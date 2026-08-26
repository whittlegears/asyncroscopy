"""Hybrid semantic + keyword search index over the skill store.

The index is derived, disposable state: it lives in one sqlite file, is rebuilt
incrementally from content hashes on every sync, and deleting it loses nothing.
Embeddings come from a local Ollama server; keyword matching from sqlite's
FTS5, which is a hard requirement — silently degrading to vector-only search
would quietly change the designed ranking.
"""

import hashlib
import json
import re
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .store import SkillRecord

schema_version = "1"
rrf_offset = 60
candidate_pool = 20


class SkillIndexUnavailable(RuntimeError):
    """Raised when search cannot honestly run — the reason goes to the caller verbatim."""


class OllamaEmbedder:
    """Embeds text through a local Ollama server's /api/embed endpoint."""

    def __init__(self, host: str, model: str = "nomic-embed-text", timeout: float = 60.0):
        self.host = (host or "http://localhost:11434").rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SkillIndexUnavailable(
                f"The embedding model '{self.model}' is not reachable at {self.host}: {exc}. "
                f"Start Ollama and run: ollama pull {self.model}"
            ) from exc
        embeddings = body.get("embeddings")
        if not embeddings:
            raise SkillIndexUnavailable(
                f"Ollama returned no embeddings for model '{self.model}': {json.dumps(body)[:300]}"
            )
        return np.asarray(embeddings, dtype=np.float32)


def chunk_skill(record: SkillRecord) -> list[tuple[str, str]]:
    """Split one skill into (section, text) chunks; metadata is always chunk one."""
    chunks = [("metadata", f"name: {record.name}\ndescription: {record.description}")]
    section = "body"
    lines: list[str] = []
    for line in record.text.splitlines():
        heading = re.match(r"^#{1,3} +(.+)", line)
        if heading:
            body = "\n".join(lines).strip()
            if body:
                chunks.append((section, body))
            section = heading.group(1).strip()[:120]
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if body:
        chunks.append((section, body))
    return chunks


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    return " OR ".join(tokens)


class SkillIndex:
    def __init__(self, db_path: Path, embedder):
        self.db_path = Path(db_path)
        self.embedder = embedder
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            self._connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, skill_id UNINDEXED, text)"
            )
        except sqlite3.OperationalError as exc:
            raise SkillIndexUnavailable(
                f"This python's sqlite3 has no FTS5 support, which hybrid skill search requires: {exc}"
            ) from exc
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                skill_id TEXT NOT NULL,
                section TEXT NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL,
                version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        self._connection.commit()

    def _meta(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def sync(self, records: list[SkillRecord]) -> dict:
        if self._meta("embedding_model") not in (None, self.embedder.model) or self._meta(
            "schema_version"
        ) not in (None, schema_version):
            self._connection.execute("DELETE FROM chunks")
            self._connection.execute("DELETE FROM chunks_fts")
            self._connection.execute("DELETE FROM skills")

        wanted_ids = {record.id for record in records}
        for (skill_id,) in self._connection.execute("SELECT id FROM skills").fetchall():
            if skill_id not in wanted_ids:
                self._delete_skill(skill_id)

        embedded = 0
        for record in records:
            chunks = chunk_skill(record)
            hashes = [_content_hash(f"{section}\n{text}") for section, text in chunks]
            existing = {
                row[0]
                for row in self._connection.execute(
                    "SELECT content_hash FROM chunks WHERE skill_id = ?", (record.id,)
                ).fetchall()
            }
            if existing != set(hashes):
                self._delete_skill(record.id)
                vectors = self.embedder.embed([text for _, text in chunks])
                for (section, text), content_hash, vector in zip(chunks, hashes, vectors):
                    cursor = self._connection.execute(
                        "INSERT INTO chunks (skill_id, section, text, content_hash, embedding) VALUES (?, ?, ?, ?, ?)",
                        (record.id, section, text, content_hash, np.asarray(vector, dtype=np.float32).tobytes()),
                    )
                    self._connection.execute(
                        "INSERT INTO chunks_fts (chunk_id, skill_id, text) VALUES (?, ?, ?)",
                        (cursor.lastrowid, record.id, text),
                    )
                embedded += len(chunks)
            self._connection.execute(
                "INSERT INTO skills (id, enabled, version) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET enabled = excluded.enabled, version = excluded.version",
                (record.id, int(record.enabled), record.version),
            )

        self._set_meta("embedding_model", self.embedder.model)
        self._set_meta("schema_version", schema_version)
        self._connection.commit()
        return {"skills": len(records), "chunks_embedded": embedded}

    def _delete_skill(self, skill_id: str) -> None:
        self._connection.execute("DELETE FROM chunks WHERE skill_id = ?", (skill_id,))
        self._connection.execute("DELETE FROM chunks_fts WHERE skill_id = ?", (skill_id,))
        self._connection.execute("DELETE FROM skills WHERE id = ?", (skill_id,))

    def search(self, query: str, k: int = 5) -> list[dict]:
        rows = self._connection.execute(
            "SELECT c.id, c.skill_id, c.section, c.embedding FROM chunks c "
            "JOIN skills s ON s.id = c.skill_id WHERE s.enabled = 1"
        ).fetchall()
        if not rows:
            raise SkillIndexUnavailable(
                "The skill index is empty — no enabled skills have been synced to this device."
            )

        query_vector = self.embedder.embed([query])[0]
        matrix = np.vstack([np.frombuffer(row[3], dtype=np.float32) for row in rows])
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query_vector) or 1.0)
        norms[norms == 0] = 1.0
        similarity = matrix @ query_vector / norms
        vector_order = np.argsort(similarity)[::-1][:candidate_pool]
        vector_ranks = {rows[int(i)][0]: rank for rank, i in enumerate(vector_order)}

        enabled_chunk_ids = {row[0] for row in rows}
        keyword_ranks = {}
        fts = _fts_query(query)
        if fts:
            keyword_rows = self._connection.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (fts, candidate_pool),
            ).fetchall()
            for rank, (chunk_id,) in enumerate(keyword_rows):
                if chunk_id in enabled_chunk_ids:
                    keyword_ranks[chunk_id] = rank

        scores: dict[int, float] = {}
        for chunk_id, rank in vector_ranks.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_offset + rank)
        for chunk_id, rank in keyword_ranks.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_offset + rank)

        chunk_info = {row[0]: (row[1], row[2]) for row in rows}
        best_per_skill: dict[str, tuple[float, str]] = {}
        for chunk_id, score in scores.items():
            skill_id, section = chunk_info[chunk_id]
            if skill_id not in best_per_skill or score > best_per_skill[skill_id][0]:
                best_per_skill[skill_id] = (score, section)

        results = []
        for skill_id, (score, section) in sorted(
            best_per_skill.items(), key=lambda item: item[1][0], reverse=True
        )[:k]:
            results.append({"id": skill_id, "score": round(float(score), 6), "matched_section": section})
        return results

    def close(self) -> None:
        self._connection.close()

    def rebuild(self, records: list[SkillRecord]) -> dict:
        self._connection.execute("DELETE FROM chunks")
        self._connection.execute("DELETE FROM chunks_fts")
        self._connection.execute("DELETE FROM skills")
        self._connection.execute("DELETE FROM meta")
        self._connection.commit()
        return self.sync(records)
