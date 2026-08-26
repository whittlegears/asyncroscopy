"""Skill store and hybrid search index shared by every agent backend."""

from pathlib import Path

from .index import OllamaEmbedder, SkillIndex, SkillIndexUnavailable
from .store import SkillRecord, SkillStore

__all__ = [
    "OllamaEmbedder",
    "SkillIndex",
    "SkillIndexUnavailable",
    "SkillRecord",
    "SkillStore",
    "SkillsService",
]


class SkillsService:
    """One handle for backends and device commands: sync, search, and load."""

    def __init__(self, store: SkillStore, index: SkillIndex):
        self.store = store
        self.index = index

    @classmethod
    def at(cls, root: Path, ollama_host: str, embedding_model: str) -> "SkillsService":
        store = SkillStore(root)
        index = SkillIndex(root / "index.db", OllamaEmbedder(ollama_host, embedding_model))
        return cls(store, index)

    def sync_from_payload(self, skills: list[dict]) -> dict:
        report = self.store.replace_all(skills)
        report.update(self.index.sync(self.store.list_skills()))
        return report

    def find_skills(self, query: str, k: int = 5) -> list[dict]:
        results = self.index.search(query, k)
        records = {record.id: record for record in self.store.list_skills()}
        enriched = []
        for result in results:
            record = records.get(result["id"])
            if record is None:
                continue
            enriched.append(
                {
                    "id": result["id"],
                    "name": record.name,
                    "description": record.description,
                    "score": result["score"],
                    "matched_section": result["matched_section"],
                }
            )
        return enriched

    def close(self) -> None:
        self.index.close()

    def load_skill(self, skill_id: str) -> str:
        records = {record.id: record for record in self.store.list_skills()}
        record = records.get(skill_id)
        if record is None or not record.enabled:
            known = ", ".join(sorted(r.id for r in records.values() if r.enabled)) or "none"
            raise KeyError(
                f"No enabled skill has the id '{skill_id}'. Enabled skill ids: {known}"
            )
        return record.text
