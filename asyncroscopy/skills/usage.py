"""Skill usage log feeding a capped rank prior and the pruning report.

Unlike the index, this state is NOT disposable — run outcomes cannot be
rebuilt from the skill store. It lives in ``usage.db`` directly under the
store root, which ``replace_all`` never touches. Each operation opens its own
sqlite connection so any thread can use it.

The prior obeys two hard rules from the design: it never resurrects a
disabled skill (it runs after the index's enabled filter), and it is capped
at ``prior_cap`` so semantic match always dominates — a popular wrong skill
must not beat a rare right one.
"""

import hashlib
import sqlite3
import time
from pathlib import Path

prior_cap = 0.15
prior_damping = 5.0


def task_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


class UsageLog:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS usage ("
                "id INTEGER PRIMARY KEY, "
                "skill_id TEXT NOT NULL, "
                "task_hash TEXT NOT NULL, "
                "success INTEGER NOT NULL, "
                "at REAL NOT NULL)"
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record(self, skill_ids: list[str], run_task_hash: str, success: bool) -> int:
        cleaned = [str(skill_id).strip() for skill_id in skill_ids if str(skill_id).strip()]
        if not cleaned:
            return 0
        now = time.time()
        connection = self._connect()
        try:
            connection.executemany(
                "INSERT INTO usage (skill_id, task_hash, success, at) VALUES (?, ?, ?, ?)",
                [(skill_id, str(run_task_hash), int(bool(success)), now) for skill_id in cleaned],
            )
            connection.commit()
        finally:
            connection.close()
        return len(cleaned)

    def stats(self) -> dict[str, dict]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT skill_id, "
                "COUNT(*), "
                "SUM(success), "
                "MAX(at) "
                "FROM usage GROUP BY skill_id"
            ).fetchall()
        finally:
            connection.close()
        collected = {}
        for skill_id, loads, successes, last_used_at in rows:
            successes = int(successes or 0)
            collected[skill_id] = {
                "loads": int(loads),
                "successes": successes,
                "failures": int(loads) - successes,
                "last_used_at": float(last_used_at),
            }
        return collected

    def priors(self) -> dict[str, float]:
        priors = {}
        for skill_id, entry in self.stats().items():
            priors[skill_id] = (entry["successes"] - entry["failures"]) / (
                entry["loads"] + prior_damping
            )
        return priors


def apply_usage_prior(results: list[dict], priors: dict[str, float]) -> list[dict]:
    """Re-rank search results by score * (1 + prior_cap * prior), keeping both numbers visible."""
    adjusted = []
    for result in results:
        prior = max(-1.0, min(1.0, priors.get(result["id"], 0.0)))
        entry = dict(result)
        entry["usage_prior"] = round(prior, 4)
        entry["score"] = round(float(result["score"]) * (1.0 + prior_cap * prior), 6)
        adjusted.append(entry)
    adjusted.sort(key=lambda entry: entry["score"], reverse=True)
    return adjusted
