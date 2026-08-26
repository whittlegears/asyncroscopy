"""Skill store and hybrid search index shared by every agent backend."""

from pathlib import Path

from .index import OllamaEmbedder, SkillIndex, SkillIndexUnavailable
from .store import SkillRecord, SkillStore
from .usage import UsageLog, apply_usage_prior, task_hash

__all__ = [
    "OllamaEmbedder",
    "SkillIndex",
    "SkillIndexUnavailable",
    "SkillRecord",
    "SkillStore",
    "SkillsService",
    "UsageLog",
    "task_hash",
]

rerank_pool = 10


class SkillsService:
    """One handle for backends and device commands: sync, search, load, and usage."""

    def __init__(self, store: SkillStore, index: SkillIndex, usage: UsageLog | None = None):
        self.store = store
        self.index = index
        self.usage = usage

    @classmethod
    def at(cls, root: Path, ollama_host: str, embedding_model: str) -> "SkillsService":
        store = SkillStore(root)
        index = SkillIndex(root / "index.db", OllamaEmbedder(ollama_host, embedding_model))
        return cls(store, index, UsageLog(Path(root) / "usage.db"))

    def sync_from_payload(self, skills: list[dict]) -> dict:
        report = self.store.replace_all(skills)
        report.update(self.index.sync(self.store.list_skills()))
        return report

    def find_skills(self, query: str, k: int = 5) -> list[dict]:
        results = self.index.search(query, max(k, rerank_pool))
        if self.usage is not None:
            results = apply_usage_prior(results, self.usage.priors())
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
                    "usage_prior": result.get("usage_prior", 0.0),
                }
            )
        return enriched[:k]

    def record_usage(self, skill_ids: list[str], run_task_hash: str, success: bool) -> int:
        if self.usage is None:
            raise RuntimeError("This skill store has no usage log.")
        return self.usage.record(skill_ids, run_task_hash, success)

    def usage_report(self) -> list[dict]:
        stats = self.usage.stats() if self.usage is not None else {}
        report = []
        for record in self.store.list_skills():
            entry = stats.get(
                record.id, {"loads": 0, "successes": 0, "failures": 0, "last_used_at": None}
            )
            report.append(
                {
                    "id": record.id,
                    "name": record.name,
                    "enabled": record.enabled,
                    **entry,
                }
            )
        report.sort(key=lambda entry: (-entry["loads"], entry["id"]))
        return report

    def propose_skill(self, name: str, content: str) -> str:
        return self.store.add_proposal(name, content)

    def list_proposals(self) -> list[dict]:
        return self.store.list_proposals()

    def remove_proposal(self, proposal_id: str) -> bool:
        return self.store.remove_proposal(proposal_id)

    def enabled_skill_summaries(self) -> list[dict]:
        return [
            {"id": record.id, "name": record.name, "description": record.description}
            for record in self.store.list_skills()
            if record.enabled
        ]

    def load_skill(self, skill_id: str) -> str:
        records = {record.id: record for record in self.store.list_skills()}
        record = records.get(skill_id)
        if record is None or not record.enabled:
            known = ", ".join(sorted(r.id for r in records.values() if r.enabled)) or "none"
            raise KeyError(
                f"No enabled skill has the id '{skill_id}'. Enabled skill ids: {known}"
            )
        return record.text
