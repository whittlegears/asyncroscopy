"""Filesystem skill store in the agentskills.io layout, synced one-way from the GUI."""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SkillRecord:
    """One skill as held on disk: full SKILL.md text plus sync provenance."""
    id: str
    name: str
    description: str
    text: str
    enabled: bool = True
    version: int = 1
    agent_authored: bool = False
    source: str = "workspace"


def parse_frontmatter(text: str) -> tuple[str, str]:
    """Return (name, description) from a leading ``---`` frontmatter block.

    Only flat ``key: value`` lines are read; anything else is ignored. Either
    field falls back to an empty string so the caller can substitute defaults.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", ""
    name = ""
    description = ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        if key.strip() == "name":
            name = value.strip()
        elif key.strip() == "description":
            description = value.strip()
    return name, description


def first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.lstrip("#").strip()
        if stripped and stripped.strip("-"):
            return stripped[:200]
    return ""


class SkillStore:
    """Owns ``<root>/<skill-id>/SKILL.md`` directories plus a provenance sidecar.

    The GUI is authoritative: ``replace_all`` makes the store match its payload
    exactly. Files directly under the root (the search index database among
    them) are never touched — only skill subdirectories are created or removed.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_skills(self) -> list[SkillRecord]:
        records = []
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                records.append(self.read(entry.name))
        return records

    def read(self, skill_id: str) -> SkillRecord:
        skill_dir = self.root / skill_id
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        name, description = parse_frontmatter(text)
        provenance = {}
        sidecar = skill_dir / "provenance.json"
        if sidecar.is_file():
            try:
                provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                provenance = {}
        return SkillRecord(
            id=skill_id,
            name=str(provenance.get("name") or name or skill_id.replace("-", " ")),
            description=str(provenance.get("description") or description or first_meaningful_line(text)),
            text=text,
            enabled=bool(provenance.get("enabled", True)),
            version=int(provenance.get("version", 1)),
            agent_authored=bool(provenance.get("agent_authored", False)),
            source=str(provenance.get("source", "workspace")),
        )

    def replace_all(self, skills: list[dict]) -> dict:
        wanted = {}
        for skill in skills:
            skill_id = str(skill.get("id", "")).strip()
            if not skill_id or "/" in skill_id or "\\" in skill_id or skill_id.startswith("."):
                raise ValueError(f"'{skill_id}' is not a valid skill id")
            wanted[skill_id] = skill

        synced_at = datetime.now(timezone.utc).isoformat()
        for skill_id, skill in wanted.items():
            skill_dir = self.root / skill_id
            skill_dir.mkdir(exist_ok=True)
            (skill_dir / "SKILL.md").write_text(str(skill.get("text", "")), encoding="utf-8")
            provenance = {
                "name": str(skill.get("name", "")),
                "description": str(skill.get("description", "")),
                "enabled": bool(skill.get("enabled", True)),
                "version": int(skill.get("version", 1)),
                "agent_authored": bool(skill.get("agent_authored", False)),
                "source": str(skill.get("source", "workspace")),
                "synced_at": synced_at,
            }
            (skill_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

        removed = 0
        for entry in self.root.iterdir():
            if entry.is_dir() and entry.name not in wanted:
                shutil.rmtree(entry)
                removed += 1

        return {"written": len(wanted), "removed": removed}
