"""Filesystem skill store in the agentskills.io layout, synced one-way from the GUI."""

import json
import shutil
import uuid
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
    Directories starting with ``_`` are the store's own state (``_proposals``
    holds agent-written skill drafts awaiting pickup by the GUI) and survive a
    sync untouched.
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
            if entry.is_dir() and not entry.name.startswith("_") and entry.name not in wanted:
                shutil.rmtree(entry)
                removed += 1

        return {"written": len(wanted), "removed": removed}

    @property
    def proposals_dir(self) -> Path:
        return self.root / "_proposals"

    def add_proposal(self, name: str, content: str) -> str:
        """Persist an agent-written skill draft for the GUI to pick up and review.

        Proposals never touch the live skill directories; the GUI pulls them,
        routes them through its own review gate, and removes them.
        """
        cleaned_name = str(name).strip()
        cleaned_content = str(content).strip()
        if not cleaned_name or not cleaned_content:
            raise ValueError("A skill proposal needs both a name and content.")
        proposal_id = uuid.uuid4().hex
        self.proposals_dir.mkdir(exist_ok=True)
        payload = {
            "id": proposal_id,
            "name": cleaned_name,
            "content": cleaned_content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.proposals_dir / f"{proposal_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return proposal_id

    def list_proposals(self) -> list[dict]:
        if not self.proposals_dir.is_dir():
            return []
        proposals = []
        for entry in sorted(self.proposals_dir.glob("*.json")):
            try:
                payload = json.loads(entry.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("id") and payload.get("name") and payload.get("content"):
                proposals.append(payload)
        return proposals

    def remove_proposal(self, proposal_id: str) -> bool:
        cleaned = str(proposal_id).strip()
        if not cleaned.isalnum():
            raise ValueError(f"'{proposal_id}' is not a proposal id")
        target = self.proposals_dir / f"{cleaned}.json"
        if not target.is_file():
            return False
        target.unlink()
        return True
