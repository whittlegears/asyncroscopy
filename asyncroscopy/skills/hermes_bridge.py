"""Two one-way bridges between the device skill store and a Hermes Agent install."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from .store import SkillRecord, parse_frontmatter

# Store skills are mirrored into <hermes>/skills/<EXPORT_SUBDIR>/; that subtree
# is owned by the export and never imported back.
EXPORT_SUBDIR = "asyncroscopy"
IMPORT_STATE_FILE = "_hermes_imported.json"


def resolve_hermes_home(configured: str = "") -> Path | None:
    """Resolve the Hermes install root, or None when there is none.

    An explicitly configured path wins outright (no fallback), so tests and
    multi-install setups stay deterministic; otherwise HERMES_HOME, then
    ~/.hermes.
    """
    if configured:
        path = Path(configured)
        return path if path.is_dir() else None
    for candidate in (os.environ.get("HERMES_HOME", ""), str(Path.home() / ".hermes")):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def _with_frontmatter(record: SkillRecord) -> str:
    name, _ = parse_frontmatter(record.text)
    if name:
        return record.text
    # Hermes refuses SKILL.md files without name/description frontmatter.
    header = f"---\nname: {record.id}\ndescription: {record.description or record.name}\n---\n\n"
    return header + record.text


def export_skills(records: list[SkillRecord], hermes_skills_dir: Path) -> dict:
    """Mirror the enabled store skills into <hermes_skills_dir>/asyncroscopy/."""
    target = Path(hermes_skills_dir) / EXPORT_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    wanted = {record.id: _with_frontmatter(record) for record in records if record.enabled}
    for skill_id, text in wanted.items():
        skill_dir = target / skill_id
        skill_dir.mkdir(exist_ok=True)
        (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    removed = 0
    for entry in target.iterdir():
        if entry.is_dir() and entry.name not in wanted:
            shutil.rmtree(entry)
            removed += 1
    return {"exported": len(wanted), "removed": removed}


def import_new_skills(hermes_skills_dir: Path, service, state_path: Path) -> list[str]:
    """Propose SKILL.md files the Hermes agent wrote since the last scan.

    The first scan is a silent baseline: everything already present (the
    bundled Hermes library included) is recorded, not proposed. Later scans
    propose new or changed files into the store's review queue.
    """
    hermes_skills_dir = Path(hermes_skills_dir)
    state_path = Path(state_path)
    first_scan = not state_path.exists()
    state: dict[str, str] = {}
    if not first_scan:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}

    proposed = []
    changed = False
    for skill_md in sorted(hermes_skills_dir.rglob("SKILL.md")):
        rel = skill_md.relative_to(hermes_skills_dir)
        if rel.parts[0] == EXPORT_SUBDIR:
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        key = rel.parent.as_posix()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if state.get(key) == digest:
            continue
        state[key] = digest
        changed = True
        if first_scan:
            continue
        name, _ = parse_frontmatter(text)
        service.propose_skill(name or rel.parent.name, text)
        proposed.append(key)

    if changed or first_scan:
        state_path.write_text(json.dumps(state), encoding="utf-8")
    return proposed
