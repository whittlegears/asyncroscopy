"""Parse and render ``SKILL.md`` files (YAML frontmatter + Markdown body)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

SKILL_FILENAME = "SKILL.md"
SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z", re.DOTALL)


class SkillError(ValueError):
    """Raised when a SKILL.md file is malformed."""


@dataclass(frozen=True)
class Skill:
    """One skill: metadata from the frontmatter plus the Markdown body."""

    name: str
    description: str
    version: str = "1.0.0"
    tags: tuple[str, ...] = field(default_factory=tuple)
    triggers: tuple[str, ...] = field(default_factory=tuple)
    tools: tuple[str, ...] = field(default_factory=tuple)
    body: str = ""
    path: Path | None = None

    def summary(self) -> dict[str, Any]:
        """Metadata only (what the agent sees before loading the body)."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": list(self.tags),
            "triggers": list(self.triggers),
            "tools": list(self.tools),
        }


def validate_skill_name(name: str) -> str:
    """Return ``name`` if it is a safe skill/folder name, else raise ``SkillError``."""
    if not isinstance(name, str) or not SKILL_NAME_RE.match(name):
        raise SkillError(
            f"Invalid skill name {name!r}: use lowercase letters, digits, '-' or '_' "
            "(1-64 characters, starting with a letter or digit)."
        )
    return name


def _as_tuple(value: Any, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise SkillError(f"Frontmatter key {key!r} must be a string or a list of strings.")


def parse_skill_markdown(text: str, path: Path | None = None) -> Skill:
    """Parse the text of a SKILL.md file.

    The file must start with a ``---`` YAML frontmatter block containing at
    least ``name`` and ``description``. When ``path`` is given and the name does
    not match the folder name, a ``SkillError`` is raised.
    """
    where = f" ({path})" if path else ""
    match = _FRONTMATTER_RE.match(text.lstrip("﻿"))
    if not match:
        raise SkillError(f"SKILL.md{where} must start with a '---' YAML frontmatter block.")

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"Invalid YAML frontmatter{where}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SkillError(f"Frontmatter{where} must be a mapping.")

    name = meta.get("name")
    if not name:
        raise SkillError(f"Frontmatter{where} is missing required key 'name'.")
    name = validate_skill_name(str(name).strip())
    description = str(meta.get("description") or "").strip()
    if not description:
        raise SkillError(f"Frontmatter{where} is missing required key 'description'.")

    if path is not None:
        folder = Path(path).parent.name
        if folder and folder != name:
            raise SkillError(f"Skill name {name!r} does not match its folder {folder!r}{where}.")

    return Skill(
        name=name,
        description=" ".join(description.split()),
        version=str(meta.get("version") or "1.0.0").strip(),
        tags=_as_tuple(meta.get("tags"), "tags"),
        triggers=_as_tuple(meta.get("triggers"), "triggers"),
        tools=_as_tuple(meta.get("tools"), "tools"),
        body=match.group(2).strip("\n"),
        path=Path(path) if path is not None else None,
    )


def load_skill_file(path: Path) -> Skill:
    """Read and parse one SKILL.md file."""
    path = Path(path)
    return parse_skill_markdown(path.read_text(encoding="utf-8"), path=path)


def render_skill_markdown(skill: Skill) -> str:
    """Serialise a skill back to SKILL.md text (inverse of ``parse_skill_markdown``)."""
    meta: dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
        "version": skill.version,
    }
    if skill.tags:
        meta["tags"] = list(skill.tags)
    if skill.triggers:
        meta["triggers"] = list(skill.triggers)
    if skill.tools:
        meta["tools"] = list(skill.tools)
    frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, default_flow_style=False).strip()
    body = skill.body.strip("\n")
    return f"---\n{frontmatter}\n---\n\n{body}\n"
