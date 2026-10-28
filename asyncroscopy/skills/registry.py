"""Find and load SKILL.md files.

A skill is a folder holding ``SKILL.md``: YAML front matter (name,
description, version, min_asyncroscopy, envelope_schema, tools, requires) followed by
instructions in Markdown. Agents see only names and descriptions until they
load one, which keeps small-model prompts short.

Skills come from this package's folders and from any directory listed in
``$ASYNCROSCOPY_SKILLS_PATH`` (``os.pathsep``-separated), so a lab can keep
its own skills outside the repository; an outside skill replaces a packaged
one with the same name. A skill whose minimum versions are newer than what is
installed, or whose ``requires`` tools (glob patterns) the endpoint does not
offer, is listed as unavailable rather than loaded: an indenter agent is not
offered the STEM mapping skill.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from asyncroscopy.contracts import ENVELOPE_VERSION

HERE = Path(__file__).resolve().parent


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    version: str = "0.0.0"
    tools: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    min_asyncroscopy: str | None = None
    envelope_schema: str | None = None
    problem: str | None = None  # why it cannot be used here, if it cannot

    @property
    def available(self) -> bool:
        return self.problem is None


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def _installed_version() -> str | None:
    """Installed asyncroscopy version; from pyproject.toml when running from a checkout; None if unknown."""
    try:
        return version("asyncroscopy")
    except PackageNotFoundError:
        pass
    pyproject = HERE.parents[1] / "pyproject.toml"
    try:
        import tomllib

        return str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"])
    except (OSError, KeyError, ValueError):
        return None


def _parse(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path} has no front matter")
    _, front, _body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    for key in ("name", "description"):
        if not meta.get(key):
            raise ValueError(f"{path} front matter is missing '{key}'")
    skill = Skill(
        name=str(meta["name"]),
        description=" ".join(str(meta["description"]).split()),
        path=path,
        version=str(meta.get("version", "0.0.0")),
        tools=list(meta.get("tools") or []),
        requires=list(meta.get("requires") or []),
        min_asyncroscopy=meta.get("min_asyncroscopy"),
        envelope_schema=meta.get("envelope_schema"),
    )
    installed = _installed_version()
    if skill.min_asyncroscopy and installed and _version_tuple(installed) < _version_tuple(str(skill.min_asyncroscopy)):
        skill.problem = f"needs asyncroscopy >= {skill.min_asyncroscopy}, installed {installed}"
    elif skill.envelope_schema and _version_tuple(str(skill.envelope_schema))[:1] != _version_tuple(ENVELOPE_VERSION)[:1]:
        skill.problem = f"written for envelope schema {skill.envelope_schema}, installed {ENVELOPE_VERSION}"
    return skill


def skill_dirs() -> list[Path]:
    extra = [Path(p) for p in os.environ.get("ASYNCROSCOPY_SKILLS_PATH", "").split(os.pathsep) if p]
    return [HERE, *extra]


def list_skills(tool_names: set[str] | None = None) -> list[Skill]:
    """Every skill found; with ``tool_names``, those whose ``requires`` are not all offered are unavailable."""
    skills: dict[str, Skill] = {}
    for base in skill_dirs():
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/SKILL.md")):
            try:
                skill = _parse(path)
            except (ValueError, yaml.YAMLError) as exc:
                skill = Skill(name=path.parent.name, description="(unreadable skill file)", path=path, problem=str(exc))
            skills[skill.name] = skill  # later folders (a lab's own) replace same-named packaged skills
    if tool_names is not None:
        for skill in skills.values():
            missing = [p for p in skill.requires if not any(fnmatch.fnmatchcase(t, p) for t in tool_names)]
            if missing and skill.problem is None:
                skill.problem = f"needs tools this endpoint does not offer: {', '.join(missing)}"
    return sorted(skills.values(), key=lambda s: s.name)


def load_skill(name: str, tool_names: set[str] | None = None) -> str:
    """Full instructions of one skill. Raises KeyError if unknown, RuntimeError if not usable here."""
    for skill in list_skills(tool_names):
        if skill.name == name:
            if not skill.available:
                raise RuntimeError(f"skill {name!r} cannot be used: {skill.problem}")
            return skill.path.read_text(encoding="utf-8").split("---", 2)[2].strip()
    raise KeyError(f"no skill named {name!r}; available: {[s.name for s in list_skills()]}")
