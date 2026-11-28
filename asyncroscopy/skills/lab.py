"""The lab's own skills, written from a client such as SciAgentGUI's Skills page.

They live in one folder (``$ASYNCROSCOPY_LAB_SKILLS``, default ``outputs/lab_skills``), one
``<name>/SKILL.md`` each, and are read like any other skill folder (``registry.skill_dirs``):
a lab skill with a shipped skill's name replaces it. ``save`` checks a skill with the same
linter as promotion (``asyncroscopy.agents.learning.lint_skill``) before it is written, and
writes through a temporary file, so a half-written skill is never read.

Any skill, shipped ones too, can be switched off here without deleting it: its name goes in
``off.txt`` in the same folder, and it is listed as unavailable ("switched off on this
endpoint") so no agent is given it, until it is switched on again.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
from pathlib import Path

import yaml

from asyncroscopy import settings
from asyncroscopy.skills.registry import read_front_matter

OFF_FILE = "off.txt"
_LOCK = threading.Lock()


def lab_dir() -> Path:
    return settings.get().lab_skills


def slug(name: str) -> str:
    """A skill name as a folder and tool-safe name: lower case, words joined by underscores."""
    text = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not text:
        raise ValueError("a skill needs a name with at least one letter or digit")
    return text[:64]


def _description(body: str) -> str:
    """The first line of the manual that is not a heading, as a description when none is given."""
    for line in body.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if line and not line.startswith("#"):
            return line[:200]
    return ""


def compose(name: str, text: str, description: str | None = None) -> str:
    """A SKILL.md for ``text``: its own front matter with the name set to ``slug(name)``, or front matter made
    from the name and ``description`` (else the manual's first line). A skill that does not say ``acts`` counts
    as acting on the instrument, as every skill does."""
    meta: dict = {}
    body = text
    if text.lstrip("﻿").startswith("---"):
        _, front, body = text.lstrip("﻿").split("---", 2)
        meta = yaml.safe_load(front) or {}
        if not isinstance(meta, dict):
            raise ValueError("the front matter is not a mapping")
    meta["name"] = slug(name)
    meta["description"] = " ".join(str(description or meta.get("description") or _description(body)).split())
    meta.setdefault("version", "1.0.0")
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n\n" + body.strip() + "\n"


def save(name: str, text: str, description: str | None = None) -> Path:
    """Write the lab skill ``name`` (replacing one of that name). Raises ValueError listing what is wrong with it."""
    from asyncroscopy.agents.learning import lint_skill

    composed = compose(name, text, description)
    folder = lab_dir() / slug(name)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".SKILL.md.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(composed, encoding="utf-8")
    try:
        problems = lint_skill(tmp)
        if problems:
            raise ValueError("; ".join(problems))
        read_front_matter(tmp)
        path = folder / "SKILL.md"
        os.replace(tmp, path)
        return path
    finally:
        tmp.unlink(missing_ok=True)


def delete(name: str) -> None:
    """Delete the lab skill ``name``. Raises KeyError when the lab has no skill of that name: shipped skills are
    switched off instead (``switch``)."""
    folder = lab_dir() / slug(name)
    if not (folder / "SKILL.md").is_file():
        raise KeyError(f"the lab has no skill named {name!r}; a shipped skill cannot be deleted, only switched off")
    shutil.rmtree(folder)
    switch(name, on=True)  # nothing stays off under a vanished name


def switched_off() -> set[str]:
    """Names of the skills switched off on this endpoint."""
    path = lab_dir() / OFF_FILE
    try:
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    except OSError:
        return set()


def switch(name: str, on: bool) -> set[str]:
    """Switch the skill ``name`` on or off here; returns the names now switched off."""
    with _LOCK:
        off = switched_off()
        off = (off - {name}) if on else (off | {name})
        path = lab_dir() / OFF_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{OFF_FILE}.{os.getpid()}.tmp")
        tmp.write_text("".join(f"{n}\n" for n in sorted(off)), encoding="utf-8")
        os.replace(tmp, path)
    return off
