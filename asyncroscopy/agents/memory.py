"""The agent's Markdown memory: what the instrument is, lessons from past runs, and what each run did.

Kept as plain Markdown files a person can read and edit, in one folder
(``$ASYNCROSCOPY_AGENT_MEMORY``, default ``outputs/agent_memory``):

* ``INSTRUMENT.md``: what this instrument (or its twin) is, its devices, limits and
  conventions. Seeded from the live tool list on first use; edit it by hand.
* ``LESSONS.md``: short dated lessons, one per line ("- 2026-10-28: ..."), added
  when a goal is not met.
* ``experiments/<date>_<run>.md``: one page per goal run (request, goal, steps,
  keys produced, outcome, issues).
* ``skills/<name>/SKILL.md``: skills learned from successful runs
  (``asyncroscopy.agents.learning``).

``notes_for(request)`` returns the sections most relevant to a request,
chosen with the same word matching as skills, within a character budget:
never the whole folder. Data keys, map keys and ids in the notes are replaced
by placeholders, so an agent never takes an earlier run's key for this one's.

Nothing here is reviewed before an agent reads it (unlike learned skills), so
what is written is kept small and checked: a lesson is one short line,
refused if it carries keys, ids or unsafe instructions (the learned-skill
lint), deduplicated, and LESSONS.md keeps the newest ``MAX_LESSONS``; at most
``MAX_EXPERIMENT_PAGES`` experiment pages are kept. Writes go through a
temporary file under a lock, so concurrent runs do not lose each other's lines.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from asyncroscopy.skills.registry import content_words, learned_dir

NOTES_BUDGET_CHARS = 1500  # about 400 tokens of notes per model call
MAX_EXPERIMENTS_SCANNED = 50
MAX_LESSONS = 100
MAX_LESSON_CHARS = 240
MAX_EXPERIMENT_PAGES = 200
_WRITE_LOCK = threading.Lock()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def redact(text: str) -> str:
    """Text with data keys, map keys and request/job ids replaced by placeholders."""
    from asyncroscopy.agents.goals import DATA_KEY, HEX_ID, MAP_KEY

    text = MAP_KEY.sub("<an earlier map>", text)
    text = DATA_KEY.sub("<an earlier key>", text)
    return HEX_ID.sub("<an id>", text)


def memory_dir() -> Path:
    """$ASYNCROSCOPY_AGENT_MEMORY, default outputs/agent_memory (the registry looks for learned skills inside it)."""
    return learned_dir().parent


def _slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:limit] or "run"


@dataclass
class Memory:
    root: Path

    @classmethod
    def from_env(cls) -> Memory:
        return cls(memory_dir())

    @property
    def instrument_file(self) -> Path:
        return self.root / "INSTRUMENT.md"

    @property
    def lessons_file(self) -> Path:
        return self.root / "LESSONS.md"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    @staticmethod
    def shared_instrument_file() -> Path | None:
        """A team's own instrument notes kept in their repository ($ASYNCROSCOPY_INSTRUMENT_NOTES), if set and present.

        outputs/ is not tracked by git, so a team that wants its INSTRUMENT.md versioned points
        this at a file in the repo (e.g. configs/instrument_notes/raman.md); it is read as well as
        the memory folder's, and no INSTRUMENT.md is seeded into the memory folder then.
        """
        path = os.environ.get("ASYNCROSCOPY_INSTRUMENT_NOTES")
        return Path(path) if path and Path(path).is_file() else None

    # ------------------------------------------------------------------ reading
    def _sections(self) -> list[tuple[str, str]]:
        """(source, text) chunks: INSTRUMENT.md by heading, each lesson, each experiment page's summary."""
        chunks: list[tuple[str, str]] = []
        for notes in (self.shared_instrument_file(), self.instrument_file):
            if notes is not None and notes.is_file():
                text = notes.read_text(encoding="utf-8")
                for part in re.split(r"(?m)^(?=## )", text):
                    if part.strip():
                        chunks.append((notes.name, part.strip()))
        if self.lessons_file.is_file():
            for line in self.lessons_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("- "):
                    chunks.append(("LESSONS.md", line))
        if self.experiments_dir.is_dir():
            pages = sorted(self.experiments_dir.glob("*.md"), reverse=True)[:MAX_EXPERIMENTS_SCANNED]
            for page in pages:
                head = page.read_text(encoding="utf-8")[:600]
                chunks.append((f"experiments/{page.name}", head.strip()))
        return chunks

    def notes_for(self, request: str, budget: int = NOTES_BUDGET_CHARS) -> str:
        """The memory most relevant to ``request``, best first, within ``budget`` characters ("" if none)."""
        asked = content_words(request)
        if not asked:
            return ""
        scored = []
        for order, (source, text) in enumerate(self._sections()):
            hits = len(asked & content_words(text))
            if hits:
                scored.append((hits, -order, source, text))
        out, used = [], 0
        for _hits, _order, source, text in sorted(scored, reverse=True):
            entry = f"[{source}] {redact(text)}"
            if used + len(entry) > budget:
                if not out:
                    out.append(entry[:budget])
                break
            out.append(entry)
            used += len(entry)
        return "\n".join(out)

    # ------------------------------------------------------------------ writing
    def seed_instrument(self, tool_names: set[str]) -> bool:
        """Write INSTRUMENT.md from the endpoint's tools if it does not exist yet. True if written."""
        if self.instrument_file.exists() or self.shared_instrument_file() is not None:
            return False
        devices: dict[str, list[str]] = {}
        native = []
        for name in sorted(tool_names):
            device, _, command = name.partition("_")
            if command and device[:1].isupper():
                devices.setdefault(device, []).append(command)
            else:
                native.append(name)
        jobs = sorted(d for d, cmds in devices.items() if "reserve" in cmds and "start" in cmds)
        lines = [
            "# Instrument",
            "",
            "Seeded from the MCP tool list on " + time.strftime("%Y-%m-%d") + ". Edit it: agents read the sections that match their task.",
            "",
            "## Devices",
            "",
        ]
        for device, commands in devices.items():
            shown = ", ".join(commands[:12]) + (f" and {len(commands) - 12} more" if len(commands) > 12 else "")
            lines.append(f"- {device}: {shown}")
        lines += ["", "## Jobs", ""]
        lines.append(f"- Job devices (run_job, then wait_for_job): {', '.join(jobs) or 'none'}")
        lines.append("- A long experiment is one job, not a loop of moves and acquisitions.")
        lines += [
            "",
            "## Conventions",
            "",
            "- Lengths are metres; a STEM stage position is [x, y, z, alpha, beta] (metres, degrees).",
            "- Data are keys in Tiled; look at them with view_data (text or png).",
            "- High-risk calls may be held for an operator (APPROVAL REQUIRED); never try to approve them yourself.",
            "",
            "## Limits",
            "",
            "- (fill in: stage travel, field of view range, dose limits, detectors to avoid)",
            "",
            "## Tools without a device",
            "",
            f"- {', '.join(native) or 'none'}",
            "",
        ]
        with _WRITE_LOCK:
            if not self.instrument_file.exists():
                _atomic_write(self.instrument_file, "\n".join(lines))
        return True

    def add_lesson(self, lesson: str) -> str:
        """Add one short dated lesson line. Returns "added", "duplicate" or "refused: why".

        Keys and ids are taken out first (a lesson is about what to do, not about one run's
        data); a lesson the learned-skill lint still refuses (unsafe advice) is not written.
        """
        from asyncroscopy.agents.learning import lint

        text = redact(" ".join(lesson.split()))[:MAX_LESSON_CHARS]
        problems = [p for p in lint(text, []) if not p.startswith("longer than")]
        if problems:
            return "refused: " + "; ".join(problems)
        line = "- " + time.strftime("%Y-%m-%d") + ": " + text
        with _WRITE_LOCK:
            existing = self.lessons_file.read_text(encoding="utf-8") if self.lessons_file.is_file() else ""
            lessons = [entry for entry in existing.splitlines() if entry.startswith("- ")]
            if any(entry.split(": ", 1)[-1] == text for entry in lessons):
                return "duplicate"
            lessons = (lessons + [line])[-MAX_LESSONS:]
            _atomic_write(self.lessons_file, "# Lessons from past runs\n\n" + "\n".join(lessons) + "\n")
        return "added"

    def record_experiment(self, run_id: str, request: str, goal: str, steps: list[str], keys: list[str], outcome: str, issues: list[str]) -> Path:
        """Write experiments/<date>_<run>.md and return its path (the oldest pages beyond MAX_EXPERIMENT_PAGES go)."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        path = self.experiments_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{_slug(run_id, 20)}.md"
        lines = [
            f"# {request[:120]}",
            "",
            f"- Outcome: {outcome}",
            f"- Goal: {goal}",
            f"- Keys: {', '.join(keys[:20]) or 'none'}",
            "",
            "## Steps",
            "",
            *[f"{i + 1}. {s}" for i, s in enumerate(steps[:40])],
            "",
            "## Issues",
            "",
            *([f"- {i}" for i in issues[:20]] or ["- none"]),
            "",
        ]
        with _WRITE_LOCK:
            _atomic_write(path, "\n".join(lines))
            for old in sorted(self.experiments_dir.glob("*.md"))[:-MAX_EXPERIMENT_PAGES]:
                old.unlink(missing_ok=True)
        return path
