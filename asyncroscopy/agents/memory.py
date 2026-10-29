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
never the whole folder.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from asyncroscopy.skills.registry import content_words, learned_dir

NOTES_BUDGET_CHARS = 1500  # about 400 tokens of notes per model call
MAX_EXPERIMENTS_SCANNED = 50


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

    # ------------------------------------------------------------------ reading
    def _sections(self) -> list[tuple[str, str]]:
        """(source, text) chunks: INSTRUMENT.md by heading, each lesson, each experiment page's summary."""
        chunks: list[tuple[str, str]] = []
        if self.instrument_file.is_file():
            text = self.instrument_file.read_text(encoding="utf-8")
            for part in re.split(r"(?m)^(?=## )", text):
                if part.strip():
                    chunks.append(("INSTRUMENT.md", part.strip()))
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
            entry = f"[{source}] {text}"
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
        if self.instrument_file.exists():
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
        self.root.mkdir(parents=True, exist_ok=True)
        self.instrument_file.write_text("\n".join(lines), encoding="utf-8")
        return True

    def add_lesson(self, lesson: str) -> None:
        """Append one dated lesson line (an identical lesson is not added twice)."""
        line = "- " + time.strftime("%Y-%m-%d") + ": " + " ".join(lesson.split())[:400]
        existing = self.lessons_file.read_text(encoding="utf-8") if self.lessons_file.is_file() else "# Lessons from past runs\n\n"
        if line.split(": ", 1)[1] in existing:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.lessons_file.write_text(existing.rstrip("\n") + "\n" + line + "\n", encoding="utf-8")

    def record_experiment(self, run_id: str, request: str, goal: str, steps: list[str], keys: list[str], outcome: str, issues: list[str]) -> Path:
        """Write experiments/<date>_<run>.md and return its path."""
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
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
