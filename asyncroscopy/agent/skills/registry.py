"""A Hermes-style skills database backed by ``SKILL.md`` folders.

Skills live in git as ``<skills_dir>/<name>/SKILL.md``. At load time they are
indexed into a SQLite FTS5 table so the agent can search them by keyword. The
agent is shown only names and descriptions (``render_skills_index``) and calls
``skill_view`` to read a body: progressive disclosure keeps the system prompt
small while still giving the model deep context on demand.
"""

from __future__ import annotations

import json
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Callable, Iterable, Sequence

from asyncroscopy.agent.skills.frontmatter import (
    SKILL_FILENAME,
    Skill,
    SkillError,
    load_skill_file,
    render_skill_markdown,
    validate_skill_name,
)

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


class SkillRegistry:
    """Discover, search, and save skills from one or more directories.

    Args:
        dirs: directories containing ``<name>/SKILL.md`` folders. Missing
            directories are ignored; the first one is where new skills are saved.
        db_path: SQLite database path for the FTS index (default in-memory).
    """

    def __init__(self, dirs: Iterable[str | Path], db_path: str = ":memory:") -> None:
        self.dirs: list[Path] = [Path(d).expanduser() for d in dirs]
        if not self.dirs:
            raise ValueError("SkillRegistry needs at least one skills directory.")
        self._db_path = db_path
        self._skills: dict[str, Skill] = {}
        self._errors: list[str] = []
        self._conn: sqlite3.Connection | None = None
        self._fts_available = True
        self.reload()

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_dirs(cls, dirs: Iterable[str | Path] | str | Path, db_path: str = ":memory:") -> "SkillRegistry":
        if isinstance(dirs, (str, Path)):
            dirs = [dirs]
        return cls(dirs, db_path=db_path)

    @property
    def save_dir(self) -> Path:
        """Directory new skills are written to (the first configured directory)."""
        return self.dirs[0]

    @property
    def errors(self) -> list[str]:
        """Parse errors encountered during the last reload (malformed files are skipped)."""
        return list(self._errors)

    def _iter_skill_files(self) -> Iterable[Path]:
        seen: set[Path] = set()
        for base in self.dirs:
            if not base.is_dir():
                continue
            for skill_file in sorted(base.glob(f"*/{SKILL_FILENAME}")):
                resolved = skill_file.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield skill_file

    def reload(self) -> int:
        """Re-scan the directories and rebuild the search index. Returns the skill count."""
        skills: dict[str, Skill] = {}
        errors: list[str] = []
        for skill_file in self._iter_skill_files():
            try:
                skill = load_skill_file(skill_file)
            except (SkillError, OSError, UnicodeDecodeError) as exc:
                errors.append(f"{skill_file}: {exc}")
                continue
            if skill.name in skills:
                errors.append(f"{skill_file}: duplicate skill name {skill.name!r} (first one wins)")
                continue
            skills[skill.name] = skill
        self._skills = skills
        self._errors = errors
        for message in errors:
            warnings.warn(f"Skipping skill: {message}", stacklevel=2)
        self._rebuild_index()
        return len(self._skills)

    def _rebuild_index(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("DROP TABLE IF EXISTS skills")
            conn.execute(
                "CREATE VIRTUAL TABLE skills USING fts5("
                "name, description, tags, triggers, body, tokenize='porter unicode61')"
            )
            conn.executemany(
                "INSERT INTO skills(name, description, tags, triggers, body) VALUES (?, ?, ?, ?, ?)",
                [
                    (s.name, s.description, " ".join(s.tags), " ".join(s.triggers), s.body)
                    for s in self._skills.values()
                ],
            )
            conn.commit()
            self._conn = conn
            self._fts_available = True
        except sqlite3.Error as exc:
            # FTS5 missing from this SQLite build: fall back to substring search.
            self._conn = None
            self._fts_available = False
            warnings.warn(f"SQLite FTS5 unavailable ({exc}); using substring search for skills.", stacklevel=2)

    # ------------------------------------------------------------------ reads
    def list(self) -> list[Skill]:
        """All skills sorted by name."""
        return [self._skills[name] for name in sorted(self._skills)]

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get((name or "").strip().lower())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def search(self, query: str, limit: int = 5) -> list[tuple[Skill, float]]:
        """Rank skills for a free-text query.

        Returns ``(skill, score)`` pairs, best first. With FTS5 the score is the
        (negated) bm25 rank, so higher is better; the fallback scores by the
        number of query terms that appear in the skill.
        """
        terms = [t.lower() for t in _WORD_RE.findall(query or "")]
        if not terms or not self._skills:
            return []
        limit = max(1, int(limit))

        if self._conn is not None and self._fts_available:
            for joiner in (" AND ", " OR "):
                match = joiner.join(f'"{term}"' for term in terms)
                try:
                    rows = self._conn.execute(
                        "SELECT name, bm25(skills) AS rank FROM skills WHERE skills MATCH ? ORDER BY rank LIMIT ?",
                        (match, limit),
                    ).fetchall()
                except sqlite3.Error:
                    rows = []
                if rows:
                    return [(self._skills[name], -float(rank)) for name, rank in rows if name in self._skills]
            return []

        scored: list[tuple[Skill, float]] = []
        for skill in self._skills.values():
            haystack = " ".join(
                [skill.name, skill.description, " ".join(skill.tags), " ".join(skill.triggers), skill.body]
            ).lower()
            hits = sum(1 for term in terms if term in haystack)
            if hits:
                scored.append((skill, float(hits)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].name))
        return scored[:limit]

    # ----------------------------------------------------------------- writes
    def save(
        self,
        name: str,
        description: str,
        body: str,
        *,
        version: str = "1.0.0",
        tags: Sequence[str] = (),
        triggers: Sequence[str] = (),
        tools: Sequence[str] = (),
        overwrite: bool = False,
        target_dir: str | Path | None = None,
    ) -> Path:
        """Write ``<dir>/<name>/SKILL.md`` and re-index.

        Refuses to overwrite an existing skill unless ``overwrite`` is True.
        The name is validated so it can never escape the skills directory.
        """
        name = validate_skill_name((name or "").strip().lower())
        description = " ".join((description or "").split())
        if not description:
            raise SkillError("A skill needs a one-line description.")
        if not (body or "").strip():
            raise SkillError("A skill needs a non-empty body.")

        base = Path(target_dir).expanduser() if target_dir is not None else self.save_dir
        base.mkdir(parents=True, exist_ok=True)
        skill_dir = (base / name)
        if not skill_dir.resolve().parent == base.resolve():
            raise SkillError(f"Refusing to write outside the skills directory: {skill_dir}")
        skill_file = skill_dir / SKILL_FILENAME
        if (skill_file.exists() or name in self._skills) and not overwrite:
            raise SkillError(f"Skill {name!r} already exists; pass overwrite=True to replace it.")

        skill = Skill(
            name=name,
            description=description,
            version=str(version or "1.0.0"),
            tags=tuple(str(t).strip() for t in tags if str(t).strip()),
            triggers=tuple(str(t).strip() for t in triggers if str(t).strip()),
            tools=tuple(str(t).strip() for t in tools if str(t).strip()),
            body=body.strip("\n"),
            path=skill_file,
        )
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(render_skill_markdown(skill), encoding="utf-8")
        self.reload()
        return skill_file


# ---------------------------------------------------------------- rendering
def render_skills_index(registry: SkillRegistry, max_items: int = 50) -> str:
    """Compact skills catalogue for a system prompt (names + descriptions only)."""
    skills = registry.list()
    if not skills:
        return "## Skills\n(No skills are installed.)"
    lines = [
        "## Skills",
        "The following skills hold detailed procedures. Before performing a task that a skill covers, "
        "call `skill_view` with its name and follow the instructions. Use `skill_search` to find skills "
        "by keyword and `skill_save` to record a new reusable procedure.",
    ]
    for skill in skills[:max_items]:
        lines.append(f"- {skill.name} (v{skill.version}): {skill.description}")
    if len(skills) > max_items:
        lines.append(f"- ... and {len(skills) - max_items} more (use skill_search).")
    return "\n".join(lines)


def skill_functions(registry: SkillRegistry) -> dict[str, Callable[..., str]]:
    """Plain callables implementing the four skill tools (no LangChain needed).

    Keys: ``skills_list``, ``skill_search``, ``skill_view``, ``skill_save``.
    Every function returns a string so it can be used directly as a tool result.
    """

    def skills_list() -> str:
        """List all installed skills with their descriptions (JSON)."""
        return json.dumps([s.summary() for s in registry.list()], indent=2)

    def skill_search(query: str, limit: int = 5) -> str:
        """Search skills by keyword; returns the best matches with descriptions (JSON)."""
        hits = registry.search(query, limit=limit)
        if not hits:
            return json.dumps({"query": query, "results": [], "hint": "No skill matched; try other keywords or skills_list."})
        return json.dumps(
            {"query": query, "results": [{**s.summary(), "score": round(score, 3)} for s, score in hits]},
            indent=2,
        )

    def skill_view(name: str) -> str:
        """Return the full instructions (Markdown body) of one skill by name."""
        skill = registry.get(name)
        if skill is None:
            available = ", ".join(registry.names()) or "<none>"
            return f"Unknown skill {name!r}. Available skills: {available}"
        header = f"# {skill.name} (v{skill.version})\n{skill.description}\n"
        if skill.tools:
            header += f"\nTools used: {', '.join(skill.tools)}\n"
        return f"{header}\n{skill.body}"

    def skill_save(
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
        triggers: list[str] | None = None,
        tools: list[str] | None = None,
        overwrite: bool = False,
    ) -> str:
        """Save a new reusable skill as <skills_dir>/<name>/SKILL.md and index it."""
        try:
            path = registry.save(
                name,
                description,
                body,
                tags=tags or (),
                triggers=triggers or (),
                tools=tools or (),
                overwrite=overwrite,
            )
        except SkillError as exc:
            return f"Could not save skill: {exc}"
        return f"Saved skill {name!r} to {path}"

    return {
        "skills_list": skills_list,
        "skill_search": skill_search,
        "skill_view": skill_view,
        "skill_save": skill_save,
    }
