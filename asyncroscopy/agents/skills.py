"""Check and review skills: list, show and lint any skill; promote the ones agents learned.

    uv run python -m asyncroscopy.agents.skills list
    uv run python -m asyncroscopy.agents.skills show mapping
    uv run python -m asyncroscopy.agents.skills lint mapping                      # a skill by name
    uv run python -m asyncroscopy.agents.skills lint my_lab/raman/SKILL.md        # or a file (before adding it)
    uv run python -m asyncroscopy.agents.skills promote map_with_eds
    uv run python -m asyncroscopy.agents.skills promote map_with_eds --check-endpoint   # also check its tools are offered

``list`` shows every skill: those shipped here, a lab's own ($ASYNCROSCOPY_SKILLS_PATH) and
the ones agents learned ($ASYNCROSCOPY_AGENT_MEMORY/skills, default outputs/agent_memory),
with each one's status and any problem. A learned skill is a draft until promoted; only
promoted ones are put in agents' prompts. ``lint`` checks a skill file the way promotion
does: front matter, length, no advice to get around approvals or safety, and (with
--check-endpoint) only tools the agent endpoint offers.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from asyncroscopy.agents.learning import learned_skills, lint_skill, promote
from asyncroscopy.agents.memory import Memory
from asyncroscopy.skills.registry import list_skills


def _offered() -> set[str]:
    from asyncroscopy.agents.config import AgentConfig

    return {t.name for t in asyncio.run(AgentConfig.from_env().tools(allow=["*"]))}


def _find(name: str, memory: Memory) -> Path | None:
    """A skill file by path, by a shipped or lab skill's name, or by a learned skill's name."""
    path = Path(name)
    if path.is_file():
        return path
    if (path / "SKILL.md").is_file():
        return path / "SKILL.md"
    found = next((s.path for s in list_skills() if s.name == name), None)
    return found or (learned_skills(memory).get(name) or (None,))[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="every skill, its status and any problem")
    for name in ("show", "lint", "promote"):
        cmd = sub.add_parser(name)
        cmd.add_argument("name", help="a skill name, or (show/lint) a SKILL.md path or its folder")
        cmd.add_argument("--check-endpoint", action="store_true", help="also check its tools are offered by the agent endpoint")
    args = parser.parse_args(argv)
    memory = Memory.from_env()
    if args.command == "list":
        unreadable: list[str] = []
        learned = learned_skills(memory, unreadable)
        for skill in list_skills():
            where = "learned" if skill.learned else ("lab" if not str(skill.path).startswith(str(Path(__file__).resolve().parents[1])) else "shipped")
            acts = "acts" if skill.acts else "reads"
            print(f"{skill.name:28s} {where:8s} {acts:6s} {skill.version:8s} {skill.problem or skill.description}")
        for name, (_path, meta) in learned.items():
            if meta.get("status") != "promoted":
                info = meta.get("learned") or {}
                print(f"{name:28s} {'draft':8s} {'':6s} {'':8s} {info.get('successes', 1)} successes, last {info.get('last', '?')}: {meta.get('description', '')}")
        for problem in unreadable:
            print(f"UNREADABLE {problem}", file=sys.stderr)
        return 1 if unreadable else 0
    path = _find(args.name, memory)
    if path is None:
        print(f"no skill or file {args.name!r}; `list` shows the skills", file=sys.stderr)
        return 2
    offered = _offered() if args.check_endpoint else None
    if args.command == "show":
        print(path.read_text(encoding="utf-8"))
        return 0
    if args.command == "lint":
        problems = lint_skill(path, offered)
        print("ok" if not problems else "\n".join(problems))
        return 0 if not problems else 1
    if args.name not in learned_skills(memory):
        print(f"{args.name!r} is not a learned skill; only learned drafts are promoted (shipped and lab skills are used as they are)", file=sys.stderr)
        return 2
    try:
        print(promote(args.name, memory, offered))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
