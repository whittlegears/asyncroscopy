"""Review the skills agents learned: list, show, lint and promote them.

    uv run python -m asyncroscopy.agents.skills list
    uv run python -m asyncroscopy.agents.skills show map_with_eds
    uv run python -m asyncroscopy.agents.skills promote map_with_eds
    uv run python -m asyncroscopy.agents.skills promote map_with_eds --check-endpoint   # also check its tools are offered

A learned skill is a draft until promoted; only promoted ones are put in agents' prompts.
The memory folder is $ASYNCROSCOPY_AGENT_MEMORY (default outputs/agent_memory).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from asyncroscopy.agents.learning import _front, learned_skills, lint, promote
from asyncroscopy.agents.memory import Memory


def _offered() -> set[str]:
    from asyncroscopy.agents.config import AgentConfig

    return {t.name for t in asyncio.run(AgentConfig.from_env().tools(allow=["*"]))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="learned skills with their status and successes")
    for name in ("show", "lint", "promote"):
        cmd = sub.add_parser(name)
        cmd.add_argument("name")
        cmd.add_argument("--check-endpoint", action="store_true", help="also check its tools are offered by the agent endpoint")
    args = parser.parse_args(argv)
    memory = Memory.from_env()
    learned = learned_skills(memory)
    if args.command == "list":
        if not learned:
            print(f"no learned skills in {memory.skills_dir}")
        for name, (path, meta) in learned.items():
            info = meta.get("learned") or {}
            print(f"{name:32s} {meta.get('status', 'draft'):9s} {info.get('successes', 1):3d} successes  last {info.get('last', '?')}  {meta.get('description', '')}")
        return 0
    if args.name not in learned:
        print(f"no learned skill {args.name!r}; learned: {sorted(learned)}", file=sys.stderr)
        return 2
    path, meta = learned[args.name]
    offered = _offered() if args.check_endpoint else None
    if args.command == "show":
        print(path.read_text(encoding="utf-8"))
        return 0
    if args.command == "lint":
        problems = lint(_front(path)[1] + str(meta.get("description", "")), list(meta.get("tools") or []), offered)
        print("ok" if not problems else "\n".join(problems))
        return 0 if not problems else 1
    try:
        print(promote(args.name, memory, offered))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
