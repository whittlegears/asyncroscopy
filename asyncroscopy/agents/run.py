"""Run a named graph once, or work toward a goal, from Python or the command line.

    uv run python -m asyncroscopy.agents.run mapping "map the sample with an EDS spectrum at each tile"
    uv run python -m asyncroscopy.agents.run team "map a 600 nm square here and find 3 interesting regions"
    uv run python -m asyncroscopy.agents.run team --goal "map a 600 nm square here with EDS" --max-minutes 20
    uv run python -m asyncroscopy.agents.run team --goal-file my_goal.yaml

With a goal the graph's work is checked from the servers' replies and retried
with feedback until the goal is met or its budget runs out (see
``asyncroscopy.agents.goals``); the exit code is 1 when it is not met.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from langchain_core.messages import HumanMessage

from .config import AgentConfig

GRAPHS = ("mapping", "general", "team")


async def run_named_graph(name: str, prompt: str, model: str | None = None) -> str:
    """Run ``mapping``, ``general`` or ``team`` on one request and return its final text."""
    return (await _run(name, prompt, model))[0]


async def run_goal_text(goal, worker: str = "team", model: str | None = None) -> tuple[str, bool]:
    """Work toward ``goal`` (a Goal, a dict as in a goal file, or a statement) with ``team`` or ``general``."""
    from .goals import Goal
    from .graphs.goal import run_goal

    if isinstance(goal, str):
        goal = Goal.from_text(goal)
    elif isinstance(goal, dict):
        goal = Goal.from_dict(goal)
    return await run_goal(goal, worker, AgentConfig.from_env(model=model, caller=f"goal-{worker}"))


async def _final_state(graph, inputs: dict, config: dict | None, progress: bool, name: str) -> dict:
    """Run ``graph`` to the end; with ``progress``, print a line to stderr as each step finishes."""
    if not progress:
        return await graph.ainvoke(inputs, config)
    state: dict = {}
    started = time.monotonic()
    last = {"node": "start", "at": started}

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(20)
            if time.monotonic() - last["at"] >= 20:
                print(f"[{name} {time.monotonic() - started:5.0f}s] still working (last finished: {last['node']})", file=sys.stderr, flush=True)

    beat = asyncio.create_task(heartbeat())
    try:
        async for mode, chunk in graph.astream(inputs, config, stream_mode=["updates", "values"]):
            if mode == "values":
                state = chunk
                continue
            for node, update in (chunk or {}).items():
                calls = [c["name"] for m in (update or {}).get("messages", []) for c in getattr(m, "tool_calls", None) or []]
                detail = f"calling {', '.join(calls)}" if calls else ("error" if (update or {}).get("error") else "done")
                print(f"[{name} {time.monotonic() - started:5.0f}s] {node}: {detail}", file=sys.stderr, flush=True)
                last.update(node=node, at=time.monotonic())
    finally:
        beat.cancel()
    return state


async def _run(name: str, prompt: str, model: str | None, progress: bool = False) -> tuple[str, bool]:
    """(final text, whether the graph reported a failure)."""
    if name == "mapping":
        from .graphs.mapping import MappingGraph

        state = await _final_state(MappingGraph(AgentConfig.from_env(model=model)).build(), {"request": prompt}, None, progress, name)
        return state.get("report", json.dumps({k: v for k, v in state.items() if k != "context"}, default=str)), bool(state.get("error"))
    if name == "general":
        from .graphs.general import make_graph

        graph = await make_graph({"configurable": {"model": model}})
        state = await _final_state(graph, {"messages": [HumanMessage(prompt)]}, {"recursion_limit": 40}, progress, name)
        return state["messages"][-1].content, False
    if name == "team":
        from .graphs.team import MAX_STEPS, TeamGraph

        graph = TeamGraph(AgentConfig.from_env(model=model, caller="team")).build()
        state = await _final_state(graph, {"request": prompt}, {"recursion_limit": 4 * MAX_STEPS + 6}, progress, name)
        return state.get("answer", ""), False
    raise ValueError(f"unknown graph {name!r}; choose one of {GRAPHS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", choices=GRAPHS)
    parser.add_argument("prompt", nargs="?", help="the request (not needed with --goal or --goal-file)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--goal", default=None, help="work toward this goal: checked from facts, retried with feedback")
    parser.add_argument("--goal-file", default=None, help="a YAML or JSON goal: statement, criteria, budget")
    parser.add_argument("--max-minutes", type=float, default=None, help="goal time budget (default 30)")
    parser.add_argument("--max-attempts", type=int, default=None, help="goal attempts (default 2)")
    parser.add_argument("--max-dose", type=float, default=None, help="goal dose budget in e/A^2")
    args = parser.parse_args()
    if args.goal or args.goal_file:
        from .goals import Goal

        if args.graph == "mapping":
            parser.error("goals run the team or general graph; the mapping graph is a fixed workflow")
        goal = Goal.load(args.goal_file) if args.goal_file else Goal.from_text(args.goal)
        if args.max_minutes is not None:
            goal.budget.max_seconds = 60.0 * args.max_minutes
        if args.max_attempts is not None:
            goal.budget.max_attempts = args.max_attempts
        if args.max_dose is not None:
            goal.budget.max_dose_e_per_a2 = args.max_dose
        print(f"[goal] {goal.statement} -- reached when: {goal.describe()}", file=sys.stderr, flush=True)
        text, met = asyncio.run(run_goal_text(goal, args.graph, args.model))
        print(text)
        sys.exit(0 if met else 1)
    if not args.prompt:
        parser.error("give a prompt, --goal or --goal-file")
    text, failed = asyncio.run(_run(args.graph, args.prompt, args.model, progress=True))
    print(text)
    if failed:  # scripts and CI see a map that did not complete
        sys.exit(1)


if __name__ == "__main__":
    main()
