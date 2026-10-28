"""Run a named graph once, from Python or the command line.

    uv run python -m asyncroscopy.agents.run mapping "map the sample with an EDS spectrum at each tile"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from langchain_core.messages import HumanMessage

from .config import AgentConfig

GRAPHS = ("mapping", "general")


async def run_named_graph(name: str, prompt: str, model: str | None = None) -> str:
    """Run ``mapping`` or ``general`` on one request and return its final text."""
    return (await _run(name, prompt, model))[0]


async def _final_state(graph, inputs: dict, config: dict | None, progress: bool, name: str) -> dict:
    """Run ``graph`` to the end; with ``progress``, print a line to stderr as each step finishes."""
    if not progress:
        return await graph.ainvoke(inputs, config)
    state: dict = {}
    started = time.monotonic()
    async for mode, chunk in graph.astream(inputs, config, stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk
            continue
        for node, update in (chunk or {}).items():
            calls = [c["name"] for m in (update or {}).get("messages", []) for c in getattr(m, "tool_calls", None) or []]
            detail = f"calling {', '.join(calls)}" if calls else ("error" if (update or {}).get("error") else "done")
            print(f"[{name} {time.monotonic() - started:5.0f}s] {node}: {detail}", file=sys.stderr, flush=True)
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
    raise ValueError(f"unknown graph {name!r}; choose one of {GRAPHS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", choices=GRAPHS)
    parser.add_argument("prompt")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    text, failed = asyncio.run(_run(args.graph, args.prompt, args.model, progress=True))
    print(text)
    if failed:  # scripts and CI see a map that did not complete
        sys.exit(1)


if __name__ == "__main__":
    main()
