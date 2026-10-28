"""Run a named graph once, from Python or the command line.

    uv run python -m asyncroscopy.agents.run mapping "map the sample with an EDS spectrum at each tile"
"""

from __future__ import annotations

import argparse
import asyncio
import json

from langchain_core.messages import HumanMessage

from .config import AgentConfig

GRAPHS = ("mapping", "general")


async def run_named_graph(name: str, prompt: str, model: str | None = None) -> str:
    """Run ``mapping`` or ``general`` on one request and return its final text."""
    if name == "mapping":
        from .graphs.mapping import MappingGraph

        state = await MappingGraph(AgentConfig.from_env(model=model)).build().ainvoke({"request": prompt})
        return state.get("report", json.dumps({k: v for k, v in state.items() if k != "context"}, default=str))
    if name == "general":
        from .graphs.general import make_graph

        graph = await make_graph({"configurable": {"model": model}})
        state = await graph.ainvoke({"messages": [HumanMessage(prompt)]}, {"recursion_limit": 40})
        return state["messages"][-1].content
    raise ValueError(f"unknown graph {name!r}; choose one of {GRAPHS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", choices=GRAPHS)
    parser.add_argument("prompt")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    print(asyncio.run(run_named_graph(args.graph, args.prompt, args.model)))


if __name__ == "__main__":
    main()
