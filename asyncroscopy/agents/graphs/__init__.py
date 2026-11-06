"""LangGraph graphs: ``mapping`` (fixed workflow), ``general`` (tool-using agent with skills),
``team`` (planner and specialists), ``goal`` (work toward a goal checked from facts) and ``swarm``
(a supervisor and the worker agents spawned into it)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


async def run_keeping_partial(graph: Runnable, inputs: dict, config: dict | None = None, timeout: float | None = None) -> tuple[dict, str]:
    """Run a graph to the end and return (its state, ""); on a recursion limit, a timeout or a crash,
    return (the last state it reached, what went wrong) instead of raising.

    A failed run must not lose the calls it made: a job it started is still running on the
    instrument, and whoever retries has to know about it. A call still waiting for its reply when
    the run stopped appears without one (a failed call) in the messages.
    """
    last: dict = {"messages": list(inputs.get("messages", []))}
    if not hasattr(graph, "astream"):  # a plain runnable (and test stand-ins): no partial state to keep
        try:
            run = graph.ainvoke(inputs, config)
            return (await asyncio.wait_for(run, timeout) if timeout else await run), ""
        except (asyncio.TimeoutError, TimeoutError):
            return last, "stopped: the time budget ran out"
        except Exception as exc:
            return last, f"{type(exc).__name__}: {str(exc)[:300]}"

    async def consume() -> None:
        nonlocal last
        async for state in graph.astream(inputs, config, stream_mode="values"):
            last = state

    try:
        await (asyncio.wait_for(consume(), timeout) if timeout else consume())
        return last, ""
    except (asyncio.TimeoutError, TimeoutError):
        return last, "stopped: the time budget ran out"
    except Exception as exc:
        return last, f"{type(exc).__name__}: {str(exc)[:300]}"
