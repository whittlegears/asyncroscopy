"""Start a workflow graph, pause at its approvals, and resume it later.

A graph that calls ``interrupt()`` stops and waits. ``WorkflowRunner`` keeps the
compiled graphs with one checkpointer and a thread per run, so a caller that
cannot hold a Python object between calls (a model over MCP, a GUI) can start a
run, get back the question it is waiting on, and answer it later with the
run's ``thread_id``.

The default checkpointer keeps runs in memory, so they do not survive a
restart. Pass a persistent LangGraph checkpointer (for example SqliteSaver) to
keep them.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class WorkflowRunner:
    def __init__(self, checkpointer=None, recursion_limit: int = 100):
        self.checkpointer = checkpointer or InMemorySaver()
        self.recursion_limit = recursion_limit
        self._builders: Dict[str, Callable[[Any], Any]] = {}
        self._graphs: Dict[str, Any] = {}
        self._threads: Dict[str, str] = {}  # thread_id -> workflow name

    def register(self, name: str, build: Callable[..., Any]) -> None:
        """``build(checkpointer)`` returns a compiled graph."""
        self._builders[name] = build
        self._graphs.pop(name, None)

    def workflows(self) -> list[str]:
        return sorted(self._builders)

    def _graph(self, name: str):
        if name not in self._builders:
            raise ValueError(f"No workflow {name!r}; registered: {self.workflows()}")
        if name not in self._graphs:
            self._graphs[name] = self._builders[name](self.checkpointer)
        return self._graphs[name]

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.recursion_limit}

    async def _report(self, name: str, thread_id: str, result: dict) -> dict:
        graph = self._graph(name)
        snapshot = await graph.aget_state(self._config(thread_id))
        waiting = [i.value for task in snapshot.tasks for i in getattr(task, "interrupts", ())]
        return {
            "workflow": name,
            "thread_id": thread_id,
            "waiting_for": waiting[0] if waiting else None,
            "finished": not snapshot.next,
            "state": {k: v for k, v in (result or snapshot.values).items() if k != "__interrupt__"},
        }

    async def start(self, name: str, inputs: dict) -> dict:
        thread_id = uuid.uuid4().hex[:12]
        self._threads[thread_id] = name
        result = await self._graph(name).ainvoke(inputs, self._config(thread_id))
        return await self._report(name, thread_id, result)

    async def resume(self, thread_id: str, answer: Any) -> dict:
        name = self._threads.get(thread_id)
        if name is None:
            raise ValueError(f"No run {thread_id!r}")
        result = await self._graph(name).ainvoke(Command(resume=answer), self._config(thread_id))
        return await self._report(name, thread_id, result)

    async def status(self, thread_id: str) -> dict:
        name = self._threads.get(thread_id)
        if name is None:
            raise ValueError(f"No run {thread_id!r}")
        return await self._report(name, thread_id, {})
