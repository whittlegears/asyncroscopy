"""What a workflow module uses to build its LangGraph graph from blocks, and the registry of workflows.

A workflow is a Python module that builds a LangGraph ``StateGraph``. ``graphs/map_grid.py`` is one.

    def build(use: Steps) -> StateGraph:
        graph = StateGraph(WorkflowState)
        graph.add_node("plan", use.block("plan_grid", size_m="{inputs.size_m}", fov_m="{inputs.fov_m}"))
        graph.add_node("tile", use.block("acquire_tile", tiles="{plan.tiles}", fov_m="{inputs.fov_m}"))

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "tile")
        graph.add_conditional_edges("tile", use.when("tile.remaining > 0", then="tile", otherwise=END, max_visits=400))
        return graph

    WORKFLOW = Workflow(name="map_tiles", summary="...", inputs={...}, budget={...}, build=build)

``use.block`` turns a registered block into a graph node. A block input is a value, or ``"{inputs.name}"`` for a
run input, or ``"{node.output}"`` for an earlier node's output. ``use.when`` makes the router for a conditional edge
from a condition over those same names. A loop needs ``max_visits``. The rest is plain LangGraph.

Two plain edges leaving one node are parallel branches. Their nodes run at the same time. If one fails, the other
is not repeated when the run is resumed. ``graph.add_edge(["eds", "eels"], "report")`` joins them, so ``report``
runs once both have finished. The block ``ask_person`` stops the run until a person answers. ``run_tool`` calls one
named tool.

Each node and router ``Steps`` makes carries which block it runs and what it is given, or its condition.
``Workflow.blueprint`` reads that back from the graph, together with the graph's edges. ``lint`` checks the blueprint before anything runs, and a run's
record keeps it. ``Workflow.hash`` covers the module's source and the code of every block it uses. An approval
names that hash, so changing either makes it a different workflow.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END as GRAPH_END
from langgraph.graph import START as GRAPH_START
from langgraph.graph import StateGraph

from .blocks import Block, BlockContext, BlockFailed, all_blocks
from .checks import END, START, edges_of, lint
from .values import check_inputs, evaluate, fill


def _function(runnable: Any) -> Any:
    """The Python function LangGraph wrapped as a node or a router, so what ``Steps`` attached to it can be read."""
    return getattr(runnable, "afunc", None) or getattr(runnable, "func", None) or runnable


def _merge(old: dict | None, new: dict | None) -> dict:
    return {**(old or {}), **(new or {})}


class WorkflowState(TypedDict, total=False):
    """The run's inputs, each node's latest outputs, and how often each node ran.

    It is kept small and plain so a checkpoint is easy to read. The visit counts are what bound loops.
    """

    inputs: dict
    outputs: Annotated[dict, _merge]
    visits: Annotated[dict, _merge]


class Steps:
    """Passed to a workflow's ``build``. It makes graph nodes from blocks and routers from conditions for one run.

    Neither needs to be told which node it belongs to. A block node reads its own name from LangGraph when it runs,
    and ``Workflow.blueprint`` reads what each node and router is from the graph that ``build`` returned.
    """

    def __init__(self, ctx: BlockContext | None = None, blocks: dict[str, Block] | None = None) -> None:
        self.ctx = ctx
        self.blocks = blocks if blocks is not None else all_blocks()

    def block(self, block: str, **given: Any) -> Callable:
        """Make a graph node that runs the registered block ``block`` with ``given``.

        Add it with ``graph.add_node("name", use.block("block_name", input=value, ...))``.
        """
        if block not in self.blocks:
            raise KeyError(f"there is no block {block!r}; blocks: {sorted(self.blocks)}")
        chosen, ctx = self.blocks[block], self.ctx

        async def run(state: WorkflowState, config: RunnableConfig) -> dict:
            node = config["metadata"]["langgraph_node"]
            outputs = state.get("outputs") or {}
            try:
                args = fill(given, state.get("inputs") or {}, outputs)
            except KeyError as exc:
                raise BlockFailed(f"node {node}: {exc.args[0]}") from exc
            defaults = {k: v["default"] for k, v in chosen.inputs.items() if "default" in v}
            args = {**defaults, **args}
            result = await chosen.run(ctx, args, outputs.get(node))
            missing = sorted(set(chosen.outputs) - set(result or {}))
            if missing:
                raise BlockFailed(f"node {node}: block {chosen.name} did not give its output(s) {missing}")
            visits = int((state.get("visits") or {}).get(node, 0))
            return {"outputs": {node: result}, "visits": {node: visits + 1}}

        run.__name__ = f"block_{block}"
        run.block_node = {"block": block, "with": dict(given)}  # read back by Workflow.blueprint
        return run

    def when(self, condition: str, *, then: str, otherwise: str, max_visits: int | None = None) -> Callable:
        """Make the router for a conditional edge.

        Add it with ``graph.add_conditional_edges("source", use.when("condition", then=..., otherwise=...))``. It goes
        to ``then`` while ``condition`` holds and to ``otherwise`` when it does not. ``max_visits`` is the most times
        the chosen node may run. Every loop needs one. LangGraph learns the two targets from the router's return type,
        so they are not listed again.
        """
        name = {GRAPH_END: END, GRAPH_START: START}
        edge = {"when": condition, "to": name.get(then, then), "otherwise": name.get(otherwise, otherwise)}
        if max_visits is not None:
            edge["max_visits"] = max_visits

        def route(state: WorkflowState) -> str:
            holds = evaluate(condition, state.get("inputs") or {}, state.get("outputs") or {})
            chosen = edge["to"] if holds else edge["otherwise"]
            if max_visits and chosen != END and int((state.get("visits") or {}).get(chosen, 0)) >= max_visits:
                raise BlockFailed(f"node {chosen} already ran {max_visits} times, the bound (max_visits) on the edge `when {condition}`")
            return GRAPH_END if chosen == END else chosen

        targets = tuple(dict.fromkeys(GRAPH_END if t == END else t for t in (edge["to"], edge["otherwise"])))
        route.__annotations__ = {"state": WorkflowState, "return": Literal[targets]}
        route.__name__ = f"when {condition}"  # LangGraph names the branch after it
        route.when_edge = edge  # read back by Workflow.blueprint
        return route


@dataclass(frozen=True)
class Workflow:
    """One workflow with its name, inputs, budget and the function that builds its graph.

    ``inputs`` maps a name to {"type", "default", "min", "max", "doc"}. A run input outside its range is refused.
    ``budget`` is {"max_seconds", "max_tool_calls"}.
    ``build(use)`` returns the ``StateGraph``. The module docstring shows one.
    """

    name: str
    summary: str
    build: Callable[[Steps], StateGraph] = field(repr=False, compare=False)
    inputs: dict[str, dict] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    # Set for a workflow proposed as text because it has no module file to read
    source_text: str = field(default="", repr=False, compare=False)

    def _built(self, ctx: BlockContext | None) -> StateGraph:
        graph = self.build(Steps(ctx))
        if not isinstance(graph, StateGraph):
            raise TypeError(f"workflow {self.name}: build must return the langgraph StateGraph it made, not {type(graph).__name__}")
        return graph

    def blueprint(self) -> dict:
        """The graph as plain data.

        It has "name", "summary", "inputs", "budget", "nodes" as {node: {"block", "with"}} and "edges". It is read
        from the graph that ``build`` made, so it always matches what the graph does.
        """
        graph = self._built(None)
        name = {GRAPH_START: START, GRAPH_END: END}
        edges: list = [[name.get(a, a), name.get(b, b)] for a, b in sorted(graph.edges)]
        for starts, end in sorted(graph.waiting_edges):
            edges.extend({"from": start, "to": name.get(end, end), "after": sorted(starts)} for start in sorted(starts))
        for source in sorted(graph.branches):
            for branch in graph.branches[source].values():
                edge = getattr(_function(branch.path), "when_edge", None)
                if edge is not None:
                    edges.append({"from": name.get(source, source), **edge})
                else:
                    edges.append({"from": name.get(source, source), "unreadable": "a conditional edge not made with use.when"})
        nodes = {node: dict(getattr(_function(spec.runnable), "block_node", None) or {"block": None, "with": {}}) for node, spec in graph.nodes.items()}
        return {"name": self.name, "summary": self.summary, "inputs": dict(self.inputs), "budget": dict(self.budget), "nodes": nodes, "edges": edges}

    def compile(self, ctx: BlockContext, checkpointer: Any = None):
        """Compile the graph for one run. Lint first, because this assumes the workflow has no problems."""
        return self._built(ctx).compile(checkpointer=checkpointer)

    def source(self) -> str:
        """The text of the whole module that ``build`` is written in."""
        if self.source_text:
            return self.source_text
        module = inspect.getmodule(self.build)
        try:
            return inspect.getsource(module) if module is not None else inspect.getsource(self.build)
        except (OSError, TypeError):
            return ""

    def hash(self) -> str:
        """A hash of the workflow's source, its blueprint and the code of every block it uses.

        Approvals and resumed runs identify the workflow by this hash.

        A block's code means the whole module its ``run`` is in. Blocks use helpers in the same module, so a change
        to a helper is a change to the block. The module of ``BlockContext`` is included too because every call a
        block makes goes through it.
        """
        blueprint = self.blueprint()
        digest = hashlib.sha256(self.source().encode())
        digest.update(json.dumps(blueprint, sort_keys=True, separators=(",", ":"), default=str).encode())
        blocks = all_blocks()
        modules = {BlockContext.__module__: inspect.getmodule(BlockContext)}
        for name in sorted({str(n.get("block")) for n in blueprint["nodes"].values()}):
            if name in blocks:
                module = inspect.getmodule(blocks[name].run)
                if module is not None:
                    modules[module.__name__] = module
                else:
                    modules[f"block {name}"] = blocks[name].run
        for key in sorted(modules):
            digest.update(key.encode())
            try:
                digest.update(inspect.getsource(modules[key]).encode())
            except (OSError, TypeError):
                pass
        return digest.hexdigest()

    def lint(self, tool_names: list[str] | None = None, analysis_tool_names: list[str] | None = None) -> dict:
        """Check the workflow without running it. Returns the result of ``checks.lint`` plus this workflow's "hash".

        The graph is compiled first, so anything LangGraph refuses is reported in its own words.
        """
        try:
            self._built(None).compile()
            blueprint = self.blueprint()
        except Exception as exc:
            return {"ok": False, "problems": [f"the graph could not be built: {exc.args[0] if exc.args else type(exc).__name__}"],
                    "name": self.name, "hash": None, "acts": False, "tools": [], "blocks": [], "needs_analysis": False}
        return {**lint(blueprint, tool_names, analysis_tool_names), "hash": self.hash()}

    def check_inputs(self, given: dict | None) -> dict:
        """Return a run's inputs, the declared defaults with ``given`` on top. Raises ValueError for a bad one."""
        return check_inputs(self.name, self.inputs, given)

    def describe(self) -> dict:
        return {"name": self.name, "summary": self.summary, "inputs": dict(self.inputs)}


def recursion_limit(blueprint: dict) -> int:
    """Enough graph steps to run every node once and every loop up to its bound.

    One pass round a loop can take a step for every node in it, so each bound is multiplied by the number of nodes.
    The edges' own ``max_visits`` stop a loop well before this, so a generous limit costs nothing.
    """
    nodes = len(blueprint.get("nodes") or {})
    return 10 + nodes + sum(int(e.get("max_visits", 0)) for e in edges_of(blueprint)) * max(2, nodes)


_WORKFLOWS: dict[str, Workflow] = {}
_loaded = False


def register_workflow(workflow: Workflow) -> Workflow:
    """Make a workflow runnable by name. A name already taken by another workflow is refused."""
    if workflow.name in _WORKFLOWS and _WORKFLOWS[workflow.name] is not workflow:
        raise ValueError(f"a workflow named {workflow.name!r} is already registered")
    _WORKFLOWS[workflow.name] = workflow
    return workflow


def all_workflows() -> dict[str, Workflow]:
    """Every registered workflow by name. Shipped ones are loaded from the ``graphs`` package on first use."""
    global _loaded
    if not _loaded:
        from . import graphs

        for info in pkgutil.iter_modules(graphs.__path__):
            found = getattr(importlib.import_module(f"{graphs.__name__}.{info.name}"), "WORKFLOW", None)
            if isinstance(found, Workflow):
                register_workflow(found)
        _loaded = True
    return dict(_WORKFLOWS)


def get_workflow(source: str | Workflow) -> Workflow:
    """Look up a workflow by name. A ``Workflow`` is returned as it is."""
    if isinstance(source, Workflow):
        return source
    found = all_workflows()
    if source not in found:
        raise KeyError(f"there is no workflow named {source!r}; workflows: {sorted(found)}")
    return found[source]
