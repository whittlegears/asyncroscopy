"""Check a workflow's graph before it runs.

``Workflow.blueprint()`` reads the built ``StateGraph`` into plain data. It lists each node with its block and what
the block is given, and each edge. ``lint`` checks that data without running anything. ``Workflow.lint`` compiles the
graph first, so an edge to a node that does not exist or a graph with no start is refused by LangGraph itself.

    {"name": "map_grid", "summary": "...", "inputs": {"size_m": {"type": "number", "default": 6.0e-7, "min": 1.0e-7}},
     "budget": {"max_seconds": 3600, "max_tool_calls": 2000},
     "nodes": {"plan": {"block": "plan_grid", "with": {"size_m": "{inputs.size_m}"}},
               "tile": {"block": "acquire_tile", "with": {"tiles": "{plan.tiles}"}}},
     "edges": [["START", "plan"], ["plan", "tile"],
               {"from": "tile", "when": "tile.remaining > 0", "to": "tile", "otherwise": "END", "max_visits": 400}]}

A two-item edge comes from ``add_edge``. An edge with ``when`` comes from ``add_conditional_edges`` with ``use.when``.
An edge with ``after`` is a join made by ``add_edge(["eds", "eels"], "report")``.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

from .blocks import TYPES, Block, all_blocks
from .values import condition_names, references

START, END = "START", "END"
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
BUDGET_KEYS = {"max_seconds", "max_tool_calls"}
INPUT_KEYS = {"type", "default", "min", "max", "doc"}


def edges_of(blueprint: dict) -> list[dict]:
    """Every edge as a dict with "from" and "to"."""
    return [dict(edge) if isinstance(edge, dict) else {"from": edge[0], "to": edge[1]} for edge in blueprint.get("edges") or []]


def lint(blueprint: dict, tool_names: list[str] | None = None, analysis_tool_names: list[str] | None = None,
         blocks: dict[str, Block] | None = None) -> dict:
    """Check a blueprint from ``Workflow.blueprint()`` without running it.

    Returns {"ok", "problems", "name", "acts", "tools", "blocks", "needs_analysis"}. ``problems`` are sentences a
    person or a model can act on. ``tools`` lists what the workflow can call.

    ``tool_names`` are the instrument endpoint's tools. When given, every tool a block calls must be among them.
    ``analysis_tool_names`` does the same for the analysis server.
    """
    problems: list[str] = []
    blocks = blocks if blocks is not None else all_blocks()

    name = _lint_heading(blueprint, problems)
    inputs = _lint_inputs(blueprint, problems)
    nodes, used, named = _lint_nodes(blueprint, inputs, blocks, problems)
    _lint_edges(blueprint, inputs, nodes, blocks, problems)
    tools, analysis = _lint_tools(used, named, tool_names, analysis_tool_names, problems)
    return {
        "ok": not problems, "problems": problems, "name": name,
        "acts": any(b.acts for b in used.values()), "tools": tools + analysis, "blocks": sorted({b.name for b in used.values()}),
        "needs_analysis": bool(analysis),
    }


def _lint_heading(blueprint: dict, problems: list[str]) -> Any:
    """Check the name and the summary, and return the name."""
    name = blueprint.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        problems.append(f"name {name!r} must be lowercase letters, digits and _, starting with a letter")
    if not isinstance(blueprint.get("summary"), str) or not blueprint["summary"].strip():
        problems.append("summary is missing: one sentence saying what the workflow does")
    return name


def _lint_inputs(blueprint: dict, problems: list[str]) -> dict:
    """Check the declared inputs and the budget, and return the inputs."""
    inputs = blueprint.get("inputs") or {}
    if not isinstance(inputs, dict):
        problems.append("inputs must be a mapping of name -> {type, default, min, max}")
        inputs = {}
    for port, declared in inputs.items():
        declared = declared or {}
        if not isinstance(declared, dict) or set(declared) - INPUT_KEYS:
            problems.append(f"input {port}: allowed keys are {sorted(INPUT_KEYS)}")
        elif declared.get("type", "any") not in TYPES:
            problems.append(f"input {port}: unknown type {declared.get('type')!r}; types: {sorted(TYPES)}")
    budget = blueprint.get("budget") or {}
    if not isinstance(budget, dict) or set(budget) - BUDGET_KEYS:
        problems.append(f"budget: allowed keys are {sorted(BUDGET_KEYS)}")
    return inputs


def _lint_nodes(blueprint: dict, inputs: dict, blocks: dict[str, Block],
                problems: list[str]) -> tuple[dict, dict[str, Block], dict[str, set[str]]]:
    """Check every node, and return the nodes, the block each one uses and the tools they name."""
    nodes = blueprint.get("nodes") or {}
    if not nodes:
        problems.append("the graph has no nodes: add one with graph.add_node(name, use.block(block, ...))")
    used: dict[str, Block] = {}
    named: dict[str, set[str]] = {"instrument": set(), "analysis": set()}
    for node, body in nodes.items():
        block = _lint_node(node, body, blocks, problems)
        if block is None:
            continue
        used[node] = block
        given = body.get("with") or {}
        _lint_given(node, block, given, named, problems)
        _lint_values(node, block, given, inputs, nodes, blocks, problems)
    return nodes, used, named


def _lint_node(node: Any, body: dict, blocks: dict[str, Block], problems: list[str]) -> Block | None:
    """Check a node's name, and return its block if it has one."""
    if not NAME.match(str(node)) or node == "inputs":
        problems.append(f"node name {node!r} must be lowercase letters, digits and _ (and not 'inputs')")
    if body.get("block") is None:
        problems.append(f"node {node} is not a block: make every node with use.block, so it can be checked and its tools are known")
        return None
    block = blocks.get(str(body["block"]))
    if block is None:
        problems.append(f"node {node}: there is no block {body['block']!r}; blocks: {sorted(blocks)}")
    return block


def _lint_given(node: Any, block: Block, given: dict, named: dict[str, set[str]], problems: list[str]) -> None:
    """Check which of the block's inputs a node gives, and note the tool it names."""
    extra = sorted(set(given) - set(block.inputs))
    if extra:
        problems.append(f"node {node}: block {block.name} has no input(s) {extra}; it takes {sorted(block.inputs)}")
    missing = sorted(k for k, v in block.inputs.items() if "default" not in v and k not in given)
    if missing:
        problems.append(f"node {node}: block {block.name} needs input(s) {missing}")
    if block.tool_input and block.tool_input in given:
        tool, server = given[block.tool_input], given.get("server", "instrument")
        tool_written = isinstance(tool, str) and bool(tool.strip()) and not references(tool)
        server_written = isinstance(server, str) and server in named
        if not (tool_written and server_written):
            problems.append(f"node {node}: {block.tool_input} (and server) must be written out as text, not taken from an input or "
                            "another node, so that whoever approves the workflow sees which tool it calls")
        else:
            named[server].add(tool)


def _lint_values(node: Any, block: Block, given: dict, inputs: dict, nodes: dict, blocks: dict[str, Block],
                 problems: list[str]) -> None:
    """Check each value a node gives its block, by reference or by type."""
    for port, value in given.items():
        for ref in references(value):
            problem = _check_reference(ref, inputs, nodes, blocks)
            if problem:
                problems.append(f"node {node}, input {port}: {problem}")
        kind = block.inputs.get(port, {}).get("type", "any")
        if not references(value) and kind in TYPES and kind != "any" and value is not None:
            if (isinstance(value, bool) and kind in ("number", "integer")) or not isinstance(value, TYPES[kind]):
                problems.append(f"node {node}, input {port}: {value!r} is not a {kind}")


@dataclass
class _Links:
    """How the edges join the nodes, gathered while the edges are checked."""

    names: set[str]
    graph: dict[str, list[str]]
    unbounded: dict[str, list[str]]
    sources: dict[str, int] = field(default_factory=dict)
    conditions: dict[str, int] = field(default_factory=dict)


def _lint_edges(blueprint: dict, inputs: dict, nodes: dict, blocks: dict[str, Block], problems: list[str]) -> None:
    """Check every edge, then the shape of the graph they make."""
    names = set(nodes)
    links = _Links(names, graph={n: [] for n in [START, *names]}, unbounded={n: [] for n in [START, *names]})
    for edge in edges_of(blueprint):
        _lint_edge(edge, links, inputs, nodes, blocks, problems)
    _lint_shape(links, problems)


def _lint_edge(edge: dict, links: _Links, inputs: dict, nodes: dict, blocks: dict[str, Block], problems: list[str]) -> None:
    """Check one edge and add it to the links."""
    sources, conditions = links.sources, links.conditions
    src = edge["from"]
    if "unreadable" in edge:
        problems.append(f"edge from {src}: {edge['unreadable']}; its condition cannot be checked")
        sources[src] = sources.get(src, 0) + 1
        conditions[src] = conditions.get(src, 0) + 1
        return
    sources[src] = sources.get(src, 0) + 1
    if "when" in edge:
        conditions[src] = conditions.get(src, 0) + 1
        _lint_when(edge, inputs, nodes, blocks, problems)
    if "max_visits" in edge and not (isinstance(edge["max_visits"], int) and edge["max_visits"] > 0):
        problems.append(f"edge from {src}: max_visits must be a whole number above 0")
    for target in (edge["to"], *([edge["otherwise"]] if "otherwise" in edge else [])):
        if target in links.names:
            links.graph[src].append(target)
            if "max_visits" not in edge:
                links.unbounded[src].append(target)


def _lint_when(edge: dict, inputs: dict, nodes: dict, blocks: dict[str, Block], problems: list[str]) -> None:
    """Check that an edge's condition reads and names only declared inputs and outputs."""
    src = edge["from"]
    try:
        for ref in condition_names(edge["when"]):
            problem = _check_reference(ref, inputs, nodes, blocks)
            if problem:
                problems.append(f"edge from {src}, when: {problem}")
    except (ValueError, SyntaxError) as exc:
        problems.append(f"edge from {src}: when {edge['when']!r} is not a condition ({exc})")


def _lint_shape(links: _Links, problems: list[str]) -> None:
    """Check that every node is reached, has a way out and sits in no unbounded loop."""
    for src, count in links.sources.items():
        if links.conditions.get(src, 0) and count > 1:
            problems.append(f"node {src} has a condition and {count - 1} other edge(s) leaving it; a node with a condition has only that "
                            "one way out (several plain edges are parallel branches)")
    reached = _reach(links.graph, START)
    for node in sorted(links.names - reached):
        problems.append(f"node {node} is never reached from START")
    for node in sorted(links.names & reached):
        if node not in links.sources:
            problems.append(f'node {node} has no edge leaving it; end with graph.add_edge("{node}", END)')
    cycle = _cycle(links.unbounded)
    if cycle:
        problems.append(f"the loop {' -> '.join(cycle)} has no bound: give max_visits to use.when on one of its edges")


def _lint_tools(used: dict[str, Block], named: dict[str, set[str]], tool_names: list[str] | None,
                analysis_tool_names: list[str] | None, problems: list[str]) -> tuple[list[str], list[str]]:
    """Check that the servers offer every tool the workflow calls, and return those tools."""
    tools = sorted({t for b in used.values() for t in (*b.tools, *b.always)} | named["instrument"])
    analysis = sorted({t for b in used.values() for t in b.analysis_tools} | named["analysis"])
    for offered, wanted, where in ((tool_names, tools, "the instrument endpoint"), (analysis_tool_names, analysis, "the analysis server")):
        if offered is not None:
            for pattern in wanted:
                if not any(fnmatch.fnmatchcase(n, pattern) for n in offered):
                    problems.append(f"{where} offers no tool matching {pattern}, which a block of this workflow calls")
    return tools, analysis


def _check_reference(ref: str, inputs: dict, nodes: dict, blocks: dict[str, Block]) -> str | None:
    head, *rest = ref.split(".")
    if head == "inputs":
        if not rest or rest[0] not in inputs:
            return f"{{{ref}}} names no declared input; inputs: {sorted(inputs)}"
        return None
    if head not in nodes:
        return f"{{{ref}}} names no node; nodes: {sorted(nodes)}"
    block = blocks.get(str((nodes[head] or {}).get("block")))
    if block is not None and (not rest or rest[0] not in block.outputs):
        return f"{{{ref}}}: block {block.name} has no output {rest[0] if rest else ''!r}; it gives {sorted(block.outputs)}"
    return None


def _reach(graph: dict[str, list[str]], start: str) -> set[str]:
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        for nxt in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _cycle(graph: dict[str, list[str]]) -> list[str] | None:
    """Find a loop in ``graph``. Returns its nodes, ending where it starts, or None."""
    state: dict[str, int] = {}
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        path.append(node)
        for nxt in graph.get(node, []):
            if state.get(nxt) == 1:
                return [*path[path.index(nxt):], nxt]
            if state.get(nxt) is None:
                found = visit(nxt)
                if found:
                    return found
        path.pop()
        state[node] = 2
        return None

    for node in list(graph):
        if state.get(node) is None:
            found = visit(node)
            if found:
                return found
    return None
