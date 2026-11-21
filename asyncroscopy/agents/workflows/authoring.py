"""Check and load a workflow that arrives as text, without trusting it.

A proposed workflow is a Python module that builds a LangGraph ``StateGraph`` from blocks, the same as a shipped one.
``TEMPLATE`` below is a whole example. The text may come from a model, so it is parsed first and only these
statements are accepted.

* Imports of ``StateGraph``, ``START``, ``END``, ``Steps``, ``Workflow`` and ``WorkflowState``.
* Constants that are plain data such as numbers, text, lists and dicts.
* One ``def build(use)`` that makes the graph, calls ``add_node``, ``add_edge`` and ``add_conditional_edges`` on it
  with plain data, and returns it.
* ``WORKFLOW = Workflow(name=..., summary=..., inputs=..., budget=..., build=build)``.

Anything else is reported as a problem with its line number and the module is not run. What passes can only add
blocks and edges to a graph, so building it is safe. ``lint`` then checks the graph like a shipped one.

This module decides what counts as a workflow. ``library.py`` decides whether it may run on an instrument.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import Any

from langgraph.graph import END, START, StateGraph

from .graph import Steps, Workflow, WorkflowState
from .checks import NAME

GRAPH_NAMES = {"END", "START", "StateGraph"}
OUR_NAMES = {"Steps", "Workflow", "WorkflowState"}
OUR_MODULES = {"asyncroscopy.agents.workflows", "asyncroscopy.agents.workflows.graph", "graph"}
GRAPH_CALLS = {"add_node", "add_edge", "add_conditional_edges"}
WORKFLOW_KEYS = {"name", "summary", "inputs", "budget", "build"}
MAX_SOURCE_CHARS = 20_000

TEMPLATE = '''"""Take every tile of a square grid, moving the stage, and stop there (no stitch)."""

from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.workflows import Steps, Workflow, WorkflowState

INPUTS = {
    "size_m": {"type": "number", "default": 6.0e-7, "min": 1.0e-7, "max": 5.0e-6, "doc": "Side of the square, metres."},
    "fov_m": {"type": "number", "default": 3.0e-7, "min": 2.0e-8, "max": 5.0e-6, "doc": "Field of view of one tile, metres."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)
    graph.add_node("plan", use.block("plan_grid", size_m="{inputs.size_m}", fov_m="{inputs.fov_m}", move_by="stage"))
    graph.add_node("tile", use.block(
        "acquire_tile",
        tiles="{plan.tiles}",
        fov_m="{inputs.fov_m}",
        stage_rest="{plan.stage_rest}",
        start_fov_m="{plan.start_fov_m}",
    ))
    graph.add_node("home", use.block("return_start", xy_m="{plan.start_xy_m}", stage_rest="{plan.stage_rest}", fov_m="{plan.start_fov_m}"))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "tile")
    graph.add_conditional_edges("tile", use.when("tile.remaining > 0", then="tile", otherwise="home", max_visits=400))
    graph.add_edge("home", END)
    return graph


WORKFLOW = Workflow(
    name="tiles_only",
    summary="Take every tile of a square grid around the current position by moving the stage, then return; no stitch.",
    inputs=INPUTS,
    budget={"max_seconds": 3600, "max_tool_calls": 2000},
    build=build,
)
'''

TEMPLATE_PARALLEL = '''"""Image, ask a person, then zoom in and take an EDS and an EELS spectrum at the same time."""

from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.workflows import Steps, Workflow, WorkflowState

INPUTS = {
    "fov_m": {"type": "number", "default": 2.0e-8, "min": 5.0e-9, "max": 5.0e-6, "doc": "Field of view for the spectra, metres."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)
    graph.add_node("image", use.block("run_tool", tool="*_acquire_scanned_image", arguments={"detector_list": ["haadf"]}))
    graph.add_node("ok", use.block("ask_person", question="Zoom in here and take an EDS and an EELS spectrum?", options=["yes", "no"], show="{image.key}"))
    graph.add_node("zoom", use.block("run_tool", tool="*_set_fov", arguments={"fov": "{inputs.fov_m}"}))
    graph.add_node("eds", use.block("run_tool", tool="*_acquire_spectrum", arguments={"detector_name": "eds"}))
    graph.add_node("eels", use.block("run_tool", tool="*_acquire_eels_spectrum"))
    graph.add_node("look", use.block("run_tool", tool="*_acquire_scanned_image", arguments={"detector_list": ["haadf"]}))
    graph.add_edge(START, "image")
    graph.add_edge("image", "ok")
    graph.add_conditional_edges("ok", use.when('ok.answer == "yes"', then="zoom", otherwise=END))
    graph.add_edge("zoom", "eds")
    graph.add_edge("zoom", "eels")
    graph.add_edge(["eds", "eels"], "look")
    graph.add_edge("look", END)
    return graph


WORKFLOW = Workflow(
    name="spectra_together",
    summary="Take an image, ask a person, then zoom in and take an EDS and an EELS spectrum at the same time, and an image after.",
    inputs=INPUTS,
    budget={"max_seconds": 1800, "max_tool_calls": 20},
    build=build,
)
'''

RULES = (
    "A workflow is a Python module that builds a LangGraph StateGraph from blocks, written exactly in the shape of the "
    "examples: the same imports; constants that are plain data; def build(use) with graph = StateGraph(WorkflowState), "
    "graph.add_node(name, use.block(block, input=value, ...)), graph.add_edge(a, b), "
    "graph.add_conditional_edges(name, use.when(condition, then=..., otherwise=..., max_visits=N)) and "
    "return graph; then WORKFLOW = Workflow(name=..., summary=..., inputs=..., budget=..., build=build). "
    'A block input is a value, or "{inputs.name}" / "{node.output}" for a run input or an earlier node\'s output. '
    "A condition compares those names (and, or, not, numbers, text in quotes). A loop needs max_visits. "
    "At the same time: give one node two plain edges (add_edge(a, b) and add_edge(a, c)) and b and c run together; "
    'add_edge(["b", "c"], d) runs d once both have finished. A node with a condition has only that way out. '
    "Any tool of the endpoint is a step with the block run_tool (the tool's name written out). "
    "To have a person decide something during the run, use the block ask_person and branch on its answer. "
    "Nothing else is accepted: no other imports, calls, loops or functions. Use only blocks from list_blocks."
)


class NotAWorkflow(ValueError):
    """The text is not a workflow we accept. ``problems`` lists why, one sentence each."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _data(node: ast.AST, constants: set[str]) -> bool:
    """True for a literal or the name of a module constant."""
    if isinstance(node, ast.Name):
        return node.id in constants
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, float, bool, type(None)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool)
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_data(item, constants) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and _data(key, constants) and _data(value, constants) for key, value in zip(node.keys, node.values))
    return False


def _text(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _node_name(node: ast.AST) -> bool:
    return _text(node) or (isinstance(node, ast.Name) and node.id in ("START", "END"))


def _method(node: ast.AST, owner: str, names: set[str]) -> str | None:
    """Return the method name if ``node`` is a call ``owner.name(...)`` with name in ``names``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)):
        return None
    if node.func.value.id == owner and node.func.attr in names:
        return node.func.attr
    return None


def _build_argument(function: ast.FunctionDef, say) -> str | None:
    """Check how build is declared and return the name of its one argument."""
    args = function.args
    extras = args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults or function.decorator_list
    if len(args.args) != 1 or extras:
        say(function, "build takes exactly one argument, written def build(use)")
        return None
    for annotation in (args.args[0].annotation, function.returns):
        if annotation is not None and not (isinstance(annotation, ast.Name) and annotation.id in GRAPH_NAMES | OUR_NAMES):
            say(function, "build's annotations may only be Steps and StateGraph")
    return args.args[0].arg


def _makes_graph(statement: ast.stmt) -> bool:
    """True for a statement written name = StateGraph(WorkflowState)."""
    if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)):
        return False
    call = statement.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "StateGraph"):
        return False
    takes_state = len(call.args) == 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == "WorkflowState"
    return takes_state and not call.keywords


def _check_add_edge(statement: ast.stmt, call: ast.Call, say) -> None:
    """Check an add_edge call between two nodes or from a list of nodes to one."""
    two = len(call.args) == 2
    join = two and isinstance(call.args[0], ast.List) and len(call.args[0].elts) > 1 and all(_text(a) for a in call.args[0].elts)
    if not (two and (join or _node_name(call.args[0])) and _node_name(call.args[1])):
        say(statement, 'add_edge takes two node names (text, START or END); or a list of node names and one node name, '
                       'add_edge(["a", "b"], "c"), to run c once a and b have both finished')


def _check_add_node(statement: ast.stmt, call: ast.Call, use: str, constants: set[str], say) -> None:
    """Check an add_node call and the use.block call inside it."""
    block = call.args[1] if len(call.args) == 2 else None
    written = len(call.args) == 2 and _text(call.args[0]) and _method(block, use, {"block"})
    if not written:
        say(statement, f'add_node is written add_node("name", {use}.block("block_name", input=value, ...))')
        return
    if not (len(block.args) == 1 and _text(block.args[0])):
        say(statement, f"{use}.block takes the block's name as text, then its inputs by name")
        return
    for keyword in block.keywords:
        if keyword.arg is None or not _data(keyword.value, constants):
            say(statement, f"{use}.block input {keyword.arg or '**'} must be plain data (a value, or \"{{node.output}}\" text)")


def _plain_when(router: ast.Call) -> bool:
    """True when a use.when call takes a condition as text and only the keywords accepted."""
    given = {k.arg: k.value for k in router.keywords}
    bound = given.get("max_visits")
    condition = len(router.args) == 1 and _text(router.args[0])
    known_keys = set(given) <= {"then", "otherwise", "max_visits"} and {"then", "otherwise"} <= set(given)
    names_targets = _node_name(given.get("then")) and _node_name(given.get("otherwise"))
    whole_bound = bound is None or (isinstance(bound, ast.Constant) and isinstance(bound.value, int) and not isinstance(bound.value, bool))
    return condition and known_keys and names_targets and whole_bound


def _check_conditional_edges(statement: ast.stmt, call: ast.Call, use: str, say) -> None:
    """Check an add_conditional_edges call and the use.when call inside it."""
    router = call.args[1] if len(call.args) == 2 else None
    if not (router is not None and _text(call.args[0]) and _method(router, use, {"when"})):
        say(statement, f'add_conditional_edges is written add_conditional_edges("name", {use}.when("condition", '
                       'then=..., otherwise=..., max_visits=N))')
        return
    if not _plain_when(router):
        say(statement, f'{use}.when takes a condition as text, then=, otherwise= (node names) and max_visits= (a whole number)')


def _check_graph_call(statement: ast.stmt, graph: str, use: str, constants: set[str], say) -> None:
    """Check one statement of build that should add a node or an edge to the graph."""
    call = statement.value if isinstance(statement, ast.Expr) else None
    method = _method(call, graph, GRAPH_CALLS) if call is not None else None
    if method is None:
        say(statement, f"inside build only {graph}.add_node, {graph}.add_edge, {graph}.add_conditional_edges and return {graph} are accepted")
    elif call.keywords:
        say(statement, f"{graph}.{method} takes its arguments by position here, as in the example")
    elif method == "add_edge":
        _check_add_edge(statement, call, say)
    elif method == "add_node":
        _check_add_node(statement, call, use, constants, say)
    else:
        _check_conditional_edges(statement, call, use, say)


def _check_build(function: ast.FunctionDef, constants: set[str], say) -> None:
    use = _build_argument(function, say)
    if use is None:
        return
    graph: str | None = None
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and _text(body[0].value):
        body = body[1:]
    for position, statement in enumerate(body):
        if graph is None:
            if not _makes_graph(statement):
                say(statement, "build must start with graph = StateGraph(WorkflowState)")
                return
            graph = statement.targets[0].id
        elif isinstance(statement, ast.Return):
            last = position == len(body) - 1
            if not (last and isinstance(statement.value, ast.Name) and statement.value.id == graph):
                say(statement, f"build ends with return {graph}, and nothing follows it")
        else:
            _check_graph_call(statement, graph, use, constants, say)
    if graph is None or not body or not isinstance(body[-1], ast.Return):
        say(function, "build must end with return graph")


def _is_workflow_call(call: ast.AST, constants: set[str]) -> bool:
    """True for a call written Workflow(name=..., summary=..., build=build) with plain data for the rest."""
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Workflow" and not call.args):
        return False
    given = {k.arg: k.value for k in call.keywords}
    if not (set(given) <= WORKFLOW_KEYS and {"name", "summary", "build"} <= set(given)):
        return False
    described = _text(given["name"]) and _text(given["summary"])
    builds = isinstance(given["build"], ast.Name) and given["build"].id == "build"
    plain = all(_data(given[k], constants) for k in ("inputs", "budget") if k in given)
    return described and builds and plain


def check_source(source: str) -> list[str]:
    """Return the reasons ``source`` is not an accepted workflow, or [] when it is. Nothing in it is run."""
    if not isinstance(source, str) or not source.strip():
        return ["the workflow is empty: send the whole Python module as text"]
    if len(source) > MAX_SOURCE_CHARS:
        return [f"the workflow is {len(source)} characters, over the limit of {MAX_SOURCE_CHARS}"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"line {exc.lineno}: not valid Python ({exc.msg})"]
    problems: list[str] = []

    def say(node: ast.AST, message: str) -> None:
        problems.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    constants: set[str] = set()
    seen = {"build": 0, "WORKFLOW": 0}
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and _text(body[0].value):
        body = body[1:]
    for statement in body:
        if isinstance(statement, ast.ImportFrom):
            names = {alias.name for alias in statement.names}
            renamed = any(alias.asname for alias in statement.names)
            module = statement.module or ""
            if module == "__future__" and names == {"annotations"} and not renamed:
                continue
            if module == "langgraph.graph" and statement.level == 0 and names <= GRAPH_NAMES and not renamed:
                continue
            if module in OUR_MODULES and names <= OUR_NAMES and not renamed:
                continue
            say(statement, "only these imports are accepted: from langgraph.graph import END, START, StateGraph and "
                           "from asyncroscopy.agents.workflows import Steps, Workflow, WorkflowState")
        elif isinstance(statement, ast.FunctionDef) and statement.name == "build":
            seen["build"] += 1
            _check_build(statement, constants, say)
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target = statement.targets[0].id
            if target == "WORKFLOW":
                seen["WORKFLOW"] += 1
                if not _is_workflow_call(statement.value, constants):
                    say(statement, "WORKFLOW is written WORKFLOW = Workflow(name=\"...\", summary=\"...\", inputs=..., budget=..., build=build), "
                                   "with inputs and budget as plain data")
            elif target in GRAPH_NAMES | OUR_NAMES | {"build"} or not _data(statement.value, constants):
                say(statement, f"{target} must be plain data (numbers, text, lists, dicts), under a name of its own")
            else:
                constants.add(target)
        else:
            say(statement, "only imports, constants, def build(use) and WORKFLOW = Workflow(...) are accepted at the top of a workflow")
    for name, count in seen.items():
        if count != 1:
            problems.append(f"the module needs exactly one {'def build(use)' if name == 'build' else 'WORKFLOW = Workflow(...)'}; it has {count}")
    return problems


def load_source(source: str) -> Workflow:
    """Build the workflow from a module's text. Raises ``NotAWorkflow`` when the text is not one."""
    problems = check_source(source)
    if problems:
        raise NotAWorkflow(problems)
    tree = ast.parse(source)
    # Imports are dropped because the names they bring in are passed in below and there are no builtins to import with.
    tree.body = [statement for statement in tree.body if not isinstance(statement, ast.ImportFrom)]
    scope: dict[str, Any] = {"__builtins__": {}, "__name__": "proposed_workflow", "END": END, "START": START, "StateGraph": StateGraph,
                             "Steps": Steps, "Workflow": Workflow, "WorkflowState": WorkflowState}
    try:
        exec(compile(tree, "<proposed workflow>", "exec"), scope)  # noqa: S102  check_source passed it
    except Exception as exc:
        raise NotAWorkflow([f"the module could not be read as a workflow: {type(exc).__name__}: {exc}"]) from exc
    workflow = scope.get("WORKFLOW")
    if not isinstance(workflow, Workflow):
        raise NotAWorkflow(["WORKFLOW is not a Workflow"])
    if not NAME.match(workflow.name):
        raise NotAWorkflow([f"name {workflow.name!r} must be lowercase letters, digits and _, starting with a letter"])
    return dataclasses.replace(workflow, source_text=source)
