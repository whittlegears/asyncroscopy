"""Focus by moving the stage in z: notebook 14's hill climb as a LangGraph graph.

Graph::

    plan -> approve -> explore -> climb -> finalize -> report
              |           |        ^  |        |
              | declined  |        |__|        |
              v           v  (any failure)     v
            report      report <------------- report

- ``plan`` checks the request and reads the stage (``get_stage``, which does not
  move anything) for the starting position.
- ``approve`` pauses (``interrupt``) with the plan: start z, step, z limits and
  the most measurements the climb can make. Resume with ``{"approved": true}``
  or ``{"approved": false}``.
- ``explore`` measures at the start, start - step and start + step.
- ``climb`` runs one iteration per pass and loops on itself without pausing: fit
  a quadratic to every measurement, move to the best predicted z inside the trust
  radius and inside the hard limit (start +- ``max_offset_m``), measure, then grow
  or shrink the trust radius. It stops after ``max_iterations``, when the trust
  radius falls below ``min_step_m``, or when no unmeasured z is left.
- ``finalize`` moves to the best z measured.
- ``report`` summarises the run. If anything failed after the stage first moved,
  it moves the stage back to the starting position first.

MCP reaches Tango commands only, not the ``stage.z`` attribute the notebook sets,
so every move is ``move_stage`` with the full [x, y, z, alpha, beta] vector from
``get_stage``, with only z changed. Each image comes from
``acquire_scanned_image`` as a data key; the score is ``fft_sharpness`` from the
ANALYSIS device's ``resolution`` command, or one of the notebook's pixel scores
(``contrast``, ``edge_sharpness``) when the graph is built with ``read_image``.

Tools (any instrument prefix): ``get_stage``, ``move_stage``,
``acquire_scanned_image``, and ``resolution`` for ``fft_sharpness``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import climb as maths
from .tools import ToolBox, ToolCallError

DEFAULT_STEP_M = 0.1e-6
DEFAULT_MAX_OFFSET_M = 2e-6
DEFAULT_MAX_ITERATIONS = 8
# Each climb iteration is one graph step; WorkflowRunner's default recursion_limit is 100.
MAX_ITERATIONS_LIMIT = 60
INITIAL_MEASUREMENTS = 3
Z = 2  # index of z in [x, y, z, alpha, beta]


class StageZRequest(TypedDict, total=False):
    step_m: float  # first step and trust radius
    max_offset_m: float  # hard limit: never further than this from the starting z
    max_iterations: int
    min_step_m: float  # stop when the trust radius falls below this; default step_m / 16
    metric: str  # fft_sharpness, contrast or edge_sharpness
    detectors: list


class StageZState(TypedDict, total=False):
    request: StageZRequest
    plan: dict
    status: str  # planned, approved, declined, explored, climbing, converged, done, failed
    start_position: list
    history: list  # one entry per measurement: z_m, offset_m, score, key, trial
    trust_radius_m: float
    iteration: int
    stop_reason: str
    moved: bool  # the stage was asked to move at least once
    final_z_m: float
    report: dict
    errors: list


def plan_request(request: StageZRequest, start_position, read_image=None) -> dict:
    step = float(request.get("step_m", DEFAULT_STEP_M))
    max_offset = float(request.get("max_offset_m", DEFAULT_MAX_OFFSET_M))
    min_step = float(request.get("min_step_m") or step / 16)
    iterations = int(request.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    metric = maths.choose_metric(request.get("metric"), read_image)
    problems = maths.metric_problems(metric, read_image)
    if not (step > 0 and max_offset > 0 and min_step > 0):
        problems.append("step_m, max_offset_m and min_step_m must be > 0")
    elif step > max_offset:
        problems.append("step_m must not exceed max_offset_m")
    if not 0 <= iterations <= MAX_ITERATIONS_LIMIT:
        problems.append(f"max_iterations must be between 0 and {MAX_ITERATIONS_LIMIT}")
    if start_position is None or len(start_position) != 5:
        problems.append(f"get_stage must return [x, y, z, alpha, beta], got {start_position!r}")
        start_z = None
    else:
        start_z = float(start_position[Z])
    return {
        "start_position": None if start_position is None else [float(v) for v in start_position],
        "start_z_m": start_z,
        "step_m": step,
        "min_step_m": min_step,
        "max_offset_m": max_offset,
        "z_limits_m": None if start_z is None else [start_z - max_offset, start_z + max_offset],
        "max_iterations": iterations,
        "max_measurements": INITIAL_MEASUREMENTS + iterations,
        "metric": metric,
        "detectors": list(request.get("detectors") or ["HAADF"]),
        "problems": problems,
    }


def build_graph(tools, checkpointer=None, prefer: tuple[str, ...] = (), read_image: Optional[Callable[[str], Any]] = None):
    """Compile the graph. ``tools``: a list or dict of tools, or a ToolBox. A checkpointer is needed to pause.

    ``read_image``: data key -> 2D array (sync or async), needed only for the pixel scores.
    """
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)

    async def read_stage() -> list[float]:
        position = await box.call("get_stage")
        return [float(v) for v in position]

    async def move_to_z(state: StageZState, z: float) -> list[float]:
        """Move only z; refuse anything outside the hard limit. Returns the position read back."""
        p = state["plan"]
        lower, upper = p["z_limits_m"]
        if not lower - 1e-15 <= z <= upper + 1e-15:
            raise ToolCallError("stage_z_focus", f"z {z:.6g} m is outside the limits {lower:.6g}..{upper:.6g} m")
        target = list(p["start_position"])
        target[Z] = float(z)
        await box.call("move_stage", position=target)
        return await read_stage()

    async def measure(state: StageZState, score, z: float, trial: str) -> dict:
        actual = await move_to_z(state, z)
        key = await box.call("acquire_scanned_image", detector_list=state["plan"]["detectors"])
        value = await score(key)
        start_z = state["plan"]["start_z_m"]
        return {"trial": trial, "z_m": actual[Z], "offset_m": actual[Z] - start_z, "commanded_z_m": float(z), "score": value, "key": key}

    def scorer(state: StageZState):
        return maths.make_scorer(box, state["plan"]["metric"], read_image)

    def failed(exc: Exception, **extra) -> StageZState:
        return {"status": "failed", "errors": [str(exc)], **extra}

    async def plan(state: StageZState) -> StageZState:
        try:
            position = await read_stage()
        except ToolCallError as exc:
            return {"plan": plan_request(state.get("request", {}), None, read_image), **failed(exc)}
        p = plan_request(state.get("request", {}), position, read_image)
        return {"plan": p, "start_position": p["start_position"], "status": "planned", "history": [], "moved": False, "errors": []}

    def approve(state: StageZState) -> StageZState:
        p = state["plan"]
        if p["problems"]:
            return {"status": "failed", "errors": list(p["problems"])}
        answer = interrupt({
            "kind": "approve_stage_z_climb",
            "plan": {k: v for k, v in p.items() if k != "problems"},
            "ask": (f"Approve this stage-Z hill climb? It moves the stage in z only, between {p['z_limits_m'][0]:.6g} and "
                    f"{p['z_limits_m'][1]:.6g} m, for at most {p['max_measurements']} images, then to the best z found. "
                    "If anything fails it moves back to the start."),
        })
        approved = answer is True or (isinstance(answer, dict) and answer.get("approved") is True)
        return {"status": "approved" if approved else "declined"}

    async def explore(state: StageZState) -> StageZState:
        p, history = state["plan"], []
        score = scorer(state)
        start, step = p["start_z_m"], p["step_m"]
        try:
            for z, trial in ((start, "baseline"), (start - step, "initial -"), (start + step, "initial +")):
                history.append(await measure(state, score, z, trial))
        except ToolCallError as exc:
            return failed(exc, history=history, moved=True)
        return {"history": history, "trust_radius_m": step, "iteration": 0, "status": "explored", "moved": True}

    async def climb(state: StageZState) -> StageZState:
        p = state["plan"]
        history, radius, iteration = list(state["history"]), float(state["trust_radius_m"]), int(state["iteration"])
        if iteration >= p["max_iterations"]:
            return {"status": "converged", "stop_reason": "max_iterations"}
        scores = [h["score"] for h in history]
        proposal = maths.propose_next_z([h["z_m"] for h in history], scores, radius, p["start_z_m"], p["max_offset_m"])
        if proposal is None:
            return {"status": "converged", "stop_reason": "no unmeasured z inside the trust radius and limits"}
        try:
            entry = await measure(state, scorer(state), proposal, f"model {iteration}")
        except ToolCallError as exc:
            return failed(exc, history=history)
        history.append(entry)
        cap = min(p["step_m"] * 2, p["max_offset_m"])
        radius = maths.next_trust_radius(radius, maths.improved(entry["score"], max(scores)), cap)
        out: StageZState = {"history": history, "trust_radius_m": radius, "iteration": iteration + 1, "status": "climbing"}
        if radius < p["min_step_m"]:
            out.update(status="converged", stop_reason="trust radius below min_step_m")
        elif iteration + 1 >= p["max_iterations"]:
            out.update(status="converged", stop_reason="max_iterations")
        return out

    async def finalize(state: StageZState) -> StageZState:
        best = max(state["history"], key=lambda h: h["score"])
        try:
            actual = await move_to_z(state, best["commanded_z_m"])
        except ToolCallError as exc:
            return failed(exc)
        return {"final_z_m": actual[Z], "status": "done"}

    async def report(state: StageZState) -> StageZState:
        p, status = state.get("plan") or {}, state.get("status")
        errors = list(state.get("errors", []))
        restored = None
        if status == "failed" and state.get("moved") and state.get("start_position"):
            try:
                await box.call("move_stage", position=list(state["start_position"]))
                restored = (await read_stage())[Z]
            except ToolCallError as exc:
                errors.append(f"could not move back to the start: {exc}")
        final = status if status in ("failed", "declined") else "done"
        history = state.get("history", [])
        best = max(history, key=lambda h: h["score"]) if history else None
        out = {
            "status": final,
            "metric": p.get("metric"),
            "start_z_m": p.get("start_z_m"),
            "final_z_m": state.get("final_z_m") if final == "done" else restored,
            "restored_to_start": restored is not None,
            "best": None if best is None else {k: best[k] for k in ("z_m", "offset_m", "score", "key")},
            "improvement": None if best is None or not history[0]["score"] else best["score"] / history[0]["score"],
            "measurements": len(history),
            "iterations": state.get("iteration", 0),
            "stop_reason": state.get("stop_reason"),
            "trust_radius_m": state.get("trust_radius_m"),
            "history": [{k: h[k] for k in ("trial", "z_m", "offset_m", "score", "key")} for h in history],
            "errors": errors,
        }
        return {"report": out, "status": final, "errors": errors}

    graph = StateGraph(StageZState)
    for node in (plan, approve, explore, climb, finalize, report):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "approve" if s["status"] == "planned" else "report", ["approve", "report"])
    graph.add_conditional_edges("approve", lambda s: "explore" if s["status"] == "approved" else "report", ["explore", "report"])
    graph.add_conditional_edges("explore", lambda s: "climb" if s["status"] == "explored" else "report", ["climb", "report"])
    graph.add_conditional_edges("climb", lambda s: {"climbing": "climb", "converged": "finalize"}.get(s["status"], "report"),
                                ["climb", "finalize", "report"])
    graph.add_edge("finalize", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


REQUIRED_TOOLS = ("get_stage", "move_stage", "acquire_scanned_image")


def missing_tools(tools, prefer: tuple[str, ...] = (), metric: str = "fft_sharpness") -> list[str]:
    """Tools the graph needs and cannot find. ``resolution`` (ANALYSIS) is needed for fft_sharpness."""
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)
    needed = REQUIRED_TOOLS + (("resolution",) if metric == "fft_sharpness" else ())
    return [name for name in needed if not box.has(name)]


def describe(result: Optional[dict]) -> dict:
    """The run's report, or what is known so far."""
    result = result or {}
    return result.get("report") or {"status": result.get("status"), "plan": result.get("plan"), "errors": result.get("errors", [])}


WORKFLOW = {
    "name": "stage_z_focus",
    "what": ("Focus by moving the stage in z: a trust-region hill climb on an image score (notebook 14). "
             "Pauses once with the plan; moves z only, within start +- max_offset_m; returns to the start on any failure."),
    "inputs": {
        "request": {
            "step_m": f"first step and trust radius, m (default {DEFAULT_STEP_M})",
            "max_offset_m": f"hard limit from the starting z, m (default {DEFAULT_MAX_OFFSET_M})",
            "max_iterations": f"model-guided steps after the 3 first images (default {DEFAULT_MAX_ITERATIONS}, at most {MAX_ITERATIONS_LIMIT})",
            "min_step_m": "stop when the trust radius falls below this (default step_m / 16)",
            "metric": "fft_sharpness (ANALYSIS.resolution), or contrast / edge_sharpness when the graph can read pixels",
            "detectors": 'detectors for acquire_scanned_image (default ["HAADF"])',
        },
    },
}
