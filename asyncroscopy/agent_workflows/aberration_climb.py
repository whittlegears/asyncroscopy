"""Tune one aberration (C1 or A1) with the corrector: notebook 12's trust-region hill climb as a LangGraph graph.

Graph::

    plan -> approve -> explore -> climb -> finalize -> report
              |           |        ^  |        |
              | declined  |        |__|        |
              v           v  (any failure)     v
            report      report <------------- report

- ``plan`` checks the request. If the corrector offers ``get_correction_limits``
  (read only), its ``max_step`` for this aberration is read so that every
  correction is sent in pieces no larger than it.
- ``approve`` pauses (``interrupt``) with the plan: aberration, step, hard limit
  and the most measurements the climb can make. Resume with
  ``{"approved": true}`` or ``{"approved": false}``.
- ``explore`` measures at zero offset, then at the notebook's first probes:
  +-step for C1, eight points on a circle of radius step for A1.
- ``climb`` runs one iteration per pass and loops on itself without pausing: fit
  a quadratic surface to every measurement, move to the best predicted offset
  inside the trust radius and inside the hard limit (``max_offset_m`` from the
  starting value), measure, then grow or shrink the trust radius.
- ``finalize`` sets the best offset measured.
- ``report`` summarises the run. If anything failed after the first correction,
  it first undoes every correction sent, back to the starting value.

An offset is the sum of the corrections sent with ``correct_aberration``
(``"<name> <x> [y]"``, metres), as in the notebook. The score is
``fft_sharpness`` from the ANALYSIS device's ``resolution`` command, or one of the
notebook's pixel scores (``contrast``, ``entropy``) when the graph is built with
``read_image``.

Tools (any instrument prefix): ``correct_aberration``, ``acquire_scanned_image``,
``resolution`` for ``fft_sharpness``, and optionally ``get_correction_limits``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import climb as maths
from .tools import ToolBox, ToolCallError

# Components and the notebook's step sizes. C1 is round (one value); the others have a direction.
ABERRATIONS = {"C1": (1, 10e-9), "A1": (2, 2e-9), "A2": (2, 25e-9), "B2": (2, 25e-9)}
DEFAULT_MAX_ITERATIONS = 8
DEFAULT_MAX_OFFSET_STEPS = 5  # default hard limit: this many steps from the starting value
MAX_ITERATIONS_LIMIT = 60  # each iteration is one graph step; WorkflowRunner's recursion_limit is 100
LIMIT_TOLERANCE = 1e-6  # relative slack when checking the hard limit


class AberrationRequest(TypedDict, total=False):
    name: str  # C1 or A1 (A2, B2 also accepted, as in the notebook)
    step_m: float
    max_offset_m: float  # hard limit on the total correction from the starting value
    max_iterations: int
    min_step_m: float  # stop when the trust radius falls below this; default step_m / 16
    metric: str  # fft_sharpness, contrast or entropy
    detectors: list


class AberrationState(TypedDict, total=False):
    request: AberrationRequest
    plan: dict
    status: str  # planned, approved, declined, explored, climbing, converged, done, failed
    applied_m: list  # corrections sent so far (the current offset)
    history: list  # one entry per measurement: offset_m, score, key, trial
    trust_radius_m: float
    iteration: int
    stop_reason: str
    report: dict
    errors: list


def plan_request(request: AberrationRequest, limits: Optional[dict] = None, read_image=None) -> dict:
    name = str(request.get("name", "C1"))
    problems = []
    if name not in ABERRATIONS:
        problems.append(f"name must be one of {sorted(ABERRATIONS)}")
        components, default_step = 1, 10e-9
    else:
        components, default_step = ABERRATIONS[name]
    step = float(request.get("step_m", default_step))
    max_offset = float(request.get("max_offset_m", DEFAULT_MAX_OFFSET_STEPS * step))
    min_step = float(request.get("min_step_m") or step / 16)
    iterations = int(request.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    metric = maths.choose_metric(request.get("metric"), read_image)
    problems += maths.metric_problems(metric, read_image)
    if not (step > 0 and max_offset > 0 and min_step > 0):
        problems.append("step_m, max_offset_m and min_step_m must be > 0")
    elif step > max_offset:
        problems.append("step_m must not exceed max_offset_m")
    if not 0 <= iterations <= MAX_ITERATIONS_LIMIT:
        problems.append(f"max_iterations must be between 0 and {MAX_ITERATIONS_LIMIT}")
    max_step = ((limits or {}).get("max_step") or {}).get(name)
    max_total = ((limits or {}).get("max_total_since_tableau") or {}).get(name)
    initial = len(maths.initial_offsets(step, components)) + 1
    return {
        "name": name,
        "components": components,
        "step_m": step,
        "min_step_m": min_step,
        "max_offset_m": max_offset,
        "max_iterations": iterations,
        "max_measurements": initial + iterations,
        "metric": metric,
        "detectors": list(request.get("detectors") or ["HAADF"]),
        "corrector_max_step_m": None if max_step is None else float(max_step),
        "corrector_max_total_m": None if max_total is None else float(max_total),
        "problems": problems,
    }


def correction_text(name: str, values) -> str:
    return " ".join([name, *[f"{float(v):.12g}" for v in values]])


def build_graph(tools, checkpointer=None, prefer: tuple[str, ...] = (), read_image: Optional[Callable[[str], Any]] = None):
    """Compile the graph. ``tools``: a list or dict of tools, or a ToolBox. A checkpointer is needed to pause.

    ``read_image``: data key -> 2D array (sync or async), needed only for the pixel scores.
    """
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)

    async def apply(plan: dict, applied: list, delta) -> None:
        """Send ``delta`` as corrections no larger than the corrector's max_step; ``applied`` follows what was accepted."""
        size = math.hypot(*delta)
        if size == 0:
            return
        limit = plan.get("corrector_max_step_m")
        pieces = max(1, math.ceil(size / limit * (1 - 1e-9))) if limit else 1
        piece = [float(v) / pieces for v in delta]
        for _ in range(pieces):
            await box.call("correct_aberration", args=correction_text(plan["name"], piece))
            for i, v in enumerate(piece):
                applied[i] += v

    def check_limit(plan: dict, target) -> None:
        if math.hypot(*target) > plan["max_offset_m"] * (1 + LIMIT_TOLERANCE):
            raise ToolCallError("aberration_climb", f"{plan['name']} offset {list(target)} m is beyond max_offset_m {plan['max_offset_m']:.6g} m")

    async def measure(plan: dict, applied: list, score, target, trial: str) -> dict:
        check_limit(plan, target)
        await apply(plan, applied, [t - a for t, a in zip(target, applied)])
        key = await box.call("acquire_scanned_image", detector_list=plan["detectors"])
        return {"trial": trial, "offset_m": [float(v) for v in applied], "score": await score(key), "key": key}

    def scorer(plan: dict):
        return maths.make_scorer(box, plan["metric"], read_image)

    def failed(exc: Exception, **extra) -> AberrationState:
        return {"status": "failed", "errors": [str(exc)], **extra}

    async def plan(state: AberrationState) -> AberrationState:
        limits = None
        if box.has("get_correction_limits"):
            try:
                limits = await box.call("get_correction_limits")
            except ToolCallError as exc:
                return {"plan": plan_request(state.get("request", {}), None, read_image), **failed(exc)}
        p = plan_request(state.get("request", {}), limits if isinstance(limits, dict) else None, read_image)
        return {"plan": p, "applied_m": [0.0] * p["components"], "history": [], "status": "planned", "errors": []}

    def approve(state: AberrationState) -> AberrationState:
        p = state["plan"]
        if p["problems"]:
            return {"status": "failed", "errors": list(p["problems"])}
        answer = interrupt({
            "kind": "approve_aberration_climb",
            "plan": {k: v for k, v in p.items() if k != "problems"},
            "ask": (f"Approve this {p['name']} hill climb? It sends corrections of up to {p['max_offset_m']:.3g} m in total from "
                    f"the current value, for at most {p['max_measurements']} images, then sets the best value found. "
                    "If anything fails it undoes every correction."),
        })
        approved = answer is True or (isinstance(answer, dict) and answer.get("approved") is True)
        return {"status": "approved" if approved else "declined"}

    async def explore(state: AberrationState) -> AberrationState:
        p = state["plan"]
        applied, history, score = list(state["applied_m"]), [], scorer(p)
        try:
            history.append(await measure(p, applied, score, [0.0] * p["components"], "baseline"))
            for index, point in enumerate(maths.initial_offsets(p["step_m"], p["components"])):
                history.append(await measure(p, applied, score, point, f"initial {index}"))
        except ToolCallError as exc:
            return failed(exc, history=history, applied_m=applied)
        return {"history": history, "applied_m": applied, "trust_radius_m": p["step_m"], "iteration": 0, "status": "explored"}

    async def climb(state: AberrationState) -> AberrationState:
        p = state["plan"]
        applied, history = list(state["applied_m"]), list(state["history"])
        radius, iteration = float(state["trust_radius_m"]), int(state["iteration"])
        if iteration >= p["max_iterations"]:
            return {"status": "converged", "stop_reason": "max_iterations"}
        scores = [h["score"] for h in history]
        proposal = maths.propose_next_point([h["offset_m"] for h in history], scores, radius, p["max_offset_m"])
        if proposal is None:
            return {"status": "converged", "stop_reason": "no unmeasured offset inside the trust radius and limit"}
        try:
            entry = await measure(p, applied, scorer(p), [float(v) for v in proposal], f"model {iteration}")
        except ToolCallError as exc:
            return failed(exc, history=history, applied_m=applied)
        history.append(entry)
        radius = maths.next_trust_radius(radius, maths.improved(entry["score"], max(scores)), min(p["step_m"] * 2, p["max_offset_m"]))
        out: AberrationState = {"history": history, "applied_m": applied, "trust_radius_m": radius, "iteration": iteration + 1, "status": "climbing"}
        if radius < p["min_step_m"]:
            out.update(status="converged", stop_reason="trust radius below min_step_m")
        elif iteration + 1 >= p["max_iterations"]:
            out.update(status="converged", stop_reason="max_iterations")
        return out

    async def finalize(state: AberrationState) -> AberrationState:
        p, applied = state["plan"], list(state["applied_m"])
        best = max(state["history"], key=lambda h: h["score"])
        try:
            check_limit(p, best["offset_m"])
            await apply(p, applied, [b - a for b, a in zip(best["offset_m"], applied)])
        except ToolCallError as exc:
            return failed(exc, applied_m=applied)
        return {"applied_m": applied, "status": "done"}

    async def report(state: AberrationState) -> AberrationState:
        p, status = state.get("plan") or {}, state.get("status")
        errors = list(state.get("errors", []))
        applied = list(state.get("applied_m") or [])
        restored = False
        if status == "failed" and any(applied):
            try:
                await apply(p, applied, [-a for a in applied])
                restored = True
            except ToolCallError as exc:
                errors.append(f"could not undo the corrections; {p.get('name')} is still offset by {applied} m: {exc}")
        final = status if status in ("failed", "declined") else "done"
        history = state.get("history", [])
        best = max(history, key=lambda h: h["score"]) if history else None
        out = {
            "status": final,
            "name": p.get("name"),
            "metric": p.get("metric"),
            "offset_m": applied,  # the total correction left applied (zero after a failure is undone)
            "restored_to_start": restored,
            "best": None if best is None else {k: best[k] for k in ("offset_m", "score", "key")},
            "improvement": None if best is None or not history[0]["score"] else best["score"] / history[0]["score"],
            "measurements": len(history),
            "iterations": state.get("iteration", 0),
            "stop_reason": state.get("stop_reason"),
            "trust_radius_m": state.get("trust_radius_m"),
            "history": [{k: h[k] for k in ("trial", "offset_m", "score", "key")} for h in history],
            "errors": errors,
        }
        return {"report": out, "status": final, "applied_m": applied, "errors": errors}

    graph = StateGraph(AberrationState)
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


REQUIRED_TOOLS = ("correct_aberration", "acquire_scanned_image")


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
    "name": "aberration_climb",
    "what": ("Tune C1 or A1 with the corrector: a trust-region hill climb on an image score (notebook 12). "
             "Pauses once with the plan; stays within max_offset_m of the starting value; undoes every correction on any failure."),
    "inputs": {
        "request": {
            "name": "C1 or A1 (A2, B2 also accepted)",
            "step_m": "first step and trust radius, m (default: C1 10e-9, A1 2e-9, A2/B2 25e-9)",
            "max_offset_m": f"hard limit on the total correction, m (default {DEFAULT_MAX_OFFSET_STEPS} x step_m)",
            "max_iterations": f"model-guided steps after the first probes (default {DEFAULT_MAX_ITERATIONS}, at most {MAX_ITERATIONS_LIMIT})",
            "min_step_m": "stop when the trust radius falls below this (default step_m / 16)",
            "metric": "fft_sharpness (ANALYSIS.resolution), or contrast / entropy when the graph can read pixels",
            "detectors": 'detectors for acquire_scanned_image (default ["HAADF"])',
        },
    },
}
