"""Correct the probe from CEOS tableaus, one approved correction at a time.

Graph::

    plan -> read_start -> measure -> score -> decide -> approve -> apply
                             ^                  ^  |       |  |       |
                             |                  |  |       |  |       |
                             |                  +--|-------+  |       | (rejected: next best
                             |                     |          |       |  from the same tableau)
                             +---------------------|----------|-------+ (applied: next round)
                                                   v          v
      any tool failure -> restore ------------> report <-- stopped (restore first if asked)
                                                   ^
                    target reached, no improvement, step limit, nothing left to correct

This is the correction loop of ``notebooks/01_Aberrations.ipynb``
(``run_correction``) as plain nodes, with no model inside it:

- ``plan`` checks the request.
- ``read_start`` reads the corrector's limits (``get_correction_limits``,
  optional). The corrector has no command that reads back its settings: the
  CEOS-RPC protocol has none, and ``get_aberrations_coeff_sim`` is twin-only
  and blocked over MCP. So the state at the start is known only as the first
  tableau of the run, and a restore undoes this run's own corrections.
- ``measure`` acquires a tableau (``acquire_tableau``, default "Enhanced 40":
  scoring needs third order, so a Fast tableau is refused at ``plan``).
- ``score`` scores it on the analysis server (``correction``): the
  aberration-free angle and how much correcting each aberration would gain.
- ``decide`` stops when the target is reached, when the angle has not improved
  by ``min_gain_mrad`` for ``patience`` measurements, after ``max_rounds``
  corrections, or when nothing has a positive gain. Otherwise it proposes the
  aberration with the largest gain, corrected by the notebook's damping factor
  times its measured value, and shrunk to the corrector's ``max_step`` and
  ``max_total_since_tableau`` limits so the server does not refuse it.
- ``approve`` pauses (``interrupt``) with the proposal and the score. Resume
  with ``"approve"``, ``"reject"`` or ``"stop"`` (or ``{"action": ...}``;
  ``{"action": "stop", "restore": true}`` also undoes the run's corrections).
  A rejection proposes the next best aberration from the same tableau.
- ``apply`` sends ``correct_aberration`` once, then the loop measures again.
- ``restore`` runs after any tool failure: it sends the negative of each
  correction this run applied, newest first, then measures a tableau to check
  the result against the first one.
- ``report`` summarises the run. Every path ends here.

Tools (any class prefix): ``acquire_tableau`` and ``correct_aberration`` on the
corrector, ``correction`` on the analysis server, and optionally
``get_correction_limits`` on the corrector.
"""

from __future__ import annotations

import json
import math
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .tools import ToolBox, ToolCallError

# Damping per aberration, from notebooks/01_Aberrations.ipynb: correct this
# fraction of the measured value in one step.
FACTORS = {"C1": 1.0, "A1": 1.0, "B2": 0.75, "A2": 0.75, "C3": 0.5, "S3": 0.75, "A3": 0.75,
           "A4": 0.5, "D4": 0.5, "B4": 0.3, "C5": 0.5, "A5": 0.5}
ROUND = {"C1", "C3", "C5"}  # no direction: sent as one value
DEFAULT_TABLEAU = "Enhanced 40"
MAX_ROUNDS_CAP = 15  # each round is five graph steps; the runner's recursion limit is 100
# A restore counts as checked when every aberration is back within this much of the first tableau.
RESTORE_RELATIVE_TOLERANCE = 0.02
RESTORE_ABSOLUTE_TOLERANCE_M = 1e-11
NO_READBACK = ("The corrector has no command that reads back its settings (CEOS-RPC has none; "
               "get_aberrations_coeff_sim is twin-only and blocked over MCP). The start state is "
               "the run's first tableau; a restore undoes this run's own corrections, newest first.")

WORKFLOW = {
    "name": "correction",
    "what": ("Correct the probe from CEOS tableaus: measure, score the aberration-free angle, propose the "
             "single correction that gains most, and apply it only after the operator approves. Repeats until "
             "the target angle is reached, the angle stops improving, or the round limit."),
    "inputs": {
        "tableau": f"Tableau to acquire, '<type> <angle mrad>', Standard or Enhanced (default '{DEFAULT_TABLEAU}')",
        "target_mrad": "Aberration-free angle to reach; default the analysis server's convergence semi-angle",
        "max_rounds": f"Most corrections to apply (default 6, at most {MAX_ROUNDS_CAP})",
        "patience": "Stop after this many measurements without improvement (default 2)",
        "min_gain_mrad": "Smallest angle increase that counts as an improvement (default 0.1)",
        "aberrations": "Names the graph may correct (default every scored one: C1 A1 B2 A2 C3 S3 A3 A4 D4 B4)",
        "factors": "Damping per aberration, overriding the notebook's, e.g. {\"B2\": 0.5}",
    },
}


class CorrectionRequest(TypedDict, total=False):
    tableau: str
    target_mrad: float
    max_rounds: int
    patience: int
    min_gain_mrad: float
    aberrations: list
    factors: dict


class CorrectionState(TypedDict, total=False):
    request: CorrectionRequest
    plan: dict
    status: str  # planned, ready, measured, scored, proposed, approved, rejected, applied, done, stopped, failed
    limits: dict
    initial: dict  # aberrations of the first tableau
    tableau: dict  # the latest tableau payload (small: about a dozen pairs)
    score: dict
    best_angle_mrad: float
    stale: int  # measurements in a row without improvement
    rounds: int  # corrections applied
    proposal: dict
    rejected: list  # rejected this tableau
    stop_reason: str
    restore: dict
    applied: Annotated[list, operator.add]
    history: Annotated[list, operator.add]
    decisions: Annotated[list, operator.add]
    report: dict
    errors: Annotated[list, operator.add]


def plan_request(request: CorrectionRequest) -> dict:
    tableau = " ".join(str(request.get("tableau", DEFAULT_TABLEAU)).split())
    problems = []
    parts = tableau.split()
    if len(parts) != 2 or parts[0] not in ("Standard", "Enhanced"):
        problems.append("tableau must be 'Standard <angle>' or 'Enhanced <angle>': scoring needs third order, which Fast does not measure")
    else:
        try:
            if not float(parts[1]) > 0:
                problems.append("tableau angle must be > 0 mrad")
        except ValueError:
            problems.append(f"tableau angle {parts[1]!r} is not a number")
    max_rounds = int(request.get("max_rounds", 6))
    if not 0 <= max_rounds <= MAX_ROUNDS_CAP:
        problems.append(f"max_rounds must be between 0 and {MAX_ROUNDS_CAP}")
    patience = int(request.get("patience", 2))
    if patience < 1:
        problems.append("patience must be >= 1")
    target = request.get("target_mrad")
    if target is not None and not float(target) > 0:
        problems.append("target_mrad must be > 0")
    factors = {**FACTORS, **{str(k): float(v) for k, v in (request.get("factors") or {}).items()}}
    if any(not 0 < f <= 1 for f in factors.values()):
        problems.append("factors must be in (0, 1]")
    allowed = [str(a) for a in request.get("aberrations") or [a for a in FACTORS if a not in ("C5", "A5")]]
    unknown = sorted(set(allowed) - set(FACTORS))
    if unknown:
        problems.append(f"unknown aberrations: {unknown}")
    return {"tableau": tableau, "target_mrad": None if target is None else float(target), "max_rounds": max_rounds,
            "patience": patience, "min_gain_mrad": float(request.get("min_gain_mrad", 0.1)), "aberrations": allowed,
            "factors": factors, "problems": problems}


def _size(values: list[float]) -> float:
    return math.hypot(*values)


def propose(score: dict, aberrations: dict, plan: dict, limits: dict, exclude=()) -> dict | None:
    """The next correction: the allowed aberration with the largest positive gain, damped and kept within limits."""
    gains = score.get("gain_if_corrected_mrad") or {}
    pool = [n for n, g in gains.items() if g > 0 and n in aberrations and n in plan["aberrations"] and n not in exclude]
    if not pool:
        return None
    name = max(pool, key=lambda n: (gains[n], n))
    measured = [float(v) for v in aberrations[name]]
    factor = plan["factors"].get(name, 1.0)
    values = [factor * v for v in (measured[:1] if name in ROUND else measured[:2])]
    bounds = [b for b in ((limits.get("max_step") or {}).get(name), (limits.get("max_total_since_tableau") or {}).get(name)) if b]
    limit = min(bounds) if bounds else None
    size = _size(values)
    clipped = limit is not None and size > limit
    if clipped:
        scale = limit / size * (1 - 1e-9)  # stay just inside: the server refuses size > limit
        values = [v * scale for v in values]
    return {"aberration": name, "measured": measured, "factor": factor, "values": values, "size": _size(values),
            "limit": limit, "clipped": clipped, "gain_if_fully_corrected_mrad": gains[name]}


def correction_argument(name: str, values: list[float]) -> str:
    """The text correct_aberration takes: 'C1 x' for round aberrations, 'B2 x y' otherwise."""
    return " ".join([name, *(repr(float(v)) for v in values)])


def parse_answer(answer: Any) -> tuple[str, bool]:
    """(action, restore) from the operator's answer. Anything not clearly an approval or a stop is a rejection."""
    restore = False
    if isinstance(answer, dict):
        restore = bool(answer.get("restore"))
        if "action" in answer:
            answer = answer["action"]
        elif "approved" in answer:
            answer = answer["approved"] is True
    if answer is True:
        return "approve", restore
    action = str(answer).strip().lower() if isinstance(answer, str) else "reject"
    return (action if action in ("approve", "reject", "stop") else "reject"), restore


def build_graph(tools, checkpointer=None, prefer: tuple[str, ...] = ()):
    """Compile the graph. ``tools``: a list or dict of tools, or a ToolBox. A checkpointer is needed to pause."""
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)

    def failed(exc: Exception) -> CorrectionState:
        return {"status": "failed", "errors": [str(exc)]}

    def plan(state: CorrectionState) -> CorrectionState:
        p = plan_request(state.get("request", {}))
        if p["problems"]:
            return {"plan": p, "status": "failed", "errors": p["problems"]}
        return {"plan": p, "status": "planned", "rounds": 0, "stale": 0, "rejected": []}

    async def read_start(state: CorrectionState) -> CorrectionState:
        limits: dict = {}
        if box.has("get_correction_limits"):
            try:
                limits = await box.call("get_correction_limits")
            except ToolCallError as exc:
                return failed(exc)
        return {"limits": limits if isinstance(limits, dict) else {}, "status": "ready"}

    async def measure(state: CorrectionState) -> CorrectionState:
        try:
            payload = await box.call("acquire_tableau", arg=state["plan"]["tableau"])
        except ToolCallError as exc:
            return failed(exc)
        if not isinstance(payload, dict) or "error" in payload:
            return failed(ToolCallError("acquire_tableau", f"unexpected reply {str(payload)[:200]}"))
        aberrations = (payload.get("result") or {}).get("aberrations") or {}
        out: CorrectionState = {"tableau": payload, "status": "measured", "rejected": []}
        if not state.get("initial"):
            out["initial"] = aberrations
        return out

    async def score(state: CorrectionState) -> CorrectionState:
        try:
            result = await box.call("correction", arg=json.dumps(state["tableau"]))
        except ToolCallError as exc:
            return failed(exc)
        if not isinstance(result, dict) or "aberration_free_angle_mrad" not in result:
            return failed(ToolCallError("correction", f"unexpected reply {str(result)[:200]}"))
        p = state["plan"]
        angle = float(result["aberration_free_angle_mrad"])
        best = state.get("best_angle_mrad")
        stale = 0 if best is None or angle >= best + p["min_gain_mrad"] else state.get("stale", 0) + 1
        target = p["target_mrad"] if p["target_mrad"] is not None else result.get("target_mrad")
        entry = {"round": state.get("rounds", 0), "angle_mrad": angle, "target_mrad": target,
                 "passed": None if target is None else angle >= target, "best_next": result.get("best_next_correction")}
        return {"score": {**result, "target_mrad": target}, "best_angle_mrad": angle if best is None else max(best, angle),
                "stale": stale, "history": [entry], "status": "scored"}

    def decide(state: CorrectionState) -> CorrectionState:
        p, s = state["plan"], state["score"]
        target = s.get("target_mrad")
        if target is not None and s["aberration_free_angle_mrad"] >= target:
            return {"status": "done", "stop_reason": "target_reached"}
        if state.get("rounds", 0) >= p["max_rounds"]:
            return {"status": "done", "stop_reason": "step_limit"}
        if state.get("stale", 0) >= p["patience"]:
            return {"status": "done", "stop_reason": "no_improvement"}
        aberrations = (state["tableau"].get("result") or {}).get("aberrations") or {}
        proposal = propose(s, aberrations, p, state.get("limits") or {}, exclude=state.get("rejected", []))
        if proposal is None:
            return {"status": "done", "stop_reason": "all_rejected" if state.get("rejected") else "nothing_to_correct"}
        return {"proposal": proposal, "status": "proposed"}

    def approve(state: CorrectionState) -> CorrectionState:
        s, proposal = state["score"], state["proposal"]
        answer = interrupt({
            "kind": "approve_correction",
            "round": state.get("rounds", 0) + 1,
            "proposal": proposal,
            "command": correction_argument(proposal["aberration"], proposal["values"]),
            "score": {k: s.get(k) for k in ("aberration_free_angle_mrad", "target_mrad", "passed", "gain_if_corrected_mrad")},
            "history": state.get("history", []),
            "ask": "Apply this correction? It changes the corrector. Answer approve, reject (try the next best) or stop.",
        })
        action, restore = parse_answer(answer)
        decision = {"round": state.get("rounds", 0) + 1, "aberration": proposal["aberration"], "action": action}
        if action == "approve":
            return {"status": "approved", "decisions": [decision]}
        if action == "stop":
            return {"status": "stopped", "stop_reason": "operator_stop", "decisions": [decision],
                    "restore": {"requested": restore}}
        return {"status": "rejected", "rejected": [*state.get("rejected", []), proposal["aberration"]], "decisions": [decision]}

    async def apply(state: CorrectionState) -> CorrectionState:
        proposal = state["proposal"]
        try:
            await box.call("correct_aberration", arg=correction_argument(proposal["aberration"], proposal["values"]))
        except ToolCallError as exc:
            return failed(exc)
        done = {"round": state.get("rounds", 0) + 1, "aberration": proposal["aberration"], "values": proposal["values"],
                "clipped": proposal["clipped"]}
        return {"applied": [done], "rounds": state.get("rounds", 0) + 1, "status": "applied"}

    async def restore(state: CorrectionState) -> CorrectionState:
        out = {"attempted": True, "note": NO_READBACK, "undone": [], "failed": [], "checked": None}
        for entry in reversed(state.get("applied", [])):
            inverse = [-v for v in entry["values"]]
            try:
                await box.call("correct_aberration", arg=correction_argument(entry["aberration"], inverse))
                out["undone"].append({"aberration": entry["aberration"], "values": inverse})
            except ToolCallError as exc:
                out["failed"].append({"aberration": entry["aberration"], "values": inverse, "error": str(exc)})
        initial = state.get("initial") or {}
        if out["undone"] and initial:
            try:
                payload = await box.call("acquire_tableau", arg=state["plan"]["tableau"])
                now = (payload.get("result") or {}).get("aberrations") or {}
                off = {n: [a - b for a, b in zip(now[n], v)] for n, v in initial.items() if n in now}
                bad = {n: d for n, d in off.items()
                       if _size(d) > RESTORE_RELATIVE_TOLERANCE * _size(initial[n]) + RESTORE_ABSOLUTE_TOLERANCE_M}
                out["checked"] = {"back_to_start": not bad and not out["failed"], "still_off": bad}
            except (ToolCallError, AttributeError, TypeError) as exc:
                out["checked"] = {"back_to_start": None, "error": f"could not measure after the restore: {exc}"}
        elif not state.get("applied"):
            out["note"] = "Nothing to undo: this run applied no corrections."
        return {"restore": {**(state.get("restore") or {}), **out}}

    def report(state: CorrectionState) -> CorrectionState:
        history = state.get("history", [])
        final = state.get("status") if state.get("status") in ("failed", "stopped") else "done"
        out = {
            "status": final,
            "stop_reason": state.get("stop_reason") or ("tool_failure" if final == "failed" else None),
            "tableau": (state.get("plan") or {}).get("tableau"),
            "target_mrad": (state.get("score") or {}).get("target_mrad"),
            "start_angle_mrad": history[0]["angle_mrad"] if history else None,
            "final_angle_mrad": history[-1]["angle_mrad"] if history else None,
            "best_angle_mrad": state.get("best_angle_mrad"),
            "corrections_applied": state.get("rounds", 0),
            "applied": state.get("applied", []),
            "decisions": state.get("decisions", []),
            "history": history,
            "limits": state.get("limits"),
            "restore": state.get("restore") or {"attempted": False, "note": NO_READBACK},
            "errors": state.get("errors", []),
        }
        if final == "failed" and history and state.get("applied"):
            out["final_angle_mrad_note"] = "measured before the failure and the restore"
        return {"report": out, "status": final}

    def after_stop(s: CorrectionState) -> str:
        return "restore" if (s.get("restore") or {}).get("requested") else "report"

    routes = {
        "plan": lambda s: "read_start" if s["status"] == "planned" else "report",
        "read_start": lambda s: "measure" if s["status"] == "ready" else "restore",
        "measure": lambda s: "score" if s["status"] == "measured" else "restore",
        "score": lambda s: "decide" if s["status"] == "scored" else "restore",
        "decide": lambda s: "approve" if s["status"] == "proposed" else "report",
        "approve": lambda s: {"approved": "apply", "rejected": "decide"}.get(s["status"]) or after_stop(s),
        "apply": lambda s: "measure" if s["status"] == "applied" else "restore",
    }
    graph = StateGraph(CorrectionState)
    for node in (plan, read_start, measure, score, decide, approve, apply, restore, report):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", routes["plan"], ["read_start", "report"])
    graph.add_conditional_edges("read_start", routes["read_start"], ["measure", "restore"])
    graph.add_conditional_edges("measure", routes["measure"], ["score", "restore"])
    graph.add_conditional_edges("score", routes["score"], ["decide", "restore"])
    graph.add_conditional_edges("decide", routes["decide"], ["approve", "report"])
    graph.add_conditional_edges("approve", routes["approve"], ["apply", "decide", "restore", "report"])
    graph.add_conditional_edges("apply", routes["apply"], ["measure", "restore"])
    graph.add_edge("restore", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


REQUIRED_TOOLS = ("acquire_tableau", "correction", "correct_aberration")


def missing_tools(tools, prefer: tuple[str, ...] = ()) -> list[str]:
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)
    return [name for name in REQUIRED_TOOLS if not box.has(name)]


def describe(result: dict | None) -> dict:
    """The run's report, or what is known so far."""
    result = result or {}
    return result.get("report") or {"status": result.get("status"), "plan": result.get("plan"),
                                    "proposal": result.get("proposal"), "history": result.get("history", []),
                                    "errors": result.get("errors", [])}
