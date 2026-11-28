"""MAPED beam-tilt 4D-STEM series (notebook 15) as a LangGraph graph.

Graph::

    plan -> read_start -> approve -> scan -> ... -> scan -> restore -> [analyse] -> report
              | (bad      | (fails)   | (declined)  | (fails)            ^                ^
              |  request) v           v             v                    |                |
              +-------> report      report        restore ---------------+----------------+

- ``plan`` checks the request and lists the beam tilts: by default the
  notebook's five (untilted, then +x, +y, -x, -y at ``radius_rad``), or the
  ``tilts_rad`` list given.
- ``read_start`` reads the starting beam tilt.
- ``approve`` pauses (``interrupt``) with the tilts and a time estimate. Resume
  with ``{"approved": true}`` or ``{"approved": false}``.
- ``scan`` does one tilt per graph step, so a checkpointer can resume after an
  interruption: ``set_beam_tilt`` then ``acquire_scanned_data_advanced``.
- ``restore`` puts back the starting beam tilt, on success and on failure.
- ``analyse`` runs only when ``build_graph`` is given an ``analyse`` function
  and the series finished. The notebook's registration and averaging use
  quantEM, an optional extra (``uv sync --extra diffraction``), so they stay
  out of this module: pass ``analyse(keys, tilts_rad) -> dict`` to run them
  (or anything else) on the saved data. Its small summary dict goes in the
  report; the datacubes never pass through the model.
- ``report`` returns one data key per tilt.

Scan size and dwell time are SCAN attributes that MCP cannot set; the
operator sets them before the run.

Tools (any instrument prefix): ``get_beam_tilt``, ``set_beam_tilt``,
``acquire_scanned_data_advanced``.
"""

from __future__ import annotations

import math
import operator
from typing import Annotated, Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .tools import ToolBox, ToolCallError

DEFAULT_RADIUS_RAD = 30e-3
MAX_TILT_RAD = 50e-3  # refuse beam tilts beyond this
MAX_TILTS = 40
DEFAULT_SECONDS_PER_SCAN = 60.0  # set from scan points x dwell time plus overhead

WORKFLOW = {
    "name": "maped",
    "what": "MAPED: acquire a 4D-STEM scan at each of several beam tilts (default: untilted and four tilts at 30 mrad) "
    "and return one data key per tilt. Pauses for approval before the beam tilt changes and always restores it. "
    "Set the scan size and dwell time beforehand.",
    "inputs": {"request": {"radius_rad": "float (default 0.03, at most 0.05)",
                           "tilts_rad": "list of [tx, ty] in radians (optional; replaces the default five)",
                           "seconds_per_scan": "float (for the time estimate)"}},
}


class MapedRequest(TypedDict, total=False):
    radius_rad: float
    tilts_rad: list
    seconds_per_scan: float


class MapedState(TypedDict, total=False):
    request: MapedRequest
    plan: dict
    status: str  # planned, ready, approved, declined, scanning, acquired, restored, analysed, done, failed
    start_tilt: list
    moved: bool
    index: int
    scans: Annotated[list, operator.add]
    restored: bool
    analysis: dict
    report: dict
    errors: Annotated[list, operator.add]


def default_tilts(radius: float) -> list[list[float]]:
    return [[0.0, 0.0], [radius, 0.0], [0.0, radius], [-radius, 0.0], [0.0, -radius]]


def plan_request(request: MapedRequest) -> dict:
    problems = []
    radius = float(request.get("radius_rad", DEFAULT_RADIUS_RAD))
    tilts = request.get("tilts_rad")
    try:
        tilts = default_tilts(radius) if tilts is None else [[float(t[0]), float(t[1])] for t in tilts if len(t) == 2]
    except (TypeError, ValueError, IndexError):
        tilts = []
    if request.get("tilts_rad") is not None and len(tilts) != len(request["tilts_rad"]):
        problems.append("tilts_rad must be a list of [tx, ty] pairs in radians")
    if not tilts:
        problems.append("no beam tilts to acquire")
    if len(tilts) > MAX_TILTS:
        problems.append(f"at most {MAX_TILTS} tilts")
    too_far = [t for t in tilts if math.hypot(*t) > MAX_TILT_RAD]
    if too_far:
        problems.append(f"beam tilts beyond {MAX_TILT_RAD * 1e3:g} mrad: {too_far}")
    seconds = float(request.get("seconds_per_scan", DEFAULT_SECONDS_PER_SCAN)) * len(tilts)
    return {
        "tilts_rad": tilts,
        "tilts_mrad": [[round(v * 1e3, 6) for v in t] for t in tilts],
        "scans": len(tilts),
        "estimated_minutes": round(seconds / 60.0, 1),
        "problems": problems,
    }


def build_graph(tools, checkpointer=None, prefer: tuple[str, ...] = (),
                analyse: Optional[Callable[[list, list], Any]] = None):
    """Compile the graph. ``tools``: a list or dict of tools, or a ToolBox. A checkpointer is needed to pause.

    ``analyse(keys, tilts_rad) -> dict``: optional step after the series (for
    example the notebook's quantEM registration and averaging). It may be async.
    """
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)

    def plan(state: MapedState) -> MapedState:
        p = plan_request(state.get("request", {}))
        if p["problems"]:
            return {"plan": p, "status": "failed", "errors": p["problems"]}
        return {"plan": p, "status": "planned"}

    async def read_start(state: MapedState) -> MapedState:
        try:
            tilt = [float(v) for v in await box.call("get_beam_tilt")]
        except (ToolCallError, TypeError, ValueError) as exc:
            return {"status": "failed", "errors": [str(exc)]}
        if len(tilt) != 2:
            return {"status": "failed", "errors": [f"expected a beam tilt [tx, ty], got {tilt}"]}
        return {"start_tilt": tilt, "status": "ready"}

    def approve(state: MapedState) -> MapedState:
        p = state["plan"]
        answer = interrupt({
            "kind": "approve_maped",
            "plan": p,
            "start_tilt_mrad": [v * 1e3 for v in state["start_tilt"]],
            "ask": f"Approve {p['scans']} 4D-STEM scans at these beam tilts? The starting beam tilt is restored at the end.",
        })
        approved = answer is True or (isinstance(answer, dict) and answer.get("approved") is True)
        return {"status": "approved" if approved else "declined", "index": 0}

    async def scan(state: MapedState) -> MapedState:
        tilts = state["plan"]["tilts_rad"]
        index = int(state.get("index", 0))
        record: dict[str, Any] = {"tilt_rad": tilts[index], "tilt_mrad": state["plan"]["tilts_mrad"][index]}
        try:
            await box.call("set_beam_tilt", tilt=tilts[index])
            record["key"] = str(await box.call("acquire_scanned_data_advanced"))
        except ToolCallError as exc:
            return {"status": "failed", "moved": True, "errors": [str(exc)], "scans": [{**record, "error": str(exc)}]}
        done = index + 1 >= len(tilts)
        return {"index": index + 1, "scans": [record], "moved": True, "status": "acquired" if done else "scanning"}

    async def restore(state: MapedState) -> MapedState:
        try:
            await box.call("set_beam_tilt", tilt=state["start_tilt"])
        except ToolCallError as exc:
            return {"restored": False, "status": "failed", "errors": [f"restore failed: {exc}"]}
        return {"restored": True, "status": "failed" if state.get("status") == "failed" else "restored"}

    async def analyse_node(state: MapedState) -> MapedState:
        scans = state.get("scans", [])
        try:
            result = analyse([s["key"] for s in scans], [s["tilt_rad"] for s in scans])
            if hasattr(result, "__await__"):
                result = await result
        except Exception as exc:  # the data is saved and the tilt restored; report the analysis error only
            return {"analysis": {"error": str(exc)}, "status": "analysed", "errors": [f"analysis failed: {exc}"]}
        return {"analysis": result if isinstance(result, dict) else {"result": result}, "status": "analysed"}

    def report(state: MapedState) -> MapedState:
        final = state["status"] if state.get("status") in ("failed", "declined") else "done"
        scans = state.get("scans", [])
        done = [s for s in scans if s.get("key")]
        out = {
            "status": final,
            "keys": [s["key"] for s in done],
            "tilts_mrad": [s["tilt_mrad"] for s in done],
            "scans": scans,
            "start_tilt_rad": state.get("start_tilt"),
            "restored": state.get("restored") if state.get("moved") else None,
            "analysis": state.get("analysis"),
            "errors": state.get("errors", []),
        }
        return {"report": out, "status": final}

    after_restore = ["report"] + (["analyse"] if analyse is not None else [])

    def route_restore(s: MapedState) -> str:
        return "analyse" if analyse is not None and s["status"] == "restored" else "report"

    graph = StateGraph(MapedState)
    for name, node in (("plan", plan), ("read_start", read_start), ("approve", approve), ("scan", scan),
                       ("restore", restore), ("report", report)):
        graph.add_node(name, node)
    if analyse is not None:
        graph.add_node("analyse", analyse_node)
        graph.add_edge("analyse", "report")
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "read_start" if s["status"] == "planned" else "report", ["read_start", "report"])
    graph.add_conditional_edges("read_start", lambda s: "approve" if s["status"] == "ready" else "report", ["approve", "report"])
    graph.add_conditional_edges("approve", lambda s: "scan" if s["status"] == "approved" else "report", ["scan", "report"])
    graph.add_conditional_edges("scan", lambda s: "scan" if s["status"] == "scanning" else "restore", ["scan", "restore"])
    graph.add_conditional_edges("restore", route_restore, after_restore)
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


REQUIRED_TOOLS = ("get_beam_tilt", "set_beam_tilt", "acquire_scanned_data_advanced")


def missing_tools(tools, prefer: tuple[str, ...] = ()) -> list[str]:
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)
    return [name for name in REQUIRED_TOOLS if not box.has(name)]


def describe(result: Optional[dict]) -> dict:
    """The run's report, or what is known so far."""
    result = result or {}
    return result.get("report") or {"status": result.get("status"), "plan": result.get("plan"), "errors": result.get("errors", [])}
