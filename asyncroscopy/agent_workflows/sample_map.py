"""Map a sample, find candidate features while it maps, and review them with a person.

Graph::

    plan -> approve_scan -> acquire -> build_map -> shortlist -> review -> finalize
              |  (declined)        |  (failed)                    ^          |
              v                    v                              |__________| (more rounds)
             END                  END

- ``plan`` checks the request and works out the grid spec and a time estimate.
- ``approve_scan`` pauses (``interrupt``) with the plan. The run continues only
  when the operator resumes with ``{"approved": true}``; they may also change
  the request, which is planned again.
- ``acquire`` calls the instrument's ``acquire_map_grid`` once. The stage loop,
  the calibration test moves and the per-tile acquisition all happen on the
  instrument, not in the model. A calibration failure ends the run with the
  instrument's own message.
- ``build_map`` starts a live map on the mapping server, attaches the detector
  and adds the grid manifest in one call, so the manifest never passes through
  a model.
- ``shortlist`` picks a varied set of provisional findings: the strongest, then
  ones far from those already picked, so the reviewer does not see ten copies
  of the same feature.
- ``review`` pauses with the shortlist. The operator resumes with
  ``{"decisions": [{"id", "status": "accepted"|"rejected", "note"}], "actor":
  "human:<name>", "more": bool}``; decisions are written to the map's history.
  ``more`` asks for another shortlist from the findings not yet reviewed.
- ``finalize`` exports the zoomable bundle with the reviewed findings.

Tools needed (names as the MCP servers publish them, any instrument prefix):
``acquire_map_grid`` and optionally ``set_fov`` on the instrument;
``start_map``, ``attach_detector``, ``add_grid_manifest``, ``list_findings``,
``review_finding`` and ``finalize_map`` on the mapping server.
"""

from __future__ import annotations

import json
import math
import operator
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .tools import ToolBox, ToolCallError

DEFAULT_SECONDS_PER_TILE = 15.0  # move, settle, scan, save: replace with a measured value


class SampleMapRequest(TypedDict, total=False):
    name: str
    rows: int
    cols: int
    overlap: float
    fov_m: float  # set the field of view first; omit to keep the current one
    calibration_fov_m: float
    detector: dict  # {"name", "kind", "params"} for attach_detector
    shortlist_size: int
    seconds_per_tile: float


class SampleMapState(TypedDict, total=False):
    request: SampleMapRequest
    plan: dict
    status: str  # planned, declined, acquired, mapped, reviewing, done, failed
    manifest: dict
    map_id: str
    map_summary: dict
    shortlist: list
    reviewed_ids: Annotated[list, operator.add]
    decisions: Annotated[list, operator.add]
    more: bool
    bundle: dict
    errors: Annotated[list, operator.add]


def _plan(request: SampleMapRequest) -> dict:
    rows, cols = int(request.get("rows", 3)), int(request.get("cols", 3))
    overlap = float(request.get("overlap", 0.2))
    problems = []
    if not (1 <= rows and 1 <= cols and rows * cols <= 400):
        problems.append("rows x cols must be between 1 and 400")
    if not 0.0 < overlap < 0.6:
        problems.append("overlap must be between 0 and 0.6")
    fov = request.get("fov_m")
    if fov is not None and not float(fov) > 0:
        problems.append("fov_m must be > 0")
    spec: dict[str, Any] = {"rows": rows, "cols": cols, "overlap": overlap}
    if request.get("calibration_fov_m"):
        spec["calibration_fov_m"] = float(request["calibration_fov_m"])
    seconds = float(request.get("seconds_per_tile", DEFAULT_SECONDS_PER_TILE)) * (rows * cols + 3)  # + 3 calibration images
    plan = {
        "spec": spec,
        "fov_m": None if fov is None else float(fov),
        "tiles": rows * cols,
        "estimated_minutes": round(seconds / 60.0, 1),
        "span_fov": [1 + (rows - 1) * (1 - overlap), 1 + (cols - 1) * (1 - overlap)],
        "problems": problems,
    }
    if fov is not None:
        plan["span_m"] = [s * float(fov) for s in plan["span_fov"]]
    return plan


def diverse_shortlist(findings: list[dict], k: int) -> list[dict]:
    """Strongest finding first, then each next one as far as possible from those picked (by canvas position)."""
    pool = [f for f in findings if f.get("canvas_xy") is not None]
    if not pool or k <= 0:
        return []
    pool.sort(key=lambda f: -(f.get("score") or 0.0))
    picked = [pool.pop(0)]
    while pool and len(picked) < k:
        def gap(f):
            return min(math.dist(f["canvas_xy"], p["canvas_xy"]) for p in picked)

        best = max(pool, key=lambda f: (gap(f), f.get("score") or 0.0))
        pool.remove(best)
        picked.append(best)
    return picked


def build_graph(tools, checkpointer=None, prefer: tuple[str, ...] = ()):
    """Compile the sample-map graph. ``tools``: a list or dict of tools; ``checkpointer`` is required to pause."""
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)

    def plan(state: SampleMapState) -> SampleMapState:
        p = _plan(state.get("request", {}))
        return {"plan": p, "status": "planned"}

    def approve_scan(state: SampleMapState) -> SampleMapState:
        p = state["plan"]
        if p["problems"]:
            return {"status": "failed", "errors": p["problems"]}
        answer = interrupt({"kind": "approve_scan", "plan": p, "ask": "Approve this grid scan? It will move the stage."})
        if isinstance(answer, dict) and answer.get("changes"):
            request = {**state.get("request", {}), **answer["changes"]}
            return {"request": request, "plan": _plan(request), "status": "replanned"}
        approved = answer is True or (isinstance(answer, dict) and answer.get("approved") is True)
        return {"status": "approved" if approved else "declined"}

    async def acquire(state: SampleMapState) -> SampleMapState:
        p = state["plan"]
        try:
            if p.get("fov_m") is not None:
                await box.call("set_fov", fov=p["fov_m"])
            manifest = await box.call("acquire_map_grid", spec=json.dumps(p["spec"]))
        except ToolCallError as exc:
            return {"status": "failed", "errors": [str(exc)]}
        if isinstance(manifest, str):
            manifest = json.loads(manifest)
        return {"manifest": manifest, "status": "acquired"}

    async def build_map(state: SampleMapState) -> SampleMapState:
        request, manifest = state.get("request", {}), state["manifest"]
        try:
            started = await box.call(
                "start_map",
                name=request.get("name", "sample_map"),
                overlap=float(manifest.get("overlap", state["plan"]["spec"]["overlap"])),
                pixel_size_nm=manifest.get("pixel_size_nm"),
                stage_origin_m=manifest.get("stage_origin_m"),
            )
            map_id = started["map_id"]
            detector = request.get("detector") or {"name": "blob", "kind": "blob", "params": {}}
            await box.call("attach_detector", map_id=map_id, name=detector.get("name", "blob"), kind=detector.get("kind", "blob"), params=detector.get("params", {}), run_now=False)
            summary = await box.call("add_grid_manifest", map_id=map_id, manifest_json=json.dumps(manifest))
        except ToolCallError as exc:
            return {"status": "failed", "errors": [str(exc)]}
        return {"map_id": map_id, "map_summary": summary, "status": "mapped"}

    async def shortlist(state: SampleMapState) -> SampleMapState:
        k = int(state.get("request", {}).get("shortlist_size", 5))
        findings = await box.call("list_findings", map_id=state["map_id"], status="provisional", limit=1000)
        seen = set(state.get("reviewed_ids", []))
        candidates = [f for f in findings if f["id"] not in seen and f.get("placed", True)]
        return {"shortlist": diverse_shortlist(candidates, k), "status": "reviewing"}

    async def review(state: SampleMapState) -> SampleMapState:
        items = state.get("shortlist", [])
        if not items:
            return {"more": False}
        answer = interrupt(
            {
                "kind": "review_findings",
                "map_id": state["map_id"],
                "shortlist": items,
                "ask": "Accept or reject each finding. Use get_crop on the mapping server to see the raw pixels.",
            }
        )
        answer = answer if isinstance(answer, dict) else {}
        actor = answer.get("actor", "human:operator")
        applied, errors = [], []
        for decision in answer.get("decisions", []):
            try:
                await box.call("review_finding", map_id=state["map_id"], finding_id=decision["id"], actor=actor, status=decision["status"], note=decision.get("note", ""))
                applied.append({**decision, "actor": actor})
            except (ToolCallError, KeyError) as exc:
                errors.append(f"review {decision}: {exc}")
        return {"decisions": applied, "reviewed_ids": [i["id"] for i in items], "more": bool(answer.get("more")), "errors": errors}

    async def finalize(state: SampleMapState) -> SampleMapState:
        try:
            bundle = await box.call("finalize_map", map_id=state["map_id"], name=state.get("request", {}).get("name"))
        except ToolCallError as exc:
            return {"status": "failed", "errors": [str(exc)]}
        return {"bundle": bundle, "status": "done"}

    def after_approve(state: SampleMapState) -> str:
        return {"approved": "acquire", "replanned": "approve_scan"}.get(state["status"], END)

    def after_acquire(state: SampleMapState) -> str:
        return "build_map" if state["status"] == "acquired" else END

    def after_build(state: SampleMapState) -> str:
        return "shortlist" if state["status"] == "mapped" else END

    def after_review(state: SampleMapState) -> str:
        return "shortlist" if state.get("more") else "finalize"

    graph = StateGraph(SampleMapState)
    for node in (plan, approve_scan, acquire, build_map, shortlist, review, finalize):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "approve_scan")
    graph.add_conditional_edges("approve_scan", after_approve, ["acquire", "approve_scan", END])
    graph.add_conditional_edges("acquire", after_acquire, ["build_map", END])
    graph.add_conditional_edges("build_map", after_build, ["shortlist", END])
    graph.add_edge("shortlist", "review")
    graph.add_conditional_edges("review", after_review, ["shortlist", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


REQUIRED_TOOLS = ("acquire_map_grid", "start_map", "attach_detector", "add_grid_manifest", "list_findings", "review_finding", "finalize_map")


def missing_tools(tools, prefer: tuple[str, ...] = ()) -> list[str]:
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)
    return [name for name in REQUIRED_TOOLS if not box.has(name)]


def describe(result: Optional[dict]) -> dict:
    """A short summary of a run's state for a model or a person."""
    result = result or {}
    return {
        "status": result.get("status"),
        "plan": result.get("plan"),
        "map_id": result.get("map_id"),
        "map_summary": {k: v for k, v in (result.get("map_summary") or {}).items() if k != "new_findings"},
        "decisions": result.get("decisions", []),
        "bundle": result.get("bundle"),
        "errors": result.get("errors", []),
    }
