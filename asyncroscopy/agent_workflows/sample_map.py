"""Map a sample, stitch it on the pyTEMlib analysis server, and review what it found with a person.

Graph::

    plan -> approve_scan -> acquire -> collect -> assemble -> score -> shortlist -> review -> report
              |  (declined)     | (any step fails: report with the error)          ^          |
              v                 v                                                  |__________| more rounds
             END              report

There is no mapping server in asyncroscopy. The instrument does the grid scan;
the pyTEMlib analysis MCP server (``python -m pyTEMlib.analysis_server``) does
the stitching, scoring, points of interest and notes. This graph joins them, and
no model sits inside the loop:

- ``plan`` checks the request and works out the grid spec and a time estimate.
- ``approve_scan`` pauses (``interrupt``) with the plan, since the scan moves the
  stage. Resume with ``{"approved": true}``, ``{"approved": false}``, or
  ``{"changes": {...}}`` to plan again.
- ``acquire`` calls the instrument's ``acquire_map_grid`` once. It measures which
  way the stage moves the image, steps the stage and saves every tile.
- ``collect`` turns the grid into a pyTEMlib map run, with tile centres in the
  image frame pyTEMlib expects (columns +x, rows +y), converted with the grid's
  calibration. With ``run_dir`` it writes the run file itself (the analysis
  server must be started with ``--local-root`` on that folder). Otherwise it
  calls ``collect_map_run`` on the server, which reads the tiles from Tiled.
- ``assemble`` stitches with ``assemble_map`` and waits for the job.
- ``score`` ranks regions with ``score_map`` and lists the candidates.
- ``shortlist`` picks a varied set: the best candidate, then each next one as far
  as possible from those picked.
- ``review`` pauses with the shortlist. Resume with ``{"actor": "human:<name>",
  "decisions": [{"id": <poi id>, "status": "accepted"|"rejected", "note": ...}],
  "more": bool}``. Each decision becomes a note on the map (``annotate``), so it
  is kept with its author in the map's history.
- ``report`` summarises the stitch, the candidates and the decisions.

Tools (any instrument prefix): ``acquire_map_grid`` and optionally ``set_fov`` on
the instrument bridge; ``collect_map_run`` (unless ``run_dir``),
``assemble_map``, ``analysis_job_status``, ``analysis_job_results``,
``score_map``, ``list_pois``, ``annotate`` and ``map_info`` on pyTEMlib.
"""

from __future__ import annotations

import json
import math
import operator
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from asyncroscopy.mapping.run_file import image_frame_positions, write_map_run

from .tools import ToolBox, ToolCallError

DEFAULT_SECONDS_PER_TILE = 15.0  # move, settle, scan, save: replace with a measured value
DEFAULT_ANALYSIS_WAIT_S = 900.0
JOB_DONE = {"done", "failed", "aborted", "error", "cancelled"}


class SampleMapRequest(TypedDict, total=False):
    name: str
    rows: int
    cols: int
    overlap: float
    fov_m: float  # set the field of view first; omit to keep the current one
    calibration_fov_m: float  # do the stage test moves at this field of view
    scorer: str  # one of pyTEMlib's list_scorers(); default "baseline"
    top_k: int  # candidates the scorer adds
    shortlist_size: int
    seconds_per_tile: float
    analysis_wait_s: float
    stage_sigma_m: float  # how well the stage repeats; bounds pyTEMlib's overlap search
    scan_linear: Any  # 2 x 2 matrix, "measure", or omit for the default below


# pyTEMlib fits its own scan stretch and turn (scan_linear) once 9 tiles match. On
# a 3 x 3 grid of atomic-resolution twin tiles that fit came out as a 0.2-0.9 degree
# turn that is not there, from matches one lattice period off, and placed tiles up
# to 9 px wrong; with the identity every tile was within 0.6 px (tests/test_mapping.py).
# The grid's own calibration already gives the direction, so a small grid passes
# the identity unless the request says otherwise.
SCAN_LINEAR_MIN_TILES = 25


def assemble_arguments(request: SampleMapRequest, manifest: dict) -> dict:
    """Extra assemble_map arguments: stage_sigma_m and scan_linear, chosen as described above."""
    args: dict[str, Any] = {}
    if request.get("stage_sigma_m"):
        args["stage_sigma_m"] = float(request["stage_sigma_m"])
    choice = request.get("scan_linear")
    if isinstance(choice, (list, tuple)):
        args["scan_linear"] = [[float(v) for v in row] for row in choice]
    elif choice != "measure" and manifest.get("calibration") and len(manifest.get("tiles", [])) < SCAN_LINEAR_MIN_TILES:
        args["scan_linear"] = [[1.0, 0.0], [0.0, 1.0]]
    return args


class SampleMapState(TypedDict, total=False):
    request: SampleMapRequest
    plan: dict
    status: str  # planned, approved, declined, acquired, collected, assembled, scored, reviewing, done, failed
    manifest: dict
    run: str
    map: str
    stitch: dict
    candidates: list
    shortlist: list
    reviewed_ids: Annotated[list, operator.add]
    decisions: Annotated[list, operator.add]
    more: bool
    report: dict
    errors: Annotated[list, operator.add]


def plan_request(request: SampleMapRequest) -> dict:
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


def diverse_shortlist(candidates: list[dict], k: int) -> list[dict]:
    """Best score first, then each next one as far as possible (in stage metres) from those picked."""
    pool = [p for p in candidates if p.get("x_m") is not None and p.get("y_m") is not None]
    if not pool or k <= 0:
        return []
    pool.sort(key=lambda p: -(p.get("score") or 0.0))
    picked = [pool.pop(0)]
    while pool and len(picked) < k:
        def gap(p):
            return min(math.dist((p["x_m"], p["y_m"]), (q["x_m"], q["y_m"])) for q in picked)

        best = max(pool, key=lambda p: (gap(p), p.get("score") or 0.0))
        pool.remove(best)
        picked.append(best)
    return picked


def build_graph(tools, checkpointer=None, prefer: tuple[str, ...] = (), run_dir: str | Path | None = None):
    """Compile the graph. ``tools``: a list or dict of tools, or a ToolBox. A checkpointer is needed to pause.

    ``run_dir``: write map runs here instead of calling ``collect_map_run`` (the
    analysis server must be allowed to read it, with ``--local-root``).
    """
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)
    run_dir = Path(run_dir) if run_dir is not None else None

    async def wait_for(job: dict, wait_s: float) -> dict:
        """Wait for an analysis job and return its results; raise ToolCallError when it fails."""
        if "job_id" not in job:
            raise ToolCallError("analysis job", f"no job_id in {str(job)[:200]}")
        deadline = time.monotonic() + wait_s
        status = job
        while status.get("state") not in JOB_DONE:
            if time.monotonic() > deadline:
                raise ToolCallError("analysis_job_status", f"job {job['job_id']} still {status.get('state')} after {wait_s:.0f} s")
            status = await box.call("analysis_job_status", job_id=job["job_id"], wait_s=min(25.0, max(1.0, deadline - time.monotonic())))
        if status["state"] != "done":
            raise ToolCallError("analysis job", f"{job['job_id']} {status['state']}: {status.get('last_error') or status}")
        return await box.call("analysis_job_results", job_id=job["job_id"])

    def failed(exc: Exception) -> SampleMapState:
        return {"status": "failed", "errors": [str(exc)]}

    def plan(state: SampleMapState) -> SampleMapState:
        return {"plan": plan_request(state.get("request", {})), "status": "planned"}

    def approve_scan(state: SampleMapState) -> SampleMapState:
        p = state["plan"]
        if p["problems"]:
            return {"status": "failed", "errors": p["problems"]}
        answer = interrupt({"kind": "approve_scan", "plan": p, "ask": "Approve this grid scan? It moves the stage."})
        if isinstance(answer, dict) and answer.get("changes"):
            request = {**state.get("request", {}), **answer["changes"]}
            return {"request": request, "plan": plan_request(request), "status": "replanned"}
        approved = answer is True or (isinstance(answer, dict) and answer.get("approved") is True)
        return {"status": "approved" if approved else "declined"}

    async def acquire(state: SampleMapState) -> SampleMapState:
        p = state["plan"]
        try:
            if p.get("fov_m") is not None:
                await box.call("set_fov", fov=p["fov_m"])
            manifest = await box.call("acquire_map_grid", spec=json.dumps(p["spec"]))
        except ToolCallError as exc:
            return failed(exc)
        if isinstance(manifest, str):
            manifest = json.loads(manifest)
        return {"manifest": manifest, "status": "acquired"}

    async def collect(state: SampleMapState) -> SampleMapState:
        manifest, request = state["manifest"], state.get("request", {})
        facts = {"name": request.get("name", "sample_map"), "grid": manifest.get("grid"), "overlap": manifest.get("overlap"),
                 "fov_m": manifest.get("fov_m"), "calibration": manifest.get("calibration"), "workflow": "asyncroscopy.sample_map"}
        try:
            if run_dir is not None:
                path = write_map_run(run_dir / f"map_{uuid.uuid4().hex[:12]}.h5", manifest, plan=facts)
                return {"run": str(path), "status": "collected"}
            pixel_size_m = float(manifest["pixel_size_nm"]) * 1e-9
            matrix = (manifest.get("calibration") or {}).get("px_per_m")
            stage = [t["stage_xy_m"] for t in manifest["tiles"]]
            centres = image_frame_positions(stage, matrix, pixel_size_m) if matrix else stage
            out = await box.call("collect_map_run", image_keys=[t["key"] for t in manifest["tiles"]], plan=facts,
                                 stage_xy_m=centres, fov_m=float(manifest["fov_m"]))
        except (ToolCallError, OSError, ValueError, KeyError) as exc:
            return failed(exc)
        return {"run": out["run"], "status": "collected"}

    async def assemble(state: SampleMapState) -> SampleMapState:
        wait_s = float(state.get("request", {}).get("analysis_wait_s", DEFAULT_ANALYSIS_WAIT_S))
        try:
            extra = assemble_arguments(state.get("request", {}), state.get("manifest") or {})
            results = await wait_for(await box.call("assemble_map", run=state["run"], **extra), wait_s)
        except ToolCallError as exc:
            return failed(exc)
        return {"map": results["result_keys"]["map"], "stitch": results.get("summary", {}), "status": "assembled"}

    async def score(state: SampleMapState) -> SampleMapState:
        request = state.get("request", {})
        wait_s = float(request.get("analysis_wait_s", DEFAULT_ANALYSIS_WAIT_S))
        try:
            await wait_for(await box.call("score_map", map=state["map"], scorer=request.get("scorer", "baseline"),
                                          top_k=int(request.get("top_k", 8)), tag="sample_map"), wait_s)
            found = await box.call("list_pois", map=state["map"], kind="candidate", limit=500)
        except ToolCallError as exc:
            return failed(exc)
        return {"candidates": found.get("pois", []), "status": "scored"}

    def shortlist(state: SampleMapState) -> SampleMapState:
        k = int(state.get("request", {}).get("shortlist_size", 5))
        seen = set(state.get("reviewed_ids", []))
        return {"shortlist": diverse_shortlist([c for c in state.get("candidates", []) if c["id"] not in seen], k), "status": "reviewing"}

    async def review(state: SampleMapState) -> SampleMapState:
        items = state.get("shortlist", [])
        if not items:
            return {"more": False}
        answer = interrupt({
            "kind": "review_candidates",
            "map": state["map"],
            "shortlist": items,
            "ask": "Accept or reject each candidate. get_region on the analysis server shows the pixels around a stage box.",
        })
        answer = answer if isinstance(answer, dict) else {}
        actor = str(answer.get("actor", "human:operator"))
        by_id = {c["id"]: c for c in items}
        applied, errors = [], []
        for decision in answer.get("decisions", []):
            poi = by_id.get(decision.get("id"))
            status = decision.get("status")
            if poi is None or status not in ("accepted", "rejected"):
                errors.append(f"skipped decision {decision}: unknown id or status")
                continue
            try:
                note = await box.call("annotate", dataset=state["map"], shape="pin", coords={"x": poi["x_m"], "y": poi["y_m"]},
                                      frame="stage_m", kind=status, note=decision.get("note", ""), links=[poi["id"]], reported_by=actor)
                applied.append({"id": poi["id"], "status": status, "note": decision.get("note", ""), "actor": actor, "note_id": note.get("id")})
            except ToolCallError as exc:
                errors.append(str(exc))
        return {"decisions": applied, "reviewed_ids": list(by_id), "more": bool(answer.get("more")), "errors": errors}

    async def report(state: SampleMapState) -> SampleMapState:
        info = None
        if state.get("map"):
            try:
                info = await box.call("map_info", map=state["map"])
            except ToolCallError as exc:
                info = {"error": str(exc)}
        stitch = state.get("stitch", {})
        final = state["status"] if state.get("status") in ("failed", "declined") else "done"
        out = {
            "status": final,
            "run": state.get("run"),
            "map": state.get("map"),
            "map_info": info,
            "stitch": {k: stitch.get(k) for k in ("tiles", "matches_used", "matches_found", "median_residual_px", "p95_residual_px", "fallback_tiles", "unplaced_tiles", "scan_linear") if k in stitch},
            "calibration": (state.get("manifest") or {}).get("calibration"),
            "candidates": len(state.get("candidates", [])),
            "decisions": state.get("decisions", []),
            "errors": state.get("errors", []),
        }
        return {"report": out, "status": final}

    routes = {
        "approve_scan": lambda s: {"approved": "acquire", "replanned": "approve_scan", "declined": END}.get(s["status"], "report"),
        "acquire": lambda s: "collect" if s["status"] == "acquired" else "report",
        "collect": lambda s: "assemble" if s["status"] == "collected" else "report",
        "assemble": lambda s: "score" if s["status"] == "assembled" else "report",
        "score": lambda s: "shortlist" if s["status"] == "scored" else "report",
        "review": lambda s: "shortlist" if s.get("more") else "report",
    }
    graph = StateGraph(SampleMapState)
    for node in (plan, approve_scan, acquire, collect, assemble, score, shortlist, review, report):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "approve_scan")
    graph.add_conditional_edges("approve_scan", routes["approve_scan"], ["acquire", "approve_scan", "report", END])
    graph.add_conditional_edges("acquire", routes["acquire"], ["collect", "report"])
    graph.add_conditional_edges("collect", routes["collect"], ["assemble", "report"])
    graph.add_conditional_edges("assemble", routes["assemble"], ["score", "report"])
    graph.add_conditional_edges("score", routes["score"], ["shortlist", "report"])
    graph.add_edge("shortlist", "review")
    graph.add_conditional_edges("review", routes["review"], ["shortlist", "report"])
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


WORKFLOW = {
    "name": "sample_map",
    "what": ("Grid-scan a sample, stitch it on the pyTEMlib analysis server, score regions, and review the "
             "candidates with a person. Pauses for approval before the stage moves."),
    "inputs": {"request": {"name": "str", "rows": "int", "cols": "int", "overlap": "0-0.6", "fov_m": "float",
                           "calibration_fov_m": "float", "scorer": "str", "top_k": "int", "shortlist_size": "int",
                           "stage_sigma_m": "float", "scan_linear": "2 x 2 list or 'measure'"}},
}

REQUIRED_TOOLS = ("acquire_map_grid", "assemble_map", "analysis_job_status", "analysis_job_results", "score_map", "list_pois", "annotate", "map_info")


def missing_tools(tools, prefer: tuple[str, ...] = (), local_runs: bool = False) -> list[str]:
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)
    needed = REQUIRED_TOOLS + (() if local_runs else ("collect_map_run",))
    return [name for name in needed if not box.has(name)]


def describe(result: Optional[dict]) -> dict:
    """The run's report, or what is known so far."""
    result = result or {}
    return result.get("report") or {"status": result.get("status"), "plan": result.get("plan"), "errors": result.get("errors", [])}
