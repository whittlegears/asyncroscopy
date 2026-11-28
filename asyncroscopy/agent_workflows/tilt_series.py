"""Alpha-tilt diffraction series with image-shift tracking (notebook 16) as a LangGraph graph.

Graph::

    plan -> read_start -> approve -> reference -> step -> ... -> step -> restore -> report
              | (bad request,  | (declined)    | (fails)    | (fails)              ^
              |  alpha != 0)   v               v            v                      |
              +-------------> report         restore ---> restore ---------------->+

- ``plan`` checks the request and lists the angles: pre-tilt steps from
  ``-alpha_step`` down to ``-max_alpha_deg`` (tracking only), then the series
  from ``-max_alpha_deg`` to ``+max_alpha_deg`` (tracking and one camera
  diffraction pattern per angle).
- ``read_start`` reads the stage, image shift and field of view. The stage must
  start at alpha 0, as in the notebook; otherwise the run stops before asking.
- ``approve`` pauses (``interrupt``) with the plan, the starting position and a
  time estimate. Resume with ``{"approved": true}`` or ``{"approved": false}``.
- ``reference`` sets the field of view (if ``fov_m`` is given) and takes the
  tracking reference image at alpha 0.
- ``step`` does one angle per graph step, so a checkpointer can resume a long
  series after an interruption: move alpha, take a tracking image, measure its
  shift against the reference (``analysis.drift.correlation_shift``, so no
  quantEM is needed), correct the image shift, and in the series phase acquire
  a camera image.
- ``restore`` puts back the starting stage, image shift and field of view. Every
  path that moved or changed anything goes through it, failures included.
- ``report`` lists one camera key per series angle and the tracking record.

Images are read inside the node from the returned data keys (``read_image``;
by default a local HDF5 file, which is what the digital twins return without a
DATA device). They never pass through the model.

Over MCP only Tango commands are reachable, so the camera and scan settings
(``camera.imsize``, ``exposure_time``, ``scan.imsize``, ``dwell_time``) are
attributes the operator sets before the run.

Tools (any instrument prefix): ``get_stage``, ``move_stage``,
``get_image_shift``, ``set_image_shift``, ``get_fov``, ``acquire_scanned_image``,
``acquire_camera_image``, and ``set_fov`` when ``fov_m`` is given.
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, Callable, Optional, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from asyncroscopy.analysis.drift import correlation_shift

from .tools import ToolBox, ToolCallError

DEFAULT_ALPHA_STEP_DEG = 6.0
DEFAULT_MAX_ALPHA_DEG = 30.0
MAX_ALPHA_LIMIT_DEG = 70.0
DEFAULT_SECONDS_PER_TRACK = 10.0  # move, settle, one tracking scan: replace with a measured value
DEFAULT_SECONDS_PER_PATTERN = 5.0  # one camera exposure and save
DEFAULT_MAX_SHIFT_FRACTION = 0.25  # the notebook's max_shift = imsize / 4
DEFAULT_ALPHA_TOLERANCE_DEG = 0.1
ALPHA = 3  # stage position is [x, y, z, alpha, beta]
# Each angle is one graph step. WorkflowRunner's recursion limit is 100, and a run
# that hits the limit stops without restoring, so the plan refuses longer series.
MAX_STEPS = 90

WORKFLOW = {
    "name": "tilt_series",
    "what": "Tilt the stage in alpha from -max to +max, tracking a feature with image shift, and acquire one camera "
    "diffraction pattern per angle. The stage must start at alpha 0. Pauses for approval before the stage moves "
    "and always restores the stage and image shift.",
    "inputs": {"request": {"alpha_step_deg": "float (default 6)", "max_alpha_deg": "float (default 30, at most 70)",
                           "fov_m": "float (optional; tracking field of view)", "detectors": "list (default ['haadf'])",
                           "tracking_sign": "[sx, sy] (default [1, 1])", "max_shift_fraction": "float (default 0.25)",
                           "seconds_per_track": "float", "seconds_per_pattern": "float"}},
}


class TiltSeriesRequest(TypedDict, total=False):
    alpha_step_deg: float
    max_alpha_deg: float
    fov_m: float
    detectors: list
    tracking_sign: list  # +1 when a positive image shift moves the content to larger columns (x) / rows (y)
    max_shift_fraction: float  # a tracking shift larger than this fraction of the image stops the run
    alpha_tolerance_deg: float
    seconds_per_track: float
    seconds_per_pattern: float


class TiltSeriesState(TypedDict, total=False):
    request: TiltSeriesRequest
    plan: dict
    status: str  # planned, ready, approved, declined, tracking, acquired, restored, done, failed
    start: dict  # stage, image_shift, fov_m as read before anything moved
    moved: bool  # something on the instrument changed, so restore must run
    reference_key: str
    pixel_size_m: float
    index: int
    steps: Annotated[list, operator.add]
    restored: bool
    report: dict
    errors: Annotated[list, operator.add]


def plan_request(request: TiltSeriesRequest) -> dict:
    step = float(request.get("alpha_step_deg", DEFAULT_ALPHA_STEP_DEG))
    top = float(request.get("max_alpha_deg", DEFAULT_MAX_ALPHA_DEG))
    problems = []
    if not step > 0:
        problems.append("alpha_step_deg must be > 0")
    if not 0 < top <= MAX_ALPHA_LIMIT_DEG:
        problems.append(f"max_alpha_deg must be between 0 and {MAX_ALPHA_LIMIT_DEG}")
    elif step > 0 and step > top:
        problems.append("alpha_step_deg must not exceed max_alpha_deg")
    fov = request.get("fov_m")
    if fov is not None and not float(fov) > 0:
        problems.append("fov_m must be > 0")
    sign = list(request.get("tracking_sign", [1.0, 1.0]))
    if len(sign) != 2:
        problems.append("tracking_sign must be [sx, sy]")
    if problems:
        return {"problems": problems, "steps": []}
    pretilt = np.arange(-step, -top - step / 2, -step)
    series = np.arange(-top, top + step / 2, step)
    steps = [{"phase": "pretilt", "alpha_deg": round(float(a), 6)} for a in pretilt]
    steps += [{"phase": "series", "alpha_deg": round(float(a), 6)} for a in series]
    if len(steps) > MAX_STEPS:
        return {"problems": [f"{len(steps)} tilt steps is more than {MAX_STEPS}; use a larger alpha_step_deg"], "steps": []}
    track_s = float(request.get("seconds_per_track", DEFAULT_SECONDS_PER_TRACK))
    pattern_s = float(request.get("seconds_per_pattern", DEFAULT_SECONDS_PER_PATTERN))
    seconds = track_s * (len(steps) + 1) + pattern_s * len(series)  # + 1 reference image
    return {
        "alpha_step_deg": step,
        "max_alpha_deg": top,
        "pretilt_deg": [s["alpha_deg"] for s in steps if s["phase"] == "pretilt"],
        "series_deg": [s["alpha_deg"] for s in steps if s["phase"] == "series"],
        "steps": steps,
        "patterns": len(series),
        "fov_m": None if fov is None else float(fov),
        "detectors": list(request.get("detectors", ["haadf"])),
        "tracking_sign": [float(v) for v in sign],
        "max_shift_fraction": float(request.get("max_shift_fraction", DEFAULT_MAX_SHIFT_FRACTION)),
        "estimated_minutes": round(seconds / 60.0, 1),
        "problems": [],
    }


def read_local_image(key: str) -> np.ndarray:
    """The first 2D image (image/HAADF first) in a local acquisition file."""
    from asyncroscopy.mapping.run_file import _first_2d

    if not os.path.isfile(str(key)):
        raise ValueError(f"{key!r} is not a local file; pass read_image= to build_graph to read DATA/Tiled keys")
    return np.asarray(_first_2d(str(key)), dtype=float)


def _floats(value: Any) -> list[float]:
    return [float(v) for v in np.asarray(value, dtype=float).ravel()]


def build_graph(tools, checkpointer=None, prefer: tuple[str, ...] = (), read_image: Optional[Callable[[str], Any]] = None):
    """Compile the graph. ``tools``: a list or dict of tools, or a ToolBox. A checkpointer is needed to pause.

    ``read_image(key) -> 2D array`` reads a tracking image; the default reads local HDF5 files.
    """
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)
    read = read_image or read_local_image

    def failed(exc: Exception) -> TiltSeriesState:
        return {"status": "failed", "errors": [str(exc)]}

    async def scan_image(detectors: list) -> str:
        return str(await box.call("acquire_scanned_image", detector_list=list(detectors)))

    def plan(state: TiltSeriesState) -> TiltSeriesState:
        p = plan_request(state.get("request", {}))
        if p["problems"]:
            return {"plan": p, "status": "failed", "errors": p["problems"]}
        return {"plan": p, "status": "planned"}

    async def read_start(state: TiltSeriesState) -> TiltSeriesState:
        tolerance = float(state.get("request", {}).get("alpha_tolerance_deg", DEFAULT_ALPHA_TOLERANCE_DEG))
        try:
            stage = _floats(await box.call("get_stage"))
            shift = _floats(await box.call("get_image_shift"))
            fov = float(await box.call("get_fov"))
        except (ToolCallError, TypeError, ValueError) as exc:
            return failed(exc)
        start = {"stage": stage, "image_shift": shift, "fov_m": fov}
        if len(stage) != 5:
            return {"start": start, "status": "failed", "errors": [f"expected a 5-axis stage [x, y, z, alpha, beta], got {stage}"]}
        if abs(stage[ALPHA]) > tolerance:
            return {"start": start, "status": "failed",
                    "errors": [f"Set the stage alpha to 0 degrees before starting (it is {stage[ALPHA]:.3f})."]}
        return {"start": start, "status": "ready"}

    def approve(state: TiltSeriesState) -> TiltSeriesState:
        p = state["plan"]
        answer = interrupt({
            "kind": "approve_tilt_series",
            "plan": {k: v for k, v in p.items() if k != "steps"},
            "start": state["start"],
            "ask": f"Approve this alpha-tilt series? It tilts the stage from -{p['max_alpha_deg']:g} to "
            f"+{p['max_alpha_deg']:g} deg and changes the image shift; both are restored at the end.",
        })
        approved = answer is True or (isinstance(answer, dict) and answer.get("approved") is True)
        return {"status": "approved" if approved else "declined"}

    async def reference(state: TiltSeriesState) -> TiltSeriesState:
        p, start = state["plan"], state["start"]
        try:
            stage = _floats(await box.call("get_stage"))
            if abs(stage[ALPHA] - start["stage"][ALPHA]) > DEFAULT_ALPHA_TOLERANCE_DEG:
                return {"status": "failed", "errors": [f"the stage moved while waiting for approval (alpha {stage[ALPHA]:.3f})"]}
            fov = start["fov_m"]
            if p.get("fov_m") is not None:
                await box.call("set_fov", fov=p["fov_m"])
                fov = p["fov_m"]
            key = await scan_image(p["detectors"])
            image = read(key)
        except (ToolCallError, OSError, ValueError, KeyError) as exc:
            return {**failed(exc), "moved": p.get("fov_m") is not None}
        return {"reference_key": key, "pixel_size_m": fov / image.shape[1], "index": 0, "status": "tracking",
                "moved": p.get("fov_m") is not None}

    async def step(state: TiltSeriesState) -> TiltSeriesState:
        p, start = state["plan"], state["start"]
        index = int(state.get("index", 0))
        planned = p["steps"][index]
        position = list(start["stage"])
        position[ALPHA] = planned["alpha_deg"]
        record: dict[str, Any] = dict(planned)
        try:
            await box.call("move_stage", position=position)
            reference_image = read(state["reference_key"])
            record["tracking_key"] = key = await scan_image(p["detectors"])
            image = read(key)
            row, col, confidence = correlation_shift(reference_image, image)
            record.update(shift_px=[row, col], confidence=confidence)
            limit = p["max_shift_fraction"] * min(image.shape)
            if max(abs(row), abs(col)) > limit:
                raise ValueError(f"tracking shift ({row:.1f}, {col:.1f}) px at alpha {planned['alpha_deg']:g} is larger "
                                 f"than {limit:.0f} px; the feature was lost")
            # correlation_shift is + when the content moved to larger indices; shift it back.
            sx, sy = p["tracking_sign"]
            correction = [-sx * col * state["pixel_size_m"], -sy * row * state["pixel_size_m"]]
            current = _floats(await box.call("get_image_shift"))
            shift = [current[0] + correction[0], current[1] + correction[1]]
            await box.call("set_image_shift", shift=shift)
            record.update(correction_m=correction, image_shift_m=shift)
            if planned["phase"] == "series":
                record["camera_key"] = str(await box.call("acquire_camera_image"))
        except (ToolCallError, OSError, ValueError, KeyError) as exc:
            return {**failed(exc), "moved": True, "steps": [{**record, "error": str(exc)}]}
        done = index + 1 >= len(p["steps"])
        return {"index": index + 1, "steps": [record], "moved": True, "status": "acquired" if done else "tracking"}

    async def restore(state: TiltSeriesState) -> TiltSeriesState:
        start, errors = state["start"], []
        for command, value in (("move_stage", {"position": start["stage"]}), ("set_image_shift", {"shift": start["image_shift"]})):
            try:
                await box.call(command, **value)
            except ToolCallError as exc:
                errors.append(f"restore failed: {exc}")
        if state["plan"].get("fov_m") is not None:
            try:
                await box.call("set_fov", fov=start["fov_m"])
            except ToolCallError as exc:
                errors.append(f"restore failed: {exc}")
        status = "failed" if errors or state.get("status") == "failed" else "restored"
        return {"restored": not errors, "status": status, "errors": errors}

    def report(state: TiltSeriesState) -> TiltSeriesState:
        final = state["status"] if state.get("status") in ("failed", "declined") else "done"
        steps = state.get("steps", [])
        series = [s for s in steps if s.get("phase") == "series" and s.get("camera_key")]
        out = {
            "status": final,
            "alpha_deg": [s["alpha_deg"] for s in series],
            "camera_keys": [s["camera_key"] for s in series],
            "reference_key": state.get("reference_key"),
            "pixel_size_m": state.get("pixel_size_m"),
            "tracking": [{k: s.get(k) for k in ("phase", "alpha_deg", "tracking_key", "shift_px", "confidence", "correction_m", "error")}
                         for s in steps],
            "start": state.get("start"),
            "restored": state.get("restored") if state.get("moved") else None,
            "errors": state.get("errors", []),
        }
        return {"report": out, "status": final}

    def after_step(s: TiltSeriesState) -> str:
        return "step" if s["status"] == "tracking" else "restore"

    graph = StateGraph(TiltSeriesState)
    for node in (plan, read_start, approve, reference, step, restore, report):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "read_start" if s["status"] == "planned" else "report", ["read_start", "report"])
    graph.add_conditional_edges("read_start", lambda s: "approve" if s["status"] == "ready" else "report", ["approve", "report"])
    graph.add_conditional_edges("approve", lambda s: "reference" if s["status"] == "approved" else "report", ["reference", "report"])
    graph.add_conditional_edges("reference", lambda s: "step" if s["status"] == "tracking" else ("restore" if s.get("moved") else "report"),
                                ["step", "restore", "report"])
    graph.add_conditional_edges("step", after_step, ["step", "restore"])
    graph.add_edge("restore", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


REQUIRED_TOOLS = ("get_stage", "move_stage", "get_image_shift", "set_image_shift", "get_fov", "acquire_scanned_image", "acquire_camera_image")


def missing_tools(tools, prefer: tuple[str, ...] = (), set_fov: bool = False) -> list[str]:
    box = tools if isinstance(tools, ToolBox) else ToolBox(tools, prefer=prefer)
    needed = REQUIRED_TOOLS + (("set_fov",) if set_fov else ())
    return [name for name in needed if not box.has(name)]


def describe(result: Optional[dict]) -> dict:
    """The run's report, or what is known so far."""
    result = result or {}
    return result.get("report") or {"status": result.get("status"), "plan": result.get("plan"), "errors": result.get("errors", [])}
