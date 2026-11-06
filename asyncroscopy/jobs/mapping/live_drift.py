"""Live drift tracking during a map: a bounded image shift before each tile, from the drift measured so far.

Off by default. A plan turns it on with ``"drift": {"track": true}`` (item 8 of
docs/designs/concepts_brief.md; docs/Operation/agentic_mapping.md, "Live drift
tracking"). The Mapper's loop calls it between the stage move and the acquisition
of each tile, whatever the instrument; what an image shift is belongs to the
instrument profile (``image_shift_limit_m``, ``get_image_shift``,
``set_image_shift``). A profile that cannot shift the image has no limit, and a
plan asking for tracking on it is refused at reserve, by name.

For each tile:

1. **Correct** (before acquiring): the tracker (``asyncroscopy.routines.drift``;
   ``linear`` by default, or a registered plug-in) predicts the tile's placement
   error at the middle of its frame on the instrument's clock (``clock_s``), and
   the image shift is set to cancel it. Each correction changes the shift by at
   most ``max_shift_m`` per axis and never takes it more than ``max_total_m`` from
   where it was before the map. Every correction is written to the Instrument
   Trace as ``Mapper.drift_correction`` (ok false, and the job fails, if the
   instrument refuses the shift).
2. **Measure** (after acquiring): the tile is registered against an overlapping
   tile already taken (the previous one in serpentine order, else the one in the
   row before), with the stitch's bounded cross-correlation, searched near where
   the prediction put it. The placement error that gives (image shift taken out)
   is the tracker's next observation. A featureless or ambiguous overlap is no
   measurement; the tracker keeps its prediction.

The stage only ever goes to the plan's tile centres, so tracking never takes it
outside the plan box. The image shift is put back to its value before the map
when the tiles are done and whenever the job stops. The tracker's state, the
corrections and their totals are saved with the job (a resumed job carries on),
the per-tile correction and measured error go in the run file
(``drift_correction_m``, ``drift_error_m``), and ``results`` reports a summary
(``summary.drift_tracking``). Revisits add the tracker's estimate to the drift
the stitch fitted, since the stitch only sees what tracking left.

A dry run tracks nothing (there are no images) but rehearses the corrections:
each ``set_image_shift`` is recorded in the trace and checked against
``max_total_m`` (see ``check_shift``).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from asyncroscopy.routines.drift import DEFAULT_WINDOW, bounded_correction, get_tracker, tile_offset_m
from asyncroscopy.routines.grid import index_at

if TYPE_CHECKING:
    from .profile import InstrumentProfile, MapSettings
    from .run_file import RunFile

DEFAULTS = {"track": False, "tracker": "linear", "window": DEFAULT_WINDOW}
MAX_SHIFT_FOV = 0.1  # default max_shift_m: a tenth of a tile per correction
REGISTRATION_MARGIN_PX = 3.0  # added to every registration search bound
KEY = "drift_tracking"


def tracking(plan: dict) -> dict | None:
    """The plan's resolved tracking settings when tracking is on, else None."""
    drift = plan.get("drift")
    return drift if isinstance(drift, dict) and drift.get("track") else None


def resolve(p: dict, profile: InstrumentProfile, tile_size_m: tuple[float, float], tiles: int) -> None:
    """Check and complete ``p["drift"]`` at reserve; refuse, by name, what tracking cannot do here."""
    drift = p.get("drift")
    if drift is None:
        return
    drift = {**DEFAULTS, **drift}
    p["drift"] = drift
    if not drift["track"]:
        return
    limit = profile.image_shift_limit_m
    if limit is None:
        raise ValueError(
            f"live drift tracking (drift.track) corrects with an image shift, which the {profile.name!r} instrument profile "
            f"cannot apply ({profile.doc}). Nothing was run; resend the plan without drift.track (the stitch still fits drift after the map)"
        )
    get_tracker(str(drift["tracker"]))  # unknown tracker: refused, with the registered ones
    if tiles < 2 or float(p["overlap"]) <= 0:
        raise ValueError("live drift tracking measures drift where neighbouring tiles overlap: it needs at least 2 tiles and overlap > 0")
    spec = importlib.util.find_spec("pyTEMlib")
    if spec is None or not any((Path(d) / "image_tools" / "mosaic.py").exists() for d in spec.submodule_search_locations or []):
        raise ValueError("live drift tracking registers tiles with pyTEMlib's image_tools.mosaic, which is not installed here")
    total = float(drift.get("max_total_m", limit))
    if total > limit:
        raise ValueError(f"drift.max_total_m {total:g} m is beyond the {profile.name!r} profile's image shift limit of {limit:g} m")
    step = float(drift.get("max_shift_m", MAX_SHIFT_FOV * min(tile_size_m)))
    if step > total:
        raise ValueError(f"drift.max_shift_m {step:g} m (one correction) is larger than drift.max_total_m {total:g} m (all of them)")
    drift["max_total_m"] = total
    drift["max_shift_m"] = step
    drift["window"] = int(drift["window"])


def needed_commands(plan: dict, profile: InstrumentProfile) -> tuple:
    return tuple(profile.image_shift_commands) if tracking(plan) else ()


# ----------------------------------------------------------------------
# During the job
# ----------------------------------------------------------------------
def _state(job: dict) -> dict:
    return job[KEY]


def _tracker(job: dict):
    settings = tracking(job["resolved_plan"])
    state = _state(job)
    return get_tracker(str(settings["tracker"]))(settings, state.get("tracker_state"))


def start(job: dict, profile: InstrumentProfile) -> None:
    """At the start (and on a resume): remember the image shift before the map, once."""
    if not tracking(job["resolved_plan"]):
        return
    state = job.setdefault(KEY, {
        "corrections": 0, "measurements": 0, "missed": 0, "limited": 0, "max_step_m": 0.0, "max_correction_m": 0.0,
        "shift_m": [0.0, 0.0], "tiles": {}, "last_tile": None,
    })
    if "base_shift_m" not in state:
        state["base_shift_m"] = profile.get_image_shift(job)


def correct(job: dict, profile: InstrumentProfile, tile: dict, trace: Callable[..., None]) -> dict | None:
    """Before acquiring ``tile``: set the bounded image shift that cancels the predicted placement error.

    ``trace(job, ok, arguments, clock_s, error=None)`` records the correction (the Mapper's Mapper.drift_correction)."""
    settings = tracking(job["resolved_plan"])
    if not settings:
        return None
    plan = job["resolved_plan"]
    state = _state(job)
    tracker = _tracker(job)
    now = profile.clock_s()
    t_mid = now + profile.tile_seconds(plan) / 2.0  # where the frame's middle will be on the instrument's clock
    predicted = tracker.predict(t_mid) if tracker.observations else (0.0, 0.0)
    previous = list(state["shift_m"])
    shift, limited = bounded_correction([-predicted[0], -predicted[1]], previous, [0.0, 0.0], float(settings["max_shift_m"]), float(settings["max_total_m"]))
    base = state["base_shift_m"]
    absolute = [base[0] + shift[0], base[1] + shift[1]]
    step = [shift[0] - previous[0], shift[1] - previous[1]]
    arguments = {
        "tile": tile["index"], "image_shift_m": absolute, "correction_m": shift, "step_m": step,
        "predicted_error_m": list(predicted), "limited": limited, "at_clock_s": t_mid,
        "max_shift_m": settings["max_shift_m"], "max_total_m": settings["max_total_m"], "tracker": settings["tracker"],
    }
    try:
        profile.set_image_shift(job, absolute)
    except Exception as exc:
        trace(job, False, arguments, now, error=f"{type(exc).__name__}: {exc}")
        raise RuntimeError(f"the image shift for live drift tracking was refused ({exc}); the job stops with the shift put back") from exc
    trace(job, True, arguments, now)
    state["shift_m"] = shift
    state["restored"] = False
    state["corrections"] += 1
    state["limited"] += int(limited)
    state["max_step_m"] = max(state["max_step_m"], max(abs(v) for v in step))
    state["max_correction_m"] = max(state["max_correction_m"], max(abs(v) for v in shift))
    state["last_predicted_error_m"] = list(predicted)
    return {"correction_m": shift, "predicted_error_m": list(predicted), "limited": limited}


def _image(images: dict) -> np.ndarray | None:
    """The image the run file keeps as ``tiles`` (the first by name): the one tiles are registered on."""
    for _name, (array, _attrs) in sorted(images.items()):
        array = np.asarray(array)
        return array if array.ndim == 2 else None
    return None


def measure(job: dict, tile: dict, images: dict | None, acquired: float, run: RunFile | None, bounds: MapSettings) -> dict | None:
    """After acquiring ``tile``: register it on an overlapping neighbour (read back from the run file) and give
    the tracker the placement error. ``bounds`` gives the stage and drift errors that bound the search.
    ``images`` and ``run`` are None in a dry run, which measures nothing."""
    settings = tracking(job["resolved_plan"])
    if not settings:
        return None
    plan = job["resolved_plan"]
    state = _state(job)
    tracker = _tracker(job)
    image = _image(images) if images else None
    index, row, col = tile["index"], tile["row"], tile["col"]
    planned = np.array([tile["x"], tile["y"]], dtype=np.float64)
    shift = np.asarray(state["shift_m"], dtype=np.float64)
    tiles = state["tiles"]
    measured = None
    confidence = None
    if image is None:
        pass  # a dry run: nothing to register
    elif not tiles:
        measured = np.zeros(2)  # the map's first tile is the reference
    else:
        last = state.get("last_tile")
        neighbour = None
        if last is not None and abs(last["row"] - row) + abs(last["col"] - col) == 1:
            neighbour = last["index"]
        elif row > 0:
            neighbour = index_at(plan["grid"], plan["order"], row - 1, col)
        record = tiles.get(str(neighbour)) if neighbour is not None else None
        previous = run.tile_image(neighbour) if record is not None and run is not None else None
        if previous is not None and previous.shape == image.shape:
            width = float((plan.get("tile_m") or [plan["fov_m"]])[0])
            pixel = width / image.shape[1]
            p_planned, p_shift, p_error = (np.asarray(record[k], dtype=np.float64) for k in ("p", "s", "e"))
            predicted = np.asarray(tracker.predict(acquired) if tracker.observations else (0.0, 0.0))
            expected = (planned + shift + predicted) - (p_planned + p_shift + p_error)
            bound = 4.0 * math.sqrt(2.0) * float(bounds.stage_sigma_m) + float(bounds.max_drift_m_per_s) * abs(acquired - record["t"]) + REGISTRATION_MARGIN_PX * pixel
            found = tile_offset_m(previous, image, pixel, expected, bound)
            if found is not None:
                measured = np.asarray(found[:2]) - (planned - p_planned) - (shift - p_shift) + p_error
                confidence = found[2]
    if measured is not None:
        tracker.observe(acquired, measured)
        state["measurements"] += 1
    elif image is not None:
        state["missed"] += 1
    state["tracker_state"] = tracker.state()
    rate = tracker.rate_m_per_s()
    if rate is not None:
        state["rate_m_per_s"] = list(rate)
    # Kept for the neighbours still to come: a row of tiles plus one is enough in either order.
    chained = measured if measured is not None else np.asarray(tracker.predict(acquired) if tracker.observations else (0.0, 0.0))
    tiles[str(index)] = {"t": float(acquired), "p": planned.tolist(), "s": shift.tolist(), "e": [float(v) for v in chained], "measured": measured is not None}
    for key in list(tiles)[: max(0, len(tiles) - (int(plan["grid"]["cols"]) + 2))]:
        del tiles[key]
    state["last_tile"] = {"index": index, "row": row, "col": col}
    out = {"measured_error_m": None if measured is None else [float(v) for v in measured]}
    if confidence is not None:
        out["confidence"] = round(float(confidence), 3)
    return out


def tile_arrays(job: dict, record: dict | None) -> dict:
    """Per-tile arrays for the run file: the correction applied and the error measured (NaN: none)."""
    if not tracking(job["resolved_plan"]) or record is None:
        return {}
    error = record.get("measured_error_m")
    return {"drift_correction_m": list(record["correction_m"]), "drift_error_m": list(error) if error is not None else [math.nan, math.nan]}


def finish(job: dict, profile: InstrumentProfile) -> None:
    """Put the image shift back to its value before the map (once; again after another correction)."""
    if not tracking(job.get("resolved_plan") or {}) or KEY not in job:
        return
    state = _state(job)
    if state.get("restored") or "base_shift_m" not in state:
        return
    profile.set_image_shift(job, list(state["base_shift_m"]))
    state["restored"] = True
    state["shift_m"] = [0.0, 0.0]


def estimate_at(job: dict, t_s: float) -> tuple[float, float]:
    """The tracker's placement error at ``t_s`` (for revisits); (0, 0) without tracking or measurements."""
    if not tracking(job.get("resolved_plan") or {}) or KEY not in job:
        return 0.0, 0.0
    tracker = _tracker(job)
    return tracker.predict(t_s) if tracker.observations else (0.0, 0.0)


def summary(job: dict) -> dict | None:
    if not tracking(job.get("resolved_plan") or {}) or KEY not in job:
        return None
    state = _state(job)
    settings = tracking(job["resolved_plan"])
    out = {k: state.get(k) for k in ("corrections", "measurements", "missed", "limited", "max_step_m", "max_correction_m", "rate_m_per_s", "last_predicted_error_m")}
    out.update({"tracker": settings["tracker"], "max_shift_m": settings["max_shift_m"], "max_total_m": settings["max_total_m"], "restored": bool(state.get("restored"))})
    return {k: v for k, v in out.items() if v is not None}


def check_shift(job: dict, shift: list[float]) -> dict | None:
    """A dry run's check of a recorded image shift [x, y]: within max_total_m of the shift before the map?"""
    settings = tracking(job["resolved_plan"])
    if not settings or KEY not in job:
        return None
    value = [float(v) for v in list(shift)[:2]]
    base = _state(job)["base_shift_m"]
    inside = all(abs(v - b) <= float(settings["max_total_m"]) + 1e-15 for v, b in zip(value, base))
    return {"ok": inside, "image_shift_m": value, "base_shift_m": list(base), "max_total_m": settings["max_total_m"]}
