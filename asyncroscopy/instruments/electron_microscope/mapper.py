"""Mapper: acquire a grid of tiles over a stage box, then stitch and explore it, as one background job.

An agent (or a person, or a script) asks for a map in one call instead of
issuing a thousand moves and acquisitions. The plan (asyncroscopy/contracts/
map_plan.schema.json) fixes a stage box, field of view, overlap and, for the
instrument, its own fields (a STEM's detectors and per-tile EDS spectra, an
AFM's setpoint). The worker visits tiles in serpentine order (to limit
backlash) and never leaves the box.

The Mapper is instrument-agnostic: what differs between a STEM, an SEM and an
AFM is an instrument profile (asyncroscopy/jobs/map_profile.py), chosen with the
``instrument_profile`` device property: ``stem`` (the default), ``sem``,
``yaml`` (any YAML twin with an image acquisition) or one registered by a
plug-in. The profile acquires each tile, makes the instrument safe between
tiles and whenever the job stops (blanks the beam, retracts the tip), prepares
before a move and recovers after it, and says what a plan costs. `reserve`
refuses a plan over any budget it gives (``budgets``: electron dose for a beam,
time for any instrument, tip force for an AFM) before anything moves, and, on a
stack that is not simulated, a dose budget computed from an assumed beam
current; the plan cannot change after that, so there is no separate budget
check while it runs. The instrument is made safe whenever the job ends: done,
aborted or failed.

The job is a pipeline whose phases are steps of the same job, so abort,
approval and progress cover all of them:

1. tiles: one step per tile. Steps are computed from the plan's grid (origin,
   step and count per axis) when needed, so a plan of millions of tiles
   stores a few numbers.
2. stitch (``stitch``, default on, needs ``analysis_url``): the finished run is
   stitched on the analysis server (pyTEMlib ``assemble_map``) with search
   bounds from this Mapper's ``stage_sigma_m`` and ``max_drift_m_per_s``;
   every EDS point spectrum becomes a point of interest linked to its key.
3. score (``find_regions``: N > 0): the best N regions become points of interest.
4. revisit (``revisit``: {"fov_m": ...}): the stage goes to each of those
   regions, corrected for the drift the stitch fitted, and only inside the
   plan box, and takes a higher-magnification image, added as a point of
   interest linked to the image and to its region.

Analysis calls go through one session to the analysis server. Each phase
waits for its analysis job with backoff and an overall deadline that grows
with the map, checking for an abort between waits. The analysis job's id is
saved in the job before waiting, so a resumed job waits for that same job
(or, if another client already started one on the same run, adopts it)
instead of starting a second. If the analysis server is down, restarted or
too slow, the job still ends done with every tile safe; ``summary.stitch_error``
says why and how to retry (assemble_map on the run). Scoring and revisits are
idempotent: a resumed job never images a region twice or adds its points twice.

Every tile, its reported stage position and any spectra go into one HDF5 run
file, ``maps/map_<run>.h5`` (``.h5.partial`` while tiles are being taken, so
DATA does not register it half-written). The run file uses layout 2 (see
pyTEMlib.mcp_server.map_store): ``tiles`` is one chunked array [n, rows,
cols] with per-tile arrays ``written``, ``stage_reported_m``,
``commanded_xy_m``, ``acquired_s`` and ``row_col``; ``spectra`` [n, channels]
and ``spectrum_keys`` [n] hold EDS point spectra. The file is closed and
synced after each tile, so a crash loses at most the tile being written.
Nothing in it depends on the vendor or on whether the microscope was the
digital twin. ``acquired_s`` is the microscope's ``clock_s`` (simulated time on
a twin), the clock the drift fit and revisits use.

A dry run (plan ``"dry_run": true``) moves nothing: stage moves and acquisitions
are recorded in the trace and checked against the plan box (see JobDevice), the
beam is still blanked, no image or run file is made, and the summary says so.

Live drift tracking (plan ``"drift": {"track": true}``, off by default) corrects each tile with a
bounded image shift between the stage move and the acquisition, from drift measured by registering
overlapping tiles on the instrument's clock (asyncroscopy/jobs/live_drift.py). The profile says how to
shift the image; a profile that cannot is refused at reserve. Every correction is traced.

The grid, dose and drift arithmetic are pure functions in asyncroscopy/routines
(grid.py, dose.py, drift.py); they are imported here under their old names.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import tango
from tango.server import device_property

from asyncroscopy.contracts import ENVELOPE_VERSION, load, validate
from asyncroscopy.data.data_reader import read_acquisition
from asyncroscopy.jobs import JobDevice, live_drift
from asyncroscopy.jobs.job_device import done_count
from asyncroscopy.jobs.map_profile import GENERIC_PLAN_DEFAULTS, GENERIC_PLAN_FIELDS, TIME_BUDGET, Cost, InstrumentProfile, get_profile
from asyncroscopy.routines.beam import DEFAULT_MAX_BEAM_CURRENT_AGE_S, beam_current
from asyncroscopy.routines.dose import ELECTRON_CHARGE, eds_dose, frame_dose  # noqa: F401  (their old home; tests and plug-ins import them here)
from asyncroscopy.routines.drift import drift_at
from asyncroscopy.routines.grid import EDGE_TOLERANCE_M, compact_grid, grid_axes, inside_box, max_exposures, tile_at, tile_centres  # noqa: F401

# A STEM plan's defaults (the generic ones plus the stem profile's); each profile has its own.
PLAN_DEFAULTS = {
    "overlap": 0.15,
    "imsize": 512,
    "dwell_s": 1e-6,
    "detectors": ["haadf"],
    "per_tile": [],
    "eds_exposure_s": 1.0,
    "eds_map_points": 8,
    "order": "serpentine",
    "stitch": True,
    "find_regions": 0,
}
LEGACY_BUDGETS = {"dose_budget_e_per_a2": "dose_e_per_a2"}  # plan fields that are one budget under another name
RENAME_RETRY_DELAYS_S = (0.0, 0.2, 0.5, 1.0, 2.0)  # a finished run file may briefly be open elsewhere (Windows)
RUN_LAYOUT = 2
STAGE_POLL_S = 0.2  # how often a long stage move is checked for arrival (and the job for an abort)
ANALYSIS_CALL_S = 60.0  # longest one call to the analysis server may take
ANALYSIS_WAIT_S = 5.0  # longest one wait on an analysis job; abort is checked between waits
ANALYSIS_BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 20.0)  # after a failed call to the analysis server
STITCH_DEADLINE_S = (600.0, 0.05)  # base, plus per tile: the longest a stitch may take before it counts as failed
SCORE_DEADLINE_S = 900.0
TRACE_LINES_KEPT = 2000  # job events copied into the run file; the rest are in the shared trace (see trace_file)
REVISIT_FOV_MIN_FRACTION = 1 / 20  # a revisit image may zoom in at most 20x on a tile
REVISIT_EDGE_SHIFT = 0.8  # a revisit image may sit this fraction of its half-width off a region near the box edge
# What a dry run (plan "dry_run": true, see JobDevice) reports in place of every image and spectrum.
DRY_RUN_SKIPPED = "skipped (dry run): nothing was acquired and the sample was not exposed; the commands are in the trace with dry_run: true"


FIELD_SYNONYMS = {
    "max_dose_e_per_a2": "dose_budget_e_per_a2", "max_dose": "dose_budget_e_per_a2", "dose_limit": "dose_budget_e_per_a2",
    "dose_budget": "dose_budget_e_per_a2", "dose": "dose_budget_e_per_a2", "fov": "fov_m", "field_of_view_m": "fov_m",
    "size": "size_m", "box": "box_m", "centre": "centre_m", "center_m": "centre_m", "center": "centre_m",
    "dwell": "dwell_s", "dwell_time": "dwell_s", "image_size": "imsize", "pixels": "imsize", "regions": "find_regions",
}


REVISIT_ONLY_KEYS = ("map", "top_k", "points")  # fields of a revisit plan ({"revisit": {...}}) that do not exist at the top level


def check_plan_fields(plan: dict) -> None:
    """Refuse unknown plan fields by name, with the closest allowed field and the full list, before anything else."""
    allowed = sorted(load("map_plan")["properties"])
    unknown = [k for k in plan if k not in allowed]
    if not unknown:
        return
    revisit_keys = [k for k in unknown if k in REVISIT_ONLY_KEYS]
    if revisit_keys:
        # Fields of a revisit plan given at the top level: show the plan they belong in, with the caller's own values.
        nested = {**(plan.get("revisit") if isinstance(plan.get("revisit"), dict) else {}), **{k: plan[k] for k in revisit_keys}}
        nested.setdefault("fov_m", plan.get("fov_m", "<a few times smaller than the map's tiles>"))
        raise ValueError(f"{', '.join(map(repr, revisit_keys))} belong inside revisit. Nothing was run; send the plan as {json.dumps({'revisit': nested}, default=str)}")
    hints = []
    for key in unknown:
        guess = FIELD_SYNONYMS.get(key) or next(iter(difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)), None)
        hints.append(f"{key!r} (did you mean {guess!r}?)" if guess else repr(key))
    raise ValueError(f"unknown plan field(s): {', '.join(hints)}. Nothing was dropped or changed: resend the plan with the right names. Allowed fields: {', '.join(allowed)}")


def profile_for(mapper) -> InstrumentProfile:
    """The mapper's instrument profile (its ``instrument_profile`` property; ``stem`` when it has none), made once."""
    name = getattr(mapper, "instrument_profile", None) or "stem"
    profile = getattr(mapper, "_map_profile", None)
    if profile is None or profile.name != name:
        profile = get_profile(name)(mapper)
        mapper._map_profile = profile
    return profile


def check_profile_fields(plan: dict, profile: InstrumentProfile) -> None:
    """Refuse plan fields this instrument's profile does not use (a tip has no dwell time), by name, before anything else."""
    allowed = GENERIC_PLAN_FIELDS | profile.plan_fields
    unused = [k for k in plan if k not in allowed]
    if unused:
        raise ValueError(
            f"{', '.join(map(repr, unused))} not used when mapping with the {profile.name!r} instrument profile ({profile.doc}). "
            f"Nothing was run; resend the plan without them. Fields it takes: {', '.join(sorted(allowed))}"
        )


def plan_budgets(p: dict) -> dict:
    """The plan's budgets by name: ``budgets`` plus fields that are a budget under an older name (dose_budget_e_per_a2)."""
    budgets = dict(p.get("budgets") or {})
    for field, name in LEGACY_BUDGETS.items():
        if p.get(field) is not None:
            budgets.setdefault(name, p[field])
    return budgets


def check_budgets(budgets: dict, costs: dict[str, Cost], duration_s: float, profile: InstrumentProfile) -> dict | None:
    """Refuse a plan over any budget it gives (every profile can budget time_s); what each budgeted cost comes to."""
    if not budgets:
        return None
    costs = {**costs, TIME_BUDGET: Cost(duration_s, "s", why="the estimated duration of the whole job", advice="map a smaller area, find fewer regions, or shorten each tile")}
    unknown = sorted(set(budgets) - set(costs))
    if unknown:
        raise ValueError(f"unknown budget(s) {unknown} for the {profile.name!r} instrument profile; it can budget {sorted(costs)}")
    for name, limit in budgets.items():
        if costs[name].value > float(limit):
            raise ValueError(costs[name].refusal(limit))
    return {name: {"value": round(costs[name].value, 3), "budget": limit, "units": costs[name].units} for name, limit in budgets.items()}


def write_tile(path: Path, n: int, tile: dict, plan: dict, reported, acquired, image_key, images, spectrum, spectrum_key, arrays: dict | None = None) -> None:
    """Put one tile into a run file's per-tile arrays (created at the first tile, sized for the whole grid).

    ``arrays``: more per-tile values by dataset name (live drift tracking's drift_correction_m and
    drift_error_m), each a float or a list of floats; created NaN-filled at the first tile that has them.

    The file is closed and synced to disk before returning, so a crash later loses nothing written here.
    """
    k = tile["index"]
    with h5py.File(path, "a") as h5:
        if "written" not in h5:
            h5.create_dataset("written", shape=(n,), dtype=bool)
            h5.create_dataset("stage_reported_m", shape=(n, 5), dtype=np.float64, fillvalue=np.nan)
            h5.create_dataset("commanded_xy_m", shape=(n, 2), dtype=np.float64, fillvalue=np.nan)
            h5.create_dataset("acquired_s", shape=(n,), dtype=np.float64, fillvalue=np.nan)
            h5.create_dataset("row_col", shape=(n, 2), dtype=np.int32, fillvalue=-1)
            h5.create_dataset("source_keys", shape=(n,), dtype=h5py.string_dtype())
        for m, (name, (array, attrs)) in enumerate(sorted(images.items())):
            detector = str(attrs.get("detector", name.rsplit("/", 1)[-1])).upper()
            target = "tiles" if m == 0 else f"channels/{detector}"
            if target not in h5:
                array = np.asarray(array)
                dset = h5.create_dataset(target, shape=(n, *array.shape), dtype=array.dtype, chunks=(1, *array.shape))
                dset.attrs.update({"detector": detector, "fov_m": float(plan["fov_m"]), "pixel_size_m": float(plan["fov_m"]) / int(plan["imsize"])})
            h5[target][k] = array
        h5["stage_reported_m"][k] = (list(reported) + [0.0] * 5)[:5]
        h5["commanded_xy_m"][k] = [tile["x"], tile["y"]]
        h5["acquired_s"][k] = acquired
        h5["row_col"][k] = [tile["row"], tile["col"]]
        h5["source_keys"][k] = image_key
        if spectrum is not None:
            array, attrs = next(iter(spectrum.values()))
            array = np.asarray(array)
            target = "spectrum_image" if spectrum_key == "eds_map" else "spectra"
            if target not in h5:
                dset = h5.create_dataset(target, shape=(n, *array.shape), dtype=array.dtype, chunks=(1, *array.shape))
                dset.attrs.update({key: (json.dumps(v) if isinstance(v, (list, dict)) else v) for key, v in attrs.items() if v is not None})
                h5.create_dataset("spectrum_keys", shape=(n,), dtype=h5py.string_dtype())
            h5[target][k] = array
            h5["spectrum_keys"][k] = spectrum_key if spectrum_key != "eds_map" else ""
        for name, value in (arrays or {}).items():
            value = np.asarray(value, dtype=np.float64)
            if name not in h5:
                h5.create_dataset(name, shape=(n, *value.shape), dtype=np.float64, fillvalue=np.nan)
            h5[name][k] = value
        h5["written"][k] = True
    with open(path, "rb+") as handle:
        os.fsync(handle.fileno())


class AnalysisSession:
    """One MCP session to the analysis server, used from any thread; reopened after a failure."""

    def __init__(self, url: str):
        self.url = url
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="mapper-analysis", daemon=True).start()
                self._loop = loop
            return self._loop

    async def _call(self, tool: str, arguments: dict, timeout: float) -> dict:
        from fastmcp import Client

        if self._client is None:
            client = Client(self.url, timeout=timeout)
            await client.__aenter__()
            self._client = client
        try:
            result = await asyncio.wait_for(self._client.call_tool(tool, arguments, raise_on_error=False), timeout)
        except BaseException:
            await self._drop()
            raise
        if result.is_error:
            raise RuntimeError(" ".join(getattr(b, "text", "") for b in result.content) or f"{tool} failed")
        if result.structured_content is not None:
            return result.structured_content
        return json.loads("".join(getattr(b, "text", "") for b in result.content))

    async def _drop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except BaseException:
                pass

    def call(self, tool: str, arguments: dict, timeout: float = ANALYSIS_CALL_S) -> dict:
        future = asyncio.run_coroutine_threadsafe(self._call(tool, arguments, timeout), self._ensure_loop())
        return future.result(timeout + 10.0)

    def close(self) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._drop(), self._loop).result(10.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None


class AnalysisFailed(RuntimeError):
    """The analysis server could not do (or finish) the job; the message says why and how to retry."""


class Mapper(JobDevice):
    JOB_KIND = "map acquisition: a grid of images (optionally an EDS spectrum at each) over a stage box, stitched into a zoomable map when it finishes"
    # A 600 nm square around the current stage position; per_tile ["eds_point"] adds a spectrum at each tile.
    PLAN_EXAMPLE = {"size_m": [6e-7, 6e-7], "fov_m": 1.6e-7, "per_tile": []}

    instrument_profile = device_property(
        dtype=str,
        default_value="stem",
        doc="What kind of instrument this Mapper drives (asyncroscopy/jobs/map_profile.py): stem (the default), sem, yaml "
        "(a YAML twin with an image acquisition), or a profile registered by a plug-in.",
    )
    microscope_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="Instrument to drive (a microscope, or any device the instrument profile drives).")
    scan_device_address = device_property(dtype=str, default_value="asyncroscopy/scan/default", doc="SCAN settings device.")
    eds_device_address = device_property(dtype=str, default_value="", doc="EDS settings device (needed for EDS per-tile actions).")
    data_device_address = device_property(dtype=str, default_value="", doc="DATA device; when set, the run file is registered with Tiled.")
    stage_device_address = device_property(dtype=str, default_value="asyncroscopy/stage/default", doc="STAGE device whose x/y limits every plan is checked against at reserve; blank: not checked (the microscope still refuses moves outside them).")
    output_directory = device_property(dtype=str, default_value="outputs/maps", doc="Where run files are written when there is no DATA device; with one, they go in its save path (which Tiled serves) under maps/.")
    max_tiles = device_property(dtype=int, default_value=4_000_000, doc="Refuse plans with more tiles than this (a 2000 x 2000 grid by default).")
    estimate_move_s = device_property(dtype=float, default_value=2.0, doc="Assumed time per stage move, for estimates only.")
    stage_timeout_s = device_property(dtype=float, default_value=600.0, doc="Longest one stage move may report stage_moving before the job fails (beam blanked; resumable).")
    default_beam_current_pa = device_property(dtype=float, default_value=50.0, doc="Used for dose estimates when the microscope does not report beam_current_pa; a dose budget on a stack that is not simulated refuses it.")
    max_beam_current_age_s = device_property(
        dtype=float, default_value=DEFAULT_MAX_BEAM_CURRENT_AGE_S,
        doc="Oldest screen-current measurement a dose budget accepts on hardware (asyncroscopy/routines/beam.py says why 15 min).",
    )
    eds_dose_spread_m = device_property(
        dtype=float,
        default_value=1e-8,
        doc="Side of the square (m) an EDS spectrum's parked-beam dose is counted over in the dose budget. The probe is "
        "far smaller; 10 nm is a typical damage and contamination spread, and the twin's dose cell on a 2 um sample.",
    )
    analysis_url = device_property(dtype=str, default_value="", doc="pyTEMlib analysis MCP server (e.g. http://127.0.0.1:8003/mcp) that stitches and scores finished maps; blank: no stitching.")
    stage_sigma_m = device_property(
        dtype=float,
        default_value=5e-8,
        doc="1-sigma error of the stage positions this microscope reports (readout, not where it was told to go); bounds the stitching search. "
        "50 nm suits a typical TEM goniometer read back by encoders; set it from your stage's specification.",
    )
    max_drift_m_per_s = device_property(dtype=float, default_value=5e-11, doc="Most the sample drifts relative to the stage (m/s); bounds the stitching search and the fitted drift. 0: no drift model.")

    def _beam_current(self) -> dict:
        """The current dose estimates use: {"pa", "source" ("measured" or "default"), "reason" (when default),
        "refused" (why a dose budget may not rely on it; asyncroscopy.routines.beam), "simulated"}."""
        parameters: dict = {}
        failure = None
        try:
            parameters = json.loads(self.proxy(self.microscope_device_address).get_parameters())
        except (tango.DevFailed, ValueError, TypeError) as error:
            failure = f"get_parameters failed: {str(error).strip()[:200]}"
        beam = beam_current(parameters, self._stack_simulated(parameters), float(self.max_beam_current_age_s), float(self.default_beam_current_pa))
        current = {"simulated": beam.simulated, "pa": beam.pa, "source": beam.source, "refused": beam.refused}
        if beam.age_s is not None:
            current["age_s"] = beam.age_s
        if beam.source == "default":
            current["reason"] = failure or parameters.get("beam_current_reason") or beam.refused or "the microscope reports an assumed current"
        return current

    def _stack_simulated(self, parameters: dict) -> bool:
        """Whether this stack drives a simulation: the stack's ``simulated`` device property where it has one,
        else ASYNCROSCOPY_SIMULATED, else what the microscope reports (the twin says true; AutoScript nothing, so false)."""
        flag = getattr(self, "simulated", None)
        if flag is not None:
            return bool(flag)
        env = os.environ.get("ASYNCROSCOPY_SIMULATED", "").strip().lower()
        if env:
            return env in ("1", "true", "yes", "on")
        return bool(parameters.get("simulated", False))

    def _refuse_assumed_current(self, current: dict, budget) -> None:
        """A dose budget means nothing with an assumed current on a real instrument: refuse it there."""
        if budget is not None and current.get("refused"):
            what = "is assumed" if current["source"] == "default" else "cannot be used for a dose budget"
            raise ValueError(
                f"this plan has a dose budget, but the beam current {what} ({current['pa']:g} pA; {current['refused']}). "
                "On an instrument the budget needs a recent measured current: insert the screen, call get_screen_current, "
                "retract the screen and reserve again, or leave out the dose budget"
            )

    @staticmethod
    def _clock_s(microscope) -> float:
        """The microscope's clock_s (simulated time on a twin); this computer's time if it reports none."""
        try:
            return float(microscope.read_attribute("clock_s").value)
        except (tango.DevFailed, AttributeError, TypeError, ValueError):
            return time.time()

    @staticmethod
    def _tile_half(plan: dict, fov: float | None = None) -> tuple[float, float]:
        """Half the width and height of an image of ``fov`` (default a tile); tiles may be rectangular (plan tile_m)."""
        if fov is not None:
            return fov / 2.0, fov / 2.0
        width, height = plan.get("tile_m") or (plan["fov_m"], plan["fov_m"])
        return float(width) / 2.0, float(height) / 2.0

    def check_dry_call(self, job: dict, device: str, name: str, arguments) -> dict | None:
        """A dry run's check of each recorded stage move (does the image it is for lie inside the plan box?)
        and image shift (live drift tracking: within the plan's max_total_m of the shift before the map?)."""
        if name == "set_image_shift":
            return live_drift.check_shift(job, arguments)
        target = profile_for(self).move_target(job["resolved_plan"], name, arguments)
        if target is None:
            return None
        plan = job["resolved_plan"]
        x_min, x_max, y_min, y_max = plan["box_m"]
        step = self.current_step()
        phase = self.step_at(job, step)["phase"] if step is not None else "tiles"
        half_x, half_y = self._tile_half(plan, float(plan["revisit"]["fov_m"]) if phase == "revisit" else None)
        tol = EDGE_TOLERANCE_M
        inside = all(
            low + half - tol <= float(target[axis]) <= high - half + tol
            for axis, low, high, half in (("x", x_min, x_max, half_x), ("y", y_min, y_max, half_y))
            if axis in target
        )
        xy = [float(target["x"]), float(target["y"])] if "x" in target and "y" in target else {k: float(v) for k, v in target.items()}
        return {"ok": inside, "inside_box": inside, "box_m": [x_min, x_max, y_min, y_max], "xy_m": xy, "image_half_width_m": half_x}

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def read_job_kind(self) -> str:
        return profile_for(self).job_kind or self.JOB_KIND

    def read_plan_example(self) -> str:
        example = profile_for(self).plan_example
        return json.dumps(example) if example is not None else super().read_plan_example()

    def example_plan(self, request: str) -> dict | None:
        profile = profile_for(self)
        if profile.plan_example is not None:
            return dict(profile.plan_example)
        example = dict(self.PLAN_EXAMPLE)
        if "dose" in request.lower():  # a caller who mentioned dose sees where a dose limit goes
            example["dose_budget_e_per_a2"] = 50.0
        return example

    def _check_needs(self, profile: InstrumentProfile, plan: dict) -> None:
        """Refuse a plan whose instrument lacks a command the profile needs, at reserve, before anything moves."""
        needs = profile.needed_commands(plan)
        shift_commands = live_drift.needed_commands(plan, profile)
        if shift_commands:
            needs.setdefault(self.microscope_device_address, []).extend(shift_commands)
        for address, commands in needs.items():
            try:
                offered = {str(n).lower() for n in self.proxy(address, timeout_s=3.0).get_command_list()}
            except Exception as exc:  # DevFailed: the device is down; the job's first step would fail the same way
                raise ValueError(f"the {profile.name!r} instrument profile drives {address}, which did not answer ({str(exc).strip().splitlines()[0][:200]})") from None
            missing = [c for c in dict.fromkeys(commands) if c.lower() not in offered]
            if missing:
                raise ValueError(
                    f"the {profile.name!r} instrument profile needs {', '.join(missing)} on {address}, which does not offer them; "
                    "set this Mapper's instrument_profile to one the instrument supports"
                )

    def resolve_plan(self, plan: dict) -> dict:
        profile = profile_for(self)
        check_plan_fields(plan)
        validate(plan, "map_plan")
        check_profile_fields(plan, profile)
        revisit_only = isinstance(plan.get("revisit"), dict) and ("points" in plan["revisit"] or "map" in plan["revisit"])
        if revisit_only:
            return self._resolve_revisit_only(plan)
        if ("box_m" in plan) == ("size_m" in plan):
            raise ValueError("give the area as box_m [x_min, x_max, y_min, y_max] or as size_m [width, height], not both or neither")
        p = {**GENERIC_PLAN_DEFAULTS, **profile.plan_defaults, **plan}
        profile.resolve(p)
        if "fov_m" not in p:
            raise ValueError("fov_m (the field of view of each tile, metres) is required: add \"fov_m\": 2e-7 (200 nm tiles) unless the request gives a tile size")
        if "size_m" in p:
            # The device does the arithmetic: callers (often small models) get centre +/- half wrong.
            width, height = (float(v) for v in p["size_m"])
            cx, cy = (float(v) for v in (p.get("centre_m") or profile.current_xy()))
            p["box_m"] = [cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2]
        x_min, x_max, y_min, y_max = (float(v) for v in p["box_m"])
        if not (x_min < x_max and y_min < y_max):
            raise ValueError("box_m must be [x_min, x_max, y_min, y_max] with min < max")
        if (p["find_regions"] or p.get("revisit")) and not (p["stitch"] and self.analysis_url):
            raise ValueError("find_regions and revisit need the map stitched: keep stitch on, on a Mapper with an analysis_url")
        if p.get("revisit") and not p["find_regions"]:
            raise ValueError("revisit images the regions find_regions finds (give find_regions > 0), or give revisit points [[x, y], ...] or an existing map (revisit map, top_k)")
        fov = float(p["fov_m"])
        revisit = None
        if p.get("revisit"):
            revisit = profile.resolve_revisit(p, p["revisit"])
            if not fov * REVISIT_FOV_MIN_FRACTION <= revisit["fov_m"] <= fov:
                raise ValueError(f"revisit fov_m must be between {fov * REVISIT_FOV_MIN_FRACTION:.3g} m (a twentieth of the tile) and the tile's {fov:.3g} m")
            p["revisit"] = revisit
        tile_w, tile_h = profile.tile_size_m(p)
        if tile_h != tile_w:
            p["tile_m"] = [tile_w, tile_h]
        grid = compact_grid(*grid_axes([x_min, x_max, y_min, y_max], tile_w, float(p["overlap"]), limit=int(self.max_tiles), fov_y=tile_h if tile_h != tile_w else None))
        tiles = grid["rows"] * grid["cols"]
        live_drift.resolve(p, profile, (tile_w, tile_h), tiles)
        if tiles == 1:
            # One tile is an image, not a map: almost always the area's size put in fov_m. A strip one tile high is a map.
            raise ValueError(f"fov_m {fov:.3g} m covers the whole area, so the map would be one tile; fov_m is the field of view of each tile (2e-7 when the request gives no tile size). To image given places, use a revisit plan: {{\"revisit\": {{\"fov_m\": ..., \"points\": [[x, y], ...]}}}}")

        stitch = bool(p["stitch"] and self.analysis_url)
        regions = int(p["find_regions"]) if stitch else 0
        revisits = regions if revisit else 0
        budgets = plan_budgets(p)
        costs, extra = profile.costs(p, grid, revisit, revisits, budgets)
        revisit_s = revisits * (float(self.estimate_move_s) + (profile.revisit_seconds(revisit) if revisit else 0.0))
        duration = tiles * (float(self.estimate_move_s) + profile.tile_seconds(p)) + revisit_s
        budgeted = check_budgets(budgets, costs, duration, profile)
        estimate = {
            "steps": tiles + int(stitch) + int(regions > 0) + revisits,
            "tiles": tiles,
            "duration_s": round(duration, 1),
            **extra,
            "phases": ["tiles"] + (["stitch"] if stitch else []) + (["score"] if regions else []) + (["revisit"] if revisits else []),
        }
        if budgeted:
            estimate["budgets"] = budgeted
        if not stitch:
            estimate["stitch"] = "off" if not p["stitch"] else "this Mapper has no analysis_url; stitch the run with assemble_map"
        resolved = {
            **p,
            "box_m": [x_min, x_max, y_min, y_max],
            "stitch": stitch,
            "find_regions": regions,
            "step_count": tiles + int(stitch) + int(regions > 0) + revisits,
            "estimate": estimate,
            "grid": grid,
        }
        if isinstance(self, JobDevice):  # a device (not a test's stand-in): the instrument must offer what the profile drives
            self._check_needs(profile, resolved)
        return resolved

    def _resolve_revisit_only(self, plan: dict) -> dict:
        """A plan that only images given points, or an existing map's best regions: no tiles, no stitch."""
        profile = profile_for(self)
        p = {**GENERIC_PLAN_DEFAULTS, **profile.plan_defaults, **plan}
        revisit = dict(p["revisit"])
        if ("points" in revisit) == ("map" in revisit):
            raise ValueError("give revisit points [[x, y], ...] or an existing map (revisit map, top_k), not both")
        if p.get("find_regions") or p.get("per_tile"):
            raise ValueError("a revisit-only plan takes no tiles: leave out find_regions and per_tile, or give an area to map")
        if live_drift.tracking(p):
            raise ValueError("live drift tracking (drift.track) corrects tiles of a map, and a revisit-only plan has none: leave out drift")
        profile.resolve(p)
        revisit.update(profile.resolve_revisit(p, revisit))
        fov = float(revisit["fov_m"])
        half = fov / 2.0
        regions: list[dict] = []
        if "points" in revisit:
            regions = [{"id": f"point:{k}", "x_m": float(x), "y_m": float(y)} for k, (x, y) in enumerate(revisit["points"])]
            xs = [r["x_m"] for r in regions]
            ys = [r["y_m"] for r in regions]
            box = p.get("box_m") or [min(xs) - half, max(xs) + half, min(ys) - half, max(ys) + half]
            count = len(regions)
        else:
            if not self.analysis_url:
                raise ValueError("revisiting an existing map's regions needs a Mapper with an analysis_url")
            info = self._analysis("map_info", {"map": revisit["map"]})
            box = p.get("box_m") or list(info["stage_box_m"])
            count = int(revisit.get("top_k", 3))
        x_min, x_max, y_min, y_max = (float(v) for v in box)
        budgets = plan_budgets(p)
        costs, extra = profile.revisit_only_costs(p, revisit, count, budgets)
        duration = count * (float(self.estimate_move_s) + profile.revisit_seconds(revisit))
        budgeted = check_budgets(budgets, costs, duration, profile)
        for r in regions:
            if not (x_min + half <= r["x_m"] <= x_max - half and y_min + half <= r["y_m"] <= y_max - half):
                raise ValueError(f"revisit point ({r['x_m']:.3e}, {r['y_m']:.3e}) m is outside the approved area {[x_min, x_max, y_min, y_max]} for a {fov:.2e} m image")
        steps = count + (1 if "map" in revisit else 0)
        estimate = {
            "steps": steps, "tiles": 0, "revisits": count,
            "duration_s": round(duration, 1),
            **extra,
            "phases": (["select"] if "map" in revisit else []) + ["revisit"],
        }
        if budgeted:
            estimate["budgets"] = budgeted
        resolved = {
            **p, "fov_m": fov, "revisit": revisit, "box_m": [x_min, x_max, y_min, y_max], "stitch": False, "find_regions": 0,
            "step_count": steps, "estimate": estimate, "grid": {"rows": 0, "cols": 0, "x0": 0.0, "dx": 0.0, "y0": 0.0, "dy": 0.0},
            "regions": regions, "revisit_only": True,
        }
        if isinstance(self, JobDevice):
            self._check_needs(profile, resolved)
        return resolved

    def step_at(self, job: dict, index: int) -> dict:
        if "steps" in job:  # a job saved before plans were computed lazily: tiles only
            return {"phase": "tiles", **job["steps"][index]}
        plan = job["resolved_plan"]
        if plan.get("revisit_only"):
            if "map" in plan["revisit"]:
                return {"phase": "select"} if index == 0 else {"phase": "revisit", "region": index - 1}
            return {"phase": "revisit", "region": index}
        grid = plan["grid"]
        tiles = grid["rows"] * grid["cols"]
        if index < tiles:
            return {"phase": "tiles", **tile_at(grid, plan["order"], index)}
        k = index - tiles
        if plan["stitch"] and k == 0:
            return {"phase": "stitch"}
        k -= int(plan["stitch"])
        if plan["find_regions"] and k == 0:
            return {"phase": "score"}
        return {"phase": "revisit", "region": k - int(bool(plan["find_regions"]))}

    def step_targets(self, job: dict) -> list[dict]:
        """The stage x and y the plan goes to at its extremes (outermost tile centres, given revisit points),
        checked at reserve against the stage's limits (the profile says where they are: the STAGE device, or
        the instrument's own stage settings), so a plan off the stage is refused before anything moves."""
        plan = job["resolved_plan"]
        grid = plan["grid"]
        xs: list[tuple[str, float]] = []
        ys: list[tuple[str, float]] = []
        if grid["rows"] and grid["cols"]:
            xs += [("tiles (left column)", grid["x0"]), ("tiles (right column)", grid["x0"] + (grid["cols"] - 1) * grid["dx"])]
            ys += [("tiles (first row)", grid["y0"]), ("tiles (last row)", grid["y0"] + (grid["rows"] - 1) * grid["dy"])]
        for region in plan.get("regions") or []:
            xs.append((f"revisit {region['id']}", region["x_m"]))
            ys.append((f"revisit {region['id']}", region["y_m"]))
        if plan.get("revisit_only") and "map" in plan["revisit"]:  # the map's regions are chosen later, inside its box
            x_min, x_max, y_min, y_max = plan["box_m"]
            xs += [("revisit area", x_min), ("revisit area", x_max)]
            ys += [("revisit area", y_min), ("revisit area", y_max)]
        return profile_for(self).stage_targets(plan, xs, ys)

    def _tile_count(self, job: dict) -> int:
        if "steps" in job:
            return len(job["steps"])
        grid = job["resolved_plan"]["grid"]
        return grid["rows"] * grid["cols"]

    def phase(self, job: dict) -> str | None:
        """Which phase a job is in, for job_status."""
        if "resolved_plan" not in job:
            return None
        tiles = self._tile_count(job)
        done = done_count(job)
        if done < tiles:
            return f"acquiring tiles ({done} of {tiles})"
        if job.get("state") in {"done", "aborted", "failed", "released"}:
            return None
        return self.step_at(job, done)["phase"] if done < job.get("step_count", tiles) else None

    # ------------------------------------------------------------------
    # Run file
    # ------------------------------------------------------------------
    def _run_path(self, job: dict) -> Path:
        if "run_path" not in job:
            # Tiled only serves files under the DATA save path, so the run file must live there.
            base = Path(self._data_proxy().save_path) / "maps" if self.data_device_address else Path(self.output_directory)
            job["run_path"] = str(base / f"map_{job['reservation_id']}.h5")
        return Path(job["run_path"])

    def _file(self, job: dict) -> Path:
        """Where the run file is now: ``<run>.h5.partial`` until its tiles are done, so DATA's reconcile leaves it alone."""
        final = self._run_path(job)
        return final if final.exists() else final.with_name(final.name + ".partial")

    def _finalize_run_file(self, job: dict) -> tuple[Path, str | None]:
        """Give the run file its final ``.h5`` name; (path, note). Never raises: the data is complete either way.

        On Windows a rename fails while anything has the file open, so it is retried
        for a few seconds and then copied instead.
        """
        final = self._run_path(job)
        working = self._file(job)
        if working == final:
            return final, None
        last: OSError | None = None
        for delay in RENAME_RETRY_DELAYS_S:
            time.sleep(delay)
            try:
                working.replace(final)
                return final, None
            except OSError as exc:
                last = exc
        try:
            shutil.copy2(working, final)
        except OSError as exc:
            return working, f"the run file could not be renamed ({last}) or copied ({exc}); it stays at {working}"
        try:
            working.unlink()
        except OSError:
            return final, f"the run file was in use, so it was copied to {final.name}; {working.name} is left behind ({last})"
        return final, f"the run file was in use, so it was copied to {final.name} ({last})"

    def _register_run(self, path: Path) -> tuple[str | None, str | None]:
        """Register a finished run file with DATA; (key, note). A failure is a note: the data is on disk."""
        if not self.data_device_address or path.suffix != ".h5":
            return None, None
        try:
            return self.proxy(self.data_device_address).register_path(str(path)), None
        except Exception as exc:  # Tango raises DevFailed
            return None, f"the run file is complete but was not registered with Tiled ({exc}); DATA's reconcile registers it later"

    def _close_run_file(self, job: dict) -> None:
        """Finish the run file once: trace, final name, Tiled registration; results go in result_keys."""
        if job.get("run_closed"):
            return
        self._write_trace(job)
        path, note = self._finalize_run_file(job)
        key, register_note = self._register_run(path)
        with self._lock:
            job["result_keys"]["run_file"] = str(path)
            if key is not None:
                job["result_keys"]["run"] = key
            notes = [n for n in (note, register_note) if n]
            if notes:
                job["result_keys"]["run_file_note"] = "; ".join(notes)
            job["run_closed"] = True
            self._save(job)

    def on_start(self, job: dict) -> None:
        plan = job["resolved_plan"]
        # The instrument's settings for the plan, ending safe (beam blanked, tip retracted): it stays safe
        # between tiles and is made unsafe only while a tile is acquired (see the profile's acquire_tile).
        profile_for(self).setup(job)
        live_drift.start(job, profile_for(self))  # tracking on: the image shift before the map, put back at the end
        if job["resolved_plan"].get("revisit_only") or job.get("dry_run"):
            job["run_closed"] = True  # no tiles (or, in a dry run, no images), so no run file
        if job.get("run_closed"):
            return  # resuming after the tiles: only pipeline steps are left
        final = self._run_path(job)
        if final.exists():  # on_start runs only for unfinished jobs
            # A job persisted before run files were .partial wrote the final name directly;
            # move it back so reconcile does not register it half-written.
            try:
                final.replace(final.with_name(final.name + ".partial"))
            except OSError:
                pass
        path = self._file(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "a") as h5:
            h5.attrs["schema_version"] = ENVELOPE_VERSION
            h5.attrs["run_layout"] = RUN_LAYOUT
            h5.attrs["run_id"] = job["reservation_id"]
            h5.attrs["plan_id"] = job["plan_id"]
            h5.attrs["plan"] = json.dumps(plan)
            h5.attrs["grid"] = json.dumps({"rows": plan["grid"]["rows"], "cols": plan["grid"]["cols"]})
            h5.attrs["axes"] = "images: rows=+y, columns=+x; positions in stage metres"
            h5.attrs["microscope"] = self.microscope_device_address
            h5.attrs["instrument_profile"] = profile_for(self).name
            h5.attrs["created"] = time.time()
        job["result_keys"]["run_file"] = str(path)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def run_step(self, job: dict, index: int) -> dict:
        step = self.step_at(job, index)
        if step["phase"] == "tiles":
            return self._tile(job, step)
        live_drift.finish(job, profile_for(self))  # the tiles are done: the image shift goes back before any revisit move
        if job.get("dry_run") and not (step["phase"] == "revisit" and job["resolved_plan"].get("regions")):
            # Stitch, score and their revisits need images; a dry run took none. Given revisit points still rehearse their moves.
            return {"phase": step["phase"], "skipped": DRY_RUN_SKIPPED}
        self._close_run_file(job)  # the tiles are done: stitching reads the finished, registered run
        if step["phase"] == "select":
            return self._select(job)
        if step["phase"] == "stitch":
            return self._stitch(job)
        if step["phase"] == "score":
            return self._score(job)
        return self._revisit(job, step["region"])

    def _tile(self, job: dict, tile: dict) -> dict:
        plan = job["resolved_plan"]
        index = tile["index"]
        half_x, half_y = self._tile_half(plan)
        if not inside_box(plan["box_m"], tile["x"], tile["y"], half_x, half_y):
            raise RuntimeError(f"tile {index} at ({tile['x']:.3e}, {tile['y']:.3e}) lies outside the plan box; refusing to move")
        profile = profile_for(self)
        if not self._move(job, profile, tile["x"], tile["y"]):
            return None  # aborted during a long move; a resume moves again
        reported = [float(v) for v in profile.stage_reported(job)]
        # Live drift tracking (plan drift.track): a bounded image shift cancels the drift predicted for this tile.
        correction = live_drift.correct(self, job, profile, tile)
        # The profile makes the instrument unsafe (unblanks the beam) only for this tile's acquisitions, timed on
        # the instrument's clock, and safe again before returning, even on an error.
        acquired_tile = profile.acquire_tile(job, tile)
        if acquired_tile is None:
            return None  # aborted part way through the tile (a spectrum image); a resume redoes this tile
        image_key, acquired = acquired_tile["image_key"], acquired_tile["acquired"]
        spectrum, spectrum_key = acquired_tile.get("spectrum"), acquired_tile.get("spectrum_key")
        job.setdefault("first_tile_s", acquired)
        if job.get("dry_run"):
            # The commands above were recorded, not sent: there is no image to read or store.
            live_drift.measure(self, job, tile, None, acquired)
            record = {"index": index, "row": tile["row"], "col": tile["col"], "commanded_xy_m": [tile["x"], tile["y"]], "stage_reported_m": reported, "clock_s": acquired, "acquisition": DRY_RUN_SKIPPED}
            if correction is not None:
                record["drift"] = correction
            return record
        images = read_acquisition(image_key, self._data_proxy())
        if correction is not None:
            correction.update(live_drift.measure(self, job, tile, images, acquired) or {})
        if spectrum is None and spectrum_key is not None:
            spectrum = read_acquisition(spectrum_key, self._data_proxy())
        record = {"index": index, "row": tile["row"], "col": tile["col"], "commanded_xy_m": [tile["x"], tile["y"]], "stage_reported_m": reported, "key": image_key, "clock_s": acquired}
        if spectrum_key is not None:
            record["spectrum_key"] = spectrum_key
        if correction is not None:
            record["drift"] = correction
        write_tile(self._file(job), self._tile_count(job), tile, plan, reported, acquired, image_key, images, spectrum, spectrum_key, live_drift.tile_arrays(job, correction))
        return record

    @staticmethod
    def _move(job: dict, profile: InstrumentProfile, x: float, y: float) -> bool:
        """One stage move with the profile's preparation and recovery (a probe retracts, moves, approaches); False if aborted."""
        profile.before_move(job)
        if not profile.move(job, x, y):
            return False
        profile.after_move(job)
        return True

    def _wait_for_stage(self, job: dict, microscope: tango.DeviceProxy) -> bool:
        """Wait for a long move to end (a microscope that returns before arriving says so in stage_moving); False if aborted.

        A stage still moving after stage_timeout_s (a stuck encoder, a lost arrival signal) fails the
        step, so the job stops with the beam blanked and can be resumed, rather than waiting forever.
        """
        deadline = time.monotonic() + float(self.stage_timeout_s)
        while True:
            try:
                moving = bool(microscope.read_attribute("stage_moving").value)
            except tango.DevFailed:  # a microscope whose move_stage returns only on arrival
                return True
            if not moving:
                return True
            if self.aborting(job):
                return False
            if time.monotonic() > deadline:
                raise RuntimeError(f"the stage still reports stage_moving after {float(self.stage_timeout_s):g} s (stage_timeout_s); check the stage, then resume the job")
            time.sleep(STAGE_POLL_S)

    # ------------------------------------------------------------------
    # Pipeline: stitch, score, revisit (on the analysis server)
    # ------------------------------------------------------------------
    def _analysis(self, tool: str, arguments: dict) -> dict:
        """One call to the analysis server over this Mapper's session."""
        session = getattr(self, "_analysis_session", None)
        if session is None or session.url != self.analysis_url:
            session = self._analysis_session = AnalysisSession(self.analysis_url)
        return session.call(tool, arguments)

    def _submit(self, job: dict, slot: str, tool: str, arguments: dict) -> str:
        """Start (or adopt) an analysis job and save its id in the job before waiting on it."""
        pipeline = self._pipeline(job)
        if pipeline.get(slot):
            return pipeline[slot]  # a resumed job waits for the analysis job it already started
        started = None
        for delay in (0.0, *ANALYSIS_BACKOFF_S):
            time.sleep(delay)
            if self.aborting(job):
                raise AnalysisFailed("aborted")
            try:
                started = self._analysis(tool, arguments)
                break
            except Exception as exc:  # down, restarting, or the call timed out: try again; a busy reply below adopts
                last = exc
        if started is None:
            raise AnalysisFailed(f"the analysis server at {self.analysis_url} did not answer ({type(last).__name__}: {last})")
        if "error" in started:
            if started["error"].get("code") == "busy" and started.get("job_id"):
                started = {"job_id": started["job_id"]}  # someone (or this job, before a timeout) is already on it
            else:
                raise AnalysisFailed(started["error"].get("message", str(started["error"])))
        with self._lock:
            pipeline[slot] = started["job_id"]
            self._save(job)
        return started["job_id"]

    def _await(self, job: dict, job_id: str, deadline_s: float) -> dict | None:
        """Results of an analysis job; None if this job was aborted; AnalysisFailed if it failed, vanished or took too long."""
        deadline = time.monotonic() + deadline_s
        failures = 0
        while True:
            if self.aborting(job):
                try:
                    self._analysis("analysis_job_abort", {"job_id": job_id})
                except Exception:
                    pass
                return None
            if time.monotonic() > deadline:
                raise AnalysisFailed(f"analysis job {job_id} did not finish within {deadline_s:.0f} s")
            try:
                status = self._analysis("analysis_job_status", {"job_id": job_id, "wait_s": ANALYSIS_WAIT_S})
                failures = 0
            except Exception as exc:
                failures += 1
                if failures > len(ANALYSIS_BACKOFF_S):
                    raise AnalysisFailed(f"lost the analysis server at {self.analysis_url} ({type(exc).__name__}: {exc})") from exc
                time.sleep(ANALYSIS_BACKOFF_S[failures - 1])
                continue
            if "error" in status or "state" not in status:
                # unknown_job: the server restarted and forgot the job.
                raise AnalysisFailed(f"the analysis server no longer knows job {job_id} ({(status.get('error') or {}).get('message', status)}); it may have restarted")
            if status["state"] in {"done", "failed", "aborted", "released"}:
                break
        if status["state"] != "done":
            raise AnalysisFailed(f"the analysis job ended {status['state']}: {(status.get('last_error') or {}).get('message', '')}")
        return self._analysis("analysis_job_results", {"job_id": job_id})

    def _pipeline(self, job: dict) -> dict:
        return job.setdefault("pipeline", {})

    def _stitch(self, job: dict) -> dict:
        pipeline = self._pipeline(job)
        run = job["result_keys"].get("run") or job["result_keys"].get("run_file")
        plan = {"run": run, "stage_sigma_m": float(self.stage_sigma_m), "max_drift_m_per_s": float(self.max_drift_m_per_s), "eds_pois": True}
        deadline = STITCH_DEADLINE_S[0] + STITCH_DEADLINE_S[1] * self._tile_count(job)
        try:
            results = self._await(job, self._submit(job, "stitch_job", "assemble_map", plan), deadline)
        except AnalysisFailed as exc:  # the tiles are safe, the job goes on
            if str(exc) == "aborted":
                return None
            pipeline["stitch_error"] = f"{exc}. The tiles are in {run}; stitch them later with assemble_map(run={run!r})."
            pipeline.pop("stitch_job", None)
            return {"phase": "stitch", "ok": False}
        if results is None:
            return None  # aborted while stitching; a resume waits for (or restarts) the same stitch
        pipeline["map"] = results["result_keys"]["map"]
        summary = results.get("summary", {})
        pipeline["drift_fit"] = summary.get("drift")  # kept for revisits; the reply shows only its totals
        pipeline["stitching"] = {k: v for k, v in summary.items() if k in ("tiles", "levels", "canvas_shape", "matches_used", "median_residual_px", "p95_residual_px", "fallback_tiles", "eds_pois", "solver")}
        if summary.get("drift"):
            pipeline["stitching"]["drift"] = {k: summary["drift"][k] for k in ("total_nm", "mean_rate_nm_per_s") if k in summary["drift"]}
        with self._lock:
            job["result_keys"]["map"] = pipeline["map"]
        return {"phase": "stitch", "ok": True, "map": pipeline["map"]}

    def _score(self, job: dict) -> dict:
        pipeline = self._pipeline(job)
        if "map" not in pipeline:
            return {"phase": "score", "skipped": "no map"}
        if "regions" in pipeline:
            return {"phase": "score", "ok": True, "regions": len(pipeline["regions"])}  # already scored before a resume
        n = int(job["resolved_plan"]["find_regions"])
        # The job's tag makes scoring idempotent: a retry replaces the regions it added before.
        arguments = {"map": pipeline["map"], "top_k": n, "tag": job["reservation_id"]}
        try:
            results = self._await(job, self._submit(job, "score_job", "score_map", arguments), SCORE_DEADLINE_S)
        except AnalysisFailed as exc:
            if str(exc) == "aborted":
                return None
            pipeline["score_error"] = f"{exc}. Score again with score_map(map={pipeline['map']!r}, top_k={n})."
            pipeline.pop("score_job", None)
            return {"phase": "score", "ok": False}
        if results is None:
            return None
        pipeline["regions"] = [
            {k: p.get(k) for k in ("id", "x_m", "y_m", "score", "reason", "scorer_version", "window_m")} for p in results.get("summary", {}).get("added_pois", [])
        ]
        return {"phase": "score", "ok": True, "regions": len(pipeline["regions"])}

    def _select(self, job: dict) -> dict:
        """Take an existing map's best scored regions (revisit-only plans with a map)."""
        pipeline = self._pipeline(job)
        if "regions" in pipeline:
            return {"phase": "select", "regions": len(pipeline["regions"])}  # already chosen before a resume
        revisit = job["resolved_plan"]["revisit"]
        reply = self._analysis("list_pois", {"map": revisit["map"], "kind": "candidate", "limit": 500})
        candidates = sorted((p for p in reply.get("pois", []) if p.get("score") is not None), key=lambda p: p["score"], reverse=True)
        top = candidates[: int(revisit.get("top_k", 3))]
        with self._lock:
            pipeline["map"] = revisit["map"]
            pipeline["regions"] = [{k: c.get(k) for k in ("id", "x_m", "y_m", "score", "reason", "scorer_version", "window_m")} for c in top]
            if not top:
                pipeline["select_error"] = f"{revisit['map']} has no scored regions; score it first with score_map(map={revisit['map']!r})"
            self._save(job)
        return {"phase": "select", "regions": len(top)}

    def _link_pending(self, job: dict) -> None:
        """Add the point of interest of every revisit image that has none yet (a link lost to a failure or an abort)."""
        pipeline = self._pipeline(job)
        if "map" not in pipeline:
            return
        regions = {r["id"]: r for r in pipeline.get("regions") or []}
        for visit in pipeline.get("revisits", []):
            if "image" not in visit or "poi" in visit or visit["region"] not in regions:
                continue
            region = regions[visit["region"]]
            try:
                poi = self._analysis(
                    "add_poi",
                    {"map": pipeline["map"], "x_m": float(region["x_m"]), "y_m": float(region["y_m"]), "kind": "high_mag", "reason": f"higher-magnification image of region {region['id']}", "links": [visit["image"], f"poi:{region['id']}"], "id": f"{job['reservation_id']}:revisit:{region['id']}"},
                )
                with self._lock:
                    visit["poi"] = poi.get("id")
                    visit.pop("poi_error", None)
                    self._save(job)
            except Exception as exc:  # the image is saved either way; the next revisit step or the finish tries again
                visit["poi_error"] = f"{type(exc).__name__}: {exc}"

    def _revisit(self, job: dict, k: int) -> dict:
        pipeline = self._pipeline(job)
        if "regions" not in pipeline and job["resolved_plan"].get("regions"):
            pipeline["regions"] = job["resolved_plan"]["regions"]  # revisit-only plan with given points
        self._link_pending(job)
        regions = pipeline.get("regions") or []
        if k >= len(regions):
            return {"phase": "revisit", "skipped": "no such region"}
        region = regions[k]
        visits = pipeline.setdefault("revisits", [])
        visit = next((v for v in visits if v.get("region") == region["id"]), None)
        plan = job["resolved_plan"]
        revisit = plan["revisit"]
        fov = float(revisit["fov_m"])
        if visit is None:
            profile = profile_for(self)
            # The map places content at stage reading + drift(t) (pyTEMlib's fit), so to put the region under
            # the probe now the stage goes to its map position minus the drift, extrapolated to now on the
            # instrument's clock (the one the tiles were timed with).
            now = profile.clock_s()
            dx, dy = drift_at(pipeline.get("drift_fit"), now - float(job.get("first_tile_s", now)))
            # With live tracking the tiles were corrected, so the stitch fitted only what tracking left; the
            # tracker's own estimate (on the same clock, from the same first tile) is the rest.
            tx, ty = live_drift.estimate_at(job, now)
            dx, dy = dx + tx, dy + ty
            x, y = float(region["x_m"]) - dx, float(region["y_m"]) - dy
            x_min, x_max, y_min, y_max = plan["box_m"]
            half = fov / 2.0
            # The image must stay inside the approved box. A region near its edge is still imaged, off centre,
            # with the image moved inward, as long as the region keeps a margin inside that image.
            cx = min(max(x, x_min + half), x_max - half) if x_max - x_min >= fov else None
            cy = min(max(y, y_min + half), y_max - half) if y_max - y_min >= fov else None
            if cx is None or cy is None or abs(cx - x) > REVISIT_EDGE_SHIFT * half or abs(cy - y) > REVISIT_EDGE_SHIFT * half:
                note = f"region {region['id']} at ({x:.3e}, {y:.3e}) m (drift-corrected) is too close to the edge of the plan box for a {fov:.2e} m image; not visited"
                visits.append({"region": region["id"], "skipped": note})
                return {"phase": "revisit", "skipped": note}
            x, y = cx, cy
            if not self._move(job, profile, x, y):
                return None
            image_key = profile.acquire_revisit(job, revisit)
            # clock_s: the instrument time the drift correction (drift_xy_m) was extrapolated to.
            visit = {"region": region["id"], "image": image_key, "fov_m": fov, "stage_xy_m": [x, y], "clock_s": now, "drift_xy_m": [dx, dy]}
            if job.get("dry_run"):
                visit = {"region": region["id"], "acquisition": DRY_RUN_SKIPPED, "fov_m": fov, "stage_xy_m": [x, y], "clock_s": now, "drift_xy_m": [dx, dy]}
            with self._lock:  # saved before the point is added, so a resume links this image instead of taking another
                visits.append(visit)
                self._save(job)
        self._link_pending(job)
        return {"phase": "revisit", **visit}

    # ------------------------------------------------------------------
    # Ending
    # ------------------------------------------------------------------
    def _data_proxy(self):
        return self.proxy(self.data_device_address) if self.data_device_address else None

    def on_stop(self, job: dict, reason: str) -> None:
        profile_for(self).make_safe(job)  # blank the beam, retract the tip
        live_drift.finish(job, profile_for(self))  # tracking on: the image shift goes back to its value before the map
        if not job.get("run_closed"):
            self._write_trace(job)

    def on_finish(self, job: dict) -> dict:
        # A finished map must not leave the beam parked (or the tip engaged) on the last tile.
        profile_for(self).make_safe(job)
        live_drift.finish(job, profile_for(self))
        self._close_run_file(job)
        self._link_pending(job)
        return {}

    def _release(self, request: str) -> str:
        """Release; an unfinished run gets its final name and is sent for stitching in the background, so its tiles stay reachable."""
        reply = super()._release(request)
        try:
            reservation_id = json.loads(request)["reservation_id"]
            job = self._jobs.get(reservation_id)
            if job is not None and not job.get("run_closed") and self._file(job).exists():
                self._close_run_file(job)
                run = job["result_keys"].get("run")
                if run and job["resolved_plan"].get("stitch"):
                    # Never a call to the analysis server from this Tango command: it would block the device.
                    threading.Thread(target=self._stitch_released, args=(job, run), name="mapper-release-stitch", daemon=True).start()
        except Exception as exc:  # the release itself succeeded; never turn it into an error
            self.warn_stream(f"Released, but the run file was not finalized: {exc}")
        return reply

    def _stitch_released(self, job: dict, run: str) -> None:
        with tango.EnsureOmniThread():
            try:
                started = self._analysis("assemble_map", {"run": run, "stage_sigma_m": float(self.stage_sigma_m), "max_drift_m_per_s": float(self.max_drift_m_per_s), "eds_pois": True})
                with self._lock:
                    job["result_keys"]["map_job"] = started.get("job_id")
            except Exception as exc:
                with self._lock:
                    self._pipeline(job)["stitch_error"] = f"{type(exc).__name__}: {exc}. Stitch the tiles with assemble_map(run={run!r})."
            with self._lock:
                self._save(job)

    def _write_trace(self, job: dict) -> None:
        """This job's own events (reserve, start, stop, approvals, failed steps) into the run file.

        Only the part of the shared trace written since this job was reserved is read, line by
        line, so the cost is this job's, not the history's. Each tile's position and time are
        already in the run's per-tile arrays, so successful per-tile events are left out, and at
        most TRACE_LINES_KEPT lines are copied; the run file names the trace file and offset.
        """
        path = self._file(job)
        if not path.exists():
            return
        trace = Path(self.trace_path)
        offset = int(job.get("trace_offset", 0))
        events: list[str] = []
        dropped = 0
        marker = f'"run_id": "{job["reservation_id"]}"'
        if trace.exists():
            with trace.open("rb") as handle:
                handle.seek(min(offset, trace.stat().st_size))
                for raw in handle:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if marker not in line or ('.step"' in line and '"ok": true' in line):
                        continue
                    if len(events) < TRACE_LINES_KEPT:
                        events.append(line)
                    else:
                        dropped += 1
        with h5py.File(path, "a") as h5:
            if "trace" in h5:
                del h5["trace"]
            h5.create_dataset("trace", data=np.array(events, dtype=h5py.string_dtype()))
            h5["trace"].attrs.update({"trace_file": str(trace), "trace_offset": offset, "events_not_copied": dropped})

    SUMMARY_KEYS_SHOWN = 3  # spectra keys listed in a reply; the map has all of them as points of interest

    def summary(self, job: dict) -> dict:
        """A short reply: counts and a few keys. Every spectrum is a point of interest on the map
        (list_pois), and the run file and view_data have the rest."""
        plan = job["resolved_plan"]
        spectra = sorted(
            (r for r in job["step_results"].values() if isinstance(r, dict) and r.get("spectrum_key") and r["spectrum_key"] != "eds_map"),
            key=lambda r: r["index"],
        )
        grid = plan.get("grid") or {}
        shown = self.SUMMARY_KEYS_SHOWN
        pipeline = {k: v for k, v in self._pipeline(job).items() if not k.endswith("_job") and k != "drift_fit"}
        if "regions" in pipeline:
            regions = pipeline["regions"]
            pipeline["regions"] = {"count": len(regions), "first": [{k: r.get(k) for k in ("id", "x_m", "y_m", "score")} for r in regions[:shown]]}
        if "revisits" in pipeline:
            visits = pipeline["revisits"]
            pipeline["revisits"] = {
                "imaged": sum(1 for v in visits if "image" in v),
                "skipped": sum(1 for v in visits if "skipped" in v),
                "first": visits[:shown],
            }
        out = {
            "grid": {"rows": grid.get("rows"), "cols": grid.get("cols")},
            "box_m": plan.get("box_m"),
            "fov_m": plan.get("fov_m"),
            **({} if plan.get("revisit_only") or job.get("dry_run") else {"run_file": str(self._file(job))}),
            **pipeline,
        }
        tracked = live_drift.summary(job)
        if tracked:
            out["drift_tracking"] = tracked
        if job.get("dry_run"):
            out["acquisitions"] = DRY_RUN_SKIPPED  # no image, spectrum or run file is a stand-in for a real one
        if "eds_point" in plan.get("per_tile", []) and not job.get("dry_run"):
            out["spectra"] = {
                "taken": min(done_count(job), self._tile_count(job)),
                "first_keys": [r["spectrum_key"] for r in spectra[:shown]],
                "all": "the map's points of interest of kind eds_point (list_pois), or the run file's spectrum_keys",
            }
        return out

    def delete_device(self) -> None:
        session = getattr(self, "_analysis_session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        super().delete_device()


if __name__ == "__main__":
    Mapper.run_server()
