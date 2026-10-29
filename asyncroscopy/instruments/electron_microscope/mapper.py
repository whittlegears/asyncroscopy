"""Mapper: acquire a grid of tiles over a stage box, then stitch and explore it, as one background job.

An agent (or a person, or a script) asks for a map in one call instead of
issuing a thousand moves and acquisitions. The plan (asyncroscopy/contracts/
map_plan.schema.json) fixes a stage box, field of view, overlap, detectors and
per-tile actions such as one EDS spectrum at each tile centre. The worker
visits tiles in serpentine order (to limit backlash) and never leaves the box.
`reserve` refuses a plan whose dose anywhere (overlapping tiles plus a
revisit image) would exceed the budget; the plan cannot change after that,
so there is no separate dose check while it runs (EDS point spectra are
reported, not budgeted). The beam is blanked whenever the job ends: done,
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
digital twin.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import math
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
from asyncroscopy.jobs import JobDevice
from asyncroscopy.jobs.job_device import done_count

ELECTRON_CHARGE = 1.602176634e-19
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
EDGE_TOLERANCE_M = 1e-12
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


def grid_axes(box: list[float], fov: float, overlap: float, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tile-centre x and y coordinates covering ``box`` with every tile inside it; more than ``limit`` tiles is refused."""
    x_min, x_max, y_min, y_max = box
    step = fov * (1.0 - overlap)

    def axis(lo: float, hi: float) -> np.ndarray:
        span = hi - lo
        if span < fov - EDGE_TOLERANCE_M:
            raise ValueError(f"box side {span:.3e} m is smaller than the field of view {fov:.3e} m")
        n = max(1, int(math.ceil((span - fov) / step - 1e-9)) + 1)
        if limit is not None and n > limit:
            raise ValueError(f"plan needs more than {limit} tiles; this Mapper allows at most {limit}")
        if n == 1:
            return np.array([(lo + hi) / 2.0])
        # Spread the tiles evenly so the last one ends exactly at the box edge.
        return np.linspace(lo + fov / 2.0, hi - fov / 2.0, n)

    xs, ys = axis(x_min, x_max), axis(y_min, y_max)
    if limit is not None and len(xs) * len(ys) > limit:
        raise ValueError(f"plan needs {len(xs) * len(ys)} tiles; this Mapper allows at most {limit}")
    return xs, ys


# Names a caller (often a small model) uses for a field the plan calls something else.
FIELD_SYNONYMS = {
    "max_dose_e_per_a2": "dose_budget_e_per_a2", "max_dose": "dose_budget_e_per_a2", "dose_limit": "dose_budget_e_per_a2",
    "dose_budget": "dose_budget_e_per_a2", "dose": "dose_budget_e_per_a2", "fov": "fov_m", "field_of_view_m": "fov_m",
    "size": "size_m", "box": "box_m", "centre": "centre_m", "center_m": "centre_m", "center": "centre_m",
    "dwell": "dwell_s", "dwell_time": "dwell_s", "image_size": "imsize", "pixels": "imsize", "regions": "find_regions",
}


def check_plan_fields(plan: dict) -> None:
    """Refuse unknown plan fields by name, with the closest allowed field and the full list, before anything else."""
    allowed = sorted(load("map_plan")["properties"])
    unknown = [k for k in plan if k not in allowed]
    if not unknown:
        return
    hints = []
    for key in unknown:
        guess = FIELD_SYNONYMS.get(key) or next(iter(difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)), None)
        hints.append(f"{key!r} (did you mean {guess!r}?)" if guess else repr(key))
    raise ValueError(f"unknown plan field(s): {', '.join(hints)}. Nothing was dropped or changed: resend the plan with the right names. Allowed fields: {', '.join(allowed)}")


def compact_grid(xs, ys) -> dict:
    """A grid as origin, step and count per axis (the tiles are evenly spaced)."""
    return {
        "rows": len(ys), "cols": len(xs),
        "x0": float(xs[0]), "dx": float(xs[1] - xs[0]) if len(xs) > 1 else 0.0,
        "y0": float(ys[0]), "dy": float(ys[1] - ys[0]) if len(ys) > 1 else 0.0,
    }


def tile_at(grid: dict, order: str, index: int) -> dict:
    """Tile ``index`` of the grid, in acquisition order (serpentine reverses every other row)."""
    cols = grid["cols"]
    row, k = divmod(int(index), cols)
    col = cols - 1 - k if order == "serpentine" and row % 2 == 1 else k
    return {"index": int(index), "row": row, "col": col, "x": grid["x0"] + col * grid["dx"], "y": grid["y0"] + row * grid["dy"]}


def tile_centres(box: list[float], fov: float, overlap: float, order: str, limit: int | None = None) -> list[dict]:
    """Every tile of a (small) grid as a list; plans use ``compact_grid`` and ``tile_at`` instead."""
    grid = compact_grid(*grid_axes(box, fov, overlap, limit))
    return [tile_at(grid, order, k) for k in range(grid["rows"] * grid["cols"])]


def max_exposures(grid: dict, fov: float) -> int:
    """Most tiles that cover any one point, from the actual tile spacing.

    The grid spreads tiles evenly, so the real overlap can exceed the
    requested one; the dose check must use the real spacing.
    """

    def per_axis(n: int, step: float) -> int:
        return 1 if n < 2 or step <= 0 else min(n, int(math.ceil(fov / step - 1e-9)))

    return per_axis(grid["cols"], grid["dx"]) * per_axis(grid["rows"], grid["dy"])


def frame_dose(current_pa: float, dwell_s: float, fov_m: float, imsize: int) -> float:
    """Electrons per square angstrom one frame gives."""
    pixel_a = fov_m / imsize / 1e-10
    return current_pa * 1e-12 * dwell_s / ELECTRON_CHARGE / (pixel_a**2)


def drift_at(drift: dict | None, t_s: float) -> tuple[float, float]:
    """The stitch's fitted drift (x, y in metres) at ``t_s`` after the first tile; extrapolated from the last segment."""
    if not drift or len(drift.get("times_s", [])) < 2:
        return 0.0, 0.0
    times = np.asarray(drift["times_s"], dtype=np.float64)
    xy = np.asarray(drift["xy_m"], dtype=np.float64)
    if t_s <= times[-1]:
        return float(np.interp(t_s, times, xy[:, 0])), float(np.interp(t_s, times, xy[:, 1]))
    rate = (xy[-1] - xy[-2]) / max(times[-1] - times[-2], 1e-9)
    end = xy[-1] + rate * (t_s - times[-1])
    return float(end[0]), float(end[1])


def write_tile(path: Path, n: int, tile: dict, plan: dict, reported, acquired, image_key, images, spectrum, spectrum_key) -> None:
    """Put one tile into a run file's per-tile arrays (created at the first tile, sized for the whole grid).

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

    microscope_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="Microscope to drive.")
    scan_device_address = device_property(dtype=str, default_value="asyncroscopy/scan/default", doc="SCAN settings device.")
    eds_device_address = device_property(dtype=str, default_value="", doc="EDS settings device (needed for EDS per-tile actions).")
    data_device_address = device_property(dtype=str, default_value="", doc="DATA device; when set, the run file is registered with Tiled.")
    stage_device_address = device_property(dtype=str, default_value="asyncroscopy/stage/default", doc="STAGE device whose x/y limits every plan is checked against at reserve; blank: not checked (the microscope still refuses moves outside them).")
    output_directory = device_property(dtype=str, default_value="outputs/maps", doc="Where run files are written when there is no DATA device; with one, they go in its save path (which Tiled serves) under maps/.")
    max_tiles = device_property(dtype=int, default_value=4_000_000, doc="Refuse plans with more tiles than this (a 2000 x 2000 grid by default).")
    estimate_move_s = device_property(dtype=float, default_value=2.0, doc="Assumed time per stage move, for estimates only.")
    default_beam_current_pa = device_property(dtype=float, default_value=50.0, doc="Used for dose estimates when the microscope does not report beam_current_pa.")
    analysis_url = device_property(dtype=str, default_value="", doc="pyTEMlib analysis MCP server (e.g. http://127.0.0.1:8003/mcp) that stitches and scores finished maps; blank: no stitching.")
    stage_sigma_m = device_property(
        dtype=float,
        default_value=5e-8,
        doc="1-sigma error of the stage positions this microscope reports (readout, not where it was told to go); bounds the stitching search. "
        "50 nm suits a typical TEM goniometer read back by encoders; set it from your stage's specification.",
    )
    max_drift_m_per_s = device_property(dtype=float, default_value=5e-11, doc="Most the sample drifts relative to the stage (m/s); bounds the stitching search and the fitted drift. 0: no drift model.")

    def _beam_current_pa(self) -> tuple[float, str]:
        """(current, source): "microscope" when reported, "default_beam_current_pa" when assumed."""
        try:
            parameters = json.loads(self.proxy(self.microscope_device_address).get_parameters())
            if "beam_current_pa" in parameters:
                return float(parameters["beam_current_pa"]), "microscope"
        except (tango.DevFailed, ValueError, TypeError):
            pass
        return float(self.default_beam_current_pa), "default_beam_current_pa"

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def example_plan(self, request: str) -> dict | None:
        example = dict(self.PLAN_EXAMPLE)
        if "dose" in request.lower():  # a caller who mentioned dose sees where a dose limit goes
            example["dose_budget_e_per_a2"] = 50.0
        return example

    def resolve_plan(self, plan: dict) -> dict:
        check_plan_fields(plan)
        validate(plan, "map_plan")
        revisit_only = isinstance(plan.get("revisit"), dict) and ("points" in plan["revisit"] or "map" in plan["revisit"])
        if revisit_only:
            return self._resolve_revisit_only(plan)
        if ("box_m" in plan) == ("size_m" in plan):
            raise ValueError("give the area as box_m [x_min, x_max, y_min, y_max] or as size_m [width, height], not both or neither")
        if "fov_m" not in plan:
            raise ValueError("fov_m (the field of view of each tile, metres) is required")
        p = {**PLAN_DEFAULTS, **plan}
        if "size_m" in p:
            # The device does the arithmetic: callers (often small models) get centre +/- half wrong.
            width, height = (float(v) for v in p["size_m"])
            cx, cy = (float(v) for v in p.get("centre_m") or list(self.proxy(self.microscope_device_address).get_stage())[:2])
            p["box_m"] = [cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2]
        x_min, x_max, y_min, y_max = (float(v) for v in p["box_m"])
        if not (x_min < x_max and y_min < y_max):
            raise ValueError("box_m must be [x_min, x_max, y_min, y_max] with min < max")
        if "eds_point" in p["per_tile"] and "eds_map" in p["per_tile"]:
            raise ValueError("choose eds_point or eds_map, not both")
        if (p["find_regions"] or p.get("revisit")) and not (p["stitch"] and self.analysis_url):
            raise ValueError("find_regions and revisit need the map stitched: keep stitch on, on a Mapper with an analysis_url")
        if p.get("revisit") and not p["find_regions"]:
            raise ValueError("revisit images the regions find_regions finds (give find_regions > 0), or give revisit points [[x, y], ...] or an existing map (revisit map, top_k)")
        imsize, dwell, fov = int(p["imsize"]), float(p["dwell_s"]), float(p["fov_m"])
        revisit = None
        if p.get("revisit"):
            revisit = {"fov_m": float(p["revisit"]["fov_m"]), "imsize": int(p["revisit"].get("imsize", imsize)), "dwell_s": float(p["revisit"].get("dwell_s", dwell))}
            if not fov * REVISIT_FOV_MIN_FRACTION <= revisit["fov_m"] <= fov:
                raise ValueError(f"revisit fov_m must be between {fov * REVISIT_FOV_MIN_FRACTION:.3g} m (a twentieth of the tile) and the tile's {fov:.3g} m")
            p["revisit"] = revisit
        grid = compact_grid(*grid_axes([x_min, x_max, y_min, y_max], fov, float(p["overlap"]), limit=int(self.max_tiles)))
        tiles = grid["rows"] * grid["cols"]

        current, current_source = self._beam_current_pa()
        tile_dose = frame_dose(current, dwell, fov, imsize) * max_exposures(grid, fov)
        stitch = bool(p["stitch"] and self.analysis_url)
        regions = int(p["find_regions"]) if stitch else 0
        revisits = regions if revisit else 0
        # A revisit adds its frame on top of whatever the tiles gave that spot; regions lie a scoring window apart,
        # so one spot gets at most one revisit.
        revisit_dose = frame_dose(current, revisit["dwell_s"], revisit["fov_m"], revisit["imsize"]) if revisit else 0.0
        max_dose = tile_dose + revisit_dose
        budget = p.get("dose_budget_e_per_a2")
        if budget is not None and max_dose > float(budget):
            parts = f"{tile_dose:.3g} from overlapping tiles" + (f" plus {revisit_dose:.3g} from a revisit image" if revisit else "")
            raise ValueError(f"plan would give up to {max_dose:.3g} e/A^2 ({parts}), above the budget of {budget} e/A^2; widen revisit fov_m or lower dwell_s")

        frame_s = imsize * imsize * dwell
        eds_s = 0.0
        if "eds_point" in p["per_tile"]:
            eds_s = float(p["eds_exposure_s"])
        elif "eds_map" in p["per_tile"]:
            eds_s = float(p["eds_exposure_s"]) * int(p["eds_map_points"]) ** 2
        revisit_s = revisits * (float(self.estimate_move_s) + (revisit["imsize"] ** 2 * revisit["dwell_s"] if revisit else 0.0))
        estimate = {
            "steps": tiles + int(stitch) + int(regions > 0) + revisits,
            "tiles": tiles,
            "duration_s": round(tiles * (float(self.estimate_move_s) + frame_s + eds_s) + revisit_s, 1),
            "max_dose_e_per_a2": round(max_dose, 3),
            "beam_current_pa": current,
            # The dose estimate is only as good as the current; say when it was assumed.
            "beam_current_source": current_source,
            "phases": ["tiles"] + (["stitch"] if stitch else []) + (["score"] if regions else []) + (["revisit"] if revisits else []),
        }
        if revisit:
            estimate["revisit_dose_e_per_a2"] = round(revisit_dose, 3)
        if not stitch:
            estimate["stitch"] = "off" if not p["stitch"] else "this Mapper has no analysis_url; stitch the run with assemble_map"
        if "eds_point" in p["per_tile"]:
            estimate["note"] = "EDS point spectra concentrate dose at each tile centre; that dose is not in max_dose_e_per_a2."
        elif "eds_map" in p["per_tile"]:
            estimate["note"] = "The EDS spectrum image parks the beam at eds_map_points x eds_map_points points per tile; that dose is not in max_dose_e_per_a2."
        return {
            **p,
            "box_m": [x_min, x_max, y_min, y_max],
            "stitch": stitch,
            "find_regions": regions,
            "step_count": tiles + int(stitch) + int(regions > 0) + revisits,
            "estimate": estimate,
            "grid": grid,
        }

    def _resolve_revisit_only(self, plan: dict) -> dict:
        """A plan that only images given points, or an existing map's best regions: no tiles, no stitch."""
        p = {**PLAN_DEFAULTS, **plan}
        revisit = dict(p["revisit"])
        if ("points" in revisit) == ("map" in revisit):
            raise ValueError("give revisit points [[x, y], ...] or an existing map (revisit map, top_k), not both")
        if p.get("find_regions") or p.get("per_tile"):
            raise ValueError("a revisit-only plan takes no tiles: leave out find_regions and per_tile, or give an area to map")
        fov = float(revisit["fov_m"])
        imsize = int(revisit.get("imsize", p["imsize"]))
        dwell = float(revisit.get("dwell_s", p["dwell_s"]))
        revisit.update({"fov_m": fov, "imsize": imsize, "dwell_s": dwell})
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
        current, current_source = self._beam_current_pa()
        dose = frame_dose(current, dwell, fov, imsize)
        budget = p.get("dose_budget_e_per_a2")
        if budget is not None and dose > float(budget):
            raise ValueError(f"each revisit image would give {dose:.3g} e/A^2, above the budget of {budget} e/A^2; widen revisit fov_m or lower dwell_s")
        for r in regions:
            if not (x_min + half <= r["x_m"] <= x_max - half and y_min + half <= r["y_m"] <= y_max - half):
                raise ValueError(f"revisit point ({r['x_m']:.3e}, {r['y_m']:.3e}) m is outside the approved area {[x_min, x_max, y_min, y_max]} for a {fov:.2e} m image")
        steps = count + (1 if "map" in revisit else 0)
        estimate = {
            "steps": steps, "tiles": 0, "revisits": count,
            "duration_s": round(count * (float(self.estimate_move_s) + imsize * imsize * dwell), 1),
            "max_dose_e_per_a2": round(dose, 3), "revisit_dose_e_per_a2": round(dose, 3),
            "beam_current_pa": current, "beam_current_source": current_source,
            "phases": (["select"] if "map" in revisit else []) + ["revisit"],
        }
        return {
            **p, "fov_m": fov, "revisit": revisit, "box_m": [x_min, x_max, y_min, y_max], "stitch": False, "find_regions": 0,
            "step_count": steps, "estimate": estimate, "grid": {"rows": 0, "cols": 0, "x0": 0.0, "dx": 0.0, "y0": 0.0, "dy": 0.0},
            "regions": regions, "revisit_only": True,
        }

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
        checked at reserve against the STAGE device's limits, so a plan off the stage is refused before anything moves."""
        if not self.stage_device_address:
            return []
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
        stage = self.stage_device_address
        return [{"step": s, "device": stage, "attribute": "x", "value": float(v)} for s, v in xs] + [
            {"step": s, "device": stage, "attribute": "y", "value": float(v)} for s, v in ys
        ]

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
        microscope = self.proxy(self.microscope_device_address)
        scan = self.proxy(self.scan_device_address)
        scan.imsize = int(plan["imsize"])
        scan.dwell_time = float(plan["dwell_s"])
        scan.scan_region = [0.0, 0.0, 1.0, 1.0]
        microscope.set_fov(float(plan["fov_m"]))
        current = list(microscope.get_stage())
        job.setdefault("pose", {"z": float(plan.get("z_m", current[2])), "alpha": float(current[3]), "beta": float(current[4])})
        if any(a in plan["per_tile"] for a in ("eds_point", "eds_map")) and self.eds_device_address:
            eds = self.proxy(self.eds_device_address)
            eds.exposure_time = float(plan["eds_exposure_s"])
        # The beam stays blanked between tiles: unblanked only while acquiring (see run_step).
        microscope.blank_beam()
        if job["resolved_plan"].get("revisit_only"):
            job["run_closed"] = True  # no tiles, so no run file
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
            h5.attrs["created"] = time.time()
        job["result_keys"]["run_file"] = str(path)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def run_step(self, job: dict, index: int) -> dict:
        step = self.step_at(job, index)
        if step["phase"] == "tiles":
            return self._tile(job, step)
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
        x_min, x_max, y_min, y_max = plan["box_m"]
        half = float(plan["fov_m"]) / 2.0
        if not (x_min + half - EDGE_TOLERANCE_M <= tile["x"] <= x_max - half + EDGE_TOLERANCE_M and y_min + half - EDGE_TOLERANCE_M <= tile["y"] <= y_max - half + EDGE_TOLERANCE_M):
            raise RuntimeError(f"tile {index} at ({tile['x']:.3e}, {tile['y']:.3e}) lies outside the plan box; refusing to move")
        microscope = self.proxy(self.microscope_device_address)
        pose = job["pose"]
        microscope.move_stage([tile["x"], tile["y"], pose["z"], pose["alpha"], pose["beta"]])
        if not self._wait_for_stage(job, microscope):
            return None  # aborted during a long move; a resume moves again
        reported = [float(v) for v in microscope.get_stage()]
        # Unblank only for this tile's acquisitions: a beam left on during the stage move, file
        # writes and Tiled reads would park on one spot and put dose there that no estimate counts.
        microscope.unblank_beam()
        try:
            image_key = microscope.acquire_scanned_image([d.lower() for d in plan["detectors"]])
            spectrum, spectrum_key = None, None
            if "eds_point" in plan["per_tile"]:
                microscope.place_beam([0.5, 0.5])
                spectrum_key = microscope.acquire_spectrum("eds")
            elif "eds_map" in plan["per_tile"]:
                spectrum = self._spectrum_map(job, microscope, int(plan["eds_map_points"]))
                spectrum_key = "eds_map"
        finally:
            microscope.blank_beam()
        if spectrum_key == "eds_map" and spectrum is None:
            return None  # aborted part way through the spectrum image; a resume redoes this tile
        acquired = time.time()
        job.setdefault("first_tile_s", acquired)
        images = read_acquisition(image_key, self._data_proxy())
        if "eds_point" in plan["per_tile"]:
            spectrum = read_acquisition(spectrum_key, self._data_proxy())
        record = {"index": index, "row": tile["row"], "col": tile["col"], "commanded_xy_m": [tile["x"], tile["y"]], "stage_reported_m": reported, "key": image_key}
        if spectrum_key is not None:
            record["spectrum_key"] = spectrum_key
        write_tile(self._file(job), self._tile_count(job), tile, plan, reported, acquired, image_key, images, spectrum, spectrum_key)
        return record

    def _wait_for_stage(self, job: dict, microscope: tango.DeviceProxy) -> bool:
        """Wait for a long move to end (a microscope that returns before arriving says so in stage_moving); False if aborted."""
        while True:
            try:
                moving = bool(microscope.read_attribute("stage_moving").value)
            except tango.DevFailed:  # a microscope whose move_stage returns only on arrival
                return True
            if not moving:
                return True
            if self.aborting(job):
                return False
            time.sleep(STAGE_POLL_S)

    def _spectrum_map(self, job: dict, microscope: tango.DeviceProxy, points: int) -> dict | None:
        coords = (np.arange(points) + 0.5) / points
        cube = None
        attrs: dict = {}
        for r, fy in enumerate(coords):
            for c, fx in enumerate(coords):
                if self.aborting(job):  # points x points spectra can take long; stop with the beam blanked
                    return None
                microscope.place_beam([float(fx), float(fy)])
                data = read_acquisition(microscope.acquire_spectrum("eds"), self._data_proxy())
                array, attrs = next(iter(data.values()))
                if cube is None:
                    cube = np.zeros((points, points, len(array)), dtype=np.asarray(array).dtype)
                cube[r, c] = array
        attrs = dict(attrs)
        attrs["spectrum_image_axes"] = "rows=+y, columns=+x, channels"
        return {"spectrum_image": (cube, attrs)}

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
            # The map places content at stage reading + drift(t) (pyTEMlib's fit), so to put the region under
            # the beam now the stage goes to its map position minus the drift, extrapolated to now.
            dx, dy = drift_at(pipeline.get("drift_fit"), time.time() - float(job.get("first_tile_s", time.time())))
            x, y = float(region["x_m"]) - dx, float(region["y_m"]) - dy
            x_min, x_max, y_min, y_max = plan["box_m"]
            half = fov / 2.0
            if not (x_min + half <= x <= x_max - half and y_min + half <= y <= y_max - half):
                note = f"region {region['id']} at ({x:.3e}, {y:.3e}) m (drift-corrected) is too close to the edge of the plan box for a {fov:.2e} m image; not visited"
                visits.append({"region": region["id"], "skipped": note})
                return {"phase": "revisit", "skipped": note}
            microscope = self.proxy(self.microscope_device_address)
            pose = job["pose"]
            microscope.move_stage([x, y, pose["z"], pose["alpha"], pose["beta"]])
            if not self._wait_for_stage(job, microscope):
                return None
            microscope.set_fov(fov)
            scan = self.proxy(self.scan_device_address)
            scan.imsize = int(revisit["imsize"])
            scan.dwell_time = float(revisit["dwell_s"])
            microscope.unblank_beam()
            try:
                image_key = microscope.acquire_scanned_image([d.lower() for d in plan["detectors"]])
            finally:
                microscope.blank_beam()
                microscope.set_fov(float(plan["fov_m"]))
                scan.imsize = int(plan["imsize"])
                scan.dwell_time = float(plan["dwell_s"])
            visit = {"region": region["id"], "image": image_key, "fov_m": fov, "stage_xy_m": [x, y]}
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
        self.proxy(self.microscope_device_address).blank_beam()
        if not job.get("run_closed"):
            self._write_trace(job)

    def on_finish(self, job: dict) -> dict:
        # A finished map must not leave the beam parked on the last tile.
        self.proxy(self.microscope_device_address).blank_beam()
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
            **({} if plan.get("revisit_only") else {"run_file": str(self._file(job))}),
            **pipeline,
        }
        if "eds_point" in plan.get("per_tile", []):
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
