"""Mapper: acquire a grid of tiles over a stage box, then stitch and explore it, as one background job.

An agent (or a person, or a script) asks for a map in one call instead of
issuing a thousand moves and acquisitions. The plan (asyncroscopy/contracts/
map_plan.schema.json) fixes a stage box, field of view, overlap, detectors and
per-tile actions such as one EDS spectrum at each tile centre. The worker
visits tiles in serpentine order (to limit backlash) and never leaves the box.
`reserve` refuses a plan whose overlapping tiles would give any point more
than the dose budget; the plan cannot change after that, so there is no
separate dose check while it runs (EDS point spectra are reported, not
budgeted). The beam is blanked whenever the job ends: done, aborted or
failed.

The job is a pipeline whose phases are steps of the same job, so abort,
approval and progress cover all of them:

1. tiles: one step per tile. Steps are computed from the plan's grid when
   needed (``step_at``), so a plan of millions of tiles stores a few numbers.
2. stitch (``stitch``, default on, needs ``analysis_url``): the finished run is
   stitched on the analysis server (pyTEMlib ``assemble_map``) with search
   bounds from this Mapper's ``stage_sigma_m`` and ``max_drift_m_per_s``;
   every EDS point spectrum becomes a point of interest linked to its key.
3. score (``find_regions``: N > 0): the best N regions become points of interest.
4. revisit (``revisit``: {"fov_m": ...}): the stage goes to each of those
   regions (only inside the plan box) and takes a higher-magnification image,
   added as a point of interest linked to the image and to its region.

If the analysis server is down or stitching fails, the job still ends done
with every tile safe; ``summary.stitch_error`` says why and how to retry
(assemble_map on the run).

Every tile, its reported stage position and any spectra go into one HDF5 run
file, ``maps/map_<run>.h5`` (``.h5.partial`` while tiles are being taken, so
DATA does not register it half-written). The run file uses layout 2 (see
pyTEMlib.mcp_server.map_store): ``tiles`` is one chunked array [n, rows,
cols] with per-tile arrays ``written``, ``stage_reported_m``,
``commanded_xy_m``, ``acquired_s`` and ``row_col``; ``spectra`` [n, channels]
and ``spectrum_keys`` [n] hold EDS point spectra. Nothing in it depends on the
vendor or on whether the microscope was the digital twin.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import tango
from tango.server import device_property

from asyncroscopy.contracts import ENVELOPE_VERSION, validate
from asyncroscopy.data.data_reader import read_acquisition
from asyncroscopy.jobs import JobDevice
from asyncroscopy.mcp.tool_log import read_tool_log

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
ANALYSIS_POLL_S = 5.0  # how long one wait on the analysis server lasts; abort is checked between waits
RUN_LAYOUT = 2


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


def tile_at(xs, ys, order: str, index: int) -> dict:
    """Tile ``index`` of the grid, in acquisition order (serpentine reverses every other row)."""
    cols = len(xs)
    row, k = divmod(int(index), cols)
    col = cols - 1 - k if order == "serpentine" and row % 2 == 1 else k
    return {"index": int(index), "row": row, "col": col, "x": float(xs[col]), "y": float(ys[row])}


def tile_centres(box: list[float], fov: float, overlap: float, order: str, limit: int | None = None) -> list[dict]:
    """Every tile of a (small) grid as a list; large plans use ``grid_axes`` and ``tile_at`` instead."""
    xs, ys = grid_axes(box, fov, overlap, limit)
    return [tile_at(xs, ys, order, k) for k in range(len(xs) * len(ys))]


def max_exposures(xs, ys, fov: float) -> int:
    """Most tiles that cover any one point, from the actual tile spacing.

    The grid spreads tiles evenly, so the real overlap can exceed the
    requested one; the dose check must use the real spacing.
    """

    def per_axis(values) -> int:
        values = np.unique(np.round(np.asarray(values, dtype=np.float64), 15))
        if len(values) < 2:
            return 1
        spacing = float(np.min(np.diff(values)))
        return min(len(values), int(math.ceil(fov / spacing - 1e-9)))

    return per_axis(xs) * per_axis(ys)


class Mapper(JobDevice):
    JOB_KIND = "map acquisition: a grid of images (optionally an EDS spectrum at each) over a stage box, stitched into a zoomable map when it finishes"
    # A 600 nm square around the current stage position; per_tile ["eds_point"] adds a spectrum at each tile.
    PLAN_EXAMPLE = {"size_m": [6e-7, 6e-7], "fov_m": 1.6e-7, "per_tile": []}

    microscope_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="Microscope to drive.")
    scan_device_address = device_property(dtype=str, default_value="asyncroscopy/scan/default", doc="SCAN settings device.")
    eds_device_address = device_property(dtype=str, default_value="", doc="EDS settings device (needed for EDS per-tile actions).")
    data_device_address = device_property(dtype=str, default_value="", doc="DATA device; when set, the run file is registered with Tiled.")
    output_directory = device_property(dtype=str, default_value="outputs/maps", doc="Where run files are written when there is no DATA device; with one, they go in its save path (which Tiled serves) under maps/.")
    max_tiles = device_property(dtype=int, default_value=4_000_000, doc="Refuse plans with more tiles than this (a 2000 x 2000 grid by default).")
    estimate_move_s = device_property(dtype=float, default_value=2.0, doc="Assumed time per stage move, for estimates only.")
    default_beam_current_pa = device_property(dtype=float, default_value=50.0, doc="Used for dose estimates when the microscope does not report beam_current_pa.")
    analysis_url = device_property(dtype=str, default_value="", doc="pyTEMlib analysis MCP server (e.g. http://127.0.0.1:8003/mcp) that stitches and scores finished maps; blank: no stitching.")
    stage_sigma_m = device_property(dtype=float, default_value=2e-7, doc="1-sigma error of reported stage positions; bounds the stitching search and weighs the stage prior.")
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
    def resolve_plan(self, plan: dict) -> dict:
        if ("box_m" in plan) == ("size_m" in plan):
            raise ValueError("give the area as box_m [x_min, x_max, y_min, y_max] or as size_m [width, height], not both or neither")
        validate(plan, "map_plan")
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
            raise ValueError("revisit images the regions find_regions finds; give find_regions > 0")
        xs, ys = grid_axes([x_min, x_max, y_min, y_max], float(p["fov_m"]), float(p["overlap"]), limit=int(self.max_tiles))
        tiles = len(xs) * len(ys)

        imsize, dwell, fov = int(p["imsize"]), float(p["dwell_s"]), float(p["fov_m"])
        pixel_a = fov / imsize / 1e-10
        current, current_source = self._beam_current_pa()
        frame_dose = current * 1e-12 * dwell / ELECTRON_CHARGE / (pixel_a**2)
        max_dose = frame_dose * max_exposures(xs, ys, fov)
        budget = p.get("dose_budget_e_per_a2")
        if budget is not None and max_dose > float(budget):
            raise ValueError(f"plan would give up to {max_dose:.3g} e/A^2 where tiles overlap, above the budget of {budget} e/A^2")

        frame_s = imsize * imsize * dwell
        eds_s = 0.0
        if "eds_point" in p["per_tile"]:
            eds_s = float(p["eds_exposure_s"])
        elif "eds_map" in p["per_tile"]:
            eds_s = float(p["eds_exposure_s"]) * int(p["eds_map_points"]) ** 2
        stitch = bool(p["stitch"] and self.analysis_url)
        regions = int(p["find_regions"]) if stitch else 0
        revisits = regions if p.get("revisit") else 0
        estimate = {
            "steps": tiles + int(stitch) + int(regions > 0) + revisits,
            "tiles": tiles,
            "duration_s": round(tiles * (float(self.estimate_move_s) + frame_s + eds_s), 1),
            "max_dose_e_per_a2": round(max_dose, 3),
            "beam_current_pa": current,
            # The dose estimate is only as good as the current; say when it was assumed.
            "beam_current_source": current_source,
            "phases": ["tiles"] + (["stitch"] if stitch else []) + (["score"] if regions else []) + (["revisit"] if revisits else []),
        }
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
            "grid": {"rows": len(ys), "cols": len(xs), "xs": [float(v) for v in xs], "ys": [float(v) for v in ys]},
        }

    def step_at(self, job: dict, index: int) -> dict:
        plan = job["resolved_plan"]
        grid = plan["grid"]
        tiles = grid["rows"] * grid["cols"]
        if index < tiles:
            return {"phase": "tiles", **tile_at(grid["xs"], grid["ys"], plan["order"], index)}
        k = index - tiles
        if plan["stitch"] and k == 0:
            return {"phase": "stitch"}
        k -= int(plan["stitch"])
        if plan["find_regions"] and k == 0:
            return {"phase": "score"}
        return {"phase": "revisit", "region": k - int(bool(plan["find_regions"]))}

    def phase(self, job: dict) -> str | None:
        """Which phase a job is in, for job_status."""
        plan = job.get("resolved_plan", {})
        grid = plan.get("grid")
        if not grid:
            return None
        tiles = grid["rows"] * grid["cols"]
        from asyncroscopy.jobs.job_device import done_count

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
            h5.attrs["plan"] = json.dumps({k: v for k, v in plan.items() if k != "grid"})
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
        images = read_acquisition(image_key, self._data_proxy())
        if "eds_point" in plan["per_tile"]:
            spectrum = read_acquisition(spectrum_key, self._data_proxy())
        record = {"index": index, "row": tile["row"], "col": tile["col"], "commanded_xy_m": [tile["x"], tile["y"]], "stage_reported_m": reported, "key": image_key}
        if spectrum_key is not None:
            record["spectrum_key"] = spectrum_key
        self._write_tile(job, tile, reported, acquired, image_key, images, spectrum, spectrum_key)
        return record

    def _write_tile(self, job, tile, reported, acquired, image_key, images, spectrum, spectrum_key) -> None:
        """Put one tile into the run file's per-tile arrays (created at the first tile, sized for the whole grid)."""
        plan = job["resolved_plan"]
        n = plan["grid"]["rows"] * plan["grid"]["cols"]
        k = tile["index"]
        with h5py.File(self._file(job), "a") as h5:
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
            h5["stage_reported_m"][k] = (reported + [0.0] * 5)[:5]
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
        from fastmcp import Client

        async def call() -> dict:
            async with Client(self.analysis_url, timeout=60) as client:
                result = await client.call_tool(tool, arguments, raise_on_error=False)
                if result.is_error:
                    raise RuntimeError(" ".join(getattr(b, "text", "") for b in result.content) or f"{tool} failed")
                if result.structured_content is not None:
                    return result.structured_content
                return json.loads("".join(getattr(b, "text", "") for b in result.content))

        return asyncio.run(call())

    def _analysis_job(self, job: dict, started: dict) -> dict | None:
        """Wait for an analysis job, checking for an abort between waits; its results, or None if aborted."""
        if "error" in started:
            raise RuntimeError(started["error"].get("message", str(started["error"])))
        job_id = started["job_id"]
        while True:
            if self.aborting(job):
                try:
                    self._analysis("analysis_job_abort", {"job_id": job_id})
                except Exception:
                    pass
                return None
            status = self._analysis("analysis_job_status", {"job_id": job_id, "wait_s": ANALYSIS_POLL_S})
            if status.get("state") in {"done", "failed", "aborted"}:
                break
        if status["state"] != "done":
            raise RuntimeError(f"the analysis job ended {status['state']}: {(status.get('last_error') or {}).get('message', '')}")
        return self._analysis("analysis_job_results", {"job_id": job_id})

    def _pipeline(self, job: dict) -> dict:
        return job.setdefault("pipeline", {})

    def _stitch(self, job: dict) -> dict:
        pipeline = self._pipeline(job)
        run = job["result_keys"].get("run") or job["result_keys"].get("run_file")
        plan = {"run": run, "stage_sigma_m": float(self.stage_sigma_m), "max_drift_m_per_s": float(self.max_drift_m_per_s), "eds_pois": True}
        try:
            results = self._analysis_job(job, self._analysis("assemble_map", plan))
        except Exception as exc:  # analysis down or stitching failed: the tiles are safe, the job goes on
            pipeline["stitch_error"] = f"{type(exc).__name__}: {exc}. The tiles are in {run}; stitch them later with assemble_map(run={run!r})."
            return {"phase": "stitch", "ok": False}
        if results is None:
            return None  # aborted while stitching; a resume stitches again
        pipeline["map"] = results["result_keys"]["map"]
        pipeline["stitching"] = {k: v for k, v in results.get("summary", {}).items() if k in ("tiles", "levels", "canvas_shape", "matches_used", "median_residual_px", "p95_residual_px", "fallback_tiles", "drift", "eds_pois")}
        with self._lock:
            job["result_keys"]["map"] = pipeline["map"]
        return {"phase": "stitch", "ok": True, "map": pipeline["map"]}

    def _score(self, job: dict) -> dict:
        pipeline = self._pipeline(job)
        if "map" not in pipeline:
            return {"phase": "score", "skipped": "no map"}
        n = int(job["resolved_plan"]["find_regions"])
        try:
            results = self._analysis_job(job, self._analysis("score_map", {"map": pipeline["map"], "top_k": n}))
        except Exception as exc:
            pipeline["score_error"] = f"{type(exc).__name__}: {exc}. Score again with score_map(map={pipeline['map']!r}, top_k={n})."
            return {"phase": "score", "ok": False}
        if results is None:
            return None
        pipeline["regions"] = [
            {k: p.get(k) for k in ("id", "x_m", "y_m", "score", "reason", "scorer_version", "window_m")} for p in results.get("summary", {}).get("added_pois", [])
        ]
        return {"phase": "score", "ok": True, "regions": len(pipeline["regions"])}

    def _revisit(self, job: dict, k: int) -> dict:
        pipeline = self._pipeline(job)
        regions = pipeline.get("regions") or []
        if k >= len(regions):
            return {"phase": "revisit", "skipped": "no such region"}
        region = regions[k]
        plan = job["resolved_plan"]
        fov = float(plan["revisit"]["fov_m"])
        x_min, x_max, y_min, y_max = plan["box_m"]
        half = fov / 2.0
        x, y = float(region["x_m"]), float(region["y_m"])
        if not (x_min + half <= x <= x_max - half and y_min + half <= y <= y_max - half):
            note = f"region {region['id']} at ({x:.3e}, {y:.3e}) m is too close to the edge of the plan box for a {fov:.2e} m image; not visited"
            pipeline.setdefault("revisits", []).append({"region": region["id"], "skipped": note})
            return {"phase": "revisit", "skipped": note}
        microscope = self.proxy(self.microscope_device_address)
        pose = job["pose"]
        microscope.move_stage([x, y, pose["z"], pose["alpha"], pose["beta"]])
        microscope.set_fov(fov)
        microscope.unblank_beam()
        try:
            image_key = microscope.acquire_scanned_image([d.lower() for d in plan["detectors"]])
        finally:
            microscope.blank_beam()
            microscope.set_fov(float(plan["fov_m"]))
        visit = {"region": region["id"], "image": image_key, "fov_m": fov}
        try:
            poi = self._analysis("add_poi", {"map": pipeline["map"], "x_m": x, "y_m": y, "kind": "high_mag", "reason": f"higher-magnification image of region {region['id']}", "links": [image_key, f"poi:{region['id']}"]})
            visit["poi"] = poi.get("id")
        except Exception as exc:  # the image is saved either way
            visit["poi_error"] = f"{type(exc).__name__}: {exc}"
        pipeline.setdefault("revisits", []).append(visit)
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
        return {}

    def _release(self, request: str) -> str:
        """Release; an unfinished run gets its final name, and is sent for stitching, so its tiles stay reachable."""
        reply = super()._release(request)
        try:
            reservation_id = json.loads(request)["reservation_id"]
            job = self._jobs.get(reservation_id)
            if job is not None and not job.get("run_closed") and self._file(job).exists():
                self._close_run_file(job)
                run = job["result_keys"].get("run")
                if run and job["resolved_plan"].get("stitch"):
                    try:  # started, not waited for: release must return at once
                        started = self._analysis("assemble_map", {"run": run, "stage_sigma_m": float(self.stage_sigma_m), "max_drift_m_per_s": float(self.max_drift_m_per_s), "eds_pois": True})
                        job["result_keys"]["map_job"] = started.get("job_id")
                    except Exception as exc:
                        self._pipeline(job)["stitch_error"] = f"{type(exc).__name__}: {exc}. Stitch the tiles with assemble_map(run={run!r})."
                with self._lock:
                    self._save(job)
        except Exception as exc:  # the release itself succeeded; never turn it into an error
            self.warn_stream(f"Released, but the run file was not finalized: {exc}")
        return reply

    def _write_trace(self, job: dict) -> None:
        """The job's own events (reserve, start, stop, approvals, failed steps) into the run file.

        Each tile's position and time are already in the run's per-tile arrays, so the per-step
        events are left out: a map of a million tiles keeps a short trace.
        """
        events = [e for e in read_tool_log(self.trace_path) if e.get("run_id") == job["reservation_id"] and not (e.get("action", "").endswith(".step") and e.get("ok"))]
        path = self._file(job)
        if not path.exists():
            return
        with h5py.File(path, "a") as h5:
            if "trace" in h5:
                del h5["trace"]
            h5.create_dataset("trace", data=np.array([json.dumps(e) for e in events], dtype=h5py.string_dtype()))

    MAX_SUMMARY_SPECTRA = 500

    def summary(self, job: dict) -> dict:
        plan = job["resolved_plan"]
        spectra = [
            {"index": r["index"], "stage_xy_m": r["stage_reported_m"][:2], "key": r["spectrum_key"]}
            for r in job["step_results"].values()
            if isinstance(r, dict) and r.get("spectrum_key") and r["spectrum_key"] != "eds_map"
        ]
        grid = plan.get("grid") or {}
        return {
            "grid": {"rows": grid.get("rows"), "cols": grid.get("cols")},
            "box_m": plan.get("box_m"),
            "fov_m": plan.get("fov_m"),
            "run_file": str(self._file(job)),
            # Where EDS point spectra were taken (the latest, capped); the map has every one as a point of interest.
            "spectra": sorted(spectra, key=lambda s: s["index"])[: self.MAX_SUMMARY_SPECTRA],
            **self._pipeline(job),
        }


if __name__ == "__main__":
    Mapper.run_server()
