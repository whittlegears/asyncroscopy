"""Mapper: acquire a grid of tiles over a stage box as one background job.

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

Every tile, its reported stage position and any spectra go into one HDF5 run
file, ``maps/map_<run>.h5`` (``.h5.partial`` while the job runs, so DATA does not
register it half-written), which is registered with Tiled when the job
finishes. Analysis (stitching, scoring) reads that file; nothing in it depends
on the vendor or on whether the microscope was the digital twin.
"""

from __future__ import annotations

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
}
EDGE_TOLERANCE_M = 1e-12
RENAME_RETRY_DELAYS_S = (0.0, 0.2, 0.5, 1.0, 2.0)  # a finished run file may briefly be open elsewhere (Windows)


def tile_centres(box: list[float], fov: float, overlap: float, order: str, limit: int | None = None) -> list[dict]:
    """Tile centres covering ``box`` with every tile inside it; more than ``limit`` tiles is refused before any is built."""
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
    tiles = []
    for row, y in enumerate(ys):
        cols = range(len(xs))
        if order == "serpentine" and row % 2 == 1:
            cols = reversed(range(len(xs)))
        for col in cols:
            tiles.append({"index": len(tiles), "row": row, "col": col, "x": float(xs[col]), "y": float(y)})
    return tiles


def max_exposures(tiles: list[dict], fov: float) -> int:
    """Most tiles that cover any one point, from the actual tile centres.

    tile_centres spreads tiles evenly, so the real overlap can exceed the
    requested one; the dose check must use the real spacing.
    """

    def per_axis(values: list[float]) -> int:
        unique = sorted(set(round(v, 15) for v in values))
        if len(unique) < 2:
            return 1
        spacing = min(b - a for a, b in zip(unique, unique[1:]))
        return min(len(unique), int(math.ceil(fov / spacing - 1e-9)))

    return per_axis([t["x"] for t in tiles]) * per_axis([t["y"] for t in tiles])


class Mapper(JobDevice):
    JOB_KIND = "map acquisition: a grid of images (optionally an EDS spectrum at each) over a stage box. Not stitching"
    # A 600 nm square around the current stage position; per_tile ["eds_point"] adds a spectrum at each tile.
    PLAN_EXAMPLE = {"size_m": [6e-7, 6e-7], "fov_m": 1.6e-7, "per_tile": []}

    microscope_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="Microscope to drive.")
    scan_device_address = device_property(dtype=str, default_value="asyncroscopy/scan/default", doc="SCAN settings device.")
    eds_device_address = device_property(dtype=str, default_value="", doc="EDS settings device (needed for EDS per-tile actions).")
    data_device_address = device_property(dtype=str, default_value="", doc="DATA device; when set, the run file is registered with Tiled.")
    output_directory = device_property(dtype=str, default_value="outputs/maps", doc="Where run files are written when there is no DATA device; with one, they go in its save path (which Tiled serves) under maps/.")
    max_tiles = device_property(dtype=int, default_value=5000, doc="Refuse plans with more tiles than this.")
    estimate_move_s = device_property(dtype=float, default_value=2.0, doc="Assumed time per stage move, for estimates only.")
    default_beam_current_pa = device_property(dtype=float, default_value=50.0, doc="Used for dose estimates when the microscope does not report beam_current_pa.")

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
        steps = tile_centres([x_min, x_max, y_min, y_max], float(p["fov_m"]), float(p["overlap"]), p["order"], limit=int(self.max_tiles))

        imsize, dwell, fov = int(p["imsize"]), float(p["dwell_s"]), float(p["fov_m"])
        pixel_a = fov / imsize / 1e-10
        current, current_source = self._beam_current_pa()
        frame_dose = current * 1e-12 * dwell / ELECTRON_CHARGE / (pixel_a**2)
        multiplicity = max_exposures(steps, fov)
        max_dose = frame_dose * multiplicity
        budget = p.get("dose_budget_e_per_a2")
        if budget is not None and max_dose > float(budget):
            raise ValueError(f"plan would give up to {max_dose:.3g} e/A^2 where tiles overlap, above the budget of {budget} e/A^2")

        frame_s = imsize * imsize * dwell
        eds_s = 0.0
        if "eds_point" in p["per_tile"]:
            eds_s = float(p["eds_exposure_s"])
        elif "eds_map" in p["per_tile"]:
            eds_s = float(p["eds_exposure_s"]) * int(p["eds_map_points"]) ** 2
        estimate = {
            "steps": len(steps),
            "duration_s": round(len(steps) * (float(self.estimate_move_s) + frame_s + eds_s), 1),
            "max_dose_e_per_a2": round(max_dose, 3),
            "beam_current_pa": current,
            # The dose estimate is only as good as the current; say when it was assumed.
            "beam_current_source": current_source,
        }
        if "eds_point" in p["per_tile"]:
            estimate["note"] = "EDS point spectra concentrate dose at each tile centre; that dose is not in max_dose_e_per_a2."
        elif "eds_map" in p["per_tile"]:
            estimate["note"] = "The EDS spectrum image parks the beam at eds_map_points x eds_map_points points per tile; that dose is not in max_dose_e_per_a2."
        return {**p, "box_m": [x_min, x_max, y_min, y_max], "steps": steps, "estimate": estimate, "grid": {"rows": 1 + max(s["row"] for s in steps), "cols": 1 + max(s["col"] for s in steps)}}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _run_path(self, job: dict) -> Path:
        if "run_path" not in job:
            # Tiled only serves files under the DATA save path, so the run file must live there.
            base = Path(self._data_proxy().save_path) / "maps" if self.data_device_address else Path(self.output_directory)
            job["run_path"] = str(base / f"map_{job['reservation_id']}.h5")
        return Path(job["run_path"])

    def _file(self, job: dict) -> Path:
        """Where the run file is now: ``<run>.h5.partial`` until the job finishes, so DATA's reconcile leaves it alone."""
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
            h5.attrs["run_id"] = job["reservation_id"]
            h5.attrs["plan_id"] = job["plan_id"]
            h5.attrs["plan"] = json.dumps(plan)
            h5.attrs["grid"] = json.dumps(plan["grid"])
            h5.attrs["axes"] = "images: rows=+y, columns=+x; positions in stage metres"
            h5.attrs["microscope"] = self.microscope_device_address
            h5.attrs["created"] = time.time()
            h5.require_group("tiles")
        job["result_keys"]["run_file"] = str(path)

    def run_step(self, job: dict, index: int) -> dict:
        plan = job["resolved_plan"]
        tile = job["steps"][index]
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
        images = read_acquisition(image_key, self._data_proxy())
        if "eds_point" in plan["per_tile"]:
            spectrum = read_acquisition(spectrum_key, self._data_proxy())
        record = {"index": index, "row": tile["row"], "col": tile["col"], "commanded_xy_m": [tile["x"], tile["y"]], "stage_reported_m": reported, "key": image_key}

        if spectrum_key is not None:
            record["spectrum_key"] = spectrum_key

        with h5py.File(self._file(job), "a") as h5:
            for n, (name, (array, attrs)) in enumerate(sorted(images.items())):
                detector = str(attrs.get("detector", name.rsplit("/", 1)[-1])).upper()
                group = h5.require_group("tiles") if n == 0 else h5.require_group(f"channels/{detector}")
                dset_name = f"tile_{index:05d}"
                if dset_name in group:
                    del group[dset_name]
                dset = group.create_dataset(dset_name, data=array)
                dset.attrs.update(
                    {
                        "index": index,
                        "row": tile["row"],
                        "col": tile["col"],
                        "detector": detector,
                        "commanded_xy_m": [tile["x"], tile["y"]],
                        "stage_reported_m": reported,
                        "fov_m": float(plan["fov_m"]),
                        "pixel_size_m": float(plan["fov_m"]) / int(plan["imsize"]),
                        "source_key": image_key,
                        "acquired": time.time(),
                    }
                )
            if spectrum is not None:
                group = h5.require_group("spectra")
                name = f"spectrum_{index:05d}"
                if name in group:
                    del group[name]
                array, attrs = next(iter(spectrum.values()))
                dset = group.create_dataset(name, data=array)
                dset.attrs.update({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in attrs.items() if v is not None})
                dset.attrs.update({"index": index, "stage_xy_m": reported[:2], "kind": "eds_map" if "eds_map" in plan["per_tile"] else "eds_point"})
        return record

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

    def _data_proxy(self):
        return self.proxy(self.data_device_address) if self.data_device_address else None

    def on_stop(self, job: dict, reason: str) -> None:
        self.proxy(self.microscope_device_address).blank_beam()
        self._write_trace(job)

    def on_finish(self, job: dict) -> dict:
        # A finished map must not leave the beam parked on the last tile.
        self.proxy(self.microscope_device_address).blank_beam()
        self._write_trace(job)
        path, note = self._finalize_run_file(job)
        key, register_note = self._register_run(path)
        keys = {"run_file": str(path)}
        if key is not None:
            keys["run"] = key
        notes = [n for n in (note, register_note) if n]
        if notes:
            keys["run_file_note"] = "; ".join(notes)
        return keys

    def _release(self, request: str) -> str:
        """Release, then give an unfinished (aborted or failed) run its final name, so its tiles stay reachable."""
        reply = super()._release(request)
        try:
            reservation_id = json.loads(request)["reservation_id"]
            job = self._jobs.get(reservation_id)
            if job is not None and job.get("result_keys", {}).get("run_file") and self._file(job).name.endswith(".partial") and self._file(job).exists():
                path, note = self._finalize_run_file(job)
                key, register_note = self._register_run(path)
                with self._lock:
                    job["result_keys"]["run_file"] = str(path)
                    if key is not None:
                        job["result_keys"]["run"] = key
                    notes = [n for n in (note, register_note) if n]
                    if notes:
                        job["result_keys"]["run_file_note"] = "; ".join(notes)
                    self._save(job)
        except Exception as exc:  # the release itself succeeded; never turn it into an error
            self.warn_stream(f"Released, but the run file was not finalized: {exc}")
        return reply

    def _write_trace(self, job: dict) -> None:
        events = [e for e in read_tool_log(self.trace_path) if e.get("run_id") == job["reservation_id"]]
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
            if r.get("spectrum_key") and r["spectrum_key"] != "eds_map"
        ]
        return {
            "grid": plan.get("grid"),
            "box_m": plan.get("box_m"),
            "fov_m": plan.get("fov_m"),
            "run_file": str(self._file(job)),
            # Where each EDS point spectrum was taken, so a map can mark them (capped to keep replies small).
            "spectra": sorted(spectra, key=lambda s: s["index"])[: self.MAX_SUMMARY_SPECTRA],
        }


if __name__ == "__main__":
    Mapper.run_server()
