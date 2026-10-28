"""Mapper: acquire a grid of tiles over a stage box as one background job.

An agent (or a person, or a script) asks for a map in one call instead of
issuing a thousand moves and acquisitions. The plan (asyncroscopy/contracts/
map_plan.schema.json) fixes a stage box, field of view, overlap, detectors and
per-tile actions such as one EDS spectrum at each tile centre. The worker
visits tiles in serpentine order (to limit backlash), never leaves the box,
stops when the dose budget would be exceeded, and blanks the beam whenever it
stops early.

Every tile, its reported stage position and any spectra go into one HDF5 run
file, ``maps/map_<run>.h5``, which is registered with Tiled when the job
finishes. Analysis (stitching, scoring) reads that file; nothing in it depends
on the vendor or on whether the microscope was the digital twin.
"""

from __future__ import annotations

import json
import math
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


def tile_centres(box: list[float], fov: float, overlap: float, order: str) -> list[dict]:
    """Tile centres covering ``box`` with every tile inside it."""
    x_min, x_max, y_min, y_max = box
    step = fov * (1.0 - overlap)

    def axis(lo: float, hi: float) -> np.ndarray:
        span = hi - lo
        if span < fov - EDGE_TOLERANCE_M:
            raise ValueError(f"box side {span:.3e} m is smaller than the field of view {fov:.3e} m")
        n = max(1, int(math.ceil((span - fov) / step - 1e-9)) + 1)
        if n == 1:
            return np.array([(lo + hi) / 2.0])
        # Spread the tiles evenly so the last one ends exactly at the box edge.
        return np.linspace(lo + fov / 2.0, hi - fov / 2.0, n)

    xs, ys = axis(x_min, x_max), axis(y_min, y_max)
    tiles = []
    for row, y in enumerate(ys):
        cols = range(len(xs))
        if order == "serpentine" and row % 2 == 1:
            cols = reversed(range(len(xs)))
        for col in cols:
            tiles.append({"index": len(tiles), "row": row, "col": col, "x": float(xs[col]), "y": float(y)})
    return tiles


class Mapper(JobDevice):
    microscope_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="Microscope to drive.")
    scan_device_address = device_property(dtype=str, default_value="asyncroscopy/scan/default", doc="SCAN settings device.")
    eds_device_address = device_property(dtype=str, default_value="", doc="EDS settings device (needed for EDS per-tile actions).")
    data_device_address = device_property(dtype=str, default_value="", doc="DATA device; when set, the run file is registered with Tiled.")
    output_directory = device_property(dtype=str, default_value="outputs/maps", doc="Where run files are written when there is no DATA device; with one, they go in its save path (which Tiled serves) under maps/.")
    max_tiles = device_property(dtype=int, default_value=5000, doc="Refuse plans with more tiles than this.")
    estimate_move_s = device_property(dtype=float, default_value=2.0, doc="Assumed time per stage move, for estimates only.")
    default_beam_current_pa = device_property(dtype=float, default_value=50.0, doc="Used for dose estimates when the microscope does not report beam_current_pa.")

    def _proxy(self, address: str, timeout_ms: int = 60_000) -> tango.DeviceProxy:
        proxy = tango.DeviceProxy(address)
        proxy.set_timeout_millis(timeout_ms)
        return proxy

    def _beam_current_pa(self) -> float:
        try:
            parameters = json.loads(self._proxy(self.microscope_device_address).get_parameters())
            return float(parameters.get("beam_current_pa", self.default_beam_current_pa))
        except (tango.DevFailed, ValueError, TypeError):
            return float(self.default_beam_current_pa)

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def resolve_plan(self, plan: dict) -> dict:
        validate(plan, "map_plan")
        p = {**PLAN_DEFAULTS, **plan}
        x_min, x_max, y_min, y_max = (float(v) for v in p["box_m"])
        if not (x_min < x_max and y_min < y_max):
            raise ValueError("box_m must be [x_min, x_max, y_min, y_max] with min < max")
        if "eds_point" in p["per_tile"] and "eds_map" in p["per_tile"]:
            raise ValueError("choose eds_point or eds_map, not both")
        steps = tile_centres([x_min, x_max, y_min, y_max], float(p["fov_m"]), float(p["overlap"]), p["order"])
        if len(steps) > int(self.max_tiles):
            raise ValueError(f"plan needs {len(steps)} tiles; this Mapper allows at most {self.max_tiles}")

        imsize, dwell, fov = int(p["imsize"]), float(p["dwell_s"]), float(p["fov_m"])
        pixel_a = fov / imsize / 1e-10
        current = self._beam_current_pa()
        frame_dose = current * 1e-12 * dwell / ELECTRON_CHARGE / (pixel_a**2)
        step = fov * (1.0 - float(p["overlap"]))
        multiplicity = min(len(steps), int(math.ceil(fov / step - 1e-9)) ** 2)
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
        }
        if "eds_point" in p["per_tile"]:
            estimate["note"] = "EDS point spectra concentrate dose at each tile centre; that dose is not in max_dose_e_per_a2."
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

    def on_start(self, job: dict) -> None:
        plan = job["resolved_plan"]
        microscope = self._proxy(self.microscope_device_address)
        scan = self._proxy(self.scan_device_address)
        scan.imsize = int(plan["imsize"])
        scan.dwell_time = float(plan["dwell_s"])
        scan.scan_region = [0.0, 0.0, 1.0, 1.0]
        microscope.set_fov(float(plan["fov_m"]))
        current = list(microscope.get_stage())
        job.setdefault("pose", {"z": float(plan.get("z_m", current[2])), "alpha": float(current[3]), "beta": float(current[4])})
        if any(a in plan["per_tile"] for a in ("eds_point", "eds_map")) and self.eds_device_address:
            eds = self._proxy(self.eds_device_address)
            eds.exposure_time = float(plan["eds_exposure_s"])
        microscope.unblank_beam()
        path = self._run_path(job)
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
        microscope = self._proxy(self.microscope_device_address)
        pose = job["pose"]
        microscope.move_stage([tile["x"], tile["y"], pose["z"], pose["alpha"], pose["beta"]])
        reported = [float(v) for v in microscope.get_stage()]
        image_key = microscope.acquire_scanned_image([d.lower() for d in plan["detectors"]])
        images = read_acquisition(image_key, self._data_proxy())
        record = {"index": index, "row": tile["row"], "col": tile["col"], "commanded_xy_m": [tile["x"], tile["y"]], "stage_reported_m": reported, "key": image_key}

        spectrum = None
        if "eds_point" in plan["per_tile"]:
            microscope.place_beam([0.5, 0.5])
            spectrum_key = microscope.acquire_spectrum("eds")
            spectrum = read_acquisition(spectrum_key, self._data_proxy())
            record["spectrum_key"] = spectrum_key
        elif "eds_map" in plan["per_tile"]:
            spectrum = self._spectrum_map(microscope, int(plan["eds_map_points"]))
            record["spectrum_key"] = "eds_map"

        with h5py.File(self._run_path(job), "a") as h5:
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

    def _spectrum_map(self, microscope: tango.DeviceProxy, points: int) -> dict:
        coords = (np.arange(points) + 0.5) / points
        cube = None
        attrs: dict = {}
        for r, fy in enumerate(coords):
            for c, fx in enumerate(coords):
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
        return self._proxy(self.data_device_address) if self.data_device_address else None

    def on_stop(self, job: dict, reason: str) -> None:
        self._proxy(self.microscope_device_address).blank_beam()
        self._write_trace(job)

    def on_finish(self, job: dict) -> dict:
        self._write_trace(job)
        path = self._run_path(job)
        keys = {"run_file": str(path)}
        if self.data_device_address:
            keys["run"] = self._proxy(self.data_device_address).register_path(str(path))
        return keys

    def _write_trace(self, job: dict) -> None:
        events = [e for e in read_tool_log(self.trace_path) if e.get("run_id") == job["reservation_id"]]
        path = self._run_path(job)
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
            "run_file": str(self._run_path(job)),
            # Where each EDS point spectrum was taken, so a map can mark them (capped to keep replies small).
            "spectra": sorted(spectra, key=lambda s: s["index"])[: self.MAX_SUMMARY_SPECTRA],
        }


if __name__ == "__main__":
    Mapper.run_server()
