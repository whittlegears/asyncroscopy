"""The STEM map profile: the scanning-beam profile plus per-tile EDS.

STEM adds ``eds_point`` (one spectrum at each tile centre) and ``eds_map`` (a coarse spectrum image), which need
``place_beam`` and ``acquire_spectrum``. Its own Mapper settings beyond the beam's are ``eds_device_address`` and
``max_native_spectrum_image_s``.

An ``eds_map`` tile asks the instrument for its native ``acquire_spectrum_image`` when it is expected to take no longer
than ``max_native_spectrum_image_s`` (it cannot be aborted part way, while the point-by-point loop can), and uses the loop
when it takes longer or the backend raises UnsupportedCommand. Either way the cube goes to the run file a row at a time.

It is registered through the ``asyncroscopy.map_profiles`` entry point (pyproject.toml), so the Mapper finds it
without importing this module by name.
"""

from __future__ import annotations

import numpy as np
import tango

from pathlib import Path

from asyncroscopy.core.names import REGISTRABLE_SUFFIXES

from asyncroscopy.instruments.scanning_beam_profile import BEAM_SETTINGS, ScanningBeamProfile
from asyncroscopy.jobs.job_device import done_count
from asyncroscopy.jobs.mapping.profile import SUMMARY_KEYS_SHOWN, Setting, register_profile
from asyncroscopy.routines.dose import EDS_DOSE_SPREAD_M, eds_dose

EDS_COMMANDS = ("place_beam", "acquire_spectrum")


@register_profile
class STEMProfile(ScanningBeamProfile):
    """A scanning transmission electron microscope: the beam profile plus per-tile EDS."""

    name = "stem"
    doc = "STEM: scanned images per tile (detectors, imsize, dwell_s), optional EDS per tile (per_tile eds_point or eds_map), electron-dose budget."
    job_kind = "map acquisition: a grid of images (optionally an EDS spectrum at each) over a stage box, stitched into a zoomable map when it finishes"
    plan_fields = ScanningBeamProfile.plan_fields | {"per_tile", "eds_exposure_s", "eds_map_points"}
    plan_defaults = {**ScanningBeamProfile.plan_defaults, "per_tile": [], "eds_exposure_s": 1.0, "eds_map_points": 8}
    instrument_types = ("TEM", "STEM")
    plan_example = {**ScanningBeamProfile.plan_example, "per_tile": []}  # per_tile ["eds_point"] adds a spectrum at each tile
    tile_fields = frozenset({"per_tile"})
    settings = {
        **BEAM_SETTINGS,
        "eds_device_address": Setting(str, "", "EDS settings device (needed for EDS per-tile actions)."),
        "max_native_spectrum_image_s": Setting(float, 120.0, "Longest an eds_map tile may hold the instrument in one acquire_spectrum_image call, which cannot be aborted part way; longer ones go point by point."),
    }

    def needed_commands(self, plan: dict) -> dict[str, list[str]]:
        out = super().needed_commands(plan)
        if plan.get("per_tile"):
            out.setdefault(self.ctx.settings.microscope_device_address, []).extend(EDS_COMMANDS)
        return out

    def summary(self, job: dict) -> dict:
        plan = job["resolved_plan"]
        if "eds_point" not in plan.get("per_tile", []) or job.get("dry_run"):
            return {}
        spectra = sorted(
            (r for r in job["step_results"].values() if isinstance(r, dict) and r.get("spectrum_key") and r["spectrum_key"] != "eds_map"),
            key=lambda r: r["index"],
        )
        tiles = plan["grid"]["rows"] * plan["grid"]["cols"]
        return {"spectra": {
            "taken": min(done_count(job), tiles),
            "first_keys": [r["spectrum_key"] for r in spectra[:SUMMARY_KEYS_SHOWN]],
            "all": "the map's points of interest of kind eds_point (list_pois), or the run file's spectrum_keys",
        }}

    def resolve(self, p: dict) -> None:
        super().resolve(p)
        if "eds_point" in p["per_tile"] and "eds_map" in p["per_tile"]:
            raise ValueError("choose eds_point or eds_map, not both")

    def _eds_seconds(self, p: dict) -> float:
        if "eds_point" in p["per_tile"]:
            return float(p["eds_exposure_s"])
        if "eds_map" in p["per_tile"]:
            return float(p["eds_exposure_s"]) * int(p["eds_map_points"]) ** 2
        return 0.0

    def _spectra_dose(self, p, current, fov, exposures):
        return eds_dose(current, float(p["eds_exposure_s"]), EDS_DOSE_SPREAD_M, p["per_tile"], int(p["eds_map_points"]), fov, exposures)

    def _setup_extras(self, job: dict, plan: dict) -> None:
        if any(a in plan["per_tile"] for a in ("eds_point", "eds_map")) and self.setting("eds_device_address"):
            eds = self.ctx.proxy(self.setting("eds_device_address"))
            eds.exposure_time = float(plan["eds_exposure_s"])

    def _tile_extras(self, job: dict, plan: dict, tile: dict, microscope) -> dict | None:
        if "eds_point" in plan["per_tile"]:
            microscope.place_beam([0.5, 0.5])
            return {"spectrum_key": microscope.acquire_spectrum("eds")}
        if "eds_map" in plan["per_tile"] and not job.get("dry_run"):
            return self._spectrum_map(job, plan, tile, microscope, int(plan["eds_map_points"]))
        return {}

    def _spectrum_map(self, job: dict, plan: dict, tile: dict, microscope: tango.DeviceProxy, points: int) -> dict | None:
        """A tile's spectrum image: the instrument's own when it has one and it is short enough, else point by point."""
        seconds = points * points * float(plan["eds_exposure_s"])
        if job.get("spectrum_image_native", True) and seconds <= float(self.setting("max_native_spectrum_image_s")):
            try:
                key = self.ctx.proxy(self.ctx.settings.microscope_device_address, timeout_s=1.5 * seconds + 60.0).acquire_spectrum_image(points)
            except tango.DevFailed as exc:
                if not exc.args or exc.args[0].reason != "UnsupportedCommand":
                    raise
                job["spectrum_image_native"] = False  # asked once a job: the backend's answer does not change
            else:
                return {"spectrum_key": "eds_map", "spectrum_stored": True, "spectrum_image_source": key}
        return self._spectrum_map_by_points(job, tile, microscope, points)

    def _spectrum_map_by_points(self, job: dict, tile: dict, microscope: tango.DeviceProxy, points: int) -> dict | None:
        coords = (np.arange(points) + 0.5) / points
        for r, fy in enumerate(coords):
            row = []
            attrs: dict = {}
            for fx in coords:
                if self.ctx.aborting(job):  # points x points spectra can take long; stop with the beam blanked
                    return None
                microscope.place_beam([float(fx), float(fy)])
                data = self.ctx.read_acquisition(microscope.acquire_spectrum("eds"))
                array, attrs = next(iter(data.values()))
                row.append(np.asarray(array))
            self.ctx.spectrum_row(job, tile, points, r, np.stack(row), {**attrs, "spectrum_image_axes": "rows=+y, columns=+x, channels"})
        return {"spectrum_key": "eds_map", "spectrum_stored": True}

    def _store_extras(self, job: dict, plan: dict, tile: dict, extras: dict) -> dict:
        """A native spectrum image goes into the run file a row at a time, or by name when it is a vendor file."""
        source = extras.pop("spectrum_image_source", None)
        if source is None:
            return extras
        if Path(source).suffix.lower() not in REGISTRABLE_SUFFIXES:
            self.ctx.spectrum_image_key(job, tile, source)
            return extras
        points = int(plan["eds_map_points"])
        for r, (spectra, attrs) in enumerate(self.ctx.read_rows(source, "spectrum_image")):
            self.ctx.spectrum_row(job, tile, points, r, spectra, attrs)
        return extras
