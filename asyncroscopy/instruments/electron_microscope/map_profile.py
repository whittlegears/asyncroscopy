"""The STEM map profile: the scanning-beam profile plus per-tile EDS.

STEM adds ``eds_point`` (one spectrum at each tile centre) and ``eds_map`` (a coarse spectrum image), which need
``place_beam`` and ``acquire_spectrum``. Its own Mapper setting beyond the beam's is ``eds_device_address``.

It is registered through the ``asyncroscopy.map_profiles`` entry point (pyproject.toml), so the Mapper finds it
without importing this module by name.
"""

from __future__ import annotations

import numpy as np
import tango

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
    settings = {**BEAM_SETTINGS, "eds_device_address": Setting(str, "", "EDS settings device (needed for EDS per-tile actions).")}

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

    def _tile_extras(self, job: dict, plan: dict, microscope) -> dict | None:
        if "eds_point" in plan["per_tile"]:
            microscope.place_beam([0.5, 0.5])
            return {"spectrum_key": microscope.acquire_spectrum("eds")}
        if "eds_map" in plan["per_tile"] and not job.get("dry_run"):
            spectrum = self._spectrum_map(job, microscope, int(plan["eds_map_points"]))
            return None if spectrum is None else {"spectrum_key": "eds_map", "spectrum": spectrum}
        return {}

    def _spectrum_map(self, job: dict, microscope: tango.DeviceProxy, points: int) -> dict | None:
        coords = (np.arange(points) + 0.5) / points
        cube = None
        attrs: dict = {}
        for r, fy in enumerate(coords):
            for c, fx in enumerate(coords):
                if self.ctx.aborting(job):  # points x points spectra can take long; stop with the beam blanked
                    return None
                microscope.place_beam([float(fx), float(fy)])
                data = self.ctx.read_acquisition(microscope.acquire_spectrum("eds"))
                array, attrs = next(iter(data.values()))
                if cube is None:
                    cube = np.zeros((points, points, len(array)), dtype=np.asarray(array).dtype)
                cube[r, c] = array
        attrs = dict(attrs)
        attrs["spectrum_image_axes"] = "rows=+y, columns=+x, channels"
        return {"spectrum_image": (cube, attrs)}
