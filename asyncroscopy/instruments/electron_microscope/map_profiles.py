"""Map profiles for scanned electron-beam instruments: STEM (the default) and SEM.

Both drive the instrument through the same command subset, which every
microscope device here offers (the twin, AutoScript, JEOL): ``move_stage``,
``get_stage``, ``set_fov``, ``acquire_scanned_image``, ``blank_beam`` and
``unblank_beam`` on the microscope, and the SCAN device's ``imsize``,
``dwell_time`` and ``scan_region``. The beam is blanked between tiles and
whenever the job stops, and unblanked only while a tile or revisit image is
acquired. Their budget is electron dose: ``dose_e_per_a2`` (the plan's
``dose_budget_e_per_a2``), counted where overlapping tiles stack and refused on
a real instrument when the beam current is only assumed.

STEM adds per-tile EDS (``eds_point``: one spectrum at each tile centre;
``eds_map``: a coarse spectrum image), which needs ``place_beam`` and
``acquire_spectrum``. SEM is the same beam and dose model without them; any
device offering the subset above maps as an SEM. Live drift tracking (plan
``drift.track``) also needs ``set_image_shift`` and ``get_image_shift``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tango

from asyncroscopy.data.data_reader import read_acquisition
from asyncroscopy.jobs.map_profile import Cost, InstrumentProfile, register_profile
from asyncroscopy.routines.dose import eds_dose, frame_dose
from asyncroscopy.routines.grid import max_exposures

BEAM_COMMANDS = ("move_stage", "get_stage", "set_fov", "acquire_scanned_image", "blank_beam", "unblank_beam")
EDS_COMMANDS = ("place_beam", "acquire_spectrum")
IMAGE_SHIFT_LIMIT_M = 1e-6
STEM_PLAN_DEFAULTS = {
    "imsize": 512,
    "dwell_s": 1e-6,
    "detectors": ["haadf"],
    "per_tile": [],
    "eds_exposure_s": 1.0,
    "eds_map_points": 8,
}


class ElectronBeamProfile(InstrumentProfile):
    """A scanned electron beam: stage moves, a blanker, a field of view, an image per tile, electron dose."""

    plan_fields = frozenset({"imsize", "dwell_s", "detectors", "dose_budget_e_per_a2", "z_m"})
    plan_defaults = {k: STEM_PLAN_DEFAULTS[k] for k in ("imsize", "dwell_s", "detectors")}
    needs = {"microscope_device_address": BEAM_COMMANDS}
    # Live drift tracking moves the beam with set_image_shift (the beam shift in STEM on AutoScript).
    # 1 um is the twin's own limit (max_image_shift) and well inside a Spectra's beam-shift range.
    image_shift_limit_m = IMAGE_SHIFT_LIMIT_M
    image_shift_commands = ("set_image_shift", "get_image_shift")

    def get_image_shift(self, job: dict) -> list[float]:
        return [float(v) for v in list(self.instrument().get_image_shift())[:2]]

    def set_image_shift(self, job: dict, xy: list[float]) -> None:
        self.instrument().set_image_shift([float(xy[0]), float(xy[1])])

    # -- plan ------------------------------------------------------------
    def current_xy(self) -> tuple[float, float]:
        x, y = list(self.instrument().get_stage())[:2]
        return float(x), float(y)

    def current_fov_m(self) -> float | None:
        return float(self.instrument().get_fov())

    def resolve_revisit(self, p: dict, revisit: dict) -> dict:
        return {"fov_m": float(revisit["fov_m"]), "imsize": int(revisit.get("imsize", int(p["imsize"]))), "dwell_s": float(revisit.get("dwell_s", float(p["dwell_s"])))}

    def tile_seconds(self, p: dict) -> float:
        return int(p["imsize"]) * int(p["imsize"]) * float(p["dwell_s"]) + self._eds_seconds(p)

    def _eds_seconds(self, p: dict) -> float:
        return 0.0

    def revisit_seconds(self, revisit: dict) -> float:
        return revisit["imsize"] ** 2 * revisit["dwell_s"]

    def _spectra_dose(self, p: dict, current: float, fov: float, exposures: int) -> float:
        return 0.0

    def costs(self, p: dict, grid: dict, revisit: dict | None, revisits: int, budgets: dict) -> tuple[dict[str, Cost], dict]:
        mapper = self.mapper
        imsize, dwell, fov = int(p["imsize"]), float(p["dwell_s"]), float(p["fov_m"])
        beam = mapper._beam_current()
        current = beam["pa"]
        budget = budgets.get("dose_e_per_a2")
        mapper._refuse_assumed_current(beam, budget)
        exposures = max_exposures(grid, fov)
        tile_dose = frame_dose(current, dwell, fov, imsize) * exposures
        # A revisit adds its frame on top of whatever the tiles gave that spot; regions lie a scoring window apart,
        # so one spot gets at most one revisit.
        revisit_dose = frame_dose(current, revisit["dwell_s"], revisit["fov_m"], revisit["imsize"]) if revisit else 0.0
        # EDS spectra park the beam inside a tile, on top of its frames: counted where they land, conservatively added.
        spectra_dose = self._spectra_dose(p, current, fov, exposures)
        max_dose = tile_dose + revisit_dose + spectra_dose
        parts = f"{tile_dose:.3g} from overlapping tiles" + (f" plus {spectra_dose:.3g} from EDS spectra" if spectra_dose else "") + (f" plus {revisit_dose:.3g} from a revisit image" if revisit else "")
        # Only the remedies for what adds dose here: told to "widen revisit fov_m" for a plan with no revisit, qwen3:8b added one.
        remedies = ["lower dwell_s or overlap"] + (["lower eds_exposure_s"] if spectra_dose else []) + (["widen revisit fov_m"] if revisit else [])
        advice = ", ".join(remedies) + "; the budget is the user's: give dose_budget_e_per_a2 only when they set a limit"
        extra = {
            "max_dose_e_per_a2": round(max_dose, 3),
            "beam_current_pa": current,
            # The dose estimate is only as good as the current; say when it was assumed (or is too old to rely on).
            **self.current_fields(beam),
        }
        if revisit:
            extra["revisit_dose_e_per_a2"] = round(revisit_dose, 3)
        if spectra_dose:
            extra["eds_dose_e_per_a2"] = round(spectra_dose, 3)
            extra["note"] = f"max_dose_e_per_a2 includes the EDS spectra's parked-beam dose, counted over {float(mapper.eds_dose_spread_m):.2g} m squares (eds_dose_spread_m)."
        return {"dose_e_per_a2": Cost(max_dose, "e/A^2", why=parts, advice=advice)}, extra

    @staticmethod
    def current_fields(beam: dict) -> dict:
        """The estimate's account of the current: where it came from, its age, and why a dose budget may not rely on it."""
        fields = {"beam_current_source": beam["source"]}
        if "age_s" in beam:
            fields["beam_current_age_s"] = beam["age_s"]
        reason = beam.get("reason") if beam["source"] == "default" else beam.get("refused")
        if reason:
            fields["beam_current_reason"] = reason
        return fields

    def revisit_only_costs(self, p: dict, revisit: dict, count: int, budgets: dict) -> tuple[dict[str, Cost], dict]:
        beam = self.mapper._beam_current()
        current = beam["pa"]
        budget = budgets.get("dose_e_per_a2")
        self.mapper._refuse_assumed_current(beam, budget)
        dose = frame_dose(current, revisit["dwell_s"], revisit["fov_m"], revisit["imsize"])
        message = "each revisit image would give {value:.3g} e/A^2, above the budget of {budget} e/A^2; widen revisit fov_m or lower dwell_s"
        extra = {"max_dose_e_per_a2": round(dose, 3), "revisit_dose_e_per_a2": round(dose, 3), "beam_current_pa": current, **self.current_fields(beam)}
        return {"dose_e_per_a2": Cost(dose, "e/A^2", message=message)}, extra

    def sample_stage(self) -> str | None:
        return self.mapper.stage_device_address or None

    def stage_targets(self, plan: dict, xs: list[tuple[str, float]], ys: list[tuple[str, float]]) -> list[dict]:
        stage = self.mapper.stage_device_address
        if not stage:
            return []
        return [{"step": s, "device": stage, "attribute": "x", "value": float(v)} for s, v in xs] + [
            {"step": s, "device": stage, "attribute": "y", "value": float(v)} for s, v in ys
        ]

    def move_target(self, plan: dict, name: str, arguments: Any) -> dict | None:
        if name != "move_stage":
            return None
        x, y = (float(v) for v in list(arguments[0])[:2])
        return {"x": x, "y": y}

    # -- run -------------------------------------------------------------
    def _scan(self):
        return self.mapper.proxy(self.mapper.scan_device_address)

    def setup(self, job: dict) -> None:
        plan = job["resolved_plan"]
        microscope = self.instrument()
        scan = self._scan()
        scan.imsize = int(plan["imsize"])
        scan.dwell_time = float(plan["dwell_s"])
        scan.scan_region = [0.0, 0.0, 1.0, 1.0]
        microscope.set_fov(float(plan["fov_m"]))
        current = list(microscope.get_stage())
        job.setdefault("pose", {"z": float(plan.get("z_m", current[2])), "alpha": float(current[3]), "beta": float(current[4])})
        self._setup_extras(job, plan)
        # The beam stays blanked between tiles: unblanked only while acquiring (see acquire_tile).
        microscope.blank_beam()

    def _setup_extras(self, job: dict, plan: dict) -> None:
        pass

    def make_safe(self, job: dict) -> None:
        self.instrument().blank_beam()

    def move(self, job: dict, x: float, y: float) -> bool:
        microscope = self.instrument()
        pose = job["pose"]
        # A move already under way (someone else's, just before the job started) would make this one refused.
        if not self.mapper._wait_for_stage(job, microscope):
            return False
        microscope.move_stage([x, y, pose["z"], pose["alpha"], pose["beta"]])
        return self.mapper._wait_for_stage(job, microscope)

    def stage_reported(self, job: dict) -> list[float]:
        return [float(v) for v in self.instrument().get_stage()]

    def acquire_tile(self, job: dict, tile: dict) -> dict | None:
        plan = job["resolved_plan"]
        microscope = self.instrument()
        # Unblank only for this tile's acquisitions: a beam left on during the stage move, file
        # writes and Tiled reads would park on one spot and put dose there that no estimate counts.
        # The tile is timed on the instrument's clock, not this computer's (on a twin with time_scale 0 the two
        # differ by any factor): the stitch fits drift, and revisits extrapolate it, against these times.
        frame_start = self.mapper._clock_s(microscope)
        microscope.unblank_beam()
        try:
            image_key = microscope.acquire_scanned_image([d.lower() for d in plan["detectors"]])
            acquired = (frame_start + self.mapper._clock_s(microscope)) / 2.0  # the middle of the frame
            extras = self._tile_extras(job, plan, microscope)
        finally:
            microscope.blank_beam()
        if extras is None:
            return None  # aborted part way through the spectrum image; a resume redoes this tile
        return {"image_key": image_key, "acquired": acquired, **extras}

    def _tile_extras(self, job: dict, plan: dict, microscope) -> dict | None:
        return {}

    def acquire_revisit(self, job: dict, revisit: dict) -> str:
        plan = job["resolved_plan"]
        microscope = self.instrument()
        microscope.set_fov(float(revisit["fov_m"]))
        scan = self._scan()
        scan.imsize = int(revisit["imsize"])
        scan.dwell_time = float(revisit["dwell_s"])
        microscope.unblank_beam()
        try:
            return microscope.acquire_scanned_image([d.lower() for d in plan["detectors"]])
        finally:
            microscope.blank_beam()
            microscope.set_fov(float(plan["fov_m"]))
            scan.imsize = int(plan["imsize"])
            scan.dwell_time = float(plan["dwell_s"])


@register_profile
class STEMProfile(ElectronBeamProfile):
    """A scanning transmission electron microscope: the beam profile plus per-tile EDS."""

    name = "stem"
    doc = "STEM: scanned images per tile (detectors, imsize, dwell_s), optional EDS per tile (per_tile eds_point or eds_map), electron-dose budget."
    plan_fields = ElectronBeamProfile.plan_fields | {"per_tile", "eds_exposure_s", "eds_map_points"}
    plan_defaults = dict(STEM_PLAN_DEFAULTS)

    def needed_commands(self, plan: dict) -> dict[str, list[str]]:
        out = super().needed_commands(plan)
        if plan.get("per_tile"):
            out.setdefault(self.mapper.microscope_device_address, []).extend(EDS_COMMANDS)
        return out

    def resolve(self, p: dict) -> None:
        if "eds_point" in p["per_tile"] and "eds_map" in p["per_tile"]:
            raise ValueError("choose eds_point or eds_map, not both")

    def _eds_seconds(self, p: dict) -> float:
        if "eds_point" in p["per_tile"]:
            return float(p["eds_exposure_s"])
        if "eds_map" in p["per_tile"]:
            return float(p["eds_exposure_s"]) * int(p["eds_map_points"]) ** 2
        return 0.0

    def _spectra_dose(self, p, current, fov, exposures):
        return eds_dose(current, float(p["eds_exposure_s"]), float(self.mapper.eds_dose_spread_m), p["per_tile"], int(p["eds_map_points"]), fov, exposures)

    def _setup_extras(self, job: dict, plan: dict) -> None:
        if any(a in plan["per_tile"] for a in ("eds_point", "eds_map")) and self.mapper.eds_device_address:
            eds = self.mapper.proxy(self.mapper.eds_device_address)
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
                if self.mapper.aborting(job):  # points x points spectra can take long; stop with the beam blanked
                    return None
                microscope.place_beam([float(fx), float(fy)])
                data = read_acquisition(microscope.acquire_spectrum("eds"), self.mapper._data_proxy())
                array, attrs = next(iter(data.values()))
                if cube is None:
                    cube = np.zeros((points, points, len(array)), dtype=np.asarray(array).dtype)
                cube[r, c] = array
        attrs = dict(attrs)
        attrs["spectrum_image_axes"] = "rows=+y, columns=+x, channels"
        return {"spectrum_image": (cube, attrs)}


@register_profile
class SEMProfile(ElectronBeamProfile):
    """A scanning electron microscope: the same beam, blanker and dose model, no EDS per tile.

    It drives only the beam command subset (BEAM_COMMANDS), so any microscope device offering
    it maps as an SEM. Its detectors default to ``se`` (secondary electrons); give the
    device's own names in the plan's ``detectors``.
    """

    name = "sem"
    doc = "SEM: scanned images per tile (detectors, default se; imsize, dwell_s), electron-dose budget; no per-tile EDS."
    plan_defaults = {**ElectronBeamProfile.plan_defaults, "detectors": ["se"]}

