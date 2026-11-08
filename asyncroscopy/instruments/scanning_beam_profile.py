"""The map profile every scanning-beam instrument shares: ``ScanningBeamProfile``.

It drives the instrument through the command subset of ``ScanningBeamInstrument``: ``move_stage``,
``get_stage``, ``set_fov``, ``acquire_scanned_image``, ``blank_beam`` and ``unblank_beam`` on the microscope,
and the SCAN device's ``imsize``, ``dwell_time`` and ``scan_region``. The beam is blanked between tiles and
whenever the job stops, and unblanked only while a tile or revisit image is acquired. The budget is electron
dose: ``dose_e_per_a2`` (the plan's ``dose_budget_e_per_a2``), counted where overlapping tiles stack and refused
on a real instrument when the beam current is only assumed; a budgeted job checks the current again when it
starts or resumes (asyncroscopy/routines/beam.py).

Its own Mapper settings (``settings``; set under the Mapper's ``properties:``): ``scan_device_address`` and
``max_beam_current_age_s``. The current is the microscope's own (``get_parameters``): measured, or its
``default_beam_current_pa``, assumed. A plan without ``detectors`` uses the instrument's ``default_detector``,
and live drift tracking (plan ``drift.track``) moves the image by ``set_image_shift`` within the instrument's
``image_shift_limit_m``.

The modalities register their subclasses: ``stem`` in electron_microscope/map_profile.py and ``sem`` in
scanning_electron_microscope/map_profile.py.
"""

from __future__ import annotations

import json
from typing import Any

import tango

from asyncroscopy.jobs.mapping.profile import Cost, InstrumentProfile, Setting
from asyncroscopy.routines.beam import DEFAULT_MAX_BEAM_CURRENT_AGE_S, BeamCurrentRefused, beam_current, check_current_at_start
from asyncroscopy.routines.dose import EDS_DOSE_SPREAD_M, frame_dose
from asyncroscopy.routines.grid import max_exposures

BEAM_COMMANDS = ("move_stage", "get_stage", "set_fov", "acquire_scanned_image", "blank_beam", "unblank_beam")
DOSE_BUDGET = "dose_e_per_a2"
BEAM_SETTINGS = {
    "scan_device_address": Setting(str, "asyncroscopy/scan/default", "SCAN settings device (imsize, dwell_time, scan_region)."),
    "max_beam_current_age_s": Setting(
        float, DEFAULT_MAX_BEAM_CURRENT_AGE_S, "Oldest screen-current measurement a dose budget accepts on hardware (asyncroscopy/routines/beam.py says why 15 min).",
    ),
}


class ScanningBeamProfile(InstrumentProfile):
    """A scanned electron beam: stage moves, a blanker, a field of view, an image per tile, electron dose."""

    plan_fields = frozenset({"imsize", "dwell_s", "detectors", "dose_budget_e_per_a2", "z_m"})
    plan_defaults = {"imsize": 512, "dwell_s": 1e-6}
    plan_example = {"size_m": [6e-7, 6e-7], "fov_m": 1.6e-7}  # a 600 nm square around the current stage position
    image_fields = frozenset({"imsize", "dwell_s", "detectors"})
    budget_fields = {"dose_budget_e_per_a2": DOSE_BUDGET}
    settings = BEAM_SETTINGS
    needs = {"microscope_device_address": BEAM_COMMANDS}
    # The instrument_type values this profile drives; a plan for any other instrument is refused at reserve.
    instrument_types: tuple[str, ...] = ()

    # Live drift tracking moves the beam with set_image_shift (the beam shift in STEM on AutoScript).
    image_shift_commands = ("set_image_shift", "get_image_shift")

    @property
    def image_shift_limit_m(self) -> float:
        """How far the instrument lets the image move without the stage (its max_image_shift)."""
        return float(self.instrument().read_attribute("image_shift_limit_m").value)

    def get_image_shift(self, job: dict) -> list[float]:
        return [float(v) for v in list(self.instrument().get_image_shift())[:2]]

    def set_image_shift(self, job: dict, xy: list[float]) -> None:
        self.instrument().set_image_shift([float(xy[0]), float(xy[1])])

    def example_plan(self, request: str) -> dict | None:
        if '"zoom"' in request:
            return None  # the Mapper's zoom example
        example = dict(self.plan_example)
        if "dose" in request.lower():  # a caller who mentioned dose sees where a dose limit goes
            example["dose_budget_e_per_a2"] = 50.0
        return example

    # -- beam current ----------------------------------------------------
    def _parameters(self) -> tuple[dict, str | None]:
        """The microscope's get_parameters, and why it could not be read."""
        try:
            return json.loads(self.instrument().get_parameters()), None
        except (tango.DevFailed, ValueError, TypeError) as error:
            return {}, f"get_parameters failed: {str(error).strip()[:200]}"

    def beam_current(self) -> dict:
        """The current dose estimates use: {"pa", "source" ("measured" or "default"), "reason" (when default),
        "refused" (why a dose budget may not rely on it; asyncroscopy.routines.beam), "simulated"}."""
        parameters, failure = self._parameters()
        beam = beam_current(parameters, self.ctx.settings.simulated, float(self.setting("max_beam_current_age_s")))
        current = {"simulated": beam.simulated, "pa": beam.pa, "source": beam.source, "refused": beam.refused}
        if beam.age_s is not None:
            current["age_s"] = beam.age_s
        if beam.source == "default":
            current["reason"] = failure or parameters.get("beam_current_reason") or beam.refused or "the microscope reports an assumed current"
        return current

    @staticmethod
    def refuse_assumed_current(current: dict, budget) -> None:
        """A dose budget means nothing with an assumed current on a real instrument: refuse it there."""
        if budget is not None and current.get("refused"):
            what = "is assumed" if current["source"] == "default" else "cannot be used for a dose budget"
            raise ValueError(
                f"this plan has a dose budget, but the beam current {what} ({current['pa']:g} pA; {current['refused']}). "
                "On an instrument the budget needs a recent measured current: insert the screen, call get_screen_current, "
                "retract the screen and reserve again, or leave out the dose budget"
            )

    def check_start(self, job: dict) -> None:
        """A plan with a dose budget, at start and on every resume: the approval may come long after the reserve, so
        the current must still be usable and no higher than the one the budget was checked with (routines/beam.py)."""
        plan = job["resolved_plan"]
        if plan.get("dose_budget_e_per_a2") is None and DOSE_BUDGET not in (plan.get("budgets") or {}):
            return
        parameters, _ = self._parameters()
        try:
            check_current_at_start(
                parameters, self.ctx.settings.simulated, job["estimate"].get("beam_current_pa"),
                float(self.setting("max_beam_current_age_s")),
            )
        except BeamCurrentRefused as exc:
            raise RuntimeError(f"the map's dose budget no longer holds at start: {exc}") from exc

    # -- plan ------------------------------------------------------------
    def resolve(self, p: dict) -> None:
        """Refuse an instrument of another kind, and take the plan's detectors from the instrument when it names none."""
        if self.instrument_types:
            found = str(self.instrument().read_attribute("instrument_type").value)
            if found not in self.instrument_types:
                raise ValueError(f"the {self.name!r} instrument profile drives {' or '.join(self.instrument_types)} instruments; "
                                 f"{self.ctx.settings.microscope_device_address} is a {found}: set this Mapper's instrument_profile to one for it")
        if "detectors" not in p:
            p["detectors"] = [str(self.instrument().read_attribute("default_detector").value).lower()]

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
        imsize, dwell, fov = int(p["imsize"]), float(p["dwell_s"]), float(p["fov_m"])
        beam = self.beam_current()
        current = beam["pa"]
        self.refuse_assumed_current(beam, budgets.get(DOSE_BUDGET))
        exposures = max_exposures(grid, fov)
        tile_dose = frame_dose(current, dwell, fov, imsize) * exposures
        # A revisit adds its frame on top of whatever the tiles gave that spot; regions lie a scoring window apart,
        # so one spot gets at most one revisit.
        revisit_dose = frame_dose(current, revisit["dwell_s"], revisit["fov_m"], revisit["imsize"]) if revisit else 0.0
        # EDS spectra park the beam inside a tile, on top of its frames: counted where they land, conservatively added.
        spectra_dose = self._spectra_dose(p, current, fov, exposures)
        max_dose = tile_dose + revisit_dose + spectra_dose
        parts = f"{tile_dose:.3g} from overlapping tiles" + (f" plus {spectra_dose:.3g} from EDS spectra" if spectra_dose else "") + (f" plus {revisit_dose:.3g} from a revisit image" if revisit else "")
        advice = "widen revisit fov_m, lower dwell_s or eds_exposure_s" if "per_tile" in self.plan_fields else "widen revisit fov_m or lower dwell_s"
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
            extra["note"] = f"max_dose_e_per_a2 includes the EDS spectra's parked-beam dose, counted over {EDS_DOSE_SPREAD_M:.2g} m squares (asyncroscopy.routines.dose.EDS_DOSE_SPREAD_M)."
        return {DOSE_BUDGET: Cost(max_dose, "e/A^2", why=parts, advice=advice)}, extra

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
        beam = self.beam_current()
        current = beam["pa"]
        self.refuse_assumed_current(beam, budgets.get(DOSE_BUDGET))
        dose = frame_dose(current, revisit["dwell_s"], revisit["fov_m"], revisit["imsize"])
        message = "each revisit image would give {value:.3g} e/A^2, above the budget of {budget} e/A^2; widen revisit fov_m or lower dwell_s"
        extra = {"max_dose_e_per_a2": round(dose, 3), "revisit_dose_e_per_a2": round(dose, 3), "beam_current_pa": current, **self.current_fields(beam)}
        return {DOSE_BUDGET: Cost(dose, "e/A^2", message=message)}, extra

    def stage_targets(self, plan: dict, xs: list[tuple[str, float]], ys: list[tuple[str, float]]) -> list[dict]:
        stage = self.ctx.settings.stage_device_address
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

    def image_shift_target(self, name: str, arguments: Any) -> list[float] | None:
        return [float(v) for v in list(arguments[0])[:2]] if name == "set_image_shift" else None

    # -- run -------------------------------------------------------------
    def _scan(self) -> tango.DeviceProxy:
        return self.ctx.proxy(self.setting("scan_device_address"))

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
        pose = job["pose"]
        # A move already under way (someone else's, just before the job started) would make this one refused.
        if not self.ctx.wait_for_stage(job):
            return False
        self.instrument().move_stage([x, y, pose["z"], pose["alpha"], pose["beta"]])
        return self.ctx.wait_for_stage(job)

    def stage_reported(self, job: dict) -> list[float]:
        return [float(v) for v in self.instrument().get_stage()]

    def acquire_tile(self, job: dict, tile: dict) -> dict | None:
        plan = job["resolved_plan"]
        microscope = self.instrument()
        # Unblank only for this tile's acquisitions: a beam left on during the stage move, file
        # writes and Tiled reads would park on one spot and put dose there that no estimate counts.
        # The tile is timed on the instrument's clock, not this computer's (on a twin with time_scale 0 the two
        # differ by any factor): the stitch fits drift, and revisits extrapolate it, against these times.
        frame_start = self.ctx.clock_s()
        microscope.unblank_beam()
        try:
            image_key = microscope.acquire_scanned_image([d.lower() for d in plan["detectors"]])
            acquired = (frame_start + self.ctx.clock_s()) / 2.0  # the middle of the frame
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
