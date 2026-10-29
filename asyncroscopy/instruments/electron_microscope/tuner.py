"""Tuner: autofocus and astigmatism correction as one bounded, abortable background job.

An agent decides *when* to tune; this job runs the loop, so dozens of
set-image-score calls never go through the LLM or the approval gate one by one
(docs/designs/concepts_brief.md, 2.3 and 2.7). ``run_job(device="Tuner",
plan={...})`` reserves, approves and starts it like any job.

How the tuning is done is a **strategy**, chosen by name in the plan:

* ``optistem``: the microscope's own autofocus (AutoScript runs OptiSTEM C1A1
  in STEM; the twin simulates it). The default (decision D2).
* ``image_metric``: bounded searches over the defocus, and over the stigmator
  where the microscope offers one, scoring images with
  ``asyncroscopy.routines.focus``.
* ``corrector``: the aberration corrector's own measure and correct commands
  (CEOS ``measureC1A1`` / ``correctAberration``; the twin's corrector).

A strategy is one small class registered with ``register_strategy`` (or the
``asyncroscopy.focus_strategies`` entry point). It never talks to Tango: it
calls the ``TuneContext`` it is given, which enforces the safety rules for
every strategy alike:

* the defocus, the stigmator and every corrector write are snapshotted before
  the first change, and restored when the job is aborted or fails;
* every write is bounded: no value leaves snapshot +/- the plan's range, and no
  single write changes it by more than ``max_step_m`` (longer moves are ramped);
* the beam is unblanked only while an image (or a vendor or corrector
  measurement) is taken, and blanked in ``finally``;
* the dose of every exposure is counted; ``reserve`` refuses a plan whose
  worst case is over its ``dose_budget_e_per_a2``, and an exposure that would
  pass the budget fails the job instead of being taken;
* the number of measurements is limited by ``max_steps``;
* a corrector correction that makes the measured error grow (a wrong sign or
  wrong units) stops the job at once, and the sharpness after tuning must not
  have fallen below the sharpness before.

* the beam current must be usable for a dose budget (``routines/beam.py``): on
  hardware a default or stale current is refused, at reserve and again at start;
* OptiSTEM's and the corrector's own scans are counted as a fixed number of frames
  at the plan's settings, which is a placeholder: on hardware those strategies are
  refused until ``vendor_dose_calibrated`` is set. OptiSTEM is refused on a
  microscope whose stigmator cannot be read and restored, and its result is checked
  against the plan's ranges.

A job is one step, so a resumed job starts again from a fresh snapshot, but only
after undoing whatever an earlier attempt left in place (a restart mid-tune, a
failed restore, corrector totals still sent); the snapshot and corrector totals are
saved to the job's state file as they change. If that undo fails, the resume is
refused.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np
import tango
from tango.server import device_property

from asyncroscopy.data.data_reader import read_acquisition
from asyncroscopy.instruments.electron_microscope.mapper import frame_dose
from asyncroscopy.jobs import JobDevice
from asyncroscopy.routines import focus
from asyncroscopy.routines.beam import DEFAULT_MAX_BEAM_CURRENT_AGE_S, BeamCurrentRefused, usable_beam_current

CORRECTIONS = ("focus", "stigmation")
PLAN_DEFAULTS: dict[str, Any] = {
    "corrections": ["focus"],
    "strategy": "optistem",
    "max_steps": 80,
    "focus_range_m": 3e-7,
    "stig_range_m": 1e-7,
    "max_step_m": 1e-7,
    "fov_m": 1e-7,
    "imsize": 256,
    "dwell_s": 2e-6,
    "detector": "haadf",
    "metric": "normalized_variance",
    "coarse_points": 13,
    "fine_points": 9,
    "rounds": 2,
    "stig_passes": 1,
    "sharpness_tolerance": 0.05,
    "corrector_gain": 1.0,
    "corrector_iterations": 5,
    "corrector_target_m": 2e-9,
    "error_tolerance_m": 1e-9,
}
PLAN_FIELDS = set(PLAN_DEFAULTS) | {"dose_budget_e_per_a2"}
HISTORY_KEPT = 200  # history entries a job keeps (the trace has every step)


def read_image(key: str, detector: str, data_proxy=None) -> np.ndarray:
    """The ``detector`` image of an acquisition (a local file, or a Tiled key through the DATA device)."""
    datasets = read_acquisition(key, data_proxy)
    named = [array for name, (array, _attrs) in datasets.items() if np.ndim(array) == 2 and detector.upper() in name.upper()]
    arrays = named or [array for array, _attrs in datasets.values() if np.ndim(array) == 2]
    if not arrays:
        raise RuntimeError(f"acquisition {key} holds no 2-D image")
    return np.asarray(arrays[0], dtype=np.float64)


class TuneFailed(RuntimeError):
    """A safety rule stopped the tuning; the job fails and the snapshot is restored."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class _Aborted(Exception):
    """abort was called; run_step returns None and the worker restores the snapshot."""


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------
class Strategy:
    """How to tune. Subclass, set ``name``, implement ``exposures`` and ``run``, and register it.

    ``corrections``: what it can correct. ``needs``: capabilities the microscope stack must offer
    ("stigmator", "autofocus", "corrector"). ``exposures(plan)``: the most frames' worth of dose
    ``run`` may use (an image is one frame; ``TuneContext.frames_per`` gives the vendor and
    corrector measurements), used for the dose and step checks at reserve. ``run(ctx)`` tunes
    through ``ctx`` only and returns a JSON-safe record.
    """

    name = ""
    corrections: frozenset = frozenset({"focus"})
    needs: frozenset = frozenset()

    def check(self, plan: dict, capabilities: set) -> None:
        """Raise ValueError when this strategy cannot run ``plan`` on a stack with ``capabilities``."""

    def exposures(self, plan: dict, frames_per: dict) -> int:
        raise NotImplementedError

    def measurements(self, plan: dict, frames_per: dict) -> int:
        """The most measurements (images, vendor runs, corrector measurements) ``run`` takes; by default
        one per frame of ``exposures``, which is right for a strategy that only takes images."""
        return int(self.exposures(plan, frames_per))

    def run(self, ctx: "TuneContext") -> dict:
        raise NotImplementedError


STRATEGIES: dict[str, type[Strategy]] = {}
_entry_points_loaded = False


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Make ``cls`` available as ``plan["strategy"] = cls.name``; usable as a class decorator."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a name")
    STRATEGIES[cls.name] = cls
    return cls


def strategies() -> dict[str, type[Strategy]]:
    """Registered strategies, including those installed with the ``asyncroscopy.focus_strategies`` entry point."""
    global _entry_points_loaded
    if not _entry_points_loaded:
        _entry_points_loaded = True
        from importlib.metadata import entry_points

        for point in entry_points(group="asyncroscopy.focus_strategies"):
            try:
                register_strategy(point.load())
            except Exception:  # a broken plug-in must not take the Tuner down
                pass
    return STRATEGIES


@register_strategy
class OptiStem(Strategy):
    """The vendor routine: one OptiSTEM C1A1 run (it corrects C1 and A1 together)."""

    name = "optistem"
    corrections = frozenset(CORRECTIONS)
    needs = frozenset({"autofocus"})

    def exposures(self, plan: dict, frames_per: dict) -> int:
        return frames_per["autofocus"]

    def measurements(self, plan: dict, frames_per: dict) -> int:
        return 1

    def check(self, plan: dict, capabilities: set) -> None:
        # OptiSTEM C1A1 changes the stigmator whatever the plan's corrections. A microscope whose stigmator the
        # Tuner cannot read and write (AutoScriptMicroscope offers no get_stigmator/set_stigmator) could not
        # put it back on an abort or a failure, so the run is refused rather than left unrestorable.
        if "stigmator" not in capabilities:
            raise ValueError(
                "strategy optistem runs OptiSTEM C1A1, which changes the stigmator, and this microscope offers no "
                "get_stigmator/set_stigmator to snapshot and restore it; use strategy image_metric for focus only"
            )

    def run(self, ctx: "TuneContext") -> dict:
        ctx.auto_focus()
        return {"runs": 1, "note": "OptiSTEM C1A1 corrects C1 and A1 together, whatever the plan's corrections"}


@register_strategy
class ImageMetric(Strategy):
    """Searches defocus (then the stigmator, where offered) for the sharpest image (routines/focus.py)."""

    name = "image_metric"
    corrections = frozenset(CORRECTIONS)

    def check(self, plan: dict, capabilities: set) -> None:
        if "stigmation" in plan["corrections"] and "stigmator" not in capabilities:
            raise ValueError("this microscope offers no stigmator (set_stigmator/get_stigmator), so image_metric can only focus; use strategy optistem or corrector for stigmation")
        if plan["metric"] not in focus.METRICS:
            raise ValueError(f"unknown metric {plan['metric']!r}; known: {sorted(focus.METRICS)}")

    @staticmethod
    def _options(plan: dict) -> dict:
        return {"coarse_points": int(plan["coarse_points"]), "fine_points": int(plan["fine_points"]), "rounds": int(plan["rounds"])}

    def exposures(self, plan: dict, frames_per: dict) -> int:
        options = self._options(plan)
        total = focus.search_1d_evaluations(**options) if "focus" in plan["corrections"] else 0
        if "stigmation" in plan["corrections"]:
            total += focus.search_2d_evaluations(passes=int(plan["stig_passes"]), **options)
        return total

    def run(self, ctx: "TuneContext") -> dict:
        plan, options, out = ctx.plan, self._options(ctx.plan), {}
        if "focus" in plan["corrections"]:
            centre, half = ctx.snapshot["defocus_m"], float(plan["focus_range_m"])
            found = focus.search_1d(lambda d: ctx.sharpness_at(defocus=d), centre, half, **options)
            ctx.check_inside(found["best"], centre, half, options, "defocus", "focus_range_m")
            ctx.set_defocus(found["best"])
            out["focus"] = {"best_m": found["best"], "images": found["evaluations"]}
        if "stigmation" in plan["corrections"]:
            centre, half = ctx.snapshot["stigmator_m"], float(plan["stig_range_m"])
            found = focus.search_2d(lambda a, b: ctx.sharpness_at(stigmator=[a, b]), centre, half, passes=int(plan["stig_passes"]), **options)
            for axis in (0, 1):
                ctx.check_inside(found["best"][axis], centre[axis], half, options, "stigmator", "stig_range_m")
            ctx.set_stigmator(found["best"])
            out["stigmation"] = {"best_m": found["best"], "images": found["evaluations"]}
        return out


@register_strategy
class Corrector(Strategy):
    """Measure C1/A1 with the corrector, send a bounded correction, measure again; stop if the error grows."""

    name = "corrector"
    corrections = frozenset(CORRECTIONS)
    needs = frozenset({"corrector"})

    @staticmethod
    def iterations(plan: dict) -> int:
        return max(1, int(plan["corrector_iterations"]))

    def exposures(self, plan: dict, frames_per: dict) -> int:
        return (self.iterations(plan) + 1) * frames_per["corrector_measure"]

    def measurements(self, plan: dict, frames_per: dict) -> int:
        return self.iterations(plan) + 1

    def run(self, ctx: "TuneContext") -> dict:
        plan = ctx.plan
        names = (["C1"] if "focus" in plan["corrections"] else []) + (["A1"] if "stigmation" in plan["corrections"] else [])
        target, gain = float(plan["corrector_target_m"]), float(plan["corrector_gain"])

        def error(measured: dict) -> float:
            return float(math.sqrt(sum(float(v) ** 2 for name in names for v in measured[name])))

        measured = ctx.corrector_measure()
        first = error(measured)
        iterations = 0
        while error(measured) > target and iterations < self.iterations(plan):
            before = error(measured)
            for name in names:
                ctx.corrector_correct(name, [gain * float(v) for v in measured[name]])
            measured = ctx.corrector_measure()
            iterations += 1
            if focus.error_grew(before, error(measured), float(plan["error_tolerance_m"])):
                raise TuneFailed(
                    "error_grew",
                    f"the corrector's {'+'.join(names)} error grew from {before:.3g} m to {error(measured):.3g} m after a correction: "
                    "the correction's sign or units are wrong for this corrector (see the first-light sign test). Stopped and restored.",
                )
        return {"names": names, "error_before_m": first, "error_after_m": error(measured), "iterations": iterations, "measured": measured}


# ----------------------------------------------------------------------
# The context a strategy tunes through
# ----------------------------------------------------------------------
class TuneContext:
    """Every instrument call a strategy can make, each bounded, counted, blanked and abortable."""

    def __init__(self, device: "Tuner", job: dict):
        self.device, self.job = device, job
        self.plan: dict = job["resolved_plan"]
        self.state: dict = job["tune"]
        self.snapshot: dict = self.state["snapshot"]
        self.frames_per: dict = self.plan["frames_per"]
        self._frame_dose = float(self.plan["frame_dose_e_per_a2"])
        self.microscope = device.proxy(device.microscope_device_address)
        self._defocus = float(self.snapshot["defocus_m"])
        self._stigmator = list(self.snapshot["stigmator_m"])

    # -- bookkeeping -------------------------------------------------
    def _check_abort(self) -> None:
        if self.device.aborting(self.job):
            raise _Aborted()

    def _expose(self, kind: str, frames: int) -> None:
        """Count one measurement and its dose before it is taken; refuse it over the plan's limits."""
        self._check_abort()
        plan, state = self.plan, self.state
        if state["steps"] + 1 > int(plan["max_steps"]):
            raise TuneFailed("max_steps", f"the plan allows {plan['max_steps']} measurements and the strategy asked for another ({kind})")
        dose = frames * self._frame_dose
        budget = float(plan["dose_budget_e_per_a2"])
        if state["dose_e_per_a2"] + dose > budget * (1 + 1e-9):
            raise TuneFailed("dose_budget", f"a {kind} would bring the dose to {state['dose_e_per_a2'] + dose:.4g} e/A^2, over the budget of {budget:g}")
        state["steps"] += 1
        state["dose_e_per_a2"] += dose

    def log(self, **record) -> None:
        history = self.state["history"]
        history.append({"step": self.state["steps"], **record})
        if len(history) > HISTORY_KEPT:
            del history[0]

    def _bounded(self, value: float, centre: float, half: float, what: str, field: str) -> float:
        value = float(value)
        if not math.isfinite(value) or abs(value - centre) > half * (1 + 1e-9) + 1e-15:
            raise TuneFailed("out_of_range", f"{what} {value:.4g} m is outside the snapshot {centre:.4g} m +/- {field} {half:g} m")
        return value

    def check_inside(self, best: float, centre: float, half: float, options: dict, what: str, field: str) -> None:
        """A search whose best value is at the edge of its range has not found a peak: fail rather than settle there."""
        margin = half / max(1, options.get("coarse_points", 2) - 1)
        if abs(best - centre) > half - margin:
            raise TuneFailed("peak_at_edge", f"the sharpest {what} ({best:.4g} m) is at the edge of the search range {centre:.4g} +/- {half:g} m: the error may be larger than {field}; widen it or start closer")

    # -- optics ------------------------------------------------------
    def get_defocus(self) -> float:
        return self._defocus

    def set_defocus(self, value: float) -> None:
        """Set the defocus, within snapshot +/- focus_range_m, ramping in steps of at most max_step_m."""
        target = self._bounded(value, self.snapshot["defocus_m"], float(self.plan["focus_range_m"]), "defocus", "focus_range_m")
        step = float(self.plan["max_step_m"])
        while abs(target - self._defocus) > 1e-15:
            self._check_abort()
            delta = max(-step, min(step, target - self._defocus))
            self.microscope.set_defocus(self._defocus + delta)
            self._defocus += delta
            self.state["writes"] += 1

    def get_stigmator(self) -> list[float]:
        return list(self._stigmator)

    def set_stigmator(self, value) -> None:
        """Set the stigmator [a, b], each within snapshot +/- stig_range_m, ramping in steps of at most max_step_m."""
        if "stigmator" not in self.plan["capabilities"]:
            raise TuneFailed("unsupported", "this microscope offers no stigmator")
        half = float(self.plan["stig_range_m"])
        target = [self._bounded(v, c, half, "stigmator", "stig_range_m") for v, c in zip(value, self.snapshot["stigmator_m"])]
        step = float(self.plan["max_step_m"])
        while max(abs(t - c) for t, c in zip(target, self._stigmator)) > 1e-15:
            self._check_abort()
            nxt = [c + max(-step, min(step, t - c)) for t, c in zip(target, self._stigmator)]
            self.microscope.set_stigmator(nxt)
            self._stigmator = nxt
            self.state["writes"] += 1

    # -- measurements ------------------------------------------------
    def image(self) -> np.ndarray:
        """One frame at the plan's field of view, size and dwell; the beam is on only while it is taken."""
        self._expose("image", 1)
        self.microscope.unblank_beam()
        try:
            key = self.microscope.acquire_scanned_image([self.plan["detector"]])
        finally:
            self.microscope.blank_beam()
        self.state["images"] += 1
        return self.device._read_image(key, self.plan["detector"])

    def sharpness(self) -> float:
        value = focus.sharpness(self.image(), self.plan["metric"])
        self.log(kind="image", defocus_m=self._defocus, stigmator_m=list(self._stigmator), sharpness=value)
        return value

    def sharpness_at(self, defocus: float | None = None, stigmator=None) -> float:
        """Move (bounded), then image and score: the ``evaluate`` callback the searches use."""
        if defocus is not None:
            self.set_defocus(defocus)
        if stigmator is not None:
            self.set_stigmator(stigmator)
        return self.sharpness()

    def _require_vendor_dose(self, kind: str) -> None:
        """A vendor or corrector measurement scans at settings the Tuner cannot read: on hardware only once calibrated."""
        if not self.plan.get("simulated") and not self.plan.get("vendor_dose_calibrated"):
            raise TuneFailed("dose_uncalibrated", f"a {kind} scans at its own settings, and this instrument's vendor_dose_calibrated is not set; its dose is unknown")

    def auto_focus(self) -> None:
        """The microscope's own autofocus (OptiSTEM C1A1 on the Spectra); it scans, so the beam is on for it.

        The vendor routine sets the defocus and stigmator itself, so they are read back and checked against the
        plan's ranges: a result outside them fails the job, and the snapshot is restored."""
        self._require_vendor_dose("autofocus")
        self._expose("autofocus", self.frames_per["autofocus"])
        self.microscope.unblank_beam()
        try:
            self.microscope.auto_focus()
        finally:
            self.microscope.blank_beam()
        self._defocus = float(self.microscope.get_defocus())
        if "stigmator" in self.plan["capabilities"]:
            self._stigmator = [float(v) for v in self.microscope.get_stigmator()]
        self.log(kind="autofocus", defocus_m=self._defocus, stigmator_m=list(self._stigmator))
        self._bounded(self._defocus, self.snapshot["defocus_m"], float(self.plan["focus_range_m"]), "the defocus the vendor autofocus set", "focus_range_m")
        if "stigmator" in self.plan["capabilities"]:
            for value, centre in zip(self._stigmator, self.snapshot["stigmator_m"]):
                self._bounded(value, centre, float(self.plan["stig_range_m"]), "the stigmator the vendor autofocus set", "stig_range_m")

    def _corrector(self) -> tango.DeviceProxy:
        return self.device.proxy(self.device.corrector_device_address)

    @staticmethod
    def _reply(text: str, command: str) -> dict:
        try:
            reply = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise TuneFailed("corrector_error", f"{command} returned something that is not JSON: {str(text)[:200]}") from exc
        if not isinstance(reply, dict) or "error" in reply:  # a JSON-RPC error is a failure, never a result
            raise TuneFailed("corrector_error", f"{command} failed: {json.dumps(reply.get('error') if isinstance(reply, dict) else reply)[:300]}")
        return reply.get("result", reply)

    def corrector_measure(self) -> dict:
        """C1 and A1 as the corrector measures them ({"C1": [m], "A1": [m, m]}); the beam is on for it."""
        self._require_vendor_dose("corrector measurement")
        self._expose("corrector_measure", self.frames_per["corrector_measure"])
        self.microscope.unblank_beam()
        try:
            text = self._corrector().measure_c1a1()
        finally:
            self.microscope.blank_beam()
        result = self._reply(text, "measure_c1a1")
        aberrations = result.get("aberrations", result)
        try:
            measured = {name: [float(v) for v in np.atleast_1d(aberrations[name])] for name in ("C1", "A1")}
        except (KeyError, TypeError, ValueError) as exc:
            raise TuneFailed("corrector_error", f"measure_c1a1 gave no C1/A1: {json.dumps(result)[:300]}") from exc
        self.log(kind="corrector_measure", **{f"{k}_m": v for k, v in measured.items()})
        return measured

    def corrector_correct(self, name: str, values) -> None:
        """Send one correction, each component at most max_step_m, and the running total within the plan's range."""
        if name not in ("C1", "A1"):
            raise TuneFailed("unsupported", f"the Tuner corrects C1 and A1 only, not {name}")
        step = float(self.plan["max_step_m"])
        half = float(self.plan["focus_range_m"] if name == "C1" else self.plan["stig_range_m"])
        values = [max(-step, min(step, float(v))) for v in values]
        sent = self.state["corrector_sent"].setdefault(name, [0.0] * len(values))
        totals = [s + v for s, v in zip(sent, values)]
        if any(abs(t) > half for t in totals):
            raise TuneFailed("out_of_range", f"corrections of {name} would add up to {totals} m, beyond the plan's +/-{half:g} m")
        self._check_abort()
        command = f"{name} " + " ".join(f"{v:.6e}" for v in values)
        self._reply(self._corrector().correct_aberration(command), f"correct_aberration {command}")
        self.state["corrector_sent"][name] = totals  # recorded only once sent, so a restore undoes exactly this
        self.state["writes"] += 1
        self.device._persist(self.job)  # on disk too: a device restart must not forget a correction it never undid
        self.log(kind="corrector_correct", name=name, values_m=values)


# ----------------------------------------------------------------------
# The job device
# ----------------------------------------------------------------------
class Tuner(JobDevice):
    """Autofocus and stigmation as a job: a pluggable strategy, bounded steps, a dose budget, snapshot and restore."""

    JOB_KIND = "autofocus and astigmatism correction (strategies: optistem, image_metric, corrector), bounded and restored on abort"
    PLAN_EXAMPLE = {"corrections": ["focus"], "strategy": "image_metric", "dose_budget_e_per_a2": 3000.0}

    microscope_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="Microscope to tune.")
    scan_device_address = device_property(dtype=str, default_value="asyncroscopy/scan/default", doc="SCAN settings device (image size and dwell for the tuning images).")
    corrector_device_address = device_property(dtype=str, default_value="", doc="Aberration corrector, for the corrector strategy; blank: none.")
    data_device_address = device_property(dtype=str, default_value="", doc="DATA device that resolves image keys; blank: keys are local files.")
    default_beam_current_pa = device_property(dtype=float, default_value=50.0, doc="Used for the dose when the microscope does not report beam_current_pa; refused on a stack that is not simulated.")
    autofocus_frames = device_property(dtype=int, default_value=8, doc="Frames of dose one vendor autofocus run counts as, each at the plan's fov_m, imsize and dwell_s (OptiSTEM scans several at its own settings; 8 is a placeholder until first light measures it).")
    corrector_measure_frames = device_property(dtype=int, default_value=4, doc="Frames of dose one corrector measurement counts as, at the plan's settings (a placeholder until first light measures it).")
    vendor_dose_calibrated = device_property(
        dtype=bool, default_value=False,
        doc="Set True once first light has checked autofocus_frames and corrector_measure_frames against the scan settings "
        "OptiSTEM and the corrector really use. Until then, on a stack that is not simulated, strategies that take vendor "
        "or corrector measurements are refused: their dose cannot be counted against the budget.",
    )
    max_beam_current_age_s = device_property(
        dtype=float, default_value=DEFAULT_MAX_BEAM_CURRENT_AGE_S,
        doc="Oldest screen-current measurement a dose budget accepts on hardware (asyncroscopy/routines/beam.py says why 15 min).",
    )
    allow_corrector_writes = device_property(dtype=bool, default_value=False, doc="Allow the corrector strategy on a stack that is not simulated (only once first light has settled the correction's sign and units; decision D2).")

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def _stack_simulated(self, parameters: dict) -> bool:
        """The stack's ``simulated`` property where it has one, else ASYNCROSCOPY_SIMULATED, else what the microscope reports (as the Mapper)."""
        flag = getattr(self, "simulated", None)
        if flag is not None:
            return bool(flag)
        env = os.environ.get("ASYNCROSCOPY_SIMULATED", "").strip().lower()
        if env:
            return env in ("1", "true", "yes", "on")
        return bool(parameters.get("simulated", False))

    def _capabilities(self, microscope) -> set:
        commands = {c.cmd_name for c in microscope.command_list_query()}
        caps = set()
        if {"set_stigmator", "get_stigmator"} <= commands:
            caps.add("stigmator")
        if "auto_focus" in commands:
            caps.add("autofocus")
        if self.corrector_device_address:
            caps.add("corrector")
        return caps

    def resolve_plan(self, plan: dict) -> dict:
        unknown = sorted(set(plan) - PLAN_FIELDS)
        if unknown:
            raise ValueError(f"unknown plan field(s) {unknown}; allowed: {sorted(PLAN_FIELDS)}")
        if "dose_budget_e_per_a2" not in plan:
            raise ValueError("dose_budget_e_per_a2 is required: the most electrons per square angstrom tuning may put on the field")
        p = {**PLAN_DEFAULTS, **plan}
        corrections = p["corrections"]
        if isinstance(corrections, str):
            corrections = [corrections]
        if not isinstance(corrections, list) or not corrections or any(c not in CORRECTIONS for c in corrections):
            raise ValueError(f"corrections must be a non-empty list of {list(CORRECTIONS)}, got {p['corrections']!r}")
        p["corrections"] = [c for c in CORRECTIONS if c in corrections]
        registry = strategies()
        if p["strategy"] not in registry:
            raise ValueError(f"unknown strategy {p['strategy']!r}; registered: {sorted(registry)}")
        for key in ("focus_range_m", "stig_range_m", "max_step_m", "fov_m", "dwell_s", "dose_budget_e_per_a2", "corrector_target_m"):
            if isinstance(p[key], bool) or not isinstance(p[key], (int, float)) or not float(p[key]) > 0:
                raise ValueError(f"{key} must be a number > 0, got {p[key]!r}")
        for key in ("max_steps", "imsize", "coarse_points", "fine_points"):
            if isinstance(p[key], bool) or not isinstance(p[key], int) or p[key] < 3:
                raise ValueError(f"{key} must be an integer >= 3, got {p[key]!r}")
        if not 0 < float(p["corrector_gain"]) <= 1:
            raise ValueError("corrector_gain must be in (0, 1]: a correction never sends more than was measured")
        strategy = registry[p["strategy"]]()
        missing = set(p["corrections"]) - set(strategy.corrections)
        if missing:
            raise ValueError(f"strategy {strategy.name} cannot correct {sorted(missing)}")

        microscope = self.proxy(self.microscope_device_address)
        try:
            parameters = json.loads(microscope.get_parameters())
        except (tango.DevFailed, ValueError, TypeError):
            parameters = {}
        simulated = self._stack_simulated(parameters)
        capabilities = self._capabilities(microscope)
        lacking = set(strategy.needs) - capabilities
        if lacking:
            raise ValueError(f"strategy {strategy.name} needs {sorted(lacking)}, which this stack does not offer (corrector_device_address, auto_focus, set_stigmator)")
        if strategy.name == "corrector" and not simulated and not bool(self.allow_corrector_writes):
            raise ValueError("corrector writes are not allowed on this instrument until first light has settled their sign and units (allow_corrector_writes); use strategy optistem")
        strategy.check(p, capabilities)

        # The budget is only as good as the current: on hardware a default or stale current is refused (routines/beam.py).
        beam = usable_beam_current(parameters, simulated, float(self.max_beam_current_age_s), float(self.default_beam_current_pa))
        current, source = beam.pa, beam.source
        frames_per = {"image": 1, "autofocus": int(self.autofocus_frames), "corrector_measure": int(self.corrector_measure_frames)}
        # Images are taken at the plan's own fov_m, imsize and dwell_s (on_start sets them), so their dose is known.
        # OptiSTEM and the corrector scan at their own settings, which the Tuner cannot read: each run is counted as
        # autofocus_frames / corrector_measure_frames frames at the plan's settings, an assumption the estimate states.
        vendor = sorted(set(strategy.needs) & {"autofocus", "corrector"})
        calibrated = bool(self.vendor_dose_calibrated)
        if vendor and not simulated and not calibrated:
            raise ValueError(
                f"strategy {strategy.name} takes {' and '.join(vendor)} measurements, which scan at their own settings; their dose is "
                f"counted as {frames_per['autofocus']} (autofocus) / {frames_per['corrector_measure']} (corrector) frames at the plan's "
                "settings, a placeholder until first light measures it. On an instrument that is refused until the Tuner's "
                "vendor_dose_calibrated is set; use strategy image_metric, whose images are all at the plan's settings"
            )
        per_frame = frame_dose(current, float(p["dwell_s"]), float(p["fov_m"]), int(p["imsize"]))
        frames = int(strategy.exposures(p, frames_per)) + 2  # plus the sharpness images before and after
        measurements = int(strategy.measurements(p, frames_per)) + 2
        worst = frames * per_frame
        budget = float(p["dose_budget_e_per_a2"])
        if measurements > int(p["max_steps"]):
            raise ValueError(f"strategy {strategy.name} needs up to {measurements} measurements with these settings, above max_steps {p['max_steps']}; raise max_steps or use fewer points")
        if worst > budget:
            raise ValueError(
                f"tuning could give up to {worst:.4g} e/A^2 ({frames} frames of {per_frame:.3g} e/A^2 at {current:g} pA), above the budget of {budget:g} e/A^2; "
                "raise dose_budget_e_per_a2, lower dwell_s, use a larger fov_m or fewer search points"
            )
        frame_s = int(p["imsize"]) ** 2 * float(p["dwell_s"])
        estimate = {
            "steps": 1,
            "measurements": measurements,
            "frames": frames,
            "duration_s": round(frames * (frame_s + 0.5), 1),
            "max_dose_e_per_a2": round(worst, 3),
            "frame_dose_e_per_a2": round(per_frame, 4),
            **beam.estimate(),
            "strategy": strategy.name,
            "dose_assumption": (
                f"{' and '.join(vendor)} runs scan at their own settings, which the Tuner cannot read; each is counted as "
                f"{frames_per['autofocus'] if 'autofocus' in vendor else frames_per['corrector_measure']} frames at fov_m {p['fov_m']:g} m, "
                f"imsize {p['imsize']}, dwell_s {p['dwell_s']:g} s ("
                + ("checked at first light: vendor_dose_calibrated" if calibrated else "a placeholder, not calibrated") + ")"
                if vendor else "every frame is at the plan's fov_m, imsize and dwell_s"
            ),
        }
        return {
            **p,
            "steps": [{"phase": "tune"}],
            "estimate": estimate,
            "frame_dose_e_per_a2": per_frame,
            "frames_per": frames_per,
            "capabilities": sorted(capabilities),
            "simulated": simulated,
            "vendor_dose_calibrated": calibrated,
        }

    # ------------------------------------------------------------------
    # Snapshot, run, restore
    # ------------------------------------------------------------------
    def _read_image(self, key: str, detector: str) -> np.ndarray:
        return read_image(key, detector, self.proxy(self.data_device_address) if self.data_device_address else None)

    def phase(self, job: dict) -> str | None:
        tune = job.get("tune") or {}
        if not tune.get("steps"):
            return None
        return f"{tune.get('strategy')}: {tune['steps']} measurements, {tune['dose_e_per_a2']:.4g} of {job['resolved_plan']['dose_budget_e_per_a2']:g} e/A^2"

    def _persist(self, job: dict, quiet: bool = False) -> None:
        """Save the job's state file now, so a device restart finds the snapshot and any corrector totals still sent."""
        try:
            with self._lock:
                self._save(job)
        except Exception:
            if not quiet:
                raise

    @staticmethod
    def _left_unrestored(previous: dict) -> list[str]:
        """What an earlier attempt of this job left on the instrument: why its snapshot must be restored first."""
        if not previous.get("snapshot"):
            return []
        why = []
        if previous.get("outcome") == "running":
            why.append("the earlier attempt stopped without restoring (the device restarted mid-tune)")
        if previous.get("restore_errors"):
            why.append(f"restoring after the earlier attempt failed ({'; '.join(previous['restore_errors'])[:200]})")
        if previous.get("corrector_sent"):
            why.append(f"corrector corrections still outstanding: {json.dumps(previous['corrector_sent'])}")
        return why

    def _check_current_at_start(self, plan: dict, microscope) -> None:
        """The approval may come long after the reserve: the current must still be usable, and no higher than budgeted."""
        try:
            parameters = json.loads(microscope.get_parameters())
        except (tango.DevFailed, ValueError, TypeError):
            parameters = {}
        try:
            beam = usable_beam_current(parameters, bool(plan["simulated"]), float(self.max_beam_current_age_s), float(self.default_beam_current_pa))
        except BeamCurrentRefused as exc:
            raise TuneFailed("beam_current", f"at start: {exc}") from exc
        budgeted = float(plan["estimate"]["beam_current_pa"])
        if beam.pa > budgeted * 1.05:
            raise TuneFailed("beam_current", f"the beam current is now {beam.pa:g} pA, above the {budgeted:g} pA the dose budget was checked with; reserve again")

    def on_start(self, job: dict) -> None:
        plan = job["resolved_plan"]
        microscope = self.proxy(self.microscope_device_address)
        scan = self.proxy(self.scan_device_address)
        microscope.blank_beam()  # nothing is exposed except while a measurement is taken
        # A resume: an earlier attempt that left optics or corrector changes in place is undone first, against the
        # earlier (true) snapshot. A fresh snapshot taken over them would make those changes the new baseline.
        previous = job.get("tune") or {}
        unrestored = self._left_unrestored(previous)
        if unrestored:
            errors: list[str] = []
            restored = self._restore_optics(job, errors)
            self._restore_imaging(job, errors)
            if errors:
                raise TuneFailed(
                    "resume_refused",
                    f"{'; '.join(unrestored)}; restoring that first failed ({'; '.join(errors)}), so the job will not take a new "
                    "snapshot over it. Put the optics back by hand (see the job's tune.snapshot) before resuming",
                )
            previous.update(outcome="restored_before_resume", restored=restored)
            previous.pop("restore_errors", None)
            job["tune"] = previous
            self._persist(job)
        self._check_current_at_start(plan, microscope)
        snapshot = {
            "defocus_m": float(microscope.get_defocus()),
            "stigmator_m": [float(v) for v in microscope.get_stigmator()] if "stigmator" in plan["capabilities"] else [0.0, 0.0],
            "fov_m": float(microscope.get_fov()),
            "imsize": int(scan.imsize),
            "dwell_s": float(scan.dwell_time),
            "scan_region": [float(v) for v in scan.scan_region],
        }
        job["tune"] = {
            "strategy": plan["strategy"], "corrections": plan["corrections"], "outcome": "running", "snapshot": snapshot,
            "steps": 0, "images": 0, "writes": 0, "dose_e_per_a2": 0.0, "dose_budget_e_per_a2": float(plan["dose_budget_e_per_a2"]),
            "corrector_sent": {}, "history": [],
        }
        self._persist(job)  # before the first change: a restart mid-tune must find this snapshot
        microscope.set_fov(float(plan["fov_m"]))
        scan.imsize = int(plan["imsize"])
        scan.dwell_time = float(plan["dwell_s"])
        scan.scan_region = [0.0, 0.0, 1.0, 1.0]

    def run_step(self, job: dict, index: int) -> dict:
        plan, state = job["resolved_plan"], job["tune"]
        ctx = TuneContext(self, job)
        strategy = strategies()[plan["strategy"]]()
        try:
            state["sharpness_before"] = ctx.sharpness()
            state["strategy_result"] = strategy.run(ctx)
            state["sharpness_after"] = ctx.sharpness()
        except _Aborted:
            return None
        except TuneFailed as exc:
            state["outcome"], state["reason"] = "failed", str(exc)
            raise
        before, after = state["sharpness_before"], state["sharpness_after"]
        if after < before * (1.0 - float(plan["sharpness_tolerance"])):
            state["outcome"] = "failed"
            state["reason"] = f"sharpness_fell: {before:.4g} before, {after:.4g} after"
            raise TuneFailed("sharpness_fell", f"the image is less sharp after tuning ({after:.4g}) than before ({before:.4g}); restoring the snapshot")
        state["outcome"] = "done"
        microscope = ctx.microscope
        state["defocus_after_m"] = float(microscope.get_defocus())
        if "stigmator" in plan["capabilities"]:
            state["stigmator_after_m"] = [float(v) for v in microscope.get_stigmator()]
        return {key: state.get(key) for key in ("strategy", "outcome", "sharpness_before", "sharpness_after", "steps", "images", "dose_e_per_a2", "defocus_after_m")}

    def _restore_imaging(self, job: dict, errors: list) -> None:
        snapshot = (job.get("tune") or {}).get("snapshot")
        if not snapshot:
            return
        for what, action in (
            ("fov", lambda: self.proxy(self.microscope_device_address).set_fov(snapshot["fov_m"])),
            ("scan", lambda: self._restore_scan(snapshot)),
        ):
            try:
                action()
            except Exception as exc:
                errors.append(f"{what}: {str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__}")

    def _restore_scan(self, snapshot: dict) -> None:
        scan = self.proxy(self.scan_device_address)
        scan.imsize = int(snapshot["imsize"])
        scan.dwell_time = float(snapshot["dwell_s"])
        scan.scan_region = snapshot["scan_region"]

    def _restore_optics(self, job: dict, errors: list) -> dict:
        """Put the defocus, stigmator and corrector back as the snapshot found them, in bounded steps."""
        state = job.get("tune") or {}
        snapshot = state.get("snapshot")
        if not snapshot:
            return {}
        plan = job["resolved_plan"]
        step = float(plan["max_step_m"])
        microscope = self.proxy(self.microscope_device_address)
        restored = {}
        try:
            here = float(microscope.get_defocus())
            target = float(snapshot["defocus_m"])
            while abs(target - here) > 1e-15:
                here += max(-step, min(step, target - here))
                microscope.set_defocus(here)
            restored["defocus_m"] = target
        except Exception as exc:
            errors.append(f"defocus: {exc}".splitlines()[0][:200])
        if "stigmator" in plan["capabilities"]:
            try:
                here = [float(v) for v in microscope.get_stigmator()]
                target = [float(v) for v in snapshot["stigmator_m"]]
                while max(abs(t - h) for t, h in zip(target, here)) > 1e-15:
                    here = [h + max(-step, min(step, t - h)) for t, h in zip(target, here)]
                    microscope.set_stigmator(here)
                restored["stigmator_m"] = target
            except Exception as exc:
                errors.append(f"stigmator: {exc}".splitlines()[0][:200])
        sent = state.get("corrector_sent") or {}
        if sent:
            corrector = self.proxy(self.corrector_device_address)
            for name, totals in list(sent.items()):
                # Undo exactly what was sent, in the backend's own convention: send the negated total, in bounded pieces.
                try:
                    remaining = [-float(v) for v in totals]
                    while max(abs(v) for v in remaining) > 1e-15:
                        piece = [max(-step, min(step, v)) for v in remaining]
                        reply = json.loads(corrector.correct_aberration(f"{name} " + " ".join(f"{v:.6e}" for v in piece)))
                        if isinstance(reply, dict) and "error" in reply:
                            raise RuntimeError(json.dumps(reply["error"])[:200])
                        remaining = [r - p for r, p in zip(remaining, piece)]
                        state["corrector_sent"][name] = [-r for r in remaining]
                        self._persist(job, quiet=True)
                    state["corrector_sent"].pop(name, None)
                    self._persist(job, quiet=True)
                    restored[f"corrector_{name}_undone_m"] = [float(v) for v in totals]
                except Exception as exc:
                    errors.append(f"corrector {name}: {exc}".splitlines()[0][:200])
        return restored

    def on_stop(self, job: dict, reason: str) -> None:
        errors: list[str] = []
        try:
            self.proxy(self.microscope_device_address).blank_beam()
        except Exception as exc:
            errors.append(f"blank_beam: {exc}".splitlines()[0][:200])
        restored = self._restore_optics(job, errors)
        self._restore_imaging(job, errors)
        state = job.setdefault("tune", {})
        if state.get("outcome") in (None, "running", "done"):
            state["outcome"] = reason
        state["restored"] = restored
        if errors:
            state["restore_errors"] = errors
            raise RuntimeError("restoring the snapshot failed: " + "; ".join(errors))

    def on_finish(self, job: dict) -> dict:
        errors: list[str] = []
        self.proxy(self.microscope_device_address).blank_beam()
        self._restore_imaging(job, errors)  # the tuned optics stay; the imaging settings go back
        if errors:
            job["tune"]["restore_errors"] = errors
        return {}

    def summary(self, job: dict) -> dict:
        state = dict(job.get("tune") or {})
        history = state.pop("history", [])
        return {"tune": {**state, "history": history[-20:], "history_omitted": max(0, len(history) - 20)}}


if __name__ == "__main__":
    Tuner.run_server()
