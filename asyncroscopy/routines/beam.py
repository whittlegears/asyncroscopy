"""Whether a beam current may be used for a dose budget: one rule for every job (Tuner, Mapper).

A microscope reports its current in ``get_parameters`` as ``beam_current_pa`` with
``beam_current_source`` ("measured" from the fluorescent screen, or "default": an assumed
nominal value) and, when measured, ``beam_current_age_s``. On a simulated stack any
current the twin reports is usable. On hardware:

* a missing ``beam_current_source`` counts as "default" (a backend that does not say where
  its number came from has not measured it);
* a "default" current is refused: a dose budget computed from an assumed current is not a
  budget;
* a measured current is refused when it reports no age, or is older than ``max_age_s``;
* a current that is not a finite number > 0 is refused everywhere (it would make every
  dose zero and so pass any budget).

``DEFAULT_MAX_BEAM_CURRENT_AGE_S`` is 15 minutes. The gun's emission drifts: a cold FEG's
(X-CFEG) falls steadily between flashes, and a Schottky FEG's (X-FEG) wanders more slowly.
A screen reading older than a quarter of an hour is not evidence of the current now. The
AutoScript backend already drops its reading when anything that sets the current changes
(monochromator focus, spot size, apertures, optical mode, high tension), so the age limit
only has to cover drift. Lengthen it per device (``max_beam_current_age_s``) once first
light has measured how far this gun drifts in that time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_MAX_BEAM_CURRENT_AGE_S = 900.0


class BeamCurrentRefused(ValueError):
    """The reported current may not be used for a dose budget on this stack; the message says why."""


@dataclass(frozen=True)
class BeamCurrent:
    pa: float
    source: str  # "measured" or "default"
    age_s: float | None
    simulated: bool
    refused: str | None  # why it may not be used for a dose budget; None when it may

    def estimate(self) -> dict:
        """The fields a job's estimate reports about the current it used."""
        out = {"beam_current_pa": self.pa, "beam_current_source": self.source}
        if self.age_s is not None:
            out["beam_current_age_s"] = self.age_s
        if self.refused:
            out["beam_current_reason"] = self.refused
        return out


def _number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def beam_current(parameters: dict, simulated: bool, max_age_s: float = DEFAULT_MAX_BEAM_CURRENT_AGE_S, default_pa: float = 50.0) -> BeamCurrent:
    """The current a dose estimate should use, and (in ``refused``) why a budget may not rely on it."""
    parameters = parameters if isinstance(parameters, dict) else {}
    reported = _number(parameters.get("beam_current_pa"))
    age = _number(parameters.get("beam_current_age_s"))
    if "beam_current_pa" not in parameters or parameters.get("beam_current_pa") is None:
        pa, source = float(default_pa), "default"
        refused = f"the microscope reports no beam_current_pa; {default_pa:g} pA is assumed"
    else:
        pa = reported if reported is not None else float("nan")
        source = "measured" if parameters.get("beam_current_source") == "measured" else "default"
        refused = None
        if source == "default":
            refused = parameters.get("beam_current_reason") or (
                "the microscope reports an assumed current" if "beam_current_source" in parameters
                else "the microscope does not say the current was measured (no beam_current_source)"
            )
    if not (math.isfinite(pa) and pa > 0):
        # Never usable, simulated or not: a zero or negative current makes every dose zero.
        return BeamCurrent(pa, source, age, simulated, f"beam_current_pa {parameters.get('beam_current_pa')!r} is not a current > 0")
    if simulated:
        return BeamCurrent(pa, source, age, simulated, None)
    if refused is None:
        if age is None:
            refused = "the measured current reports no age (beam_current_age_s), so it cannot be shown to be current"
        elif age < 0 or age > float(max_age_s):
            refused = f"the screen-current measurement is {age:g} s old, beyond the {float(max_age_s):g} s this stack accepts"
    return BeamCurrent(pa, source, age, simulated, refused)


def usable_beam_current(parameters: dict, simulated: bool, max_age_s: float = DEFAULT_MAX_BEAM_CURRENT_AGE_S, default_pa: float = 50.0) -> BeamCurrent:
    """The current, for a dose budget; raises ``BeamCurrentRefused`` with the reason when it may not be used."""
    current = beam_current(parameters, simulated, max_age_s, default_pa)
    if current.refused:
        what = "is assumed" if current.source == "default" else "cannot be used for a dose budget"
        raise BeamCurrentRefused(
            f"the beam current {what} ({current.pa:g} pA, {current.source}): {current.refused}. "
            "Insert the screen, call get_screen_current, retract the screen, then reserve again"
        )
    return current
