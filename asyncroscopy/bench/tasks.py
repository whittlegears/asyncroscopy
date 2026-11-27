"""Scored tasks for agents controlling a microscope, refereed by the digital twin.

Each task has three parts:

``setup(env, rng)``
    Puts the twin into a seeded, perturbed state: defocused, aberrated,
    off zone, or with a dropped corrector link. Returns what the referee needs
    later; the agent never sees it.
``score(env, state)``
    Reads ground truth that only the twin knows (``get_ground_truth``, the
    corrector's simulated coefficients) and returns an error, a pass/fail, and
    a 0–1 score.
``baseline(env)``
    Our own routine for the task, the score to beat.

The referee commands are blocked in ``configs/mcp_benchmark.yaml``, so an
agent working through MCP cannot read the answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from asyncroscopy.autonomy.routines import Imager, Resilient, align_zone_axis, focus, measure_and_correct
from asyncroscopy.health.monitor import HealthMonitor


@dataclass
class Env:
    instrument: Any                    # DigitalTwinTilt proxy
    corrector: Any = None              # DigitalTwinCorrector proxy
    fetch_image: Callable[[str], np.ndarray] | None = None     # scanned-image key -> 2-D array
    fetch_pattern: Callable[[str], np.ndarray] | None = None   # camera key -> 2-D diffraction pattern
    corrector_name: str = ""
    extra: dict = field(default_factory=dict)

    def truth(self) -> dict:
        return json.loads(self.instrument.get_ground_truth())

    def aberrations(self) -> dict:
        return json.loads(self.corrector.get_aberrations_coeff_sim())


@dataclass
class Task:
    name: str
    instructions: str
    setup: Callable[[Env, np.random.Generator], dict]
    score: Callable[[Env, dict], dict]
    baseline: Callable[[Env], dict]
    needs: tuple[str, ...] = ()


def _score(error: float, tolerance: float) -> float:
    """1 at zero error, 0.5 at the pass tolerance, falling towards 0."""
    return float(1.0 / (1.0 + (error / tolerance) ** 2))


# -- focus ------------------------------------------------------------------------------

def _focus_setup(env: Env, rng) -> dict:
    env.instrument.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    offset = float(rng.choice([-1, 1]) * rng.uniform(30e-9, 60e-9))
    t = env.truth()
    env.instrument.set_defocus(t["true_focus_m"] + offset)
    return {"start_offset_m": offset}


def _focus_score(env: Env, state: dict) -> dict:
    t = env.truth()
    err = abs(t["defocus_m"] - t["true_focus_m"]) * 1e9
    return {"error": err, "unit": "nm", "pass": err <= 10.0, "score": _score(err, 10.0),
            "start_error": abs(state["start_offset_m"]) * 1e9}


def _focus_baseline(env: Env) -> dict:
    imager = Imager(env.instrument, env.fetch_image, Resilient())
    return focus(imager, mode="defocus", step_m=15e-9, max_offset_m=90e-9, score="edge_sharpness")


# -- aberrations ------------------------------------------------------------------------

ABERRATION_TOLERANCE_M = {"C1": 2e-9, "A1": 2e-9}


def _ab_setup(env: Env, rng) -> dict:
    c1 = float(rng.uniform(15e-9, 40e-9) * rng.choice([-1, 1]))
    a1 = (rng.uniform(4e-9, 10e-9) * np.array([np.cos(t := rng.uniform(0, 2 * np.pi)), np.sin(t)])).tolist()
    env.corrector.set_aberrations_coeff_sim(json.dumps({"C1": [c1], "A1": a1}))
    return {"start": {"C1": [c1], "A1": a1}}


def _ab_score(env: Env, state: dict) -> dict:
    ab = env.aberrations()
    errs = {n: float(np.linalg.norm(ab[n])) * 1e9 for n in ABERRATION_TOLERANCE_M}
    passed = all(errs[n] <= ABERRATION_TOLERANCE_M[n] * 1e9 for n in errs)
    score = float(np.mean([_score(errs[n], ABERRATION_TOLERANCE_M[n] * 1e9) for n in errs]))
    return {"error": max(errs.values()), "unit": "nm", "components": errs, "pass": passed, "score": score}


def _ab_baseline(env: Env) -> dict:
    return measure_and_correct(env.corrector, thresholds_m=ABERRATION_TOLERANCE_M)


# -- zone axis --------------------------------------------------------------------------

def _zone_setup(env: Env, rng) -> dict:
    t = env.truth()
    alpha = t["on_zone_alpha_deg"] + float(rng.choice([-1, 1]) * rng.uniform(0.2, 0.45))
    env.instrument.move_stage([0.0, 0.0, 0.0, alpha, 0.0])
    return {"start_alpha_deg": alpha}


def _zone_score(env: Env, state: dict) -> dict:
    t = env.truth()
    err = abs(t["stage"][3] - t["on_zone_alpha_deg"])
    start = abs(state["start_alpha_deg"] - t["on_zone_alpha_deg"])
    return {"error": err, "unit": "deg", "pass": err <= 0.05, "score": _score(err, 0.05), "start_error": start}


def _zone_baseline(env: Env) -> dict:
    return align_zone_axis(env.instrument, env.fetch_pattern, Resilient(), axes=("alpha",), step_deg=0.15,
                           max_offset_deg=0.6)


# -- recovery ---------------------------------------------------------------------------

def _rec_setup(env: Env, rng) -> dict:
    env.corrector.command_inout("inject_link_drop")
    return {}


def _rec_score(env: Env, state: dict) -> dict:
    try:
        ok = str(env.corrector.state()) == "ON" and "aberrations" in env.corrector.measure_c1a1()
    except Exception:
        ok = False
    return {"error": 0.0 if ok else 1.0, "unit": "failed", "pass": ok, "score": 1.0 if ok else 0.0}


def _rec_baseline(env: Env) -> dict:
    from asyncroscopy.health.monitor import recover
    name = env.corrector_name or env.corrector.dev_name()
    return recover(HealthMonitor([name]), name)


TASKS: dict[str, Task] = {
    "focus": Task("focus", "The image is out of focus. Bring it into focus using defocus.",
                  _focus_setup, _focus_score, _focus_baseline, ("instrument",)),
    "aberrations": Task("aberrations", "The probe is aberrated. Correct C1 and A1 below 2 nm with the corrector.",
                        _ab_setup, _ab_score, _ab_baseline, ("instrument", "corrector")),
    "zone_axis": Task("zone_axis", "The crystal is off its zone axis. Tilt alpha onto the zone axis using "
                                   "diffraction patterns from the camera.",
                      _zone_setup, _zone_score, _zone_baseline, ("instrument",)),
    "recovery": Task("recovery", "The corrector stopped responding. Get it working again and take a C1/A1 "
                                 "measurement.",
                     _rec_setup, _rec_score, _rec_baseline, ("corrector",)),
}
