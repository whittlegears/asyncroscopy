"""Autonomous routines on our Tango devices: focus, aberration tuning, measure-and-correct, ROI finding.

Each routine:

* talks only to the standard device commands (``acquire_scanned_image``,
  ``set_defocus``/``get_defocus``, ``move_stage``/``get_stage``, and the
  CORRECTOR API ``acquire_tableau``/``correct_aberration``), so it runs the
  same on a twin and on a vendor backend that passes the conformance check;
* runs every instrument operation through a *step* callback (the supervision
  queue records, pauses and aborts them, and applies the hard limits);
* survives a dropped device link: a failure diagnosed as ``unreachable``,
  ``link_down`` or ``timeout`` triggers the automatic recovery rungs
  (reconnect, Init) and one retry, and is recorded in the result;
* ends in a known state (best verified setting, or the start setting).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from asyncroscopy.autonomy.metrics import all_metrics, metric
from asyncroscopy.autonomy.optimize import SearchConfig, Step, direct_step, search
from asyncroscopy.health.monitor import HealthMonitor, diagnose, recover

RECOVERABLE = ("unreachable", "link_down", "timeout")


class Resilient:
    """Calls a device command; on a recoverable failure, recovers the device and retries once."""

    def __init__(self, monitor: HealthMonitor | None = None, allowed: tuple[str, ...] = ("reconnect", "init")):
        self.monitor = monitor
        self.allowed = allowed
        self.events: list[dict] = []

    def call(self, proxy, command: str, *args):
        try:
            return proxy.command_inout(command, *args) if args else proxy.command_inout(command)
        except Exception as exc:
            kind = diagnose(exc)
            device = _name(proxy)
            if self.monitor is None or kind not in RECOVERABLE or not device:
                raise
            outcome = recover(self.monitor, device, allowed=self.allowed, settle_s=10.0)
            self.events.append({"t": time.time(), "device": device, "command": command, "failure": kind,
                                "recovery": outcome["status"], "by": outcome.get("by")})
            if outcome["status"] not in ("recovered", "healthy"):
                raise RuntimeError(f"{command} on {device} failed ({kind}) and recovery did not succeed: "
                                   f"{outcome.get('message', outcome['status'])}") from exc
            return proxy.command_inout(command, *args) if args else proxy.command_inout(command)


def _name(proxy) -> str:
    try:
        return proxy.dev_name()
    except Exception:
        try:
            return proxy.name()
        except Exception:
            return ""


@dataclass
class Imager:
    """Acquire and score images with the instrument's scanning detector."""

    instrument: Any
    fetch_array: Callable[[str], np.ndarray]
    resilient: Resilient
    detector: str = "haadf"

    def acquire(self) -> tuple[str, np.ndarray]:
        key = self.resilient.call(self.instrument, "acquire_scanned_image", [self.detector])
        image = self.fetch_array(key)
        if image is None:
            raise RuntimeError(f"acquisition {key!r} has no 2-D image")
        return key, np.asarray(image, dtype=np.float64)

    def measure(self) -> dict:
        key, image = self.acquire()
        return {**all_metrics(image), "key": key}


# ----------------------------------------------------------------------------
# Focus
# ----------------------------------------------------------------------------

def focus(imager: Imager, *, mode: str = "defocus", step_m: float = 10e-9, max_offset_m: float | None = None,
          score: str = "edge_sharpness", max_iterations: int = 8, step: Step = direct_step) -> dict:
    """Find the best focus by defocus (``set_defocus``) or by stage z (``move_stage``)."""
    metric(score)
    r = imager.resilient
    inst = imager.instrument
    if mode == "defocus":
        start = float(r.call(inst, "get_defocus"))

        def apply(offset):
            r.call(inst, "set_defocus", float(start + offset[0]))
    elif mode == "stage_z":
        start_pos = [float(v) for v in r.call(inst, "get_stage")]
        start = start_pos[2]

        def apply(offset):
            pos = list(start_pos)
            pos[2] = float(start + offset[0])
            r.call(inst, "move_stage", pos)
    else:
        raise ValueError("mode must be 'defocus' or 'stage_z'")
    cfg = SearchConfig(step=step_m, components=1, max_iterations=max_iterations,
                       max_offset=max_offset_m if max_offset_m is not None else 6 * step_m)
    res = search(apply, imager.measure, score, cfg, step=step, label=f"focus ({mode})")
    out = res.to_json()
    out.update(routine="focus", mode=mode, start_value_m=start,
               final_value_m=start + res.applied_offset[0], recoveries=list(r.events))
    return out


# ----------------------------------------------------------------------------
# Aberrations
# ----------------------------------------------------------------------------

ABERRATION_COMPONENTS = {"C1": 1, "C3": 1, "A1": 2, "A2": 2, "B2": 2, "S3": 2, "A3": 2, "WD": 2}


def tune_aberration(imager: Imager, corrector, name: str, *, step_m: float, components: int | None = None,
                    max_offset_m: float | None = None, score: str = "contrast", max_iterations: int = 8,
                    step: Step = direct_step) -> dict:
    """Image-based tuning of one aberration with the corrector (the notebook 12 method, made safe)."""
    metric(score)
    n = components or ABERRATION_COMPONENTS.get(name, 2)
    r = imager.resilient
    applied = np.zeros(n)

    def apply(offset):
        nonlocal applied
        delta = np.asarray(offset, dtype=float) - applied
        if np.any(delta):
            payload = " ".join([name, *[f"{v:.12g}" for v in delta]])
            r.call(corrector, "correct_aberration", payload)
            applied = np.asarray(offset, dtype=float).copy()

    cfg = SearchConfig(step=step_m, components=n, max_iterations=max_iterations,
                       max_offset=max_offset_m if max_offset_m is not None else 4 * step_m)
    res = search(apply, imager.measure, score, cfg, step=step, label=f"{name}")
    out = res.to_json()
    out.update(routine="tune_aberration", aberration=name, recoveries=list(r.events))
    return out


def _aberrations(raw: str) -> dict[str, list[float]]:
    data = json.loads(raw)
    result = data.get("result", data)
    return {k: [float(x) for x in v] for k, v in (result.get("aberrations") or {}).items()}


def measure_and_correct(corrector, *, thresholds_m: dict[str, float], resilient: Resilient | None = None,
                        tableau: tuple[str, float] = ("Fast", 18.0), gain: float = 0.8, max_rounds: int = 4,
                        max_correction_m: dict[str, float] | None = None, step: Step = direct_step) -> dict:
    """Corrector workflow: measure a tableau, correct what is above threshold, measure again.

    A correction that makes an aberration *larger* is undone and that
    aberration is left alone, so a sign or calibration mistake in a backend
    cannot run away. Corrections larger than ``max_correction_m`` are refused.
    """
    r = resilient or Resilient()
    limits = dict(max_correction_m or {})
    rounds = []
    frozen: set[str] = set()
    t0 = time.time()

    def measure() -> dict[str, list[float]]:
        raw = step("acquire_tableau", {"type": tableau[0], "angle": tableau[1]},
                   lambda: r.call(corrector, "acquire_tableau", f"{tableau[0]} {tableau[1]:g}"))
        return _aberrations(raw)

    current = measure()
    start = current
    for rnd in range(max_rounds):
        over = {n: v for n, v in current.items() if n in thresholds_m and n not in frozen
                and float(np.hypot(*v) if len(v) == 2 else abs(v[0])) > thresholds_m[n]}
        if not over:
            break
        applied = {}
        for name, values in over.items():
            correction = [gain * v for v in values]
            size = float(np.linalg.norm(correction))
            if name in limits and size > limits[name]:
                frozen.add(name)
                applied[name] = {"refused": f"correction {size:.3g} m exceeds limit {limits[name]:.3g} m"}
                continue
            payload = " ".join([name, *[f"{v:.12g}" for v in correction]])
            step("correct_aberration", {"name": name, "values": correction},
                 lambda p=payload: r.call(corrector, "correct_aberration", p))
            applied[name] = {"correction": correction}
        after = measure()
        for name, info in applied.items():
            if "correction" not in info:
                continue
            before_mag, after_mag = np.linalg.norm(current[name]), np.linalg.norm(after.get(name, current[name]))
            if after_mag > before_mag * 1.05:
                undo = " ".join([name, *[f"{-v:.12g}" for v in info["correction"]]])
                step("correct_aberration", {"name": name, "undo": True},
                     lambda p=undo: r.call(corrector, "correct_aberration", p))
                frozen.add(name)
                info["undone"] = f"{name} grew from {before_mag:.3g} to {after_mag:.3g} m"
        rounds.append({"round": rnd + 1, "before": current, "applied": applied})
        current = measure() if any("undone" in i for i in applied.values()) else after

    remaining = {n: float(np.linalg.norm(current[n])) for n in thresholds_m if n in current}
    within = {n: remaining[n] <= thresholds_m[n] for n in remaining}
    return {"routine": "measure_and_correct", "status": "corrected" if all(within.values()) else "partial",
            "start": start, "final": current, "within_threshold": within, "frozen": sorted(frozen),
            "rounds": rounds, "recoveries": list(r.events), "duration_s": time.time() - t0}


# ----------------------------------------------------------------------------
# Regions of interest
# ----------------------------------------------------------------------------

def find_regions(image, n: int = 5, tile: int | None = None, min_separation_tiles: float = 1.5) -> list[dict]:
    """Rank tiles of a survey image by structure (contrast × entropy); return the best *n*, spread out."""
    a = np.asarray(image, dtype=np.float64)
    h, w = a.shape
    t = tile or max(8, min(h, w) // 8)
    tiles = []
    for r0 in range(0, h - t + 1, t // 2):
        for c0 in range(0, w - t + 1, t // 2):
            patch = a[r0:r0 + t, c0:c0 + t]
            if patch.std() <= 0:
                continue
            score = metric("contrast")(patch) * (0.25 + metric("entropy")(patch))
            tiles.append({"row": r0 + t / 2, "col": c0 + t / 2, "score": float(score)})
    tiles.sort(key=lambda d: d["score"], reverse=True)
    picked: list[dict] = []
    for cand in tiles:
        if all(np.hypot(cand["row"] - p["row"], cand["col"] - p["col"]) >= min_separation_tiles * t for p in picked):
            picked.append(cand)
        if len(picked) >= n:
            break
    return picked
