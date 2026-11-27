"""Autonomous routines on our Tango devices: focus, aberration tuning, measure-and-correct, ROI finding.

Each routine:

* talks only to the standard device commands (``acquire_scanned_image``,
  ``set_defocus``/``get_defocus``, ``move_stage``/``get_stage``, and the
  CORRECTOR API ``acquire_tableau``/``correct_aberration``), so it runs the
  same on a twin and on a vendor backend that passes the conformance check;
* runs every instrument operation through a *step* callback, so a caller can
  record, rate-limit or abort them;
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


# ----------------------------------------------------------------------------
# Zone axis
# ----------------------------------------------------------------------------

def _block_reduce(p: np.ndarray, max_side: int = 256) -> np.ndarray:
    f = int(np.ceil(max(p.shape) / max_side))
    if f <= 1:
        return p
    h, w = (p.shape[0] // f) * f, (p.shape[1] // f) * f
    return p[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def laue_offset(pattern, disk_fraction: float = 0.08, max_side: int = 256) -> dict:
    """How far a diffraction pattern is from on-zone.

    The direct beam is the brightest blob; the *Laue offset* is the vector from
    it to the centroid of the significant diffracted intensity outside the
    central disk, as a fraction of the pattern size. On zone the diffracted
    intensity is balanced around the direct beam (offset ~ 0); off zone it
    piles up on the Laue circle on one side. Smaller is better.

    It is independent of camera size (patterns are block-averaged to at most
    *max_side* pixels) and of the detector noise floor (only intensity more
    than 3 robust standard deviations above the background counts), so a
    1024-pixel camera with read noise gives the same answer as a clean
    128-pixel one.
    """
    from scipy import ndimage

    p = _block_reduce(np.asarray(pattern, dtype=np.float64), max_side)
    h, w = p.shape
    smooth = ndimage.gaussian_filter(p, 1.0)
    background = float(np.median(smooth))
    noise = 1.4826 * float(np.median(np.abs(smooth - background))) + 1e-12
    signal = np.clip(smooth - background - 3.0 * noise, 0.0, None)
    cy, cx = np.unravel_index(int(np.argmax(ndimage.gaussian_filter(p, max(1.0, min(h, w) / 64)))), p.shape)
    yy, xx = np.mgrid[:h, :w]
    outer = np.hypot(yy - cy, xx - cx) > disk_fraction * min(h, w)
    weights = np.sqrt(signal) * outer
    total = weights.sum()
    if total <= 0:
        return {"offset": float("nan"), "dx": 0.0, "dy": 0.0, "beam": [float(cx), float(cy)],
                "note": "no diffracted intensity above the noise floor"}
    dx = float((weights * (xx - cx)).sum() / total) / w
    dy = float((weights * (yy - cy)).sum() / total) / h
    return {"offset": float(np.hypot(dx, dy)), "dx": dx, "dy": dy, "beam": [float(cx), float(cy)]}


def align_zone_axis(instrument, fetch_array: Callable[[str], np.ndarray], resilient: Resilient, *,
                    axes: tuple[str, ...] = ("alpha",), step_deg: float = 0.1, max_offset_deg: float = 1.0,
                    max_iterations: int = 6, tolerance: float = 0.0, tilt_tolerance_deg: float = 0.02,
                    coarse_step_deg: float | None = 0.1, refine_repeats: int = 4, damping: float = 0.8,
                    bisect_iterations: int = 3, landing_attempts: int = 4, landing_repeats: int = 8,
                    approach_deg: float = 0.1,
                    step: Step = direct_step,
                    tracker: "Callable[[], dict] | None" = None) -> dict:
    """Tilt onto the zone axis using the Laue offset of camera diffraction patterns.

    The Laue offset is a 2-vector that tilting moves roughly linearly near the
    zone. So instead of a blind search, this measures how it responds to each
    tilt axis (± *step_deg*), then takes Newton steps on the part of the offset
    those axes control. With one axis, the component the other axis would fix
    is ignored, so it cannot drag the answer sideways. Each step is re-measured
    and the response refreshed (Broyden update).

    Safety: the tilt never moves more than *max_offset_deg* from the start on
    any axis; the start tilt is restored unless the controllable offset ends
    up smaller than at the start. *tracker*, when given, re-centres the region
    after every tilt.
    """
    t0 = time.time()
    index = {"alpha": 3, "beta": 4}
    for a in axes:
        if a not in index:
            raise ValueError("axes must be alpha and/or beta")
    n = len(axes)
    start = [float(v) for v in resilient.call(instrument, "get_stage")]
    trials: list[dict] = []
    tracked: list[dict] = []
    noise: dict = {}

    def tilt_to(offset, why):
        offset = np.clip(np.asarray(offset, dtype=float), -max_offset_deg, max_offset_deg)
        pos = list(start)
        for a, o in zip(axes, offset):
            pos[index[a]] = start[index[a]] + float(o)

        def do():
            resilient.call(instrument, "move_stage", pos)
            if tracker is not None:
                tracked.append(tracker())
        step("tilt", {"offset_deg": offset.tolist(), "why": why}, do)
        return offset

    def laue(offset, why, repeats: int = 1) -> np.ndarray:
        """Mean Laue vector of *repeats* patterns at the current tilt (more patterns, no extra moves)."""
        vs = []
        for _ in range(max(1, repeats)):
            key = step("acquire_diffraction", {"offset_deg": list(map(float, offset)), "why": why},
                       lambda: resilient.call(instrument, "acquire_camera_image"))
            lo = laue_offset(fetch_array(key))
            vs.append([lo["dx"], lo["dy"]])
            trials.append({"offset_deg": list(map(float, offset)), "why": why, "laue": vs[-1], "key": key})
        arr = np.array(vs)
        noise["sem"] = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else noise.get("sem")
        return arr.mean(axis=0)

    def controllable(v: np.ndarray, j: np.ndarray) -> float:
        """Norm of the part of v that the tilt axes can change (projection onto range(J))."""
        q, _ = np.linalg.qr(j)
        return float(np.linalg.norm(q.T @ v))

    status, reasons = "failed", []
    current = np.zeros(n)
    scans: list[dict] = []
    try:
        l0 = laue(current, "start", refine_repeats)
        def probe(center: np.ndarray) -> np.ndarray:
            # Keep both probes inside the tilt limit, so the finite difference is symmetric.
            center = np.clip(center, -max_offset_deg + step_deg, max_offset_deg - step_deg)
            cols = []
            for k in range(n):
                e = np.zeros(n)
                e[k] = step_deg
                lp = laue(tilt_to(center + e, f"probe +{axes[k]}"), f"probe +{axes[k]}")
                lm = laue(tilt_to(center - e, f"probe -{axes[k]}"), f"probe -{axes[k]}")
                cols.append((lp - lm) / (2 * step_deg))
            return np.column_stack(cols)                         # d(laue)/d(tilt), 2 x n

        center = np.zeros(n)
        slopes: list[np.ndarray | None] = [None] * n
        brackets: list[tuple[float, float, np.ndarray] | None] = [None] * n
        if coarse_step_deg:
            # Bracket the zone on each axis: the Laue component that this axis moves changes sign at the
            # zone. A sign change is robust to noise and to the response being non-linear far out.
            for k in range(n):
                grid = np.arange(-max_offset_deg, max_offset_deg + 1e-9, coarse_step_deg)
                vecs = []
                for g in grid:
                    c = center.copy()
                    c[k] = g
                    vecs.append(laue(tilt_to(c, f"coarse {axes[k]}"), f"coarse {axes[k]}"))
                vecs = np.array(vecs)
                slope = np.polyfit(grid, vecs, 1)[0]                 # d(laue)/d(tilt) along this axis
                if np.linalg.norm(slope) < 1e-12:
                    continue
                slopes[k] = slope
                u = slope / np.linalg.norm(slope)
                comp = vecs @ u                                      # signed controllable component
                crossings = [i for i in range(len(grid) - 1) if np.sign(comp[i]) != np.sign(comp[i + 1])
                             and comp[i] < comp[i + 1]]
                if crossings:
                    # The steepest rising crossing is the zone; interpolate between the two patterns.
                    i = max(crossings, key=lambda j: comp[j + 1] - comp[j])
                    root = grid[i] - comp[i] * (grid[i + 1] - grid[i]) / (comp[i + 1] - comp[i])
                    brackets[k] = (float(grid[i]), float(grid[i + 1]), u)
                else:
                    root = grid[int(np.argmin(np.abs(comp)))]
                center[k] = float(np.clip(root, -max_offset_deg, max_offset_deg))
                scans.append({"axis": axes[k], "grid": grid.tolist(), "component": comp.tolist(),
                              "zone_estimate": center[k]})
        # The response fitted over the whole coarse scan is far more robust than a ±step finite
        # difference, whose two tilts each carry the stage's tilt error.
        jac = np.column_stack(slopes) if all(sl is not None for sl in slopes) else probe(center)
        if np.linalg.matrix_rank(jac, tol=1e-6) < n:
            raise RuntimeError("diffraction does not respond to the tilt axes (no crystal under the beam, "
                               "or the probe step is too small)")
        # Stop when the remaining offset corresponds to less than tilt_tolerance_deg of tilt.
        tol = max(tolerance, float(np.linalg.svd(jac, compute_uv=False).min()) * tilt_tolerance_deg)
        # Near the zone the component can be step-like (the Laue circle sweeping through a row of
        # spots), so refine by its sign, which flips cleanly: bisect each bracket.
        for k in range(n):
            if brackets[k] is None:
                continue
            lo, hi, u = brackets[k]
            for _ in range(bisect_iterations):
                c = center.copy()
                c[k] = (lo + hi) / 2
                comp = float(laue(tilt_to(c, f"bisect {axes[k]}"), f"bisect {axes[k]}", refine_repeats) @ u)
                lo, hi = ((lo + hi) / 2, hi) if comp < 0 else (lo, (lo + hi) / 2)
            center[k] = (lo + hi) / 2
        bracketed = all(b is not None for b in brackets)
        if bracketed:
            # Land on the estimate from the same side every time (consistent backlash). The landing
            # carries tilt error, so measure it; if it is outside the noise band, back off and land again
            # (re-rolls the error; on a stage without tilt error a repeat landing is identical).
            for attempt in range(landing_attempts):
                tilt_to(center - approach_deg, "approach")
                current = tilt_to(center, f"land {attempt + 1}")
                lv = laue(current, f"land {attempt + 1}", landing_repeats)
                sem = noise.get("sem")
                landing_tol = max(float(np.linalg.svd(jac, compute_uv=False).min()) * tilt_tolerance_deg,
                                  2.0 * float(np.linalg.norm(sem)) if sem is not None else 0.0)
                if controllable(lv, jac) <= landing_tol:
                    break
            max_newton = 0
        else:
            # No bracket (no sign change in range): Newton steps from the measured state instead.
            current = tilt_to(center, "to zone estimate")
            lv = laue(current, "start of refinement", refine_repeats)
            max_newton = max_iterations
        stalls = 0
        smin = float(np.linalg.svd(jac, compute_uv=False).min())
        for it in range(max_newton):
            score = controllable(lv, jac)
            # Stop inside the measurement's own noise band: another move would only re-roll tilt error.
            sem = noise.get("sem")
            noise_tol = 2.0 * float(np.linalg.norm(sem)) if sem is not None else 0.0
            if score <= max(tol, noise_tol):
                break
            delta = -damping * np.linalg.lstsq(jac, lv, rcond=None)[0]
            current = tilt_to(current + delta, f"newton {it}")
            new = laue(current, f"newton {it}", refine_repeats)
            # No Broyden update: with tilt error on every move, commanded steps are not the steps taken.
            if controllable(new, jac) >= score:
                stalls += 1
                if stalls >= 2:                                  # not linear here: re-linearise in place
                    fresh = probe(current)
                    if np.linalg.matrix_rank(fresh, tol=1e-6) == n:
                        jac = fresh
                    current = tilt_to(current, "back after probing")
                    new = laue(current, "after re-probe", refine_repeats)
                    stalls = 0
            else:
                stalls = 0
            lv = new
        # Verify where the crystal actually is: more patterns, no move (a move would re-roll tilt error).
        verified = controllable((lv + laue(current, "verify", refine_repeats)) / 2, jac)
        baseline = controllable(l0, jac)
        if verified < baseline and np.any(current):
            status = "improved"
        else:
            current = tilt_to(np.zeros(n), "restore start (no verified gain)")
            status = "no_improvement"
            reasons.append(f"controllable Laue offset {verified:.4f} not below start {baseline:.4f}")
    except BaseException as exc:
        reasons.append(f"{type(exc).__name__}: {exc}")
        try:
            current = tilt_to(np.zeros(n), "restore start after error")
        except Exception as restore_exc:
            reasons.append(f"could not restore the start tilt: {restore_exc}")
        if "abort" in str(exc).lower():
            raise
        baseline = verified = float("nan")
        jac = np.zeros((2, n))
    final = [float(v) for v in resilient.call(instrument, "get_stage")]
    return {"routine": "align_zone_axis", "status": status, "reasons": reasons, "axes": list(axes),
            "start_tilt_deg": [start[index[a]] for a in axes], "final_tilt_deg": [final[index[a]] for a in axes],
            "baseline_laue_offset": baseline, "verified_laue_offset": verified,
            "response_per_deg": np.asarray(jac).tolist(), "coarse_scans": scans, "trials": trials, "tracking": tracked[-3:],
            "recoveries": list(resilient.events), "duration_s": time.time() - t0}


def image_shift_tracker(imager: Imager, *, pixel_size_m: float, tracking_sign=(1.0, 1.0),
                        max_correction_m: float | None = None) -> Callable[[], dict]:
    """Keep the region centred during a tilt series, as in the alpha-tilt notebook.

    Takes a HAADF reference now; each call re-images, registers against the
    reference, and moves the image shift by the measured offset (times
    *tracking_sign*, per axis, which absorbs the scan-to-image-shift handedness).
    A registration that is not a confident match is not applied.
    """
    from asyncroscopy.imaging.registration import MatchCriteria, center_crop, judge, locate, offset_from_center

    r = imager.resilient
    _, reference = imager.acquire()
    template = center_crop(reference, 0.6)
    sign = np.asarray(tracking_sign, dtype=float)
    limit = max_correction_m if max_correction_m is not None else 0.25 * pixel_size_m * reference.shape[1]

    def track() -> dict:
        _, image = imager.acquire()
        loc = locate(template, image)
        verdict = judge(loc, MatchCriteria(min_ncc=0.3, max_ambiguity=0.9, max_sigma_px=3.0))
        dx, dy = offset_from_center(loc, image.shape)
        correction = sign * np.array([dx, dy]) * pixel_size_m
        applied = False
        if verdict.ok and float(np.hypot(*correction)) <= limit:
            shift = np.asarray(r.call(imager.instrument, "get_image_shift"), dtype=float) + correction
            r.call(imager.instrument, "set_image_shift", shift.tolist())
            applied = True
        return {"offset_px": [dx, dy], "verdict": verdict.status, "applied": applied,
                "correction_m": correction.tolist()}

    return track
