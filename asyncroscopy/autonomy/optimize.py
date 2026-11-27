"""Trust-region response-surface search, with the safety net an autonomous run needs.

The search is the team's notebook method (``hill_climb_aberration`` in
``notebooks/12_Digital_twin_Aberration_Hill_Climbing.ipynb``): measure a
baseline and a ring of initial points, fit a quadratic to *every* measured
point, measure where the fit predicts the best score inside a trust radius,
grow the radius after an improvement and shrink it otherwise, and stop when
the radius falls below a minimum step.

What this module adds for unattended use:

* **hard limits**: no trial is applied further than ``max_offset`` from the
  start, per component (the notebook's ``MAX_Z_OFFSET`` made general);
* **verification**: after moving to the best point, it is measured again
  (``verify_repeats`` times). If the verified score does not beat a fresh
  baseline by ``min_gain`` (relative), the start setting is restored and the
  result says ``no_improvement``, so noise is never reported as progress;
* **always restores** a known setting: on success the best one, on any error
  or abort the start one (when it can still talk to the instrument);
* every trial is recorded, and every instrument operation runs through a
  *step* callback, so the supervision queue records, pauses and aborts it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

Step = Callable[[str, dict, Callable[[], Any]], Any]


def direct_step(name: str, args: dict, fn: Callable[[], Any]) -> Any:
    return fn()


@dataclass
class SearchConfig:
    step: float                          # initial trust radius / ring size, in setting units
    components: int = 1
    max_iterations: int = 8
    min_step: float | None = None        # stop when the trust radius falls below this (default step / 16)
    max_offset: float | None = None      # hard limit on |offset| per component from the start (default 4 * step)
    verify_repeats: int = 2
    min_gain: float = 0.002              # verified relative gain needed to keep the change
    improve_factor: float = 1.001        # the notebook's "previous_best * 1.001"


@dataclass
class SearchResult:
    status: str                          # improved | no_improvement | aborted | failed
    best_offset: list[float]
    baseline_score: float
    best_score: float
    verified_score: float | None
    applied_offset: list[float]
    trials: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def fit_quadratic(points: np.ndarray, scores: np.ndarray):
    """Quadratic response surface over 1 or 2 components (the notebook's ridge fit)."""
    center = points.mean(axis=0)
    scale = np.maximum(points.std(axis=0), 1e-12)

    def design(p):
        z = (np.atleast_2d(p) - center) / scale
        if points.shape[1] == 2:
            x, y = z.T
            return np.column_stack([np.ones(len(z)), x, y, x * x, x * y, y * y])
        x = z[:, 0]
        return np.column_stack([np.ones(len(z)), x, x * x])

    a = design(points)
    coefficients = np.linalg.solve(a.T @ a + 1e-6 * np.eye(a.shape[1]), a.T @ scores)
    return lambda p: design(p) @ coefficients


def propose(points: np.ndarray, scores: np.ndarray, radius: float, limit: float) -> np.ndarray | None:
    best = points[int(np.argmax(scores))]
    predict = fit_quadratic(points, scores)
    if points.shape[1] == 2:
        axis = np.linspace(-radius, radius, 81)
        dx, dy = np.meshgrid(axis, axis)
        inside = dx * dx + dy * dy <= radius * radius
        candidates = np.column_stack([best[0] + dx[inside], best[1] + dy[inside]])
    else:
        candidates = np.linspace(best[0] - radius, best[0] + radius, 401)[:, None]
    candidates = candidates[np.all(np.abs(candidates) <= limit + 1e-15, axis=1)]
    if not len(candidates):
        return None
    separation = np.linalg.norm(candidates[:, None, :] - points[None, :, :], axis=2).min(axis=1)
    candidates = candidates[separation > max(radius / 40, 1e-15)]
    if not len(candidates):
        return None
    return candidates[int(np.argmax(predict(candidates)))]


def search(apply: Callable[[np.ndarray], None], measure: Callable[[], dict], score_key: str,
           cfg: SearchConfig, *, step: Step = direct_step, label: str = "setting",
           should_stop: Callable[[], bool] | None = None) -> SearchResult:
    """Maximise ``measure()[score_key]`` over offsets from the current setting.

    *apply(offset)* sets the instrument to start + offset (absolute offsets, so
    a lost increment can't accumulate). *measure()* acquires and returns a
    dict of metrics (and optionally ``key``/``image``).
    """
    t0 = time.time()
    n = cfg.components
    min_step = cfg.min_step if cfg.min_step is not None else cfg.step / 16
    limit = cfg.max_offset if cfg.max_offset is not None else 4 * cfg.step
    current = np.zeros(n)
    trials: list[dict] = []

    def go(offset: np.ndarray, why: str) -> None:
        nonlocal current
        offset = np.clip(np.asarray(offset, dtype=float), -limit, limit)
        if np.any(offset != current):
            step(f"apply {label}", {"offset": offset.tolist(), "why": why}, lambda o=offset: apply(o))
            current = offset

    def measure_at(offset: np.ndarray, why: str) -> float:
        go(offset, why)
        m = step(f"measure {label}", {"offset": current.tolist(), "why": why}, measure)
        score = float(m[score_key])
        trials.append({"offset": current.tolist(), "why": why, "score": score,
                       "metrics": {k: v for k, v in m.items() if isinstance(v, (int, float))},
                       "key": m.get("key")})
        return score

    try:
        baseline = measure_at(np.zeros(n), "baseline")
        ring = ([cfg.step * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
                if n == 2 else [np.array([-cfg.step]), np.array([cfg.step])])
        for i, p in enumerate(ring):
            measure_at(p, f"initial {i}")
            if should_stop and should_stop():
                raise InterruptedError("stopped")

        radius = cfg.step
        for it in range(cfg.max_iterations):
            pts = np.array([t["offset"] for t in trials])
            scs = np.array([t["score"] for t in trials])
            proposal = propose(pts, scs, radius, limit)
            if proposal is None:
                break
            previous_best = scs.max()
            s = measure_at(proposal, f"model {it}")
            radius = min(radius * 1.20, cfg.step * 2) if s > previous_best * cfg.improve_factor else radius * 0.65
            if radius < min_step:
                break
            if should_stop and should_stop():
                raise InterruptedError("stopped")

        scs = np.array([t["score"] for t in trials])
        best = np.array(trials[int(np.argmax(scs))]["offset"])
        best_score = float(scs.max())

        # Verify: a fresh baseline and repeats at the best point, so noise is not progress.
        fresh_base = np.mean([measure_at(np.zeros(n), "verify baseline") for _ in range(cfg.verify_repeats)])
        verified = np.mean([measure_at(best, "verify best") for _ in range(cfg.verify_repeats)])
        if not np.any(best) or verified <= fresh_base * (1 + cfg.min_gain):
            go(np.zeros(n), "restore start (no verified gain)")
            return SearchResult("no_improvement", best.tolist(), float(baseline), best_score, float(verified),
                                current.tolist(), trials,
                                [f"verified {verified:.6g} vs baseline {fresh_base:.6g}: below the "
                                 f"{cfg.min_gain:.1%} gain needed; start setting restored"], time.time() - t0)
        return SearchResult("improved", best.tolist(), float(fresh_base), best_score, float(verified),
                            current.tolist(), trials, [], time.time() - t0)
    except BaseException as exc:
        status = "aborted" if isinstance(exc, InterruptedError) or "abort" in str(exc).lower() else "failed"
        reasons = [f"{type(exc).__name__}: {exc}"]
        try:
            go(np.zeros(n), "restore start after error")
        except Exception as restore_exc:
            reasons.append(f"could not restore the start setting: {restore_exc}")
        result = SearchResult(status, [0.0] * n, float(trials[0]["score"]) if trials else float("nan"),
                              float(max(t["score"] for t in trials)) if trials else float("nan"), None,
                              current.tolist(), trials, reasons, time.time() - t0)
        if status == "aborted" and not isinstance(exc, InterruptedError):
            raise                       # let the queue mark the action aborted
        return result
