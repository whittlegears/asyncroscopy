"""Image sharpness and bounded focus searches, as pure functions (no Tango).

A focus routine is a loop: set a value, take an image, score it, choose the
next value. Only the scoring and the choosing live here; the ``Tuner`` job
(asyncroscopy/instruments/electron_microscope/tuner.py) owns the instrument,
the step limits, the dose and the abort. Every search takes an ``evaluate(x)``
callback that sets the value, images and returns the score, so the same code
runs on the twin, on the instrument, or on a function in a test.

Scores are "higher is sharper". Over-focus and under-focus blur the same way,
so the score near focus is a symmetric peak; the searches fit a parabola to
its top rather than trusting the single best image, which makes the estimate
much finer than the sampling and robust to shot noise.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

# ----------------------------------------------------------------------
# Sharpness metrics: image (2-D array) -> float, higher is sharper
# ----------------------------------------------------------------------


def _as_image(image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3:
        raise ValueError(f"a sharpness metric needs a 2-D image of at least 3 x 3 pixels, got shape {array.shape}")
    return array


def normalized_variance(image: ArrayLike) -> float:
    """Variance over mean squared: independent of beam current and gain."""
    array = _as_image(image)
    mean = float(array.mean())
    return float(array.var() / (mean * mean)) if mean > 0 else 0.0


def gradient_energy(image: ArrayLike) -> float:
    """Mean squared gradient (Tenengrad-like) over mean squared, with the shot-noise part removed.

    Poisson noise adds about ``2 * mean`` to the squared difference of two
    neighbouring pixels whatever the focus; subtracting it keeps a dim, noisy
    image from scoring as sharp.
    """
    array = _as_image(image)
    mean = float(array.mean())
    if mean <= 0:
        return 0.0
    dx = np.diff(array, axis=1)
    dy = np.diff(array, axis=0)
    energy = 0.5 * (float((dx * dx).mean()) + float((dy * dy).mean()))
    return max(0.0, energy - 2.0 * mean) / (mean * mean)


def smoothed_gradient_energy(image: ArrayLike) -> float:
    """Gradient energy of the image after a 3 x 3 box filter, which averages shot noise down first."""
    array = _as_image(image)
    kernel = np.ones(3) / 3.0
    smooth = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, array)
    smooth = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, smooth)
    mean = float(smooth.mean())
    if mean <= 0:
        return 0.0
    dx = np.diff(smooth, axis=1)
    dy = np.diff(smooth, axis=0)
    # The box filter leaves 1/9 of the Poisson variance, correlated over 3 pixels: about 2/27 * mean per difference.
    energy = 0.5 * (float((dx * dx).mean()) + float((dy * dy).mean()))
    return max(0.0, energy - 2.0 * mean / 27.0) / (mean * mean)


METRICS: dict[str, Callable[[np.ndarray], float]] = {
    "normalized_variance": normalized_variance,
    "gradient_energy": gradient_energy,
    "smoothed_gradient_energy": smoothed_gradient_energy,
}


def sharpness(image: ArrayLike, metric: str = "normalized_variance") -> float:
    """Score ``image`` with the named metric (see METRICS)."""
    try:
        return METRICS[metric](image)
    except KeyError:
        raise ValueError(f"unknown sharpness metric {metric!r}; known: {sorted(METRICS)}") from None


# ----------------------------------------------------------------------
# Fitting the top of a peak
# ----------------------------------------------------------------------


def parabola_vertex(xs: ArrayLike, ys: ArrayLike) -> float | None:
    """x of the top of the least-squares parabola through (xs, ys); None if it opens upwards or is flat."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 3 or np.ptp(x) == 0:
        return None
    centre, scale = float(x.mean()), float(np.ptp(x))
    u = (x - centre) / scale
    a, b, _c = np.polyfit(u, y, 2)
    if not a < 0:
        return None
    return centre + scale * float(-b / (2.0 * a))


def peak_estimate(xs: ArrayLike, ys: ArrayLike) -> float:
    """Where the peak of a sampled score lies: the vertex of a parabola through the best sample and its
    neighbours (in x order), kept within those neighbours; the best sample itself when no parabola fits."""
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x = np.asarray(xs, dtype=np.float64)[order]
    y = np.asarray(ys, dtype=np.float64)[order]
    best = int(np.argmax(y))
    lo, hi = max(0, best - 1), min(len(x), best + 2)
    if hi - lo < 3:  # the best sample is at an end: widen inwards
        lo, hi = max(0, min(lo, len(x) - 3)), min(len(x), max(hi, 3))
    vertex = parabola_vertex(x[lo:hi], y[lo:hi]) if hi - lo >= 3 else None
    if vertex is None:
        return float(x[best])
    return float(min(max(vertex, x[lo]), x[hi - 1]))


# ----------------------------------------------------------------------
# Searches
# ----------------------------------------------------------------------


def sweep(evaluate: Callable[[float], float], low: float, high: float, points: int) -> tuple[list[float], list[float]]:
    """Evaluate ``points`` evenly spaced values from ``low`` to ``high`` in order (one direction, so
    hysteresis acts the same way on every point); returns (xs, scores)."""
    if points < 2:
        raise ValueError("a sweep needs at least 2 points")
    xs = [float(v) for v in np.linspace(low, high, points)]
    return xs, [float(evaluate(x)) for x in xs]


def search_1d(
    evaluate: Callable[[float], float],
    centre: float,
    half_range: float,
    coarse_points: int = 9,
    fine_points: int = 7,
    rounds: int = 2,
    tolerance: float = 0.0,
    fine_half_range: float | None = None,
    shrink: float = 1.0,
) -> dict:
    """Find the sharpest value within ``centre +/- half_range``.

    A coarse sweep over the whole range finds the peak to within one spacing.
    Each refine round then sweeps ``fine_points`` over +/- ``fine_half_range``
    (default: one coarse spacing) around the current estimate and fits a
    parabola to the whole sweep. The window stays the same size by default
    (``shrink`` 1): near focus the score is flat, so the flanks carry the
    information, and a window re-centred on the last estimate is symmetric
    about the peak, which cancels the parabola's misfit. It stops early once
    the window is smaller than ``tolerance``. Nothing outside the range is ever
    evaluated. Returns {"best", "evaluations", "samples": [[x, score], ...]}.
    """
    if half_range <= 0:
        raise ValueError("half_range must be > 0")
    low_bound, high_bound = centre - half_range, centre + half_range
    samples: list[tuple[float, float]] = []

    xs, ys = sweep(evaluate, low_bound, high_bound, coarse_points)
    samples += zip(xs, ys)
    estimate = peak_estimate(xs, ys)
    window = float(fine_half_range) if fine_half_range else 2.0 * half_range / (coarse_points - 1)
    for _ in range(max(0, rounds)):
        if window < tolerance:
            break
        low, high = max(low_bound, estimate - window), min(high_bound, estimate + window)
        if high - low <= 0:
            break
        xs, ys = sweep(evaluate, low, high, fine_points)
        samples += zip(xs, ys)
        # Fit the whole fine sweep: it spans the peak symmetrically, which cancels most of the noise.
        vertex = parabola_vertex(xs, ys)
        estimate = float(min(max(vertex, low), high)) if vertex is not None else peak_estimate(xs, ys)
        window *= float(shrink)
    return {"best": estimate, "evaluations": len(samples), "samples": [[x, y] for x, y in samples]}


def search_2d(
    evaluate: Callable[[float, float], float],
    centre: tuple[float, float],
    half_range: float,
    passes: int = 2,
    **search_options: Any,
) -> dict:
    """Find the sharpest (a, b) within ``centre +/- half_range`` per axis: alternate 1-D searches along a
    then b, ``passes`` times, each pass around the last estimate with half the range. For two-fold
    astigmatism the score depends on |A1 + stigmator|, so each axis has one peak with the other held."""
    a, b = float(centre[0]), float(centre[1])
    a_bounds = (a - half_range, a + half_range)
    b_bounds = (b - half_range, b + half_range)
    evaluations = 0
    history = []
    span = float(half_range)
    for _ in range(max(1, passes)):
        for axis in (0, 1):
            here, bounds = (a, a_bounds) if axis == 0 else (b, b_bounds)
            # Keep each pass inside the original box.
            low, high = max(bounds[0], here - span), min(bounds[1], here + span)
            mid, half = (low + high) / 2.0, (high - low) / 2.0
            if axis == 0:
                found = search_1d(lambda v: evaluate(v, b), mid, half, **search_options)
                a = found["best"]
            else:
                found = search_1d(lambda v: evaluate(a, v), mid, half, **search_options)
                b = found["best"]
            evaluations += found["evaluations"]
            history.append({"axis": "ab"[axis], "best": found["best"], "evaluations": found["evaluations"]})
        span /= 2.0
    return {"best": [a, b], "evaluations": evaluations, "history": history}


def search_1d_evaluations(coarse_points: int = 9, fine_points: int = 7, rounds: int = 2) -> int:
    """Most images ``search_1d`` takes with these options (for dose and step estimates)."""
    return int(coarse_points) + max(0, int(rounds)) * int(fine_points)


def search_2d_evaluations(passes: int = 2, **search_options: int) -> int:
    """Most images ``search_2d`` takes with these options."""
    return 2 * max(1, int(passes)) * search_1d_evaluations(**search_options)


def error_grew(before: float, after: float, tolerance: float = 0.0) -> bool:
    """True when a correction made the error larger than before (beyond ``tolerance``): a wrong-sign
    or wrong-unit correction, which must stop the loop rather than be repeated."""
    return math.isfinite(before) and (not math.isfinite(after) or abs(after) > abs(before) + abs(tolerance))
