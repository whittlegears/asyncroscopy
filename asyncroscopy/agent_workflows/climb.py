"""Image scores and trust-region maths shared by the hill-climb graphs.

Moved from notebooks 12 (aberration hill climbing) and 14 (stage-Z hill
climbing) without changing what they compute, so a graph and its notebook
propose the same next point from the same history. The one addition: the
aberration proposal can be held inside a hard limit (``max_offset_m``), as the
stage-Z proposal already was.

Scores:
- ``fft_sharpness`` comes from the ANALYSIS device's ``resolution`` command
  (finest spacing whose FFT spots stand above the noise, as a fraction of the
  Nyquist limit). It needs only a data key, so it works over MCP.
- ``contrast``, ``edge_sharpness`` and ``entropy`` need the pixels. A graph gets
  them only when it is built with ``read_image`` (a function from data key to a
  2D array, for example a Tiled reader), since MCP passes keys, not images.
"""

from __future__ import annotations

import inspect
import json
import math
from typing import Any, Callable, Optional

import numpy as np

from .tools import ToolBox, ToolCallError

PIXEL_METRICS = ("contrast", "edge_sharpness", "entropy")
METRICS = ("fft_sharpness",) + PIXEL_METRICS


# --- image scores (notebooks 12 and 14) -------------------------------------------------

def contrast(image) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    clipped = np.clip(image, low, high)
    return float(np.std(clipped) / (abs(np.mean(clipped)) + 1e-12))


def edge_sharpness(image) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    clipped = np.clip(image, low, high)
    normalized = clipped / (abs(np.mean(clipped)) + 1e-12)
    gradient_y, gradient_x = np.gradient(normalized)
    return float(np.sqrt(np.mean(gradient_x * gradient_x + gradient_y * gradient_y)))


def entropy(image, bins: int = 256) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return 0.0
    histogram, _ = np.histogram(np.clip(image, low, high), bins=bins, range=(low, high))
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(bins))


def fft_sharpness(resolution_result: dict) -> float:
    """Fraction of the Nyquist limit an image resolves (0 to 1), from ANALYSIS.resolution's JSON."""
    if resolution_result.get("resolution_angstrom") is None:
        return 0.0
    return float(min(1.0, resolution_result["nyquist_angstrom"] / resolution_result["resolution_angstrom"]))


PIXEL_SCORES = {"contrast": contrast, "edge_sharpness": edge_sharpness, "entropy": entropy}


def choose_metric(requested: Optional[str], read_image) -> str:
    """The request's metric; by default contrast (as the notebooks) when pixels can be read, else fft_sharpness."""
    if requested:
        return str(requested)
    return "contrast" if read_image is not None else "fft_sharpness"


def metric_problems(metric: str, read_image) -> list[str]:
    if metric not in METRICS:
        return [f"metric must be one of {list(METRICS)}"]
    if metric in PIXEL_METRICS and read_image is None:
        return [f"metric {metric!r} needs the pixels: build the graph with read_image, or use fft_sharpness"]
    return []


def make_scorer(box: ToolBox, metric: str, read_image: Optional[Callable[[str], Any]]):
    """An async function: data key -> score (higher is better). Raises ToolCallError on failure."""

    async def score(key: str) -> float:
        if metric == "fft_sharpness":
            result = await box.call("resolution", key=key)
            if isinstance(result, str):
                result = json.loads(result)
            value = fft_sharpness(result)
        else:
            try:
                image = read_image(key)
                if inspect.isawaitable(image):
                    image = await image
                value = PIXEL_SCORES[metric](image)
            except Exception as exc:  # a reader can fail in many ways
                raise ToolCallError("read_image", str(exc)) from exc
        if not math.isfinite(value):
            raise ToolCallError(metric, f"score for {key!r} is not finite: {value}")
        return float(value)

    return score


# --- stage Z (notebook 14) --------------------------------------------------------------

def fit_response_z(z_m, scores):
    """Quadratic fit of score against stage z (ridge-regularised, in standardised units)."""
    z_m = np.asarray(z_m, dtype=float)
    scores = np.asarray(scores, dtype=float)
    center = z_m.mean()
    scale = max(z_m.std(), 1e-12)
    x = (z_m - center) / scale
    design = np.column_stack([np.ones(len(x)), x, x * x])
    ridge = 1e-6 * np.eye(design.shape[1])
    coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ scores)

    def predict(candidate_z_m):
        candidate = np.atleast_1d(candidate_z_m)
        x = (candidate - center) / scale
        return np.column_stack([np.ones(len(x)), x, x * x]) @ coefficients

    return predict


def propose_next_z(z_m, scores, trust_radius_m, starting_z_m, max_offset_m) -> Optional[float]:
    """Best predicted z within the trust radius of the best point and within max_offset of the start, not yet measured."""
    measured_z = np.asarray(z_m, dtype=float)
    scores = np.asarray(scores, dtype=float)
    best_z = measured_z[int(np.argmax(scores))]
    lower = max(best_z - trust_radius_m, starting_z_m - max_offset_m)
    upper = min(best_z + trust_radius_m, starting_z_m + max_offset_m)
    candidates = np.linspace(lower, upper, 401)
    separation = np.min(np.abs(candidates[:, None] - measured_z[None, :]), axis=1)
    candidates = candidates[separation > max(trust_radius_m / 40, 1e-12)]
    if not len(candidates):
        return None
    predict = fit_response_z(measured_z, scores)
    return float(candidates[int(np.argmax(predict(candidates)))])


# --- aberrations (notebook 12) ----------------------------------------------------------

def fit_response_surface(offsets_m, scores):
    """Quadratic fit of score against 1 or 2 aberration components (fitted in nm)."""
    points = np.asarray(offsets_m, dtype=float) * 1e9
    scores = np.asarray(scores, dtype=float)
    center = points.mean(axis=0)
    scale = np.maximum(points.std(axis=0), 1e-6)
    normalized = (points - center) / scale
    if points.shape[1] == 2:
        x, y = normalized.T
        design = np.column_stack([np.ones(len(points)), x, y, x * x, x * y, y * y])
    else:
        x = normalized[:, 0]
        design = np.column_stack([np.ones(len(points)), x, x * x])
    ridge = 1e-6 * np.eye(design.shape[1])
    coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ scores)

    def predict(candidate_points_nm):
        candidate = np.atleast_2d(candidate_points_nm)
        z = (candidate - center) / scale
        if points.shape[1] == 2:
            x, y = z.T
            matrix = np.column_stack([np.ones(len(z)), x, y, x * x, x * y, y * y])
        else:
            x = z[:, 0]
            matrix = np.column_stack([np.ones(len(z)), x, x * x])
        return matrix @ coefficients

    return predict


def propose_next_point(offsets_m, scores, trust_radius_m, max_offset_m: Optional[float] = None):
    """Best predicted offset (m) within the trust radius of the best point, not yet measured; None if none.

    With ``max_offset_m``, candidates farther than that from zero offset are dropped
    (the notebook has no such limit; without it the result is the notebook's).
    """
    points_nm = np.asarray(offsets_m, dtype=float) * 1e9
    scores = np.asarray(scores, dtype=float)
    best = points_nm[int(np.argmax(scores))]
    radius_nm = trust_radius_m * 1e9
    predict = fit_response_surface(offsets_m, scores)
    if points_nm.shape[1] == 2:
        axis = np.linspace(-radius_nm, radius_nm, 81)
        dx, dy = np.meshgrid(axis, axis)
        inside = dx * dx + dy * dy <= radius_nm * radius_nm
        candidates = np.column_stack([best[0] + dx[inside], best[1] + dy[inside]])
    else:
        candidates = np.linspace(best[0] - radius_nm, best[0] + radius_nm, 401)[:, None]
    separation = np.linalg.norm(candidates[:, None, :] - points_nm[None, :, :], axis=2).min(axis=1)
    keep = separation > max(radius_nm / 40, 1e-6)
    if max_offset_m is not None:
        keep &= np.linalg.norm(candidates, axis=1) <= max_offset_m * 1e9
    candidates = candidates[keep]
    if not len(candidates):
        return None
    return candidates[int(np.argmax(predict(candidates)))] * 1e-9


def initial_offsets(step_m: float, components: int) -> list[list[float]]:
    """The notebook's first probes: +-step for one component, eight points on a circle for two."""
    if components == 2:
        return [[step_m * math.cos(a), step_m * math.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    return [[-step_m], [step_m]]


def next_trust_radius(trust_radius: float, improved: bool, cap: float) -> float:
    """Grow by 20% (up to ``cap``) after an improvement of more than 0.1%, else shrink to 65%."""
    return min(trust_radius * 1.20, cap) if improved else trust_radius * 0.65


def improved(score: float, previous_best: float) -> bool:
    return score > previous_best * 1.001
