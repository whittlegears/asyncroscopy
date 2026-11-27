"""Image-quality metrics used to tune focus and aberrations.

These are the team's notebook metrics (``notebooks/12_*`` and ``14_*``), moved
here unchanged so every routine, workflow and agent tool scores images the same
way.

Not every metric suits every job. Measured on the tilt twin across defocus
(64 px, shot noise): ``contrast`` and ``edge_sharpness`` peak cleanly at true
focus; ``high_frequency_power`` has a false minimum a few blur widths out,
because shot noise keeps its high-frequency share up in blurred images; and
``entropy`` is *lowest* at focus. Focus and sessions therefore default to
``edge_sharpness`` (the stage-z notebook's choice).
"""

from __future__ import annotations

import numpy as np


def contrast(image) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    clipped = np.clip(image, low, high)
    return float(np.std(clipped) / (abs(np.mean(clipped)) + 1e-12))


def entropy(image, bins: int = 256) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return 0.0
    histogram, _ = np.histogram(np.clip(image, low, high), bins=bins, range=(low, high))
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(bins))


def high_frequency_power(image, cutoff: float = 0.20) -> float:
    centered = np.asarray(image, dtype=np.float64) - np.mean(image)
    power = np.abs(np.fft.rfft2(centered)) ** 2
    fy = np.fft.fftfreq(centered.shape[0])[:, None]
    fx = np.fft.rfftfreq(centered.shape[1])[None, :]
    high_frequency = np.hypot(fx, fy) >= cutoff
    return float(power[high_frequency].sum() / (power.sum() + 1e-12))


def edge_sharpness(image) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    clipped = np.clip(image, low, high)
    normalized = clipped / (abs(np.mean(clipped)) + 1e-12)
    gradient_y, gradient_x = np.gradient(normalized)
    return float(np.sqrt(np.mean(gradient_x * gradient_x + gradient_y * gradient_y)))


METRICS = {
    "contrast": contrast,
    "entropy": entropy,
    "high_frequency_power": high_frequency_power,
    "edge_sharpness": edge_sharpness,
}


def all_metrics(image) -> dict[str, float]:
    return {name: fn(image) for name, fn in METRICS.items()}


def metric(name: str):
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}; choose from {', '.join(METRICS)}") from None
