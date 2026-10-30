"""Where on the sample each pixel of a scan really lands.

Real scans are not a perfect grid: the left edge is compressed after
flyback, each line jitters, the scan is slightly rotated and sheared, the
field of view is miscalibrated, and the specimen drifts while the frame is
being written. All of that is expressed as one coordinate map; rendering
then samples the sample at those points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class ScanParams:
    flyback_s: float = 0.0
    line_jitter_px: float = 0.0
    rotation_deg: float = 0.0
    shear: float = 0.0
    flyback_compression: float = 0.0
    fov_scale_error: float = 0.0


def line_jitter_m(rng: np.random.Generator, rows: int, std_m: float) -> np.ndarray:
    """Each line's x offset (m) as ``pixel_sample_coordinates`` drew it, given a fresh generator for the same
    event (the twin's ``EventRng(seed).for_event("scan", acquisition_index)``) and the frame's ``line_jitter``."""
    return rng.normal(0.0, std_m, size=(rows, 1))[:, 0]


def frame_time(rows: int, cols: int, dwell_s: float, params: ScanParams) -> float:
    return rows * (cols * dwell_s + params.flyback_s)


def pixel_sample_coordinates(
    imsize: int,
    fov_m: float,
    region: tuple[float, float, float, float],
    dwell_s: float,
    params: ScanParams,
    origin_at: Callable[[float], np.ndarray],
    t0: float,
    rng: np.random.Generator,
    truth: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sample-frame (x, y) in metres for every pixel, shape (rows, cols).

    ``origin_at(t)`` gives the sample point under the scan centre at time t
    (true stage position plus drift). ``region`` is [x0, y0, x1, y1] as
    fractions of the full field of view. ``truth``, when given, receives
    ``line_jitter``: what ``line_jitter_m`` needs to give each line's random x
    offset again (ground truth stays small however many lines a map has).
    ``rng`` must be fresh: the jitter is its first draw.
    """
    x0, y0, x1, y1 = (float(v) for v in region)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"scan_region must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {list(region)}")
    cols = max(1, int(round((x1 - x0) * imsize)))
    rows = max(1, int(round((y1 - y0) * imsize)))
    pixel = fov_m / imsize
    scale = 1.0 + params.fov_scale_error

    col = np.arange(cols, dtype=np.float64)
    row = np.arange(rows, dtype=np.float64)
    x_lin = (x0 - 0.5) * fov_m + (col + 0.5) * pixel
    if params.flyback_compression > 0:
        tau = max(1.0, 0.05 * imsize)
        x_lin = x_lin - params.flyback_compression * tau * pixel * (1.0 - np.exp(-col / tau))
    y_lin = (y0 - 0.5) * fov_m + (row + 0.5) * pixel

    xx, yy = np.meshgrid(x_lin * scale, y_lin * scale)
    if params.line_jitter_px > 0:
        xx = xx + rng.normal(0.0, params.line_jitter_px * pixel, size=(rows, 1))
        if truth is not None:
            truth["line_jitter"] = {"rows": rows, "std_m": params.line_jitter_px * pixel}
    if params.shear:
        xx = xx + params.shear * yy
    if params.rotation_deg:
        angle = np.deg2rad(params.rotation_deg)
        c, s = np.cos(angle), np.sin(angle)
        xx, yy = c * xx - s * yy, s * xx + c * yy

    line_time = cols * dwell_s + params.flyback_s
    origins = np.array([origin_at(t0 + r * line_time) for r in range(rows)])
    return xx + origins[:, 0:1], yy + origins[:, 1:2]
