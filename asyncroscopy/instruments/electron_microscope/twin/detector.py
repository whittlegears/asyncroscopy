"""Electron counting for scanning detectors.

Output is raw counts: electrons per pixel from beam current and dwell, times
the fraction reaching the detector, with Poisson shot noise, gain, offset and
read noise. There is no per-tile normalisation, so overlapping tiles agree in
brightness the way they do on the instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ELECTRON_CHARGE = 1.602176634e-19


@dataclass
class DetectorParams:
    beam_current_pa: float = 50.0
    gain: float = 1.0
    offset: float = 0.0
    read_noise: float = 0.0


def electrons_per_pixel(beam_current_pa: float, dwell_s: float) -> float:
    return beam_current_pa * 1e-12 * dwell_s / ELECTRON_CHARGE


def count(fraction: np.ndarray, dwell_s: float, params: DetectorParams, rng: np.random.Generator) -> np.ndarray:
    expected = np.clip(fraction, 0.0, None) * electrons_per_pixel(params.beam_current_pa, dwell_s)
    counts = rng.poisson(expected).astype(np.float64) * params.gain + params.offset
    if params.read_noise > 0:
        counts += rng.normal(0.0, params.read_noise, size=counts.shape)
    return np.clip(np.rint(counts), 0, np.iinfo(np.uint16).max).astype(np.uint16)
