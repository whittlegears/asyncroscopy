"""Electron counting for scanning detectors.

Output is raw counts: electrons per pixel from beam current and dwell, times
the fraction reaching the detector, with Poisson shot noise, gain, offset and
read noise. There is no per-tile normalisation, so overlapping tiles agree in
brightness the way they do on the instrument, unless the detector drifts:
``DetectorDrift`` moves the gain and the offset slowly with the twin's clock,
the way a real amplifier warms up, so a stitcher has to match brightness
between tiles taken minutes apart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from asyncroscopy.routines.beam import DEFAULT_BEAM_CURRENT_PA

ELECTRON_CHARGE = 1.602176634e-19
DRIFT_PERIODS_S = (600.0, 7200.0)  # the slowest and fastest swings of gain and offset
DRIFT_WAVES = 5


@dataclass
class DetectorParams:
    beam_current_pa: float = DEFAULT_BEAM_CURRENT_PA
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


class DetectorDrift:
    """Gain and offset that wander smoothly with time: each is a seeded sum of slow sine waves whose swing never
    exceeds ``gain_drift`` (a fraction of the gain) or ``offset_drift`` (counts). Both 0: the detector is steady."""

    def __init__(self, gain_drift: float, offset_drift: float, rng: np.random.Generator) -> None:
        if gain_drift < 0 or gain_drift >= 1 or offset_drift < 0:
            raise ValueError("detector gain drift must be in [0, 1) and offset drift >= 0")
        self.gain_drift, self.offset_drift = float(gain_drift), float(offset_drift)
        periods = rng.uniform(*DRIFT_PERIODS_S, size=(2, DRIFT_WAVES))
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(2, DRIFT_WAVES))
        weights = rng.uniform(0.5, 1.0, size=(2, DRIFT_WAVES))
        self._waves = (2.0 * np.pi / periods, phases, weights / weights.sum(axis=1, keepdims=True))

    def _swing(self, which: int, t_s: float) -> float:
        frequency, phase, weight = (a[which] for a in self._waves)
        return float(np.sum(weight * np.sin(frequency * float(t_s) + phase)))

    def at(self, params: DetectorParams, t_s: float) -> DetectorParams:
        """``params`` with the gain and offset the detector has at twin time ``t_s``."""
        if not (self.gain_drift or self.offset_drift):
            return params
        return replace(
            params,
            gain=params.gain * (1.0 + self.gain_drift * self._swing(0, t_s)),
            offset=params.offset + self.offset_drift * self._swing(1, t_s),
        )
