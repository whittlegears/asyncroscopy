"""Electron probe from the current aberrations.

At atomic resolution the probe comes from pyTEMlib (the same code the
polycrystalline-gold twin uses). At survey pixel sizes only its width matters,
so a Gaussian whose width grows with |defocus| and astigmatism stands in.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyTEMlib.image_tools as it
import pyTEMlib.probe_tools as pt

# pyTEMlib's one Spectra300 target-aberration table is for 200 kV; the twins use it at every voltage.
TARGET_ABERRATIONS_V = 200000

CORRECTOR_TO_PYTEMLIB = {
    "C1": ("C10",),
    "A1": ("C12a", "C12b"),
    "B2": ("C21a", "C21b"),
    "A2": ("C23a", "C23b"),
    "C3": ("C30",),
    "S3": ("C32a", "C32b"),
    "A3": ("C34a", "C34b"),
    "D4": ("C41a", "C41b"),
    "B4": ("C43a", "C43b"),
    "A4": ("C45a", "C45b"),
}

# The Spectra 300 datasheet quotes 50 pm resolution.
IDEAL_PROBE_SIGMA_M = 0.05e-9 / 2.355

FOCUS_DRIFT_PERIODS_S = (1200.0, 21600.0)  # fastest and slowest swings of focus and astigmatism
FOCUS_DRIFT_WAVES = 5


class FocusDrift:
    """Focus (C1) and two-fold astigmatism (A1) that wander with the twin's clock, the way a column does as lens
    currents and the room's temperature change. Each is a seeded sum of slow sine waves, like the detector's gain
    in ``detector.DetectorDrift``. It is measured from the start: at time 0 the drift is 0, and it never moves
    further than twice ``focus_drift_m`` (or ``astigmatism_drift_m``, per A1 component) from there. Both 0: steady.

    So a focus that was right goes wrong after a while, and a long map has to decide when to focus again.
    """

    def __init__(self, focus_drift_m: float, astigmatism_drift_m: float, rng: np.random.Generator) -> None:
        if focus_drift_m < 0 or astigmatism_drift_m < 0:
            raise ValueError("focus and astigmatism drift must be >= 0")
        self.swing_m = np.array([focus_drift_m, astigmatism_drift_m, astigmatism_drift_m], dtype=np.float64)  # C1, A1a, A1b
        periods = rng.uniform(*FOCUS_DRIFT_PERIODS_S, size=(3, FOCUS_DRIFT_WAVES))
        self._frequency = 2.0 * np.pi / periods
        self._phase = rng.uniform(0.0, 2.0 * np.pi, size=(3, FOCUS_DRIFT_WAVES))
        weights = rng.uniform(0.5, 1.0, size=(3, FOCUS_DRIFT_WAVES))
        self._weight = weights / weights.sum(axis=1, keepdims=True)

    def _swing(self, t_s: float) -> np.ndarray:
        return np.sum(self._weight * np.sin(self._frequency * float(t_s) + self._phase), axis=1)

    def at(self, t_s: float) -> tuple[float, list[float]]:
        """(C1, [A1a, A1b]) in metres that the drift adds at twin time ``t_s``."""
        if not self.swing_m.any():
            return 0.0, [0.0, 0.0]
        c1, a, b = self.swing_m * (self._swing(t_s) - self._swing(0.0))
        return float(c1), [float(a), float(b)]


def target_aberrations() -> dict:
    """pyTEMlib's Spectra300 target aberrations (nm), the starting point of every twin probe."""
    return pt.get_target_aberrations("Spectra300", TARGET_ABERRATIONS_V)


def aberrations_nm(corrector_coefficients: dict | None, voltage_v: float, extra_defocus_m: float) -> dict:
    """pyTEMlib aberration dict (nm) from corrector coefficients (m) plus a defocus offset."""
    ab = target_aberrations()
    for source, destinations in CORRECTOR_TO_PYTEMLIB.items():
        values = (corrector_coefficients or {}).get(source)
        if values is None:
            continue
        for destination, value_m in zip(destinations, np.atleast_1d(values)):
            ab[destination] = float(value_m) * 1e9
    ab["C10"] = float(ab.get("C10", 0.0)) + extra_defocus_m * 1e9
    return ab


def survey_blur_sigma_m(ab_nm: dict, convergence_mrad: float, local_defocus_m: float = 0.0) -> float:
    """Gaussian width standing in for the probe when pixels are much larger than it."""
    alpha = convergence_mrad * 1e-3
    defocus = ab_nm.get("C10", 0.0) * 1e-9 - local_defocus_m
    astig = np.hypot(ab_nm.get("C12a", 0.0), ab_nm.get("C12b", 0.0)) * 1e-9
    geometric = alpha * np.hypot(defocus, astig) / 2.0
    return float(np.hypot(IDEAL_PROBE_SIGMA_M, geometric))


def atomic_probe(ab_nm: dict, size: int, fov_m: float, voltage_v: float, convergence_mrad: float) -> np.ndarray:
    """Probe intensity on a size x size grid spanning fov_m, normalised to sum 1, centred at [0, 0]."""
    key = tuple(sorted((k, round(float(v), 6)) for k, v in ab_nm.items() if isinstance(v, (int, float))))
    return _cached_probe(key, int(size), round(float(fov_m), 15), float(voltage_v), float(convergence_mrad)).copy()


@lru_cache(maxsize=16)
def _cached_probe(key: tuple, size: int, fov_m: float, voltage_v: float, convergence_mrad: float) -> np.ndarray:
    ab = dict(key)
    ab["acceleration_voltage"] = voltage_v
    ab["convergence_angle"] = convergence_mrad
    ab["wavelength"] = it.get_wavelength(voltage_v)
    ab["FOV"] = fov_m * 1e9
    probe, _a_k, _chi = pt.get_probe(ab, size, size, verbose=False)
    probe = np.abs(np.asarray(probe, dtype=np.float64))
    total = probe.sum()
    probe = probe / total if total > 0 else probe
    return np.fft.ifftshift(probe)
