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

# CEOS corrector names (metres) -> pyTEMlib names (nanometres).
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

# Diffraction-limited probe width (Gaussian sigma) of an aberration-corrected
# 300 kV probe; the Spectra 300 datasheet quotes 50 pm resolution.
IDEAL_PROBE_SIGMA_M = 0.05e-9 / 2.355


def aberrations_nm(corrector_coefficients: dict | None, voltage_v: float, extra_defocus_m: float) -> dict:
    """pyTEMlib aberration dict (nm) from corrector coefficients (m) plus a defocus offset."""
    ab = pt.get_target_aberrations("Spectra300", int(voltage_v))
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
