"""How far out in angle the probe is free of aberrations, from a CEOS tableau.

Coefficients are the [x, y] pairs CEOS returns, in meters (C1, C3 and C5 may
be given as [x]). Angles are in radians unless a name ends in _mrad.

The aberration function chi is written in the notation the CEOS software
reports (Uhlemann & Haider, Ultramicroscopy 72 (1998) 109). With the complex
angle w = theta * exp(i*phi) and each coefficient read as x + iy:

    chi(w) = (2 pi / lambda) * Re{ 1/2 w w* C1 + 1/2 w*^2 A1 + w^2 w* B2 + 1/3 w*^3 A2
                                   + 1/4 (w w*)^2 C3 + w^3 w* S3 + 1/4 w*^4 A3
                                   + w^3 w*^2 B4 + w^4 w* D4 + 1/5 w*^5 A4
                                   + 1/6 (w w*)^3 C5 + 1/6 w*^6 A5 }

The quarter-wave rule says the phase error across the probe-forming aperture
should stay below pi/4. The aberration-free angle is the largest aperture
semi-angle for which |chi| < pi/4 at every azimuth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.constants import c, e, h, m_e

QUARTER_WAVE = math.pi / 4
# The angles searched. The result is a lower bound, at most one step low.
MAX_ANGLE_MRAD = 100.0  # far beyond any STEM convergence angle
ANGLE_STEP_MRAD = 0.05
N_AZIMUTHS = 360  # the highest symmetry is six-fold (A5); 1 degree steps resolve it

# Haider notation: name -> (power of w, power of conj(w), prefactor).
TERMS: dict[str, tuple[int, int, float]] = {
    "C1": (1, 1, 1 / 2),
    "A1": (0, 2, 1 / 2),
    "B2": (2, 1, 1.0),
    "A2": (0, 3, 1 / 3),
    "C3": (2, 2, 1 / 4),
    "S3": (3, 1, 1.0),
    "A3": (0, 4, 1 / 4),
    "B4": (3, 2, 1.0),
    "D4": (4, 1, 1.0),
    "A4": (0, 5, 1 / 5),
    "C5": (3, 3, 1 / 6),
    "A5": (0, 6, 1 / 6),
}

# Without the third-order terms the angle is overestimated. A Fast tableau
# measures only up to second order, so it cannot be scored.
REQUIRED = ("C1", "A1", "A2", "B2", "C3", "A3", "S3")


def electron_wavelength_m(accelerating_voltage_v: float) -> float:
    """Relativistic electron wavelength in meters."""
    energy = e * float(accelerating_voltage_v)
    return h / math.sqrt(2 * m_e * energy * (1 + energy / (2 * m_e * c**2)))


def _xy(name: str, values: list[float]) -> tuple[float, float]:
    """A coefficient as (x, y). Only the round aberrations (p == q) may omit y."""
    p, q, _ = TERMS[name]
    if len(values) == 2:
        return float(values[0]), float(values[1])
    if len(values) == 1 and p == q:
        return float(values[0]), 0.0
    raise ValueError(f"{name} needs an [x, y] pair, got {list(values)}")


def chi(aberrations: dict[str, list[float]], wavelength_m: float, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Aberration function in radians at polar angles ``theta`` and ``phi`` (radians)."""
    w = np.asarray(theta) * np.exp(1j * np.asarray(phi))
    total = np.zeros(w.shape, dtype=complex)
    for name, values in aberrations.items():
        if name in TERMS:  # other names CEOS reports (WD, a beam tilt) are not aberrations of the probe
            p, q, prefactor = TERMS[name]
            x, y = _xy(name, values)
            total += prefactor * (x + 1j * y) * w**p * np.conj(w) ** q
    return (2 * math.pi / wavelength_m) * total.real


def _free_angle(chi_grid: np.ndarray, theta: np.ndarray) -> float:
    """The angle just inside the first theta where any azimuth reaches pi/4.

    The aperture must be clear all the way out, so a later dip back below pi/4
    does not count. chi is 0 at theta = 0, so the first crossing is never there.
    """
    reaches = (np.abs(chi_grid) >= QUARTER_WAVE).any(axis=1)
    return float(theta[-1]) if not reaches.any() else float(theta[reaches.argmax() - 1])


@dataclass
class CorrectionScore:
    aberration_free_angle_mrad: float
    target_mrad: float | None
    passed: bool | None
    gain_if_corrected_mrad: dict[str, float]
    best_next_correction: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def correction_score(
    aberrations: dict[str, list[float]],
    accelerating_voltage_v: float,
    target_mrad: float | None = None,
) -> CorrectionScore:
    """Score CEOS aberrations against the quarter-wave rule.

    ``target_mrad`` is the convergence semi-angle in use; ``passed`` is None without it.
    For each aberration, ``gain_if_corrected_mrad`` is how much the angle would grow
    if that one aberration were corrected to zero. A negative gain means it currently
    helps (for example defocus balancing C3). Raises ValueError for a Fast tableau.
    """
    missing = [name for name in REQUIRED if name not in aberrations]
    if missing:
        raise ValueError(
            f"Cannot score: {', '.join(missing)} missing. A Fast tableau measures only up to "
            "second order; acquire a Standard or Enhanced tableau first."
        )
    wavelength = electron_wavelength_m(accelerating_voltage_v)
    theta = np.arange(0.0, MAX_ANGLE_MRAD + ANGLE_STEP_MRAD / 2, ANGLE_STEP_MRAD)[:, None] * 1e-3
    phi = np.linspace(0.0, 2 * math.pi, N_AZIMUTHS, endpoint=False)[None, :]

    # chi is a sum of one term per aberration, so each term is computed once and
    # "corrected to zero" means subtracting it.
    terms = {name: chi({name: values}, wavelength, theta, phi) for name, values in aberrations.items() if name in TERMS}
    total = sum(terms.values())
    angle = _free_angle(total, theta[:, 0])
    gains = {name: round((_free_angle(total - term, theta[:, 0]) - angle) * 1e3, 2) for name, term in terms.items()}
    best = max(gains, key=gains.get)

    angle_mrad = round(angle * 1e3, 2)
    return CorrectionScore(
        aberration_free_angle_mrad=angle_mrad,
        target_mrad=target_mrad,
        passed=None if target_mrad is None else angle_mrad >= target_mrad,
        gain_if_corrected_mrad=gains,
        best_next_correction=best if gains[best] > 0 else None,
    )


# CEOS (Haider) name -> pyTEMlib (Krivanek) names, factor, sign of y.
# Matching the two forms term by term: Haider writes p * (x + iy) * w^k * conj(w)^l;
# pyTEMlib writes theta^(n+1)/(n+1) * (C_a cos(m phi) + C_b sin(m phi)) with n + 1 = k + l.
# So C_a = (n+1) p x, and C_b = (n+1) p y for conj(w) terms (A1..A5) but
# -(n+1) p y for w terms (B2, S3, B4, D4), whose phase turns the other way.
_TO_PYTEMLIB: dict[str, tuple[tuple[str, ...], float, float]] = {
    "C1": (("C10",), 1.0, 1.0),
    "A1": (("C12a", "C12b"), 1.0, 1.0),
    "B2": (("C21a", "C21b"), 3.0, -1.0),
    "A2": (("C23a", "C23b"), 1.0, 1.0),
    "C3": (("C30",), 1.0, 1.0),
    "S3": (("C32a", "C32b"), 4.0, -1.0),
    "A3": (("C34a", "C34b"), 1.0, 1.0),
    "B4": (("C41a", "C41b"), 5.0, -1.0),
    "D4": (("C43a", "C43b"), 5.0, -1.0),
    "A4": (("C45a", "C45b"), 1.0, 1.0),
    "C5": (("C50",), 1.0, 1.0),
    "A5": (("C56a", "C56b"), 1.0, 1.0),
}


def ceos_to_pytemlib(aberrations: dict[str, list[float]]) -> dict[str, float]:
    """Convert CEOS aberrations (meters) to pyTEMlib keys (nm).

    pyTEMlib's ``make_chi`` on the result equals ``chi`` above on the input.
    """
    converted: dict[str, float] = {}
    for name, values in aberrations.items():
        if name not in _TO_PYTEMLIB:
            continue
        keys, factor, y_sign = _TO_PYTEMLIB[name]
        x, y = _xy(name, values)
        converted[keys[0]] = factor * x * 1e9
        if len(keys) == 2:
            converted[keys[1]] = factor * y_sign * y * 1e9
    return converted
