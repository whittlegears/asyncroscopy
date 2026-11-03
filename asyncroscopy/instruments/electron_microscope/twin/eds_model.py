"""EDS spectrum from what is under the beam.

The spectrum has an energy axis and looks like a real silicon-drift detector
spectrum: a Kramers bremsstrahlung continuum cut off at low energy by the
detector window, characteristic lines with energy-dependent resolution, Si
escape peaks, pile-up (sum) peaks and paralyzable dead time. Line energies
and relative weights come from pyTEMlib's X-ray tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pyTEMlib import xrpa_x_sections

from .sample_model import ELEMENTS

SI_ESCAPE_EV = 1740.0
MN_KA_EV = 5895.0
# Rough fluorescence-yield constants: omega = Z^4 / (a + Z^4) per shell family.
FAMILY_YIELD_A = {"K": 1.0e6, "L": 1.0e8, "M": 1.0e9}
FAMILY_FACTOR = {"K": 1.0, "L": 0.5, "M": 0.2}


@dataclass
class EdsParams:
    energy_max_ev: float = 20000.0
    channels: int = 2048
    fwhm_mn_ka_ev: float = 130.0
    solid_angle_sr: float = 0.7
    dead_time_s: float = 1e-6
    escape_peaks: bool = True
    sum_peaks: bool = True
    count_scale: float = 1.4e6
    continuum_scale: float = 4.0
    voltage_v: float = 300e3


@lru_cache(maxsize=64)
def element_lines(symbol: str, energy_max_ev: float) -> tuple[tuple[float, float], ...]:
    """(energy eV, relative intensity) for one element's lines inside the detector range."""
    z = ELEMENTS[symbol][0]
    lines = xrpa_x_sections.x_sections.get(str(z), {}).get("lines", {})
    families: dict[str, list[tuple[float, float]]] = {}
    for name, line in lines.items():
        energy = float(line["position"])
        if 150.0 <= energy <= energy_max_ev:
            families.setdefault(name[0], []).append((energy, float(line["weight"])))
    out = []
    for family, entries in families.items():
        if family not in FAMILY_YIELD_A:
            continue
        omega = z**4 / (FAMILY_YIELD_A[family] + z**4)
        total = sum(weight for _, weight in entries) or 1.0
        for energy, weight in entries:
            out.append((energy, FAMILY_FACTOR[family] * omega * weight / total))
    return tuple(out)


def energy_axis(params: EdsParams) -> np.ndarray:
    width = params.energy_max_ev / params.channels
    return (np.arange(params.channels) + 0.5) * width


def fwhm_ev(energy_ev: float | np.ndarray, params: EdsParams) -> np.ndarray:
    return np.sqrt(np.maximum(params.fwhm_mn_ka_ev**2 + 2.5 * (np.asarray(energy_ev) - MN_KA_EV), 30.0**2))


def point_spectrum(
    areal_density: dict[str, float],
    beam_current_pa: float,
    real_time_s: float,
    params: EdsParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (energy axis eV, counts, live time s).

    ``areal_density`` maps element symbol to atoms per square angstrom along
    the beam path at the probe position.
    """
    energies = energy_axis(params)
    channel_width = energies[1] - energies[0]
    current_na = beam_current_pa * 1e-3
    geometry = params.solid_angle_sr / (4.0 * np.pi)

    line_energies: list[float] = []
    line_rates: list[float] = []
    mean_z_density = 0.0
    for symbol, density in areal_density.items():
        if density <= 0:
            continue
        mean_z_density += ELEMENTS[symbol][0] * density
        for energy, intensity in element_lines(symbol, params.energy_max_ev):
            line_energies.append(energy)
            line_rates.append(params.count_scale * current_na * geometry * density * intensity)

    window = np.exp(-((400.0 / energies) ** 3))
    continuum = (
        params.continuum_scale * current_na * geometry * mean_z_density
        * (params.voltage_v - energies) / energies * window * (channel_width / 1000.0)
    )
    rate = continuum.copy()

    def add_peak(center: float, counts_per_s: float) -> None:
        sigma = fwhm_ev(center, params) / 2.355
        profile = np.exp(-0.5 * ((energies - center) / sigma) ** 2)
        total = profile.sum()
        if total > 0:
            rate[:] += counts_per_s * profile / total

    for energy, line_rate in zip(line_energies, line_rates):
        add_peak(energy, line_rate)
        if params.escape_peaks and energy > SI_ESCAPE_EV + 100.0:
            add_peak(energy - SI_ESCAPE_EV, 0.01 * line_rate)

    input_rate = float(rate.sum())
    if params.sum_peaks and line_rates:
        strongest = int(np.argmax(line_rates))
        pileup_fraction = min(0.05, input_rate * params.dead_time_s * 0.5)
        target = 2.0 * line_energies[strongest]
        if target < params.energy_max_ev:
            add_peak(target, pileup_fraction * line_rates[strongest])

    live_fraction = float(np.exp(-input_rate * params.dead_time_s))
    live_time = real_time_s * live_fraction
    counts = rng.poisson(np.clip(rate * live_time, 0.0, None)).astype(np.float64)
    return energies, counts, live_time
