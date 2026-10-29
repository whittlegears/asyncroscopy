"""Electron dose: what one scanned frame or a parked-beam spectrum puts on the sample, in e/A^2.

Pure numbers; no instrument. Used by the Mapper's electron-beam profiles (STEM, SEM)
and the Tuner; see docs/designs/concepts_brief.md, decision D1.
"""

from __future__ import annotations

import math

ELECTRON_CHARGE = 1.602176634e-19


def frame_dose(current_pa: float, dwell_s: float, fov_m: float, imsize: int) -> float:
    """Electrons per square angstrom one frame gives."""
    pixel_a = fov_m / imsize / 1e-10
    return current_pa * 1e-12 * dwell_s / ELECTRON_CHARGE / (pixel_a**2)


def eds_dose(current_pa: float, exposure_s: float, spread_m: float, per_tile: list, map_points: int, fov_m: float, exposures: int) -> float:
    """Most electrons per square angstrom a tile's EDS spectra put on any ``spread_m`` square.

    A parked beam's dose is counted over a ``spread_m`` square (the probe is far smaller, but damage
    and contamination spread). An eds_map grid of ``map_points``^2 spectra can put several in one
    square, and overlapping tiles stack their grids (``exposures``); one eds_point sits at each tile centre.
    """
    if "eds_point" in per_tile:
        spectra = 1
    elif "eds_map" in per_tile:
        per_axis = min(int(map_points), int(math.floor(spread_m / (fov_m / int(map_points)) + 1e-9)) + 1)
        spectra = per_axis * per_axis * int(exposures)
    else:
        return 0.0
    return spectra * current_pa * 1e-12 * exposure_s / ELECTRON_CHARGE / (spread_m / 1e-10) ** 2
