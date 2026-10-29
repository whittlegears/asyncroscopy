"""Value checks every electron-microscope backend applies before a value reaches the column.

The digital twin and the AutoScript backend call the same functions, so they refuse the
same values with the same messages (tests/test_parity.py checks this).
"""

from __future__ import annotations

import math


def check_fov(fov, min_fov: float, max_fov: float) -> float:
    """Field of view in metres, refused outside [min_fov, max_fov]."""
    fov = float(fov)
    if not (math.isfinite(fov) and float(min_fov) <= fov <= float(max_fov)):
        raise ValueError(f"field of view must be between {min_fov} and {max_fov} m, got {fov}")
    return fov


def check_defocus(defocus, max_defocus: float) -> float:
    """Defocus in metres, refused beyond +/-max_defocus."""
    defocus = float(defocus)
    if not (math.isfinite(defocus) and abs(defocus) <= float(max_defocus)):
        raise ValueError(f"defocus must be within +/-{max_defocus} m (did you pass nanometres?), got {defocus}")
    return defocus


def check_scan(imsize, dwell_time, scan_region) -> tuple[float, float, float, float]:
    """A scan's [left, top, width, height] (fractions of the field), refused unless it lies inside the field."""
    region = tuple(float(v) for v in scan_region) if scan_region is not None else (0.0, 0.0, 1.0, 1.0)
    if len(region) != 4 or not all(math.isfinite(v) for v in region):
        raise ValueError(f"scan_region must be four finite numbers [left, top, width, height], got {list(region)}")
    left, top, width, height = region
    if not (left >= 0 and top >= 0 and width > 0 and height > 0 and left + width <= 1 + 1e-9 and top + height <= 1 + 1e-9):
        raise ValueError(f"scan_region [left, top, width, height] must lie inside the field (0..1), got {list(region)}")
    if not (math.isfinite(imsize) and int(imsize) >= 1 and math.isfinite(dwell_time) and float(dwell_time) > 0):
        raise ValueError(f"imsize must be >= 1 and dwell_time > 0, got {imsize}, {dwell_time}")
    return region
