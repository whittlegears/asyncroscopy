"""Sample drift: where a stitch's fitted drift puts the sample at a given time.

Pure numbers; no instrument. The Mapper's revisits use it to put a mapped region back
under the probe; see docs/designs/concepts_brief.md, decision D1.
"""

from __future__ import annotations

import numpy as np


def drift_at(drift: dict | None, t_s: float) -> tuple[float, float]:
    """The stitch's fitted drift (x, y in metres) at ``t_s`` after the first tile; extrapolated from the last segment."""
    if not drift or len(drift.get("times_s", [])) < 2:
        return 0.0, 0.0
    times = np.asarray(drift["times_s"], dtype=np.float64)
    xy = np.asarray(drift["xy_m"], dtype=np.float64)
    if t_s <= times[-1]:
        return float(np.interp(t_s, times, xy[:, 0])), float(np.interp(t_s, times, xy[:, 1]))
    rate = (xy[-1] - xy[-2]) / max(times[-1] - times[-2], 1e-9)
    end = xy[-1] + rate * (t_s - times[-1])
    return float(end[0]), float(end[1])
