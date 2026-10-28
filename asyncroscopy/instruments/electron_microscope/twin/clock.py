"""Simulated time.

Stage moves, frames and spectra advance the clock by what they would take on
the instrument. ``time_scale`` sets how much real time passes per simulated
second: 1.0 waits like the microscope would, 0.0 never sleeps (tests).
"""

from __future__ import annotations

import time


class SimClock:
    def __init__(self, time_scale: float = 0.0):
        self.time_scale = max(0.0, float(time_scale))
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        seconds = max(0.0, float(seconds))
        self._now += seconds
        if self.time_scale > 0.0 and seconds > 0.0:
            time.sleep(seconds * self.time_scale)
        return self._now
