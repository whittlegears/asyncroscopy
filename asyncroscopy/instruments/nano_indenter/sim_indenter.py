"""A simulated nanoindenter: the NanoIndenter base with simulated physics.

It is the worked example in docs/Adding_New_Hardware/new_instrument.md. A real backend subclasses
``NanoIndenter`` (nano_indenter.py) the same way and replaces the simulated physics with calls to the
instrument's API; the base keeps the commands, their docstrings (the agent's tool descriptions) and the
limit checks, so the agent sees the same tools on either.

Physics: the sample is two phases split at x = 0 (soft below, hard above).
Hardness follows from the peak load and a Berkovich contact depth, with
seeded noise, so the same plan gives the same numbers.
"""

from __future__ import annotations

import math

import numpy as np
from tango import DevState
from tango.server import device_property

from asyncroscopy.instruments.nano_indenter.nano_indenter import NanoIndenter

BERKOVICH_AREA = 24.5  # projected contact area = 24.5 * h_c**2


class SimIndenter(NanoIndenter):
    seed = device_property(dtype=int, default_value=0, doc="Random seed for the simulated measurement noise.")

    def _init_device_attributes(self) -> None:
        self._xy = [0.0, 0.0]
        self._tip_down = False
        self._rng = np.random.default_rng(int(self.seed))

    def _connect(self) -> None:
        self.set_state(DevState.ON)

    def _get_position(self) -> list[float]:
        return list(self._xy)

    def _move_xy(self, x: float, y: float) -> None:
        self._xy = [x, y]

    def _indent(self, load_n: float) -> dict:
        self._tip_down = True
        hardness_gpa = (2.0 if self._xy[0] < 0 else 8.0) * (1 + 0.03 * self._rng.standard_normal())
        depth_m = math.sqrt(load_n / (BERKOVICH_AREA * hardness_gpa * 1e9))
        self._tip_down = False
        return {"hardness_gpa": hardness_gpa, "contact_depth_m": depth_m, "load_n": load_n}

    def _retract(self) -> None:
        self._tip_down = False


if __name__ == "__main__":
    SimIndenter.run_server()
