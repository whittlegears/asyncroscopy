"""A simulated nanoindenter: the smallest instrument that shows the asyncroscopy pattern.

It is the worked example in docs/Adding_New_Hardware/new_instrument.md. Copy
it, replace the simulated physics with calls to your instrument's API, and
keep the command names and docstrings in the same style: the MCP server turns
each command into an agent tool, and ``asyncroscopy.utils.commands.command``
carries each docstring to the agent (plain ``tango.server.command`` does not).

Physics: the sample is two phases split at x = 0 (soft below, hard above).
Hardness follows from the peak load and a Berkovich contact depth, with
seeded noise, so the same plan gives the same numbers.
"""

from __future__ import annotations

import json
import math

import numpy as np
from tango import DevState
from tango.server import Device, device_property

from asyncroscopy.utils.commands import command

BERKOVICH_AREA = 24.5  # projected contact area = 24.5 * h_c**2


class SimIndenter(Device):
    max_load_n = device_property(dtype=float, default_value=0.5, doc="Largest peak load the head allows, in newtons.")
    travel_m = device_property(dtype=float, default_value=0.01, doc="Stage travel from the origin in each direction, in metres.")
    seed = device_property(dtype=int, default_value=0, doc="Random seed for the simulated measurement noise.")

    def init_device(self) -> None:
        Device.init_device(self)
        self._xy = [0.0, 0.0]
        self._tip_down = False
        self._rng = np.random.default_rng(int(self.seed))
        self.set_state(DevState.ON)

    @command(dtype_out=str)
    def get_limits(self) -> str:
        """JSON with the stage travel (travel_m, from the origin in each direction) and the largest load (max_load_n)."""
        return json.dumps({"travel_m": float(self.travel_m), "max_load_n": float(self.max_load_n)})

    @command(dtype_out=[float])
    def get_position(self) -> list[float]:
        """Stage position [x, y] in metres."""
        return list(self._xy)

    @command(dtype_in=[float], doc_in=":param xy: Target stage position [x, y] in metres.")
    def move_xy(self, xy: list[float]) -> None:
        """Move the stage. Retracts the tip first."""
        x, y = float(xy[0]), float(xy[1])
        if max(abs(x), abs(y)) > float(self.travel_m):
            raise ValueError(f"({x}, {y}) is outside the stage travel of +/-{self.travel_m} m")
        self._tip_down = False
        self._xy = [x, y]

    @command(dtype_in=float, dtype_out=str, doc_in=":param load_n: Peak load in newtons.")
    def indent(self, load_n: float) -> str:
        """Make one indent at the current position. Returns JSON with hardness (GPa) and contact depth (m)."""
        if not 0 < load_n <= float(self.max_load_n):
            raise ValueError(f"load must be in (0, {self.max_load_n}] N")
        self._tip_down = True
        hardness_gpa = (2.0 if self._xy[0] < 0 else 8.0) * (1 + 0.03 * self._rng.standard_normal())
        depth_m = math.sqrt(load_n / (BERKOVICH_AREA * hardness_gpa * 1e9))
        self._tip_down = False
        return json.dumps({"hardness_gpa": hardness_gpa, "contact_depth_m": depth_m, "load_n": load_n})

    @command
    def retract(self) -> None:
        """Lift the tip off the sample. Always safe."""
        self._tip_down = False


if __name__ == "__main__":
    SimIndenter.run_server()
