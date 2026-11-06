"""NanoIndenter: the Tango device every nanoindenter backend shares.

The base owns the public commands (the agent's tools) and the checks every backend must apply:
``move_xy`` refuses a target outside ``travel_m`` and ``indent`` a load outside (0, ``max_load_n``],
before a backend sees them. A backend (a vendor driver, or ``SimIndenter``) implements only the
``_x`` methods: ``_move_xy``, ``_indent``, ``_retract`` and ``_get_position``.

It is the worked example in docs/Adding_New_Hardware/new_instrument.md: copy this base-plus-backend
shape for a new kind of instrument.
"""

from __future__ import annotations

import json
import math
from abc import abstractmethod

from tango.server import device_property

from asyncroscopy.instruments.instrument import Instrument
from asyncroscopy.utils.commands import command


class NanoIndenter(Instrument):
    """A nanoindenter: an XY stage and an indenting head."""

    max_load_n = device_property(dtype=float, default_value=0.5, doc="Largest peak load the head allows, in newtons.")
    travel_m = device_property(dtype=float, default_value=0.01, doc="Stage travel from the origin in each direction, in metres.")

    def read_instrument_type(self) -> str:
        return "NANOINDENTER"

    def _disconnect(self) -> None:
        self._retract()

    # ------------------------------------------------------------------
    # Commands (the agent's tools)
    # ------------------------------------------------------------------
    @command(dtype_out=str)
    def get_limits(self) -> str:
        """JSON with the stage travel (travel_m, from the origin in each direction) and the largest load (max_load_n)."""
        return json.dumps({"travel_m": float(self.travel_m), "max_load_n": float(self.max_load_n)})

    @command(dtype_out=[float])
    def get_position(self) -> list[float]:
        """Stage position [x, y] in metres."""
        return [float(v) for v in self._get_position()]

    @command(dtype_in=[float], doc_in=":param xy: Target stage position [x, y] in metres.")
    def move_xy(self, xy: list[float]) -> None:
        """Move the stage. Retracts the tip first."""
        x, y = float(xy[0]), float(xy[1])
        if not (math.isfinite(x) and math.isfinite(y)) or max(abs(x), abs(y)) > float(self.travel_m):
            raise ValueError(f"({x}, {y}) is outside the stage travel of +/-{self.travel_m} m")
        self._retract()
        self._move_xy(x, y)

    @command(dtype_in=float, dtype_out=str, doc_in=":param load_n: Peak load in newtons.")
    def indent(self, load_n: float) -> str:
        """Make one indent at the current position. Returns JSON with hardness (GPa) and contact depth (m)."""
        if not 0 < float(load_n) <= float(self.max_load_n):
            raise ValueError(f"load must be in (0, {self.max_load_n}] N")
        return json.dumps(self._indent(float(load_n)))

    @command
    def retract(self) -> None:
        """Lift the tip off the sample. Always safe."""
        self._retract()

    # ------------------------------------------------------------------
    # What a backend implements
    # ------------------------------------------------------------------
    @abstractmethod
    def _get_position(self) -> list[float]:
        """The stage position [x, y] in metres."""

    @abstractmethod
    def _move_xy(self, x: float, y: float) -> None:
        """Move the stage to (x, y) metres (inside the travel; the tip is already retracted)."""

    @abstractmethod
    def _indent(self, load_n: float) -> dict:
        """One indent at the current position, to ``load_n`` (already checked); {"hardness_gpa", "contact_depth_m", "load_n"}."""

    @abstractmethod
    def _retract(self) -> None:
        """Lift the tip off the sample; must never fail for a tip already up."""
