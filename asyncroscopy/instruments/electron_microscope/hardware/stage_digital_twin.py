"""The digital twin's STAGE device.

The twin microscope moves its stage model (commanded, true and reported positions: bias, Gaussian landing
error, backlash on a reversal, settling) on its own simulated clock, and writes the reported position here,
as a real stage controller would report it. This device keeps that position and the travel limits every
move is checked against, and reports how far the stage really lands from where it says it is
(``position_sigma_m``), which the twin works out from its stage model. A Mapper whose ``stage_sigma_m`` is
not set reads it, so a twin configuration never has to derive the stitching bound by hand.
"""

from __future__ import annotations

import math

from tango import AttrWriteType
from tango.server import attribute

from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE


class DigitalTwinSTAGE(STAGE):
    """The STAGE device of a simulated stack: positions written by the twin microscope, and its landing error."""

    position_sigma_m = attribute(
        dtype=float,
        unit="m",
        access=AttrWriteType.READ_WRITE,
        doc="1-sigma distance between where the stage lands and where it reports (m); written by the twin microscope "
        "from its stage model (landing noise, backlash, readback noise). 0 until the twin has written it.",
    )

    def init_device(self) -> None:
        super().init_device()
        self._sigma_m = float(self.position_error_m)

    def _read_position(self) -> list[float]:
        return list(self._position)

    def _write_position(self, value) -> None:
        position = [float(component) for component in value]
        if len(position) != 5:
            raise ValueError("Stage position must be [x, y, z, alpha, beta]")
        self._position = position

    def _read_beta_tilt_enabled(self) -> bool:
        return True

    def read_position_sigma_m(self) -> float:
        return self._sigma_m

    def write_position_sigma_m(self, value: float) -> None:
        if not (math.isfinite(float(value)) and float(value) >= 0):
            raise ValueError(f"position_sigma_m must be a distance >= 0 in metres, got {value}")
        self._sigma_m = float(value)


if __name__ == "__main__":
    DigitalTwinSTAGE.run_server()
