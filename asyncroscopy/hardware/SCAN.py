"""
ALL SCANNING DETECTORS
This device holds acquisition settings for the HAADF and BF detectors.
It does NOT talk to AutoScript directly — the Microscope device
reads these attributes via DeviceProxy before acquiring.
"""
from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command
from tango import DevVarStringArray


class SCAN(Device):
    """SCAN detector settings device."""

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    dwell_time = attribute(
        label="Dwell Time",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="s",
        format="%e",
        min_value=1e-9,
        max_value=1e-3,
        doc="Per-pixel dwell time in seconds (e.g. 1e-6 = 1 µs)",
    )

    imsize = attribute(
        label="Image Size",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        unit="px",
        doc="Acquisition width in pixels (should match an AutoScript ImageSize preset)",
    )

    haadf = attribute(
        label="HAADF Detector",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        doc="HAADF detector state: True = active, False = inactive",
    )

    bf = attribute(
        label="BF Detector",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        doc="Bright Field detector state: True = active, False = inactive",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)
        self._dwell_time: float = 1e-6
        self._imsize: int = 512
        self._haadf: bool = False
        self._bf: bool = False
        self.info_stream("SCAN device initialised")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_dwell_time(self) -> float:
        return self._dwell_time

    def write_dwell_time(self, value: float) -> None:
        self._dwell_time = value

    def read_imsize(self) -> int:
        return self._imsize

    def write_imsize(self, value: int) -> None:
        self._imsize = value

    def read_haadf(self) -> bool:
        return self._haadf

    def write_haadf(self, value: bool) -> None:
        self._haadf = value
        self.info_stream(f"HAADF detector set to {'active' if value else 'inactive'}")

    def read_bf(self) -> bool:
        return self._bf

    def write_bf(self, value: bool) -> None:
        self._bf = value
        self.info_stream(f"BF detector set to {'active' if value else 'inactive'}")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    VALID_DETECTORS = {"haadf", "bf"}

    @command(dtype_in=DevVarStringArray, doc_in="List of detectors to activate, e.g. ['haadf', 'bf']. All others are deactivated.")
    def Activate(self, detectors) -> None:
        """Activate the named detectors and deactivate all others."""
        requested = {d.lower() for d in detectors}
        unknown = requested - self.VALID_DETECTORS
        if unknown:
            raise ValueError(f"Unknown detector(s): {unknown}. Valid options: {self.VALID_DETECTORS}")

        self._haadf = "haadf" in requested
        self._bf    = "bf"    in requested

        self.info_stream(f"Active detectors: {requested or 'none'}")


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SCAN.run_server()