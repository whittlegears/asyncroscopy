"""
STAGE Tango device.

This device holds params for the scan.
It does NOT talk to AutoScript directly — the Microscope device
reads these attributes via DeviceProxy before acquiring.
"""

from tango import AttrWriteType, DevState
from tango.server import Device, attribute


class STAGE(Device):
    """Stage/Sample settings device."""

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------

    # (no hardware connection properties needed — STAGE is settings-only)

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    beta_tilt_enabled = attribute(
        label="Beta Tilt Enabled",
        dtype=bool,
        access=AttrWriteType.READ_WRITE,
        doc="Whether the holder supports beta tilt)",
    )

    x = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        units="m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Stage X position in meters",
    )

    y = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        units="m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Stage Y position in meters",
    )

    z = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        units="m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Stage Z position in meters",
    )

    alpha = attribute(
        label="Alpha tilt",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        units="degrees",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Stage alpha tilt in degrees",
    )

    beta = attribute(
        label="Beta tilt",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        units="degrees",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Stage beta tilt in degrees",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)

        # Start with zeros, TODO: get real numbers during initialization
        self.beta_tilt_enabled: bool = False
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.alpha: float = 0.0
        self.beta: float = 0.0


        self.info_stream("STAGE device initialised")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_beta_tilt_enabled(self) -> bool:
        return self.beta_tilt_enabled
    
    def write_beta_tilt_enabled(self, value: bool) -> None:
        self.beta_tilt_enabled = value

    def read_x(self) -> float:
        return self.x

    def write_x(self, value: float) -> None:
        self.x = value
    
    def read_y(self) -> float:
        return self.y

    def write_y(self, value: float) -> None:
        self.y = value
    
    def read_z(self) -> float:
        return self.z

    def write_z(self, value: float) -> None:
        self.z = value
    
    def read_alpha(self) -> float:
        return self.alpha

    def write_alpha(self, value: float) -> None:
        self.alpha = value

    def read_beta(self) -> float:
        return self.beta

    def write_beta(self, value: float) -> None:
        self.beta = value

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    STAGE.run_server()