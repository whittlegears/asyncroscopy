"""
STAGE Tango device.

This device holds params for the scan.
It does NOT talk to AutoScript directly — the STEMMicroscope device
reads these attributes via DeviceProxy before acquiring.
"""

from tango import AttrWriteType, DevState
from tango.server import Device, attribute, device_property
from abc import abstractmethod

TILT_LIMITS_DEG = {"alpha": (-35.0, 35.0), "beta": (-20.0, 20.0)}

class STAGE(Device):
    """Stage/sample settings device.

    Public stage vectors are always [x, y, z, alpha, beta], with x/y/z in
    meters and alpha/beta in degrees.
    """

    # ------------------------------------------------------------------
    # Device properties — set per-deployment in the Tango DB
    # ------------------------------------------------------------------
    # Travel limits: a move outside them is refused, and job devices check plans against them at reserve.
    # The defaults (+/-1 mm in x and y, +/-0.5 mm in z) are conservative for a TEM goniometer; set your stage's.
    x_limits_m = device_property(dtype=(float,), default_value=[-1e-3, 1e-3], doc="[min, max] stage x travel (m).")
    y_limits_m = device_property(dtype=(float,), default_value=[-1e-3, 1e-3], doc="[min, max] stage y travel (m).")
    z_limits_m = device_property(dtype=(float,), default_value=[-5e-4, 5e-4], doc="[min, max] stage z travel (m).")

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    beta_tilt_enabled = attribute(
        label="Beta tilt enabled",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="Whether the beta tilt is enabled on this stage",
    )

    position = attribute(
        label="Position",
        dtype=(float,),
        max_dim_x=5,
        access=AttrWriteType.READ_WRITE,
        unit="m, m, m, deg, deg",
        doc="Stage position [x, y, z, alpha, beta], with tilts in degrees",
    )

    x = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        doc="Stage X position in meters (limits from the x_limits_m property)",
    )

    y = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        doc="Stage Y position in meters (limits from the y_limits_m property)",
    )

    z = attribute(
        label="Position",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        doc="Stage Z position in meters (limits from the z_limits_m property)",
    )

    alpha = attribute(
        label="Alpha tilt",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="degrees",
        min_value = -35,
        max_value = 35,
        doc="Stage alpha tilt in degrees",
    )

    beta = attribute(
        label="Beta tilt",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="degrees",
        min_value = -20,
        max_value = 20,
        doc="Stage beta tilt in degrees",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)
        self._position = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._limits = {
            "x": self._pair(self.x_limits_m, "x_limits_m"),
            "y": self._pair(self.y_limits_m, "y_limits_m"),
            "z": self._pair(self.z_limits_m, "z_limits_m"),
            **TILT_LIMITS_DEG,
        }
        attributes = self.get_device_attr()
        for axis in ("x", "y", "z"):
            low, high = self._limits[axis]
            attr = attributes.get_attr_by_name(axis)
            attr.set_min_value(low)
            attr.set_max_value(high)

        self.info_stream("STAGE device initialised")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_position(self) -> list[float]:
        return self._read_position()

    def write_position(self, value) -> None:
        for axis, v in zip(("x", "y", "z", "alpha", "beta"), value):
            low, high = self._limits[axis]
            if not low <= float(v) <= high:
                raise ValueError(f"stage {axis} = {float(v):.6g} is outside the stage limits [{low}, {high}]")
        self._write_position(value)

    @staticmethod
    def _pair(value, name: str) -> tuple[float, float]:
        low, high = (float(v) for v in value)
        if not low < high:
            raise ValueError(f"{name} must be [min, max] with min < max, got {list(value)}")
        return low, high

    def read_beta_tilt_enabled(self) -> bool:
        return self._read_beta_tilt_enabled()

    @abstractmethod
    def _read_position(self):
        pass

    @abstractmethod
    def _write_position(self, value):
        pass

    @abstractmethod
    def _read_beta_tilt_enabled(self):
        pass

    def read_x(self) -> float:
        return self.read_position()[0]

    def write_x(self, value: float) -> None:
        position = self.read_position()
        position[0] = float(value)
        self.write_position(position)

    def read_y(self) -> float:
        return self.read_position()[1]

    def write_y(self, value: float) -> None:
        position = self.read_position()
        position[1] = float(value)
        self.write_position(position)

    def read_z(self) -> float:
        return self.read_position()[2]

    def write_z(self, value: float) -> None:
        position = self.read_position()
        position[2] = float(value)
        self.write_position(position)

    def read_alpha(self) -> float:
        return self.read_position()[3]

    def write_alpha(self, value: float) -> None:
        position = self.read_position()
        position[3] = float(value)
        self.write_position(position)

    def read_beta(self) -> float:
        position = self.read_position()
        if len(position) < 5:
            return 0.0
        return position[4]

    def write_beta(self, value: float) -> None:
        if not self.read_beta_tilt_enabled():
            if float(value) == 0.0:
                return
            raise ValueError("This stage does not support beta tilt")
        position = self.read_position()
        position[4] = float(value)
        self.write_position(position)


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    STAGE.run_server()
