"""
APERTURE Tango device.

This device defines the public aperture API. Hardware-specific subclasses
implement the private read, write, and command helpers.
"""

import json
import math
from abc import abstractmethod

from tango import AttrWriteType, DevState
from tango.server import Device, attribute, command


class APERTURE(Device):
    """Base device for a motorized aperture mechanism."""

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    available_mechanisms = attribute(
        label="Available mechanisms",
        dtype=(str,),
        max_dim_x=16,
        access=AttrWriteType.READ,
        doc="Motorized aperture mechanisms available on the microscope",
    )

    mechanism = attribute(
        label="Mechanism",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        doc="Aperture mechanism used by the other attributes and commands",
    )

    available_apertures = attribute(
        label="Available apertures",
        dtype=(str,),
        max_dim_x=32,
        access=AttrWriteType.READ,
        doc="Apertures available on the selected mechanism",
    )

    selected_aperture = attribute(
        label="Selected aperture",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        doc="Name of the aperture selected on the current mechanism",
    )

    aperture_type = attribute(
        label="Aperture type",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Type of the selected aperture",
    )

    aperture_diameter = attribute(
        label="Aperture diameter",
        dtype=float,
        access=AttrWriteType.READ,
        unit="m",
        doc="Diameter of the selected circular aperture in meters",
    )

    insertion_state = attribute(
        label="Insertion state",
        dtype=str,
        access=AttrWriteType.READ,
        doc="Insertion state of the selected mechanism",
    )

    enabled = attribute(
        label="Enabled",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="Whether the selected mechanism is enabled",
    )

    retractable = attribute(
        label="Retractable",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="Whether the selected mechanism is retractable",
    )

    position = attribute(
        label="Position",
        dtype=(float,),
        max_dim_x=2,
        access=AttrWriteType.READ_WRITE,
        unit="m",
        doc="Selected aperture position [x, y] in meters",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self._mechanism = ""
        self.set_state(DevState.ON)
        self.info_stream("APERTURE device initialised")

    # ------------------------------------------------------------------
    # Attribute read / write
    # ------------------------------------------------------------------

    def read_available_mechanisms(self) -> list[str]:
        return self._read_available_mechanisms()

    def read_mechanism(self) -> str:
        return self._mechanism

    def write_mechanism(self, value: str) -> None:
        self._mechanism = self._validated_mechanism(value)

    def read_available_apertures(self) -> list[str]:
        return self._read_available_apertures()

    def read_selected_aperture(self) -> str:
        return self._read_selected_aperture()

    def write_selected_aperture(self, value: str) -> None:
        self._write_selected_aperture(value)

    def read_aperture_type(self) -> str:
        return self._read_aperture_type()

    def read_aperture_diameter(self) -> float:
        return self._read_aperture_diameter()

    def read_insertion_state(self) -> str:
        return self._read_insertion_state()

    def read_enabled(self) -> bool:
        return self._read_enabled()

    def read_retractable(self) -> bool:
        return self._read_retractable()

    def read_position(self) -> list[float]:
        return self._read_position()

    def write_position(self, value) -> None:
        self._write_position(value)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    # insert/retract/enable/disable act on the mechanism selected through the
    # ``mechanism`` attribute. Remote bridges that expose only Tango commands
    # (e.g. the MCP server) cannot write attributes, so mechanism selection is
    # also available as commands here — without set_mechanism, retract could
    # never succeed through MCP because devices boot with a non-retractable
    # condenser mechanism selected.

    @command(dtype_out=(str,))
    def get_available_mechanisms(self) -> list[str]:
        """List the motorized aperture mechanisms available on the microscope."""
        return self._read_available_mechanisms()

    @command(dtype_in=str, doc_in=":param mechanism: Mechanism name, e.g. 'OBJ'")
    def set_mechanism(self, value: str) -> None:
        """Select the mechanism targeted by the other aperture attributes and commands."""
        self._mechanism = self._validated_mechanism(value)

    def _validated_mechanism(self, value: str) -> str:
        """Reject unknown mechanism names before they poison every later read."""
        value = value.strip()
        available = list(self._read_available_mechanisms())
        if value not in available:
            raise ValueError(f"Unknown mechanism {value!r}. Available: {available}")
        return value

    @command(dtype_out=str)
    def get_mechanism(self) -> str:
        """Return the currently selected mechanism."""
        return self._mechanism

    @command(dtype_out=(str,))
    def get_available_apertures(self) -> list[str]:
        """List the apertures available on the selected mechanism."""
        return self._read_available_apertures()

    @command(dtype_in=str, doc_in=":param aperture: Aperture name, e.g. '70 um'")
    def set_selected_aperture(self, value: str) -> None:
        """Select an aperture on the current mechanism."""
        self._write_selected_aperture(value)

    @command(dtype_out=str)
    def get_aperture_info(self) -> str:
        """Return the full state of the selected mechanism as JSON."""
        diameter = self._read_aperture_diameter()
        return json.dumps(
            {
                "mechanism": self._mechanism,
                "available_mechanisms": list(self._read_available_mechanisms()),
                "available_apertures": list(self._read_available_apertures()),
                "selected_aperture": self._read_selected_aperture(),
                "aperture_type": self._read_aperture_type(),
                "aperture_diameter_m": None if math.isnan(diameter) else diameter,
                "insertion_state": self._read_insertion_state(),
                "enabled": self._read_enabled(),
                "retractable": self._read_retractable(),
                "position": list(self._read_position()),
            }
        )

    @command
    def insert(self) -> None:
        self._insert()

    @command
    def retract(self) -> None:
        self._retract()

    @command
    def enable(self) -> None:
        self._enable()

    @command
    def disable(self) -> None:
        self._disable()

    @command
    def reset_positions(self) -> None:
        self._reset_positions()

    # ------------------------------------------------------------------
    # Hardware helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def _read_available_mechanisms(self):
        pass

    @abstractmethod
    def _read_available_apertures(self):
        pass

    @abstractmethod
    def _read_selected_aperture(self):
        pass

    @abstractmethod
    def _write_selected_aperture(self, value):
        pass

    @abstractmethod
    def _read_aperture_type(self):
        pass

    @abstractmethod
    def _read_aperture_diameter(self):
        pass

    @abstractmethod
    def _read_insertion_state(self):
        pass

    @abstractmethod
    def _read_enabled(self):
        pass

    @abstractmethod
    def _read_retractable(self):
        pass

    @abstractmethod
    def _read_position(self):
        pass

    @abstractmethod
    def _write_position(self, value):
        pass

    @abstractmethod
    def _insert(self):
        pass

    @abstractmethod
    def _retract(self):
        pass

    @abstractmethod
    def _enable(self):
        pass

    @abstractmethod
    def _disable(self):
        pass

    @abstractmethod
    def _reset_positions(self):
        pass


if __name__ == "__main__":
    APERTURE.run_server()
