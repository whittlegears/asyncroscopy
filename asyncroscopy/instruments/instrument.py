import json
import time
from typing import Optional


from abc import abstractmethod, ABCMeta

import tango
import tango.server

class CombinedMeta(tango.server.DeviceMeta, ABCMeta):
    """Combines Tango DeviceMeta and ABCMeta to allow abstract methods in Devices."""
    pass

class Instrument(tango.server.Device, metaclass=CombinedMeta):

    # ------------------------------------------------------------------
    # Instrument level Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------
    data_device_address = tango.server.device_property(
        dtype=str,
        default_value="",
        doc="Optional Tango device address for the DATA device, e.g. 'asyncroscopy/data/default'.",
    )

    testing_mode_bool = tango.server.device_property(
        dtype=bool,
        default_value=False,
        doc="When True - used for running tests, passed in conftest.py")

    simulated = tango.server.device_property(
        dtype=bool,
        default_value=False,
        doc="True when this stack drives a simulator, set from run_servers' top-level simulated:. "
            "False, the default, means real hardware (safety checks key off it).",
    )

    # ------------------------------------------------------------------
    # Instrument Attributes
    # ------------------------------------------------------------------

    instrument_type = tango.server.attribute(
        label="Instrument Type",
        dtype=str,
        access=tango.AttrWriteType.READ,
        doc="Instrument modality, for example 'STEM', 'SPM', 'TEM', or 'OPTIC'.",
    )

    clock_s = tango.server.attribute(
        label="Instrument clock",
        dtype=float,
        unit="s",
        access=tango.AttrWriteType.READ,
        doc="The instrument's own time: simulated seconds on a digital twin, Unix time on hardware. "
            "Jobs time drift fits and revisits with it, so they hold whatever the twin's time_scale.",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_device(self) -> None:
        tango.server.Device.init_device(self)
        self.instrument_id = self.get_name()  # what every dataset it saves records as instrument_id
        self.set_state(tango.DevState.INIT)

        self._init_device_attributes()
        self._connect()

    # ------------------------------------------------------------------
    # Instrument methods
    # ------------------------------------------------------------------
    @abstractmethod
    def read_instrument_type(self) -> str:
        pass

    def read_clock_s(self) -> float:
        return float(self.instrument_clock_s())

    def instrument_clock_s(self) -> float:
        """The time clock_s reports; a simulator overrides this with its simulated clock."""
        return time.time()

    @abstractmethod
    def _init_device_attributes(self) -> None:
        """
        Initialize device-specific attributes.

        Define attributes that are specific to a particular instrument type (STEMMicroscope, SPMMicroscope, etc.).
        """
        pass

    @abstractmethod
    def _connect(self):
        pass

    @abstractmethod
    def _disconnect(self):
        pass


    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @tango.server.command
    def Connect(self) -> None:
        """
        Explicitly (re)connect to microscope hardware. Useful after a fault.
        Raises ConnectFailed with the reason if the connection fails.
        """
        self._connect()
        error = getattr(self, "_connect_error", None)
        if error:
            tango.Except.throw_exception("ConnectFailed", f"Connect failed: {error}", "Connect()")

    @tango.server.command
    def Disconnect(self) -> None:
        """Disconnect from microscope hardware gracefully."""
        self.set_state(tango.server.DevState.OFF)
        self._disconnect()
