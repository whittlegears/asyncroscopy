import time


from abc import abstractmethod, ABCMeta

import tango
import tango.server

from asyncroscopy.utils.commands import action_hints_attribute, command

class CombinedMeta(tango.server.DeviceMeta, ABCMeta):
    """Combines Tango DeviceMeta and ABCMeta to allow abstract methods in Devices."""
    pass

class Instrument(tango.server.Device, metaclass=CombinedMeta):
    """instrument_type reads through a lambda, because Tango binds an attribute's default read method to the declaring
    class.
    """

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


    instrument_type = tango.server.attribute(
        fget=lambda device: device.read_instrument_type(),
        name="instrument_type",
        label="Instrument Type",
        dtype=str,
        access=tango.AttrWriteType.READ,
        doc="Instrument modality, for example 'STEM', 'SPM', 'TEM', or 'OPTIC'.",
    )

    action_hints = action_hints_attribute()

    clock_s = tango.server.attribute(
        label="Instrument clock",
        dtype=float,
        unit="s",
        access=tango.AttrWriteType.READ,
        doc="The instrument's own time: simulated seconds on a digital twin, Unix time on hardware. "
            "Jobs time drift fits and revisits with it, so they hold whatever the twin's time_scale.",
    )

    def init_device(self) -> None:
        tango.server.Device.init_device(self)
        self.instrument_id = self.get_name()  # recorded as each dataset's instrument_id
        self.set_state(tango.DevState.INIT)

        self._init_device_attributes()
        self._connect()

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
        """Set the attributes of one instrument type."""

    @abstractmethod
    def _connect(self):
        pass

    @abstractmethod
    def _disconnect(self):
        pass


    @command(risk="high", role="other")
    def Connect(self) -> None:
        """
        Explicitly (re)connect to microscope hardware. Useful after a fault.
        Raises ConnectFailed with the reason if the connection fails.
        """
        self._connect()
        error = getattr(self, "_connect_error", None)
        if error:
            tango.Except.throw_exception("ConnectFailed", f"Connect failed: {error}", "Connect()")

    @command(risk="high", role="other")
    def Disconnect(self) -> None:
        """Disconnect from microscope hardware gracefully."""
        self.set_state(tango.server.DevState.OFF)
        self._disconnect()
