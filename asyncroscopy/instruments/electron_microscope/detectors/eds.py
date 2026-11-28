"""
EDS (Energy disperive X-ray spectroscopy) detector Tango device.

This device holds acquisition settings for the EDS detector.
It does NOT talk to AutoScript directly — the STEMMicroscope device (although this will change in the future)
reads these attributes via DeviceProxy before acquiring.
"""

from tango import AttrWriteType, DevState
from tango.server import Device, attribute

from asyncroscopy.utils.commands import action_hints_attribute, command


class EDS(Device):
    """EDS detector settings device."""


    action_hints = action_hints_attribute()


    exposure_time = attribute(
        label="Dwell Time",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        unit="s",
        format="%e",
        min_value=1e-6,
        max_value=1e2,
        doc="Exposure time in seconds (e.g. 1e-6 = 1 µs)",
    )


    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.ON)

        self._exposure_time: float = 1   # s

        self.info_stream("EDS device initialised")


    def read_exposure_time(self) -> float:
        return self._exposure_time

    def write_exposure_time(self, value: float) -> None:
        self._exposure_time = value


    @command(dtype_in=float, doc_in=":param seconds: Time the next EDS spectrum counts for, in seconds, from 1e-6 to 100.",
             risk="high", role="setting.set", routine=True)
    def set_eds_exposure_time(self, seconds: float) -> None:
        """Set how long each following EDS spectrum counts for, in seconds (1 s when the device starts). A longer time
        gives more counts and more dose where the beam is parked."""
        seconds = float(seconds)
        if not 1e-6 <= seconds <= 1e2:
            raise ValueError(f"the EDS exposure time must be from 1e-6 to 100 seconds, got {seconds:g}")
        self._exposure_time = seconds

    @command(dtype_out=float, risk="low", role="setting.read")
    def get_eds_exposure_time(self) -> float:
        """Read how long an EDS spectrum counts for, in seconds."""
        return self._exposure_time


if __name__ == "__main__":
    EDS.run_server()