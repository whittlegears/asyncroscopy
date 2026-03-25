"""
Shared pytest fixtures for Tango device tests.

Starts BOTH the detector device(s) and the Microscope device in ONE Tango
test device server using MultiDeviceTestContext, so the Microscope can
create DeviceProxy connections to detectors by device name.

This avoids:
- "No proxy found for detector 'scan'. Available: []"
- Needing a real Tango DB
- Flaky multi-context issues from spinning up multiple separate servers
"""

import numpy as np
import pytest
import tango
from tango.test_context import MultiDeviceTestContext

# Import device classes to test
from asyncroscopy.hardware.SCAN import SCAN
from asyncroscopy.ThermoDigitalTwin import ThermoDigitalTwin
from asyncroscopy.ThermoMicroscope import ThermoMicroscope


# We use ThermoDigitalTwin as our simulated microscope for all tests.

@pytest.fixture(scope="session")
def tango_ctx():
    """
    One Tango device server hosting SCAN + Microscope together.

    Device names here MUST match what you put into Microscope properties.
    """
    devices_info = [
        {
            "class": SCAN,
            "devices": [
                {
                    "name": "test/nodb/scan",
                    "properties": {
                        # put SCAN defaults here if you want
                        # e.g. "dwell_time": 2e-6  (only if it's a device_property)
                    },
                }
            ],
        },
        {
            "class": ThermoDigitalTwin,
            "devices": [
                {
                    "name": "test/nodb/twin",
                    "properties": {
                        "scan_device_address": "test/nodb/scan",
                    },
                }
            ],
        },

        {
            "class": ThermoMicroscope,
            "devices": [
                {
                    "name": "test/nodb/thermomicroscope",
                    "properties": {
                        # "simulate_hardware_for_tests": True,
                        "scan_device_address": "test/nodb/scan",
                    },
                }
            ],
        },
    ]

    # process=False keeps everything in the same process (fast, debuggable).
    # Also we only create ONE context, so the "second DeviceTestContext segfault"
    # issue doesn't apply.
    ctx = MultiDeviceTestContext(devices_info, process=False)
    with ctx:
        yield ctx



@pytest.fixture(scope="session")
def scan_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("test/nodb/scan"))


@pytest.fixture(scope="session")
def twin_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("test/nodb/twin"))


@pytest.fixture(scope="session")
def thermo_proxy(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("test/nodb/thermomicroscope"))



@pytest.fixture
def patched_single_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch ThermoMicroscope._acquire_stem_image so get_image() works
    without AutoScript/hardware.
    """
    def fake_acquire(self, imsize: int, dwell_time: float, detector_list: list):
        # Deterministic image makes tests stable
        arr = np.arange(imsize * imsize, dtype=np.uint16)
        return arr.reshape(imsize, imsize)

    monkeypatch.setattr(
        ThermoMicroscope,
        "_acquire_stem_image",
        fake_acquire,
    )
    monkeypatch.setattr(
        ThermoDigitalTwin,
        "_acquire_stem_image",
        fake_acquire,
    )