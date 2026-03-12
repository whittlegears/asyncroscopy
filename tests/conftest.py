"""
Shared pytest fixtures for Tango device tests.

Starts BOTH the detector device(s) and the Microscope device in ONE Tango
test device server using MultiDeviceTestContext, so the Microscope can
create DeviceProxy connections to detectors by device name.

This avoids:
- "No proxy found for detector 'haadf'. Available: []"
- Needing a real Tango DB
- Flaky multi-context issues from spinning up multiple separate servers
"""

import pytest
import tango
from tango.test_context import MultiDeviceTestContext

# Import device classes to test
from asyncroscopy.detectors.HAADF import HAADF
from asyncroscopy.ThermoDigitalTwin import ThermoDigitalTwin


# We use ThermoDigitalTwin as our simulated microscope for all tests.

@pytest.fixture(scope="session")
def tango_ctx():
    """
    One Tango device server hosting HAADF + Microscope together.

    Device names here MUST match what you put into Microscope properties.
    """
    devices_info = [
        {
            "class": HAADF,
            "devices": [
                {
                    "name": "test/nodb/haadf",
                    "properties": {
                        # put HAADF defaults here if you want
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
                        "haadf_device_address": "test/nodb/haadf",
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
def haadf_proxy(tango_ctx) -> tango.DeviceProxy:
    return tango.DeviceProxy("test/nodb/haadf")


@pytest.fixture(scope="session")
def twin_proxy(tango_ctx) -> tango.DeviceProxy:
    return tango.DeviceProxy("test/nodb/twin")


@pytest.fixture(scope="session")
def microscope_proxy(twin_proxy):
    return twin_proxy