"""
Tests for the SCAN hardware Tango device.

Runs against a real DeviceTestContext — exercises actual Tango
attribute machinery, not mocks.
"""

import pytest


class TestSCANAttributes:
    """Attribute read/write round-trips."""

    def test_default_dwell_time(self, scan_proxy):
        assert scan_proxy.dwell_time == pytest.approx(1e-6)

    def test_write_dwell_time(self, scan_proxy):
        scan_proxy.dwell_time = 5e-6
        assert scan_proxy.dwell_time == pytest.approx(5e-6)

    def test_default_image_width(self, scan_proxy):
        assert scan_proxy.image_width == 1024

    def test_write_image_width(self, scan_proxy):
        scan_proxy.image_width = 512
        assert scan_proxy.image_width == 512

    def test_default_image_height(self, scan_proxy):
        assert scan_proxy.image_height == 1024

    def test_write_image_height(self, scan_proxy):
        scan_proxy.image_height = 512
        assert scan_proxy.image_height == 512


class TestSCANState:
    """Device state checks."""

    def test_initial_state_is_on(self, scan_proxy):
        import tango
        assert scan_proxy.state() == tango.DevState.ON