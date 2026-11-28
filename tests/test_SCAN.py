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

    def test_default_imsize(self, scan_proxy):
        assert scan_proxy.imsize == 512

    def test_write_imsize(self, scan_proxy):
        scan_proxy.imsize = 256
        assert scan_proxy.imsize == 256


class TestSCANState:
    """Device state checks."""

    def test_initial_state_is_on(self, scan_proxy):
        import tango
        assert scan_proxy.state() == tango.DevState.ON


class TestSCANCommands:
    """Every scan setting must be reachable as a command, because the MCP server
    exposes Tango commands as tools but not attributes."""

    def test_imsize_round_trip(self, scan_proxy):
        scan_proxy.set_imsize(256)
        assert scan_proxy.get_imsize() == 256
        assert scan_proxy.imsize == 256

    def test_scan_region_round_trip(self, scan_proxy):
        scan_proxy.set_scan_region([0.25, 0.1, 0.5, 0.3])
        assert list(scan_proxy.get_scan_region()) == pytest.approx([0.25, 0.1, 0.5, 0.3])

    def test_output_format_round_trip(self, scan_proxy):
        scan_proxy.set_output_format(".tiff")
        assert scan_proxy.get_output_format() == ".tiff"
        scan_proxy.set_output_format(".h5")

    def test_dwell_time_round_trip(self, scan_proxy):
        scan_proxy.set_dwell_time(2e-6)
        assert scan_proxy.get_dwell_time() == pytest.approx(2e-6)

    @pytest.mark.parametrize("command, value", [
        ("set_imsize", 0),
        ("set_imsize", -64),
        ("set_scan_region", [0.8, 0.0, 0.5, 1.0]),
        ("set_scan_region", [0.0, 0.0, 1.0]),
        ("set_output_format", ".mrc"),
        ("set_dwell_time", 0.0),
        ("set_dwell_time", 60.0),
    ])
    def test_invalid_values_are_rejected_and_setting_kept(self, scan_proxy, command, value):
        import tango
        before = {
            "imsize": scan_proxy.get_imsize(),
            "scan_region": list(scan_proxy.get_scan_region()),
            "output_format": scan_proxy.get_output_format(),
            "dwell_time": scan_proxy.get_dwell_time(),
        }
        with pytest.raises(tango.DevFailed):
            getattr(scan_proxy, command)(value)
        assert scan_proxy.get_imsize() == before["imsize"]
        assert list(scan_proxy.get_scan_region()) == pytest.approx(before["scan_region"])
        assert scan_proxy.get_output_format() == before["output_format"]
        assert scan_proxy.get_dwell_time() == pytest.approx(before["dwell_time"])
