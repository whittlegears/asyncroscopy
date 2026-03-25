"""
Tests for the ThermoDigitalTwin Tango device.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pytest
import tango

# Using shared twin_proxy from conftest.py

class TestThermoDigitalTwin:

    def test_state_is_on(self, twin_proxy: tango.DeviceProxy):
        assert twin_proxy.state() == tango.DevState.ON

    def test_manufacturer_is_digital_twin(self, twin_proxy: tango.DeviceProxy):
        assert twin_proxy.manufacturer == "UTKTeam"

    def test_get_image_returns_valid_data(self, twin_proxy: tango.DeviceProxy, patched_single_image: pytest.MonkeyPatch):
        json_meta, raw_bytes = twin_proxy.get_scanned_image()
        meta = json.loads(json_meta)
        
        assert meta["detector"] == "haadf"
        assert "shape" in meta
        assert "dtype" in meta
        
        image = np.frombuffer(raw_bytes, dtype=meta["dtype"]).reshape(meta["shape"])
        assert image.shape == tuple(meta["shape"])

    def test_unknown_detector_raises(self, twin_proxy: tango.DeviceProxy):
        with pytest.raises(tango.DevFailed):
            twin_proxy.get_spectrum("void")
