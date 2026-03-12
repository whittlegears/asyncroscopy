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
from tango.test_context import MultiDeviceTestContext

from asyncroscopy.detectors.HAADF import HAADF
from asyncroscopy.ThermoDigitalTwin import ThermoDigitalTwin

# Using shared twin_proxy from conftest.py

class TestThermoDigitalTwin:

    def test_state_is_on(self, twin_proxy):
        assert twin_proxy.state() == tango.DevState.ON

    def test_manufacturer_is_digital_twin(self, twin_proxy):
        assert twin_proxy.manufacturer == "UTKTeam"

    def test_get_image_returns_valid_data(self, twin_proxy):
        json_meta, raw_bytes = twin_proxy.get_image("haadf")
        meta = json.loads(json_meta)
        
        assert meta["detector"] == "haadf"
        assert "shape" in meta
        assert "dtype" in meta
        
        image = np.frombuffer(raw_bytes, dtype=meta["dtype"]).reshape(meta["shape"])
        assert image.shape == tuple(meta["shape"])
        assert image.dtype == np.uint16

    def test_unknown_detector_raises(self, twin_proxy):
        with pytest.raises(tango.DevFailed):
            twin_proxy.get_image("void")
