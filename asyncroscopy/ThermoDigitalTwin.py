"""
Digital twin version of ThermoMicroscope for simulated image generation.

Useful for testing and development without requiring AutoScript hardware.
"""

import json
import time
import numpy as np
from tango import DevState, DeviceProxy, DevFailed
from tango.server import Device, attribute

from asyncroscopy.Microscope import Microscope

class ThermoDigitalTwin(Microscope):
    """
    A digital twin for the ThermoMicroscope.
    """

    manufacturer = attribute(
        label="ThermoDigitalTwin",
        dtype=str,
        doc="Simulation backend",
    )

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)
        
        # Internal state
        self._stem_mode = True
        self._detector_proxies = {}
        self._manufacturer = "UTKTeam"
        
        self._connect()

    def _connect(self):
        """Simulate connection by connecting to detector proxies."""
        self._connect_detector_proxies()
        self.set_state(DevState.ON)

    def _connect_detector_proxies(self) -> None:
        """
        Connect to simulated detector proxies. 
        """
        addresses: dict[str, str] = {
            "haadf": self.haadf_device_address,
            "AdvancedAcquistion": self.advanced_acquisition_device_address,
        }
        
        for name, address in addresses.items():
            if not address:
                continue
            try:
                self._detector_proxies[name] = DeviceProxy(address)
                self.info_stream(f"Connected to detector proxy: {name} @ {address}")
            except DevFailed as e:
                self.error_stream(f"Failed to connect to {name} proxy at {address}: {e}")

    def read_manufacturer(self) -> str:
        return self._manufacturer

    def _acquire_stem_image(
        self,
        detector_name: str,
        width: int,
        height: int,
        dwell_time: float,
    ) -> np.ndarray:
        """Generate a random image to simulate acquisition."""
        self.info_stream(f"Simulating {detector_name} image: {width}x{height}")
        rng = np.random.default_rng()
        return rng.integers(0, 65535, size=(height, width), dtype=np.uint16)

    def _acquire_stem_image_advanced(
        self,
        detector_names: list[str],
        base_resolution: int,
        scan_region: list[float],
        dwell_time: float,
        auto_beam_blank: bool,
    ) -> list[np.ndarray]:
        """Generate multiple random images to simulate simultaneous acquisition."""
        self.info_stream(f"Simulating advanced acquisition for detectors: {detector_names}")
        rng = np.random.default_rng()
        
        # Calculate dimensions from scan_region [left, top, width, height]
        width = int(base_resolution * scan_region[2])
        height = int(base_resolution * scan_region[3])
        
        return [rng.integers(0, 65535, size=(height, width), dtype=np.uint16) 
                for _ in detector_names]


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    ThermoDigitalTwin.run_server()
