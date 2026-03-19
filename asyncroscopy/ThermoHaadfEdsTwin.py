"""
Digital twin version of ThermoMicroscope for HAADF-EDX.

Useful for testing and development without requiring AutoScript hardware.
"""


import json
import time
import math
from typing import Optional

import numpy as np
import tango
from tango import AttrWriteType, DevEncoded, DevState
from tango.server import Device, attribute, command, device_property

from asyncroscopy.Microscope import Microscope

class ThermoHaadfEdsTwin(Microscope):
    """
    Manages the AutoScript connection and exposes acquisition commands.
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    # not finishded
    manufacturer = attribute(
        label="ThermoHaadfEdsTwin",
        dtype=str,
        doc="Simulation backend",
    )

    beam_pos = attribute(
        label="Beam Position",
        dtype=(float,),        # 1D array of floats
        max_dim_x=2,           # exactly 2 elements: [x, y]
        access=AttrWriteType.READ_WRITE,
        unit="fractional",
        min_value=0.0,
        max_value=1.0,
        doc="Beam position as [x, y] fractional coordinates, each in range [0.0, 1.0]",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

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
        """Build DeviceProxy objects for each configured detector device."""
        # Extend this dict as more detectors are added
        # later, we want to do this automatically, not with a dictionary.
        addresses: dict[str, str] = {
            "haadf": self.haadf_device_address,
            "AdvancedAcquistion": self.advanced_acquisition_device_address,
            "eds":  self.eds_device_address,
            "stage": self.stage_device_address,
            "scan": self.scan_device_address,
        }
        print(addresses)
        for name, address in addresses.items():
            if not address:   # <-- minimal fix
                self.info_stream(f"Skipping {name}: no address configured")
                continue
            try:
                self._detector_proxies[name] = tango.DeviceProxy(address)
                self.info_stream(f"Connected to detector proxy: {name} @ {address}")
            except tango.DevFailed as e:
                self.error_stream(f"Failed to connect to {name} proxy at {address}: {e}")


    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------

    def read_manufacturer(self) -> bool:
        # TODO: query self._microscope.optics.mode when AutoScript available
        return self._manufacturer


    def read_beam_pos(self):
        """Return beam position as [x, y] fractional coordinates."""
        return [self._beam_pos_x, self._beam_pos_y]

    # --- Write Method ---

    def write_beam_pos(self, value):
        """Set beam position from [x, y] fractional coordinates."""
        x, y = value[0], value[1]

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"beam_pos values must be in [0.0, 1.0], got x={x}, y={y}"
            )

        self._beam_pos_x = x
        self._beam_pos_y = y
        
    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @command(dtype_in=str, dtype_out=DevEncoded)#In PyTango, DevEncoded is a special Tango data type designed to send binary data + a small description string together as a single return value.
    def get_image(self, detector_name: str) -> tuple[str, bytes]:
        """
        Acquire a single STEM image from the named detector.

        Parameters
        ----------
        detector_name:
            Name of the detector, e.g. "haadf". Must match a key in
            self._detector_proxies.

        Returns
        -------
        DevEncoded = (json_metadata, raw_bytes)
            json_metadata includes: shape, dtype, dwell_time, detector,
            timestamp, and any other relevant metadata.
            raw_bytes is the flat numpy array bytes; reshape using shape from metadata.
        """
        detector_name = detector_name.lower().strip()

        proxy = self._detector_proxies.get(detector_name)
        if proxy is None:
            tango.Except.throw_exception(
                "UnknownDetector",
                f"No proxy found for detector '{detector_name}'. "
                f"Available: {list(self._detector_proxies.keys())}",
                "Microscope.get_image()",
            )

        # Read acquisition settings from the detector device
        dwell_time: float = proxy.dwell_time
        width: int  = proxy.imsize 
        height: int = proxy.imsize

        # TODO: map (width, height) → AutoScript ImageSize enum
        # e.g. ImageSize.PRESET_1024 when width == height == 1024

        adorned_image = self._acquire_stem_image(detector_name, width, height, dwell_time)

        metadata = {
            "detector": detector_name,
            "shape": [height, width],
            "dtype": str(adorned_image.dtype),
            "dwell_time": dwell_time,
            "timestamp": time.time(),
            # TODO: add metadata from adorned_image.metadata when using real AutoScript
        }

        return json.dumps(metadata), adorned_image.tobytes()


    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    # TODO:if self._microscope is not None: checks should go to init functions than sitting in commands 

    def _acquire_stem_image(self, detector_name: int, imsize: int, dwell_time: float, detector_list: list) -> np.ndarray:
        """
        Call AutoScript acquisition and return numpy array.

        Falls back to a simulated image when AutoScript is unavailable.
        """
        height, width = imsize, imsize
        rng = np.random.default_rng()
        HAADF_array = rng.integers(0, 65535, size=(height, width), dtype=np.uint16)
        return HAADF_array



    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> np.ndarray:
        if detector_name.upper() == "EDS":
            # set up settings object
            spectrum = np.linspace(0, 1000, 20)

        else:
            print(f"Detector {detector_name} not supported for spectroscopy")
            return

        return spectrum


    def _place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        x, y = position
        self.write_beam_pos([x, y])

    def _set_fov(self, fov) -> None:
        """set field of view in meters"""
        self._microscope.optics.scan_field_of_view = fov




# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    ThermoHaadfEdsTwin.run_server()