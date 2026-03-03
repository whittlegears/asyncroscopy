"""
Microscope Tango device.

Owns the AutoScript connection and all acquisition commands.
Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

Return convention for image commands
-------------------------------------
All image commands return DevEncoded = (str, bytes) where:
  - str  : JSON string containing metadata (shape, dtype, dwell_time, …)
  - bytes: raw numpy array bytes (reconstruct with np.frombuffer + reshape)

Client-side reconstruction example::

    import json, numpy as np
    encoded = proxy.get_haadf_image()   # returns (json_str, raw_bytes)
    meta    = json.loads(encoded[0])
    image   = np.frombuffer(encoded[1], dtype=meta["dtype"]).reshape(meta["shape"])
"""

import json
import time
from typing import Optional

import numpy as np
import tango
from tango import AttrWriteType, DevEncoded, DevState
from tango.server import Device, attribute, command, device_property

# AutoScript imports — only available on the microscope PC.
# Wrapped in try/except so the device can still be imported and tested
# on a development machine without AutoScript installed.
try:
    from autoscript_tem_microscope_client import TemMicroscopeClient
    from autoscript_tem_microscope_client.enumerations import DetectorType, ImageSize
    from autoscript_tem_microscope_client.structures import Region, Rectangle
    from autoscript_tem_microscope_client.enumerations import RegionCoordinateSystem
    from autoscript_tem_microscope_client.structures import StemAcquisitionSettings
    _AUTOSCRIPT_AVAILABLE = True
except ImportError:
    _AUTOSCRIPT_AVAILABLE = False

print(_AUTOSCRIPT_AVAILABLE)


from asyncroscopy.Microscope import Microscope

class ThermoMicroscope(Microscope):
    """
    Manages the AutoScript connection and exposes acquisition commands.
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    # not finishded
    manufacturer = attribute(
        label="Thermofisher",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="This microscope uses AutoScript for control and acquisition",
    )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_device(self) -> None:
        Device.init_device(self)
        self.set_state(DevState.INIT)

        self._microscope: Optional[object] = None  # TemMicroscopeClient instance
        self._stem_mode: bool = False

        # Dict mapping detector name string → DeviceProxy
        # Populated in _connect_detector_proxies
        self._detector_proxies: dict[str, tango.DeviceProxy] = {}

        self._connect()

    def _connect(self):
        self._connect_hardware()
        self._connect_detector_proxies()
        self.set_state(DevState.ON)

    def _connect_hardware(self) -> None:
        """Establish AutoScript connection from MPC -> hardware."""
        if not _AUTOSCRIPT_AVAILABLE:
            self.warn_stream("AutoScript not available")
            return
        try:
            self._microscope = TemMicroscopeClient()
            self._microscope.connect(self.autoscript_host_ip, self.autoscript_host_port)
            self.info_stream(f"Connected to AutoScript at {self.autoscript_host_ip}:{self.autoscript_host_port}")
        except Exception as e:
            self.error_stream(f"AutoScript connection failed: {e}")
            self.set_state(DevState.FAULT)
            self._microscope = None

    def _connect_detector_proxies(self) -> None:
        """Build DeviceProxy objects for each configured detector device."""
        # Extend this dict as more detectors are added
        # later, we want to do this automatically, not with a dictionary.
        addresses: dict[str, str] = {
            "haadf": self.haadf_device_address,
            "AdvancedAcquistion": self.advanced_acquisition_device_address,
            # "BF"
            # "eds":  self.eds_device_address,
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

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _acquire_stem_image(
        self,
        detector_name: str,
        width: int,
        height: int,
        dwell_time: float,
    ) -> np.ndarray:
        """
        Call AutoScript acquisition and return numpy array.

        Falls back to a simulated image when AutoScript is unavailable.
        """
        if self._microscope is not None:
            # Real AutoScript path
            if detector_name.upper() == "HAADF":
                detector_type = DetectorType.HAADF # :TODO --> make it general and check
                adorned = self._microscope.acquisition.acquire_stem_image(
                    detector_type, ImageSize.PRESET_1024, dwell_time
                )
                return adorned.data
            # pass  # remove this line when uncommenting above

        # Simulation fallback
        self.warn_stream("Simulating image acquisition (AutoScript not connected)")
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
        """Acquire images from multiple detectors simultaneously."""

        if self._microscope is not None:
            # Real AutoScript
            detector_types = []
            for name in detector_names:
                if name == "haadf":
                    detector_types.append(DetectorType.HAADF)
                elif name == "bf":
                    detector_types.append(DetectorType.BF)
                # Add more detector types as needed

            # Create scan region
            custom_region = Region(
                RegionCoordinateSystem.RELATIVE,
                Rectangle(
                    scan_region[0],  # left
                    scan_region[1],  # top
                    scan_region[2],  # width
                    scan_region[3]   # height
                )
            )
        
            # TODO -----> handle segments

            settings = StemAcquisitionSettings(
                dwell_time=dwell_time,
                detector_types=detector_types,
                size=base_resolution,
                region=custom_region,
                auto_beam_blank=auto_beam_blank
            )
            
            return self._microscope.acquisition.acquire_stem_images_advanced(settings)
        
        # Simulation fallback
        self.warn_stream(f"Simulating acquisition for {detector_names}")
        rng = np.random.default_rng()
        
        # Calculate cropped dimensions based on scan_region
        height = int(base_resolution * scan_region[3])
        width = int(base_resolution * scan_region[2])
        
        return [rng.integers(0, 65535, size=(height, width), dtype=np.uint16) 
                for _ in detector_names]




# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    ThermoMicroscope.run_server()