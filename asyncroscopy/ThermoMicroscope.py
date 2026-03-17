"""
Microscope Tango device.

Owns the AutoScript connection and all acquisition commands.
Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

AutoScript is an optional dependency; this module imports cleanly without it
and falls back to simulated acquisition. To enable real hardware:

    pip install asyncroscopy[autoscript]

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
import math
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
    from autoscript_tem_microscope_client.enumerations import DetectorType, ImageSize, EdsDetectorType
    from autoscript_tem_microscope_client.enumerations import RegionCoordinateSystem, ExposureTimeType
    from autoscript_tem_microscope_client.structures import Region, Rectangle, AdornedSpectrum
    from autoscript_tem_microscope_client.structures import StemAcquisitionSettings, EdsAcquisitionSettings

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
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------

    autoscript_host_ip = device_property(
        dtype=str,
        default_value="10.46.217.241",
        doc="Hostname or IP of the AutoScript microscope server",
    )

    autoscript_host_port = device_property(
        dtype=int,
        default_value=9095,
        doc="Hostname or IP of the AutoScript microscope server",
    )

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

    def _connect(self):
        self._connect_hardware()
        self._connect_detector_proxies()
        self._link_hardware_attributes()
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
            "eds":  self.eds_device_address,
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

    def _link_hardware_attributes(self) -> None:
        # TODO: read off stage position and beta tilt enabled or not, set proxy attributes with these values.
        pass
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
        image_width: int,
        image_height: int,
        dwell_time: float,
    ) -> np.ndarray:
        """
        Call AutoScript acquisition and return numpy array.

        Falls back to a simulated image when AutoScript is unavailable.
        """
        if self._microscope is not None:
            # TODO move all of this to the detector
            if detector_name.upper() == "HAADF":
                haadf = self._detector_proxies['haadf']
                detector_type = DetectorType.HAADF # :TODO --> make it general and check
                dwell_time = haadf.read_attribute("dwell_time").value
                image_width = haadf.read_attribute("image_width").value
                if image_width == 256:
                    imsize = ImageSize.PRESET_256
                elif image_width == 512:
                    imsize = ImageSize.PRESET_512
                elif image_width == 1024:
                    imsize = ImageSize.PRESET_1024
                elif image_width == 2048:
                    imsize = ImageSize.PRESET_2048
                elif image_width == 4096:
                    imsize = ImageSize.PRESET_4096

                # take image
                adorned = self._microscope.acquisition.acquire_stem_image(
                    detector_type, imsize, dwell_time
                )
                # adorned.metadata TODO get this and pass it on
                return adorned.data

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

    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> np.ndarray:
        if detector_name.upper() == "EDS":
            # set up settings object
            settings = EdsAcquisitionSettings()
            settings.eds_detector = EdsDetectorType.SUPER_X
            settings.dispersion = 5 # int
            settings.shaping_time = 3e-6 # float
            settings.exposure_time = exposure_time
            settings.exposure_time_type = ExposureTimeType.LIVE_TIME

            # take eds
            spectrum = self._microscope.analysis.eds.acquire_spectrum(settings)
            handle_byte_order = True
            if handle_byte_order == True:
                dt = np.dtype("uint32").newbyteorder("<")
                spectrum = np.frombuffer(spectrum._raw_data, dtype=dt)

        else:
            print(f"Detector {detector_name} not supported for spectroscopy")

        return spectrum


    def _place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        x = float(position[0])
        y = float(position[1])
        print(x,y)
        self._microscope.optics.paused_scan_beam_position = [x, y]


    def _blank_beam(self) -> None:
        """blank beam"""
        self._microscope.optics.blanker.blank()


    def _unblank_beam(self) -> None:
        """
        unblank beam
        """
        self._microscope.optics.blanker.unblank()

    def _get_stage(self):
        """Get the current stage position as a list of floats [x, y, z, alpha, beta]."""
        position = self._microscope.specimen.stage.position

        # set proxy attributes with current stage position
        proxy = self._detector_proxies.get('stage')
        proxy.x = position[0]
        proxy.y = position[1]
        proxy.z = position[2]
        proxy.alpha = position[3]
        proxy.beta = position[4]
      
        return position
    
    def _move_stage(self, position) -> None:
        """Move stage to specified position [x, y, z, alpha, beta]."""
        proxy = self._detector_proxies.get('stage')

        beta_enabled = proxy.read_beta_tilt_enabled()

        x = float(position[0])
        y = float(position[1])
        z = float(position[2])
        alpha = float(position[3])
        beta = float(position[4])

        if beta_enabled:
            self._microscope.specimen.stage.absolute_move((x, y, z, math.radians(alpha), math.radians(beta)))
        else:
            self._microscope.specimen.stage.absolute_move((x, y, z, math.radians(alpha), None))

        # TODO: not sure how this will work with beta if that tilt is diabled - we will test
        # there are many other ways to impu this, using the TF StagePosition class, see their docs

        self._get_stage() # link the proxy with real state


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    ThermoMicroscope.run_server()