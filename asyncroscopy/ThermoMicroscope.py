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

import os
import math
from typing import Optional
import time

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
    from autoscript_tem_microscope_client.structures import StemAcquisitionSettings, EdsAcquisitionSettings, RunOptiStemSettings

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
        self.set_state(DevState.ON)
        self.screen_current_calibration = None

    def _connect_hardware(self) -> None:
        """Establish AutoScript connection from MPC -> hardware."""
        if not _AUTOSCRIPT_AVAILABLE or self.testing_mode_bool:
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

    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    # TODO:if self._microscope is not None: checks should go to init functions than sitting in commands 

    def _acquire_stem_image(self, imsize: int, dwell_time: float, detector_list: list) -> np.ndarray:
        """
        Call AutoScript acquisition and return numpy array.

        Falls back to a simulated image when AutoScript is unavailable.
        """
        if self._microscope is not None:
            # check detectors in detector_list
            detector_list = [d.upper() for d in detector_list] # must be caps for AutoScript
            detector_type = 'HAADF'

            # take image
            adorned = self._microscope.acquisition.acquire_stem_image(detector_type, imsize, dwell_time)
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
            # TODO: don't hardcode these
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
        if self._microscope is not None:
            x = float(position[0])
            y = float(position[1])
            print(x,y)
            self._microscope.optics.paused_scan_beam_position = [x, y]

    def _set_fov(self, fov) -> None:
        """set field of view in meters"""
        self._microscope.optics.scan_field_of_view = fov

    def _get_fov(self) -> float:
        """get field of view in meters"""
        return self._microscope.optics.scan_field_of_view

    def _blank_beam(self) -> None:
        """blank beam"""
        if self._microscope is not None:
            self._microscope.optics.blanker.blank()

    def _unblank_beam(self) -> None:
        """
        unblank beam
        """
        self._microscope.optics.blanker.unblank()

    def _caibrate_screen_current(self) -> None:
        original_gun_lens = self._microscope.optics.monochromator.focus
        gun_lens_series = np.linspace(-40, 100, 15)

        # series of measurements
        current_series = []
        for val in gun_lens_series:
            self._microscope.optics.monochromator.focus = val # original_gun_lens + val
            time.sleep(2)
            screen_current = self._microscope.detectors.screen.measure_current()
            current_series.append(screen_current)
        current_series = np.array(current_series) * 1e12
        self._microscope.optics.monochromator.focus = original_gun_lens

        # fit a polynomial and save:
        coeffs = np.polyfit(gun_lens_series, current_series, 11)
        poly_func = np.poly1d(coeffs)
        self.screen_current_calibration = poly_func

    def _set_screen_current(self, current) -> None:
        """set screen current in pA"""
        if self.screen_current_calibration is not None:
            poly_func = self.screen_current_calibration
            adjusted_poly = poly_func - current
            x_candidates = adjusted_poly.r
            x_real = x_candidates[np.isreal(x_candidates)].real
            x_real = np.max(x_real) # choose the largest real root as the gun lens value
            self._microscope.optics.monochromator.focus = float(x_real)
        else:
            self.warn_stream("Screen current calibration not available. running calibration (should take 15 seconds).")
            self._caibrate_screen_current()

            poly_func = self.screen_current_calibration
            adjusted_poly = poly_func - current
            x_candidates = adjusted_poly.r
            x_real = x_candidates[np.isreal(x_candidates)].real
            x_real = np.max(x_real) # choose the largest real root as the gun lens value
            self._microscope.optics.monochromator.focus = float(x_real)

    def _get_screen_current(self) -> float:
        """get screen current in pA"""
        screen_current = self._microscope.detectors.screen.measure_current() * 1e12
        return screen_current

    def _get_stage(self):
        """Get the current stage position as a list of floats [x, y, z, alpha, beta]."""
        # set proxy attributes with current stage position
        stage = self._detector_proxies['stage']

        position = self._microscope.specimen.stage.position
        position = np.array(position)

        stage.x = float(position[1])
        stage.y = float(position[0])
        stage.z = float(position[2])
        stage.alpha = float(math.degrees(position[3]))

        if position[4] is not None:
            return position
        else:
            return position[:4]

    def _move_stage(self, position) -> None:
        """Move stage to specified position [x, y, z, alpha, beta]."""
        x = float(position[0])
        y = float(position[1])
        z = float(position[2])
        alpha = float(math.radians(position[3]))
        beta = float(math.radians(position[4]))

        self._microscope.specimen.stage.absolute_move((x, y, z, alpha, beta))
        self._get_stage() # link the proxy with real state

    def _auto_focus(self):
        """Perform autofocus routine C1A1"""
        settings = RunOptiStemSettings(method='C1A1') #method=OptiStemMethod.C1_A1, dwell_time=2e-06, cutoff_in_pixels=5)
        self._microscope.auto_functions.run_opti_stem(settings)

    def _set_image_shift(self, shift):
        """Apply image shift in meters."""
        x_shift = float(shift[0])
        y_shift = float(shift[1])
        try:
            self._microscope.optics.deflectors.image_shift = (x_shift, y_shift)
        except Exception as e:
            self.error_stream(f"Failed to set image shift: {e}")

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    ThermoMicroscope.run_server()