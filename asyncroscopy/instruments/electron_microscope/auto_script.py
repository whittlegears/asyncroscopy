"""
STEMMicroscope Tango device.

Owns the AutoScript connection and all acquisition commands.
Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

AutoScript is an optional dependency; this module imports cleanly without it
and falls back to simulated acquisition. To enable real hardware:

    pip install asyncroscopy[autoscript]

Return convention for real image commands
-----------------------------------------
Real AutoScript image commands save the adorned object on disk and return the
DATA/Tiled unique id for that saved acquisition.
"""

import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import tango
from tango import AttrWriteType, DevState
from tango.server import attribute, command, device_property

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope
from asyncroscopy.data.data_writer import DEFAULT_ACQUISITION_DIR, save_acquisition

# AutoScript imports — only available on the microscope PC.
# Wrapped in try/except so the device can still be imported and tested
# on a development machine without AutoScript installed.
try:
    import autoscript_tem_microscope_client
    from autoscript_tem_microscope_client import TemMicroscopeClient
    from autoscript_tem_microscope_client.enumerations import EdsDetectorType
    from autoscript_tem_microscope_client.enumerations import RegionCoordinateSystem, ExposureTimeType
    from autoscript_tem_microscope_client.structures import Region, Rectangle
    from autoscript_tem_microscope_client.structures import StemAcquisitionSettings, EdsAcquisitionSettings, RunOptiStemSettings, CameraAcquisitionSettings
    from autoscript_tem_microscope_client.structures import RunObjectiveAutoStigmatorSettings

    _AUTOSCRIPT_AVAILABLE = True
except ImportError:
    _AUTOSCRIPT_AVAILABLE = False

from asyncroscopy.instruments.electron_microscope.limits import check_defocus, check_fov, check_scan

AUTOSCRIPT_STEM_OUTPUT_DIRECTORY = Path(r"Y:\AutoScript TEM")
# Screen-current calibration: monochromator focus series (V) and the wait for the current to settle at each step.
SCREEN_CALIBRATION_FOCUS_V = (10.0, 150.0, 15)
SCREEN_SETTLE_S = 1.0


class AutoScriptMicroscope(ElectronMicroscope):
    """
    Manages the AutoScript connection and exposes acquisition commands.
    Detector-specific settings (dwell time, resolution) are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    # ------------------------------------------------------------------
    # Device properties — configure in Tango DB per deployment
    # ------------------------------------------------------------------
    hardware_host = device_property(
        dtype=str,
        default_value="10.46.217.241",
        doc="Hostname or IP of the AutoScript microscope server",
    )
    hardware_port = device_property(
        dtype=int,
        default_value=9095,
        doc="Port of the AutoScript microscope server",
    )
    hardware_timeout_seconds = device_property(
        dtype=int,
        default_value=120,
        doc="Hardware connection timeout in seconds.",
    )
    acquisition_save_directory = device_property(
        dtype=str,
        default_value=DEFAULT_ACQUISITION_DIR,
        doc="Directory where AutoScript acquisitions are saved before the Tiled server serves them.",
    )
    acquisition_file_format = device_property(
        dtype=str,
        default_value="h5",
        doc="Acquisition file format. HDF5 stores acquisition data and parsed metadata attributes.",
    )
    data_device_address = device_property(
        dtype=str,
        default_value="",
        doc="Optional Tango device address for the DATA device, e.g. 'asyncroscopy/data/default'.",
    )
    # Software limits, checked before a value is sent; the same defaults as the digital twin's, so both refuse the
    # same values (tests/test_parity.py). Narrow them to the Spectra 300's working range at first light.
    min_fov = device_property(dtype=float, default_value=1e-9, doc="Smallest field of view accepted (m).")
    max_fov = device_property(dtype=float, default_value=1e-4, doc="Largest field of view accepted (m).")
    max_defocus = device_property(dtype=float, default_value=1e-5, doc="Largest |defocus| accepted (m).")
    nominal_beam_current_pa = device_property(
        dtype=float,
        default_value=50.0,
        doc="Reported as beam_current_pa, with beam_current_source 'default', until the screen current has been "
        "measured (insert the screen, then get_screen_current). Dose budgets on hardware refuse it.",
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

    fov = attribute(
        label="Field of View",
        dtype=float,
        access=AttrWriteType.READ,
        unit = "m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Current field of view in metres",
    )

    defocus = attribute(
        label="Defocus",
        dtype=float,
        access=AttrWriteType.READ,
        unit = "m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Current defocus in metres (e.g. 2e-8 for 20 nm)",
    )

    camera_length = attribute(
        label="Camera Length",
        dtype=float,
        access=AttrWriteType.READ,
        unit = "m",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Current camera length in meters",
    )

    beam_state = attribute(
        label="Beam State",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="Current beam state, either 'blanked' or 'unblanked'",
    )

    acceleration_voltage = attribute(
        label="Acceleration Voltage",
        dtype=float,
        access=AttrWriteType.READ,
        unit = "V",
        # min_value= TODO: set these
        # max_value= TODO: set these
        doc="Current acceleration voltage in volts",
    )


    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _connect(self):
        self._connect_error = None
        self._connect_hardware()
        self._connect_detector_proxies()
        self.screen_current_calibration = None
        self._measured_beam_current = None  # {"pa": ..., "at": time.time()} from the last screen-current reading
        # Without a client, report FAULT instead of looking connected.
        if self._connect_error:
            self.set_state(DevState.FAULT)
            self.set_status(f"Not connected to AutoScript: {self._connect_error}")
        else:
            self.set_state(DevState.ON)
            self.set_status("Connected to AutoScript" if self._microscope is not None else "Connected (testing mode)")

    def _connect_hardware(self) -> None:
        """Establish AutoScript connection from MPC -> hardware."""
        if self.testing_mode_bool:
            return
        if not _AUTOSCRIPT_AVAILABLE:
            self._connect_error = "autoscript_tem_microscope_client is not installed in this environment"
            self.warn_stream(f"AutoScript not available: {self._connect_error}")
            return
        try:
            self._microscope = TemMicroscopeClient()
            self._microscope.connect(self.hardware_host, self.hardware_port)
            self.info_stream(f"Connected to AutoScript at {self.hardware_host}:{self.hardware_port}")
            self.is_autoscript = True
        except Exception as e:
            self._connect_error = f"connecting to {self.hardware_host}:{self.hardware_port} failed: {e}"
            self.error_stream(f"AutoScript connection failed: {e}")
            self._microscope = None
            self.is_autoscript = False

    def _connect_detector_proxies(self) -> None:
        """Build DeviceProxy objects for each configured detector device."""
        # Extend this dict as more detectors are added
        # later, we want to do this automatically, not with a dictionary.
        addresses: dict[str, str] = {
            "eds": self.eds_device_address,
            "stage": self.stage_device_address,
            "scan": self.scan_device_address,
            "camera": self.camera_device_address,
            "corrector": self.corrector_device_address,
            "data": self.data_device_address,
        }
        for name, address in addresses.items():
            if not address:  # <-- minimal fix
                self.info_stream(f"Skipping {name}: no address configured")
                continue
            try:
                proxy = tango.DeviceProxy(address)
                proxy.set_timeout_millis(3_600_000 if name == "data" else 12_000)
                self._detector_proxies[name] = proxy
                self.info_stream(f"Connected to detector proxy: {name} @ {address}")
            except tango.DevFailed as e:
                self.error_stream(f"Failed to connect to {name} proxy at {address}: {e}")

    # ------------------------------------------------------------------
    # Attribute read methods
    # ------------------------------------------------------------------

    def read_manufacturer(self) -> bool:
        # TODO: query self._microscope.optics.mode when AutoScript available
        return self._manufacturer

    def read_fov(self) -> float:
        """Field of view in meters (STEM mode only)."""
        if self._microscope is None:
            return float("nan")
        return self._get_fov()

    def read_defocus(self) -> float:
        """Defocus in meters."""
        if self._microscope is None:
            return float("nan")
        return self._get_defocus()

    def read_acceleration_voltage(self) -> float:
        """Accelerating voltage in volts."""
        if self._microscope is None:
            return float("nan")
        return self._microscope.source.acceleration_voltage.value

    def read_camera_length(self) -> float:
        """Camera length in meters (only meaningful in DIFFRACTION mode)."""
        if self._microscope is None:
            return float("nan")
        return self._microscope.optics.camera_length.value.calibrated

    def read_beam_state(self) -> bool:
        """Beam blanked state: True when blanked, False when unblanked."""
        if self._microscope is None:
            return False
        return self._microscope.optics.blanker.is_beam_blanked

    # ------------------------------------------------------------------
    # Commands pertaining to setting children attributes, e.g. stage position, scan parameters, EDS settings, etc. --> iuser accesses it in a jupyter notebook using the device proxy
    # ------------------------------------------------------------------

    @command(dtype_out=tango.DevVarFloatArray)
    def register_stage(self):
        """Read the live stage position from the instrument: [x, y, z, alpha, beta], x/y/z in metres, tilts in degrees.

        Nothing is written back: the AutoScript STAGE device reads the same hardware, and a write there
        is a stage move. A STAGE device that reports another position (a cached settings device) is logged.
        """
        position = self._get_stage()
        stage = self._detector_proxies.get("stage")
        if stage is not None:
            try:
                reported = np.asarray(stage.position, dtype=float)
                if reported.shape != position.shape or not np.allclose(reported[:3], position[:3], atol=1e-9, equal_nan=True):
                    self.warn_stream(f"STAGE device reports {reported.tolist()}, the instrument {position.tolist()}")
            except (tango.DevFailed, AttributeError, TypeError, ValueError) as error:
                self.warn_stream(f"could not compare with the STAGE device: {error}")
        return position


    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ["haadf"],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = ".h5",
    ) -> str:
        """
        Call AutoScript scanned image acquisition, save it, and return its DATA/Tiled key.
        """
        scan_region = list(check_scan(imsize, dwell_time, scan_region))  # refused here, before the instrument
        detector_list = [d.upper() for d in detector_list]
        settings = StemAcquisitionSettings(dwell_time=dwell_time, detector_types=detector_list, size=imsize, region=Region(RegionCoordinateSystem.RELATIVE, Rectangle(*scan_region)))
        adorned = self._microscope.acquisition.acquire_stem_images_advanced(settings)
        if not isinstance(adorned, list):
            adorned = [adorned]
        data_server = self._detector_proxies.get("data")
        return save_acquisition(self, data_server, "stem_image", detector_list, adorned, output_format=output_format)


    def _acquire_camera_image(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        output_format: str = ".h5",
    ) -> str:
        """
        Call advanced AutoScript camera acquisition, save the adorned image,
        and return its DATA/Tiled key.
        """
        camera_detector = {"flucam": "Flucam", "bm-ceta": "BM-Ceta"}.get(detector.lower(), detector)
        fixed_readout_area = {
            "full": "Full",
            "half": "Half",
            "quarter": "Quarter",
        }.get(readout_area.lower(), readout_area)
        settings = CameraAcquisitionSettings(
            camera_detector=camera_detector,
            size=imsize,
            exposure_time=exposure_time,
            fixed_readout_area=fixed_readout_area,
            frame_combining=frame_combining,
        )
        adorned = self._microscope.acquisition.acquire_camera_image_advanced(settings)
        data_server = self._detector_proxies.get("data")
        return save_acquisition(
            self,
            data_server,
            "camera_image",
            str(detector),
            adorned,
            output_format=output_format,
        )


    # the following file is COMPLETELY CORRECT, except the AutoScript acquire_stem_data_advanced is currently trash.
    # it only records the origin corner of the camera, which is covered by the HAADF and ureachable at short camera lengths.
    # waiting on an email back with a fix. -Austin Houston, Aug 11th 2026
    '''
    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """Acquire AutoScript 4D-STEM data and return its registered HDF5 key."""
        data_server = self._detector_proxies.get("data")
        if not AUTOSCRIPT_STEM_OUTPUT_DIRECTORY.is_dir():
            raise FileNotFoundError(f"AutoScript STEM output directory is not readable: {AUTOSCRIPT_STEM_OUTPUT_DIRECTORY}")

        files_before = set(AUTOSCRIPT_STEM_OUTPUT_DIRECTORY.glob("*.mrc"))
        region = Region(RegionCoordinateSystem.RELATIVE, Rectangle(*scan_region))
        settings = StemDataSettings(dwell_time=dwell_time, detector_types=[CameraType.BM_CETA], size=imsize, region=region)
        self._microscope.acquisition.acquire_stem_data_advanced(settings)
        for _ in range(11):
            new_files = set(AUTOSCRIPT_STEM_OUTPUT_DIRECTORY.glob("*.mrc")) - files_before
            if new_files:
                break
            time.sleep(0.5)

        if len(new_files) != 1:
            raise RuntimeError(f"Expected one new AutoScript MRC file, found {len(new_files)}")

        source_path = new_files.pop()
        request = {
            "source_path": str(source_path),
            "detector": detector,
            "scan_shape": [int(imsize), int(imsize)],
            "dwell_time": float(dwell_time),
            "scan_region": list(scan_region),
        }
        return data_server.copy_and_register_remote_file(json.dumps(request))
        '''

    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """Acquire 4D-STEM data by placing the probe and recording one Ceta image per scan point."""
        scan_size = int(imsize)
        left, top, width, height = [float(value) for value in scan_region]
        x_positions = left + (np.arange(scan_size) + 0.5) * width / scan_size
        y_positions = top + (np.arange(scan_size) + 0.5) * height / scan_size
        camera_detector = {"flucam": "Flucam", "bm-ceta": "BM-Ceta"}.get(detector.lower(), detector)
        settings = CameraAcquisitionSettings(camera_detector=camera_detector, size=256, exposure_time=dwell_time, fixed_readout_area="Half")
        starting_beam_position = self._microscope.optics.paused_scan_beam_position
        frames = []
        try:
            for y_position in y_positions:
                for x_position in x_positions:
                    self._microscope.optics.paused_scan_beam_position = [float(x_position), float(y_position)]
                    image = self._microscope.acquisition.acquire_camera_image_advanced(settings)
                    frames.append(np.array(image.data, copy=True))
        finally:
            self._microscope.optics.paused_scan_beam_position = starting_beam_position

        stem_data = np.stack(frames).reshape(scan_size, scan_size, 256, 256)
        attrs = {"scan_shape": [scan_size, scan_size], "diffraction_shape": [256, 256], "dwell_time": float(dwell_time), "scan_region": [left, top, width, height], "readout_area": "Half"}
        data_server = self._detector_proxies.get("data")
        return save_acquisition(self, data_server, "stem_data", camera_detector, stem_data, dataset_name="stem_data", dataset_attrs=attrs)


    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        settings = EdsAcquisitionSettings()
        settings.eds_detector = EdsDetectorType.SUPER_X
        settings.dispersion = 5
        settings.shaping_time = 3e-6
        settings.exposure_time = exposure_time
        settings.exposure_time_type = ExposureTimeType.LIVE_TIME
        spectrum = self._microscope.analysis.eds.acquire_spectrum(settings)
        data_server = self._detector_proxies.get("data")
        return save_acquisition(self, data_server, "spectrum", detector_name, spectrum, dataset_name="spectrum")

    def _place_beam(self, position) -> None:
        """
        sets resting beam position, [0:1]
        """
        if self._microscope is not None:
            x = float(position[0])
            y = float(position[1])
            self._microscope.optics.paused_scan_beam_position = [x, y]

    def _is_stem(self) -> bool:
        """True in STEM mode. AutoScript's OpticalMode.STEM is "Stem"; compare without case, in one place."""
        return str(self._microscope.optics.optical_mode).lower() == "stem"

    def _is_diffraction(self) -> bool:
        """True when the projector is in diffraction (ProjectorMode.DIFFRACTION, "Diffraction")."""
        return str(self._microscope.optics.projector_mode).lower() == "diffraction"

    def _set_fov(self, fov) -> None:
        """set field of view in meters"""
        self._microscope.optics.scan_field_of_view = check_fov(fov, self.min_fov, self.max_fov)

    def _get_fov(self) -> float:
        """get field of view in meters"""
        return self._microscope.optics.scan_field_of_view

    def _set_column_valves(self, state: str) -> None:
        """Set column valves state."""
        if self._microscope is not None:
            if state == "open":
                self._microscope.vacuum.column_valves.open()
            elif state == "close":
                self._microscope.vacuum.column_valves.close()
            else:
                print(f"Invalid valve state '{state}'. Use 'open' or 'close'.")

    def _blank_beam(self) -> None:
        """blank beam"""
        if self._microscope is not None:
            self._microscope.optics.blanker.blank()

    def _unblank_beam(self) -> None:
        """
        unblank beam
        """
        self._microscope.optics.blanker.unblank()
    def _set_defocus(self, defocus) -> None:
        """Set defocus in meters."""
        defocus = check_defocus(defocus, self.max_defocus)  # refused here, before the instrument
        optics = getattr(self._microscope, "optics", None)
        if optics is None:
            raise AttributeError("Microscope object has no optics")

        if hasattr(optics, "defocus"):
            optics.defocus = defocus
            return

        if self._is_stem():
            optics.focusing.stem.objective.defocus = defocus
        else:
            optics.focusing.tem.defocus = defocus

    def _get_defocus(self) -> float:
        """Get defocus in meters."""
        optics = getattr(self._microscope, "optics", None)
        if optics is None:
            raise AttributeError("Microscope object has no optics")

        if hasattr(optics, "defocus"):
            return float(optics.defocus)

        if self._is_stem():
            return float(optics.focusing.stem.objective.defocus)
        return float(optics.focusing.tem.defocus)
    
    def _set_screen(self, position: str)->None:
        if position.lower() in ['in', 'insert', 'inserted']:
            if self._microscope.detectors.screen.position == 'Retracted':
                self._microscope.detectors.screen.insert()
        elif position.lower() in ['out', 'retract', 'retracted']:
             if self._microscope.detectors.screen.position == 'Inserted':
                self._microscope.detectors.screen.retract()


    def _calibrate_screen_current(self) -> None:
        """Calibrate the screen current (pA) against the monochromator focus (V); the screen must be inserted."""
        if self._microscope.detectors.screen.position != "Inserted":
            raise ValueError("insert the screen (set_screen 'inserted') before calibrating the screen current")
        self._measured_beam_current = None  # the series below sweeps the monochromator focus
        original_gun_lens = self._microscope.optics.monochromator.focus
        gun_lens_series = np.linspace(*SCREEN_CALIBRATION_FOCUS_V)

        # series of measurements
        current_series = []
        try:
            for val in gun_lens_series:
                self._microscope.optics.monochromator.focus = float(val)
                time.sleep(SCREEN_SETTLE_S)
                current_series.append(self._microscope.detectors.screen.measure_current())
        finally:
            self._microscope.optics.monochromator.focus = original_gun_lens
        current_series = np.array(current_series) * 1e12

        # fit a polynomial and save, with the span it is valid over:
        coeffs = np.polyfit(gun_lens_series, current_series, 11)
        self.screen_current_calibration = np.poly1d(coeffs)
        self.screen_current_calibration_range = {
            "focus_v": (float(gun_lens_series.min()), float(gun_lens_series.max())),
            "current_pa": (float(current_series.min()), float(current_series.max())),
        }

    def _screen_current_focus(self, current: float) -> float:
        """The monochromator focus (V) that gives ``current`` pA, refused unless it lies inside the calibrated span.

        The calibration is a degree-11 polynomial through 15 points: outside the measured focus span its
        roots are extrapolation, and writing one would be an unbounded write to the gun.
        """
        span = self.screen_current_calibration_range
        low_v, high_v = span["focus_v"]
        low_pa, high_pa = span["current_pa"]
        current = float(current)
        if not (math.isfinite(current) and low_pa <= current <= high_pa):
            raise ValueError(
                f"screen current {current:g} pA is outside the calibrated range {low_pa:.4g} to {high_pa:.4g} pA "
                f"(monochromator focus {low_v:g} to {high_v:g} V); nothing was written"
            )
        roots = (self.screen_current_calibration - current).r
        real = roots[np.abs(roots.imag) <= 1e-6 * np.maximum(1.0, np.abs(roots.real))].real
        inside = real[(real >= low_v) & (real <= high_v)]
        if inside.size == 0:
            raise ValueError(
                f"no monochromator focus inside the calibrated {low_v:g} to {high_v:g} V gives {current:g} pA "
                f"(roots {np.round(real, 3).tolist()}); recalibrate the screen current; nothing was written"
            )
        return float(np.max(inside))  # the largest in-span root, as before

    def _set_screen_current(self, current) -> None:
        """set screen current in pA"""
        self._measured_beam_current = None  # the monochromator focus sets the current: the old reading no longer holds
        if self.screen_current_calibration is None:
            self.warn_stream("Screen current calibration not available. running calibration (should take 15 seconds).")
            self._calibrate_screen_current()
        self._microscope.optics.monochromator.focus = self._screen_current_focus(current)

    def _current_conditions(self) -> dict | None:
        """What sets the beam current, as AutoScript reports it: the monochromator focus, spot size, optical and
        illumination mode, high tension and the aperture mechanisms. None when any of it cannot be read."""
        try:
            optics = self._microscope.optics
            conditions = {
                "monochromator_focus_v": float(optics.monochromator.focus),
                "spot_size": optics.spot_size_index,
                "optical_mode": str(optics.optical_mode),
                "illumination_mode": str(optics.illumination_mode),
                "acceleration_voltage": float(optics.acceleration_voltage.value),
            }
            for mechanism_type in optics.aperture_mechanisms.get_available():
                mechanism = optics.aperture_mechanisms.get_mechanism(mechanism_type)
                if not mechanism.is_enabled:
                    conditions[f"aperture_{mechanism_type}"] = "Disabled"
                elif mechanism.insertion_state == "Retracted":
                    conditions[f"aperture_{mechanism_type}"] = "Retracted"
                else:
                    conditions[f"aperture_{mechanism_type}"] = str(mechanism.aperture.name)
            return conditions
        except Exception as error:  # cannot tell what the current depends on: no reading can be trusted
            self.warn_stream(f"could not read the optics that set the beam current: {error}")
            return None

    def _get_screen_current(self) -> float:
        """get screen current in pA; with the screen inserted it is the beam current, kept for get_parameters
        together with the optics it was measured under, so a later change to them makes it stale"""
        screen_current = self._microscope.detectors.screen.measure_current() * 1e12
        if screen_current > 0 and self._microscope.detectors.screen.position == "Inserted":
            self._measured_beam_current = {"pa": float(screen_current), "at": time.time(), "conditions": self._current_conditions()}
        return screen_current

    def _beam_current(self) -> dict:
        """beam_current_pa with its provenance: "measured" from the screen, or "default" with the reason.

        A reading counts as measured only while the optics that set the current are as they were when it was
        taken (``_current_conditions``): a spot size, aperture, mode, high tension or monochromator change made
        by any device, script or person makes it stale, and it is reported as "default" again."""
        measured = getattr(self, "_measured_beam_current", None)
        reason = "no screen-current measurement since connecting: insert the screen, call get_screen_current, then retract it"
        if measured:
            then = measured.get("conditions")
            now = self._current_conditions() if then is not None else None
            if then is not None and now == then:
                return {
                    "beam_current_pa": measured["pa"],
                    "beam_current_source": "measured",
                    "beam_current_age_s": round(time.time() - measured["at"], 1),
                }
            if then is None or now is None:
                reason = "the optics that set the beam current could not be read, so the last screen reading cannot be trusted: measure it again"
            else:
                changed = sorted(k for k in set(then) | set(now) if then.get(k) != now.get(k))
                reason = f"the optics changed since the screen current was measured ({', '.join(changed)}): measure it again"
        return {
            "beam_current_pa": float(self.nominal_beam_current_pa),
            "beam_current_source": "default",
            "beam_current_reason": reason,
        }

    def _get_stage(self) -> np.ndarray:
        """The instrument's stage position [x, y, z, alpha, beta]: x/y/z in metres, tilts in degrees.

        Read from the AutoScript client itself, so it is right (and works) whatever the STAGE device holds.
        """
        position = self._microscope.specimen.stage.position
        tilts = [float("nan") if v is None else math.degrees(float(v)) for v in (position.a, position.b)]
        if math.isnan(tilts[1]):
            tilts[1] = 0.0  # single-tilt holder: no beta
        return np.array([float(position.x), float(position.y), float(position.z), *tilts], dtype=np.float64)

    def _get_parameters(self):
        optics = self._microscope.optics
        status = {'system': self._microscope.service.system.name,
                'vacuum': self._microscope.vacuum.state,
                'column_valves': self._microscope.vacuum.column_valves.state,
                'is_accelerator_on': optics.is_accelerator_on,
                'acceleration_voltage': optics.acceleration_voltage.value,
                'optical_mode': optics.optical_mode,
                'illumination_mode': optics.illumination_mode,
                'objective_lens_mode': optics.objective_lens_mode,
                'projector_mode': optics.projector_mode,
                #'convergence_angle': optics.convergence_angle,
                'spot_size': optics.spot_size_index,
                'beam_stopper': optics.beam_stopper.insertion_state,
                'beam_blanker': optics.blanker.is_beam_blanked,
                'is_eftem_on': optics.is_eftem_on,
                }
        stem = self._is_stem()
        if stem:
            status['scan_rotation'] = optics.scan_rotation
            status['scan_field_of_view'] = optics.scan_field_of_view
        for mechanism_type in optics.aperture_mechanisms.get_available():
            mechanism = optics.aperture_mechanisms.get_mechanism(mechanism_type)
            if not mechanism.is_enabled:
                status[mechanism_type] = 'Disabled'
            elif mechanism.insertion_state == 'Retracted':
                status[mechanism_type] = 'Retracted'
            else:
                status[mechanism_type] = mechanism.aperture.name
        for deflector in optics.deflectors.get_available_deflectors():
            defl = optics.deflectors.get_deflector_value(deflector)
            status[deflector] = [defl.x, defl.y]
        # The keys every microscope backend reports, with the twin's names and units (tests/test_parity.py).
        status.update({
            'stem_mode': stem,
            'defocus_m': self._get_defocus(),
            'fov_m': float(status['scan_field_of_view']) if stem else None,
            'stage_position': self._get_stage().tolist(),
            'beam_blanked': bool(status['beam_blanker']),
            **self._beam_current(),
        })
        return json.dumps(status)

    def _move_stage(self, position) -> None:
        """Move stage to [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        target = [float(v) for v in position]
        if len(target) != 5:
            raise ValueError("Stage position must have 5 values: [x, y, z, alpha, beta]")
        if not all(math.isfinite(v) for v in target):
            raise ValueError(f"Stage position must be finite numbers, got {target}")
        stage_proxy = self._detector_proxies.get("stage")
        if stage_proxy is None:
            raise ValueError("no STAGE device is configured (stage_device_address): it checks the travel limits, so moves go through it")
        stage_proxy.position = target  # the STAGE device refuses a move outside its limits

    def _auto_focus(self):
        """Run the vendor autofocus: OptiSTEM C1A1 in STEM, the objective auto-stigmator (Flucam) in TEM."""
        if self._is_stem():
            settings = RunOptiStemSettings(method="C1A1")
            self._microscope.auto_functions.run_opti_stem(settings)
        else:
            settings = RunObjectiveAutoStigmatorSettings(camera_detector="Flucam")
            self._microscope.auto_functions.run_objective_auto_stigmator(settings)

    # The shift and tilt setters let AutoScript's errors through: a write the instrument refuses must
    # reach the caller (and the job that sent it), not be logged while the command reports success.
    def _set_image_shift(self, shift):
        """Apply image shift in meters (the beam shift in STEM)."""
        x_shift = float(shift[0])
        y_shift = float(shift[1])
        if self._is_stem():
            self._microscope.optics.deflectors.beam_shift = (x_shift, y_shift)
        else:
            self._microscope.optics.deflectors.image_shift = (x_shift, y_shift)

    def _set_diffraction_shift(self, shift):
        """Apply diffraction shift; only in diffraction (the projector's mode, as the getter reads it)."""
        x_shift = float(shift[0])
        y_shift = float(shift[1])
        if not self._is_diffraction():
            raise ValueError("diffraction shift needs the projector in Diffraction mode")
        self._microscope.optics.deflectors.image_shift = (x_shift, y_shift)

    def _get_diffraction_shift(self):
        if self._is_diffraction():
            position = self._microscope.optics.deflectors.image_shift
            return np.array([position.x, position.y])
        else:
            return np.array([0, 0])


    def _get_image_shift(self):
        if self._is_stem():
            position = self._microscope.optics.deflectors.beam_shift
        else:
            position = self._microscope.optics.deflectors.image_shift
        return np.array([position.x, position.y])

    def _get_beam_tilt(self):
        tilt = self._microscope.optics.deflectors.beam_tilt
        return np.array([tilt.x, tilt.y])

    def _set_beam_tilt(self, tilt):
        """Apply beam tilt (AutoScript: unitless, -1 to +1 for x and y)."""
        x_tilt = float(tilt[0])
        y_tilt = float(tilt[1])
        self._microscope.optics.deflectors.beam_tilt = (x_tilt, y_tilt)


def _requires_microscope(method):
    """Raise NotConnected instead of an AttributeError when there is no client."""

    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        # Testing mode has no client by design.
        if self._microscope is None and not getattr(self, "testing_mode_bool", False):
            reason = getattr(self, "_connect_error", None) or "the device has not connected"
            if not _AUTOSCRIPT_AVAILABLE:
                advice = "Reconnecting will not help; report it to the operator."
            else:
                advice = "Call Connect once; if it fails again, report it to the operator."
            tango.Except.throw_exception(
                "NotConnected",
                f"Not connected to AutoScript: {reason}. {advice}",
                f"{method.__name__}()",
            )
        return method(self, *args, **kwargs)

    return guarded


for _name in (
    "_acquire_scanned_image", "_acquire_camera_image", "_acquire_scanned_data_advanced", "_acquire_spectrum",
    "_place_beam", "_set_fov", "_get_fov", "_set_column_valves", "_blank_beam", "_unblank_beam",
    "_set_defocus", "_get_defocus", "_set_screen", "_calibrate_screen_current", "_set_screen_current",
    "_get_screen_current", "_auto_focus", "_set_image_shift", "_get_image_shift",
    "_set_diffraction_shift", "_get_diffraction_shift", "_get_beam_tilt", "_set_beam_tilt",
    "_get_parameters", "_get_stage",
):
    setattr(AutoScriptMicroscope, _name, _requires_microscope(getattr(AutoScriptMicroscope, _name)))

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    AutoScriptMicroscope.run_server()
