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

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import tango
from tango import AttrWriteType, DevState
from tango.server import attribute, command, device_property

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope, hardware_call, unsupported
from asyncroscopy.data.data_writer import DEFAULT_ACQUISITION_DIR, save_acquisition

# AutoScript imports — only available on the microscope PC.
# Wrapped in try/except so the device can still be imported and tested
# on a development machine without AutoScript installed.
try:
    from autoscript_tem_microscope_client import TemMicroscopeClient
    from autoscript_tem_microscope_client.enumerations import EdsDetectorType
    from autoscript_tem_microscope_client.enumerations import RegionCoordinateSystem, ExposureTimeType
    from autoscript_tem_microscope_client.structures import Region, Rectangle
    from autoscript_tem_microscope_client.structures import StemAcquisitionSettings, EdsAcquisitionSettings, RunOptiStemSettings, CameraAcquisitionSettings
    from autoscript_tem_microscope_client.structures import RunObjectiveAutoStigmatorSettings

    _AUTOSCRIPT_AVAILABLE = True
except ImportError:
    _AUTOSCRIPT_AVAILABLE = False

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
    # Not applied: AutoScript 1.18's client waits for every reply with no time limit and offers none to set
    # (autoscript_core ClientEndpoint.perform_call; connect has none either). A hung AutoScript server therefore
    # holds the call until the server answers or the connection drops. Kept so configs that set timeout_seconds load.
    hardware_timeout_seconds = device_property(
        dtype=int,
        default_value=120,
        doc="Not applied: the AutoScript 1.18 client has no call or connect timeout to set.",
    )
    stage_timeout_seconds = device_property(
        dtype=float,
        default_value=180.0,
        doc="How long a call to the STAGE device may take (s). AutoScript stage moves return on arrival, and a "
        "large tilt takes far longer than other calls; too short, and a move reports failure while the stage still moves.",
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
    # The software limits (min_fov, max_fov, max_defocus, max_stigmator) are the ElectronMicroscope base's: narrow
    # them to the Spectra 300's working range at first light.
    # 4D-STEM parks the probe and records one camera frame per scan point: bound it until first light has timed it.
    max_4dstem_imsize = device_property(
        dtype=int, default_value=64,
        doc="Largest 4D-STEM scan accepted, per side (64: 4096 camera frames). Larger scans are refused.",
    )
    min_4dstem_exposure_s = device_property(
        dtype=float, default_value=1e-3,
        doc="Shortest camera exposure per 4D-STEM point (s), taken from the CAMERA device's exposure_time.",
    )
    objective_stigmator_m_per_unit = device_property(
        dtype=float,
        default_value=0.0,
        doc="Metres of A1 per AutoScript objective-stigmator unit (AutoScript sets the stigmator in optical units, "
        "about -1 to 1). 0, the default, until first light has measured it: get_stigmator and set_stigmator are refused.",
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

    def _init_device_attributes(self) -> None:
        # The server runs without Tango's per-device lock (serve, below), so that blank_beam is never queued
        # behind a long command. Every other hardware call takes the base's command lock instead (hardware_call).
        super()._init_device_attributes()
        self._blank_client = None  # a second AutoScript connection, used only to blank the beam
        self._blank_count = 0  # counts blank_beam calls: a running 4D-STEM scan stops when it changes

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
        elif self._microscope is not None and self._blank_client is None:
            self.set_state(DevState.ALARM)
            self.set_status("Connected to AutoScript, but the separate connection for blank_beam failed: "
                            "an e-stop blank waits behind any AutoScript call already running")
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
            return
        # The AutoScript client runs one call at a time, so a blank sent on the main connection waits for
        # a running acquisition or stage move to end. blank_beam uses a connection of its own.
        try:
            self._blank_client = TemMicroscopeClient()
            self._blank_client.connect(self.hardware_host, self.hardware_port)
        except Exception as e:
            self.error_stream(f"AutoScript connection for blank_beam failed: {e}")
            self._blank_client = None

    def _proxy_timeout_ms(self, name: str) -> int:
        # AutoScript stage moves return on arrival, and a large tilt takes far longer than other calls.
        if name == "stage":
            return int(1000 * float(self.stage_timeout_seconds))
        return super()._proxy_timeout_ms(name)

    def _check_connected(self, command_name: str) -> None:
        """Raise NotConnected instead of an AttributeError when there is no client (testing mode has none by design)."""
        if self._microscope is not None or getattr(self, "testing_mode_bool", False):
            return
        reason = getattr(self, "_connect_error", None) or "the device has not connected"
        if not _AUTOSCRIPT_AVAILABLE:
            advice = "Reconnecting will not help; report it to the operator."
        else:
            advice = "Call Connect once; if it fails again, report it to the operator."
        tango.Except.throw_exception("NotConnected", f"Not connected to AutoScript: {reason}. {advice}", f"{command_name}()")

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
    @hardware_call()
    def register_stage(self) -> np.ndarray:
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
        detector_list = [d.upper() for d in detector_list]
        attrs = self._frame_attrs(imsize, dwell_time, scan_region)  # before the frame: where and how it is taken
        settings = StemAcquisitionSettings(dwell_time=dwell_time, detector_types=detector_list, size=imsize, region=Region(RegionCoordinateSystem.RELATIVE, Rectangle(*scan_region)))
        with self._beam_restored():
            adorned = self._microscope.acquisition.acquire_stem_images_advanced(settings)
        if not isinstance(adorned, list):
            adorned = [adorned]
        data_server = self._detector_proxies.get("data")
        return save_acquisition(self, data_server, "stem_image", detector_list, adorned, dataset_attrs=attrs, output_format=output_format)

    def _frame_attrs(self, imsize: int, dwell_time: float, scan_region: list[float]) -> dict:
        """What a saved frame records about where and how it was taken (the dose ledger reads it). A stage or field of
        view that cannot be read is said in "frame_attrs_error", and the frame is then not in the ledger."""
        try:
            stage = [float(v) for v in self._get_stage()]
            fov = float(self._get_fov())
            current = self.beam_current_fields()
        except Exception as exc:
            return {"frame_attrs_error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"}
        try:  # the field's centre is the stage plus this; a frame without it is placed as if it were zero
            shift = {"image_shift_m": [float(v) for v in self._get_image_shift()][:2]}
        except Exception as exc:
            self.debug_stream(f"frame recorded without image_shift_m: {exc}")
            shift = {}
        return {
            "stage_reported_m": stage, "fov_m": fov, "pixel_size_m": fov / int(imsize), "dwell_time_s": float(dwell_time),
            "scan_region": list(scan_region), **shift,
            **{k: v for k, v in current.items() if k in ("beam_current_pa", "beam_current_source")},
        }


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
        with self._beam_restored():
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
        """Acquire 4D-STEM data by placing the probe and recording one Ceta image per scan point.

        The scan is checked like any scan and refused above max_4dstem_imsize per side. Each frame's exposure is the
        CAMERA device's exposure_time (at least min_4dstem_exposure_s), never the STEM dwell. A beam blanked before
        the call is blanked again after every frame, so it is off while the probe moves; a blank_beam from any
        client during the scan (an e-stop) stops it before the next frame.
        """
        left, top, width, height = (float(v) for v in scan_region)  # checked by the base before it got here
        scan_size = int(imsize)
        if scan_size > int(self.max_4dstem_imsize):
            raise ValueError(f"4D-STEM scan {scan_size}x{scan_size} is larger than max_4dstem_imsize {int(self.max_4dstem_imsize)} per side; nothing was acquired")
        camera = self._detector_proxies.get("camera")
        if camera is None:
            raise ValueError("4D-STEM needs a CAMERA device (camera_device_address) for its exposure time; nothing was acquired")
        exposure_time = float(camera.exposure_time)
        if not (math.isfinite(exposure_time) and exposure_time >= float(self.min_4dstem_exposure_s)):
            raise ValueError(f"4D-STEM camera exposure {exposure_time:g} s is below min_4dstem_exposure_s {float(self.min_4dstem_exposure_s):g} s; "
                             "set the CAMERA device's exposure_time; nothing was acquired")
        x_positions = left + (np.arange(scan_size) + 0.5) * width / scan_size
        y_positions = top + (np.arange(scan_size) + 0.5) * height / scan_size
        camera_detector = {"flucam": "Flucam", "bm-ceta": "BM-Ceta"}.get(detector.lower(), detector)
        settings = CameraAcquisitionSettings(camera_detector=camera_detector, size=256, exposure_time=exposure_time, fixed_readout_area="Half")
        starting_beam_position = self._microscope.optics.paused_scan_beam_position
        blanks_before = self._blank_count
        frames = []
        try:
            for y_position in y_positions:
                for x_position in x_positions:
                    if self._blank_count != blanks_before:
                        raise RuntimeError(f"4D-STEM stopped after {len(frames)} of {scan_size * scan_size} frames: "
                                           "blank_beam was called during the scan (e-stop); nothing was saved")
                    self._microscope.optics.paused_scan_beam_position = [float(x_position), float(y_position)]
                    with self._beam_restored():
                        image = self._microscope.acquisition.acquire_camera_image_advanced(settings)
                    frames.append(np.array(image.data, copy=True))
        finally:
            self._microscope.optics.paused_scan_beam_position = starting_beam_position

        stem_data = np.stack(frames).reshape(scan_size, scan_size, 256, 256)
        attrs = {"scan_shape": [scan_size, scan_size], "diffraction_shape": [256, 256], "dwell_time": float(dwell_time),
                 "exposure_time": exposure_time, "scan_region": [left, top, width, height], "readout_area": "Half"}
        data_server = self._detector_proxies.get("data")
        return save_acquisition(self, data_server, "stem_data", camera_detector, stem_data, dataset_name="stem_data", dataset_attrs=attrs)


    def _acquire_spectrum(self, detector_name: str, exposure_time: float) -> str:
        settings = EdsAcquisitionSettings()
        settings.eds_detector = EdsDetectorType.SUPER_X
        settings.dispersion = 5
        settings.shaping_time = 3e-6
        settings.exposure_time = exposure_time
        settings.exposure_time_type = ExposureTimeType.LIVE_TIME
        with self._beam_restored():
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
        self._microscope.optics.scan_field_of_view = float(fov)

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
        """Blank the beam (the blank_beam command, and the e-stop's). It takes no command lock and uses its own
        AutoScript connection, so it does not wait for a running acquisition or stage move; it also tells a running
        4D-STEM scan to stop."""
        if self._microscope is None:  # testing mode, which has no client by design (_check_connected refuses otherwise)
            return
        self._blank_count += 1
        self._blank_now()

    def _blank_now(self) -> None:
        """Blank the beam on the blank connection (the main one when that failed to connect)."""
        client = self._blank_client if self._blank_client is not None else self._microscope
        client.optics.blanker.blank()

    @contextlib.contextmanager
    def _beam_restored(self):
        """Around one acquisition: a beam that was blanked before it, or that blank_beam blanked during it (an
        e-stop), is blanked again after it, even when it fails. An acquisition may open the blanker; the save that
        follows must not leave the beam parked on the sample. A beam that was unblanked is otherwise left as it is
        (the caller opened it, and closes it)."""
        was_blanked = bool(self._microscope.optics.blanker.is_beam_blanked)
        blanks_before = self._blank_count
        try:
            yield
        finally:
            if was_blanked or self._blank_count != blanks_before:
                self._blank_now()

    def _unblank_beam(self) -> None:
        """
        unblank beam
        """
        self._microscope.optics.blanker.unblank()
    def _set_defocus(self, defocus) -> None:
        """Set defocus in meters (checked against max_defocus by the base)."""
        defocus = float(defocus)
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
        """The beam current with its provenance: "measured" from the screen, or "default" with the reason.

        A reading counts as measured only while the optics that set the current are as they were when it was
        taken (``_current_conditions``): a spot size, aperture, mode, high tension or monochromator change made
        by any device, script or person makes it stale, and it is reported as "default" again."""
        measured = getattr(self, "_measured_beam_current", None)
        reason = "no screen-current measurement since connecting: insert the screen, call get_screen_current, then retract it"
        if measured:
            then = measured.get("conditions")
            now = self._current_conditions() if then is not None else None
            if then is not None and now == then:
                return {"pa": measured["pa"], "source": "measured", "age_s": round(time.time() - measured["at"], 1)}
            if then is None or now is None:
                reason = "the optics that set the beam current could not be read, so the last screen reading cannot be trusted: measure it again"
            else:
                changed = sorted(k for k in set(then) | set(now) if then.get(k) != now.get(k))
                reason = f"the optics changed since the screen current was measured ({', '.join(changed)}): measure it again"
        return {"pa": float(self.default_beam_current_pa), "source": "default", "age_s": None, "reason": reason}

    def _stage_moving(self) -> bool:
        return bool(self._microscope.specimen.stage.is_moving)

    def _stigmator_scale(self) -> float:
        scale = float(self.objective_stigmator_m_per_unit)
        if not scale > 0:
            unsupported(self, "the stigmator until objective_stigmator_m_per_unit is set (first light measures it)")
        return scale

    def _get_stigmator(self) -> list[float]:
        scale = self._stigmator_scale()
        value = self._microscope.optics.stigmators.objective_stigmator
        return [float(value.x) * scale, float(value.y) * scale]

    def _set_stigmator(self, stigmator: list[float]) -> None:
        scale = self._stigmator_scale()
        units = [float(v) / scale for v in stigmator]
        if any(abs(u) > 1.0 for u in units):
            raise ValueError(f"stigmator {list(stigmator)} m is {units} AutoScript units, outside -1 to 1; nothing was set")
        self._microscope.optics.stigmators.objective_stigmator = units

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
            **self.beam_current_fields(),
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


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

def _without_tango_serialization() -> None:
    """Turn off Tango's per-device command lock in this server, before any device starts: a long acquisition
    or stage move would otherwise hold it, and an e-stop's blank_beam would queue behind it until it timed out.
    The device serializes its own hardware calls instead (the base's command lock, hardware_call)."""
    tango.Util.instance().set_serial_model(tango.SerialModel.NO_SYNC)


def serve() -> None:
    """Run the AutoScriptMicroscope device server, without Tango's per-device command lock."""
    AutoScriptMicroscope.run_server(pre_init_callback=_without_tango_serialization)


if __name__ == "__main__":
    serve()
