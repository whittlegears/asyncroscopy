"""
Electron microscope Tango device.

Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

What every scanning-beam instrument shares (the beam, the field of view, image shift, stage access, the
beam current, limits and the command lock) is in ``asyncroscopy.instruments.scanning_beam``. This device adds
what is particular to a TEM/STEM, and backends implement only ``_x`` methods:

* defocus and the stigmator (``set_stigmator``/``get_stigmator``, metres of A1), checked against
  ``max_defocus`` and ``max_stigmator`` before the backend sees the value;
* beam tilt, diffraction shift, the fluorescent screen and its current, the column valves;
* the camera, 4D data, spectra and spectrum images, and the corrector and detector devices.

Return convention for image commands
-------------------------------------
Image commands return a string supplied by the concrete microscope
implementation, typically a DATA/Tiled unique id.
"""

import math
import json
from abc import abstractmethod

import numpy as np
import tango
from tango import AttrWriteType, DevDouble, DevEncoded, DevFloat, DevString, DevVarFloatArray, DevVarStringArray
from tango.server import attribute, device_property

from asyncroscopy.instruments.image_cache import ImageCache
from asyncroscopy.instruments.scanning_beam import BEAM_ADDRESSES, ScanningBeamInstrument, check_scan, hardware_call, refuses, unsupported
from asyncroscopy.utils.commands import command

DETECTOR_ADDRESSES = {
    **BEAM_ADDRESSES,
    "eds": "eds_device_address",
    "camera": "camera_device_address",
    "corrector": "corrector_device_address",
}
SPECTRUM_DETECTORS = ("eds", "eels")
DEFAULT_MAX_DEFOCUS = 1e-5
DEFAULT_MAX_DIFFRACTION_SHIFT = 1e-2  # rad; a conservative software limit, not a vendor figure
DEFAULT_MAX_STIGMATOR = 1e-6
DEFAULT_MAX_SPECTRUM_IMAGE_POINTS = 64


def check_defocus(defocus: float, max_defocus: float) -> float:
    """Defocus in metres, refused beyond +/-max_defocus."""
    defocus = float(defocus)
    if not (math.isfinite(defocus) and abs(defocus) <= float(max_defocus)):
        raise ValueError(f"defocus must be within +/-{max_defocus} m (did you pass nanometres?), got {defocus}")
    return defocus


def check_stigmator(stigmator: list[float], max_stigmator: float) -> list[float]:
    """Stigmator [a, b] in metres of A1, refused beyond +/-max_stigmator per component."""
    value = [float(v) for v in stigmator]
    if len(value) != 2 or not all(math.isfinite(v) for v in value):
        raise ValueError(f"stigmator must be [a, b] in metres, got {list(stigmator)}")
    if any(abs(v) > float(max_stigmator) for v in value):
        raise ValueError(f"stigmator must be within +/-{float(max_stigmator)} m per component, got {value}")
    return value


def _request(text: str) -> dict:
    """A command's JSON request object ({} for an empty string)."""
    found = json.loads(text) if str(text or "").strip() else {}
    if not isinstance(found, dict):
        raise ValueError(f"the request must be a JSON object, got {text!r}")
    return found


def _positive(value, name: str) -> float:
    number = float("nan") if value is None else float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a positive number of seconds, got {value!r}; nothing was acquired")
    return number


class ElectronMicroscope(ScanningBeamInstrument):
    """
    Top-level electron microscope device (TEM/STEM).

    Detector-specific settings such as dwell time and resolution are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    proxy_addresses = DETECTOR_ADDRESSES

    corrector_device_address = device_property(
        dtype=str,
        doc="Tango device address for the aberration corrector settings device. "
        "DB mode: 'test/hardware/corrector' "
        "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/corrector#dbase=no'",
    )

    eds_device_address = device_property(
        dtype=str,
        doc="Tango device address for the EDS settings device. "
        "DB mode: 'asyncroscopy/eds/default' "
        "No-DB mode: 'tango://127.0.0.1:8887/asyncroscopy/haadf/default#dbase=no'",
    )

    camera_device_address = device_property(
        dtype=str,
        doc="Tango device address for the CAMERA settings. "
        "DB mode: 'asyncroscopy/camera/default' "
        "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/camera/default#dbase=no'",
    )

    default_scan_detector = device_property(
        dtype=str,
        default_value="haadf",
        doc="Scanning detector used when a caller names none: acquire_scanned_image with an empty list, a Tuner plan without a detector.",
    )

    max_defocus = device_property(dtype=float, default_value=DEFAULT_MAX_DEFOCUS, doc="Largest |defocus| accepted (m).")
    max_stigmator = device_property(dtype=float, default_value=DEFAULT_MAX_STIGMATOR, doc="Largest |stigmator| component accepted (m of A1).")
    max_diffraction_shift = device_property(dtype=float, default_value=DEFAULT_MAX_DIFFRACTION_SHIFT, doc="Largest |diffraction shift| component accepted (rad). Set from the instrument.")
    max_camera_series_frames = device_property(dtype=int, default_value=100, doc="Most frames acquire_camera_series accepts.")
    max_camera_series_bytes = device_property(dtype=int, default_value=1 << 30, doc="Largest series acquire_camera_series accepts: frames x size^2 x camera_bytes_per_pixel (bytes).")
    camera_bytes_per_pixel = device_property(dtype=int, default_value=0, doc="Bytes per pixel of one camera frame as the camera returns it; 0: not set (acquire_camera_series is refused).")
    max_spectrum_image_points = device_property(dtype=int, default_value=DEFAULT_MAX_SPECTRUM_IMAGE_POINTS, doc="Largest points per side acquire_spectrum_image accepts.")

    image_cache_size = device_property(dtype=int, default_value=8,
                                       doc="How many of the latest saved 2-D images get_image_data_cached keeps in memory (one per detector image).")

    stem_mode = attribute(
        label="STEM Mode",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="True when the microscope is in STEM mode",
    )

    manufacturer = attribute(label="Manufacturer", dtype=str, access=AttrWriteType.READ, doc="Who makes this microscope, or the simulation backend of a twin.")
    fov = attribute(label="Field of View", dtype=float, access=AttrWriteType.READ, unit="m", doc="The field of view in metres, as get_fov reads it. set_fov changes it.")
    defocus = attribute(label="Defocus", dtype=float, access=AttrWriteType.READ, unit="m", doc="The defocus in metres, as get_defocus reads it. set_defocus changes it.")
    beam_state = attribute(label="Beam Blanked", dtype=bool, access=AttrWriteType.READ,
                           doc="True while the beam is blanked, as get_parameters' beam_blanked reads it. blank_beam and unblank_beam change it.")

    def _init_device_attributes(self) -> None:
        super()._init_device_attributes()
        self._stem_mode: bool = False
        self.image_cache = ImageCache(int(self.image_cache_size))

    def read_instrument_type(self) -> str:
        return 'TEM'

    @abstractmethod
    def _connect(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _connect_hardware(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect_hardware; "
            "this vendor backend is missing the override"
        )

    def read_stem_mode(self) -> bool:
        return self._stem_mode

    def read_manufacturer(self) -> str:
        return str(self._manufacturer_name())

    @hardware_call()
    def read_fov(self) -> float:
        return float(self._get_fov())

    @hardware_call()
    def read_defocus(self) -> float:
        return float(self._get_defocus())

    @hardware_call()
    def read_beam_state(self) -> bool:
        return bool(self._beam_blanked())

    def argument_hints(self) -> dict:
        """The base's hints, plus defocus and the screen current, at this device's limits."""
        max_defocus = float(self.max_defocus)
        return {
            **super().argument_hints(),
            "set_defocus": {"args": [{"name": "defocus", "type": "number", "units": "m", "min": -max_defocus, "max": max_defocus}]},
            "get_defocus": {"returns": {"type": "number", "units": "m"}},
            "get_screen_current": {"returns": {"type": "number", "units": "pA"}},
            "set_screen_current": {"args": [{"name": "current", "type": "number", "units": "pA"}]},
            "get_beam_tilt": {"returns": {"type": "array", "items": "number", "elements": ["x", "y"], **self._beam_tilt_units()}},
            "set_beam_tilt": {"args": [{"name": "tilt", "type": "array", "items": "number", "elements": ["x", "y"], **self._beam_tilt_units()}]},
            "get_diffraction_shift": {"returns": {"type": "array", "items": "number", "elements": ["x", "y"], "units": "rad"}},
            "set_diffraction_shift": {"args": [{"name": "shift", "type": "array", "items": "number", "elements": ["x", "y"], "units": "rad",
                                                "min": -float(self.max_diffraction_shift), "max": float(self.max_diffraction_shift)}]},
        }

    def _beam_tilt_units(self) -> dict:
        """The units (and range) of get_beam_tilt and set_beam_tilt on this backend: radians unless it says otherwise."""
        return {"units": "rad"}

    def _spectrum_detector_names(self) -> list[str]:
        """The connected detectors acquire_spectrum takes: those the backend acquires itself (SPECTRUM_DETECTORS) and
        any device that acquires its own spectrum."""
        return sorted(name for name, proxy in self._detector_proxies.items()
                      if name in SPECTRUM_DETECTORS or self._own_spectrum_command(proxy))

    @staticmethod
    def _own_spectrum_command(proxy) -> "str | None":
        """The command a detector device takes a spectrum with by itself: the argument-free command its action_hints
        give the role acquire.spectrum. None for a settings-only device (the backend then acquires, as for EDS)."""
        try:
            hints = json.loads(proxy.read_attribute("action_hints").value)
        except Exception:
            return None
        for name, hint in sorted(hints.items()):
            if isinstance(hint, dict) and hint.get("role") == "acquire.spectrum":
                try:
                    if proxy.command_query(name).in_type == tango.CmdArgType.DevVoid:
                        return name
                except Exception:
                    continue
        return None

    @command(
        dtype_in=str,
        dtype_out=str,
        doc_in=":param detector_name: A spectrum detector of this microscope, e.g. 'eds' (a refusal lists them).",
        risk="high", role="acquire.spectrum", routine=True,
    )
    @hardware_call()
    def acquire_spectrum(self, detector_name: str) -> str:
        """Acquire a single spectrum and return its DATA/Tiled unique id."""
        detector_name = detector_name.lower().strip()
        proxy = self._detector_proxies.get(detector_name)
        own = self._own_spectrum_command(proxy) if proxy is not None else None
        if proxy is None or (own is None and detector_name not in SPECTRUM_DETECTORS):
            tango.Except.throw_exception(
                'UnknownDetector',
                f"No spectrum detector named '{detector_name}'. Spectrum detectors here: {self._spectrum_detector_names()}. "
                "For a scanned image (haadf, bf, adf, se) use acquire_scanned_image.",
                'acquire_spectrum()',
            )
        if own is not None:
            return self._acquire_through(proxy, own)
        return self._acquire_spectrum(detector_name, proxy.exposure_time)

    def _acquire_through(self, proxy, command_name: str) -> str:
        """A spectrum from a detector device that acquires it itself; its DATA key. A backend that must guard the beam
        around it overrides this."""
        return str(proxy.command_inout(command_name))

    @command(dtype_in=int, dtype_out=str, doc_in=":param points: Beam positions per side of the field (2 to the max_spectrum_image_points property).", risk="high", role="acquire.other", routine=True)
    @hardware_call()
    def acquire_spectrum_image(self, points: int) -> str:
        """Acquire an EDS spectrum image of points x points positions, at the EDS device's exposure per position, and
        return its DATA/Tiled key. A backend that cannot raises UnsupportedCommand."""
        points = int(points)
        if not 2 <= points <= int(self.max_spectrum_image_points):
            raise ValueError(f"points must be between 2 and {int(self.max_spectrum_image_points)}, got {points}")
        proxy = self._detector_proxies.get("eds")
        if proxy is None:
            tango.Except.throw_exception('UnknownDetector', f"No spectrum detector named 'eds'. Spectrum detectors here: {self._spectrum_detector_names()}.", 'acquire_spectrum_image()')
        return self._acquire_spectrum_image(points, proxy.exposure_time)

    @command(dtype_out=str, risk="high", role="acquire.other", routine=True)
    @hardware_call()
    def acquire_scanned_data_advanced(self) -> str:
        """Trigger an advanced 4D scanned data acquisition with the Ceta camera."""
        scan = self._detector_proxies.get('scan')
        region = list(check_scan(scan.imsize, scan.dwell_time, list(scan.scan_region)))  # refused here, before the instrument
        return self._acquire_scanned_data_advanced(scan.imsize, scan.dwell_time, 'BM-Ceta', region)

    @command(dtype_out=str, risk="high", role="acquire.image")
    @hardware_call()
    def acquire_camera_image(self) -> str:
        """Acquire a camera image using settings from the camera device."""
        camera = self._detector_proxies.get('camera')
        return self._acquire_camera_image(
            camera.imsize,
            camera.exposure_time,
            camera.camera_detector,
            camera.readout_area,
            camera.frame_combining,
            camera.output_format,
        )

    @command(dtype_in=int, dtype_out=DevEncoded, doc_in=":param index: 0 for the newest cached image, 1 for the one before, and so on.",
             risk="low", role="data.read")
    def get_image_data_cached(self, index: int) -> tuple[str, bytes]:
        """One of the latest saved 2-D images from memory, by index (0 is the newest), without reading DATA or Tiled.
        Returns JSON {"index", "key", "acquisition_type", "detector", "shape", "dtype", "cached", "cache_size"} and the
        image's raw bytes (C order). Only the image_cache_size latest images are kept; a spectrum, a camera series or a
        4D-STEM cube is never cached."""
        try:
            entry = self.image_cache.get(int(index))
        except IndexError as error:
            tango.Except.throw_exception("InvalidIndex", str(error), "get_image_data_cached()")
        except LookupError as error:
            tango.Except.throw_exception("NoCachedImage", str(error), "get_image_data_cached()")
        image = entry["image"]
        meta = {"index": int(index), "key": entry["key"], "acquisition_type": entry["acquisition_type"], "detector": entry["detector"],
                "shape": list(image.shape), "dtype": str(image.dtype), "cached": len(self.image_cache), "cache_size": self.image_cache.size}
        return json.dumps(meta), np.ascontiguousarray(image).tobytes()

    @command(dtype_out=str, risk="low", role="setting.read")
    @hardware_call()
    def get_detectors(self) -> str:
        """The detectors this microscope has, by the names its acquisitions take, as JSON {"scanning", "camera", "eds",
        "eels"} (lists), and "cameras" with each camera's exposure limits and image sizes where the backend gives them."""
        return json.dumps(self._detectors())

    @command(dtype_in=str, doc_in=":param state: 'open' or 'close'.", risk="high", role="setting.set")
    @hardware_call()
    def set_column_valves(self, state: str) -> None:
        """Open or close the column valves."""
        self._set_column_valves(state)

    @command(dtype_in=DevDouble, doc_in=':param defocus: Defocus in metres (e.g. 2e-8 for 20 nm).', risk="low", role="optics.focus")
    @hardware_call()
    def set_defocus(self, defocus: float) -> None:
        """Set the defocus in meters."""
        self._set_defocus(check_defocus(defocus, self.max_defocus))

    @command(dtype_out=DevDouble, risk="low", role="setting.read")
    @hardware_call()
    def get_defocus(self) -> float:
        """Read the defocus in meters."""
        return self._get_defocus()

    @command(dtype_in=DevVarFloatArray, doc_in=":param stigmator: [a, b] objective stigmator in metres of A1.", risk="high", role="setting.set", routine=True)
    @hardware_call()
    def set_stigmator(self, stigmator: list[float]) -> None:
        """Set the objective stigmator: the two A1 components it adds to the column, in metres."""
        self._set_stigmator(check_stigmator(stigmator, self.max_stigmator))

    @command(dtype_out=DevVarFloatArray, risk="low", role="setting.read")
    @hardware_call()
    def get_stigmator(self) -> list[float]:
        """Read the objective stigmator [a, b] in metres of A1; the column's own astigmatism is not included."""
        return [float(v) for v in self._get_stigmator()]

    @command(dtype_out=DevVarFloatArray, risk="low", role="setting.read")
    @hardware_call()
    def get_beam_tilt(self) -> list[float]:
        """Get the current beam tilt as [x, y], in this backend's beam-tilt units (action_space gives them: radians on
        a twin, AutoScript's unitless -1 to 1 on a Thermo Fisher microscope, which has no calibration to radians here)."""
        return self._get_beam_tilt()

    @command(dtype_in=DevVarFloatArray, doc_in=":param tilt: [x, y] beam tilt in this backend's beam-tilt units (action_space gives them).",
             risk="high", role="setting.set", routine=True)
    @hardware_call()
    def set_beam_tilt(self, tilt: list[float]) -> None:
        """Set the beam tilt to [x, y], in the units get_beam_tilt returns (radians on a twin, AutoScript's unitless -1 to 1)."""
        self._set_beam_tilt(tilt)

    @command(dtype_out=DevVarFloatArray, risk="low", role="setting.read")
    @hardware_call()
    def get_diffraction_shift(self) -> list[float]:
        """Get the current diffraction shift as [x, y] in radians. Refused outside diffraction mode."""
        return self._get_diffraction_shift()

    @command(dtype_in=DevVarFloatArray, doc_in=':param shift: [x, y] diffraction shift in radians.', risk="high", role="setting.set")
    @hardware_call()
    def set_diffraction_shift(self, shift: list[float]) -> None:
        """Set the diffraction shift to [x, y] in radians, each within max_diffraction_shift. Refused outside diffraction mode."""
        values = [float(v) for v in shift]
        limit = float(self.max_diffraction_shift)
        if len(values) != 2 or not all(math.isfinite(v) and abs(v) <= limit for v in values):
            raise ValueError(f"diffraction shift must be [x, y] in radians, each within +/-{limit} (max_diffraction_shift), got {list(shift)}; nothing was changed")
        self._set_diffraction_shift(values)

    @command(dtype_in=DevString, doc_in=":param position: 'inserted' or 'retracted'.", risk="high", role="setting.set")
    @hardware_call()
    def set_screen(self, position: str) -> None:
        """Insert or retract the fluorescent screen."""
        self._set_screen(position)

    @command(risk="high", role="other")
    @hardware_call()
    def calibrate_screen_current(self) -> None:
        """Measure the beam current on the screen, so dose estimates use a measured current."""
        self._calibrate_screen_current()


    @command(dtype_in=DevFloat, doc_in=':param current: Screen current in picoamperes (pA).', risk="high", role="setting.set")
    @hardware_call()
    def set_screen_current(self, current: float) -> None:
        """Set the screen current in pA."""
        self._set_screen_current(current)

    @command(dtype_out=DevFloat, risk="low", role="setting.read")
    @hardware_call()
    def get_screen_current(self) -> float:
        """Get the screen current in pA."""
        return self._get_screen_current()

    @command(dtype_out=str, risk="high", role="optics.focus", routine=True)
    @hardware_call()
    def auto_focus(self) -> str:
        """Run the microscope's autofocus routine. Returns JSON of what the routine reports (on AutoScript in STEM,
        OptiSTEM's quality: Bad, Improved or Good); {} from a routine that reports nothing."""
        return json.dumps(self._auto_focus() or {})

    # Operator only until reviewed: an agent is not given these (docs/Microscopy/modify_auto_script_microscope.md lists them).
    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON of RunStemAutoFocusSettings fields to set (scanning_detector, size, dwell_time, auto_stigmate, accuracy_level); {} for the defaults.',
             risk="high", operator_only=True, role="optics.focus")
    @hardware_call()
    def run_stem_auto_focus(self, request: str) -> str:
        """Run the vendor STEM autofocus (STEM mode). Returns JSON {"defocus_m", "astigmatism_m"}, as the routine measured them,
        and "beam_blanked", the blanker read after it."""
        return json.dumps(self._run_stem_auto_focus(_request(request)))

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON of RunBeamTiltAutoFocusSettings fields to set (camera_detector, size, exposure_time, focus_method, auto_stigmate, ...); {} for the defaults.',
             risk="high", operator_only=True, role="optics.focus")
    @hardware_call()
    def run_beam_tilt_auto_focus(self, request: str) -> str:
        """Run the vendor beam-tilt autofocus (TEM mode). Returns JSON {"defocus_m", "astigmatism_m", "drift_m": [x, y]} and
        "beam_blanked", the blanker read after it."""
        return json.dumps(self._run_beam_tilt_auto_focus(_request(request)))

    @command(dtype_in=int, dtype_out=str, doc_in=":param length: Number of camera frames (1 to the max_camera_series_frames property, and at most max_camera_series_bytes in all).",
             risk="high", operator_only=True, role="acquire.image")
    @hardware_call()
    def acquire_camera_series(self, length: int) -> str:
        """Acquire a series of camera images with the CAMERA device's settings and return its DATA/Tiled key."""
        length = int(length)
        if not 1 <= length <= int(self.max_camera_series_frames):
            raise ValueError(f"length must be between 1 and {int(self.max_camera_series_frames)} (max_camera_series_frames), got {length}; nothing was acquired")
        camera = self._detector_proxies.get('camera')
        per_pixel = int(self.camera_bytes_per_pixel)
        if per_pixel <= 0:
            raise ValueError("camera_bytes_per_pixel is not set, so the size of a series is not known; set it from the camera's "
                             "specification. Nothing was acquired")
        size = int(camera.imsize)
        total = length * size * size * per_pixel
        if total > int(self.max_camera_series_bytes):
            raise ValueError(f"{length} frames of {size}x{size} at {per_pixel} bytes per pixel is {total} bytes, above max_camera_series_bytes "
                             f"({int(self.max_camera_series_bytes)}); take fewer or smaller frames. Nothing was acquired")
        return self._acquire_camera_series(size, camera.exposure_time, camera.camera_detector, camera.readout_area, camera.frame_combining, length)

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"detector": an EELS detector from get_detectors, "exposure_time_s": seconds}.',
             risk="high", operator_only=True, role="acquire.spectrum")
    @hardware_call()
    def acquire_eels_spectra(self, request: str) -> str:
        """Acquire EELS spectra with the vendor's EELS detector and return their DATA/Tiled key."""
        fields = _request(request)
        return self._acquire_eels_spectra(str(fields.get("detector", "")), _positive(fields.get("exposure_time_s"), "exposure_time_s"))

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"detector": an EELS detector, "exposure_time_s": seconds per point, "size": points per side}.',
             risk="high", operator_only=True, role="acquire.other")
    @hardware_call()
    def acquire_eels_spectrum_image(self, request: str) -> str:
        """Acquire an EELS spectrum image over the whole field and return its key."""
        fields = _request(request)
        size = int(fields.get("size", 0))
        if not 2 <= size <= int(self.max_spectrum_image_points):
            raise ValueError(f"size must be between 2 and {int(self.max_spectrum_image_points)}, got {size}; nothing was acquired")
        return self._acquire_eels_spectrum_image(str(fields.get("detector", "")), _positive(fields.get("exposure_time_s"), "exposure_time_s"), size)

    @command(dtype_in=DevString, doc_in=":param mode: 'Imaging' or 'Diffraction'.", risk="high", operator_only=True, role="setting.set")
    @hardware_call()
    def set_projector_mode(self, mode: str) -> None:
        """Set the projector to imaging or diffraction."""
        self._set_projector_mode(str(mode))

    @command(dtype_in=DevString, doc_in=":param label: A camera length's label, from get_parameters' camera_lengths.", risk="high", operator_only=True, role="setting.set")
    @hardware_call()
    def set_camera_length(self, label: str) -> None:
        """Set the camera length to one the microscope offers in this optical state, by its label."""
        self._set_camera_length(str(label))

    @command(dtype_in=DevString, doc_in=":param state: 'inserted', 'half_inserted' or 'retracted'.", risk="high", operator_only=True, role="setting.set")
    @hardware_call()
    def set_beam_stopper(self, state: str) -> None:
        """Insert, half insert or retract the beam stopper."""
        self._set_beam_stopper(str(state))

    @command(dtype_in=DevVarStringArray, doc_in=":param request: [detector, state]: a scanning detector from get_detectors and 'inserted' or 'retracted'.",
             risk="high", operator_only=True, role="setting.set")
    @hardware_call()
    def set_scanning_detector(self, request: list[str]) -> None:
        """Insert or retract a scanning detector."""
        if len(request) != 2:
            raise ValueError(f"set_scanning_detector takes [detector, state], got {list(request)}")
        self._set_scanning_detector(str(request[0]), str(request[1]))

    
    @refuses("acquire_camera_image")
    def _acquire_camera_image(
        self,
        imsize: int,
        exposure_time: float,
        detector: str,
        readout_area: str,
        frame_combining: int = 1,
        output_format: str = '.h5',
    ) -> str:
        """Vendor-specific camera acquisition implementation."""
        unsupported(self, "acquire_camera_image")

    @refuses("acquire_spectrum_image")
    def _acquire_spectrum_image(self, points: int, exposure_time: float) -> str:
        """Vendor-specific EDS spectrum image; the base backend has none."""
        unsupported(self, "acquire_spectrum_image")

    @refuses("acquire_scanned_data_advanced")
    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """Vendor-specific advanced 4D scanned data acquisition trigger."""
        unsupported(self, "acquire_scanned_data_advanced")

    @refuses("set_stigmator")
    def _set_stigmator(self, stigmator: list[float]) -> None:
        """Set the objective stigmator [a, b] (metres of A1, already checked against max_stigmator)."""
        unsupported(self, "set_stigmator")

    @refuses("get_stigmator")
    def _get_stigmator(self) -> list[float]:
        """The objective stigmator [a, b] in metres of A1."""
        unsupported(self, "get_stigmator")

    @refuses("set_defocus")
    def _set_defocus(self, defocus):
        unsupported(self, "set_defocus")

    def _beam_blanked(self) -> bool:
        """Whether the beam is blanked now: the one reader of get_parameters' beam_blanked and the beam_state attribute."""
        unsupported(self, "reading the blanker")

    def _manufacturer_name(self) -> str:
        """Who makes the instrument this backend drives (the manufacturer attribute)."""
        unsupported(self, "manufacturer")

    @refuses("set_column_valves")
    def _set_column_valves(self, state: str) -> None:
        unsupported(self, "set_column_valves")

    @refuses("get_detectors")
    def _detectors(self) -> dict:
        unsupported(self, "get_detectors")

    @refuses("run_stem_auto_focus")
    def _run_stem_auto_focus(self, fields: dict) -> dict:
        unsupported(self, "run_stem_auto_focus")

    @refuses("run_beam_tilt_auto_focus")
    def _run_beam_tilt_auto_focus(self, fields: dict) -> dict:
        unsupported(self, "run_beam_tilt_auto_focus")

    @refuses("acquire_camera_series")
    def _acquire_camera_series(self, imsize: int, exposure_time: float, detector: str, readout_area: str, frame_combining: int, length: int) -> str:
        unsupported(self, "acquire_camera_series")

    @refuses("acquire_eels_spectra")
    def _acquire_eels_spectra(self, detector: str, exposure_time: float) -> str:
        unsupported(self, "acquire_eels_spectra")

    @refuses("acquire_eels_spectrum_image")
    def _acquire_eels_spectrum_image(self, detector: str, exposure_time: float, size: int) -> str:
        unsupported(self, "acquire_eels_spectrum_image")

    @refuses("set_projector_mode")
    def _set_projector_mode(self, mode: str) -> None:
        unsupported(self, "set_projector_mode")

    @refuses("set_camera_length")
    def _set_camera_length(self, label: str) -> None:
        unsupported(self, "set_camera_length")

    @refuses("set_beam_stopper")
    def _set_beam_stopper(self, state: str) -> None:
        unsupported(self, "set_beam_stopper")

    @refuses("set_scanning_detector")
    def _set_scanning_detector(self, detector: str, state: str) -> None:
        unsupported(self, "set_scanning_detector")

    @abstractmethod
    def _get_defocus(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_defocus; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_screen(self, position):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_screen; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_screen_current(self, current):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_screen_current; "
            "this vendor backend is missing the override"
        )
    
    @abstractmethod
    def _calibrate_screen_current(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _calibrate_screen_current; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_screen_current(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_screen_current; "
            "this vendor backend is missing the override"
        )

    @refuses("get_beam_tilt")
    @abstractmethod
    def _get_beam_tilt(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_beam_tilt; "
            "this vendor backend is missing the override"
        )

    @refuses("set_beam_tilt")
    @abstractmethod
    def _set_beam_tilt(self,tilt):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_beam_tilt; "
            "this vendor backend is missing the override"
        )

    @refuses("get_diffraction_shift")
    @abstractmethod
    def _get_diffraction_shift(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_diffraction_shift; "
            "this vendor backend is missing the override"
        )

    @refuses("set_diffraction_shift")
    @abstractmethod
    def _set_diffraction_shift(self, tilt):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_diffraction_shift; "
            "this vendor backend is missing the override"
        )

    @refuses("auto_focus")
    @abstractmethod
    def _auto_focus(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _auto_focus; "
            "this vendor backend is missing the override"
        )


if __name__ == '__main__':
    ElectronMicroscope.run_server()
