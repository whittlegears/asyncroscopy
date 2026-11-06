"""
Electron microscope Tango device.

Detector settings are read from the corresponding detector DeviceProxy
so that each detector device is the single source of truth for its own params.

What every backend shares lives here, and backends implement only ``_x`` methods:

* limits: ``set_fov``, ``set_defocus``, ``set_stigmator`` and the scans are checked against
  ``min_fov``, ``max_fov``, ``max_defocus`` and ``max_stigmator`` (and a scan region inside
  the field) before the backend sees the value, so every backend refuses the same values
  with the same messages (tests/test_parity.py);
* the beam current: ``beam_current_pa``, ``beam_current_source`` ("measured" or "default")
  and ``beam_current_age_s``, from the backend's ``_beam_current`` (by default the
  ``default_beam_current_pa`` property, reported as assumed);
* the stigmator (``set_stigmator``/``get_stigmator``, metres of A1) and ``stage_moving``;
  a backend that has neither raises a clear UnsupportedCommand error;
* the detector proxies (``_connect_detector_proxies``);
* one command lock: every public command that reaches the instrument runs under it
  (``hardware_call``), except ``blank_beam``, which never waits for another command.
  ``_check_connected`` lets a backend refuse a call when it has no connection.

Return convention for image commands
-------------------------------------
Image commands return a string supplied by the concrete microscope
implementation, typically a DATA/Tiled unique id.
"""

import functools
import json
import math
import threading
from abc import abstractmethod
from typing import Optional

import tango
from tango import AttrWriteType, DevDouble, DevEncoded, DevFloat, DevString, DevState, DevVarFloatArray, DevVarStringArray
from tango.server import attribute, command, device_property

from asyncroscopy.data.dose_ledger import dose_at
from asyncroscopy.data.record import current_sample
from asyncroscopy.instruments.electron_microscope.limits import DEFAULT_MAX_DEFOCUS, DEFAULT_MAX_FOV, DEFAULT_MAX_STIGMATOR, DEFAULT_MIN_FOV
from asyncroscopy.instruments.instrument import Instrument
from asyncroscopy.routines.beam import DEFAULT_BEAM_CURRENT_PA

# Detector and sub-device proxies the base connects: {name: the device property holding its address}.
DETECTOR_ADDRESSES = {
    "eds": "eds_device_address",
    "stage": "stage_device_address",
    "scan": "scan_device_address",
    "camera": "camera_device_address",
    "corrector": "corrector_device_address",
    "data": "data_device_address",
}
DETECTOR_TIMEOUT_MS = 12_000
DATA_TIMEOUT_MS = 3_600_000  # registering a large acquisition with Tiled can take long


def check_fov(fov: float, min_fov: float, max_fov: float) -> float:
    """Field of view in metres, refused outside [min_fov, max_fov]."""
    fov = float(fov)
    if not (math.isfinite(fov) and float(min_fov) <= fov <= float(max_fov)):
        raise ValueError(f"field of view must be between {min_fov} and {max_fov} m, got {fov}")
    return fov


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


def check_scan(imsize: int, dwell_time: float, scan_region: list[float] | None) -> tuple[float, float, float, float]:
    """A scan's [left, top, width, height] (fractions of the field), refused unless it lies inside the field."""
    region = tuple(float(v) for v in scan_region) if scan_region is not None else (0.0, 0.0, 1.0, 1.0)
    if len(region) != 4 or not all(math.isfinite(v) for v in region):
        raise ValueError(f"scan_region must be four finite numbers [left, top, width, height], got {list(region)}")
    left, top, width, height = region
    if not (left >= 0 and top >= 0 and width > 0 and height > 0 and left + width <= 1 + 1e-9 and top + height <= 1 + 1e-9):
        raise ValueError(f"scan_region [left, top, width, height] must lie inside the field (0..1), got {list(region)}")
    if not (math.isfinite(imsize) and int(imsize) >= 1 and math.isfinite(dwell_time) and float(dwell_time) > 0):
        raise ValueError(f"imsize must be >= 1 and dwell_time > 0, got {imsize}, {dwell_time}")
    return region


def hardware_call(serialized: bool = True):
    """A public command that reaches the instrument: refused when the backend is not connected
    (``_check_connected``), and run under the device's command lock unless ``serialized`` is False."""

    def decorate(method):
        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            self._check_connected(method.__name__)
            if not serialized:
                return method(self, *args, **kwargs)
            with self._command_lock:
                return method(self, *args, **kwargs)

        return guarded

    return decorate


def unsupported(device: object, what: str) -> None:
    """Raise the Tango error a backend gives for a command it does not implement."""
    tango.Except.throw_exception(
        "UnsupportedCommand",
        f"{type(device).__name__} does not support {what}.",
        f"{what}()",
    )


class ElectronMicroscope(Instrument):
    """
    Top-level electron microscope device.

    Detector-specific settings such as dwell time and resolution are stored in
    dedicated detector devices and read via DeviceProxy at acquisition time.
    """

    scan_device_address = device_property(
        dtype=str,
        doc="Tango device address for the SCAN settings device. "
        "DB mode: 'test/detector/scan' "
        "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/scan#dbase=no'",
    )

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

    stage_device_address = device_property(
        dtype=str,
        doc="Tango device address for the STAGE settings device. "
        "DB mode: 'asyncroscopy/stage/default' "
        "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/stage/default#dbase=no'",
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

    default_detector = attribute(
        label="Default scanning detector",
        dtype=str,
        access=AttrWriteType.READ,
        doc="The scanning detector used when a caller names none (the default_scan_detector property).",
    )

    # Limits checked before any backend sees a value (the defaults are in limits.py).
    min_fov = device_property(dtype=float, default_value=DEFAULT_MIN_FOV, doc="Smallest field of view accepted (m).")
    max_fov = device_property(dtype=float, default_value=DEFAULT_MAX_FOV, doc="Largest field of view accepted (m).")
    max_defocus = device_property(dtype=float, default_value=DEFAULT_MAX_DEFOCUS, doc="Largest |defocus| accepted (m).")
    max_stigmator = device_property(dtype=float, default_value=DEFAULT_MAX_STIGMATOR, doc="Largest |stigmator| component accepted (m of A1).")
    default_beam_current_pa = device_property(
        dtype=float,
        default_value=DEFAULT_BEAM_CURRENT_PA,
        doc="Reported as beam_current_pa, with beam_current_source 'default', while the backend has no measured current "
        "(on AutoScript: until the screen current has been measured). Dose budgets on hardware refuse it.",
    )

    beam_current_pa = attribute(dtype=float, unit="pA", access=AttrWriteType.READ, doc="The beam current dose estimates use (pA).")
    beam_current_source = attribute(dtype=str, access=AttrWriteType.READ, doc="'measured', or 'default' when beam_current_pa is assumed.")
    beam_current_age_s = attribute(dtype=float, unit="s", access=AttrWriteType.READ, doc="Age of the measured current (s); NaN when it is assumed.")
    stage_moving = attribute(dtype=bool, access=AttrWriteType.READ, doc="True while a stage move is under way.")

    stem_mode = attribute(
        label="STEM Mode",
        dtype=bool,
        access=AttrWriteType.READ,
        doc="True when the microscope is in STEM mode",
    )

    def _init_device_attributes(self) -> None:
        self._microscope: Optional[object] = None
        self._stem_mode: bool = False
        self._detector_proxies: dict[str, tango.DeviceProxy] = {}
        # Every public command that reaches the instrument takes this lock (hardware_call); blank_beam does not.
        self._command_lock = threading.RLock()

    def read_instrument_type(self) -> str:
        return 'TEM'

    @abstractmethod
    def _connect(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect; "
            "this vendor backend is missing the override"
        )

    def _disconnect(self):
        self._microscope = None
        self.info_stream('Disconnected from microscope hardware')

    @abstractmethod
    def _connect_hardware(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _connect_hardware; "
            "this vendor backend is missing the override"
        )

    def _connect_detector_proxies(self) -> None:
        """A DeviceProxy for every configured detector and sub-device (DETECTOR_ADDRESSES); a blank address is skipped."""
        for name, prop in DETECTOR_ADDRESSES.items():
            address = getattr(self, prop, "")
            if not address:
                self.info_stream(f"Skipping {name}: no address configured")
                continue
            try:
                proxy = tango.DeviceProxy(address)
                proxy.set_timeout_millis(self._proxy_timeout_ms(name))
                self._detector_proxies[name] = proxy
                self.info_stream(f"Connected to detector proxy: {name} @ {address}")
            except tango.DevFailed as exc:
                self.error_stream(f"Failed to connect to {name} proxy at {address}: {exc}")

    def _proxy_timeout_ms(self, name: str) -> int:
        """How long a call to sub-device ``name`` may take."""
        return DATA_TIMEOUT_MS if name == "data" else DETECTOR_TIMEOUT_MS

    def _check_connected(self, command_name: str) -> None:
        """Raise a Tango error when the backend cannot reach the instrument; by default it always can."""

    def read_stem_mode(self) -> bool:
        return self._stem_mode

    def read_default_detector(self) -> str:
        return str(self.default_scan_detector)

    def read_beam_current_pa(self) -> float:
        return float(self._beam_current()["pa"])

    def read_beam_current_source(self) -> str:
        return str(self._beam_current()["source"])

    def read_beam_current_age_s(self) -> float:
        age = self._beam_current().get("age_s")
        return float("nan") if age is None else float(age)

    @hardware_call()
    def read_stage_moving(self) -> bool:
        return bool(self._stage_moving())

    def beam_current_fields(self) -> dict:
        """The beam current as get_parameters reports it: beam_current_pa, beam_current_source, and
        beam_current_age_s (measured) or beam_current_reason (assumed)."""
        current = self._beam_current()
        fields = {"beam_current_pa": float(current["pa"]), "beam_current_source": current["source"]}
        if current.get("age_s") is not None:
            fields["beam_current_age_s"] = float(current["age_s"])
        if current["source"] != "measured":
            fields["beam_current_reason"] = current.get("reason") or "this microscope does not measure its beam current"
        return fields

    @command
    def Disconnect(self) -> None:
        """Disconnect from microscope hardware gracefully."""
        self.set_state(DevState.OFF)
        self._disconnect()

    @command(
        dtype_in=str,
        dtype_out=str,
        doc_in=":param detector_name: Spectrum detector name, e.g. 'eds'.",
    )
    @hardware_call()
    def acquire_spectrum(self, detector_name: str) -> str:
        """Acquire a single spectrum and return its DATA/Tiled unique id."""
        detector_name = detector_name.lower().strip()
        proxy = self._detector_proxies.get(detector_name)
        if proxy is None:
            tango.Except.throw_exception(
                'UnknownDetector',
                f"No spectrum detector named '{detector_name}'. "
                f"Configured detector devices: {sorted(self._detector_proxies.keys())}.",
                'acquire_spectrum()',
            )
        return self._acquire_spectrum(detector_name, proxy.exposure_time)

    @command(
        dtype_in=DevVarStringArray,
        dtype_out=str,
        doc_in=":param detector_list: Scanning detector names, e.g. ['haadf']. "
               "An empty list uses the microscope's default_detector.",
    )
    @hardware_call()
    def acquire_scanned_image(self, detector_list: list[str]) -> str:
        """
        Acquire an image with scanning detectors and return its DATA/Tiled key.
        An empty list uses the microscope's default_detector (haadf unless configured).
        """
        # Tango's wire protocol has no concept of "omitted", so remote callers (e.g. the
        # MCP/LLM bridge) send an empty list for "no detector named".
        if not detector_list:
            detector_list = [str(self.default_scan_detector)]
        scan = self._detector_proxies.get('scan')
        region = list(check_scan(scan.imsize, scan.dwell_time, list(scan.scan_region)))  # refused here, before the instrument
        return self._acquire_scanned_image(scan.imsize, scan.dwell_time, detector_list, region, scan.output_format)

    @command(dtype_out=str)
    @hardware_call()
    def acquire_scanned_data_advanced(self) -> str:
        """Trigger an advanced 4D scanned data acquisition with the Ceta camera."""
        scan = self._detector_proxies.get('scan')
        region = list(check_scan(scan.imsize, scan.dwell_time, list(scan.scan_region)))  # refused here, before the instrument
        return self._acquire_scanned_data_advanced(scan.imsize, scan.dwell_time, 'BM-Ceta', region)

    @command(dtype_out=str)
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

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"x_m": ..., "y_m": ...} for the dose at one stage point, or {} for the exposure count only.')
    def get_dose(self, request: str) -> str:
        """Electron dose already put on the loaded sample, from the dose ledger: exposures counted (scanned frames and
        parked-beam spectra), and at a stage point the exposures that covered it and their summed dose in e/A^2;
        "not_counted" says what is not counted."""
        body = json.loads(request) if request.strip() else {}
        point = [body.get("x_m"), body.get("y_m")]
        if any(v is not None for v in point) and not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point):
            raise ValueError("give both x_m and y_m in metres, or neither")
        sample = current_sample(self._detector_proxies.get("stage"))
        if sample.get("sample_load_id") is None:
            return json.dumps({"sample_load_id": None, "sample_id": sample.get("sample_id"), "exposures": None, "error": sample.get("sample_error")})
        x_m, y_m = (float(point[0]), float(point[1])) if point[0] is not None else (None, None)
        return json.dumps({**dose_at(sample["sample_load_id"], x_m, y_m), "sample_id": sample.get("sample_id")})

    @command(dtype_in=int, dtype_out=DevEncoded, doc_in=':param index: Index of a cached image.')
    def get_image_data_cached(self, index: int) -> tuple[str, bytes]:
        """Retrieve cached image by index."""
        if not hasattr(self, '_cached_images'):
            tango.Except.throw_exception('NoCache', 'Call acquire_scanned_image() first', 'get_image_data()')
        if index >= len(self._cached_images):
            tango.Except.throw_exception('InvalidIndex', f'Index {index} out of range', 'get_image_data()')

        cached_image = self._cached_images[index]
        img_data = cached_image.data if hasattr(cached_image, 'data') else cached_image

        meta = {'shape': list(img_data.shape), 'dtype': str(img_data.dtype)}
        return json.dumps(meta), img_data.tobytes()

    @command(dtype_in=DevVarFloatArray, dtype_out=None, doc_in=':param position: [x, y] beam position as fractions (0-1) of the field of view; [0.5, 0.5] is the centre.')
    @hardware_call()
    def place_beam(self, position: list[float]) -> None:
        """Set resting beam position, [0:1]."""
        x, y = (float(v) for v in list(position)[:2])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            # Agents pass stage coordinates (metres) here; say what the command takes and how to reach a stage point.
            raise ValueError(
                f"place_beam takes [x, y] as fractions of the current image (0 to 1, [0.5, 0.5] is the centre), got [{x:g}, {y:g}]. "
                "To put the beam on a stage point given in metres (a find_features x_m, y_m), call place_beam_at with it."
            )
        self._place_beam(position)

    @command(dtype_in=DevVarFloatArray, dtype_out=None, doc_in=':param positions: Flat list [x0, y0, x1, y1, ...] of fractional beam positions.')
    @hardware_call()
    def place_beam_list(self, positions: list[float]) -> None:
        """Place beam at multiple positions sequentially."""
        if len(positions) % 2 != 0:
            raise ValueError('Input must contain pairs of (x, y) values.')

        for i in range(0, len(positions), 2):
            x = float(positions[i])
            y = float(positions[i + 1])
            self._place_beam([x, y])

    @command(dtype_in=str, doc_in=":param state: 'open' or 'close'.")
    @hardware_call()
    def set_column_valves(self, state: str) -> None:
        """Open or close the column valves."""
        self._set_column_valves(state)

    @command()
    @hardware_call(serialized=False)  # never waits for a running command: blanking is the safe state
    def blank_beam(self) -> None:
        """Blank beam."""
        self._blank_beam()

    @command()
    @hardware_call()
    def unblank_beam(self) -> None:
        """Unblank beam."""
        self._unblank_beam()

    @command(dtype_in=DevDouble, doc_in=':param defocus: Defocus in metres (e.g. 2e-8 for 20 nm).')
    @hardware_call()
    def set_defocus(self, defocus: float) -> None:
        """Set the defocus in meters."""
        self._set_defocus(check_defocus(defocus, self.max_defocus))

    @command(dtype_out=DevDouble)
    @hardware_call()
    def get_defocus(self) -> float:
        """Read the defocus in meters."""
        return self._get_defocus()

    @command(dtype_in=DevDouble, doc_in=':param fov: Field of view in metres (e.g. 2e-7 for 200 nm).')
    @hardware_call()
    def set_fov(self, fov: float) -> None:
        """Set the field of view for the next acquisition, in meters."""
        self._set_fov(check_fov(fov, self.min_fov, self.max_fov))

    @command(dtype_out=DevDouble)
    @hardware_call()
    def get_fov(self) -> float:
        """Read the field of view for the next acquisition, in meters."""
        return self._get_fov()
    
    @command(dtype_in=DevVarFloatArray, doc_in=':param shift: [x, y] image shift in metres.')
    @hardware_call()
    def set_image_shift(self, shift: list[float]) -> None:
        """Set the image shift to [x_shift, y_shift] in meters."""
        self._set_image_shift(shift)

    @command(dtype_out=DevVarFloatArray)
    @hardware_call()
    def get_image_shift(self) -> list[float]:
        """Get the image shiftas [x, y] in m."""
        return self._get_image_shift()

    @command(dtype_in=DevVarFloatArray, dtype_out=DevVarFloatArray,
             doc_in=':param position: [x, y] stage position in metres for the centre of the next image.')
    @hardware_call()
    def aim_image_shift(self, position: list[float]) -> list[float]:
        """Centre the next image on stage position [x, y] (metres) with the image shift alone; the stage does not move.
        Sets the shift to the position minus the stage's and returns it, [x, y] in metres; set_image_shift [0, 0] goes
        back to the stage's centre. The instrument refuses a shift beyond its limit (about 1 um)."""
        stage = self._get_stage()
        shift = [float(position[0]) - float(stage[0]), float(position[1]) - float(stage[1])]
        self._set_image_shift(shift)
        return shift
    
    @command(dtype_in=DevVarFloatArray, dtype_out=DevVarFloatArray,
             doc_in=':param position: [x, y] stage position in metres to park the beam on (inside the current field of view).')
    @hardware_call()
    def place_beam_at(self, position: list[float]) -> list[float]:
        """Park the beam on stage position [x, y] (metres; a find_features x_m, y_m) for a spectrum, without moving the stage.
        For the centre of the image, or any other fraction of it, use place_beam (centre: [0.5, 0.5]). Works out place_beam's
        fractions from the stage, the image shift and the field of view, and returns them. Refused outside the current
        field of view (take an image there first). A small agent asked for fractions put the beam at the centre."""
        stage, shift, fov = self._get_stage(), self._get_image_shift(), float(self._get_fov())
        fraction = [0.5 + (float(position[i]) - float(stage[i]) - float(shift[i])) / fov for i in (0, 1)]
        if not all(0.0 <= f <= 1.0 for f in fraction):
            if all(0.0 <= float(v) <= 1.0 for v in list(position)[:2]):
                # The reverse of place_beam's mistake: fractions sent here, where stage metres go.
                raise ValueError(f"place_beam_at takes a stage position in metres; [{float(position[0]):g}, {float(position[1]):g}] look like "
                                 "fractions of the image: for those (the centre is [0.5, 0.5]) call place_beam with them")
            # The exact move, with the stage's own z and tilts: a small agent told only to "image the area first" gave up.
            there = [float(f"{float(v):.6g}") for v in (*list(position)[:2], *list(stage)[2:5])]  # float32 noise off, to copy as is
            raise ValueError(f"[{float(position[0]):.4g}, {float(position[1]):.4g}] m is outside the current {fov:.3g} m field of view. "
                             f"To measure there, call move_stage with {{\"position\": {json.dumps(there)}}} (the stage keeps its z and tilts), "
                             "then place_beam_at with the same position")
        self._place_beam(fraction)
        return fraction

    @command(dtype_in=DevVarFloatArray, doc_in=":param stigmator: [a, b] objective stigmator in metres of A1.")
    @hardware_call()
    def set_stigmator(self, stigmator: list[float]) -> None:
        """Set the objective stigmator: the two A1 components it adds to the column, in metres."""
        self._set_stigmator(check_stigmator(stigmator, self.max_stigmator))

    @command(dtype_out=DevVarFloatArray)
    @hardware_call()
    def get_stigmator(self) -> list[float]:
        """Read the objective stigmator [a, b] in metres of A1; the column's own astigmatism is not included."""
        return [float(v) for v in self._get_stigmator()]

    @command(dtype_out=DevVarFloatArray)
    @hardware_call()
    def get_beam_tilt(self) -> list[float]:
        """Get the current beam tilt as [alpha, beta] in radian."""
        return self._get_beam_tilt()

    @command(dtype_in=DevVarFloatArray, doc_in=':param tilt: [x, y] beam tilt in radians.')
    @hardware_call()
    def set_beam_tilt(self, tilt: list[float]) -> None:
        """Set the beam tilt to [x_tilt, y_tilt] in radian."""
        self._set_beam_tilt(tilt)

    @command(dtype_out=DevVarFloatArray)
    @hardware_call()
    def get_diffraction_shift(self) -> list[float]:
        """Get the current  diffraction shift as [alpha, beta] in radian."""
        return self._get_diffraction_shift()
    
    @command(dtype_in=DevVarFloatArray, doc_in=':param shift: [x, y] diffraction shift.')
    @hardware_call()
    def set_diffraction_shift(self, shift: list[float]) -> None:
        """Set the diffraction shift to [x_shift, y_shift] in radian."""
        self._set_diffraction_shift(shift)

    @command(dtype_out=DevString)
    @hardware_call()
    def get_parameters(self) -> str:
        """ Get all status parameters"""
        return self._get_parameters()

    @command(dtype_in=DevString, doc_in=":param position: 'inserted' or 'retracted'.")
    @hardware_call()
    def set_screen(self, position: str)->None:
        self._set_screen(position)

    @command()
    @hardware_call()
    def calibrate_screen_current(self) -> None:
        """Set the screen current in pA."""
        self._calibrate_screen_current()


    @command(dtype_in=DevFloat, doc_in=':param current: Screen current in picoamperes (pA).')
    @hardware_call()
    def set_screen_current(self, current: float) -> None:
        """Set the screen current in pA."""
        self._set_screen_current(current)

    @command(dtype_out=DevFloat)
    @hardware_call()
    def get_screen_current(self) -> float:
        """Get the screen current in pA."""
        return self._get_screen_current()

    @command(dtype_out=DevVarFloatArray)
    @hardware_call()
    def get_stage(self) -> list[float]:
        """Get the current stage position as [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        return self._get_stage()
   
    @command(dtype_in=DevVarFloatArray, doc_in=':param position: [x, y, z, alpha, beta]; x, y, z in metres, alpha and beta in degrees.')
    @hardware_call()
    def move_stage(self, position: list[float]) -> None:
        """Move the stage to [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        self._move_stage(position)

    @command()
    @hardware_call()
    def auto_focus(self) -> None:
        """Run the microscope's autofocus routine."""
        self._auto_focus()

    
    @abstractmethod
    def _acquire_scanned_image(
        self,
        imsize: int,
        dwell_time: float,
        detector_list: list[str] = ['haadf'],
        scan_region: list[float] = [0.0, 0.0, 1.0, 1.0],
        output_format: str = '.h5',
    ) -> str:
        """Vendor-specific scanned image acquisition implementation."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _acquire_scanned_image; "
            "this vendor backend is missing the override"
        )

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
        tango.Except.throw_exception(
            'UnsupportedCommand',
            'This microscope does not support camera image acquisition.',
            '_acquire_camera_image()',
        )

    def _acquire_scanned_data_advanced(self, imsize: int, dwell_time: float, detector: str, scan_region: list[float]) -> str:
        """Vendor-specific advanced 4D scanned data acquisition trigger."""
        tango.Except.throw_exception(
            'UnsupportedCommand',
            'This microscope does not support advanced scanned data acquisition.',
            '_acquire_scanned_data_advanced()',
        )

    def _beam_current(self) -> dict:
        """The beam current: {"pa", "source" ("measured" or "default"), "age_s" (measured) or "reason" (assumed)}.
        By default the default_beam_current_pa property, assumed: a backend that measures its current overrides this."""
        return {"pa": float(self.default_beam_current_pa), "source": "default", "age_s": None,
                "reason": f"{type(self).__name__} does not measure its beam current; default_beam_current_pa is assumed"}

    def _stage_moving(self) -> bool:
        """True while a stage move is under way; a backend whose moves return only on arrival may return False."""
        unsupported(self, "stage_moving")

    def _set_stigmator(self, stigmator: list[float]) -> None:
        """Set the objective stigmator [a, b] (metres of A1, already checked against max_stigmator)."""
        unsupported(self, "set_stigmator")

    def _get_stigmator(self) -> list[float]:
        """The objective stigmator [a, b] in metres of A1."""
        unsupported(self, "get_stigmator")

    def _place_beam(self, position):
        pass

    def _blank_beam(self):
        pass

    def _unblank_beam(self):
        pass

    def _set_defocus(self, defocus):
        pass

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

    @abstractmethod
    def _move_stage(self, position):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _move_stage; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_stage(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_stage; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_image_shift(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_image_shift; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_beam_tilt(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_beam_tilt; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_beam_tilt(self,tilt):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_beam_tilt; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_diffraction_shift(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_diffraction_shift; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_diffraction_shift(self, tilt):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_diffraction_shift; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_parameters(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_parameters; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_fov(self, fov):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_fov; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _get_fov(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_fov; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _auto_focus(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _auto_focus; "
            "this vendor backend is missing the override"
        )

    @abstractmethod
    def _set_image_shift(self, shift):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_image_shift; "
            "this vendor backend is missing the override"
        )


if __name__ == '__main__':
    ElectronMicroscope.run_server()
