"""
What every scanning-beam instrument shares: ``ScanningBeamInstrument``.

A scanning-beam instrument rasters a blankable beam over a field of view on a stage: a STEM
(``electron_microscope/``), an SEM (``scanning_electron_microscope/``). This base holds what does not depend
on which, and backends implement only ``_x`` methods:

* limits: ``set_fov``, ``set_image_shift`` and the scans are checked against ``min_fov``, ``max_fov``,
  ``max_image_shift`` (and a scan region inside the field) before the backend sees the value, so every
  backend refuses the same values with the same messages (tests/test_parity.py);
* the beam current: ``beam_current_pa``, ``beam_current_source`` ("measured" or "default") and
  ``beam_current_age_s``, from the backend's ``_beam_current`` (by default the ``default_beam_current_pa``
  property, reported as assumed);
* ``stage_moving``, stage access (``get_stage``, ``move_stage``) and the instrument's ``clock_s``;
* the support-device proxies (``_connect_detector_proxies``, from ``proxy_addresses``);
* the SAMPLE device: ``get_dose`` asks it, and every dataset records what it says is loaded;
* one command lock: every public command that reaches the instrument runs under it (``hardware_call``),
  except ``blank_beam``, which never waits for another command. ``_check_connected`` lets a backend refuse
  a call when it has no connection;
* the declared risk and role of every command here (asyncroscopy.utils.commands).

Image commands return a string supplied by the concrete instrument, typically a DATA/Tiled unique id.
"""

import functools
import json
import math
import threading
from abc import abstractmethod
from typing import Optional

import numpy as np
import tango
from tango import AttrWriteType, DevDouble, DevState, DevString, DevVarFloatArray, DevVarStringArray
from tango.server import attribute, device_property

from asyncroscopy.instruments.instrument import Instrument
from asyncroscopy.routines.beam import DEFAULT_BEAM_CURRENT_PA
from asyncroscopy.utils.commands import action_hints_attribute, command

# Support devices every scanning-beam instrument connects: {name: the device property holding its address}.
BEAM_ADDRESSES = {
    "stage": "stage_device_address",
    "scan": "scan_device_address",
    "sample": "sample_device_address",
    "data": "data_device_address",
}
# The limit properties' defaults: what the base enforces when a deployment sets none.
DEFAULT_MIN_FOV = 1e-9
DEFAULT_MAX_FOV = 1e-4
DEFAULT_MAX_IMAGE_SHIFT = 1e-6  # m per axis: well inside a Spectra's beam-shift range
STAGE_ELEMENTS = ("x", "y", "z", "alpha", "beta")
STAGE_UNITS = ("m", "m", "m", "deg", "deg")
DETECTOR_TIMEOUT_MS = 12_000
DATA_TIMEOUT_MS = 3_600_000  # registering a large acquisition with Tiled can take long


def check_fov(fov: float, min_fov: float, max_fov: float) -> float:
    """Field of view in metres, refused outside [min_fov, max_fov]."""
    fov = float(fov)
    if not (math.isfinite(fov) and float(min_fov) <= fov <= float(max_fov)):
        raise ValueError(f"field of view must be between {min_fov} and {max_fov} m, got {fov}")
    return fov


def check_image_shift(shift: list[float], max_image_shift: float) -> list[float]:
    """Image shift [x, y] in metres, refused beyond +/-max_image_shift per axis."""
    value = np.asarray(shift, dtype=np.float64)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"image shift must be [x, y] in metres, got {np.atleast_1d(value).tolist()}")
    if np.any(np.abs(value) > float(max_image_shift)):
        raise ValueError(f"image shift must be within +/-{float(max_image_shift)} m per axis (did you pass nanometres?), got {value.tolist()}")
    return value.tolist()


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


class ScanningBeamInstrument(Instrument):
    """A blankable beam rastered over a field of view on a stage: the public commands every backend shares."""

    # Which support devices _connect_detector_proxies connects; a modality adds its own.
    proxy_addresses = BEAM_ADDRESSES

    scan_device_address = device_property(
        dtype=str,
        doc="Tango device address for the SCAN settings device. "
        "DB mode: 'test/detector/scan' "
        "No-DB mode: 'tango://127.0.0.1:8888/test/nodb/scan#dbase=no'",
    )

    stage_device_address = device_property(
        dtype=str,
        doc="Tango device address for the STAGE settings device. "
        "DB mode: 'asyncroscopy/stage/default' "
        "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/stage/default#dbase=no'",
    )

    sample_device_address = device_property(
        dtype=str,
        doc="Tango device address for the SAMPLE device (which sample is loaded, and the dose it has had). "
        "DB mode: 'asyncroscopy/sample/default' "
        "No-DB mode: 'tango://127.0.0.1:8888/asyncroscopy/sample/default#dbase=no'",
    )

    default_scan_detector = device_property(
        dtype=str,
        default_value="",
        doc="Scanning detector used when a caller names none: acquire_scanned_image with an empty list, a job plan without detectors.",
    )

    default_detector = attribute(
        label="Default scanning detector",
        dtype=str,
        access=AttrWriteType.READ,
        doc="The scanning detector used when a caller names none (the default_scan_detector property).",
    )

    # Limits checked before any backend sees a value; action_hints tells clients the same ranges.
    min_fov = device_property(dtype=float, default_value=DEFAULT_MIN_FOV, doc="Smallest field of view accepted (m).")
    max_fov = device_property(dtype=float, default_value=DEFAULT_MAX_FOV, doc="Largest field of view accepted (m).")
    max_image_shift = device_property(dtype=float, default_value=DEFAULT_MAX_IMAGE_SHIFT, doc="Largest |image shift| per axis accepted (m).")
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
    image_shift_limit_m = attribute(dtype=float, unit="m", access=AttrWriteType.READ, doc="Largest |image shift| per axis this instrument accepts (m): how far a job may move the image without the stage.")
    # What each command is (declared on its @command) and its arguments' units and ranges at this device's limits.
    action_hints = action_hints_attribute(extra=lambda device: device.argument_hints())

    def _init_device_attributes(self) -> None:
        self._microscope: Optional[object] = None
        self._detector_proxies: dict[str, tango.DeviceProxy] = {}
        # Every public command that reaches the instrument takes this lock (hardware_call); blank_beam does not.
        self._command_lock = threading.RLock()

    def _disconnect(self):
        self._microscope = None
        self.info_stream('Disconnected from microscope hardware')

    def _connect_detector_proxies(self) -> None:
        """A DeviceProxy for every configured detector and sub-device (``proxy_addresses``); a blank address is skipped."""
        for name, prop in self.proxy_addresses.items():
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

    def read_default_detector(self) -> str:
        return str(self.default_scan_detector)

    def read_beam_current_pa(self) -> float:
        return float(self._beam_current()["pa"])

    def read_beam_current_source(self) -> str:
        return str(self._beam_current()["source"])

    def read_beam_current_age_s(self) -> float:
        age = self._beam_current().get("age_s")
        return float("nan") if age is None else float(age)

    def read_image_shift_limit_m(self) -> float:
        return float(self.max_image_shift)

    @hardware_call()
    def read_stage_moving(self) -> bool:
        return bool(self._stage_moving())

    def argument_hints(self) -> dict:
        """The commands' arguments and returns with the units and ranges the checks enforce, at this device's
        limits, in the shape of ``action_hints`` (asyncroscopy.mcp.action_space)."""
        stage = {"type": "array", "items": "number", "elements": list(STAGE_ELEMENTS), "units": list(STAGE_UNITS)}
        return {
            "set_fov": {"args": [{"name": "fov", "type": "number", "units": "m", "min": float(self.min_fov), "max": float(self.max_fov)}]},
            "get_fov": {"returns": {"type": "number", "units": "m"}},
            "place_beam": {"args": [{"name": "position", "type": "array", "items": "number", "elements": ["x", "y"],
                                     "units": "fraction of the field of view", "min": 0.0, "max": 1.0}]},
            "move_stage": {"args": [{"name": "position", **stage}]},
            "get_stage": {"returns": stage},
            "acquire_scanned_image": {"primary": True, "args": [{"name": "detector_list", "type": "array", "items": "string",
                                                                 "doc": "scanning detectors; [] means the microscope's default_detector"}]},
        }

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

    @command(risk="high", role="other")
    def Disconnect(self) -> None:
        """Disconnect from microscope hardware gracefully."""
        self.set_state(DevState.OFF)
        self._disconnect()

    @command(
        dtype_in=DevVarStringArray,
        dtype_out=str,
        doc_in=":param detector_list: Scanning detector names, e.g. ['haadf'] on a STEM or ['se'] on an SEM. "
               "An empty list uses the microscope's default_detector.",
        risk="high", role="acquire.image", routine=True,
    )
    @hardware_call()
    def acquire_scanned_image(self, detector_list: list[str]) -> str:
        """
        Acquire an image with scanning detectors and return its DATA/Tiled key.
        An empty list uses the microscope's default_detector.
        """
        # Tango's wire protocol has no concept of "omitted", so remote callers (e.g. the
        # MCP/LLM bridge) send an empty list for "no detector named".
        if not detector_list:
            detector_list = [str(self.default_scan_detector)]
        scan = self._detector_proxies.get('scan')
        region = list(check_scan(scan.imsize, scan.dwell_time, list(scan.scan_region)))  # refused here, before the instrument
        return self._acquire_scanned_image(scan.imsize, scan.dwell_time, detector_list, region, scan.output_format)

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"x_m": ..., "y_m": ...} for the dose at one stage point, or {} for the exposure count only.', risk="low", role="setting.read")
    def get_dose(self, request: str) -> str:
        """Electron dose already put on the loaded sample, from the SAMPLE device's dose ledger: exposures counted (scanned
        frames and parked-beam spectra), and at a stage point the exposures that covered it and their summed dose in e/A^2;
        "not_counted" says what is not counted."""
        sample = self._detector_proxies.get("sample")
        if sample is None:
            return json.dumps({"sample_load_id": None, "sample_id": None, "exposures": None, "error": "no SAMPLE device is configured for this instrument"})
        return sample.get_dose(request)

    @command(dtype_in=DevVarFloatArray, dtype_out=None, doc_in=':param position: [x, y] beam position as fractions (0-1) of the field of view; [0.5, 0.5] is the centre.', risk="high", role="beam.place", routine=True)
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

    @command(dtype_in=DevVarFloatArray, dtype_out=None, doc_in=':param positions: Flat list [x0, y0, x1, y1, ...] of fractional beam positions.', risk="high", role="beam.place", routine=True)
    @hardware_call()
    def place_beam_list(self, positions: list[float]) -> None:
        """Place beam at multiple positions sequentially."""
        if len(positions) % 2 != 0:
            raise ValueError('Input must contain pairs of (x, y) values.')

        for i in range(0, len(positions), 2):
            x = float(positions[i])
            y = float(positions[i + 1])
            self._place_beam([x, y])

    @command(risk="low", role="beam.blank", stop=True)
    @hardware_call(serialized=False)  # never waits for a running command: blanking is the safe state
    def blank_beam(self) -> None:
        """Blank beam."""
        self._blank_beam()

    @command(risk="high", role="beam.unblank", routine=True)
    @hardware_call()
    def unblank_beam(self) -> None:
        """Unblank beam."""
        self._unblank_beam()

    @command(dtype_in=DevDouble, doc_in=':param fov: Field of view in metres (e.g. 2e-7 for 200 nm).', risk="low", role="optics.fov")
    @hardware_call()
    def set_fov(self, fov: float) -> None:
        """Set the field of view for the next acquisition, in meters."""
        self._set_fov(check_fov(fov, self.min_fov, self.max_fov))

    @command(dtype_out=DevDouble, risk="low", role="setting.read")
    @hardware_call()
    def get_fov(self) -> float:
        """Read the field of view for the next acquisition, in meters."""
        return self._get_fov()

    @command(dtype_in=DevVarFloatArray, doc_in=':param shift: [x, y] image shift in metres.', risk="high", role="setting.set", routine=True)
    @hardware_call()
    def set_image_shift(self, shift: list[float]) -> None:
        """Set the image shift to [x_shift, y_shift] in meters."""
        self._set_image_shift(check_image_shift(shift, self.max_image_shift))

    @command(dtype_out=DevVarFloatArray, risk="low", role="setting.read")
    @hardware_call()
    def get_image_shift(self) -> list[float]:
        """Get the image shift as [x, y] in m."""
        return self._get_image_shift()

    @command(dtype_in=DevVarFloatArray, dtype_out=DevVarFloatArray,
             doc_in=':param position: [x, y] stage position in metres for the centre of the next image.',
             risk="high", role="other", routine=True)
    @hardware_call()
    def aim_image_shift(self, position: list[float]) -> list[float]:
        """Centre the next image on stage position [x, y] (metres) with the image shift alone; the stage does not move.
        Sets the shift to the position minus the stage's and returns it, [x, y] in metres; set_image_shift [0, 0] goes
        back to the stage's centre. The instrument refuses a shift beyond its limit (image_shift_limit_m)."""
        stage = self._get_stage()
        shift = [float(position[0]) - float(stage[0]), float(position[1]) - float(stage[1])]
        self._set_image_shift(check_image_shift(shift, self.max_image_shift))
        return shift

    @command(dtype_in=DevVarFloatArray, dtype_out=DevVarFloatArray,
             doc_in=':param position: [x, y] stage position in metres to park the beam on (inside the current field of view).',
             risk="high", role="beam.place", routine=True)
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

    @command(dtype_out=DevString, risk="low", role="setting.read")
    @hardware_call()
    def get_parameters(self) -> str:
        """Get all status parameters, as JSON."""
        return self._get_parameters()

    @command(dtype_out=DevVarFloatArray, risk="low", role="stage.read")
    @hardware_call()
    def get_stage(self) -> list[float]:
        """Get the current stage position as [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        return self._get_stage()

    @command(dtype_in=DevVarFloatArray, doc_in=':param position: [x, y, z, alpha, beta]; x, y, z in metres, alpha and beta in degrees.', risk="high", role="stage.move", routine=True)
    @hardware_call()
    def move_stage(self, position: list[float]) -> None:
        """Move the stage to [x, y, z, alpha, beta], with x/y/z in meters and tilts in degrees."""
        self._move_stage(position)

    # ------------------------------------------------------------------
    # What a backend implements
    # ------------------------------------------------------------------
    @abstractmethod
    def _acquire_scanned_image(self, imsize: int, dwell_time: float, detector_list: list[str], scan_region: list[float], output_format: str = '.h5') -> str:
        """Vendor-specific scanned image acquisition implementation."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _acquire_scanned_image; this vendor backend is missing the override")

    def _beam_current(self) -> dict:
        """The beam current: {"pa", "source" ("measured" or "default"), "age_s" (measured) or "reason" (assumed)}.
        By default the default_beam_current_pa property, assumed: a backend that measures its current overrides this."""
        return {"pa": float(self.default_beam_current_pa), "source": "default", "age_s": None,
                "reason": f"{type(self).__name__} does not measure its beam current; default_beam_current_pa is assumed"}

    def _stage_moving(self) -> bool:
        """True while a stage move is under way; a backend whose moves return only on arrival may return False."""
        unsupported(self, "stage_moving")

    def _place_beam(self, position):
        pass

    def _blank_beam(self):
        pass

    def _unblank_beam(self):
        pass

    @abstractmethod
    def _move_stage(self, position):
        raise NotImplementedError(f"{type(self).__name__} does not implement _move_stage; this vendor backend is missing the override")

    @abstractmethod
    def _get_stage(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement _get_stage; this vendor backend is missing the override")

    @abstractmethod
    def _get_image_shift(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement _get_image_shift; this vendor backend is missing the override")

    @abstractmethod
    def _set_image_shift(self, shift):
        raise NotImplementedError(f"{type(self).__name__} does not implement _set_image_shift; this vendor backend is missing the override")

    @abstractmethod
    def _get_parameters(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement _get_parameters; this vendor backend is missing the override")

    @abstractmethod
    def _set_fov(self, fov):
        raise NotImplementedError(f"{type(self).__name__} does not implement _set_fov; this vendor backend is missing the override")

    @abstractmethod
    def _get_fov(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement _get_fov; this vendor backend is missing the override")
