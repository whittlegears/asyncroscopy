"""A recorded fake of the Thermo Fisher AutoScript TEM client, for tests with no instrument.

It mirrors only the part of ``autoscript_tem_microscope_client.TemMicroscopeClient``
that ``AutoScriptMicroscope`` (and the AutoScript STAGE device) use, with the
member names, types and units of AutoScript 1.18 (the client in ``stubs/``):

- ``specimen.stage``: ``position`` (a ``StagePosition``: x, y, z in metres, a and b
  in radians), ``absolute_move``, ``absolute_move_safe`` (tilt to zero first, then the
  move), ``get_holder_type``, ``get_axis_limits``;
- ``optics``: ``optical_mode`` ("Stem" / "Tem"), ``projector_mode``,
  ``scan_field_of_view`` (m), ``focusing.stem.objective.defocus`` and
  ``focusing.tem.defocus`` (m), ``blanker``, ``paused_scan_beam_position``
  (fractions 0-1), ``monochromator.focus`` (V), ``deflectors`` (beam and image
  shift in m, beam tilt unitless -1..+1, image tilt, the diffraction shift, in rad), and the read-only status members
  ``get_parameters`` reads;
- ``acquisition``: ``acquire_stem_images_advanced``, ``acquire_camera_image_advanced``;
- ``analysis.eds.acquire_spectrum``;
- ``detectors.screen``: ``position``, ``insert``, ``retract``, ``measure_current`` (A);
- ``auto_functions``: ``run_opti_stem`` (a RunOptiStemResults), ``run_objective_auto_stigmator``,
  ``run_stem_auto_focus`` and ``run_beam_tilt_auto_focus`` (their Results: defocus and astigmatism in m);
- ``specimen.stage.is_moving`` (whatever a test sets ``moving`` to: the stub does not say whether a move returns
  before the stage arrives) and ``stop_jogging``; ``acquisition.acquire_camera_series_advanced``;
  ``analysis.eels.acquire_spectra`` and ``acquire_spectrum_image``; ``optics.beam_stopper`` (insert,
  half_insert, retract); ``optics.camera_length.value`` (set from ``available_values``);
  ``detectors.get_scanning_detector`` (insert, retract);
- ``detectors``: ``scanning_detectors``, ``camera_detectors``, ``eds_detectors``,
  ``eels_detectors`` and ``get_camera_detector`` (``exposure_time_limits``, Flucam only, and
  ``get_supported_image_sizes``), with the DetectorType / CameraType names of a Spectra 300;
- ``optics.camera_length`` and ``optics.magnification`` (``value``: a CameraLengthData /
  MagnificationData; ``available_values``), ``convergence_angle`` (rad), ``beam_diameter`` (m),
  ``blanker.auto_beam_blank.mode.value``;
- ``vacuum``, ``service.system`` (name, serial_number, version), ``service.autoscript``
  (server.version, server.is_offline, client.version), ``source.acceleration_voltage`` and
  ``source.is_accelerator_on``. The deprecated ``optics.acceleration_voltage`` and
  ``optics.is_accelerator_on`` are left out, so a test fails if the code uses them.

Every read, write and call is appended to ``client.calls`` as
``{"path": "optics.scan_field_of_view", "op": "set", "value": 2e-7}``, so a test
can assert what reached the instrument, in order. A value the instrument would
not take raises ``ApplicationServerException``, as the AutoScript server does;
a value of the wrong type raises ``TypeError``, as the client does before
sending. The ``HARDWARE_LIMITS`` are placeholders until first light gives the
Spectra 300's own; they are deliberately wider than the software limits the
microscope devices apply, so a test can tell which layer refused.

The blanker starts unblanked, and every acquisition opens it and leaves it open
(``acquisition_opens_blanker``, the worst case: whether the real column does is
not known before first light), so a test sees a beam left on after a frame.
The camera refuses an exposure outside ``HARDWARE_LIMITS["camera_exposure"]``.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
from autoscript_core.common import ApplicationServerException
from autoscript_tem_microscope_client.structures import (CameraImageSize, CameraLengthData, Limits, MagnificationData, Point, RunBeamTiltAutoFocusResults,
                                                          RunOptiStemResults, RunStemAutoFocusResults, StagePosition)

HARDWARE_LIMITS = {
    "stage_x": (-1.0e-3, 1.0e-3),  # m
    "stage_y": (-1.0e-3, 1.0e-3),
    "stage_z": (-5.0e-4, 5.0e-4),
    "stage_a": (-math.radians(70.0), math.radians(70.0)),  # rad
    "stage_b": (-math.radians(30.0), math.radians(30.0)),
    "scan_field_of_view": (1.0e-10, 1.0e-3),  # m
    "defocus": (-1.0e-4, 1.0e-4),  # m
    "monochromator_focus": (0.0, 200.0),  # V
    "beam_shift": (-1.0e-5, 1.0e-5),  # m, each of x and y
    "image_shift": (-1.0e-5, 1.0e-5),
    "beam_tilt": (-1.0, 1.0),  # unitless
    "image_tilt": (-0.1, 0.1),  # rad, the diffraction shift; placeholder
    "fraction": (0.0, 1.0),  # paused beam position, scan region
    "camera_exposure": (1.0e-3, 60.0),  # s, placeholder
}


def screen_current_a(focus_v: float) -> float:
    """The fake gun: screen current (A) rises smoothly from 5 pA to 300 pA over a 0-200 V monochromator focus."""
    f = min(max(float(focus_v), 0.0), 200.0) / 200.0
    return (5.0 + 295.0 * f * f) * 1e-12


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, path: str, op: str, value: Any = None) -> None:
        self.calls.append({"path": path, "op": op, "value": value})


def _refuse_outside(name: str, value: float, what: str) -> None:
    low, high = HARDWARE_LIMITS[name]
    if not (isinstance(value, Real) and math.isfinite(float(value)) and low <= float(value) <= high):
        raise ApplicationServerException(f"{what} {value!r} is out of range [{low}, {high}]")


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Cannot set {what} with the given type. Read the documentation for details.")
    return float(value)


def _point(value: Any, name: str, what: str) -> Point:
    point = Point.convert_from(value) if not isinstance(value, Point) else value
    for component in (point.x, point.y):
        _refuse_outside(name, _number(component, what), what)
    return Point(float(point.x), float(point.y))


class _Node:
    """A member of the client tree whose reads, writes and calls are recorded under ``path``."""

    def __init__(self, record: _Recorder, path: str) -> None:
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_path", path)

    def _rec(self, member: str, op: str, value: Any = None) -> None:
        self._record(f"{self._path}.{member}", op, value)


class _Stage(_Node):
    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_position", StagePosition(0.0, 0.0, 0.0, 0.0, 0.0))
        object.__setattr__(self, "holder_type", "DoubleTilt")
        object.__setattr__(self, "moving", False)

    @property
    def position(self) -> StagePosition:
        p = self._position
        self._rec("position", "get")
        return StagePosition(p.x, p.y, p.z, p.a, p.b)

    def absolute_move(self, position: StagePosition | list[float]) -> None:
        p = position if isinstance(position, StagePosition) else StagePosition(*position)
        self._rec("absolute_move", "call", [p.x, p.y, p.z, p.a, p.b])
        self._move(p)

    def absolute_move_safe(self, position: StagePosition | list[float]) -> None:
        """AutoScript sets the tilt to zero first, so the holder clears the pole piece, then moves."""
        p = position if isinstance(position, StagePosition) else StagePosition(*position)
        self._rec("absolute_move_safe", "call", [p.x, p.y, p.z, p.a, p.b])
        self._move(p)

    def stop_jogging(self) -> None:
        self._rec("stop_jogging", "call")

    def _move(self, p: StagePosition) -> None:
        for axis in "xyz":
            _refuse_outside(f"stage_{axis}", _number(getattr(p, axis), f"stage {axis}"), f"stage {axis}")
        for axis in "ab":
            if getattr(p, axis) is not None:
                _refuse_outside(f"stage_{axis}", _number(getattr(p, axis), f"stage {axis}"), f"stage {axis}")
        if self.holder_type != "DoubleTilt" and p.b not in (None, 0.0):
            raise ApplicationServerException("this holder has no beta tilt")
        old = self._position
        object.__setattr__(self, "_position", StagePosition(p.x, p.y, p.z, old.a if p.a is None else p.a, old.b if p.b is None else p.b))

    @property
    def is_moving(self) -> bool:
        self._rec("is_moving", "get")
        return self.moving

    def get_holder_type(self) -> str:
        self._rec("get_holder_type", "call")
        return self.holder_type

    def get_axis_limits(self, axis: str) -> Limits:
        self._rec("get_axis_limits", "call", axis)
        low, high = HARDWARE_LIMITS[f"stage_{axis.lower()}"]
        return Limits(low, high)


class _Defocus(_Node):
    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_defocus", 0.0)

    @property
    def defocus(self) -> float:
        self._rec("defocus", "get")
        return self._defocus

    @defocus.setter
    def defocus(self, value: float) -> None:
        self._rec("defocus", "set", value)
        _refuse_outside("defocus", _number(value, "defocus"), "defocus")
        object.__setattr__(self, "_defocus", float(value))


class _Blanker(_Node):
    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_blanked", False)  # a column as the operator leaves it

    def blank(self) -> None:
        self._rec("blank", "call")
        object.__setattr__(self, "_blanked", True)

    def unblank(self) -> None:
        self._rec("unblank", "call")
        object.__setattr__(self, "_blanked", False)

    @property
    def is_beam_blanked(self) -> bool:
        self._rec("is_beam_blanked", "get")
        return self._blanked


class _Monochromator(_Node):
    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_focus", 80.0)

    @property
    def focus(self) -> float:
        self._rec("focus", "get")
        return self._focus

    @focus.setter
    def focus(self, value: float) -> None:
        self._rec("focus", "set", value)
        _refuse_outside("monochromator_focus", _number(value, "monochromator focus"), "monochromator focus")
        object.__setattr__(self, "_focus", float(value))


class _Deflectors(_Node):
    UNITS = {"beam_shift": "beam_shift", "image_shift": "image_shift", "beam_tilt": "beam_tilt", "image_tilt": "image_tilt"}

    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_values", {name: Point(0.0, 0.0) for name in self.UNITS})

    def __getattr__(self, name: str) -> Point:
        if name in _Deflectors.UNITS:
            self._rec(name, "get")
            value = self._values[name]
            return Point(value.x, value.y)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in _Deflectors.UNITS:
            raise AttributeError(f"the AutoScript deflectors have no member {name!r} the fake mirrors")
        self._rec(name, "set", list(value) if isinstance(value, (list, tuple)) else value)
        self._values[name] = _point(value, self.UNITS[name], name.replace("_", " "))

    def get_available_deflectors(self) -> list[str]:
        self._rec("get_available_deflectors", "call")
        return []

    def get_deflector_value(self, deflector: str) -> Point:
        self._rec("get_deflector_value", "call", deflector)
        raise ApplicationServerException(f"no deflector {deflector!r}")


class _Stigmators(_Node):
    """The objective stigmator, in AutoScript's optical units (about -1 to 1)."""

    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_objective", Point(0.0, 0.0))

    @property
    def objective_stigmator(self) -> Point:
        self._rec("objective_stigmator", "get")
        return Point(self._objective.x, self._objective.y)

    @objective_stigmator.setter
    def objective_stigmator(self, value: Any) -> None:
        self._rec("objective_stigmator", "set", list(value) if isinstance(value, (list, tuple)) else value)
        x, y = (float(v) for v in (value if isinstance(value, (list, tuple)) else (value.x, value.y)))
        if not (abs(x) <= 1.0 and abs(y) <= 1.0):
            raise ApplicationServerException(f"objective stigmator ({x}, {y}) is outside -1 to 1")
        object.__setattr__(self, "_objective", Point(x, y))


class _Apertures(_Node):
    def get_available(self) -> list[str]:
        self._rec("get_available", "call")
        return []


CAMERA_LENGTHS = tuple(CameraLengthData(nominal=n, calibrated=n * 1.03, label=f"{n * 1e3:g} mm") for n in (0.06, 0.091, 0.115, 0.185))


class _Camera(_Node):
    """One camera detector. AutoScript gives exposure_time_limits for the Flucam only and raises for the others."""

    def __init__(self, record: _Recorder, path: str, name: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "name", name)

    @property
    def exposure_time_limits(self) -> Limits:
        self._rec("exposure_time_limits", "get")
        if self.name != "Flucam":
            raise ApplicationServerException("exposure time limits are available for the Flucam only")
        return Limits(*HARDWARE_LIMITS["camera_exposure"])

    def get_supported_image_sizes(self) -> list:
        self._rec("get_supported_image_sizes", "call")
        return [CameraImageSize(size=size, fixed_readout_area=area) for size, area in ((4096, "Full"), (2048, "Half"), (1024, "Quarter"))]


class _Detectors(_Node):
    SCANNING = ["BF", "DF2", "DF4", "HAADF", "BF-S", "DF-S"]
    CAMERAS = ["Flucam", "BM-Ceta"]

    def __init__(self, record: _Recorder, path: str, monochromator: "_Monochromator") -> None:
        super().__init__(record, path)
        object.__setattr__(self, "screen", _Screen(record, f"{path}.screen", monochromator))

    @property
    def scanning_detectors(self) -> list[str]:
        self._rec("scanning_detectors", "get")
        return list(self.SCANNING)

    @property
    def camera_detectors(self) -> list[str]:
        self._rec("camera_detectors", "get")
        return list(self.CAMERAS)

    @property
    def eds_detectors(self) -> list[str]:
        self._rec("eds_detectors", "get")
        return ["SuperX"]

    @property
    def eels_detectors(self) -> list[str]:
        self._rec("eels_detectors", "get")
        return ["Zebra"]  # EelsDetectorType's one value

    def get_scanning_detector(self, detector_name: str) -> _ScanningDetector:
        self._rec("get_scanning_detector", "call", detector_name)
        if detector_name not in self.SCANNING:
            raise ApplicationServerException(f"no scanning detector {detector_name!r}")
        return _ScanningDetector(self._record, f"{self._path}.scanning[{detector_name}]")

    def get_camera_detector(self, detector_name: str) -> _Camera:
        self._rec("get_camera_detector", "call", detector_name)
        if detector_name not in self.CAMERAS:
            raise ApplicationServerException(f"no camera {detector_name!r}")
        return _Camera(self._record, f"{self._path}.camera[{detector_name}]", detector_name)


class _Optics(_Node):
    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        sub = lambda cls, name: cls(record, f"{path}.{name}")  # noqa: E731
        focusing = _Node(record, f"{path}.focusing")
        object.__setattr__(focusing, "stem", _Node(record, f"{path}.focusing.stem"))
        object.__setattr__(focusing.stem, "objective", sub(_Defocus, "focusing.stem.objective"))
        object.__setattr__(focusing, "tem", sub(_Defocus, "focusing.tem"))
        state = {
            "optical_mode": "Stem",
            "projector_mode": "Imaging",
            "scan_field_of_view": 2.0e-7,
            "paused_scan_beam_position": Point(0.5, 0.5),
            "scan_rotation": 0.0,
            "illumination_mode": "Probe",
            "convergence_angle": 0.025,
            "beam_diameter": 1.0e-10,
            "objective_lens_mode": "HM",
            "spot_size_index": 6,
            "is_eftem_on": False,
        }
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_opens_blanker", True)  # see FakeAutoScriptClient.acquisition_opens_blanker
        for name, value in {
            "focusing": focusing,
            "blanker": sub(_Blanker, "blanker"),
            "monochromator": sub(_Monochromator, "monochromator"),
            "deflectors": sub(_Deflectors, "deflectors"),
            "stigmators": sub(_Stigmators, "stigmators"),
            "aperture_mechanisms": sub(_Apertures, "aperture_mechanisms"),
            "beam_stopper": sub(_BeamStopper, "beam_stopper"),
            "camera_length": sub(_CameraLength, "camera_length"),
            "magnification": _ReadOnly(record, f"{path}.magnification", value=MagnificationData(nominal=1.0e5, calibrated=1.02e5, label="100 kx"),
                                       available_values=[MagnificationData(nominal=1.0e5, calibrated=1.02e5, label="100 kx")]),
        }.items():
            object.__setattr__(self, name, value)
        auto_blank = _Node(record, f"{path}.blanker.auto_beam_blank")
        object.__setattr__(auto_blank, "mode", _ReadOnly(record, f"{path}.blanker.auto_beam_blank.mode", value="Off"))
        object.__setattr__(self.blanker, "auto_beam_blank", auto_blank)

    WRITABLE = {"optical_mode", "projector_mode", "scan_field_of_view", "paused_scan_beam_position"}

    def __getattr__(self, name: str) -> Any:
        state = object.__getattribute__(self, "_state")
        if name in state:
            self._rec(name, "get")
            value = state[name]
            return Point(value.x, value.y) if isinstance(value, Point) else value
        raise AttributeError(f"the fake AutoScript optics have no member {name!r} (mirror it only if the code uses it)")

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in _Optics.WRITABLE:
            raise AttributeError(f"the fake AutoScript optics cannot set {name!r}")
        self._rec(name, "set", list(value) if isinstance(value, (list, tuple)) else value)
        if name == "scan_field_of_view":
            _refuse_outside("scan_field_of_view", _number(value, "scan field of view"), "scan field of view")
            value = float(value)
        elif name == "paused_scan_beam_position":
            value = _point(value, "fraction", "paused scan beam position")
        elif name == "optical_mode" and value not in ("Stem", "Tem"):
            raise ApplicationServerException(f"unknown optical mode {value!r}")
        elif name == "projector_mode" and value not in ("Imaging", "Diffraction"):
            raise ApplicationServerException(f"unknown projector mode {value!r}")
        self._state[name] = value


class _ReadOnly(_Node):
    def __init__(self, record: _Recorder, path: str, **values: Any) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in values:
            self._rec(name, "get")
            return values[name]
        raise AttributeError(name)


class FakeAdornedImage:
    """What AutoScript acquisitions return: ``data`` (a NumPy array) and ``metadata``."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.metadata = None


def _open_blanker(optics: _Optics) -> None:
    """The instrument opening the blanker for an acquisition (not a call the code sent, so not recorded)."""
    if optics._opens_blanker:
        object.__setattr__(optics.blanker, "_blanked", False)


class _Acquisition(_Node):
    def __init__(self, record: _Recorder, path: str, optics: _Optics) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_optics", optics)

    def acquire_stem_images_advanced(self, settings: Any) -> list[FakeAdornedImage]:
        self._rec("acquire_stem_images_advanced", "call", {
            "size": settings.size, "dwell_time": settings.dwell_time, "detector_types": list(settings.detector_types),
            "region": None if settings.region is None else [settings.region.rectangle.left, settings.region.rectangle.top,
                                                           settings.region.rectangle.width, settings.region.rectangle.height],
        })
        if self._optics._state["optical_mode"] != "Stem":
            raise ApplicationServerException("STEM acquisition needs the column in STEM mode")
        if not (isinstance(settings.size, int) and settings.size > 0 and float(settings.dwell_time) > 0):
            raise ApplicationServerException("size must be a positive integer and dwell_time > 0")
        if settings.region is not None:
            r = settings.region.rectangle
            values = [float(v) for v in (r.left, r.top, r.width, r.height)]
            if not (all(math.isfinite(v) for v in values) and values[0] >= 0 and values[1] >= 0 and values[2] > 0
                    and values[3] > 0 and values[0] + values[2] <= 1 + 1e-9 and values[1] + values[3] <= 1 + 1e-9):
                raise ApplicationServerException(f"scan region {values} is outside the scan area")
        size = int(settings.size)
        _open_blanker(self._optics)
        return [FakeAdornedImage(np.zeros((size, size), dtype=np.uint16)) for _ in settings.detector_types]

    def acquire_camera_series_advanced(self, settings: Any, length: int) -> list[FakeAdornedImage]:
        self._rec("acquire_camera_series_advanced", "call", {"camera_detector": settings.camera_detector, "size": settings.size,
                                                             "exposure_time": settings.exposure_time, "length": length})
        _refuse_outside("camera_exposure", _number(settings.exposure_time, "camera exposure"), "camera exposure")
        _open_blanker(self._optics)
        return [FakeAdornedImage(np.zeros((int(settings.size), int(settings.size)), dtype=np.uint16)) for _ in range(int(length))]

    def acquire_camera_image_advanced(self, settings: Any) -> FakeAdornedImage:
        self._rec("acquire_camera_image_advanced", "call", {"camera_detector": settings.camera_detector, "size": settings.size,
                                                            "exposure_time": settings.exposure_time})
        _refuse_outside("camera_exposure", _number(settings.exposure_time, "camera exposure"), "camera exposure")
        size = int(settings.size)
        _open_blanker(self._optics)
        return FakeAdornedImage(np.zeros((size, size), dtype=np.uint16))


class _Eds(_Node):
    def __init__(self, record: _Recorder, path: str, optics: _Optics) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_optics", optics)

    def acquire_spectrum(self, settings: Any) -> FakeAdornedImage:
        self._rec("acquire_spectrum", "call", {"exposure_time": settings.exposure_time})
        if not float(settings.exposure_time) > 0:
            raise ApplicationServerException("exposure_time must be > 0")
        _open_blanker(self._optics)
        return FakeAdornedImage(np.zeros(2048, dtype=np.uint32))

    def acquire_spectrum_image(self, settings: Any) -> _FakeSpectrumImage:
        self._rec("acquire_spectrum_image", "call", {"dwell_time": settings.dwell_time, "size": settings.size})
        _open_blanker(self._optics)
        return _FakeSpectrumImage()


class _Eels(_Node):
    def __init__(self, record: _Recorder, path: str, optics: _Optics) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_optics", optics)

    def acquire_spectra(self, settings: Any) -> list[FakeAdornedImage]:
        self._rec("acquire_spectra", "call", {"eels_detector": settings.eels_detector,
                                              "exposure_time": [e.exposure_time for e in settings.eels_exposure_settings]})
        _open_blanker(self._optics)
        return [FakeAdornedImage(np.zeros(2048, dtype=np.float32)) for _ in settings.eels_exposure_settings]

    def acquire_spectrum_image(self, settings: Any) -> _FakeSpectrumImage:
        self._rec("acquire_spectrum_image", "call", {"eels_detector": settings.eels_detector, "size": settings.size})
        _open_blanker(self._optics)
        return _FakeSpectrumImage()


class _BeamStopper(_Node):
    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_state", "Retracted")

    @property
    def insertion_state(self) -> str:
        self._rec("insertion_state", "get")
        return self._state

    def insert(self) -> None:
        self._rec("insert", "call")
        object.__setattr__(self, "_state", "Inserted")

    def half_insert(self) -> None:
        self._rec("half_insert", "call")
        object.__setattr__(self, "_state", "HalfInserted")

    def retract(self) -> None:
        self._rec("retract", "call")
        object.__setattr__(self, "_state", "Retracted")


class _CameraLength(_Node):
    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_value", CAMERA_LENGTHS[2])

    @property
    def value(self) -> CameraLengthData:
        self._rec("value", "get")
        return self._value

    @value.setter
    def value(self, value: CameraLengthData) -> None:
        self._rec("value", "set", getattr(value, "label", value))
        if not any(value.label == v.label for v in CAMERA_LENGTHS):
            raise ApplicationServerException(f"camera length {value!r} is not in available_values")
        object.__setattr__(self, "_value", value)

    @property
    def available_values(self) -> list[CameraLengthData]:
        self._rec("available_values", "get")
        return list(CAMERA_LENGTHS)


class _ScanningDetector(_Node):
    def insert(self) -> None:
        self._rec("insert", "call")

    def retract(self) -> None:
        self._rec("retract", "call")


class _FakeSpectrumImage:
    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            handle.write(b"velox emd bytes")


class _Screen(_Node):
    def __init__(self, record: _Recorder, path: str, monochromator: _Monochromator) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_position", "Retracted")
        object.__setattr__(self, "_monochromator", monochromator)

    @property
    def position(self) -> str:
        self._rec("position", "get")
        return self._position

    def insert(self) -> None:
        self._rec("insert", "call")
        object.__setattr__(self, "_position", "Inserted")

    def retract(self) -> None:
        self._rec("retract", "call")
        object.__setattr__(self, "_position", "Retracted")

    def measure_current(self) -> float:
        self._rec("measure_current", "call")
        if self._position != "Inserted":
            return 0.0  # the beam does not reach a retracted screen
        return screen_current_a(self._monochromator._focus)


class _AutoFunctions(_Node):
    def __init__(self, record: _Recorder, path: str, optics: _Optics) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_optics", optics)

    def run_opti_stem(self, settings: Any = None) -> RunOptiStemResults:
        self._rec("run_opti_stem", "call", getattr(settings, "method", None))
        if self._optics._state["optical_mode"] != "Stem":
            raise ApplicationServerException("OptiSTEM runs in STEM mode only")
        return RunOptiStemResults(quality="Improved")

    def run_stem_auto_focus(self, settings: Any) -> RunStemAutoFocusResults:
        self._rec("run_stem_auto_focus", "call", {"scanning_detector": settings.scanning_detector, "accuracy_level": settings.accuracy_level})
        if self._optics._state["optical_mode"] != "Stem":
            raise ApplicationServerException("the STEM autofocus runs in STEM mode only")
        return RunStemAutoFocusResults(astigmatism=2e-9, defocus=-4e-9)

    def run_beam_tilt_auto_focus(self, settings: Any) -> RunBeamTiltAutoFocusResults:
        self._rec("run_beam_tilt_auto_focus", "call", {"camera_detector": settings.camera_detector})
        if self._optics._state["optical_mode"] != "Tem":
            raise ApplicationServerException("the beam-tilt autofocus runs in TEM mode only")
        return RunBeamTiltAutoFocusResults(astigmatism=3e-9, defocus=5e-9, drift=Point(1e-10, -2e-10))

    def run_objective_auto_stigmator(self, settings: Any) -> None:
        self._rec("run_objective_auto_stigmator", "call", getattr(settings, "camera_detector", None))
        if self._optics._state["optical_mode"] != "Tem":
            raise ApplicationServerException("the objective auto-stigmator runs in TEM mode only")


class _ColumnValves(_Node):
    def __init__(self, record: _Recorder, path: str) -> None:
        super().__init__(record, path)
        object.__setattr__(self, "_state", "Closed")

    @property
    def state(self) -> str:
        self._rec("state", "get")
        return self._state

    def open(self) -> None:
        self._rec("open", "call")
        object.__setattr__(self, "_state", "Opened")

    def close(self) -> None:
        self._rec("close", "call")
        object.__setattr__(self, "_state", "Closed")


class FakeAutoScriptClient:
    """Stands in for ``TemMicroscopeClient`` after ``connect``; see the module docstring for its scope."""

    def __init__(self) -> None:
        record = _Recorder()
        self._record = record
        self.calls = record.calls
        self.specimen = _Node(record, "specimen")
        object.__setattr__(self.specimen, "stage", _Stage(record, "specimen.stage"))
        self.optics = _Optics(record, "optics")
        self.acquisition = _Acquisition(record, "acquisition", self.optics)
        self.analysis = _Node(record, "analysis")
        object.__setattr__(self.analysis, "eds", _Eds(record, "analysis.eds", self.optics))
        object.__setattr__(self.analysis, "eels", _Eels(record, "analysis.eels", self.optics))
        self.detectors = _Detectors(record, "detectors", self.optics.monochromator)
        self.auto_functions = _AutoFunctions(record, "auto_functions", self.optics)
        self.vacuum = _ReadOnly(record, "vacuum", state="Ready")
        object.__setattr__(self.vacuum, "column_valves", _ColumnValves(record, "vacuum.column_valves"))
        self.service = _Node(record, "service")
        object.__setattr__(self.service, "system", _ReadOnly(record, "service.system", name="Spectra 300 (fake)", serial_number="FAKE-0000",
                                                             version="fake TEM server"))
        autoscript = _Node(record, "service.autoscript")
        object.__setattr__(autoscript, "server", _ReadOnly(record, "service.autoscript.server", version="1.18.0 (fake)", is_offline=True))
        object.__setattr__(autoscript, "client", _ReadOnly(record, "service.autoscript.client", version="1.18.0"))
        object.__setattr__(self.service, "autoscript", autoscript)
        self.source = _ReadOnly(record, "source", is_accelerator_on=True)
        object.__setattr__(self.source, "acceleration_voltage", _ReadOnly(record, "source.acceleration_voltage", value=300e3))

    @property
    def acquisition_opens_blanker(self) -> bool:
        """Whether acquisitions open the blanker and leave it open (True, the worst case, by default)."""
        return self.optics._opens_blanker

    @acquisition_opens_blanker.setter
    def acquisition_opens_blanker(self, value: bool) -> None:
        object.__setattr__(self.optics, "_opens_blanker", bool(value))

    def connect(self, host: str = "localhost", port: int | None = None) -> None:
        self._record("connect", "call", [host, port])

    def writes(self, path: str | None = None) -> list[dict]:
        """Recorded writes and calls (not reads), optionally only those whose path starts with ``path``."""
        return [c for c in self.calls if c["op"] != "get" and (path is None or c["path"].startswith(path))]
