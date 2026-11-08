"""Behavioural parity: the digital twin and AutoScriptMicroscope (Spectra 300) on a recorded fake client.

Both backends inherit every command from ``ElectronMicroscope``, so a check of command
names would pass trivially. These checks run each command's body on both devices and
compare behaviour: the arguments they accept, the out-of-range values they refuse
(and, on AutoScript, that a refused value never reaches the instrument), and the
shapes and keys of what they return. The AutoScript side runs over
``tests/fake_autoscript.py``, which records every call that would reach the
instrument. Where one backend truly cannot do what the other does, the gap is
written down in ``TWIN_LACKS`` / ``AUTOSCRIPT_ONLY`` and asserted, not skipped, so
closing a gap fails a test and moves the command into the shared checks.

Test-only hook: ``ASYNCROSCOPY_PARITY_AUTOSCRIPT=<path to an auto_script.py>`` runs the
suite against another copy of the AutoScript backend (for example an older one), to
see which of its bugs these checks catch.
"""

from __future__ import annotations

import importlib.util
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import tango
from autoscript_core.common import ApplicationServerException
from autoscript_tem_microscope_client.structures import StagePosition

from asyncroscopy import settings
from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope as EM
from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE, TILT_LIMITS_DEG
from asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript import AutoScriptSTAGE
from tests.fake_autoscript import FakeAutoScriptClient

REFUSED = (ValueError, TypeError, tango.DevFailed, ApplicationServerException)
SOFTWARE_LIMITS = {"min_fov": 1e-9, "max_fov": 1e-4, "max_defocus": 1e-5, "max_image_shift": 1e-6}  # the twin's defaults; AutoScript's must match


def _autoscript_class():
    path = settings.get().parity_autoscript
    if not path:
        from asyncroscopy.instruments.electron_microscope.auto_script import AutoScriptMicroscope

        return AutoScriptMicroscope
    spec = importlib.util.spec_from_file_location("parity_auto_script_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AutoScriptMicroscope


AutoScriptMicroscope = _autoscript_class()


class _UnregisteredAutoScript(AutoScriptMicroscope):
    """Never registered with Tango; the default repr of an unregistered device crashes pytest's tracebacks."""

    def __repr__(self) -> str:
        return "<AutoScriptMicroscope on the fake client>"

    __str__ = __repr__


class FakeDataServer:
    def __init__(self, save_path: Path):
        self.save_path = str(save_path)

    def register_path(self, path: str) -> str:
        return path


class FakeStageProxy:
    """The AutoScript STAGE device's own code (travel limits, degrees <-> radians) over the fake client, without Tango."""

    _read_position = AutoScriptSTAGE._read_position
    _write_position = AutoScriptSTAGE._write_position
    _read_beta_tilt_enabled = AutoScriptSTAGE._read_beta_tilt_enabled
    _check_limits = STAGE._check_limits
    _check_z_and_tilt = STAGE._check_z_and_tilt
    read_beta_tilt_enabled = STAGE.read_beta_tilt_enabled
    high_tilt_deg, high_tilt_max_abs_z_m = 15.0, 1e-4  # STAGE's defaults

    def __init__(self, client: FakeAutoScriptClient):
        self._microscope = client
        self._limits = {"x": (-1e-3, 1e-3), "y": (-1e-3, 1e-3), "z": (-5e-4, 5e-4), **TILT_LIMITS_DEG}  # STAGE's defaults

    @property
    def position(self):
        return self._read_position()

    @position.setter
    def position(self, value):
        STAGE.write_position(self, value)


def autoscript_device(client: FakeAutoScriptClient, save_dir: Path):
    """An AutoScriptMicroscope connected to the fake client, with the Spectra 300 stack's settings devices."""
    device = _UnregisteredAutoScript.__new__(_UnregisteredAutoScript)
    device._tango_properties = {}
    for name, value in {**SOFTWARE_LIMITS, "default_beam_current_pa": 50.0, "default_scan_detector": "haadf"}.items():
        if hasattr(AutoScriptMicroscope, name):
            setattr(device, name, value)
    device._microscope = client
    device._connect_error = None
    device._command_lock, device._blank_client, device._blank_count = threading.RLock(), None, 0
    device.screen_current_calibration = None
    device.info_stream = device.warn_stream = device.error_stream = device.debug_stream = lambda *a, **k: None
    device._detector_proxies = {
        "data": FakeDataServer(save_dir),
        "stage": FakeStageProxy(client),
        "scan": SimpleNamespace(imsize=32, dwell_time=1e-6, scan_region=[0.0, 0.0, 1.0, 1.0], output_format=".h5"),
        "eds": SimpleNamespace(exposure_time=0.01),
    }
    return device


class Backend:
    def __init__(self, name: str, device, client: FakeAutoScriptClient | None = None):
        self.name, self.device, self.client = name, device, client

    def call(self, command: str, *args):
        """Run the Tango command's body on this device, as a client's call would."""
        return getattr(EM, command)(self.device, *args)

    def instrument_writes(self, path: str) -> list:
        return [] if self.client is None else self.client.writes(path)


@pytest.fixture
def twin(parity_twin_device, scan_proxy):
    scan_proxy.imsize, scan_proxy.dwell_time, scan_proxy.scan_region = 32, 1e-6, [0.0, 0.0, 1.0, 1.0]
    backend = Backend("twin", parity_twin_device)
    backend.call("set_fov", 2e-7)
    backend.call("set_defocus", 0.0)
    backend.call("move_stage", [0.0, 0.0, 0.0, 0.0, 0.0])
    return backend


@pytest.fixture
def autoscript(tmp_path):
    client = FakeAutoScriptClient()
    return Backend("autoscript", autoscript_device(client, tmp_path), client)


@pytest.fixture(params=["twin", "autoscript"])
def backend(request):
    return request.getfixturevalue(request.param)


# ----------------------------------------------------------------------
# Accepted arguments
# ----------------------------------------------------------------------
def test_in_range_values_are_accepted_and_read_back(backend):
    backend.call("set_fov", 3e-7)
    assert backend.call("get_fov") == pytest.approx(3e-7)
    backend.call("set_defocus", -5e-8)
    assert backend.call("get_defocus") == pytest.approx(-5e-8)
    backend.call("place_beam", [0.25, 0.75])
    backend.call("move_stage", [2e-7, -3e-7, 1e-8, 5.0, 0.0])
    stage = [float(v) for v in backend.call("get_stage")]
    assert len(stage) == 5
    assert stage == pytest.approx([2e-7, -3e-7, 1e-8, 5.0, 0.0], abs=1e-12)  # x before y, tilts in degrees


# ----------------------------------------------------------------------
# Refused values: same on both, and on AutoScript refused before the instrument sees them
# ----------------------------------------------------------------------
NAN = float("nan")
OUT_OF_RANGE = [
    ("set_fov", 0.0, "optics.scan_field_of_view"),
    ("set_fov", -2e-7, "optics.scan_field_of_view"),
    ("set_fov", 1.0, "optics.scan_field_of_view"),  # one metre
    ("set_fov", NAN, "optics.scan_field_of_view"),
    ("set_defocus", 50.0, "optics.focusing"),  # 50 nm given as 50
    ("set_defocus", 2e-5, "optics.focusing"),  # above the 10 um software limit, inside the instrument's
    ("set_defocus", NAN, "optics.focusing"),
    ("move_stage", [2e-3, 0.0, 0.0, 0.0, 0.0], "specimen.stage.absolute_move"),
    ("move_stage", [0.0, 0.0, 0.0, 50.0, 0.0], "specimen.stage.absolute_move"),
    ("move_stage", [0.0, 0.0], "specimen.stage.absolute_move"),
    ("move_stage", [NAN, 0.0, 0.0, 0.0, 0.0], "specimen.stage.absolute_move"),
    ("place_beam", [1.5, 0.5], "optics.paused_scan_beam_position"),
]


@pytest.mark.parametrize("command, value, path", OUT_OF_RANGE, ids=[f"{c}={v}" for c, v, _ in OUT_OF_RANGE])
def test_out_of_range_values_are_refused_before_the_instrument(backend, command, value, path):
    def state():
        return [float(backend.call("get_fov")), float(backend.call("get_defocus")), *(float(v) for v in backend.call("get_stage"))]

    before = state()
    with pytest.raises(REFUSED):
        backend.call(command, value)
    assert backend.instrument_writes(path) == [], f"{backend.name} sent a refused {command}({value}) to the instrument"
    assert state() == pytest.approx(before, nan_ok=True)


BAD_SCANS = [
    (32, 1e-6, [0.5, 0.5, 0.8, 0.8]),  # runs off the field
    (32, 1e-6, [0.0, 0.0, 0.0, 1.0]),  # zero width
    (32, 1e-6, [-0.1, 0.0, 0.5, 0.5]),
    (32, 1e-6, [0.0, 0.0, 1.0]),  # three values
    (0, 1e-6, [0.0, 0.0, 1.0, 1.0]),
    (32, 0.0, [0.0, 0.0, 1.0, 1.0]),
]


@pytest.mark.parametrize("imsize, dwell, region", BAD_SCANS)
def test_bad_scan_settings_are_refused_before_the_instrument(backend, imsize, dwell, region, monkeypatch):
    bad = SimpleNamespace(imsize=imsize, dwell_time=dwell, scan_region=region, output_format=".h5")
    monkeypatch.setitem(backend.device._detector_proxies, "scan", bad)
    with pytest.raises(REFUSED):
        backend.call("acquire_scanned_image", ["haadf"])
    assert backend.instrument_writes("acquisition") == []


# ----------------------------------------------------------------------
# Return shapes and keys
# ----------------------------------------------------------------------
def test_get_stage_reports_the_instrument_without_a_stage_device(backend):
    """get_stage answers from the instrument itself, so it works (and is right) with no STAGE settings device."""
    backend.call("move_stage", [4e-7, -1e-7, 0.0, 0.0, 0.0])
    proxies = backend.device._detector_proxies
    stage_device = proxies.pop("stage")
    try:
        stage = [float(v) for v in backend.call("get_stage")]
    finally:
        proxies["stage"] = stage_device
    assert stage == pytest.approx([4e-7, -1e-7, 0.0, 0.0, 0.0], abs=1e-12)


SHARED_PARAMETERS = {
    "defocus_m": float,
    "fov_m": float,
    "stage_position": list,
    "beam_blanked": bool,
    "beam_current_pa": float,
    "beam_current_source": str,
}


def test_get_parameters_shares_keys_types_and_values(backend):
    backend.call("set_fov", 2.5e-7)
    backend.call("set_defocus", 1e-8)
    parameters = json.loads(backend.call("get_parameters"))
    missing = sorted(set(SHARED_PARAMETERS) - set(parameters))
    assert not missing, f"{backend.name} get_parameters lacks {missing}"
    for key, kind in SHARED_PARAMETERS.items():
        assert isinstance(parameters[key], kind) or (kind is float and isinstance(parameters[key], int)), (key, parameters[key])
    assert parameters["fov_m"] == pytest.approx(2.5e-7)
    assert parameters["defocus_m"] == pytest.approx(1e-8)
    assert len(parameters["stage_position"]) == 5
    assert parameters["beam_current_pa"] > 0
    assert parameters["beam_current_source"] in ("measured", "default")
    if parameters["beam_current_source"] == "default":
        assert parameters["beam_current_reason"]


def test_acquisitions_return_the_same_shapes(backend):
    image_key = backend.device._acquire_scanned_image(32, 1e-6, ["haadf"], [0.0, 0.0, 1.0, 1.0])
    with h5py.File(image_key, "r") as h5:
        assert h5["image/HAADF"].shape == (32, 32)
    spectrum_key = backend.call("acquire_spectrum", "eds")
    with h5py.File(spectrum_key, "r") as h5:
        assert h5["spectrum"].ndim == 1


# ----------------------------------------------------------------------
# What only one backend can do: written down and asserted, never skipped
# ----------------------------------------------------------------------
# Commands the twin does not implement (they raise NotImplementedError there). Each is checked on
# AutoScript below. Implementing one in the twin fails test_twin_gaps_are_as_documented: move it into
# the shared checks above.
TWIN_LACKS = {
    "set_beam_tilt": ([0.1, -0.1],),
    "get_beam_tilt": (),
    "set_diffraction_shift": ([1e-7, 0.0],),
    "get_diffraction_shift": (),
    "set_screen": ("inserted",),
    "get_screen_current": (),
}
# Twin-only, by design: export_ground_truth (hidden truth for scoring), the stage_moving attribute
# (the twin's long moves return early; AutoScript's absolute_move returns on arrival), simulated_time_s,
# and set_stigmator/get_stigmator (the objective stigmator is not wired to AutoScript yet; the Tuner
# offers image-metric stigmation only where the microscope has them).


def test_image_shift_round_trips_and_refuses_out_of_range(backend):
    backend.call("set_image_shift", [1e-9, 2e-9])
    assert [float(v) for v in backend.call("get_image_shift")] == pytest.approx([1e-9, 2e-9])
    with pytest.raises(REFUSED):
        backend.call("set_image_shift", [1.0, 0.0])
    backend.call("set_image_shift", [0.0, 0.0])


def test_aim_image_shift_centres_the_image_on_a_stage_position_without_moving_the_stage(backend):
    stage = [float(v) for v in backend.call("get_stage")]
    shift = backend.call("aim_image_shift", [stage[0] + 2e-9, stage[1] - 3e-9])
    assert [float(v) for v in shift] == pytest.approx([2e-9, -3e-9], abs=1e-15)
    assert [float(v) for v in backend.call("get_image_shift")] == pytest.approx([2e-9, -3e-9], abs=1e-15)
    assert [float(v) for v in backend.call("get_stage")][:2] == pytest.approx(stage[:2], abs=1e-15)
    with pytest.raises(REFUSED):
        backend.call("aim_image_shift", [stage[0] + 1.0, stage[1]])
    backend.call("set_image_shift", [0.0, 0.0])


def test_place_beam_at_parks_the_beam_on_a_stage_position_inside_the_field_of_view(backend):
    stage = [float(v) for v in backend.call("get_stage")]
    backend.call("set_fov", 1e-7)
    backend.call("set_image_shift", [1e-8, 0.0])
    fraction = backend.call("place_beam_at", [stage[0] + 1e-8 + 2.5e-8, stage[1] - 2.5e-8])
    assert [float(v) for v in fraction] == pytest.approx([0.75, 0.25], abs=1e-9)
    with pytest.raises(REFUSED):
        backend.call("place_beam_at", [stage[0] + 1e-6, stage[1]])
    with pytest.raises(REFUSED, match="call place_beam with them"):
        backend.call("place_beam_at", [0.5, 0.5])
    backend.call("set_image_shift", [0.0, 0.0])


def test_auto_focus_runs_the_vendor_routine_or_its_stand_in(backend):
    backend.call("set_defocus", 4e-8)
    backend.call("auto_focus")
    if backend.client is not None:
        assert [c["path"] for c in backend.client.writes("auto_functions")] == ["auto_functions.run_opti_stem"]
    else:  # the twin's OptiSTEM stand-in sets C1 to its hidden truth (zero here) within its residual
        assert abs(float(backend.call("get_defocus"))) < 1e-8


@pytest.mark.parametrize("command", sorted(TWIN_LACKS))
def test_twin_gaps_are_as_documented(twin, command):
    with pytest.raises(NotImplementedError):
        twin.call(command, *TWIN_LACKS[command])


def test_autoscript_autofocus_runs_the_vendor_routine_for_the_mode(autoscript):
    client = autoscript.client
    autoscript.call("auto_focus")
    assert [c["path"] for c in client.writes("auto_functions")] == ["auto_functions.run_opti_stem"]
    client.optics.optical_mode = "Tem"
    autoscript.call("auto_focus")
    assert client.writes("auto_functions")[-1]["path"] == "auto_functions.run_objective_auto_stigmator"


def test_autoscript_shift_and_tilt_round_trip(autoscript):
    autoscript.call("set_image_shift", [1e-9, 2e-9])  # STEM: the beam shift
    assert list(autoscript.call("get_image_shift")) == pytest.approx([1e-9, 2e-9])
    autoscript.call("set_beam_tilt", [0.1, -0.1])
    assert list(autoscript.call("get_beam_tilt")) == pytest.approx([0.1, -0.1])
    # Diffraction shift is written to deflectors.image_shift (metres) in diffraction; AutoScript documents
    # deflectors.image_tilt (radians) as "equivalent to diffraction shift": to confirm at first light.
    autoscript.client.optics.projector_mode = "Diffraction"
    autoscript.call("set_diffraction_shift", [1e-7, 0.0])
    assert list(autoscript.call("get_diffraction_shift")) == pytest.approx([1e-7, 0.0])


@pytest.mark.parametrize("command, value", [("set_image_shift", [1.0, 0.0]), ("set_beam_tilt", [2.0, 0.0])])
def test_autoscript_refused_shift_or_tilt_raises(autoscript, command, value):
    """A write the instrument refuses must reach the caller as an error, not be logged and dropped."""
    with pytest.raises(REFUSED):
        autoscript.call(command, value)


def test_autoscript_diffraction_shift_outside_diffraction_is_refused(autoscript):
    with pytest.raises(REFUSED):
        autoscript.call("set_diffraction_shift", [1e-7, 0.0])
    assert autoscript.client.writes("optics.deflectors") == []


def test_autoscript_register_stage_reads_the_instrument_and_moves_nothing(autoscript):
    client = autoscript.client
    # The operator moved the stage at the console: only the instrument knows.
    object.__setattr__(client.specimen.stage, "_position", StagePosition(3e-7, -4e-7, 1e-7, math.radians(2.0), 0.0))
    position = autoscript.device.register_stage()
    assert [float(v) for v in position] == pytest.approx([3e-7, -4e-7, 1e-7, 2.0, 0.0], abs=1e-12)
    assert client.writes("specimen.stage.absolute_move") == []


def test_autoscript_defocus_attribute_is_documented_in_metres():
    attr = AutoScriptMicroscope.defocus
    assert getattr(attr, "unit", "m") == "m"
    assert "micrometer" not in (attr.__doc__ or "").lower()
