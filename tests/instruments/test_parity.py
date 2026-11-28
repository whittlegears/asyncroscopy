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

from asyncroscopy import settings
from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope as EM
from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE, TILT_LIMITS_DEG
from asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript import AutoScriptSTAGE
from tests.fake_autoscript import FakeAutoScriptClient

REFUSED = (ValueError, TypeError, tango.DevFailed, ApplicationServerException)
SOFTWARE_LIMITS = {"min_fov": 1e-9, "max_fov": 1e-4, "max_defocus": 1e-5, "max_image_shift": 1e-6}  # the twin's defaults, which AutoScript must match


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
    _refuse_if_held = STAGE._refuse_if_held
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
    for name, value in {**SOFTWARE_LIMITS, "max_diffraction_shift": AutoScriptMicroscope.max_diffraction_shift.default_value,
                        "default_beam_current_pa": 50.0, "default_scan_detector": "haadf"}.items():
        if hasattr(AutoScriptMicroscope, name):
            setattr(device, name, value)
    device._microscope = client
    device._connect_error = None
    device._command_lock, device._blank_client, device._blank_count, device._connected_versions = threading.RLock(), None, 0, None
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
    ("place_beam_list", [0.25, 0.25, 1.5, 0.5], "optics.paused_scan_beam_position"),
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


TWIN_LACKS = {
    "set_beam_tilt": ([0.1, -0.1],),
    "get_beam_tilt": (),
    "set_diffraction_shift": ([1e-3, 0.0],),
    "get_diffraction_shift": (),
}


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
    else:  # the twin's OptiSTEM stand-in sets C1 near its hidden truth, zero here
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
    autoscript.call("set_image_shift", [1e-9, 2e-9])
    assert list(autoscript.call("get_image_shift")) == pytest.approx([1e-9, 2e-9])
    assert [c["path"] for c in autoscript.client.writes("optics.deflectors")] == ["optics.deflectors.image_shift"]
    autoscript.device.stem_shift_deflector = "beam_shift"
    autoscript.call("set_image_shift", [3e-9, 0.0])
    assert autoscript.client.writes("optics.deflectors")[-1]["path"] == "optics.deflectors.beam_shift"
    assert list(autoscript.call("get_image_shift")) == pytest.approx([3e-9, 0.0])
    autoscript.device.stem_shift_deflector = "sideways"
    with pytest.raises(REFUSED):
        autoscript.call("set_image_shift", [0.0, 0.0])
    autoscript.device.stem_shift_deflector = "image_shift"
    autoscript.call("set_beam_tilt", [0.1, -0.1])
    assert list(autoscript.call("get_beam_tilt")) == pytest.approx([0.1, -0.1])
    autoscript.client.optics.projector_mode = "Diffraction"
    autoscript.call("set_diffraction_shift", [1e-3, 0.0])
    assert list(autoscript.call("get_diffraction_shift")) == pytest.approx([1e-3, 0.0])
    assert [c["path"] for c in autoscript.client.writes("optics.deflectors")][-1] == "optics.deflectors.image_tilt"
    assert autoscript.client.optics.deflectors.image_shift.x == pytest.approx(1e-9)  # the image shift set above is untouched


@pytest.mark.parametrize("command, value", [("set_image_shift", [1.0, 0.0]), ("set_beam_tilt", [2.0, 0.0])])
def test_autoscript_refused_shift_or_tilt_raises(autoscript, command, value):
    """A write the instrument refuses must reach the caller as an error, not be logged and dropped."""
    with pytest.raises(REFUSED):
        autoscript.call(command, value)


def test_autoscript_diffraction_shift_beyond_max_diffraction_shift_is_refused_before_the_instrument(autoscript):
    autoscript.client.optics.projector_mode = "Diffraction"
    autoscript.device.max_diffraction_shift = 5e-3
    for shift in ([6e-3, 0.0], [0.0, float("nan")]):
        with pytest.raises(REFUSED, match="max_diffraction_shift"):
            autoscript.call("set_diffraction_shift", shift)
    assert autoscript.client.writes("optics.deflectors") == []
    hint = autoscript.device.argument_hints()["set_diffraction_shift"]["args"][0]
    assert (hint["min"], hint["max"]) == (-5e-3, 5e-3)


def test_autoscript_diffraction_shift_outside_diffraction_is_refused(autoscript):
    with pytest.raises(REFUSED):
        autoscript.call("set_diffraction_shift", [1e-7, 0.0])
    with pytest.raises(REFUSED, match="diffraction mode only"):
        autoscript.call("get_diffraction_shift")  # never a made-up [0, 0]
    assert autoscript.client.writes("optics.deflectors") == []


def test_autoscript_beam_tilt_is_unitless_and_says_so(autoscript):
    with pytest.raises(REFUSED, match="unitless between -1 and 1"):
        autoscript.call("set_beam_tilt", [1.5, 0.0])
    assert autoscript.client.writes("optics.deflectors") == []  # refused before the instrument
    hints = autoscript.device.argument_hints()
    assert hints["set_beam_tilt"]["args"][0]["units"] == "unitless" and hints["set_beam_tilt"]["args"][0]["min"] == -1.0
    assert hints["get_beam_tilt"]["returns"]["units"] == "unitless" and hints["get_diffraction_shift"]["returns"]["units"] == "rad"


@pytest.mark.parametrize("command, word", [("set_column_valves", "ajar"), ("set_screen", "halfway")])
def test_autoscript_refuses_an_unknown_valve_or_screen_word(autoscript, command, word):
    with pytest.raises(REFUSED, match="nothing was changed"):
        autoscript.call(command, word)
    assert [c for c in autoscript.client.writes() if c["path"].startswith(("vacuum", "detectors.screen"))] == []


def test_autoscript_has_no_register_stage():
    """register_stage published the stage to a settings-only STAGE device; the AutoScript STAGE device now reads the
    hardware itself, so it only duplicated get_stage."""
    assert not hasattr(AutoScriptMicroscope, "register_stage")


@pytest.mark.parametrize("attribute, command", [("fov", "get_fov"), ("defocus", "get_defocus")])
def test_read_only_attributes_read_through_their_command_reader(backend, attribute, command):
    backend.call("set_fov", 2.5e-7)
    backend.call("set_defocus", -3e-8)
    assert getattr(EM, f"read_{attribute}")(backend.device) == pytest.approx(backend.call(command))


def test_beam_state_reads_the_blanker_get_parameters_reports(backend):
    backend.call("blank_beam")
    assert EM.read_beam_state(backend.device) is True and json.loads(backend.call("get_parameters"))["beam_blanked"] is True
    backend.call("unblank_beam")
    assert EM.read_beam_state(backend.device) is False and json.loads(backend.call("get_parameters"))["beam_blanked"] is False


def test_manufacturer_reads_as_a_string(backend):
    manufacturer = EM.read_manufacturer(backend.device)
    assert isinstance(manufacturer, str) and manufacturer == json.loads(backend.call("get_parameters")).get("manufacturer", manufacturer)


def test_autoscript_manufacturer_is_thermo_fisher_over_tango(auto_script_proxy):
    assert auto_script_proxy.manufacturer == "Thermo Fisher Scientific"


@pytest.mark.parametrize("name, value", [("fov", 1e-7), ("defocus", 1e-8), ("beam_state", True), ("manufacturer", "x")])
def test_autoscript_read_only_attributes_refuse_writes(auto_script_proxy, name, value):
    with pytest.raises(tango.DevFailed):
        setattr(auto_script_proxy, name, value)


def test_autoscript_get_parameters_reports_optics_versions_and_auto_blank_from_the_instrument(autoscript):
    parameters = json.loads(autoscript.call("get_parameters"))
    assert parameters["camera_length_m"] == {"calibrated": pytest.approx(0.115 * 1.03), "nominal": pytest.approx(0.115), "label": "115 mm"}
    assert parameters["camera_lengths"] == ["60 mm", "91 mm", "115 mm", "185 mm"]
    assert parameters["magnification"]["label"] == "100 kx" and parameters["convergence_angle_rad"] == pytest.approx(0.025)
    assert parameters["beam_diameter_m"] == pytest.approx(1e-10) and parameters["auto_beam_blank_mode"] == "Off"
    assert parameters["versions"] == {"autoscript_server_version": "1.18.0 (fake)", "autoscript_client_version": "1.18.0",
                                      "autoscript_server_offline": True, "system_serial_number": "FAKE-0000", "system_version": "fake TEM server"}
    assert parameters["is_accelerator_on"] is True and parameters["acceleration_voltage"] == pytest.approx(300e3)
    paths = {c["path"] for c in autoscript.client.calls}
    assert "source.acceleration_voltage.value" in paths and "source.is_accelerator_on" in paths  # not the deprecated optics.* members
    assert [c for c in autoscript.client.calls if c["op"] == "set"] == []  # get_parameters only reads


def test_autoscript_says_which_readings_it_cannot_give(autoscript):
    def unavailable():
        raise RuntimeError("only on systems with three condenser lenses")

    optics = autoscript.client.optics
    type(optics).convergence_angle = property(lambda self: unavailable())
    try:
        parameters = json.loads(autoscript.call("get_parameters"))
    finally:
        del type(optics).convergence_angle
    assert "three condenser lenses" in parameters["convergence_angle_rad"]["unavailable"]


def test_autoscript_lists_its_detectors_and_refuses_one_it_does_not_have(autoscript):
    detectors = json.loads(autoscript.call("get_detectors"))
    assert detectors["scanning"] == ["BF", "DF2", "DF4", "HAADF", "BF-S", "DF-S"] and detectors["camera"] == ["BM-Ceta", "Flucam"]
    assert detectors["eds"] == ["SuperX"] and detectors["eels"] == ["Zebra"]
    assert detectors["cameras"]["Flucam"]["exposure_time_limits_s"] == [1e-3, 60.0]
    assert "Flucam only" in detectors["cameras"]["BM-Ceta"]["exposure_time_limits_s"]["unavailable"]
    assert {"size": 4096, "fixed_readout_area": "Full"} in detectors["cameras"]["BM-Ceta"]["image_sizes"]
    with pytest.raises(REFUSED, match="no scanning detector"):
        autoscript.device._acquire_scanned_image(32, 1e-6, ["adf"], [0.0, 0.0, 1.0, 1.0])  # ADF is a twin name
    assert autoscript.client.writes("acquisition") == []


def test_the_twin_lists_the_detectors_it_renders(twin):
    detectors = json.loads(twin.call("get_detectors"))
    parameters = json.loads(twin.call("get_parameters"))
    assert detectors["scanning"] == parameters["scan_detectors"] and detectors["camera"] == parameters["camera_detectors"]
    assert "BM-Ceta" in detectors["camera"]  # CAMERA names that all give the twin's one bright-field camera


NEW_OPERATOR_ONLY = ("run_stem_auto_focus", "run_beam_tilt_auto_focus", "acquire_camera_series", "acquire_eels_spectra",
                     "acquire_eels_spectrum_image", "set_projector_mode", "set_camera_length", "set_beam_stopper", "set_scanning_detector")


def test_the_new_actuating_commands_are_operator_only_and_the_twin_refuses_them(twin):
    from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE
    from asyncroscopy.utils.commands import declared_hints

    hints = declared_hints(AutoScriptMicroscope)
    assert all(hints[name]["operator_only"] is True and hints[name]["risk"] == "high" for name in NEW_OPERATOR_ONLY)
    assert declared_hints(STAGE)["move_relative"]["operator_only"] is True
    stop = declared_hints(AutoScriptSTAGE)["stop_jogging"]
    assert stop["risk"] == "low" and stop["stop"] is True and "operator_only" not in stop  # a stop stays with the agent and the e-stop
    for name, argument in (("run_stem_auto_focus", "{}"), ("set_projector_mode", "Diffraction"), ("set_beam_stopper", "inserted")):
        with pytest.raises(tango.DevFailed, match="does not support"):
            twin.call(name, argument)


def test_autoscript_auto_focus_returns_optistem_quality(autoscript):
    assert json.loads(autoscript.call("auto_focus")) == {"routine": "run_opti_stem C1A1", "quality": "Improved"}


def test_autoscript_stem_and_beam_tilt_autofocus_return_what_they_measured(autoscript):
    assert json.loads(autoscript.call("run_stem_auto_focus", '{"accuracy_level": "Fine"}')) == {"defocus_m": -4e-9, "astigmatism_m": 2e-9,
                                                                                               "beam_blanked": False}
    assert autoscript.client.writes("auto_functions.run_stem_auto_focus")[-1]["value"]["accuracy_level"] == "Fine"
    with pytest.raises(REFUSED, match="takes"):
        autoscript.call("run_stem_auto_focus", '{"reset_beam_blanker_afterwards": true}')  # not a field it may set
    with pytest.raises(REFUSED, match="TEM mode only"):
        autoscript.call("run_beam_tilt_auto_focus", "{}")
    autoscript.client.optics.optical_mode = "Tem"
    autoscript.client.optics.blanker.blank()
    assert json.loads(autoscript.call("run_beam_tilt_auto_focus", '{"camera_detector": "Flucam"}')) == {
        "defocus_m": 5e-9, "astigmatism_m": 3e-9, "drift_m": [1e-10, -2e-10], "beam_blanked": True}
    with pytest.raises(REFUSED, match="STEM mode only"):
        autoscript.call("run_stem_auto_focus", "{}")
    assert len(autoscript.client.writes("auto_functions.run_stem_auto_focus")) == 1


def test_autoscript_camera_series_uses_the_camera_settings_and_is_bounded(autoscript):
    autoscript.device._detector_proxies["camera"] = SimpleNamespace(imsize=64, exposure_time=0.01, camera_detector="BM-Ceta", readout_area="Half",
                                                                   frame_combining=1)
    autoscript.device.max_camera_series_frames = 10
    autoscript.device.max_camera_series_bytes = 1 << 30
    autoscript.device.camera_bytes_per_pixel = 0
    with pytest.raises(REFUSED, match="camera_bytes_per_pixel is not set"):
        autoscript.call("acquire_camera_series", 3)
    autoscript.device.camera_bytes_per_pixel = 2
    autoscript.device.max_camera_series_bytes = 3 * 64 * 64 * 2 - 1
    with pytest.raises(REFUSED, match="above max_camera_series_bytes"):
        autoscript.call("acquire_camera_series", 3)
    assert autoscript.client.writes("acquisition.acquire_camera_series_advanced") == []
    autoscript.device.max_camera_series_bytes = 3 * 64 * 64 * 2
    key = autoscript.call("acquire_camera_series", 3)
    with h5py.File(key, "r") as h5:
        assert h5["camera_series"].shape == (3, 64, 64) and h5["camera_series"].attrs["length"] == 3
    assert autoscript.client.writes("acquisition.acquire_camera_series_advanced")[-1]["value"]["length"] == 3
    for length in (0, 11):
        with pytest.raises(REFUSED, match="nothing was acquired"):
            autoscript.call("acquire_camera_series", length)
    assert len(autoscript.client.writes("acquisition.acquire_camera_series_advanced")) == 1


def test_autoscript_eels_needs_a_detector_the_microscope_has(autoscript):
    autoscript.device.max_spectrum_image_points = 64
    key = autoscript.call("acquire_eels_spectra", '{"detector": "Zebra", "exposure_time_s": 0.1}')
    with h5py.File(key, "r") as h5:
        assert h5["spectrum"].ndim == 1
    assert autoscript.call("acquire_eels_spectrum_image", '{"detector": "Zebra", "exposure_time_s": 0.01, "size": 8}').endswith(".emd")
    for request in ('{"detector": "GIF", "exposure_time_s": 0.1}', '{"detector": "Zebra"}', '{"detector": "Zebra", "exposure_time_s": 0.01, "size": 1}'):
        with pytest.raises(REFUSED, match="nothing was acquired"):
            autoscript.call("acquire_eels_spectrum_image" if "size" in request else "acquire_eels_spectra", request)
    assert len(autoscript.client.writes("analysis.eels")) == 2


def test_autoscript_imaging_set_up_takes_only_what_the_microscope_offers(autoscript):
    autoscript.call("set_projector_mode", "diffraction")
    assert autoscript.client.optics.projector_mode == "Diffraction"
    autoscript.call("set_camera_length", "185 mm")
    assert autoscript.client.optics.camera_length.value.label == "185 mm"
    autoscript.call("set_beam_stopper", "half_inserted")
    assert autoscript.client.optics.beam_stopper.insertion_state == "HalfInserted"
    autoscript.call("set_scanning_detector", ["DF4", "retracted"])
    assert autoscript.client.writes("detectors.scanning[DF4]")[-1]["path"] == "detectors.scanning[DF4].retract"
    before = len(autoscript.client.writes())
    for command, argument in (("set_projector_mode", "Spectroscopy"), ("set_camera_length", "2 m"), ("set_beam_stopper", "ajar"),
                              ("set_scanning_detector", ["ADF", "inserted"]), ("set_scanning_detector", ["DF4", "out"])):
        with pytest.raises(REFUSED, match="nothing was changed"):
            autoscript.call(command, argument)
    assert len(autoscript.client.writes()) == before


def test_autoscript_stage_stop_jogging_reaches_the_client():
    client = FakeAutoScriptClient()
    device = AutoScriptSTAGE.__new__(AutoScriptSTAGE)
    device._microscope = client
    AutoScriptSTAGE.stop_jogging(device)
    assert client.writes("specimen.stage.stop_jogging")
