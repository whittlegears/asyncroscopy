"""What the ElectronMicroscope base owns for every backend: the stigmator, stage_moving, the beam current and the command lock."""

from __future__ import annotations

import json
import math
import types
import threading
import time

import pytest
import tango

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope as EM
from asyncroscopy.instruments.electron_microscope.jeol import JeolMicroscope
from tests.fake_autoscript import FakeAutoScriptClient
from tests.instruments.test_parity import autoscript_device


@pytest.fixture
def autoscript(tmp_path):
    client = FakeAutoScriptClient()
    return autoscript_device(client, tmp_path), client


def jeol_device():
    device = JeolMicroscope.__new__(JeolMicroscope)
    device._tango_properties = {"max_stigmator": 1e-6}
    device._command_lock = threading.RLock()
    return device


@pytest.mark.parametrize("call", [lambda d: EM.get_stigmator(d), lambda d: EM.set_stigmator(d, [0.0, 0.0]), lambda d: EM.read_stage_moving(d)])
def test_a_backend_without_a_stigmator_or_stage_moving_says_so(call):
    device = jeol_device()
    with pytest.raises(tango.DevFailed) as refused:
        call(device)
    assert refused.value.args[0].reason == "UnsupportedCommand" and "JeolMicroscope" in refused.value.args[0].desc


@pytest.mark.parametrize("call, what", [
    (lambda d: EM.blank_beam(d), "blank_beam"), (lambda d: EM.unblank_beam(d), "unblank_beam"),
    (lambda d: EM.place_beam(d, [0.5, 0.5]), "place_beam"), (lambda d: EM._set_defocus(d, 1e-9), "set_defocus"),
    (lambda d: EM._set_column_valves(d, "open"), "set_column_valves"),
])
def test_a_backend_without_a_blanker_or_defocus_refuses_instead_of_doing_nothing(call, what):
    """JEOL (and the atomistic twins) implement none of these; the base must never report success for them."""
    device = jeol_device()
    device._check_connected = lambda name: None
    with pytest.raises(tango.DevFailed) as refused:
        call(device)
    assert refused.value.args[0].reason == "UnsupportedCommand" and what in refused.value.args[0].desc


def test_autoscript_stigmator_is_refused_until_calibrated_then_set_in_its_own_units(autoscript):
    device, client = autoscript
    device.max_stigmator = 1e-6
    device.objective_stigmator_m_per_unit = 0.0
    with pytest.raises(tango.DevFailed, match="objective_stigmator_m_per_unit"):
        EM.get_stigmator(device)
    device.objective_stigmator_m_per_unit = 1e-6
    EM.set_stigmator(device, [2e-7, -1e-7])
    assert [w["value"] for w in client.writes("optics.stigmators.objective_stigmator")] == [pytest.approx([0.2, -0.1])]
    assert EM.get_stigmator(device) == pytest.approx([2e-7, -1e-7])
    with pytest.raises(ValueError, match="stigmator must be within"):
        EM.set_stigmator(device, [2e-6, 0.0])  # refused by the base, before the instrument
    assert len(client.writes("optics.stigmators.objective_stigmator")) == 1


def test_autoscript_reports_an_assumed_current_and_whether_the_stage_moves(autoscript):
    device, _client = autoscript
    assert EM.read_beam_current_source(device) == "default"
    assert EM.read_beam_current_pa(device) == pytest.approx(50.0)
    assert math.isnan(EM.read_beam_current_age_s(device))
    assert EM.read_stage_moving(device) is False


def test_blank_beam_never_waits_for_a_command_holding_the_lock(autoscript):
    device, client = autoscript
    held = threading.Event()
    release = threading.Event()

    def long_command():
        with device._command_lock:
            held.set()
            release.wait(5.0)

    worker = threading.Thread(target=long_command)
    worker.start()
    held.wait(5.0)
    try:
        t0 = time.monotonic()
        EM.blank_beam(device)
        assert time.monotonic() - t0 < 1.0
        assert client.writes("optics.blanker.blank")
        waited = threading.Thread(target=EM.get_fov, args=(device,))
        waited.start()
        waited.join(0.3)
        assert waited.is_alive()  # every other command queues behind the lock
    finally:
        release.set()
        worker.join()
    waited.join(5.0)
    assert not waited.is_alive()


@pytest.mark.parametrize("name", ["scan", "haadf"])
def test_only_a_spectrum_detector_takes_a_spectrum_and_the_refusal_names_the_right_tool(auto_script_proxy, name):
    """A sub-device such as the scan controller is not a spectrum detector, and an imaging detector belongs to another tool."""
    with pytest.raises(tango.DevFailed) as refused:
        auto_script_proxy.acquire_spectrum(name)
    message = refused.value.args[0].desc
    assert f"No spectrum detector named '{name}'" in message and "acquire_scanned_image" in message
    assert "'scan'" not in message.split("Spectrum detectors here:")[1]


class _FakeDetector:
    """A detector device: its action_hints say which command takes a spectrum by itself."""

    exposure_time = 0.5

    def __init__(self, hints):
        self.hints = hints
        self.calls = []

    def read_attribute(self, name):
        assert name == "action_hints"
        return types.SimpleNamespace(value=json.dumps(self.hints))

    def command_query(self, name):
        return types.SimpleNamespace(in_type=tango.CmdArgType.DevVoid)

    def command_inout(self, name, *args):
        self.calls.append(name)
        return f"{name}-key"


def test_a_detector_device_that_acquires_its_own_spectrum_is_used_whatever_it_is_called(autoscript):
    """No detector name is built in: a device whose action_hints declare an acquire.spectrum command is a spectrum detector."""
    device, client = autoscript
    gatan = _FakeDetector({"acquire_eels_spectrum": {"role": "acquire.spectrum"}, "get_eels_dispersion": {"role": "setting.read"}})
    device._detector_proxies["gif"] = gatan
    client.optics.blanker.blank()
    assert device.acquire_spectrum("GIF") == "acquire_eels_spectrum-key"
    assert gatan.calls == ["acquire_eels_spectrum"]
    assert client.optics.blanker.is_beam_blanked
    assert "gif" in device._spectrum_detector_names()


def test_a_settings_only_device_is_not_a_spectrum_detector(autoscript):
    device, _ = autoscript
    device._detector_proxies["camera"] = _FakeDetector({"get_exposure": {"role": "setting.read"}})
    with pytest.raises(tango.DevFailed) as refused:
        device.acquire_spectrum("camera")
    assert "No spectrum detector named 'camera'" in refused.value.args[0].desc


def test_a_scanned_image_is_refused_in_tem_mode(autoscript):
    device, client = autoscript
    client.optics.optical_mode = "Tem"
    with pytest.raises(ValueError, match="STEM mode only"):
        device._acquire_scanned_image(64, 1e-6, ["haadf"])


def test_every_device_address_a_config_sets_is_connected_without_a_property_for_it(autoscript, monkeypatch):
    """run_servers sets <key>_device_address for every support device; the instrument connects each, declared or not."""
    device, _ = autoscript

    class FakeDb:
        def get_device_property_list(self, name, wildcard):
            return types.SimpleNamespace(value_string=["gif_device_address", "eds_device_address", "instrument_device_address", "empty_device_address"])

        def get_device_property(self, name, names):
            return {"gif_device_address": ["asyncroscopy/eels/default"], "eds_device_address": ["asyncroscopy/eds/default"],
                    "instrument_device_address": ["asyncroscopy/instrument/default"], "empty_device_address": []}

    monkeypatch.setattr(tango.Util, "instance", staticmethod(lambda *a: types.SimpleNamespace(get_database=FakeDb)))
    monkeypatch.setattr(device, "get_name", lambda: "asyncroscopy/autoscriptmicroscope/default", raising=False)
    assert device._configured_addresses() == {"gif": "asyncroscopy/eels/default", "eds": "asyncroscopy/eds/default"}
