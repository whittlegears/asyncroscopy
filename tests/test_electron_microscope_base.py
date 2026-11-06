"""What the ElectronMicroscope base owns for every backend: the stigmator, stage_moving, the beam current and the command lock."""

from __future__ import annotations

import math
import threading
import time

import pytest
import tango

from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope as EM
from asyncroscopy.instruments.electron_microscope.jeol import JeolMicroscope
from tests.fake_autoscript import FakeAutoScriptClient
from tests.test_parity import autoscript_device


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
