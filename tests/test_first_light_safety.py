"""Fixes from the Spectra 300 first-light review (docs/Operation/first_light_review.md), against the fake AutoScript client.

* A2: the launcher reads each device's State after it answers, and says FAULT loudly; a STAGE or APERTURE
  device without AutoScript on a hardware stack is in FAULT, not ON with cached values.
* A3: check_stack.py runs only on a simulated stack.
* A4: the CEOS timeout is 30 s, and a JSON-RPC error reply raises.
* B1: z and tilt moves go through absolute_move_safe; a high tilt far from eucentric height and a beta on a
  holder without one are refused.
* B2: a beam blanked before an acquisition is blanked again after it, before the save.
* B3: 4D-STEM is checked, bounded, takes the camera's exposure, and stops on a blank_beam.
* B4: blank_beam does not wait for a running command; stop_all says the stage is not stopped.
* B8: the microscope's STAGE proxy gets the stage timeout.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import tango

import asyncroscopy.instruments.electron_microscope.auto_script as auto_script_module
import asyncroscopy.instruments.electron_microscope.hardware.aperture_autoscript as aperture_module
import asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript as stage_module
from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope as EM
from asyncroscopy.instruments.electron_microscope.hardware.aperture_autoscript import AutoScriptAPERTURE
from asyncroscopy.instruments.electron_microscope.hardware.corrector_ceos import CEOSCorrector
from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE
from asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript import AutoScriptSTAGE
from startup_scripts import check_stack, run_servers
from tests.fake_autoscript import FakeAutoScriptClient
from tests.test_parity import FakeStageProxy, autoscript_device


@pytest.fixture
def autoscript(tmp_path):
    client = FakeAutoScriptClient()
    device = autoscript_device(client, tmp_path)
    device.max_4dstem_imsize, device.min_4dstem_exposure_s = 64, 1e-3
    device._detector_proxies["camera"] = SimpleNamespace(exposure_time=0.01, imsize=256, camera_detector="BM-Ceta",
                                                         readout_area="Half", frame_combining=1, output_format=".h5")
    return device, client


def blanker_at_save(monkeypatch) -> list[bool]:
    """Record whether the beam is blanked each time an acquisition is saved."""
    seen: list[bool] = []
    save = auto_script_module.save_acquisition

    def recording_save(device, *args, **kwargs):
        seen.append(device._microscope.optics.blanker._blanked)
        return save(device, *args, **kwargs)

    monkeypatch.setattr(auto_script_module, "save_acquisition", recording_save)
    return seen


# ----------------------------------------------------------------------
# A2: FAULT is said, not hidden
# ----------------------------------------------------------------------
def test_the_launcher_reports_a_device_in_fault_instead_of_ready(monkeypatch, capsys):
    states = {"asyncroscopy/stage/default": (tango.DevState.FAULT, "Not connected to AutoScript: refused"),
              "asyncroscopy/scan/default": (tango.DevState.ON, "ON")}

    class Proxy:
        def __init__(self, name):
            self.name = name

        def state(self):
            return states[self.name][0]

        def status(self):
            return states[self.name][1]

    monkeypatch.setattr(run_servers.tango, "DeviceProxy", Proxy)
    faults: dict[str, str] = {}
    run_servers.report_health("asyncroscopy/scan/default", 0.5, faults)
    run_servers.report_health("asyncroscopy/stage/default", 0.5, faults)
    assert faults == {"asyncroscopy/stage/default": "Not connected to AutoScript: refused"}
    run_servers.print_summary("localhost", 9094, [], {}, faults=faults)
    out = capsys.readouterr().out
    assert "NOT READY" in out and "asyncroscopy/stage/default: Not connected to AutoScript: refused" in out
    assert "All asyncroscopy servers are ready" not in out
    run_servers.print_summary("localhost", 9094, [], {}, faults={})
    assert "All asyncroscopy servers are ready" in capsys.readouterr().out


def _bare_device(simulated: bool) -> SimpleNamespace:
    calls: list = []
    return SimpleNamespace(simulated=simulated, hardware_host="10.0.0.1", hardware_port=9095, calls=calls,
                           set_state=lambda s: calls.append(("state", s)), set_status=lambda s: calls.append(("status", s)),
                           warn_stream=lambda *a: None, error_stream=lambda *a: None, info_stream=lambda *a: None)


@pytest.mark.parametrize("cls, module", [(AutoScriptSTAGE, stage_module), (AutoScriptAPERTURE, aperture_module)])
def test_a_device_without_autoscript_is_in_fault_on_hardware(monkeypatch, cls, module):
    monkeypatch.setattr(module, "_AUTOSCRIPT_AVAILABLE", False)
    hardware = _bare_device(simulated=False)
    cls._connect_autoscript(hardware)
    assert ("state", tango.DevState.FAULT) in hardware.calls
    assert any(k == "status" and "not installed" in v for k, v in hardware.calls)
    simulated = _bare_device(simulated=True)
    cls._connect_autoscript(simulated)
    assert simulated.calls == []  # a simulated stack only warns


def test_a_refused_autoscript_connection_says_why_in_the_status(monkeypatch):
    class Refusing:
        def connect(self, host, port):
            raise ConnectionRefusedError("refused")

    monkeypatch.setattr(stage_module, "_AUTOSCRIPT_AVAILABLE", True)
    monkeypatch.setattr(stage_module, "TemMicroscopeClient", Refusing, raising=False)
    device = _bare_device(simulated=False)
    AutoScriptSTAGE._connect_autoscript(device)
    assert ("state", tango.DevState.FAULT) in device.calls
    assert any(k == "status" and "10.0.0.1:9095" in v and "refused" in v for k, v in device.calls)


# ----------------------------------------------------------------------
# A3: check_stack refuses a hardware stack
# ----------------------------------------------------------------------
class FakeOperator:
    def __init__(self, reply):
        self.reply = reply

    async def call_tool(self, name, args, raise_on_error=True):
        assert name == "Approvals_get_simulated"
        if isinstance(self.reply, Exception):
            raise self.reply
        is_error, value = self.reply
        return SimpleNamespace(is_error=is_error, structured_content={"result": value}, content=[])


@pytest.mark.parametrize("reply, simulated", [
    ((False, True), True),
    ((False, False), False),
    ((True, "unknown tool"), False),
    (ConnectionError("down"), False),
])
def test_check_stack_runs_only_on_a_simulated_stack(reply, simulated):
    ok, why = asyncio.run(check_stack.simulated_stack(FakeOperator(reply)))
    assert ok is simulated and why


# ----------------------------------------------------------------------
# A4: CEOS
# ----------------------------------------------------------------------
def _ceos_reply_once(reply: dict) -> tuple[int, threading.Thread]:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        with conn:
            CEOSCorrector._recv_netstring(conn)
            conn.sendall(CEOSCorrector._encode_netstring(reply))
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server.getsockname()[1], thread


def _ceos(port: int) -> SimpleNamespace:
    return SimpleNamespace(ceos_host="127.0.0.1", ceos_port=port, socket_timeout=5.0, _message_id=1, _last_status="",
                           set_state=lambda s: None, _encode_netstring=CEOSCorrector._encode_netstring,
                           _recv_netstring=CEOSCorrector._recv_netstring, _decode_netstring=CEOSCorrector._decode_netstring,
                           _rpc_error=CEOSCorrector._rpc_error)


def test_the_ceos_timeout_is_seconds_not_a_day():
    assert CEOSCorrector.TangoClassClass.device_property_list["socket_timeout"][2] == 30.0


def test_a_ceos_error_reply_raises_instead_of_reading_as_success():
    port, thread = _ceos_reply_once({"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "value out of range"}})
    device = _ceos(port)
    with pytest.raises(tango.DevFailed, match="value out of range") as refused:
        CEOSCorrector._call(device, "correctAberration", {"name": "C1", "value": [1e-9]})
    assert refused.value.args[0].reason == "CeosError" and "refused correctAberration" in device._last_status
    thread.join(5)
    port, thread = _ceos_reply_once({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
    assert json.loads(CEOSCorrector._call(_ceos(port), "getInfo"))["result"] == {"ok": True}
    thread.join(5)


# ----------------------------------------------------------------------
# B1: stage moves near the pole piece
# ----------------------------------------------------------------------
def test_z_or_tilt_moves_use_the_safe_move_and_xy_moves_the_plain_one():
    client = FakeAutoScriptClient()
    stage = FakeStageProxy(client)
    stage.position = [1e-6, 0.0, 5e-5, 10.0, 0.0]  # z and alpha change
    stage.position = [2e-6, 1e-6, 5e-5, 10.0, 0.0]  # x and y only
    moves = [c["path"].rsplit(".", 1)[1] for c in client.writes("specimen.stage.absolute_move")]
    assert moves == ["absolute_move_safe", "absolute_move"]


@pytest.mark.parametrize("target", [[0.0, 0.0, 3e-4, 20.0, 0.0], [0.0, 0.0, -2e-4, 0.0, -16.0]])
def test_a_high_tilt_far_from_eucentric_height_is_refused_before_the_instrument(target):
    client = FakeAutoScriptClient()
    stage = FakeStageProxy(client)
    with pytest.raises(ValueError, match="pole piece"):
        stage.position = target
    assert client.writes("specimen.stage.absolute_move") == []
    stage.position = [0.0, 0.0, 5e-5, 20.0, 0.0]  # the same tilt near eucentric height is allowed
    stage.position = [0.0, 0.0, 3e-4, 10.0, 0.0]  # and a small tilt far from it


def test_a_beta_on_a_holder_without_beta_is_refused_not_dropped():
    client = FakeAutoScriptClient()
    client.specimen.stage.holder_type = "SingleTilt"
    stage = FakeStageProxy(client)
    with pytest.raises(ValueError, match="no beta tilt"):
        stage.position = [0.0, 0.0, 0.0, 5.0, 3.0]
    assert client.writes("specimen.stage.absolute_move") == []


def test_the_z_tilt_check_reads_its_limits_from_the_device():
    device = SimpleNamespace(high_tilt_deg=30.0, high_tilt_max_abs_z_m=4e-4, read_beta_tilt_enabled=lambda: True)
    STAGE._check_z_and_tilt(device, [0.0, 0.0, 3e-4, 25.0, 0.0])
    with pytest.raises(ValueError, match="high_tilt_deg"):
        STAGE._check_z_and_tilt(device, [0.0, 0.0, 4.5e-4, 31.0, 0.0])


# ----------------------------------------------------------------------
# B2: blanked again after each acquisition, before the save
# ----------------------------------------------------------------------
@pytest.mark.parametrize("command, args", [("acquire_scanned_image", (["haadf"],)), ("acquire_spectrum", ("eds",)), ("acquire_camera_image", ())])
def test_a_blanked_beam_is_blanked_again_before_the_save(autoscript, monkeypatch, command, args):
    device, client = autoscript
    client.optics.blanker.blank()
    seen = blanker_at_save(monkeypatch)
    getattr(EM, command)(device, *args)
    assert seen == [True]  # the acquisition opened the blanker; it was closed before the file was written
    assert client.optics.blanker._blanked


def test_a_failed_acquisition_still_blanks_the_beam(autoscript, monkeypatch):
    device, client = autoscript
    client.optics.blanker.blank()

    def failing(settings):
        object.__setattr__(client.optics.blanker, "_blanked", False)  # the column opened it, then the call failed
        raise RuntimeError("acquisition failed")

    object.__setattr__(client.acquisition, "acquire_stem_images_advanced", failing)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        EM.acquire_scanned_image(device, ["haadf"])
    assert client.optics.blanker._blanked


def test_an_unblanked_beam_is_left_to_its_caller(autoscript, monkeypatch):
    # The Mapper and the Tuner unblank around acquire_scanned_image and blank after it: never unblank or blank here.
    device, client = autoscript
    EM.unblank_beam(device)
    EM.acquire_scanned_image(device, ["haadf"])
    assert client.writes("optics.blanker") == [{"path": "optics.blanker.unblank", "op": "call", "value": None}]
    EM.blank_beam(device)
    assert client.optics.blanker._blanked


# ----------------------------------------------------------------------
# B3: 4D-STEM bounds
# ----------------------------------------------------------------------
def test_4dstem_takes_the_camera_exposure_and_blanks_between_points(autoscript, monkeypatch):
    device, client = autoscript
    device._detector_proxies["scan"].imsize = 2
    client.optics.blanker.blank()
    blanked_at_move = []
    original_set = type(client.optics).__setattr__

    def watch(optics, name, value):
        if name == "paused_scan_beam_position":
            blanked_at_move.append(optics.blanker._blanked)
        original_set(optics, name, value)

    monkeypatch.setattr(type(client.optics), "__setattr__", watch)
    seen = blanker_at_save(monkeypatch)
    EM.acquire_scanned_data_advanced(device)
    frames = [c["value"] for c in client.writes("acquisition.acquire_camera_image_advanced")]
    assert len(frames) == 4 and all(f["exposure_time"] == 0.01 for f in frames)  # the camera's, not the 1 us dwell
    assert blanked_at_move == [True] * 5  # four points and the restore: the probe moves only blanked
    assert seen == [True]


@pytest.mark.parametrize("change, message", [
    (lambda d: setattr(d._detector_proxies["scan"], "imsize", 65), "max_4dstem_imsize"),
    (lambda d: setattr(d._detector_proxies["camera"], "exposure_time", 1e-6), "min_4dstem_exposure_s"),
    (lambda d: setattr(d._detector_proxies["scan"], "scan_region", [0.5, 0.5, 0.8, 0.8]), "scan_region"),
    (lambda d: d._detector_proxies.pop("camera"), "CAMERA device"),
])
def test_4dstem_outside_its_bounds_is_refused_before_the_instrument(autoscript, change, message):
    device, client = autoscript
    change(device)
    with pytest.raises(ValueError, match=message):
        EM.acquire_scanned_data_advanced(device)
    assert client.writes("optics") == [] and client.writes("acquisition") == []


def test_a_blank_during_4dstem_stops_it(autoscript):
    device, client = autoscript
    device._detector_proxies["scan"].imsize = 4
    acquire = client.acquisition.acquire_camera_image_advanced
    taken = []

    def frame(settings):
        taken.append(1)
        if len(taken) == 3:
            EM.blank_beam(device)  # the e-stop, from another client, while the third frame is taken
        return acquire(settings)

    object.__setattr__(client.acquisition, "acquire_camera_image_advanced", frame)
    with pytest.raises(RuntimeError, match="stopped after 3 of 16 frames"):
        EM.acquire_scanned_data_advanced(device)
    assert len(taken) == 3 and client.optics.blanker._blanked


# ----------------------------------------------------------------------
# B4: the e-stop's blank does not queue
# ----------------------------------------------------------------------
def test_blank_beam_does_not_wait_for_a_running_acquisition(autoscript):
    device, client = autoscript
    # Its own connection: the AutoScript client runs one call at a time, so the main one would be busy.
    device._blank_client = SimpleNamespace(optics=SimpleNamespace(blanker=client.optics.blanker))
    started, release = threading.Event(), threading.Event()
    acquire = client.acquisition.acquire_stem_images_advanced

    def long_frame(settings):
        started.set()
        release.wait(10)
        return acquire(settings)

    object.__setattr__(client.acquisition, "acquire_stem_images_advanced", long_frame)
    worker = threading.Thread(target=EM.acquire_scanned_image, args=(device, ["haadf"]))
    worker.start()
    try:
        assert started.wait(5)
        assert not device._command_lock.acquire(blocking=False)  # the acquisition holds the device
        t0 = time.monotonic()
        EM.blank_beam(device)
        assert time.monotonic() - t0 < 1.0 and client.optics.blanker._blanked
    finally:
        release.set()
        worker.join(10)


def test_the_microscope_server_runs_without_tangos_command_lock(monkeypatch):
    chosen = []
    monkeypatch.setattr(auto_script_module.tango.Util, "instance", staticmethod(lambda: SimpleNamespace(set_serial_model=chosen.append)))
    auto_script_module._without_tango_serialization()
    assert chosen == [tango.SerialModel.NO_SYNC]


# ----------------------------------------------------------------------
# B8: the stage gets time to move
# ----------------------------------------------------------------------
def test_the_stage_proxy_gets_the_stage_timeout(monkeypatch):
    timeouts = {}

    class Proxy:
        def __init__(self, address):
            self.address = address

        def set_timeout_millis(self, ms):
            timeouts[self.address] = ms

    monkeypatch.setattr(auto_script_module.tango, "DeviceProxy", Proxy)
    device = auto_script_module.AutoScriptMicroscope.__new__(auto_script_module.AutoScriptMicroscope)
    device._tango_properties = {}
    for name in ("eds", "scan", "camera", "corrector"):
        setattr(device, f"{name}_device_address", "")
    device.stage_device_address, device.data_device_address, device.stage_timeout_seconds = "s/t/1", "d/t/1", 240.0
    device._detector_proxies = {}
    device.info_stream = device.error_stream = lambda *a: None
    auto_script_module.AutoScriptMicroscope._connect_detector_proxies(device)
    assert timeouts == {"s/t/1": 240_000, "d/t/1": 3_600_000}
