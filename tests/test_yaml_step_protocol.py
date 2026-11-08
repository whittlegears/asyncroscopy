"""How a YAML instrument's job device drives it: the hold, the steps, and what happens when the job device goes away.

The devices are in tests/conftest.py: asyncroscopy/steptwin/default (one simulated second is 20 ms) with
asyncroscopy/steptwinjob/default, and asyncroscopy/steptwin/lapsing, whose hold lapses after 0.4 s.
"""

import json
import threading
import time

import pytest
import tango

from asyncroscopy.instruments.yaml_sim.device import YamlInstrument
from asyncroscopy.utils.commands import declared_hints

PROTOCOL = ("hold", "release_hold", "start_step", "step_status", "stop_step")


@pytest.fixture
def instrument(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/steptwin/default"))
    proxy.set_timeout_millis(5000)
    yield proxy
    proxy.close_beam()


@pytest.fixture
def lapsing(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/steptwin/lapsing"))
    proxy.set_timeout_millis(5000)
    return proxy


@pytest.fixture
def job_device(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/steptwinjob/default"))
    proxy.set_timeout_millis(10000)
    return proxy


def start(proxy, token, command, args=None):
    proxy.start_step(json.dumps({"job_token": token, "command": command, "args": args}))


def status(proxy):
    return json.loads(proxy.step_status())


def wait_until(condition, limit_s=5.0):
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def call(job_device, verb, **body):
    return json.loads(getattr(job_device, verb)(json.dumps(body)))


def run_job(job_device, plan):
    reserved = call(job_device, "reserve", plan=plan)
    assert "error" not in reserved, reserved
    started = call(job_device, "start", reservation_id=reserved["reservation_id"])
    return reserved["reservation_id"], started["job_id"]


def job_state(job_device, job_id):
    return call(job_device, "job_status", job_id=job_id)["state"]


def test_agents_never_see_the_hold_and_step_commands():
    hints = declared_hints(YamlInstrument)
    assert all(hints[name].get("operator_only") is True for name in PROTOCOL)
    assert hints["hold"]["risk"] == hints["start_step"]["risk"] == "high"
    assert hints["release_hold"]["risk"] == hints["step_status"]["risk"] == hints["stop_step"]["risk"] == "low"


def test_the_agent_endpoint_leaves_them_out_and_keeps_the_rest(tango_ctx, monkeypatch):
    from asyncroscopy.testing import mcp_server_for

    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", lambda name: tango.DeviceProxy(tango_ctx.get_device_access(name)))
    group = [{"class": YamlInstrument, "devices": [{"name": "asyncroscopy/steptwin/default"}]}]
    for role, visible in (("agent", False), ("operator", True)):
        server = mcp_server_for(group, endpoint_role=role)
        tools = server.list_tool_descriptions()
        assert {"StepTwin_set_exposure_s", "StepTwin_acquire_counts", "StepTwin_close_beam"} <= set(tools)
        assert {f"StepTwin_{name}" in tools for name in PROTOCOL} == {visible}


def test_while_held_every_direct_set_and_acquire_is_refused_with_a_clear_tango_error(instrument):
    instrument.hold("job-a")
    try:
        for attempt in (lambda: instrument.set_exposure_s(2.0), instrument.acquire_counts, lambda: instrument.write_attribute("exposure_s", 2.0)):
            with pytest.raises(tango.DevFailed) as raised:
                attempt()
            assert raised.value.args[0].reason == "InstrumentHeld" and "busy: job job-a holds this instrument" in raised.value.args[0].desc
        assert instrument.exposure_s == 1.0 and instrument.get_exposure_s() == 1.0  # reading still works
        instrument.open_beam()  # and so do commands, a beam or laser switch-off above all
        with pytest.raises(tango.DevFailed, match="job-a holds this instrument"):
            instrument.hold("job-b")  # one job at a time
    finally:
        instrument.release_hold("job-a")
    instrument.set_exposure_s(1.0)


def test_step_commands_need_the_holders_token(instrument):
    with pytest.raises(tango.DevFailed, match="not held"):
        start(instrument, "job-a", "set_stage_x_m", 0.1)
    instrument.hold("job-a")
    try:
        with pytest.raises(tango.DevFailed, match="holds this instrument, not job-b"):
            start(instrument, "job-b", "set_stage_x_m", 0.1)
        with pytest.raises(tango.DevFailed, match="not a setting"):
            start(instrument, "job-a", "reboot_the_lab")
        start(instrument, "job-a", "set_stage_x_m", 0.1)
        assert wait_until(lambda: status(instrument)["state"] == "done")
        assert instrument.stage_x_m == 0.1
        start(instrument, "job-a", "acquire_counts")
        assert wait_until(lambda: status(instrument)["state"] == "done")
        assert json.loads(status(instrument)["result"]) == {"counts": 100.0}
        start(instrument, "job-a", "set_stage_x_m", 5.0)  # out of range: the refusal is the step's outcome
        assert wait_until(lambda: status(instrument)["state"] == "failed")
        assert "stage_x_m = 5" in status(instrument)["error"]
    finally:
        instrument.release_hold("job-a")
        instrument.set_stage_x_m(0.0)


def test_stop_step_ends_a_running_step_within_a_fifth_of_a_second(instrument):
    instrument.hold("job-a")

    try:
        start(instrument, "job-a", "_settle", 3600.0)  # a 72 s wait on this device
        time.sleep(0.05)
        assert status(instrument)["state"] == "running"
        began = time.monotonic()
        instrument.stop_step("job-a")
        assert time.monotonic() - began < 0.2
        assert status(instrument)["state"] == "stopped"
    finally:
        instrument.release_hold("job-a")


def test_a_job_abort_stops_the_running_step_within_a_fifth_of_a_second(instrument, job_device):
    reservation, job_id = run_job(job_device, {"axes": {"stage_x_m": {"start": 0.0, "stop": 0.1, "points": 2}}, "settings": {"exposure_s": 3600.0}})
    assert wait_until(lambda: status(instrument)["state"] == "running" and status(instrument)["command"] == "acquire_counts")
    ended = []

    def watch():
        while status_of["state"] == "running":
            status_of.update(status(watcher))
        ended.append(time.monotonic())

    watcher = tango.DeviceProxy(instrument.dev_name())
    status_of = {"state": "running"}
    thread = threading.Thread(target=watch)
    thread.start()
    began = time.monotonic()
    reply = call(job_device, "abort", job_id=job_id)
    thread.join(timeout=5)
    assert status_of["state"] == "stopped" and ended[0] - began < 0.2
    assert reply["state"] == "aborted" or wait_until(lambda: job_state(job_device, job_id) == "aborted")
    instrument.set_exposure_s(1.0)  # the hold is given back
    call(job_device, "release", reservation_id=reservation)


def test_a_step_may_run_far_longer_than_one_tango_call_is_allowed(instrument, job_device):
    # 120 simulated seconds (2.4 s here): a direct call of that length is refused...
    instrument.set_exposure_s(120.0)
    with pytest.raises(tango.DevFailed, match="would take 120 s; one call is limited to 25 s"):
        instrument.acquire_counts()
    instrument.set_exposure_s(1.0)
    # ...but as a job step every Tango call returns at once, and the step goes on for its whole length.
    reservation, job_id = run_job(job_device, {"axes": {"stage_x_m": {"start": 0.0, "stop": 0.1, "points": 1}}, "settings": {"exposure_s": 120.0}})
    assert wait_until(lambda: status(instrument)["state"] == "running" and status(instrument)["command"] == "acquire_counts")
    started = time.monotonic()
    slowest = 0.0
    while job_state(job_device, job_id) == "running":
        before = time.monotonic()
        status(instrument)
        slowest = max(slowest, time.monotonic() - before)
        time.sleep(0.02)
        assert time.monotonic() - started < 30.0
    assert job_state(job_device, job_id) == "done", call(job_device, "job_status", job_id=job_id)
    assert time.monotonic() - started > 1.5  # the step really took its simulated time (0.02 real seconds per second)
    assert slowest < 0.5
    keys = call(job_device, "results", job_id=job_id)["summary"]["keys"]["acquire_counts"]
    assert json.loads(keys[0]) == {"counts": 12000.0}
    instrument.set_exposure_s(1.0)  # a finished job gives the hold back
    call(job_device, "release", reservation_id=reservation)


def test_a_slow_direct_command_during_a_hold_does_not_delay_the_jobs_calls(instrument):
    instrument.hold("job-a")
    try:
        worker = threading.Thread(target=lambda: tango.DeviceProxy(instrument.dev_name()).warm_up())  # 20 simulated seconds: 0.4 s here
        worker.start()
        time.sleep(0.1)
        began = time.monotonic()
        status(instrument)
        assert time.monotonic() - began < 0.25  # Tango's device monitor is free while the command waits
        worker.join(timeout=5)
    finally:
        instrument.release_hold("job-a")


def test_a_hold_that_is_not_renewed_lapses_and_makes_the_instrument_safe(lapsing):
    lapsing.open_beam()
    assert lapsing.beam_on is True
    lapsing.hold("job-gone")
    start(lapsing, "job-gone", "_settle", 3600.0)
    with pytest.raises(tango.DevFailed, match="busy"):
        lapsing.set_exposure_s(2.0)
    assert wait_until(lambda: status(lapsing)["state"] == "stopped")
    assert "lapsed" in status(lapsing)["error"]
    assert wait_until(lambda: not lapsing.beam_on)  # the file's on_stop commands ran
    assert wait_until(lambda: _not_busy(lapsing))  # free again
    with pytest.raises(tango.DevFailed, match="not held"):
        start(lapsing, "job-gone", "set_stage_x_m", 0.1)  # the job that lost it cannot go on
    lapsing.hold("job-next")  # the next one can take it
    lapsing.release_hold("job-next")


def _not_busy(device) -> bool:
    try:
        device.set_exposure_s(1.0)
        return True
    except tango.DevFailed:
        return False


def test_renewing_keeps_the_hold(lapsing):
    lapsing.hold("job-alive")
    began = time.monotonic()
    while time.monotonic() - began < 1.0:  # well past the 0.4 s lease
        lapsing.hold("job-alive")
        time.sleep(0.05)
    with pytest.raises(tango.DevFailed, match="busy"):
        lapsing.set_exposure_s(2.0)
    lapsing.release_hold("job-alive")
    assert _not_busy(lapsing)


def test_the_launcher_starts_the_job_device_only_for_a_file_with_jobs():
    from pathlib import Path

    from startup_scripts import run_servers

    root = Path(__file__).resolve().parents[1]
    raman = run_servers.load_config(root / "configs" / "yaml_sim_raman.yaml")
    devices = {device.key: device for device in run_servers.build_devices(raman)}
    job = devices["job"]
    assert (job.class_name, job.module_name, job.device_name) == ("RamanTwinJob", "asyncroscopy.instruments.yaml_sim.job", "asyncroscopy/job/default")
    assert job.properties["instrument_device_address"] == ["asyncroscopy/instrument/default"]
    assert job.properties["yaml_path"] == raman.instrument.properties["yaml_path"] and job.properties["simulated"] == ["true"]
    assert job.start_after_dependencies and list(devices).index("job") > list(devices).index("instrument")
    assert "RamanTwinJob job_instance" in run_servers.instrument_cleanup_patterns(raman)
    xrd = run_servers.load_config(root / "configs" / "yaml_sim_xrd.yaml")
    assert "job" not in {device.key for device in run_servers.build_devices(xrd)}
