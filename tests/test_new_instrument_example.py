"""The worked example in docs/Adding_New_Hardware/new_instrument.md, run for real.

If this test breaks, the guide is wrong: fix both together.
"""

import json
import time

import pytest
import tango

from asyncroscopy.instruments.nano_indenter.indent_grid import IndentGrid
from asyncroscopy.instruments.nano_indenter.sim_indenter import SimIndenter
from asyncroscopy.jobs import JobDevice
from asyncroscopy.safety.policy import is_envelope_device, is_probe_device, risk_of

PLAN = {"origin_m": [-2e-5, 0.0], "nx": 4, "ny": 2, "pitch_m": 1e-5, "load_n": 0.01}


@pytest.fixture
def grid(indent_grid_proxy, job_dir):
    # The devices live in the session's shared Tango context (tests/conftest.py):
    # one in-process context per interpreter is all PyTango allows.
    return indent_grid_proxy, job_dir / "indent"


def call(proxy, verb, **body):
    return json.loads(getattr(proxy, verb)(json.dumps(body)))


def wait(proxy, job_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = call(proxy, "job_status", job_id=job_id)
        if status["state"] not in ("running", "aborting"):
            return status
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_grid_runs_and_returns_hardness(grid):
    proxy, out = grid
    reserved = call(proxy, "reserve", plan=PLAN)
    assert reserved["estimate"]["indents"] == 8
    assert call(proxy, "reserve", plan=PLAN)["error"]["code"] == "busy"  # one reservation per instrument
    job_id = call(proxy, "start", reservation_id=reserved["reservation_id"])["job_id"]
    assert wait(proxy, job_id)["state"] == "done"
    indents = call(proxy, "results", job_id=job_id)["summary"]["indents"]
    assert len(indents) == 8
    soft = [i["hardness_gpa"] for i in indents if i["x_m"] < 0]
    hard = [i["hardness_gpa"] for i in indents if i["x_m"] >= 0]
    assert max(soft) < 3 < 6 < min(hard)  # the two simulated phases
    trace = (out / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count('"IndentGrid.step"') == 8
    call(proxy, "release", reservation_id=reserved["reservation_id"])


def test_a_dry_run_records_every_move_and_indent_and_makes_none(grid, tango_ctx):
    proxy, out = grid
    indenter = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/indenter/default"))
    indenter.move_xy([1e-5, 1e-5])
    reserved = call(proxy, "reserve", plan={**PLAN, "dry_run": True})
    job_id = call(proxy, "start", reservation_id=reserved["reservation_id"])["job_id"]
    status = wait(proxy, job_id)
    assert status["state"] == "done", status
    results = call(proxy, "results", job_id=job_id)
    assert results["dry_run"]["recorded"] == 8 * 2 + 1 and results["dry_run"]["failed_checks"] == 0  # moves, indents, the final retract
    assert all(i["hardness_gpa"] is None for i in results["summary"]["indents"])  # nothing was measured
    assert list(indenter.get_position()) == [1e-5, 1e-5]  # nothing moved
    call(proxy, "release", reservation_id=reserved["reservation_id"])


def test_a_bad_plan_is_refused_at_reserve(grid):
    proxy, _ = grid
    error = call(proxy, "reserve", plan={**PLAN, "nx": 0})["error"]
    assert error["code"] == "plan_rejected"
    assert "Example plan" in error["message"]  # a model can copy the shape and retry


def test_a_null_field_is_a_refused_plan_not_a_python_error(grid):
    proxy, _ = grid
    error = call(proxy, "reserve", plan={**PLAN, "pitch_m": None})["error"]  # small models send nulls
    assert error["code"] == "plan_rejected" and "Example plan" in error["message"]


def test_a_plan_beyond_the_instruments_limits_is_refused_at_reserve(grid):
    proxy, _ = grid
    too_far = call(proxy, "reserve", plan={**PLAN, "pitch_m": 0.01})["error"]
    too_hard = call(proxy, "reserve", plan={**PLAN, "load_n": 5.0})["error"]
    assert "travels" in too_far["message"] and "above" in too_hard["message"]


def test_a_restart_does_not_leave_the_device_busy(grid):
    proxy, _ = grid
    first = call(proxy, "reserve", plan=PLAN)
    proxy.Init()  # what a crash and restart would do
    second = call(proxy, "reserve", plan=PLAN)
    assert "reservation_id" in second, second
    assert call(proxy, "get_reservation", reservation_id=first["reservation_id"])["state"] == "released"
    call(proxy, "release", reservation_id=second["reservation_id"])


def test_the_agent_reads_each_commands_docstring(indent_grid_proxy, tango_ctx):
    indenter = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/indenter/default"))
    assert indenter.command_query("indent").out_type_desc.startswith("Make one indent")
    assert indenter.command_query("move_xy").in_type_desc.startswith(":param xy:")
    assert json.loads(indent_grid_proxy.plan_example)["nx"] == 5


def test_policy_sees_a_job_device_and_a_raw_instrument():
    assert is_envelope_device(dir(IndentGrid)) and issubclass(IndentGrid, JobDevice)
    assert not is_envelope_device(dir(SimIndenter))
    assert risk_of("IndentGrid", "start", envelope=True) == "high"
    assert risk_of("IndentGrid", "job_status", envelope=True) == "low"
    assert risk_of("SimIndenter", "indent") == "high"
    assert risk_of("SimIndenter", "move_xy") == "high"
    # A bare retract is low only on a device that brings a probe to the sample (a pump's withdraw aspirates);
    # the MCP server passes probe= from the device's own commands.
    assert is_probe_device(dir(SimIndenter))
    assert risk_of("SimIndenter", "retract", probe=True) == "low"
    assert risk_of("SimIndenter", "retract") == "high"
    assert risk_of("SimIndenter", "get_position") == "low"


def test_job_replies_keep_numpy_values_as_numbers():
    import numpy as np

    from asyncroscopy.jobs.job_device import _json_default
    assert json.loads(json.dumps({"i": np.int64(2), "xy": np.array([0.5, 1.5])}, default=_json_default)) == {"i": 2, "xy": [0.5, 1.5]}


def test_a_resolve_plan_bug_is_the_devices_not_the_callers():
    plan_caused = IndentGrid._plan_caused  # its PLAN_EXAMPLE names the plan fields
    assert plan_caused(KeyError("nx"), json.dumps({"plan": {"ny": 1}}))  # the plan lacks nx
    assert not plan_caused(KeyError("travl_m"), json.dumps({"plan": {"nx": 1}}))  # a typo in the device
    assert plan_caused(TypeError("x"), json.dumps({"plan": {"nx": None}}))
    assert not plan_caused(TypeError("x"), json.dumps({"plan": {"nx": 1}}))
    for confused in ({"origin_m": 5}, {"nx": [2]}, {"nx": {"a": 1}}, {"origin_m": [None, 0]}):  # wrong type or a nested null
        assert plan_caused(TypeError("x"), json.dumps({"plan": confused})), confused


def test_a_failed_state_save_on_start_leaves_the_reservation_startable():
    import threading

    class Envelope:  # JobDevice's envelope logic only, no Tango server
        _start, _parse, _active = JobDevice._start, JobDevice._parse, JobDevice._active

    grid = Envelope()
    job = {"reservation_id": "r1", "state": "reserved", "job_ids": [], "caller": None, "last_error": None}
    grid._lock, grid._jobs, grid._job_index, grid._abort, grid._threads, grid._stale_workers = threading.Lock(), {"r1": job}, {}, {}, {}, []

    def full_disk(_job):
        raise PermissionError("state file locked")

    grid._save = full_disk
    with pytest.raises(PermissionError):
        grid._start(json.dumps({"reservation_id": "r1"}))
    assert job["state"] == "reserved" and job["job_ids"] == [] and not grid._abort and not grid._job_index


def test_a_worker_that_cannot_start_leaves_the_reservation_startable(monkeypatch):
    import threading

    class Envelope:  # JobDevice's envelope logic only, no Tango server
        _start, _parse, _active = JobDevice._start, JobDevice._parse, JobDevice._active

    grid = Envelope()
    job = {"reservation_id": "r1", "state": "reserved", "job_ids": [], "caller": None, "last_error": None}
    saved = []
    grid._lock, grid._jobs, grid._job_index, grid._abort, grid._threads, grid._stale_workers = threading.Lock(), {"r1": job}, {}, {}, {}, []
    grid._save = lambda j: saved.append(j["state"])
    grid._omni_worker = lambda *a: None

    def no_threads(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", no_threads)
    with pytest.raises(RuntimeError):
        grid._start(json.dumps({"reservation_id": "r1"}))
    assert job == {"reservation_id": "r1", "state": "reserved", "job_ids": [], "caller": None, "last_error": None}
    assert saved == ["running", "reserved"] and not grid._abort and not grid._job_index and not grid._threads
