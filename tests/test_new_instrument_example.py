"""The worked example in docs/Adding_New_Hardware/new_instrument.md, run for real.

If this test breaks, the guide is wrong: fix both together.
"""

import json
import time

import pytest
import tango

from asyncroscopy.instruments.nano_indenter.indent_grid import IndentGrid
from asyncroscopy.jobs import JobDevice
from asyncroscopy.safety.policy import is_envelope_device, risk_of

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


def test_a_bad_plan_is_refused_at_reserve(grid):
    proxy, _ = grid
    error = call(proxy, "reserve", plan={**PLAN, "nx": 0})["error"]
    assert error["code"] == "plan_rejected"
    assert "Example plan" in error["message"]  # a model can copy the shape and retry


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
    grid_commands = [name for name in dir(IndentGrid) if name in ("reserve", "start", "job_status", "abort", "results", "release")]
    assert is_envelope_device(grid_commands) and issubclass(IndentGrid, JobDevice)
    assert risk_of("IndentGrid", "start", envelope=True) == "high"
    assert risk_of("IndentGrid", "job_status", envelope=True) == "low"
    assert risk_of("SimIndenter", "indent") == "high"
    assert risk_of("SimIndenter", "move_xy") == "high"
    assert risk_of("SimIndenter", "retract") == "low"
    assert risk_of("SimIndenter", "get_position") == "low"
