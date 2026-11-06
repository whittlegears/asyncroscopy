"""JobDevice's limit check at reserve and its suspenders, on JobDevice's envelope logic without a Tango server."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from asyncroscopy.contracts import validate
from asyncroscopy.jobs import JobDevice
from asyncroscopy.jobs.job_device import parse_suspenders, suspender_problem
from asyncroscopy.mcp.tool_log import TraceWriter


class FakeDevice:
    """A Tango device as far as the checks see it: attribute configs and values."""

    def __init__(self, limits=None, values=None):
        self.limits = limits or {}
        self.values = values or {}

    def get_attribute_config(self, name):
        low, high = self.limits.get(name, ("Not specified", "Not specified"))
        return SimpleNamespace(min_value=str(low), max_value=str(high))

    def read_attribute(self, name):
        value = self.values[name]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(value=value)


def envelope_class():
    """JobDevice's envelope methods on a plain class (a Tango Device cannot be built without a server)."""
    methods = {
        name: value for name, value in vars(JobDevice).items()
        if name.startswith("_") and not name.startswith("__") and (callable(value) or isinstance(value, (classmethod, staticmethod)))
    }
    for name in ("aborting", "on_start", "on_stop", "on_finish", "on_suspend", "step_targets", "summary"):
        methods[name] = vars(JobDevice)[name]
    return type("Envelope", (), methods)


class Steps(envelope_class()):
    """A job of n steps that each set stage x to a planned value."""

    PLAN_EXAMPLE = {"xs": [0.0]}

    def __init__(self, tmp_path, devices, suspenders=""):
        self._lock, self._jobs, self._unloaded, self._job_index, self._threads, self._abort = threading.RLock(), {}, {}, {}, {}, {}
        self._name, self._generation, self._stale_workers = "test/steps/1", 1, ()
        self._state_dir = tmp_path
        self._trace = TraceWriter(str(tmp_path / "trace.jsonl"), source="job")
        self._suspenders = parse_suspenders(suspenders)
        self.suspender_poll_s = 0.02
        self.devices = devices
        self.stopped, self.suspended_reasons, self.ran = [], [], []
        self.step_delay = 0.0

    def proxy(self, address, timeout_s=None):
        device = self.devices.get(address)
        if device is None:
            raise RuntimeError(f"DevFailed: device {address} not exported")
        return device

    def _omni_worker(self, job, job_id):
        self._worker(job, job_id)

    def resolve_plan(self, plan):
        return {"steps": [{"x": float(x)} for x in plan["xs"]], "estimate": {"steps": len(plan["xs"])}}

    def step_targets(self, job):
        return [{"step": i, "device": "test/stage/1", "attribute": "x", "value": s["x"]} for i, s in enumerate(job["steps"])]

    def run_step(self, job, index):
        time.sleep(self.step_delay)
        self.ran.append(index)
        return {"index": index}

    def on_stop(self, job, reason):
        self.stopped.append(reason)

    def on_suspend(self, job, reason):
        self.suspended_reasons.append(reason)

    def call(self, verb, body):
        handler = {"reserve": self._reserve, "start": self._start, "status": self._status, "abort": self._abort_job, "release": self._release}[verb]
        return json.loads(self._guard(handler, json.dumps(body)))

    def wait_for(self, job_id, states, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.call("status", {"job_id": job_id})
            if status["state"] in states:
                return status
            time.sleep(0.01)
        raise AssertionError(f"job never reached {states}: {status}")


def stage(low=-1e-3, high=1e-3):
    return FakeDevice(limits={"x": (low, high)})


# ---------------------------------------------------------------- limits at reserve
def test_a_step_outside_the_stage_limits_is_refused_before_anything_is_reserved(tmp_path):
    job = Steps(tmp_path, {"test/stage/1": stage()})
    reply = job.call("reserve", {"plan": {"xs": [0.0, 5e-4, 2e-3, -3e-3]}})
    validate(reply, "envelope", "error_response")
    assert reply["error"]["code"] == "plan_rejected"
    assert reply["error"]["violations_total"] == 2
    assert [(v["step"], v["value"], v["max"]) for v in reply["error"]["violations"]] == [(2, 2e-3, 1e-3), (3, -3e-3, 1e-3)]
    assert "step 2 sets test/stage/1/x to 0.002" in reply["error"]["message"]
    assert job._jobs == {} and not list(tmp_path.glob("*.json"))


def test_a_plan_inside_the_limits_reserves_and_says_what_was_checked(tmp_path):
    job = Steps(tmp_path, {"test/stage/1": stage()})
    reply = job.call("reserve", {"plan": {"xs": [0.0, 5e-4]}})
    validate(reply, "envelope", "reserve_response")
    assert reply["limits"] == {"targets": 2, "unchecked": []}


def test_limits_that_cannot_be_read_refuse_the_plan(tmp_path):
    job = Steps(tmp_path, {})
    reply = job.call("reserve", {"plan": {"xs": [0.0, 9.0]}})
    assert reply["error"]["code"] == "plan_rejected" and "could not be read" in reply["error"]["message"]
    assert reply["error"]["unchecked"][0]["target"] == "test/stage/1/x"
    assert job._jobs == {}


def test_a_device_may_reserve_with_unread_limits_and_report_them(tmp_path):
    job = Steps(tmp_path, {})
    job.require_limits = False
    reply = job.call("reserve", {"plan": {"xs": [0.0, 9.0]}})
    assert "reservation_id" in reply
    assert reply["limits"]["unchecked"][0]["target"] == "test/stage/1/x"
    assert "not exported" in reply["limits"]["unchecked"][0]["reason"]


def test_unspecified_tango_limits_check_nothing(tmp_path):
    job = Steps(tmp_path, {"test/stage/1": FakeDevice()})
    assert "reservation_id" in job.call("reserve", {"plan": {"xs": [1e9]}})


def test_limits_a_target_carries_are_used(tmp_path):
    class Declared(Steps):
        def step_targets(self, job):
            return [{"step": i, "device": "test/indenter/1", "attribute": "load_n", "value": s["x"], "max": 0.5} for i, s in enumerate(job["steps"])]

    reply = Declared(tmp_path, {}).call("reserve", {"plan": {"xs": [0.1, 0.9]}})
    assert reply["error"]["violations"] == [{"step": 1, "device": "test/indenter/1", "attribute": "load_n", "value": 0.9, "min": None, "max": 0.5}]


def test_a_broken_step_targets_is_the_devices_error(tmp_path):
    class Broken(Steps):
        def step_targets(self, job):
            raise AttributeError("no stage address")

    reply = Broken(tmp_path, {}).call("reserve", {"plan": {"xs": [0.0]}})
    assert reply["error"]["code"] == "device_error" and "step_targets" in reply["error"]["message"]


# ---------------------------------------------------------------- suspenders
@pytest.mark.parametrize("text, problem", [
    ("{}", "JSON list"),
    ("[{\"attr\": \"beam/current\", \"min\": 1}]", "<domain>/<family>/<member>/<attribute>"),
    ("[{\"attr\": \"a/b/c/d\"}]", "min, max or equals"),
    ("[{\"attr\": \"a/b/c/d\", \"min\": \"10\"}]", "must be a number"),
    ("[{\"attr\": \"a/b/c/d\", \"minimum\": 10}]", "unknown keys"),
])
def test_a_bad_suspender_property_is_refused_with_the_reason(text, problem):
    with pytest.raises(ValueError, match=problem):
        parse_suspenders(text)


def test_suspender_conditions():
    low = parse_suspenders('[{"attr": "a/b/c/current", "min": 10, "label": "beam current"}]')[0]
    assert low["resume_after_s"] == 10.0
    assert suspender_problem(low, 12.0) is None
    assert suspender_problem(low, 3.0) == "beam current is 3, below 10"
    assert "below" in suspender_problem(low, float("nan"))
    valve = parse_suspenders('[{"attr": "a/b/c/valve", "equals": "open"}]')[0]
    assert suspender_problem(valve, "closed") == "a/b/c/valve is 'closed', needs 'open'"


def beam_job(tmp_path, current, resume_after_s=0.2):
    beam = FakeDevice(values={"current": current})
    suspenders = json.dumps([{"attr": "test/beam/1/current", "min": 10, "resume_after_s": resume_after_s, "label": "beam current"}])
    job = Steps(tmp_path, {"test/stage/1": stage(), "test/beam/1": beam}, suspenders)
    return job, beam


def test_a_bad_condition_pauses_the_job_which_resumes_once_it_has_been_good_long_enough(tmp_path):
    job, beam = beam_job(tmp_path, 50.0)
    job.step_delay = 0.05
    reservation = job.call("reserve", {"plan": {"xs": [0.0] * 6}})["reservation_id"]
    job_id = job.call("start", {"reservation_id": reservation})["job_id"]
    deadline = time.time() + 10.0
    while len(job.ran) < 2:  # a worker that fails before its steps must fail this test, not hang it
        assert time.time() < deadline, job.call("status", {"job_id": job_id})
        time.sleep(0.005)
    beam.values["current"] = 2.0  # beam lost
    paused = job.wait_for(job_id, {"paused_suspended"})
    validate(paused, "envelope", "status_response")
    assert paused["suspended"]["reason"] == "beam current is 2, below 10"
    assert job.suspended_reasons == ["beam current is 2, below 10"]
    steps_when_paused = len(job.ran)
    time.sleep(0.1)
    assert len(job.ran) == steps_when_paused  # no step runs while paused
    beam.values["current"] = 50.0
    done = job.wait_for(job_id, {"done"})
    assert done["suspended"] is None and done["progress"] == {"done": 6, "total": 6}
    actions = [json.loads(line)["action"] for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert "Steps.suspended" in actions and "Steps.resumed" in actions


def test_a_condition_that_flickers_restarts_the_resume_wait(tmp_path):
    job, beam = beam_job(tmp_path, 2.0, resume_after_s=0.3)
    reservation = job.call("reserve", {"plan": {"xs": [0.0]}})["reservation_id"]
    job_id = job.call("start", {"reservation_id": reservation})["job_id"]
    job.wait_for(job_id, {"paused_suspended"})
    beam.values["current"] = 50.0
    status = job.wait_for(job_id, {"paused_suspended"})
    deadline = time.time() + 2
    while "resumes_in_s" not in (status.get("suspended") or {}) and time.time() < deadline:
        status = job.call("status", {"job_id": job_id})
    assert status["suspended"]["reason"] == "conditions are good again"
    beam.values["current"] = 2.0  # drops again before resume_after_s
    time.sleep(0.1)
    assert job.call("status", {"job_id": job_id})["state"] == "paused_suspended" and job.ran == []
    beam.values["current"] = 50.0
    job.wait_for(job_id, {"done"})


def test_abort_works_while_paused_and_leaves_the_instrument_safe(tmp_path):
    job, _beam = beam_job(tmp_path, 2.0, resume_after_s=60)
    reservation = job.call("reserve", {"plan": {"xs": [0.0, 0.0]}})["reservation_id"]
    job_id = job.call("start", {"reservation_id": reservation})["job_id"]
    job.wait_for(job_id, {"paused_suspended"})
    release = job.call("release", {"reservation_id": reservation})
    assert release["error"]["code"] == "bad_state"  # a paused job must be aborted first
    aborted = job.call("abort", {"job_id": job_id})
    assert aborted["state"] in {"aborting", "aborted"}
    assert job.wait_for(job_id, {"aborted"})["state"] == "aborted"
    assert job.stopped == ["aborted"] and job.ran == []


def test_a_pause_that_could_not_make_the_instrument_safe_says_so(tmp_path):
    job, _beam = beam_job(tmp_path, 2.0, resume_after_s=60)

    def broken_blank(job_record, reason):
        raise RuntimeError("DevFailed: blank_beam timed out")

    job.on_suspend = broken_blank
    reservation = job.call("reserve", {"plan": {"xs": [0.0]}})["reservation_id"]
    job_id = job.call("start", {"reservation_id": reservation})["job_id"]
    paused = job.wait_for(job_id, {"paused_suspended"})
    assert "not made safe" in paused["last_error"] and "blank_beam timed out" in paused["last_error"]
    job.call("abort", {"job_id": job_id})
    job.wait_for(job_id, {"aborted"})


def test_an_unreadable_condition_counts_as_bad(tmp_path):
    job, beam = beam_job(tmp_path, RuntimeError("DevFailed: beam monitor offline"))
    reservation = job.call("reserve", {"plan": {"xs": [0.0]}})["reservation_id"]
    job_id = job.call("start", {"reservation_id": reservation})["job_id"]
    paused = job.wait_for(job_id, {"paused_suspended"})
    assert "could not read test/beam/1/current" in paused["suspended"]["reason"]
    beam.values["current"] = 50.0
    job.wait_for(job_id, {"done"})


def test_a_job_paused_when_the_device_restarted_comes_back_resumable(tmp_path):
    job, _beam = beam_job(tmp_path, 2.0, resume_after_s=60)
    reservation = job.call("reserve", {"plan": {"xs": [0.0]}})["reservation_id"]
    job_id = job.call("start", {"reservation_id": reservation})["job_id"]
    job.wait_for(job_id, {"paused_suspended"})
    restarted, _ = beam_job(tmp_path, 50.0)
    restarted._reload()
    reloaded = restarted._jobs[reservation]
    assert reloaded["state"] == "failed" and reloaded["last_error"]["code"] == "device_restarted"
    job.call("abort", {"job_id": job_id})
    job.wait_for(job_id, {"aborted"})


# ---------------------------------------------------------------- dry runs
class FakeAfm:
    """An SPM as a DeviceProxy: get_image scans the tip, despite its read-only name."""

    def __init__(self):
        self.calls = []

    def set_timeout_millis(self, ms):
        pass

    def info(self):
        return SimpleNamespace(dev_class="AFMTwin")

    def get_command_list(self):
        return ["get_stage", "get_image", "move_xy"]

    def get_attribute_list(self):
        return []

    def get_stage(self):
        self.calls.append("get_stage")
        return [0.0, 0.0]

    def get_image(self):
        self.calls.append("get_image")
        return "image_1.h5"


class Scan(Steps):
    """One step that reads the stage and takes an image, each through JobDevice.proxy (a RecordingProxy in a dry run)."""

    SUPPORTS_DRY_RUN = True
    proxy = vars(JobDevice)["proxy"]
    check_dry_call = vars(JobDevice)["check_dry_call"]
    safe_commands = vars(JobDevice)["safe_commands"]
    SAFE_COMMANDS = frozenset()
    step_timeout_s = 5.0
    _proxies = threading.local()

    def resolve_plan(self, plan):
        return {"steps": [{}], "estimate": {"steps": 1}}

    def step_targets(self, job):
        return []

    def run_step(self, job, index):
        afm = self.proxy("test/afm/1")
        return {"stage": afm.get_stage(), "image": afm.get_image()}


@pytest.mark.parametrize("pattern", ["AFMTwin.get_image", "AFMTwin_get_image", "*.get_image"])
def test_a_dry_run_records_a_read_named_command_the_deployment_marks_high(tmp_path, monkeypatch, pattern):
    from asyncroscopy.jobs.recording_proxy import DRY_RUN_PLACEHOLDER

    afm = FakeAfm()
    monkeypatch.setattr("asyncroscopy.jobs.job_device.tango.DeviceProxy", lambda address: afm)
    job = Scan(tmp_path, {})
    reserved = job.call("reserve", {"plan": {"dry_run": True}, "risk_overrides": {pattern: "high", "AFMTwin.get_stage": "low"}})
    job_id = job.call("start", {"reservation_id": reserved["reservation_id"]})["job_id"]
    job.wait_for(job_id, {"done"})
    record = job._jobs[reserved["reservation_id"]]["step_results"]["0"]
    assert afm.calls == ["get_stage"]  # the scan was recorded, not run; the plain read still passed through
    assert record == {"stage": [0.0, 0.0], "image": DRY_RUN_PLACEHOLDER}
    assert job._jobs[reserved["reservation_id"]]["dry_run_calls"]["recorded"] == 1


def test_a_device_that_cannot_rehearse_refuses_a_dry_run_at_reserve(tmp_path):
    job = Steps(tmp_path, {"test/stage/1": stage()})
    reply = job.call("reserve", {"plan": {"xs": [0.0], "dry_run": True}})
    assert reply["error"]["code"] == "dry_run_unsupported" and "Steps does not support dry runs" in reply["error"]["message"]
    assert job._jobs == {}


# ---------------------------------------------------------------- job state on disk
def test_the_plan_is_written_once_and_a_finished_job_is_opened_only_when_asked_for(tmp_path):
    job = Steps(tmp_path, {"test/stage/1": stage()})
    reservation = job.call("reserve", {"plan": {"xs": [0.0, 1e-4, 2e-4]}})["reservation_id"]
    plan_file = tmp_path / f"test_steps_1__{reservation}.plan.json"
    progress_file = tmp_path / f"test_steps_1__{reservation}.json"
    written = plan_file.stat().st_mtime_ns
    job_id = job.call("start", {"reservation_id": reservation})["job_id"]
    job.wait_for(job_id, {"done"})
    assert plan_file.stat().st_mtime_ns == written  # rewritten per step: only the progress record
    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress["state"] == "done" and "plan" not in progress and "resolved_plan" not in progress and "steps" not in progress
    assert json.loads(plan_file.read_text(encoding="utf-8"))["plan"] == {"xs": [0.0, 1e-4, 2e-4]}

    restarted = Steps(tmp_path, {"test/stage/1": stage()})
    restarted._reload()
    assert reservation not in restarted._jobs and reservation in restarted._unloaded  # a finished job stays on disk
    status = restarted.call("status", {"job_id": job_id})  # until it is asked for by id
    assert status["state"] == "done" and status["progress"]["done"] == 3
    assert restarted._jobs[reservation]["plan"] == {"xs": [0.0, 1e-4, 2e-4]} and len(restarted._jobs[reservation]["step_results"]) == 3
