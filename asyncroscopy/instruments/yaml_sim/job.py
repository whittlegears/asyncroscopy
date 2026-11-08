"""YamlJob: the ``jobs:`` section of an instrument YAML file, as a job device driving the instrument.

The instrument is a separate device (``YamlInstrument``, device.py), driven through a proxy as IndentGrid
drives an indenter. Its tools carry the name ``<Name>Job`` (``RamanTwinJob_reserve``, and
``run_job(device="RamanTwinJob")``): one reserve, start, job_status, abort, results and release for the
whole file, with no Python written for the instrument.

Every step goes through the instrument's step protocol: ``hold`` keeps agents' ``set_*`` and ``acquire*``
calls out for the whole job (renewed so that it lasts as long as the job does, and lapses on its own if
this device dies), ``start_step`` returns at once, and ``step_status`` is polled in short calls. So one
step may take up to an hour while no Tango call takes more than milliseconds, and an abort stops the
running step within 0.2 s through ``stop_step``.
"""

from __future__ import annotations

import json
import sys
import threading
import time

import tango
from tango import DevState
from tango.server import device_property

from asyncroscopy.instruments.yaml_sim.device import registered_class
from asyncroscopy.instruments.yaml_sim.spec import (
    SETTLE_STEP,
    JobPlan,
    SpecError,
    job_estimate,
    job_targets,
    load_spec,
    plan_example,
    resolve_job_plan,
    resolve_path,
)
from asyncroscopy.jobs.job_device import JobDevice

KEYS_IN_SUMMARY = 200  # data keys per acquisition a job's results list (the newest); the rest are in its step records
CALL_TIMEOUT_S = 5.0  # a step-protocol call answers in milliseconds; if one does not, the instrument is in trouble
POLL_MIN_S = 0.002  # the wait between step_status calls starts here (so a quick set_ costs a few ms) ...
POLL_MAX_S = 0.05  # ... and doubles up to here: an abort reaches a running step in this long, plus one call


class _Stopped(Exception):
    """A step was cut short by an abort."""


class YamlJob(JobDevice):
    """The jobs of one instrument file; each step runs on the instrument through its step protocol."""

    # No dry runs: the steps read step_status replies that a recorded call does not produce.
    SUPPORTS_DRY_RUN = False

    yaml_path = device_property(dtype=str, default_value="", doc="The instrument YAML file (absolute, or relative to the working directory or the repository).")
    instrument_device_address = device_property(dtype=str, default_value="asyncroscopy/instrument/default", doc="The YamlInstrument these jobs drive.")

    def init_device(self) -> None:
        JobDevice.init_device(self)
        self._plans: dict[str, JobPlan] = {}
        self._holds: dict[str, threading.Event] = {}  # job id -> set when its hold is given back
        self.spec = None
        try:
            spec = load_spec(resolve_path(self.yaml_path))
        except SpecError as exc:
            message = f"{type(self).__name__}: the instrument file is not valid: {exc}"
            print(message, file=sys.stderr, flush=True)
            self.set_state(DevState.FAULT)
            self.set_status(message)
            return
        if not spec.jobs:
            self.set_state(DevState.FAULT)
            self.set_status(f"{type(self).__name__}: {spec.path.name} has no jobs: section; nothing to run")
            return
        self.spec = spec
        first = next(iter(spec.jobs.values()))
        kinds = "; ".join(f"{job.name}: {job.doc}" for job in spec.jobs.values())
        # The MCP server shows these with the job tools; set on the class, which the envelope reads.
        type(self).JOB_KIND = kinds
        type(self).PLAN_EXAMPLE = plan_example(spec, first)

    def trace_clock_device(self) -> str:
        return str(self.instrument_device_address)

    def _instrument(self) -> tango.DeviceProxy:
        return self.proxy(self.instrument_device_address, timeout_s=CALL_TIMEOUT_S)

    def _values(self) -> dict:
        """The instrument's current settings."""
        names = list(self.spec.settings)
        return dict(zip(names, (reply.value for reply in self._instrument().read_attributes(names)), strict=True))

    # -- job hooks --------------------------------------------------------
    def _plan(self, job: dict) -> JobPlan:
        plan = self._plans.get(job["reservation_id"])
        if plan is None:
            plan = self._plans[job["reservation_id"]] = resolve_job_plan(self.spec, job["plan"], self._values())
        return plan

    def resolve_plan(self, plan: dict) -> dict:
        if self.spec is None:
            raise ValueError(f"{type(self).__name__} has no valid instrument file: {self.get_status()}")
        values = self._values()
        resolved = resolve_job_plan(self.spec, plan, values)
        return {
            "job": resolved.job.name,
            "step_count": resolved.count,
            "settings": resolved.settings,
            "axes": {k: {"start": a, "stop": b, "points": n} for k, (a, b, n) in resolved.axes.items()},
            "estimate": {"steps": resolved.count, **job_estimate(self.spec, resolved, values)},
        }

    def step_targets(self, job: dict) -> list[dict]:
        resolved = resolve_job_plan(self.spec, job["plan"], self._values())
        return [
            {"step": step, "device": str(self.instrument_device_address), "attribute": setting, "value": value, "min": low, "max": high}
            for step, setting, value, low, high in job_targets(self.spec, resolved)
        ]

    def step_at(self, job: dict, index: int) -> dict:
        return self._plan(job).point(index)

    def phase(self, job: dict) -> str | None:
        return f"{self._plan(job).job.name}: acquiring" if job.get("state") == "running" else None

    # -- the instrument's step protocol ------------------------------------
    def _renew_hold(self, token: str, lease_s: float, worker: threading.Thread) -> None:
        """Keep the hold while the job's worker lives. A device that dies stops renewing, and the hold lapses."""
        ended = self._holds[token]
        with tango.EnsureOmniThread():
            while not ended.wait(lease_s / 3.0) and worker.is_alive():
                try:
                    self._instrument().hold(token)
                except tango.DevFailed as exc:  # a step will report the lost hold; keep trying in case it is a passing fault
                    self.warn_stream(f"Could not renew the hold of job {token}: {exc.args[0].desc.strip()[:200] if exc.args else exc}")

    def _take_hold(self, job: dict) -> None:
        token = job["job_id"]
        # step_timeout_s, not CALL_TIMEOUT_S: the first hold may wait behind a direct call still running on the instrument.
        lease_s = float(self.proxy(self.instrument_device_address).hold(token))
        self._holds[token] = threading.Event()
        threading.Thread(target=self._renew_hold, args=(token, lease_s, threading.current_thread()), name=f"hold-{token}", daemon=True).start()

    def _give_back_hold(self, job: dict) -> None:
        token = job["job_id"]
        ended = self._holds.pop(token, None)
        if ended is None:
            return
        ended.set()
        try:
            self._instrument().release_hold(token)
        except tango.DevFailed as exc:  # the lease ends the hold anyway
            self.warn_stream(f"Could not release the hold of job {token}: {exc.args[0].desc.strip()[:200] if exc.args else exc}")

    def _step(self, job: dict, command: str, args=None, file_attrs: dict | None = None, stoppable: bool = True) -> str:
        """Run one step on the instrument and wait for it in short calls; an abort stops it (unless not ``stoppable``)."""
        token = job["job_id"]
        instrument = self._instrument()
        abort = self._abort[token]
        instrument.start_step(json.dumps({"job_token": token, "command": command, "args": args, "file_attrs": file_attrs}))
        delay, stopped = POLL_MIN_S, False
        while True:
            if stoppable and not stopped and abort.is_set():
                instrument.stop_step(token)
                stopped = True
            status = json.loads(instrument.step_status())
            if status["state"] != "running":
                break
            if stopped or not stoppable:
                time.sleep(delay)
            else:
                abort.wait(delay)
            delay = min(delay * 2.0, POLL_MAX_S)
        if status["state"] == "done":
            return status["result"]
        if status["state"] == "stopped" and stopped:
            raise _Stopped()
        raise RuntimeError(f"{command}: {status['error']}")

    def _run_commands(self, job: dict, names: list[str]) -> list[str]:
        problems = []
        for name in names:
            try:
                self._step(job, name, stoppable=False)
            except Exception as exc:  # report, but never mask why the job stopped
                problems.append(f"{name}: {exc}")
        return problems

    # -- the job's life ----------------------------------------------------
    def on_start(self, job: dict) -> None:
        self._take_hold(job)
        for setting, value in self._plan(job).settings.items():
            self._step(job, f"set_{setting}", value, stoppable=False)

    def run_step(self, job: dict, index: int) -> dict | None:
        plan = self._plan(job)
        point = plan.point(index)
        try:
            for name in plan.job.before_move:  # e.g. retract a probe before the stage moves; a refusal fails the job
                self._step(job, name)
            for setting, value in point.items():
                self._step(job, f"set_{setting}", value)
            if plan.job.settle_s:
                self._step(job, SETTLE_STEP, plan.job.settle_s)
            for name in plan.job.after_move:  # e.g. approach the probe again at the new point
                self._step(job, name)
            keys = {}
            for name in plan.job.acquire:
                keys[name] = self._step(
                    job, name, file_attrs={"job": plan.job.name, "job_id": job.get("job_id", ""), "point_index": index, "point": json.dumps(point)},
                )
        except _Stopped:
            return None
        return {"index": index, "point": point, "keys": keys, "key": next(iter(keys.values()), None)}

    def on_stop(self, job: dict, reason: str) -> None:
        if job["job_id"] not in self._holds:  # it never got the instrument: nothing of its to undo
            return
        try:
            problems = self._run_commands(job, self._plan(job).job.on_stop)
        finally:
            self._give_back_hold(job)
        if problems:
            raise RuntimeError(f"on_stop commands failed: {'; '.join(problems)}")

    def on_finish(self, job: dict) -> dict:
        try:
            problems = self._run_commands(job, self._plan(job).job.on_finish)
        finally:
            self._give_back_hold(job)
        return {"on_finish_errors": "; ".join(problems)} if problems else {}

    def summary(self, job: dict) -> dict:
        plan = self._plan(job)
        records = job.get("step_results", {})
        keys: dict[str, list] = {name: [] for name in plan.job.acquire}
        for index in sorted(records, key=int):
            for name, key in (records[index].get("keys") or {}).items():
                keys.setdefault(name, []).append(key)
        shown = {name: values[-KEYS_IN_SUMMARY:] for name, values in keys.items()}
        omitted = {name: len(values) - len(shown[name]) for name, values in keys.items() if len(values) > KEYS_IN_SUMMARY}
        return {"job": plan.job.name, "points": plan.count, "keys": shown, **({"keys_omitted": omitted} if omitted else {})}


_classes: dict[str, type] = {}


def job_class(name: str) -> type:
    """A job device class called ``name``, by convention ``<instrument class name>Job`` (the MCP server names tools after the Tango class)."""
    if name == YamlJob.__name__:
        return YamlJob
    if name not in _classes:
        _classes[name] = type(name, (YamlJob,), {"__module__": __name__})
    return _classes[name]


if __name__ == "__main__":
    job_class(registered_class(sys.argv[1], YamlJob.__name__) if len(sys.argv) > 1 else YamlJob.__name__).run_server()
