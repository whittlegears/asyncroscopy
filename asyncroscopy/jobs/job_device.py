"""Tango base class for jobs that outlive a single command.

A Tango command times out after seconds; a map takes hours. ``JobDevice``
splits a job into the envelope verbs: ``reserve`` checks a plan and returns an
estimate, ``start`` launches a worker thread and returns at once, ``status``
reports progress, ``abort`` stops within one step, ``results`` returns what was
produced and ``release`` frees the device. State is saved after every step,
so a restarted device resumes a job and skips the steps already done.

Subclasses implement ``resolve_plan`` and ``run_step`` (and optionally
``on_start``, ``on_stop`` and ``on_finish``). One reservation is active per
device at a time: the device stands for one instrument.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path

import jsonschema
import tango
from tango import DevState
from tango.server import Device, attribute, command, device_property

from asyncroscopy.contracts import ENVELOPE_VERSION
from asyncroscopy.mcp.tool_log import TraceWriter

ACTIVE_STATES = {"reserved", "queued", "running", "paused_awaiting_approval", "aborting"}
RESUMABLE_STATES = {"reserved", "failed", "aborted"}
SUMMARY_STEPS = 50  # step records results returns by default; the rest are in the job's .steps.jsonl


class JobError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _json_default(value):
    """numpy values stay numbers in replies and state files: np.int64(2) -> 2 and an array (what
    Tango returns for a list of floats) -> a list, not "2" or "[0.1 0.2]"."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


def _has_null(value) -> bool:
    if isinstance(value, dict):
        return any(_has_null(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_null(v) for v in value)
    return value is None


def _kind(value) -> str:
    """JSON kind of a plan value; ints and floats are both numbers."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


class JobDevice(Device):
    # One line saying what this device's job does; MCP tool descriptions start with it.
    JOB_KIND = "background job"
    # A valid plan, returned with every refused plan so a caller (often a small model) can correct itself.
    PLAN_EXAMPLE: dict | None = None
    # Workers still finishing a step when the device was last re-initialised (see delete_device).
    _stale_workers: tuple = ()
    _generation = 0

    job_kind = attribute(dtype=str, doc="What this device's background job does.")
    plan_example = attribute(dtype=str, doc="A valid plan for reserve, as JSON ('' if the device gives none).")
    state_directory = device_property(dtype=str, default_value="outputs/jobs", doc="Where job state files are kept.")
    trace_path = device_property(dtype=str, default_value="outputs/trace/jobs.jsonl", doc="Instrument Trace JSONL file.")
    step_timeout_s = device_property(dtype=float, default_value=120.0, doc="Timeout of proxies made with proxy(): the longest one instrument command may take.")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def resolve_plan(self, plan: dict) -> dict:
        """Validate and expand a plan. Return {"steps": [...], "estimate": {...}, ...}; raise ValueError to refuse."""
        raise NotImplementedError(f"{type(self).__name__} must implement resolve_plan(plan)")

    def run_step(self, job: dict, index: int) -> dict:
        """Do step ``index``; return a JSON-safe record of what it produced, or None if it stopped early for an abort (see aborting), so a resume runs it again."""
        raise NotImplementedError(f"{type(self).__name__} must implement run_step(job, index)")

    def aborting(self, job: dict) -> bool:
        """True once abort was called for ``job``; a long run_step should check it and return None."""
        event = self._abort.get(job.get("job_id"))
        return event is not None and event.is_set()

    def on_start(self, job: dict) -> None:
        """Called in the worker before the first step (including on resume)."""

    def on_stop(self, job: dict, reason: str) -> None:
        """Called in the worker after an abort or a failure; put the instrument in a safe state here."""

    def read_job_kind(self) -> str:
        return self.JOB_KIND

    def read_plan_example(self) -> str:
        return json.dumps(self.PLAN_EXAMPLE) if self.PLAN_EXAMPLE is not None else ""

    def on_finish(self, job: dict) -> dict:
        """Called after the last step; return result keys to merge into ``results``."""
        return {}

    # ------------------------------------------------------------------
    # Tango plumbing
    # ------------------------------------------------------------------
    def init_device(self) -> None:
        Device.init_device(self)
        self._generation += 1  # a worker from an earlier generation no longer writes state files
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._job_index: dict[str, str] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._abort: dict[str, threading.Event] = {}
        self._name = self.get_name()
        self._state_dir = Path(self.state_directory)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._trace = TraceWriter(self.trace_path, source="job")
        self._reload()
        self.set_state(DevState.ON)

    def _reload(self) -> None:
        """Pick up jobs from a previous run of this device; interrupted ones become resumable."""
        for path in sorted(self._state_dir.glob(f"{self._name.replace('/', '_')}__*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("state") in {"queued", "running", "aborting", "paused_awaiting_approval"}:
                job["state"] = "failed"
                job["last_error"] = {"code": "device_restarted", "message": "The device restarted while this job was running; start it again to resume."}
            elif job.get("state") == "reserved":
                # Never started; whoever reserved it is gone. Keeping it would leave the device busy.
                job["state"] = "released"
                job["last_error"] = {"code": "device_restarted", "message": "The device restarted before this reservation was started; reserve again."}
            job["step_results"] = self._read_steps(job["reservation_id"])
            self._jobs[job["reservation_id"]] = job
            for job_id in job.get("job_ids", []):
                self._job_index[job_id] = job["reservation_id"]

    def delete_device(self) -> None:
        """Stop every worker before the device goes away (Init, restart, shutdown).

        Without this, Init would reload the job as resumable while the old
        worker kept driving the instrument, and a resume would run a second
        worker beside it. Each stopping worker runs on_stop (safe state). A
        worker still inside a long step after the wait is remembered, and start
        refuses until it has finished.
        """
        for event in list(getattr(self, "_abort", {}).values()):
            event.set()
        threads = list(getattr(self, "_threads", {}).values()) + list(self._stale_workers)
        deadline = time.time() + 60.0
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.time()))
        self._stale_workers = tuple(t for t in threads if t.is_alive())
        Device.delete_device(self)

    def _path(self, reservation_id: str) -> Path:
        return self._state_dir / f"{self._name.replace('/', '_')}__{reservation_id}.json"

    def _steps_path(self, reservation_id: str) -> Path:
        return self._path(reservation_id).with_suffix(".steps.jsonl")

    def _read_steps(self, reservation_id: str) -> dict:
        path = self._steps_path(reservation_id)
        if not path.exists():
            return {}
        steps = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line cut short by a crash
                continue
            steps[str(row["index"])] = row["record"]
        return steps

    def _save(self, job: dict) -> None:
        """Write the job's small state file. Step results go to an append-only file instead,
        so saving after each of thousands of steps does not rewrite all earlier results."""
        state = {k: v for k, v in job.items() if k != "step_results"}
        tmp = self._path(job["reservation_id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(state, default=_json_default), encoding="utf-8")
        tmp.replace(self._path(job["reservation_id"]))

    def _append_step(self, job: dict, index: int, record: dict) -> None:
        with self._steps_path(job["reservation_id"]).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"index": index, "record": record}, default=_json_default) + "\n")

    @staticmethod
    def _reply(payload: dict) -> str:
        return json.dumps({"schema_version": ENVELOPE_VERSION, **payload}, default=_json_default)

    @staticmethod
    def _error(code: str, message: str) -> str:
        return json.dumps({"schema_version": ENVELOPE_VERSION, "error": {"code": code, "message": message}})

    def _parse(self, text: str, required: str) -> dict:
        try:
            request = json.loads(text) if text.strip().startswith("{") else {required: text.strip()}
        except json.JSONDecodeError as exc:
            raise JobError("bad_request", f"expected a JSON object: {exc}") from exc
        if required not in request:
            raise JobError("bad_request", f"missing '{required}'")
        return request

    def _active(self) -> dict | None:
        for job in self._jobs.values():
            if job["state"] in ACTIVE_STATES:
                return job
        return None

    def _by_job_id(self, job_id: str) -> dict:
        reservation = self._job_index.get(job_id)
        if reservation is None:
            # Accept a reservation id too: callers often only kept that.
            if job_id in self._jobs:
                return self._jobs[job_id]
            raise JobError("unknown_job", f"no job {job_id!r}")
        return self._jobs[reservation]

    @classmethod
    def _plan_caused(cls, exc: Exception, request: str) -> bool:
        """A KeyError for a plan field the plan lacks, or a TypeError with a null or a wrongly typed value
        in the plan, is the caller's; anything else from resolve_plan (a typo'd limits key, arithmetic on
        None) is the device's bug. Plan fields and their types are PLAN_EXAMPLE's when there is one."""
        try:
            plan = json.loads(request).get("plan")
        except (ValueError, AttributeError):
            return True
        if not isinstance(plan, dict):
            return True
        if isinstance(exc, KeyError):
            if not exc.args:
                return True
            known = cls.PLAN_EXAMPLE if isinstance(cls.PLAN_EXAMPLE, dict) else None
            return exc.args[0] not in plan and (known is None or exc.args[0] in known)
        example = cls.PLAN_EXAMPLE if isinstance(cls.PLAN_EXAMPLE, dict) else {}
        return any(_has_null(v) or (k in example and _kind(v) != _kind(example[k])) for k, v in plan.items())

    def _guard(self, fn, *args) -> str:
        try:
            return fn(*args)
        except JobError as exc:
            return self._error(exc.code, str(exc))
        except (ValueError, TypeError, KeyError, jsonschema.ValidationError) as exc:
            if isinstance(exc, (TypeError, KeyError)) and fn != self._reserve:
                raise  # outside a plan check these are bugs, not a caller's mistake
            if isinstance(exc, (TypeError, KeyError)) and not self._plan_caused(exc, args[0] if args else ""):
                example = f" Example plan: {json.dumps(self.PLAN_EXAMPLE)}" if self.PLAN_EXAMPLE is not None else ""
                return self._error("device_error", f"{type(self).__name__}.resolve_plan failed: {type(exc).__name__}: {exc}.{example}")
            message = exc.message if isinstance(exc, jsonschema.ValidationError) else f"{type(exc).__name__}: {exc}" if isinstance(exc, (TypeError, KeyError)) else str(exc)
            if self.PLAN_EXAMPLE is not None:
                message += f". Example plan: {json.dumps(self.PLAN_EXAMPLE)}"
            return self._error("plan_rejected", message)

    # ------------------------------------------------------------------
    # Envelope verbs
    # ------------------------------------------------------------------
    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"plan": {...}, "caller": "name"}. The plan depends on the job; an example is in the tool description.', doc_out="Checks a plan for this device's background job and returns reservation_id with an estimate; nothing moves until start.")
    def reserve(self, request: str) -> str:
        """Validate a plan, hold the device for it, and return the resolved plan with an estimate."""
        return self._guard(self._reserve, request)

    def _reserve(self, request: str) -> str:
        body = self._parse(request, "plan")
        if not isinstance(body["plan"], dict):
            raise JobError("bad_request", "'plan' must be an object")
        with self._lock:
            active = self._active()
            if active is not None:
                raise JobError("busy", f"reservation {active['reservation_id']} is {active['state']}; release or abort it first")
            try:
                resolved = self.resolve_plan(body["plan"])
            except (JobError, ValueError, TypeError, KeyError, jsonschema.ValidationError):
                raise  # sorted out by _guard: the caller's plan, or the device's bug
            except Exception as exc:
                raise JobError("device_error", f"{type(self).__name__}.resolve_plan failed: {type(exc).__name__}: {exc}") from exc
            if not isinstance(resolved, dict) or not isinstance(resolved.get("steps"), list):
                # The device's own bug, not the caller's plan: say so, outside plan_rejected.
                raise JobError("device_error", f"{type(self).__name__}.resolve_plan must return a dict with a 'steps' list, got {type(resolved).__name__}")
            reservation_id = uuid.uuid4().hex[:12]
            job = {
                "reservation_id": reservation_id,
                "plan_id": uuid.uuid4().hex[:12],
                "device": self._name,
                "plan": body["plan"],
                "resolved_plan": {k: v for k, v in resolved.items() if k != "steps"},
                "steps": resolved["steps"],
                "estimate": resolved.get("estimate", {"steps": len(resolved["steps"])}),
                "state": "reserved",
                "done": [],
                "step_results": {},
                "result_keys": {},
                "job_ids": [],
                "caller": body.get("caller"),
                "grant_id": None,
                "last_error": None,
                "created": time.time(),
            }
            self._save(job)  # if this fails, nothing is reserved
            self._jobs[reservation_id] = job
        self._trace.record(f"{type(self).__name__}.reserve", True, time.time(), caller=job["caller"], plan_id=job["plan_id"], run_id=reservation_id, arguments=body["plan"])
        return self._reply(
            {
                "reservation_id": reservation_id,
                "plan_id": job["plan_id"],
                "resolved_plan": job["resolved_plan"],
                "estimate": job["estimate"],
            }
        )

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."} from reserve.', doc_out="The plan, resolved plan, estimate and state of one reservation.")
    def get_reservation(self, request: str) -> str:
        """What a reservation will do; shown to the operator who approves its start."""
        return self._guard(self._get_reservation, request)

    def _get_reservation(self, request: str) -> str:
        reservation_id = self._parse(request, "reservation_id")["reservation_id"]
        with self._lock:
            job = self._jobs.get(reservation_id)
            if job is None:
                raise JobError("unknown_reservation", f"no reservation {reservation_id!r}")
            return self._reply({k: job.get(k) for k in ("reservation_id", "plan_id", "plan", "resolved_plan", "estimate", "state", "caller")})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."} from reserve.', doc_out="Starts this device's background job and returns job_id at once.")
    def start(self, request: str) -> str:
        """Start a reserved job in the background and return its job id at once. Starting an aborted or failed job resumes it."""
        return self._guard(self._start, request)

    def _start(self, request: str) -> str:
        body = self._parse(request, "reservation_id")
        with self._lock:
            job = self._jobs.get(body["reservation_id"])
            if job is None:
                raise JobError("unknown_reservation", f"no reservation {body['reservation_id']!r}")
            if job["state"] not in RESUMABLE_STATES:
                raise JobError("bad_state", f"reservation is {job['state']}")
            active = self._active()
            if active is not None and active is not job:
                raise JobError("busy", f"reservation {active['reservation_id']} is {active['state']}")
            if any(t.is_alive() for t in self._stale_workers):
                raise JobError("bad_state", "a worker from before the device restarted is still finishing its step; try again shortly")
            job_id = uuid.uuid4().hex[:12]
            # Each start records its own approval (none in auto mode); a resume never inherits an old one.
            started = {
                **job, "job_ids": [*job["job_ids"], job_id], "job_id": job_id, "grant_id": body.get("grant_id"),
                "caller": body.get("caller") or job.get("caller"), "state": "running", "last_error": None,
            }
            self._save(started)  # if this fails, the reservation stays as it was and can be started again
            job.update(started)
            self._job_index[job_id] = job["reservation_id"]
            self._abort[job_id] = threading.Event()
            thread = threading.Thread(target=self._omni_worker, args=(job, job_id), name=f"job-{job_id}", daemon=True)
            self._threads[job_id] = thread
            thread.start()
        return self._reply({"job_id": job_id, "reservation_id": job["reservation_id"], "state": "running"})

    def _omni_worker(self, job: dict, job_id: str) -> None:
        # PyTango requires threads it did not create to be registered with omniORB before
        # making Tango calls from inside a device server.
        with tango.EnsureOmniThread():
            self._worker(job, job_id)

    def _worker(self, job: dict, job_id: str) -> None:
        abort = self._abort[job_id]
        generation = self._generation
        cls = type(self).__name__
        try:
            self.on_start(job)
            for index in range(len(job["steps"])):
                if index in job["done"]:
                    continue
                if abort.is_set():
                    break
                started = time.time()
                record = self.run_step(job, index)
                if record is None:
                    if abort.is_set():  # the step stopped early for the abort and did not finish
                        break
                    raise RuntimeError(f"{cls}.run_step returned None for step {index} without an abort; return a record of the step")
                with self._lock:
                    job["done"].append(index)
                    job["step_results"][str(index)] = record
                    if generation == self._generation:
                        self._append_step(job, index, record)
                        self._save(job)
                self._trace.record(
                    f"{cls}.step", True, started, duration_s=time.time() - started, job_id=job_id, plan_id=job["plan_id"],
                    run_id=job["reservation_id"], step=index, grant_id=job.get("grant_id"), caller=job.get("caller"),
                    result=record, result_key=record.get("key") if isinstance(record, dict) else None,
                    stage_reported_m=record.get("stage_reported_m") if isinstance(record, dict) else None,
                )
            if abort.is_set():
                self.on_stop(job, "aborted")
                state = "aborted"
            else:
                # Outside the lock: on_finish may do slow I/O (registering data), and abort,
                # job_status and release must still answer meanwhile.
                finished = self.on_finish(job) or {}
                with self._lock:
                    job["result_keys"].update(finished)
                state = "done"
        except Exception as exc:  # any failure leaves the instrument safe and the job resumable
            job["last_error"] = {"code": "step_failed", "message": f"{type(exc).__name__}: {exc}"}
            self._trace.record(f"{cls}.step", False, time.time(), job_id=job_id, run_id=job["reservation_id"], error=job["last_error"]["message"], result=traceback.format_exc(limit=3))
            try:
                self.on_stop(job, "failed")
            except Exception as stop_exc:  # report, but never mask the original failure
                job["last_error"]["message"] += f"; on_stop also failed: {stop_exc}"
            state = "failed"
        with self._lock:
            job["state"] = state
            job["finished"] = time.time()
            if generation == self._generation:  # after a re-init the reloaded copy owns the file
                self._save(job)
        self._trace.record(f"{cls}.{state}", state == "done", time.time(), job_id=job_id, run_id=job["reservation_id"], plan_id=job["plan_id"], result=job.get("result_keys"))

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Progress of this device's background job: state and steps done of total.")
    def job_status(self, request: str) -> str:
        """Progress of a job: state, steps done of total, last error."""
        return self._guard(self._status, request)

    def _status(self, request: str) -> str:
        job = self._by_job_id(self._parse(request, "job_id")["job_id"])
        return self._reply(
            {
                "job_id": job.get("job_id", job["reservation_id"]),
                "reservation_id": job["reservation_id"],
                "state": job["state"],
                "progress": {"done": len(job["done"]), "total": len(job["steps"])},
                "last_error": job["last_error"],
                "pending_request_id": None,
            }
        )

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Stops this device's background job after the current step (or sooner, where the step checks for it) and puts the instrument in its safe state. Always allowed.")
    def abort(self, request: str) -> str:
        """Stop a running job after the current step and put the instrument in a safe state. Always allowed."""
        return self._guard(self._abort_job, request)

    def _abort_job(self, request: str) -> str:
        job = self._by_job_id(self._parse(request, "job_id")["job_id"])
        job_id = job.get("job_id")
        with self._lock:
            if job["state"] == "running" and job_id in self._abort:
                job["state"] = "aborting"
                self._abort[job_id].set()
            elif job["state"] == "reserved":
                job["state"] = "aborted"
            self._save(job)
        # Wait briefly so a quick step reports "aborted" at once; a long step keeps the state
        # "aborting" (poll job_status) rather than blocking this device past client timeouts.
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout=2.0)
        return self._status(json.dumps({"job_id": job_id or job["reservation_id"]}))

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Tiled keys of what this device's background job produced, and a summary.")
    def results(self, request: str) -> str:
        """Tiled keys of everything the job produced so far, plus a summary."""
        return self._guard(self._results, request)

    def _results(self, request: str) -> str:
        job = self._by_job_id(self._parse(request, "job_id")["job_id"])
        with self._lock:  # the worker adds step results under this lock; summary iterates them
            reply = {
                "job_id": job.get("job_id", job["reservation_id"]),
                "state": job["state"],
                "result_keys": dict(job["result_keys"]),
                "summary": {"steps_done": len(job["done"]), "steps_total": len(job["steps"]), **self.summary(job)},
            }
        return self._reply(reply)

    def summary(self, job: dict) -> dict:
        """Extra fields for ``results``; by default the last steps' records (what run_step returned)."""
        records = job.get("step_results", {})
        last = sorted(records, key=int)[-SUMMARY_STEPS:]
        return {"steps": {i: records[i] for i in last}, "steps_omitted": max(0, len(records) - len(last))}

    def proxy(self, address: str, timeout_s: float | None = None) -> tango.DeviceProxy:
        """A proxy to another device whose commands may take longer than Tango's 3 s default
        (a stage move, an autofocus): use this, not a bare DeviceProxy, inside run_step."""
        proxy = tango.DeviceProxy(address)
        proxy.set_timeout_millis(int(1000 * (timeout_s if timeout_s is not None else self.step_timeout_s)))
        return proxy

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."}.', doc_out="Frees the instrument for the next reservation.")
    def release(self, request: str) -> str:
        """Free the device for the next reservation. A running job must be aborted first."""
        return self._guard(self._release, request)

    def _release(self, request: str) -> str:
        reservation_id = self._parse(request, "reservation_id")["reservation_id"]
        with self._lock:
            job = self._jobs.get(reservation_id)
            if job is None:
                raise JobError("unknown_reservation", f"no reservation {reservation_id!r}")
            if job["state"] in {"running", "aborting", "queued"}:
                raise JobError("bad_state", f"job is {job['state']}; abort it first")
            job["state"] = "released"
            self._save(job)
        return self._reply({"reservation_id": reservation_id, "state": "released"})
