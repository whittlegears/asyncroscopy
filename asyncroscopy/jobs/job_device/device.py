"""JobDevice: the envelope verbs, the subclass hooks and the Tango plumbing."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import jsonschema
from tango import DevState
from tango.server import Device, attribute, device_property

from asyncroscopy.contracts import ENVELOPE_VERSION, load as load_schema
from asyncroscopy.core.names import ACTIVE_JOB_STATES, RUNNING_JOB_STATES
from asyncroscopy.safety.policy import check_overrides
from asyncroscopy.trace.clock import InstrumentClock
from asyncroscopy.trace.writer import TraceWriter
from asyncroscopy.utils.commands import action_hints_attribute, command
from .dry_run import DryRunCalls
from .errors import JobError
from .files import JobFiles, _json_default
from .worker import JobWorker, WORKER, lease_seconds
from .limits import LimitChecks
from .suspenders import SuspenderChecks, parse_suspenders
from .progress import done_count, step_count

STOPPABLE_STATES = {"running", "paused_suspended", "paused"}  # abort sets the worker's abort event
RESUMABLE_STATES = {"reserved", "failed", "aborted"}
SUMMARY_STEPS = 50  # step records results returns by default; the rest are in the job's .steps.jsonl


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


class JobDevice(SuspenderChecks, LimitChecks, JobFiles, JobWorker, DryRunCalls, Device):

    # What each command is (risk, role, stop), as its @command declares it, for the MCP server.
    action_hints = action_hints_attribute()

    # One line saying what this device's job does; MCP tool descriptions start with it.
    JOB_KIND = "background job"

    # A valid plan, returned with every refused plan so a caller (often a small model) can correct itself.
    PLAN_EXAMPLE: dict | None = None

    # The plan's JSON Schema, by name in asyncroscopy/contracts ("map_plan"); None: the device gives none.
    PLAN_SCHEMA: str | None = None

    # True when run_step copes with a dry run: every recorded command returns DRY_RUN_PLACEHOLDER (no
    # image, no reply to parse). False (the default): reserve refuses a dry_run plan, naming the device.
    SUPPORTS_DRY_RUN = False

    # Commands that only make the instrument safer (blank a beam, retract a tip): they really run in a dry run.
    SAFE_COMMANDS: frozenset = frozenset()

    # Workers still finishing a step when the device was last re-initialised (see delete_device).
    _stale_workers: tuple = ()

    _generation = 0

    job_kind = attribute(dtype=str, doc="What this device's background job does.")

    plan_example = attribute(dtype=str, doc="A valid plan for reserve, as JSON ('' if the device gives none).")

    plan_schema = attribute(dtype=str, doc="The plan's JSON Schema, as JSON ('' if the device gives none).")

    state_directory = device_property(dtype=str, default_value="outputs/jobs", doc="Where job state files are kept.")

    trace_path = device_property(dtype=str, default_value="outputs/trace/jobs.jsonl", doc="Instrument Trace JSONL file.")

    step_timeout_s = device_property(dtype=float, default_value=120.0, doc="Timeout of proxies made with proxy(): the longest one instrument command may take.")

    suspenders = device_property(dtype=str, default_value="", doc='JSON list of conditions checked between steps, e.g. [{"attr": "sys/beam/1/current_pa", "min": 10, "resume_after_s": 30}]; a bad one pauses the job until it has been good again for resume_after_s.')

    suspender_poll_s = device_property(dtype=float, default_value=1.0, doc="How often a paused job re-reads its suspender conditions.")

    require_limits = device_property(dtype=bool, default_value=True, doc="Refuse a plan when an attribute it sets has limits that cannot be read (False: reserve it and list them under limits.unchecked).")

    simulated = device_property(dtype=bool, default_value=False, doc="True when the stack drives a simulator (run_servers' top-level simulated:); False, the default, means real hardware.")

    lease_s = device_property(
        dtype=float, default_value=0.0,
        doc="A running job is aborted (and the instrument put in its safe state) when nobody has checked on it (job_status, "
            "wait_for_job, renew_lease) for this long, so a crashed agent or GUI cannot leave it running unwatched. 0: off.",
    )

    pause_limit_s = device_property(
        dtype=float, default_value=0.0,
        doc="A job a person paused is aborted (its instrument already safe) when it has stayed paused this long, so a "
            "forgotten pause does not hold the device for good. 0: no limit.",
    )

    # Subclass hooks
    def resolve_plan(self, plan: dict) -> dict:
        """Validate and expand a plan. Return {"steps": [...], "estimate": {...}, ...}; raise ValueError to refuse.

        A plan of very many steps returns ``"step_count": n`` instead of the list and
        implements ``step_at``, so no step list is held or saved.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement resolve_plan(plan)")

    def step_at(self, job: dict, index: int) -> dict:
        """Step ``index`` of the job; by default from the list resolve_plan returned."""
        return job["steps"][index]

    def phase(self, job: dict) -> str | None:
        """A short description of what the job is doing now, shown by job_status; None for none."""
        return None

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

    def read_plan_schema(self) -> str:
        return json.dumps(load_schema(self.PLAN_SCHEMA)) if self.PLAN_SCHEMA else ""

    def on_finish(self, job: dict) -> dict:
        """Called after the last step; return result keys to merge into ``results``."""
        return {}

    def on_suspend(self, job: dict, reason: str) -> None:
        """Called in the worker when a suspender pauses the job; leave the instrument waiting safely (e.g. beam blanked)."""

    @staticmethod
    def current_step() -> int | None:
        """The step this worker thread is running (None outside a step)."""
        return getattr(WORKER, "step", None)

    def _stack_simulated(self) -> bool:
        """Whether this stack drives a simulator: the ``simulated`` property run_servers sets (False, real hardware, by default)."""
        return bool(getattr(self, "simulated", False))

    def check_dry_call(self, job: dict, device: str, name: str, arguments: Any) -> dict | None:
        """In a dry run, check one recorded call against the plan: {"ok": bool, ...}, or None for no check.
        Default: no check. The Mapper checks every move_stage against the plan box."""
        return None

    def example_plan(self, request: str) -> dict | None:
        """The example plan shown with a refusal; a device may tailor it to the refused request."""
        return self.PLAN_EXAMPLE

    def safe_commands(self, job: dict) -> frozenset:
        """The commands a dry run of ``job`` really runs because they only make the instrument safer (SAFE_COMMANDS)."""
        return self.SAFE_COMMANDS

    def step_targets(self, job: dict) -> list[dict]:
        """What the steps will set, checked against limits at reserve: [{"step": i, "device": address,
        "attribute": name, "value": number}, ...]. A target may carry "min"/"max" for limits the device
        declares some other way; otherwise the attribute's Tango min_value/max_value apply. Default: none."""
        return []

    # Tango plumbing
    def init_device(self) -> None:
        Device.init_device(self)
        self._generation += 1  # a worker from an earlier generation no longer writes state files
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._unloaded: dict[str, Path] = {}  # finished jobs not read yet: reservation id -> progress file
        self._job_index: dict[str, str] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._abort: dict[str, threading.Event] = {}
        self._name = self.get_name()
        self._state_dir = Path(self.state_directory)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._proxies = threading.local()  # one DeviceProxy per (address, timeout) per thread: see proxy()
        self._trace = TraceWriter(
            self.trace_path, source="job", writer=self._name, defaults={"simulated": bool(self.simulated)},
            clock=InstrumentClock(self.trace_clock_device()), may_rotate=self._no_active_job,
        )
        self._suspenders = parse_suspenders(self.suspenders)  # a bad property fails Init with the reason
        self._reload()
        self.set_state(DevState.ON)

    def _no_active_job(self) -> bool:
        """The trace file waits to be rotated while this device holds a job: the job's events are found by their offset
        in the file (``trace_offset``). Runs under the file's lock while other threads may hold self._lock, so it takes none."""
        return not any(job["state"] in ACTIVE_JOB_STATES for job in list(self._jobs.values()))

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
            if job["state"] in ACTIVE_JOB_STATES:
                return job
        return None

    def _by_job_id(self, job_id: str) -> dict:
        # Accept a reservation id too: callers often only kept that.
        job = self._job(self._job_index.get(job_id, job_id))
        if job is None:
            raise JobError("unknown_job", f"no job {job_id!r}")
        return job

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
            if exc.details:
                return json.dumps({"schema_version": ENVELOPE_VERSION, "error": {"code": exc.code, "message": str(exc), **exc.details}}, default=_json_default)
            return self._error(exc.code, str(exc))
        except (ValueError, TypeError, KeyError, jsonschema.ValidationError) as exc:
            if isinstance(exc, (TypeError, KeyError)) and fn not in (self._reserve, self._revise):
                raise  # outside a plan check these are bugs, not a caller's mistake
            if isinstance(exc, (TypeError, KeyError)) and not self._plan_caused(exc, args[0] if args else ""):
                example = f" Example plan: {json.dumps(self.PLAN_EXAMPLE)}" if self.PLAN_EXAMPLE is not None else ""
                return self._error("device_error", f"{type(self).__name__}.resolve_plan failed: {type(exc).__name__}: {exc}.{example}")
            message = exc.message if isinstance(exc, jsonschema.ValidationError) else f"{type(exc).__name__}: {exc}" if isinstance(exc, (TypeError, KeyError)) else str(exc)
            example = self.example_plan(args[0] if args else "")
            if example is not None:
                message += f". Example plan: {json.dumps(example)}"
            return self._error("plan_rejected", message)

    # Envelope verbs
    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"plan": {...}, "caller": "name"}. The plan depends on the job; an example is in the tool description.', doc_out="Checks a plan for this device's background job and returns reservation_id with an estimate; nothing moves until start.", risk="low", role="job.reserve")
    def reserve(self, request: str) -> str:
        """Validate a plan, hold the device for it, and return the resolved plan with an estimate."""
        return self._guard(self._reserve, request)

    def _reserve(self, request: str) -> str:
        body = self._parse(request, "plan")
        try:  # the deployment's risk patterns (mcp.safety.risk), so a dry run records a get_* command marked high
            risk_overrides = check_overrides(body.get("risk_overrides"))
        except (ValueError, AttributeError) as exc:
            raise JobError("bad_request", f"risk_overrides: {exc}") from exc
        self._refuse_if_busy()  # a quick answer for a busy device, before reading the instrument
        plan, dry_run, resolved, shape, limits = self._checked_plan(body["plan"])
        body["plan"] = plan
        with self._lock:
            self._refuse_if_busy()  # another reserve may have won while this one read the instrument
            reservation_id = uuid.uuid4().hex[:12]
            job = {
                "reservation_id": reservation_id,
                "plan_id": uuid.uuid4().hex[:12],
                "device": self._name,
                "plan": body["plan"],
                "resolved_plan": {k: v for k, v in resolved.items() if k not in {"steps", "step_count"}},
                **shape,
                "estimate": resolved.get("estimate", {"steps": step_count(shape)}),
                "limits": limits,
                "state": "reserved",
                "dry_run": dry_run,
                "risk_overrides": risk_overrides,
                "done_ranges": [],
                "step_results": {},
                "result_keys": {},
                "job_ids": [],
                "caller": body.get("caller"),
                "run": body.get("run") or None,  # the agent run (X-Asyncroscopy-Run) the MCP server passes on, for the trace
                "grant_id": None,
                "last_error": None,
                "created": time.time(),
                # Where this job's events start in the shared trace file, so reading them back costs this job, not the history.
                "trace_offset": self._trace.path.stat().st_size if self._trace.path.exists() else 0,
            }
            try:
                self._save_plan(job)
                self._save(job)
            except OSError:  # nothing is reserved: no half-written job is left for a restart to find
                for path in (self._plan_path(reservation_id), self._path(reservation_id)):
                    path.unlink(missing_ok=True)
                raise
            self._jobs[reservation_id] = job
        self._trace.record(f"{type(self).__name__}.reserve", True, time.time(), caller=job["caller"], plan_id=job["plan_id"], run_id=reservation_id, arguments=body["plan"], **self._dry(job))
        return self._reply(
            {
                "reservation_id": reservation_id,
                "plan_id": job["plan_id"],
                "resolved_plan": job["resolved_plan"],
                "estimate": job["estimate"],
                **({"dry_run": True} if dry_run else {}),
                **({"limits": limits} if limits["targets"] else {}),
            }
        )

    def _checked_plan(self, plan) -> tuple[dict, bool, dict, dict, dict]:
        """(plan without dry_run, dry_run, resolved plan, steps or step_count, limits): what reserve and revise check."""
        if not isinstance(plan, dict):
            raise JobError("bad_request", "'plan' must be an object")
        plan = dict(plan)
        dry_run = plan.pop("dry_run", False)  # the envelope's, not the device's: resolve_plan never sees it
        if not isinstance(dry_run, bool):
            raise JobError("bad_request", f"dry_run must be true or false, got {dry_run!r}")
        if "drives" in plan:  # auto mode trusts this list to decide the start; only the device writes it
            raise JobError("bad_request", "drives is not a plan field: the job device works out which commands it drives")
        if dry_run and not getattr(self, "SUPPORTS_DRY_RUN", False):
            raise JobError(
                "dry_run_unsupported", f"{type(self).__name__} does not support dry runs: its steps read replies a dry run "
                "does not produce (SUPPORTS_DRY_RUN is false). Nothing was reserved; rehearse this job on a simulated stack",
            )
        # Resolving the plan and reading limits call other devices (seconds each when one is slow); doing it
        # outside the lock keeps abort, release and a running job's progress from waiting behind it.
        try:
            resolved = self.resolve_plan(plan)
        except (JobError, ValueError, TypeError, KeyError, jsonschema.ValidationError):
            raise  # sorted out by _guard: the caller's plan, or the device's bug
        except Exception as exc:
            raise JobError("device_error", f"{type(self).__name__}.resolve_plan failed: {type(exc).__name__}: {exc}") from exc
        lazy = isinstance(resolved, dict) and isinstance(resolved.get("step_count"), int) and "steps" not in resolved
        if not isinstance(resolved, dict) or not (lazy or isinstance(resolved.get("steps"), list)):
            # The device's own bug, not the caller's plan: say so, outside plan_rejected.
            raise JobError("device_error", f"{type(self).__name__}.resolve_plan must return a dict with a 'steps' list or a 'step_count', got {type(resolved).__name__}")
        shape = {"step_count": resolved["step_count"]} if lazy else {"steps": resolved["steps"]}
        limits = self._check_limits({"plan": plan, "resolved_plan": resolved, **shape})
        return plan, dry_run, resolved, shape, limits

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "...", "plan": {...}}: the whole new plan.', doc_out="Replaces the plan of a reservation that has not started, checked as at reserve; returns the new resolved plan and estimate.", risk="high", operator_only=True, role="other")
    def revise(self, request: str) -> str:
        """Edit a plan before it starts (operator only): the new plan is checked like a reserve and gets a new plan_id."""
        return self._guard(self._revise, request)

    def _revise(self, request: str) -> str:
        body = self._parse(request, "reservation_id")
        if "plan" not in body:
            raise JobError("bad_request", "missing 'plan'")
        reservation_id = body["reservation_id"]
        with self._lock:
            job = self._job(reservation_id)
            if job is None:
                raise JobError("unknown_reservation", f"no reservation {reservation_id!r}")
            if job["state"] != "reserved":
                raise JobError("bad_state", f"reservation is {job['state']}; only a plan that has not started can be revised")
        plan, dry_run, resolved, shape, limits = self._checked_plan(body["plan"])
        with self._lock:
            if job["state"] != "reserved":  # started or released while the new plan was checked
                raise JobError("bad_state", f"reservation is {job['state']}; only a plan that has not started can be revised")
            previous = job["plan_id"]
            job.pop("steps", None)
            job.pop("step_count", None)
            job.update(
                plan=plan, plan_id=uuid.uuid4().hex[:12], resolved_plan={k: v for k, v in resolved.items() if k not in {"steps", "step_count"}},
                **shape, estimate=resolved.get("estimate", {"steps": step_count(shape)}), limits=limits, dry_run=dry_run,
            )
            self._save_plan(job)
            self._save(job)
        self._trace.record(f"{type(self).__name__}.revise", True, time.time(), caller=body.get("caller"), plan_id=job["plan_id"], run_id=reservation_id,
                           arguments={"plan": plan, "replaces_plan_id": previous}, **self._dry(job))
        return self._reply({
            "reservation_id": reservation_id, "plan_id": job["plan_id"], "replaces_plan_id": previous, "resolved_plan": job["resolved_plan"],
            "estimate": job["estimate"], **({"dry_run": True} if dry_run else {}), **({"limits": limits} if limits["targets"] else {}),
        })

    def _refuse_if_busy(self) -> None:
        with self._lock:
            active = self._active()
            if active is not None:
                raise JobError("busy", f"reservation {active['reservation_id']} is {active['state']}; release or abort it first")

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."} from reserve.', doc_out="The plan, resolved plan, estimate and state of one reservation.", risk="low", role="job.status")
    def get_reservation(self, request: str) -> str:
        """What a reservation will do; shown to the operator who approves its start."""
        return self._guard(self._get_reservation, request)

    def _get_reservation(self, request: str) -> str:
        reservation_id = self._parse(request, "reservation_id")["reservation_id"]
        with self._lock:
            job = self._job(reservation_id)
            if job is None:
                raise JobError("unknown_reservation", f"no reservation {reservation_id!r}")
            # dry_run comes first: the operator approving the start must see whether anything will move.
            return self._reply({"dry_run": bool(job.get("dry_run")), **{k: job.get(k) for k in ("reservation_id", "plan_id", "plan", "resolved_plan", "estimate", "state", "caller")}})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."} from reserve.', doc_out="Starts this device's background job and returns job_id at once.", risk="high", role="job.start")
    def start(self, request: str) -> str:
        """Start a reserved job in the background and return its job id at once. Starting an aborted or failed job resumes it."""
        return self._guard(self._start, request)

    def _start(self, request: str) -> str:
        body = self._parse(request, "reservation_id")
        with self._lock:
            job = self._job(body["reservation_id"])
            if job is None:
                raise JobError("unknown_reservation", f"no reservation {body['reservation_id']!r}")
            if "dry_run" in body and bool(body["dry_run"]) != bool(job.get("dry_run")):
                # A start never turns a rehearsal into a real run (or back): the approved plan decides.
                raise JobError("bad_request", f"this reservation is {'a dry run' if job.get('dry_run') else 'a real run'}; dry_run is set in the plan at reserve, not at start")
            pausing = job.get("pause_requested") and job["state"] in ("running", "paused", "paused_suspended")
            if pausing and not body.get("continue"):  # an agent's start or resume must not undo a person's pause
                raise JobError("paused", "a person paused this job; it goes on only when they continue it (start with continue: true)")
            if pausing:
                return self._continue(job, body)
            if body.get("plan_id") and body["plan_id"] != job.get("plan_id"):
                raise JobError("plan_changed", f"the plan was revised (now {job.get('plan_id')}) after the start was approved for {body['plan_id']}; ask again")
            if body.get("continue"):  # a person pressed Continue on a paused job: never resume one stopped since
                raise JobError("bad_state", f"job is {job['state']}, not paused; it was not continued")
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
                "caller": body.get("caller") or job.get("caller"), "run": body.get("run") or job.get("run"), "state": "running", "last_error": None, "suspended": None,
                "pause_requested": False,
            }
            previous = dict(job)
            self._save(started)  # if this fails, the reservation stays as it was and can be started again
            job.update(started)
            self._job_index[job_id] = job["reservation_id"]
            for finished in [j for j, t in self._threads.items() if not t.is_alive()]:  # a long-lived server keeps no dead workers
                self._threads.pop(finished, None)
                self._abort.pop(finished, None)
            self._abort[job_id] = threading.Event()
            job["lease_until"] = time.time() + lease_seconds(self) if lease_seconds(self) > 0 else None
            thread = threading.Thread(target=self._omni_worker, args=(job, job_id), name=f"job-{job_id}", daemon=True)
            self._threads[job_id] = thread
            try:
                thread.start()
                if job["lease_until"] is not None:
                    threading.Thread(target=self._watch_lease, args=(job, job_id, thread), name=f"lease-{job_id}", daemon=True).start()
            except Exception:  # no thread to run it (RuntimeError, MemoryError): put the reservation back as it was
                for table in (self._job_index, self._abort, self._threads):
                    table.pop(job_id, None)
                job.clear()
                job.update(previous)
                self._save(job)
                raise
        return self._reply({"job_id": job_id, "reservation_id": job["reservation_id"], "state": "running"})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Pauses this device's background job after the current step with the instrument waiting safely; start continues it. Always allowed.", risk="low", role="safe_stop")
    def pause(self, request: str) -> str:
        """Pause a running job after the current step; the device stays held. start continues it, abort stops it."""
        return self._guard(self._pause, request)

    def _pause(self, request: str) -> str:
        body = self._parse(request, "job_id")
        job = self._by_job_id(body["job_id"])
        with self._lock:
            if job["state"] not in ("running", "paused_suspended", "paused"):
                raise JobError("bad_state", f"job is {job['state']}; only a running job can be paused")
            first = not job.get("pause_requested")
            job["pause_requested"] = True
            self._save(job)
        if first:
            self._trace.record(f"{type(self).__name__}.pause", True, time.time(), job_id=job.get("job_id"), run_id=job["reservation_id"], caller=body.get("caller"), **self._dry(job))
        return self._status(json.dumps({"job_id": job.get("job_id") or job["reservation_id"]}))

    @command(dtype_out=str, doc_out='JSON {"jobs": [...]}: every job that holds this device or is running, paused or aborting, with its progress.', risk="low", role="setting.read")
    def get_jobs(self) -> str:
        """The jobs a supervisor can act on now: reserved, running, paused or aborting. Reading this renews no lease."""
        with self._lock:
            jobs = [
                {
                    "job_id": job.get("job_id"), "reservation_id": job["reservation_id"], "plan_id": job.get("plan_id"), "state": job["state"],
                    "pause_requested": bool(job.get("pause_requested")), "progress": {"done": done_count(job), "total": step_count(job)},
                    "caller": job.get("caller"), "dry_run": bool(job.get("dry_run")), "last_error": job.get("last_error"),
                    "suspended": job.get("suspended"), "lease_until": job.get("lease_until"),
                }
                for job in self._jobs.values() if job["state"] in ACTIVE_JOB_STATES
            ]
        return self._reply({"jobs": jobs, "lease_s": lease_seconds(self)})

    def trace_clock_device(self) -> str:
        """The device whose clock_s stamps this job device's trace events; by default its microscope_device_address
        when it has one. Step events use the clock_s their record reports. "" for none (this computer's time)."""
        return str(getattr(self, "microscope_device_address", "") or "")

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Progress of this device's background job: state and steps done of total.", risk="low", role="job.status")
    def job_status(self, request: str) -> str:
        """Progress of a job: state, steps done of total, last error."""
        return self._guard(self._status, request)

    def _status(self, request: str) -> str:
        job = self._by_job_id(self._parse(request, "job_id")["job_id"])
        with self._lock:  # the worker changes done_ranges under this lock
            done = done_count(job)
            self._renew(job)
        return self._reply(
            {
                "job_id": job.get("job_id", job["reservation_id"]),
                "reservation_id": job["reservation_id"],
                "state": job["state"],
                "progress": {"done": done, "total": step_count(job)},
                "last_error": job["last_error"],
                "pending_request_id": None,
                "suspended": job.get("suspended"),
                **({"pause_requested": True} if job.get("pause_requested") else {}),
                **({"phase": phase} if (phase := getattr(self, "phase", lambda _job: None)(job)) else {}),
            }
        )

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Stops this device's background job after the current step (or sooner, where the step checks for it) and puts the instrument in its safe state. Always allowed.", risk="low", role="job.abort")
    def abort(self, request: str) -> str:
        """Stop a running job after the current step and put the instrument in a safe state. Always allowed."""
        return self._guard(self._abort_job, request)

    def _abort_job(self, request: str) -> str:
        job = self._by_job_id(self._parse(request, "job_id")["job_id"])
        job_id = job.get("job_id")
        with self._lock:
            job["pause_requested"] = False
            if job["state"] in STOPPABLE_STATES and job_id in self._abort:
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

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."}.', doc_out="The job's status; its lease runs lease_s from now.", risk="low", role="other")
    def renew_lease(self, request: str) -> str:
        """Keep a running job's lease (lease_s) from lapsing without reading anything else. job_status renews it too."""
        return self._guard(self._status, request)

    @command(dtype_out=str, doc_out='JSON {"aborted": [status of each job that was running]}.', risk="low", role="safe_stop", stop=True)
    def abort_all(self) -> str:
        """Stop every job of this device that is running, without naming it: the e-stop. Always allowed."""
        with self._lock:
            running = [job["job_id"] for job in self._jobs.values() if job.get("job_id") and job["state"] in STOPPABLE_STATES | {"aborting"}]
        stopped = []
        for job_id in running:
            try:
                stopped.append(json.loads(self._abort_job(json.dumps({"job_id": job_id}))))
            except Exception as exc:  # one job that cannot be stopped must not keep the others running
                stopped.append({"job_id": job_id, "error": f"{type(exc).__name__}: {exc}"})
        self._trace.record(f"{type(self).__name__}.abort_all", True, time.time(), arguments={"job_ids": running})
        return self._reply({"aborted": stopped})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Tiled keys of what this device's background job produced, and a summary.", risk="low", role="job.results")
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
                "summary": {"steps_done": done_count(job), "steps_total": step_count(job), **self.summary(job)},
            }
            if job.get("dry_run"):
                reply["dry_run"] = {"recorded": 0, "performed": 0, "failed_checks": 0, **job.get("dry_run_calls", {})}
        return self._reply(reply)

    def summary(self, job: dict) -> dict:
        """Extra fields for ``results``; by default the last steps' records (what run_step returned)."""
        records = job.get("step_results", {})
        last = sorted(records, key=int)[-SUMMARY_STEPS:]
        return {"steps": {i: records[i] for i in last}, "steps_omitted": max(0, len(records) - len(last))}

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."}.', doc_out="Frees the instrument for the next reservation.", risk="low", role="job.release")
    def release(self, request: str) -> str:
        """Free the device for the next reservation. A running job must be aborted first."""
        return self._guard(self._release, request)

    def _release(self, request: str) -> str:
        reservation_id = self._parse(request, "reservation_id")["reservation_id"]
        with self._lock:
            job = self._job(reservation_id)
            if job is None:
                raise JobError("unknown_reservation", f"no reservation {reservation_id!r}")
            if job["state"] in RUNNING_JOB_STATES:
                raise JobError("bad_state", f"job is {job['state']}; abort it first")
            job["state"] = "released"
            self._save(job)
        return self._reply({"reservation_id": reservation_id, "state": "released"})
