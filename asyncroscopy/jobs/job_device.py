"""Tango base class for jobs that outlive a single command.

A Tango command times out after seconds; a map takes hours. ``JobDevice``
splits a job into the envelope verbs: ``reserve`` checks a plan and returns an
estimate, ``start`` launches a worker thread and returns at once, ``status``
reports progress, ``abort`` stops within one step, ``results`` returns what was
produced and ``release`` frees the device. State is saved after every step,
so a restarted device resumes a job and skips the steps already done.

A job is kept as three files in ``state_directory``: the plan (``<device>__<id>.plan.json``: the plan,
its resolution, estimate and limits), written once at reserve (again only by ``revise``, before the job
starts); a small progress record (``<device>__<id>.json``: state, done ranges, result keys, the hooks'
own progress) rewritten after every step; and the step results, appended one line per step
(``.steps.jsonl``). At Init only the jobs that are active or can be resumed are loaded; a finished one
(done or released) is opened from its files when it is asked for by id.

Subclasses implement ``resolve_plan`` and ``run_step`` (and optionally
``on_start``, ``on_stop``, ``on_finish``, ``on_suspend`` and ``step_targets``).
One reservation is active per device at a time: the device stands for one
instrument.

Two checks borrowed from Bluesky (its ``check_limits`` and suspenders), written
for Tango:

* ``reserve`` checks every value ``step_targets`` says a step will set against
  that attribute's Tango ``min_value``/``max_value`` (or limits the target
  carries), and refuses the plan with a list of violations before anything moves.
* The ``suspenders`` device property lists conditions on other devices'
  attributes, e.g. ``[{"attr": "sys/beam/1/current_pa", "min": 10,
  "resume_after_s": 30}]``. Between steps, a condition gone bad pauses the job
  (state ``paused_suspended``, with the reason in ``job_status``); it resumes by
  itself once every condition has been good for ``resume_after_s``. Abort works
  while paused.

A person supervising a job can ``pause`` it: after the current step the device
calls ``on_suspend`` (the instrument waits safely, e.g. beam blanked) and the job
waits in state ``paused`` holding the device. ``start`` on a paused job continues
it, through the same approval as any start; ``abort`` stops it. Before a reserved
plan starts, an operator can ``revise`` it: the plan is checked again as at
reserve and gets a new plan_id. ``get_jobs`` lists the jobs a supervisor can act on.

Dry run (decision D3 of docs/designs/concepts_brief.md): a plan with ``"dry_run": true``
is reserved, approved and started like any other, but inside its worker ``proxy()``
returns a ``RecordingProxy``: reads pass through, the device's ``safe_commands`` (making
the instrument safer, such as blanking the beam) really run, and every
other command or attribute write is recorded in the trace (``dry_run: true``), checked
by ``check_dry_call`` and not performed. ``dry_run`` is part of the plan, so it is fixed
at reserve, shown to the operator who approves the start, and kept on a resume. Only a device that
sets ``SUPPORTS_DRY_RUN`` takes one: its steps must cope with the placeholder a recorded command
returns. Any other device refuses ``dry_run`` at reserve, naming itself. The reserve request may carry
``risk_overrides`` (the MCP server passes its ``mcp.safety.risk``); a command they mark high is
recorded in a dry run even when its name says it only reads (an SPM ``get_image`` scans the tip).
"""

from __future__ import annotations

import bisect
import copy
import json
import threading
from collections import deque
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import jsonschema
import tango
from tango import DevState
from tango.server import Device, attribute, command, device_property

from asyncroscopy.contracts import ENVELOPE_VERSION, load as load_schema
from asyncroscopy.jobs.recording_proxy import RecordingProxy
from asyncroscopy.trace.writer import TraceWriter
from asyncroscopy.safety.policy import check_overrides
from asyncroscopy.trace.clock import InstrumentClock

ACTIVE_STATES = {"reserved", "queued", "running", "paused_awaiting_approval", "paused_suspended", "paused", "aborting"}
RUNNING_STATES = {"queued", "running", "aborting", "paused_awaiting_approval", "paused_suspended", "paused"}  # a worker owns the job
STOPPABLE_STATES = {"running", "paused_suspended", "paused"}  # abort sets the worker's abort event
PAUSE_POLL_S = 0.2


def lease_seconds(device: object) -> float:
    """A job device's lease_s (0: no lease), also for a stand-in that does not declare the property."""
    return float(getattr(device, "lease_s", 0.0) or 0.0)
SUSPENDER_KEYS = {"attr", "min", "max", "equals", "resume_after_s", "label"}
VIOLATIONS_SHOWN = 20  # violations listed in a refusal; the total is always given
RESUMABLE_STATES = {"reserved", "failed", "aborted"}
FINISHED_STATES = {"done", "released"}  # left on disk at Init and opened when asked for by id
# What reserve (or revise, before a start) fixes: written once to the job's plan file, never with its progress.
PLAN_KEYS = frozenset({"plan_id", "device", "plan", "resolved_plan", "steps", "step_count", "estimate", "limits", "dry_run", "risk_overrides", "created", "trace_offset"})
SUMMARY_STEPS = 50  # step records results returns by default; the rest are in the job's .steps.jsonl
# Step records kept in memory (the newest); all of them stay in the job's .steps.jsonl. Bounds memory
# for jobs of millions of steps; a device whose summary needs every record keeps its jobs below this.
KEPT_STEP_RESULTS = 10_000
# The job (and step) the current worker thread runs, so proxy() knows a dry run. Per thread, and each worker
# thread runs one job, so one object serves every device; _worker sets it, so it needs nothing from init_device.
WORKER = threading.local()


def step_count(job: dict) -> int:
    """How many steps the job has: a list from resolve_plan, or a step_count with step_at."""
    return int(job["step_count"]) if "step_count" in job else len(job["steps"])


def read_last_steps(path: Path, count: int) -> dict:
    """The last ``count`` step records of a job's .steps.jsonl, read from the end of the file.

    Start-up cost does not grow with the job's history: a job of a million steps reads only
    its tail. A line cut short by a crash is skipped.
    """
    if not path.exists():
        return {}
    newest: deque = deque(maxlen=count)
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = position = handle.tell()
        block = 1 << 16
        data = b""
        while position > 0 and data.count(b"\n") <= count:
            position = max(0, position - block)
            handle.seek(position)
            data = handle.read(end - position)
        lines = data.split(b"\n")
        if position > 0:
            lines = lines[1:]  # the first line may start before the block
    for line in lines[-(count + 1):]:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        newest.append((str(row["index"]), row["record"]))
    return dict(newest)


def done_count(job: dict) -> int:
    """Steps done, from the job's sorted [start, end) ranges."""
    return sum(end - start for start, end in job["done_ranges"])


def is_done(job: dict, index: int) -> bool:
    ranges = job["done_ranges"]
    k = bisect.bisect_right(ranges, [index, float("inf")]) - 1
    return k >= 0 and ranges[k][0] <= index < ranges[k][1]


def mark_done(job: dict, index: int) -> None:
    """Add ``index``, merging ranges, so a job of a million steps stores a few numbers, not a million."""
    ranges = job["done_ranges"]
    if is_done(job, index):
        return
    k = bisect.bisect_right(ranges, [index, float("inf")])
    joins_left = k > 0 and ranges[k - 1][1] == index
    joins_right = k < len(ranges) and ranges[k][0] == index + 1
    if joins_left and joins_right:
        ranges[k - 1][1] = ranges[k][1]
        del ranges[k]
    elif joins_left:
        ranges[k - 1][1] = index + 1
    elif joins_right:
        ranges[k][0] = index
    else:
        ranges.insert(k, [index, index + 1])


def done_ranges_from(job: dict) -> list[list[int]]:
    """Ranges from a saved job: new files store done_ranges, older ones a list of indices."""
    if "done_ranges" in job:
        return [[int(a), int(b)] for a, b in job["done_ranges"]]
    rebuilt = {"done_ranges": []}
    for index in job.get("done", []):
        mark_done(rebuilt, int(index))
    return rebuilt["done_ranges"]


class JobError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details  # extra fields of the error object, e.g. violations


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


def parse_suspenders(text: str) -> list[dict]:
    """The ``suspenders`` property: a JSON list of conditions, each {"attr": "<device>/<attribute>",
    "min" and/or "max" (numbers) or "equals", "resume_after_s" (default 10), "label" (optional)}."""
    if not text or not text.strip():
        return []
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"suspenders must be a JSON list: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("suspenders must be a JSON list of conditions")
    out = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("attr"), str) or item["attr"].count("/") < 3:
            raise ValueError(f"each suspender needs 'attr' as '<domain>/<family>/<member>/<attribute>', got {item!r}")
        unknown = set(item) - SUSPENDER_KEYS
        if unknown:
            raise ValueError(f"suspender {item['attr']}: unknown keys {sorted(unknown)}; allowed: {sorted(SUSPENDER_KEYS)}")
        if not any(k in item for k in ("min", "max", "equals")):
            raise ValueError(f"suspender {item['attr']}: give min, max or equals")
        for key in ("min", "max", "resume_after_s"):
            if key in item and (isinstance(item[key], bool) or not isinstance(item[key], (int, float))):
                raise ValueError(f"suspender {item['attr']}: {key} must be a number")
        out.append({"resume_after_s": 10.0, "label": item["attr"], **item})
    return out


def suspender_problem(suspender: dict, value: Any) -> str | None:
    """Why ``value`` trips ``suspender``, or None when it is fine."""
    label = suspender["label"]
    if "equals" in suspender and value != suspender["equals"]:
        return f"{label} is {value!r}, needs {suspender['equals']!r}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        if "min" in suspender or "max" in suspender:
            return f"{label} is {value!r}, not a number"
        return None
    if "min" in suspender and not number >= suspender["min"]:  # "not >=" also catches NaN
        return f"{label} is {number:g}, below {suspender['min']:g}"
    if "max" in suspender and not number <= suspender["max"]:
        return f"{label} is {number:g}, above {suspender['max']:g}"
    return None


def _limit(value) -> float | None:
    """A Tango attribute config bound as a number; None for "Not specified"."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Tango plumbing
    # ------------------------------------------------------------------
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
            clock=InstrumentClock(self.trace_clock_device()),
        )
        self._suspenders = parse_suspenders(self.suspenders)  # a bad property fails Init with the reason
        self._reload()
        self.set_state(DevState.ON)

    def _reload(self) -> None:
        """Pick up jobs from a previous run of this device; interrupted ones become resumable, finished ones stay on disk."""
        for path in sorted(self._state_dir.glob(f"{self._name.replace('/', '_')}__*.json")):
            if path.name.endswith(".plan.json"):
                continue
            try:
                progress = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.warn_stream(f"Skipping job state file {path}: {exc}")
                continue
            reservation_id = progress["reservation_id"]
            for job_id in progress.get("job_ids", []):
                self._job_index[job_id] = reservation_id
            if progress.get("state") in FINISHED_STATES:
                self._unloaded[reservation_id] = path
                continue
            job = self._open(reservation_id, progress)
            if job is None:
                continue
            job.pop("pause_requested", None)
            if job.get("state") in RUNNING_STATES:
                job["state"] = "failed"
                job["last_error"] = {"code": "device_restarted", "message": "The device restarted while this job was running; start it again to resume."}
            elif job.get("state") == "reserved":
                # Never started; whoever reserved it is gone. Keeping it would leave the device busy.
                job["state"] = "released"
                job["last_error"] = {"code": "device_restarted", "message": "The device restarted before this reservation was started; reserve again."}
            self._jobs[reservation_id] = job

    def _open(self, reservation_id: str, progress: dict | None = None) -> dict | None:
        """A job from its files: the plan, the progress record and the last step results; None if they cannot be read."""
        try:
            if progress is None:
                progress = json.loads(self._path(reservation_id).read_text(encoding="utf-8"))
            plan_path = self._plan_path(reservation_id)
            if plan_path.exists():
                job = {**json.loads(plan_path.read_text(encoding="utf-8")), **progress}
            else:  # a job saved whole, before plans had a file of their own: its plan is written out now
                job = dict(progress)
                self._save_plan(job)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            self.warn_stream(f"Skipping job {reservation_id}: {exc}")
            return None
        job["done_ranges"] = done_ranges_from(job)
        job.pop("done", None)
        job["step_results"] = self._read_steps(reservation_id)
        return job

    def _job(self, reservation_id: str) -> dict | None:
        """The job with this reservation id, opened from disk if it finished before this device started; None if unknown."""
        job = self._jobs.get(reservation_id)
        if job is None and reservation_id in self._unloaded:
            job = self._open(reservation_id)
            if job is not None:
                self._jobs[reservation_id] = job
                del self._unloaded[reservation_id]
        return job

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

    def _plan_path(self, reservation_id: str) -> Path:
        return self._path(reservation_id).with_suffix(".plan.json")

    def _steps_path(self, reservation_id: str) -> Path:
        return self._path(reservation_id).with_suffix(".steps.jsonl")

    def _read_steps(self, reservation_id: str) -> dict:
        return read_last_steps(self._steps_path(reservation_id), KEPT_STEP_RESULTS)

    def _save(self, job: dict) -> None:
        """Write the job's small progress record. The plan has its own file (_save_plan), written once, and
        step results go to an append-only file, so saving after each of thousands of steps rewrites neither."""
        path = self._path(job["reservation_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._snapshot(job), default=_json_default), encoding="utf-8")
        tmp.replace(path)

    def _save_plan(self, job: dict) -> None:
        """Write the job's plan file: what reserve (or revise) fixed."""
        path = self._plan_path(job["reservation_id"])
        tmp = path.with_suffix(".tmp")
        fixed = {"reservation_id": job["reservation_id"], **{k: v for k, v in job.items() if k in PLAN_KEYS}}
        tmp.write_text(json.dumps(fixed, default=_json_default), encoding="utf-8")
        tmp.replace(path)

    def _snapshot(self, job: dict) -> dict:
        """A deep copy of the job's progress (without its plan and step results), taken under the lock. Hooks (live
        drift, the Mapper's pipeline) change parts of the job outside the lock; a copy taken while one does can fail
        with "changed size during iteration", so it is tried again rather than losing the save."""
        with self._lock:
            for attempt in range(5):
                try:
                    return copy.deepcopy({k: v for k, v in job.items() if k != "step_results" and k not in PLAN_KEYS})
                except RuntimeError:
                    if attempt == 4:
                        raise
                    time.sleep(0.001)
        raise AssertionError("unreachable")

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

    # ------------------------------------------------------------------
    # Envelope verbs
    # ------------------------------------------------------------------
    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"plan": {...}, "caller": "name"}. The plan depends on the job; an example is in the tool description.', doc_out="Checks a plan for this device's background job and returns reservation_id with an estimate; nothing moves until start.")
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

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "...", "plan": {...}}: the whole new plan.', doc_out="Replaces the plan of a reservation that has not started, checked as at reserve; returns the new resolved plan and estimate.")
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

    def _check_limits(self, job: dict) -> dict:
        """Check what step_targets says the steps will set; raise plan_rejected listing every violation."""
        try:
            targets = list(self.step_targets(job) or [])
        except Exception as exc:
            raise JobError("device_error", f"{type(self).__name__}.step_targets failed: {type(exc).__name__}: {exc}") from exc
        found: dict[tuple[str, str], tuple | str] = {}
        violations, unchecked = [], {}
        for target in targets:
            key = (target["device"], target["attribute"])
            if "min" in target or "max" in target:
                low, high = _limit(target.get("min")), _limit(target.get("max"))
            else:
                if key not in found:
                    found[key] = self._attribute_limits(*key)
                if isinstance(found[key], str):
                    unchecked[f"{key[0]}/{key[1]}"] = found[key]
                    continue
                low, high = found[key]
            value = float(target["value"])
            if (low is not None and value < low) or (high is not None and value > high):
                violations.append({"step": target["step"], "device": key[0], "attribute": key[1], "value": value, "min": low, "max": high})
        if unchecked and getattr(self, "require_limits", True):
            shown = "; ".join(f"{k} ({r})" for k, r in list(unchecked.items())[:3])
            raise JobError(
                "plan_rejected", f"the limits of {len(unchecked)} attribute(s) the plan sets could not be read, so the plan cannot be checked: {shown}. Nothing moved.",
                unchecked=[{"target": k, "reason": r} for k, r in unchecked.items()][:VIOLATIONS_SHOWN],
            )
        if violations:
            shown = "; ".join(
                f"step {v['step']} sets {v['device']}/{v['attribute']} to {v['value']:g}, outside [{v['min']}, {v['max']}]" for v in violations[:3]
            )
            more = f" (and {len(violations) - 3} more)" if len(violations) > 3 else ""
            raise JobError(
                "plan_rejected", f"{len(violations)} step(s) outside the instrument's limits: {shown}{more}. Nothing moved.",
                violations=violations[:VIOLATIONS_SHOWN], violations_total=len(violations),
            )
        return {"targets": len(targets), "unchecked": [{"target": k, "reason": r} for k, r in unchecked.items()]}

    def _attribute_limits(self, device: str, attribute: str) -> tuple[float | None, float | None] | str:
        """(min, max) from the attribute's Tango config, or why they could not be read."""
        try:
            config = self.proxy(device, timeout_s=3.0).get_attribute_config(attribute)
        except Exception as exc:  # DevFailed: device down or no such attribute
            return f"could not read its limits: {exc}".splitlines()[0][:200]
        return _limit(config.min_value), _limit(config.max_value)

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."} from reserve.', doc_out="The plan, resolved plan, estimate and state of one reservation.")
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

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."} from reserve.', doc_out="Starts this device's background job and returns job_id at once.")
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

    def _continue(self, job: dict, body: dict) -> str:
        """Let a paused job go on (called by start, under the lock): the worker leaves its pause at the next poll."""
        job.update(pause_requested=False, grant_id=body.get("grant_id"), caller=body.get("caller") or job.get("caller"))
        job["lease_until"] = time.time() + lease_seconds(self) if lease_seconds(self) > 0 else None
        self._save(job)
        self._trace.record(f"{type(self).__name__}.continue", True, time.time(), job_id=job.get("job_id"), run_id=job["reservation_id"],
                           caller=job.get("caller"), grant_id=job.get("grant_id"), **self._dry(job))
        return self._reply({"job_id": job.get("job_id"), "reservation_id": job["reservation_id"], "state": "running", "continued": True})

    def _hold_while_paused(self, job: dict, job_id: str, abort: threading.Event) -> str:
        """Wait while a person has paused the job, with the instrument made safe: "running" (it was not paused),
        "continued" (it was, and they continued it) or "aborted"."""
        with self._lock:
            if not job.get("pause_requested"):
                return "running"
        cls = type(self).__name__
        unsafe = None
        try:  # make the instrument safe before reporting the pause
            self.on_suspend(job, "paused by a person")
        except Exception as exc:
            unsafe = f"{type(exc).__name__}: {exc}"
            self._trace.record(f"{cls}.on_suspend", False, time.time(), job_id=job_id, run_id=job["reservation_id"], error=unsafe, **self._dry(job))
        with self._lock:
            if job["state"] not in ("running", "paused"):  # aborting already
                return "aborted"
            job["state"] = "paused"
            if unsafe:  # a pause that could not make the instrument safe must not look safe
                job["last_error"] = f"paused, but the instrument was not made safe: {unsafe}"
            job["paused_since"] = time.time()
            self._save(job)
        self._trace.record(f"{cls}.paused", True, time.time(), job_id=job_id, run_id=job["reservation_id"], **self._dry(job))
        limit = float(getattr(self, "pause_limit_s", 0.0) or 0.0)
        while not abort.wait(PAUSE_POLL_S):
            with self._lock:
                if limit > 0 and job["state"] == "paused" and job.get("pause_requested") and time.time() - job["paused_since"] > limit:
                    job["last_error"] = {"code": "pause_lapsed", "message": f"paused for longer than pause_limit_s ({limit:g} s), so it was aborted; start it again to resume"}
                    job["state"] = "aborting"
                    job["pause_requested"] = False
                    abort.set()
                    lapsed = True
                else:
                    lapsed = False
            if lapsed:
                self._trace.record(f"{cls}.pause_lapsed", True, time.time(), job_id=job_id, run_id=job["reservation_id"], **self._dry(job))
                return "aborted"
            with self._lock:
                if not job.get("pause_requested"):
                    if job["state"] != "paused":
                        return "aborted"
                    job["state"] = "running"
                    job.pop("paused_since", None)
                    self._save(job)
                    return "continued"
        return "aborted"

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Pauses this device's background job after the current step with the instrument waiting safely; start continues it. Always allowed.")
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

    @command(dtype_out=str, doc_out='JSON {"jobs": [...]}: every job that holds this device or is running, paused or aborting, with its progress.')
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
                for job in self._jobs.values() if job["state"] in ACTIVE_STATES
            ]
        return self._reply({"jobs": jobs, "lease_s": lease_seconds(self)})

    def _watch_lease(self, job: dict, job_id: str, worker: threading.Thread) -> None:
        """Abort ``job_id`` once its lease lapses: nobody has asked about it for lease_s."""
        lease = lease_seconds(self)
        abort = self._abort.get(job_id)
        while worker.is_alive() and abort is not None and not abort.is_set():
            time.sleep(min(1.0, lease / 4))
            with self._lock:
                lapsed = job.get("job_id") == job_id and job["state"] == "running" and time.time() > (job.get("lease_until") or float("inf"))
                if lapsed:
                    job["last_error"] = {"code": "lease_lapsed", "message": f"lease lapsed: nobody checked on this job for {lease:g} s, so it was aborted and the instrument put in its safe state"}
                    job["state"] = "aborting"
                    abort.set()
            if lapsed:
                self._trace.record(f"{type(self).__name__}.lease_lapsed", True, time.time(), job_id=job_id, run_id=job["reservation_id"], caller=job.get("caller"), **self._dry(job))
                return

    def _renew(self, job: dict) -> None:
        if lease_seconds(self) > 0 and job["state"] in RUNNING_STATES:  # a paused job someone is watching keeps its lease
            job["lease_until"] = time.time() + lease_seconds(self)

    def _omni_worker(self, job: dict, job_id: str) -> None:
        # PyTango requires threads it did not create to be registered with omniORB before
        # making Tango calls from inside a device server.
        with tango.EnsureOmniThread():
            self._worker(job, job_id)

    @staticmethod
    def _dry(job: dict) -> dict:
        """Trace fields every event of ``job`` carries: dry_run, and the agent run that started it (if known)."""
        return {"dry_run": bool(job.get("dry_run")), "run": job.get("run")}

    def trace_clock_device(self) -> str:
        """The device whose clock_s stamps this job device's trace events; by default its microscope_device_address
        when it has one. Step events use the clock_s their record reports. "" for none (this computer's time)."""
        return str(getattr(self, "microscope_device_address", "") or "")

    def _worker(self, job: dict, job_id: str) -> None:
        abort = self._abort[job_id]
        generation = self._generation
        cls = type(self).__name__
        WORKER.job, WORKER.step = job, None  # this thread's job: proxy() checks it for a dry run
        try:
            self.on_start(job)
            for index in range(step_count(job)):
                if is_done(job, index):
                    continue
                if abort.is_set():
                    break
                if self._suspenders and not self._hold_while_suspended(job, job_id, abort):
                    break  # aborted while paused
                held = self._hold_while_paused(job, job_id, abort)
                if held == "aborted":
                    break
                if held == "continued" and self._suspenders and not self._hold_while_suspended(job, job_id, abort):
                    break  # a condition that went bad during the pause holds the job before its next step
                started = time.time()
                WORKER.step = index
                record = self.run_step(job, index)
                if record is None:
                    if abort.is_set():  # the step stopped early for the abort and did not finish
                        break
                    raise RuntimeError(f"{cls}.run_step returned None for step {index} without an abort; return a record of the step")
                with self._lock:
                    mark_done(job, index)
                    results = job["step_results"]
                    results[str(index)] = record
                    if len(results) > KEPT_STEP_RESULTS:
                        del results[next(iter(results))]  # the oldest; it stays in .steps.jsonl
                    if generation == self._generation:
                        self._append_step(job, index, record)
                        self._save(job)
                self._trace.record(
                    f"{cls}.step", True, started, duration_s=time.time() - started, job_id=job_id, plan_id=job["plan_id"],
                    run_id=job["reservation_id"], step=index, grant_id=job.get("grant_id"), caller=job.get("caller"),
                    result=record, result_key=record.get("key") if isinstance(record, dict) else None,
                    stage_reported_m=record.get("stage_reported_m") if isinstance(record, dict) else None,
                    # The instrument's own clock (simulated time on a twin), when the step reports it.
                    **({"clock_s": record["clock_s"]} if isinstance(record, dict) and "clock_s" in record else {}),
                    **self._dry(job),
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
            self._trace.record(f"{cls}.step", False, time.time(), job_id=job_id, run_id=job["reservation_id"], error=job["last_error"]["message"], result=traceback.format_exc(limit=3), **self._dry(job))
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
        self._trace.record(f"{cls}.{state}", state == "done", time.time(), job_id=job_id, run_id=job["reservation_id"], plan_id=job["plan_id"], result=job.get("result_keys"), **self._dry(job))
        WORKER.job = WORKER.step = None

    def _suspender_problems(self) -> list[tuple[int, str]]:
        """(index, reason) for every suspender condition that is bad now; an unreadable attribute counts as bad."""
        problems = []
        for index, suspender in enumerate(self._suspenders):
            device, attribute = suspender["attr"].rsplit("/", 1)
            try:
                value = self.proxy(device, timeout_s=3.0).read_attribute(attribute).value
            except Exception as exc:
                problems.append((index, f"{suspender['label']}: could not read {suspender['attr']} ({str(exc).splitlines()[0][:120]})"))
                continue
            reason = suspender_problem(suspender, value)
            if reason:
                problems.append((index, reason))
        return problems

    def _hold_while_suspended(self, job: dict, job_id: str, abort: threading.Event) -> bool:
        """Pause the job while a suspender condition is bad; False if it was aborted meanwhile."""
        problems = self._suspender_problems()
        if not problems:
            return True
        cls = type(self).__name__
        tripped = {i for i, _ in problems}
        reason = "; ".join(r for _, r in problems)
        unsafe = None
        try:  # make the instrument safe before reporting the pause
            self.on_suspend(job, reason)
        except Exception as exc:
            unsafe = f"{type(exc).__name__}: {exc}"
            self._trace.record(f"{cls}.on_suspend", False, time.time(), job_id=job_id, run_id=job["reservation_id"], error=unsafe, **self._dry(job))
        with self._lock:
            if job["state"] != "running":  # aborting already
                return False
            job["state"] = "paused_suspended"
            if unsafe:
                job["last_error"] = f"suspended, but the instrument was not made safe: {unsafe}"
            job["suspended"] = {"reason": reason, "since": time.time()}
            self._save(job)
        self._trace.record(f"{cls}.suspended", False, time.time(), job_id=job_id, run_id=job["reservation_id"], error=reason, **self._dry(job))
        good_since = None
        while not abort.wait(max(0.05, float(self.suspender_poll_s))):
            problems = self._suspender_problems()
            if problems:
                good_since = None
                tripped |= {i for i, _ in problems}
                with self._lock:
                    job["suspended"] = {**job["suspended"], "reason": "; ".join(r for _, r in problems)}
                    job["suspended"].pop("resumes_in_s", None)
                continue
            good_since = good_since or time.monotonic()
            wait_s = max(float(self._suspenders[i]["resume_after_s"]) for i in tripped)
            waited = time.monotonic() - good_since
            if waited >= wait_s:
                with self._lock:
                    if job["state"] != "paused_suspended":
                        return False
                    job["state"] = "running"
                    job["suspended"] = None
                    self._save(job)
                self._trace.record(f"{cls}.resumed", True, time.time(), job_id=job_id, run_id=job["reservation_id"], **self._dry(job))
                return True
            with self._lock:
                job["suspended"] = {**job["suspended"], "reason": "conditions are good again", "resumes_in_s": round(wait_s - waited, 1)}
        return False

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Progress of this device's background job: state and steps done of total.")
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

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."} from start.', doc_out="Stops this device's background job after the current step (or sooner, where the step checks for it) and puts the instrument in its safe state. Always allowed.")
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

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"job_id": "..."}.', doc_out="The job's status; its lease runs lease_s from now.")
    def renew_lease(self, request: str) -> str:
        """Keep a running job's lease (lease_s) from lapsing without reading anything else. job_status renews it too."""
        return self._guard(self._status, request)

    @command(dtype_out=str, doc_out='JSON {"aborted": [status of each job that was running]}.')
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

    def proxy(self, address: str, timeout_s: float | None = None) -> tango.DeviceProxy:
        """A proxy to another device whose commands may take longer than Tango's 3 s default
        (a stage move, an autofocus): use this, not a bare DeviceProxy, inside run_step.

        One proxy per address and timeout is kept per thread, so a step that calls the instrument
        many times does not build a new connection each time.

        Inside the worker of a dry-run job it is a RecordingProxy: reads pass through (except
        commands the job's risk_overrides mark high), the job's safe_commands run, and every other
        command and attribute write is recorded, not sent.
        """
        timeout_ms = int(1000 * (timeout_s if timeout_s is not None else self.step_timeout_s))
        cache = self._proxies.__dict__.setdefault("by_address", {})
        proxy = cache.get((address, timeout_ms))
        if proxy is None:
            proxy = tango.DeviceProxy(address)
            proxy.set_timeout_millis(timeout_ms)
            cache[(address, timeout_ms)] = proxy
        job = getattr(WORKER, "job", None)
        if job is not None and job.get("dry_run"):
            return RecordingProxy(proxy, address, lambda *call: self._record_dry_call(job, *call), job.get("risk_overrides") or None, self.safe_commands(job))
        return proxy

    def _record_dry_call(self, job: dict, device: str, name: str, kind: str, arguments, performed: bool) -> None:
        """Trace one call a dry run recorded (or, for blanking, performed), with the plan's check of it."""
        check = None
        if not performed:
            try:
                check = self.check_dry_call(job, device, name, arguments)
            except Exception as exc:  # a broken check is reported, never mistaken for a pass
                check = {"ok": False, "reason": f"check_dry_call failed: {type(exc).__name__}: {exc}"}
        ok = check is None or bool(check.get("ok", True))
        with self._lock:
            counts = job.setdefault("dry_run_calls", {"recorded": 0, "performed": 0, "failed_checks": 0})
            counts["performed" if performed else "recorded"] += 1
            counts["failed_checks"] += 0 if ok else 1
        self._trace.record(
            name, ok, time.time(), dry_run=True, run=job.get("run"), performed=performed, device=device, kind=kind, arguments=arguments,
            check=check, job_id=job.get("job_id"), run_id=job["reservation_id"], plan_id=job.get("plan_id"),
            step=getattr(WORKER, "step", None), caller=job.get("caller"),
        )

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"reservation_id": "..."}.', doc_out="Frees the instrument for the next reservation.")
    def release(self, request: str) -> str:
        """Free the device for the next reservation. A running job must be aborted first."""
        return self._guard(self._release, request)

    def _release(self, request: str) -> str:
        reservation_id = self._parse(request, "reservation_id")["reservation_id"]
        with self._lock:
            job = self._job(reservation_id)
            if job is None:
                raise JobError("unknown_reservation", f"no reservation {reservation_id!r}")
            if job["state"] in RUNNING_STATES:
                raise JobError("bad_state", f"job is {job['state']}; abort it first")
            job["state"] = "released"
            self._save(job)
        return self._reply({"reservation_id": reservation_id, "state": "released"})
