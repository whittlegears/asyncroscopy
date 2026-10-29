"""Tango base class for jobs that outlive a single command.

A Tango command times out after seconds; a map takes hours. ``JobDevice``
splits a job into the envelope verbs: ``reserve`` checks a plan and returns an
estimate, ``start`` launches a worker thread and returns at once, ``status``
reports progress, ``abort`` stops within one step, ``results`` returns what was
produced and ``release`` frees the device. State is saved after every step,
so a restarted device resumes a job and skips the steps already done.

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
"""

from __future__ import annotations

import bisect
import json
import threading
from collections import deque
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

ACTIVE_STATES = {"reserved", "queued", "running", "paused_awaiting_approval", "paused_suspended", "aborting"}
RUNNING_STATES = {"queued", "running", "aborting", "paused_awaiting_approval", "paused_suspended"}  # a worker owns the job
SUSPENDER_KEYS = {"attr", "min", "max", "equals", "resume_after_s", "label"}
VIOLATIONS_SHOWN = 20  # violations listed in a refusal; the total is always given
RESUMABLE_STATES = {"reserved", "failed", "aborted"}
SUMMARY_STEPS = 50  # step records results returns by default; the rest are in the job's .steps.jsonl
# Step records kept in memory (the newest); all of them stay in the job's .steps.jsonl. Bounds memory
# for jobs of millions of steps; a device whose summary needs every record keeps its jobs below this.
KEPT_STEP_RESULTS = 10_000


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
    def __init__(self, code: str, message: str, **details):
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


def suspender_problem(suspender: dict, value) -> str | None:
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
    # Workers still finishing a step when the device was last re-initialised (see delete_device).
    _stale_workers: tuple = ()
    _generation = 0

    job_kind = attribute(dtype=str, doc="What this device's background job does.")
    plan_example = attribute(dtype=str, doc="A valid plan for reserve, as JSON ('' if the device gives none).")
    state_directory = device_property(dtype=str, default_value="outputs/jobs", doc="Where job state files are kept.")
    trace_path = device_property(dtype=str, default_value="outputs/trace/jobs.jsonl", doc="Instrument Trace JSONL file.")
    step_timeout_s = device_property(dtype=float, default_value=120.0, doc="Timeout of proxies made with proxy(): the longest one instrument command may take.")
    suspenders = device_property(dtype=str, default_value="", doc='JSON list of conditions checked between steps, e.g. [{"attr": "sys/beam/1/current_pa", "min": 10, "resume_after_s": 30}]; a bad one pauses the job until it has been good again for resume_after_s.')
    suspender_poll_s = device_property(dtype=float, default_value=1.0, doc="How often a paused job re-reads its suspender conditions.")

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

    def on_finish(self, job: dict) -> dict:
        """Called after the last step; return result keys to merge into ``results``."""
        return {}

    def on_suspend(self, job: dict, reason: str) -> None:
        """Called in the worker when a suspender pauses the job; leave the instrument waiting safely (e.g. beam blanked)."""

    def example_plan(self, request: str) -> dict | None:
        """The example plan shown with a refusal; a device may tailor it to the refused request."""
        return self.PLAN_EXAMPLE

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
        self._job_index: dict[str, str] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._abort: dict[str, threading.Event] = {}
        self._name = self.get_name()
        self._state_dir = Path(self.state_directory)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._trace = TraceWriter(self.trace_path, source="job")
        self._suspenders = parse_suspenders(self.suspenders)  # a bad property fails Init with the reason
        self._reload()
        self.set_state(DevState.ON)

    def _reload(self) -> None:
        """Pick up jobs from a previous run of this device; interrupted ones become resumable."""
        for path in sorted(self._state_dir.glob(f"{self._name.replace('/', '_')}__*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("state") in RUNNING_STATES:
                job["state"] = "failed"
                job["last_error"] = {"code": "device_restarted", "message": "The device restarted while this job was running; start it again to resume."}
            elif job.get("state") == "reserved":
                # Never started; whoever reserved it is gone. Keeping it would leave the device busy.
                job["state"] = "released"
                job["last_error"] = {"code": "device_restarted", "message": "The device restarted before this reservation was started; reserve again."}
            job["done_ranges"] = done_ranges_from(job)
            job.pop("done", None)
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
        return read_last_steps(self._steps_path(reservation_id), KEPT_STEP_RESULTS)

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
            if exc.details:
                return json.dumps({"schema_version": ENVELOPE_VERSION, "error": {"code": exc.code, "message": str(exc), **exc.details}}, default=_json_default)
            return self._error(exc.code, str(exc))
        except (ValueError, TypeError, KeyError, jsonschema.ValidationError) as exc:
            if isinstance(exc, (TypeError, KeyError)) and fn != self._reserve:
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
            lazy = isinstance(resolved, dict) and isinstance(resolved.get("step_count"), int) and "steps" not in resolved
            if not isinstance(resolved, dict) or not (lazy or isinstance(resolved.get("steps"), list)):
                # The device's own bug, not the caller's plan: say so, outside plan_rejected.
                raise JobError("device_error", f"{type(self).__name__}.resolve_plan must return a dict with a 'steps' list or a 'step_count', got {type(resolved).__name__}")
            shape = {"step_count": resolved["step_count"]} if lazy else {"steps": resolved["steps"]}
            limits = self._check_limits({"plan": body["plan"], "resolved_plan": resolved, **shape})
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
                "done_ranges": [],
                "step_results": {},
                "result_keys": {},
                "job_ids": [],
                "caller": body.get("caller"),
                "grant_id": None,
                "last_error": None,
                "created": time.time(),
                # Where this job's events start in the shared trace file, so reading them back costs this job, not the history.
                "trace_offset": self._trace.path.stat().st_size if self._trace.path.exists() else 0,
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
                **({"limits": limits} if limits["targets"] else {}),
            }
        )

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
                "caller": body.get("caller") or job.get("caller"), "state": "running", "last_error": None, "suspended": None,
            }
            previous = dict(job)
            self._save(started)  # if this fails, the reservation stays as it was and can be started again
            job.update(started)
            self._job_index[job_id] = job["reservation_id"]
            self._abort[job_id] = threading.Event()
            thread = threading.Thread(target=self._omni_worker, args=(job, job_id), name=f"job-{job_id}", daemon=True)
            self._threads[job_id] = thread
            try:
                thread.start()
            except Exception:  # no thread to run it (RuntimeError, MemoryError): put the reservation back as it was
                for table in (self._job_index, self._abort, self._threads):
                    table.pop(job_id, None)
                job.clear()
                job.update(previous)
                self._save(job)
                raise
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
            for index in range(step_count(job)):
                if is_done(job, index):
                    continue
                if abort.is_set():
                    break
                if self._suspenders and not self._hold_while_suspended(job, job_id, abort):
                    break  # aborted while paused
                started = time.time()
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
        try:  # make the instrument safe before reporting the pause
            self.on_suspend(job, reason)
        except Exception as exc:
            self._trace.record(f"{cls}.on_suspend", False, time.time(), job_id=job_id, run_id=job["reservation_id"], error=f"{type(exc).__name__}: {exc}")
        with self._lock:
            if job["state"] != "running":  # aborting already
                return False
            job["state"] = "paused_suspended"
            job["suspended"] = {"reason": reason, "since": time.time()}
            self._save(job)
        self._trace.record(f"{cls}.suspended", False, time.time(), job_id=job_id, run_id=job["reservation_id"], error=reason)
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
                self._trace.record(f"{cls}.resumed", True, time.time(), job_id=job_id, run_id=job["reservation_id"])
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
        return self._reply(
            {
                "job_id": job.get("job_id", job["reservation_id"]),
                "reservation_id": job["reservation_id"],
                "state": job["state"],
                "progress": {"done": done_count(job), "total": step_count(job)},
                "last_error": job["last_error"],
                "pending_request_id": None,
                "suspended": job.get("suspended"),
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
            if job["state"] in {"running", "paused_suspended"} and job_id in self._abort:
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
                "summary": {"steps_done": done_count(job), "steps_total": step_count(job), **self.summary(job)},
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
            if job["state"] in RUNNING_STATES:
                raise JobError("bad_state", f"job is {job['state']}; abort it first")
            job["state"] = "released"
            self._save(job)
        return self._reply({"reservation_id": reservation_id, "state": "released"})
