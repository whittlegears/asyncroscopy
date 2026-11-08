"""The worker thread that runs a job's steps, its pauses and its lease."""

from __future__ import annotations

import threading
import time
import traceback

import tango

from asyncroscopy.core.names import RUNNING_JOB_STATES
from .files import KEPT_STEP_RESULTS
from .progress import is_done, mark_done, step_count

PAUSE_POLL_S = 0.2


def lease_seconds(device: object) -> float:
    """A job device's lease_s (0: no lease), also for a stand-in that does not declare the property."""
    return float(getattr(device, "lease_s", 0.0) or 0.0)


# The job (and step) the current worker thread runs, so proxy() knows a dry run. Per thread, and each worker
# thread runs one job, so one object serves every device; _worker sets it, so it needs nothing from init_device.
WORKER = threading.local()


class JobWorker:
    """The worker thread's loop, pauses and lease."""

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
        if lease_seconds(self) > 0 and job["state"] in RUNNING_JOB_STATES:  # a paused job someone is watching keeps its lease
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
