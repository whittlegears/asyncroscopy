"""A job's files: the plan written once, the small progress record, the appended step results."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from asyncroscopy.core.names import RUNNING_JOB_STATES
from .progress import done_ranges_from, read_last_steps

FINISHED_STATES = {"done", "released"}  # left on disk at Init and opened when asked for by id
PLAN_KEYS = frozenset({"plan_id", "device", "plan", "resolved_plan", "steps", "step_count", "estimate", "limits", "dry_run", "risk_overrides", "created", "trace_offset"})
KEPT_STEP_RESULTS = 10_000


def _json_default(value):
    """Keep numpy values numbers in JSON: np.int64(2) is 2 and an array a list."""
    return value.tolist() if hasattr(value, "tolist") else str(value)


class JobFiles:
    """Reading and writing a job's files; ``_reload`` picks jobs up after a restart."""

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
            if job.get("state") in RUNNING_JOB_STATES:
                job["state"] = "failed"
                job["last_error"] = {"code": "device_restarted", "message": "The device restarted while this job was running; start it again to resume."}
            elif job.get("state") == "reserved":
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
            else:  # an older job file holds the plan too, so write the plan to its own file
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

    def _path(self, reservation_id: str) -> Path:
        return self._state_dir / f"{self._name.replace('/', '_')}__{reservation_id}.json"

    def _plan_path(self, reservation_id: str) -> Path:
        return self._path(reservation_id).with_suffix(".plan.json")

    def _steps_path(self, reservation_id: str) -> Path:
        return self._path(reservation_id).with_suffix(".steps.jsonl")

    def _read_steps(self, reservation_id: str) -> dict:
        return read_last_steps(self._steps_path(reservation_id), KEPT_STEP_RESULTS)

    def _save(self, job: dict) -> None:
        """Write the job's small progress record; the plan and step results have their own files."""
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
        """A deep copy of the job's progress under the lock, retried when a hook changes it mid-copy."""
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
