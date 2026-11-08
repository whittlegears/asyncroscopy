"""Suspenders: conditions on other devices' attributes that pause a job between steps."""

from __future__ import annotations

import json
import threading
import time
from typing import Any


SUSPENDER_KEYS = {"attr", "min", "max", "equals", "resume_after_s", "label"}


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


class SuspenderChecks:
    """Between steps: hold a job while a suspender condition is bad."""

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
