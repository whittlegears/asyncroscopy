"""proxy() for a job's steps, and the record of what a dry run did not do."""

from __future__ import annotations

import time

import tango

from asyncroscopy.jobs.recording_proxy import RecordingProxy
from .worker import WORKER

class DryRunCalls:
    """``proxy`` and the record of a dry run's calls."""

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
