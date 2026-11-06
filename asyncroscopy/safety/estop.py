"""The emergency stop: latch the approval mode, then stop every device at once.

``stop_all`` first puts the Approvals device in approve mode and voids every unused grant, so nothing
starts again without a person. Then, on every device in parallel, it calls the stops the MCP server
picked for it (``StopTarget``): first the ones that stop energy reaching the sample (blank a beam,
close a shutter, switch a laser or lamp off), then the others (abort the device's jobs, lift its
probe); then the energy stops once more, in case a job step that ended late switched a source back
on. It never waits for approval. What a device could not do is reported, not skipped.

Which commands are stops is declared, not guessed here: a command's action role (``beam.blank``,
``probe.retract``; ``asyncroscopy.core.names.STOP_ROLES``) or its ``stop`` declaration, and the
envelope's ``abort_all``. High tension, a gun or a filament is never among them: blanking stops the
beam, and those take hours to recover.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, wait as wait_all
from dataclasses import dataclass, field
from typing import Any, Callable

# How long each phase waits for every device to answer before it reports the ones that did not.
STOP_ALL_WAIT_S = 20.0
ABORT_ALL = "abort_all"  # the job envelope's stop for every job of a device
FINISHED = ("aborted", "done", "failed")


@dataclass
class StopTarget:
    """One device and the no-argument stop commands to call on it."""

    prefix: str
    proxy: Any
    energy: list[str] = field(default_factory=list)  # stop energy reaching the sample (called in both phases)
    other: list[str] = field(default_factory=list)  # abort jobs, lift probes (first phase)
    retracts: set[str] = field(default_factory=set)  # probe lifts: a reply of True means the probe is still engaged


def stop_all(targets: list[StopTarget], *, latch: Callable[[str, dict], Any] | None, who: str,
             missing: list[dict], describe: Callable[[Exception], str], wait_s: float = STOP_ALL_WAIT_S) -> list[dict[str, Any]]:
    """Latch (``latch(command, body)`` on the Approvals device; None without one), then stop ``targets``.

    ``missing``: steps already known to have failed (a device that offers no stop for what it does).
    Returns every step, {"tool", "ok", "error"?}, in the order they finished.
    """
    steps: list[dict[str, Any]] = []
    guard = threading.Lock()

    def record(step: dict[str, Any]) -> None:
        with guard:
            steps.append(step)

    if latch is not None:
        for command, body in (("set_mode", {"mode": "approve", "who": who}), ("revoke_grants", {"who": who, "reason": "e-stop"})):
            try:
                latch(command, body)
                record({"tool": f"Approvals_{command}", "ok": True})
            except Exception as exc:
                record({"tool": f"Approvals_{command}", "ok": False, "error": describe(exc)})
    for step in missing:
        record(step)

    def run_device(target: StopTarget, commands: list[str]) -> None:
        for command in commands:
            try:
                reply = target.proxy.command_inout(command)
                step: dict[str, Any] = {"tool": f"{target.prefix}_{command}", "ok": True}
                if reply is True and command in target.retracts:  # an SPM retract answers "still approached"
                    step.update(ok=False, error="the device says the probe is still engaged")
                if command == ABORT_ALL:
                    states = [job.get("state") for job in json.loads(str(reply)).get("aborted", [])]
                    if any(state not in FINISHED for state in states):
                        step.update(ok=False, error=f"jobs not stopped yet (states {states}); poll job_status, then stop again")
                record(step)
            except Exception as exc:
                record({"tool": f"{target.prefix}_{command}", "ok": False, "error": describe(exc)})

    work = [t for t in targets if t.energy or t.other]
    if work:
        # Every device at once: sources off, then its jobs and probes; then, once all of that has returned,
        # sources off again, in case a job step that ended late switched one back on.
        pool = ThreadPoolExecutor(max_workers=len(work), thread_name_prefix="stop_all")
        phases = ({t.prefix: (t, [*t.energy, *t.other]) for t in work}, {t.prefix: (t, list(t.energy)) for t in work if t.energy})
        hung: set[str] = set()
        for phase in phases:
            futures = {pool.submit(run_device, target, commands): prefix for prefix, (target, commands) in phase.items() if prefix not in hung}
            _done, pending = wait_all(futures, timeout=wait_s)
            for future in pending:
                hung.add(futures[future])
                record({"tool": f"{futures[future]}", "ok": False, "error": f"still not answering after {wait_s:g} s; its stop calls keep running in the background"})
        pool.shutdown(wait=False)
    with guard:
        return list(steps)
