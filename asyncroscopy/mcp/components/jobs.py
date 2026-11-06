"""Jobs: run_job, wait_for_job, held runs, job limits, and the interceptors of job (envelope) devices.

run_job reserves a plan, starts it through the approval gate and waits a little; a start held for a
person keeps its reservation ("a held run") so repeating the same run_job after approval starts that
reservation instead of a new one. One reserve-and-start at a time per job device: identical retries
never file twice, and a sweep never releases a reservation a retry is starting.

Replies carry facts, not instructions: a job's ``state``, ``next_actions`` ({"tool", "args"}: the calls
to make next), ``resume_with`` (the call that continues an aborted job) and ``warnings`` ({"code", ...}).
The agent graphs turn them into sentences for a small model (``asyncroscopy.agents.guidance``).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING, Any

import tango
from fastmcp.exceptions import ToolError
from fastmcp.tools import tool
from tango import DeviceProxy

from asyncroscopy.core.names import RUNNING_JOB_STATES
from asyncroscopy.mcp.components.gate import ApprovalHeld
from asyncroscopy.mcp.interceptors import Call, Interceptor, describe_error

if TYPE_CHECKING:
    from asyncroscopy.mcp.descriptors import ToolDescriptor
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

JOB_WAIT_MAX_S = 25.0  # wait_for_job stays under MCP client timeouts
RUN_JOB_WAIT_S = 15.0  # run_job waits this long after starting, so a short job needs no wait_for_job
RUN_JOB_BUDGET_S = 25.0  # run_job as a whole (reserve, gate, start, wait) stays under the 30 s client timeout
HELD_SWEEP_S = 5.0  # how often held run_job reservations are checked against their approval requests
SWEEP_BUDGET_S = 3.0  # the most one sweep spends asking the Approvals device about held runs
HELD_MAX_S = 2 * 3600.0  # a held reservation nobody came back for is released after this long
MAX_HELD_RUNS = 32  # held reservations kept at once; the oldest is released beyond this
# Estimate fields of a job that puts a dose on the sample (a beam job): max_dose_e_per_a2 applies to it.
DOSE_FIELDS = ("max_dose_e_per_a2", "beam_current_pa", "beam_current_source")
SETTLED = ("denied", "expired", "revoked")
def wait_action(job_id: str) -> dict[str, Any]:
    return {"tool": "wait_for_job", "args": {"job_id": job_id}}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value  # NaN is not a dose


def _json_object(text: Any) -> dict | None:
    """``text`` parsed as a JSON object, or None when it is not one."""
    if not isinstance(text, str):
        return None
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None
    return body if isinstance(body, dict) else None


class Jobs:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.analysis_url = cfg.analysis_url
        self.job_limits = dict(cfg.job_limits)
        # (caller, agent run, device, plan) -> {reservation_id, request_id, device, created, summary, why, call} of a
        # run_job whose start is held. A denied or expired request, or one nobody came back for, frees it.
        self.held_runs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.held_lock = threading.Lock()
        self._device_locks: dict[str, threading.RLock] = {}
        self._device_locks_lock = threading.Lock()
        self._via = threading.local()  # set while run_job calls a job device, so the trace says so
        self._last_sweep = 0.0
        self._simulated: bool | None = None  # the stack's simulated flag once the Approvals device has said it

    # ------------------------------------------------------------------ plumbing
    def natives(self) -> list:
        if not self.devices():
            return []
        return [self.wait_for_job] if self.server.endpoint_role == "monitor" else [self.run_job, self.wait_for_job]

    def devices(self) -> list[str]:
        """The job devices' tool prefixes."""
        return sorted(p for p, d in self.server.devices.items() if d.envelope)

    def device_lock(self, device: str) -> threading.RLock:
        """The lock one job device's reserve-and-start (and the release of its held runs) runs under."""
        with self._device_locks_lock:
            return self._device_locks.setdefault(device, threading.RLock())

    def via_name(self) -> str | None:
        return getattr(self._via, "name", None)

    def job_device(self, device: str) -> DeviceProxy:
        info = self.server.devices.get(device)
        if info is None or info.proxy is None or not info.envelope:
            raise ToolError(f"no job device {device!r}; job devices here: {self.devices()}")
        return info.proxy

    def release(self, device: str, reservation_id: str) -> str:
        """Release a reservation run_job made (housekeeping, low risk); returns what happened, for the reply."""
        info = self.server.devices.get(device)
        proxy = info.proxy if info is not None else None
        started = time.time()
        if proxy is None:
            return f"could not release {device} reservation {reservation_id}: no such job device"
        try:
            reply = json.loads(proxy.release(json.dumps({"reservation_id": reservation_id})))
        except Exception as exc:
            note = f"could not release {device} reservation {reservation_id} ({describe_error(exc)}); an operator can release it"
        else:
            failed = isinstance(reply, dict) and "error" in reply
            note = (f"{device} reservation {reservation_id} was not released: {reply['error'].get('message', reply['error'])}" if failed
                    else f"{device} reservation {reservation_id} was released")
        if self.server.trace is not None:
            self.server.trace.record(f"{device}_release", "was released" in note, started, arguments={"reservation_id": reservation_id},
                                     caller="run_job", run=self.server.run_id() or None, result=note)
        return note

    # ------------------------------------------------------------------ held runs
    def settle(self, request_id: str, status: dict) -> dict:
        """approval_status's reply for a request a run_job is held on: when approved, the exact run_job that starts
        it; when denied, expired or revoked, the reservation is released."""
        with self.held_lock:
            waiting = next((v for v in self.held_runs.values() if v["request_id"] == request_id), None)
        if waiting is not None and status.get("state") == "approved":
            # A small model copied the resolved plan out of the reason into a "new" run_job, which then found the device
            # reserved, aborted the approved reservation and was stuck: say exactly which call starts it.
            status["reason"] = f"the reserved {waiting['device']} plan (reservation {waiting['reservation_id']})"
            status["next"] = (f"Approved: call run_job with exactly {json.dumps(waiting['call'], default=str)} now, the same arguments as "
                              "before. A changed plan would be a new job needing its own approval.")
        if status.get("state") in SETTLED:
            with self.held_lock:
                held = next(((k, v) for k, v in self.held_runs.items() if v["request_id"] == request_id), None)
                if held is not None:
                    del self.held_runs[held[0]]
            if held is not None:
                # A run_job held on this request reserved the device; nothing will start it now.
                status["note"] = self.release(held[1]["device"], held[1]["reservation_id"])
        return status

    def sweep_held(self, force: bool = False) -> list[str]:
        """Release held run_job reservations whose request was denied or expired, that nobody came back
        for within HELD_MAX_S, or beyond MAX_HELD_RUNS (oldest first). Returns what was released."""
        now = time.monotonic()
        with self.held_lock:
            if not force and now - self._last_sweep < HELD_SWEEP_S:
                return []
            self._last_sweep = now
            entries = list(self.held_runs.items())
        # Ask the Approvals device before taking any device lock, within SWEEP_BUDGET_S in all: a slow or
        # hung approvals service must not hold up every run_job and wait_for_job behind this sweep.
        stale = []
        budget_end = time.monotonic() + SWEEP_BUDGET_S
        address = self.server.gate.address
        for key, held in entries:
            state = None
            if held["request_id"] and address and time.monotonic() < budget_end:
                try:
                    state = json.loads(self.server.gate.approvals().request_status(held["request_id"])).get("state")
                except Exception:
                    budget_end = 0.0  # the service is not answering; the rest wait for the next sweep
            if state in SETTLED or now - held["created"] > HELD_MAX_S:
                stale.append((key, held))
        with self.held_lock:
            ordered = sorted(self.held_runs, key=lambda k: self.held_runs[k]["created"])
            surplus = [(k, self.held_runs[k]) for k in ordered[: max(0, len(ordered) - MAX_HELD_RUNS)]]
        released = []
        for key, held in self._unique(stale + surplus):
            with self.device_lock(held["device"]):  # never while a run_job is reserving or starting on that device
                with self.held_lock:
                    # Only the entries looked at above: one a run_job resumed or replaced meanwhile is not stale.
                    if self.held_runs.get(key) is not held:
                        continue
                    del self.held_runs[key]
                released.append(self.release(held["device"], held["reservation_id"]))
        return released

    @staticmethod
    def _unique(pairs: list[tuple[tuple, dict]]) -> list[tuple[tuple, dict]]:
        seen, out = set(), []
        for key, held in pairs:
            if key not in seen:
                seen.add(key)
                out.append((key, held))
        return out

    # ------------------------------------------------------------------ job limits
    def limit_reservation(self, device: str, reply: Any) -> Any:
        """A reserve reply, refused (and the reservation released) when its estimate is over mcp.job_limits.

        Applied to every <device>_reserve, so run_job and a direct reserve share the limits.
        """
        reserved = _json_object(reply)
        if reserved is None or "error" in reserved or not reserved.get("reservation_id"):
            return reply
        violations = self.over_limits(reserved)
        if not violations:
            return reply
        note = self.release(device, reserved["reservation_id"])
        summary = {k: reserved[k] for k in ("schema_version", "reservation_id", "plan_id", "resolved_plan", "estimate") if k in reserved}
        return json.dumps({**summary, "note": note, "error": self.limits_error(violations)})

    @staticmethod
    def limits_error(violations: list[dict]) -> dict:
        return {"code": "plan_rejected", "violations": violations, "message": (
            "the estimate is over this endpoint's job limits (mcp.job_limits): "
            + "; ".join(v.get("reason") or f"{v['what']} {v['value']:.4g} {v['units']} > {v['limit']:.4g}" for v in violations)
            + ". Tell the user the estimate and ask before trying a smaller plan; only an operator can raise the limit.")}

    def limit_start(self, device: str, request: Any) -> str | None:
        """An error reply when the reservation a start names is over mcp.job_limits (None when it is within them).

        run_job's resume and a direct <Device>_start never pass through the reserve wrapper, and the
        reservation may be anyone's, so its estimate is read back and checked here. A reservation that
        cannot be read is refused: the limits must be shown to hold, not assumed to.
        """
        info = self.server.devices.get(device)
        proxy = info.proxy if info is not None else None
        try:
            body = _json_object(request.strip() if isinstance(request, str) else request) or {"reservation_id": str(request or "").strip()}
            reserved = json.loads(proxy.get_reservation(json.dumps({"reservation_id": body.get("reservation_id")})))
        except Exception as exc:
            reason = describe_error(exc) if proxy is not None else f"no job device {device!r}"
            return json.dumps({"error": {"code": "plan_rejected", "message": f"could not read the reservation to check it against this endpoint's job limits ({reason}); not started"}})
        if not isinstance(reserved, dict) or "error" in reserved:
            return None  # the start itself reports an unknown reservation
        violations = self.over_limits(reserved)
        if not violations:
            return None
        summary = {k: reserved[k] for k in ("reservation_id", "plan_id", "resolved_plan", "estimate") if k in reserved}
        return json.dumps({**summary, "error": self.limits_error(violations)})

    def stack_simulated(self) -> bool:
        """Whether the stack drives a simulator, as its Approvals device says; no device, or no answer, is real hardware."""
        if self._simulated is not None:
            return self._simulated
        if not self.server.gate.address:
            return False
        try:
            self._simulated = bool(self.server.gate.approvals().get_simulated())
        except Exception:
            return False  # not cached: asked again next time
        return self._simulated

    def over_limits(self, reserved: dict) -> list[dict]:
        estimate = reserved.get("estimate") if isinstance(reserved.get("estimate"), dict) else {}
        checks = (("max_duration_s", "duration_s", "s"), ("max_dose_e_per_a2", "max_dose_e_per_a2", "e/A^2"))
        over = [
            {"what": field, "value": estimate[field], "limit": self.job_limits[limit], "units": units}
            for limit, field, units in checks
            if limit in self.job_limits and _is_number(estimate.get(field)) and estimate[field] > self.job_limits[limit]
        ]
        return over + self.dose_unchecked(estimate)

    def dose_unchecked(self, estimate: dict) -> list[dict]:
        """On hardware, a beam job whose dose the limit cannot be checked against: no numeric dose, or a dose
        computed from a current that was not measured (assumed, or too old for the device to rely on).
        A job that reports no dose concept at all (an AFM or YAML map, an indent grid) is not a beam job."""
        if "max_dose_e_per_a2" not in self.job_limits or not any(k in estimate for k in DOSE_FIELDS) or self.stack_simulated():
            return []
        limit = self.job_limits["max_dose_e_per_a2"]
        found = []
        if not _is_number(estimate.get("max_dose_e_per_a2")):
            found.append({"what": "max_dose_e_per_a2", "value": estimate.get("max_dose_e_per_a2"), "limit": limit, "units": "e/A^2",
                          "reason": "the estimate gives no dose to check against max_dose_e_per_a2"})
        source, reason = estimate.get("beam_current_source"), estimate.get("beam_current_reason")
        if source != "measured" or reason:
            found.append({"what": "beam_current_source", "value": source, "limit": limit, "units": "e/A^2", "reason": (
                f"the dose was computed from a beam current that is {source or 'not said to be measured'}"
                + (f" ({reason})" if reason else "") + ", so max_dose_e_per_a2 cannot be checked: measure the screen current, then reserve again")})
        return found

    # ------------------------------------------------------------------ run_job
    @staticmethod
    def held_message(device: str, request_id: str, reservation_id: str, summary: dict, why: str | None = None) -> str:
        return (
            f"APPROVAL REQUIRED: {device}_start is held for an operator (request_id={request_id}"
            + (f"; {why}" if why else "") + "). "
            "Call approval_status with this request_id until it is approved, then repeat this run_job call unchanged. "
            f"If it is denied, do not retry; report to the user. Reserved as {reservation_id}; estimate: {json.dumps(summary.get('estimate'), default=str)}"
        )

    @tool()
    def run_job(self, device: str, plan: dict[str, Any] | None = None, resume: str | None = None, wait: bool = True) -> dict[str, Any]:
        """Run a job on a job device in one call: the plan is checked (a refused plan comes back with the reason and an example), the job starts, and the call waits up to 15 s. device: the job device's tool prefix, e.g. "Mapper". Every reply has the estimate. If the reply's state is done it already carries the results (result_keys, summary); otherwise call wait_for_job with its job_id until it is done. A plan over this endpoint's limits is refused: tell the user the estimate and ask. To continue a job that was aborted, pass resume=<its reservation_id> and no plan: it goes on from the next step (through the approval gate, like any start). wait=false returns as soon as the job runs (when the user wants the job id now, or to stop it later)."""
        began = time.monotonic()
        info = self.server.devices.get(device)
        tools = self.server.tools.get(device) if info is not None and info.envelope else None
        if not tools or "reserve" not in tools or "start" not in tools:
            raise ToolError(f"no job device {device!r}; job devices here: {self.devices()}")
        if (plan is None) == (resume is None):
            raise ToolError("give a plan for a new job, or resume=<reservation_id> to continue an aborted one (not both)")
        self.sweep_held()
        # A held run is found again by caller and agent run, never by MCP session (agents open one per call).
        target = {"resume": resume} if resume else plan
        key = (self.server.caller() or "", self.server.run_id(), device, json.dumps(target, sort_keys=True, default=str))
        with self.held_lock:
            other = next((v for k, v in self.held_runs.items() if k[:3] == key[:3] and k != key), None)
        if other is not None:
            raise ToolError(
                f"{device} has a run of yours waiting for its approved start (approval request {other['request_id']}, reservation "
                f"{other['reservation_id']}): call run_job with exactly {json.dumps(other['call'], default=str)} to start it. Do not "
                f"abort it unless the user asked to stop."
            )
        with self.device_lock(device):
            self._via.name = "run_job"
            try:
                summary, started = self._reserve_and_start(device, plan, tools, key, resume)
            finally:
                self._via.name = None
        if "error" in started:
            return started
        reply = {**summary, **{k: v for k, v in started.items() if k != "schema_version"}}
        wait_s = max(0.0, min(RUN_JOB_WAIT_S if wait else 0.0, RUN_JOB_BUDGET_S - (time.monotonic() - began)))
        if started.get("job_id"):
            proxy = self.job_device(device)
            status = json.loads(proxy.job_status(json.dumps({"job_id": started["job_id"]})))
            reply.update(self._wait_on_device(proxy, started["job_id"], status, time.monotonic() + wait_s))
            if not wait and reply.get("state") in ("queued", "running"):  # in the background: it can be stopped, or waited for later
                reply["next_actions"] = [wait_action(started["job_id"]), {"tool": f"{device}_abort", "args": {"job_id": started["job_id"]}}]
        return reply

    def _reserve_and_start(self, device: str, plan: dict | None, tools: dict, key: tuple, resume: str | None = None) -> tuple[dict, dict]:
        """(plan summary, start reply or an error reply) for run_job; called under the device's lock."""
        with self.held_lock:
            held = self.held_runs.get(key)
        if held is not None and held["request_id"] and self.server.gate.address is not None:
            try:
                state = json.loads(self.server.gate.approvals().request_status(held["request_id"])).get("state")
            except Exception:
                state = None
            if state == "pending":  # asking again before the operator decided must not file a second request
                raise ToolError(self.held_message(device, held["request_id"], held["reservation_id"], held["summary"], held.get("why")))
            if state in SETTLED:
                with self.held_lock:
                    self.held_runs.pop(key, None)
                note = self.release(device, held["reservation_id"])
                raise ToolError(f"the approval request {held['request_id']} for this run was {state}; {note}. Do not retry; tell the user.")
        if held is None and resume:
            # Continue an aborted job: its reservation already holds the plan; starting it again resumes it.
            try:
                reserved = json.loads(self.job_device(device).get_reservation(json.dumps({"reservation_id": resume})))
            except Exception as exc:
                raise ToolError(f"no reservation {resume!r} on {device} to resume ({describe_error(exc)})") from exc
            if "error" in reserved:
                return {}, {k: v for k, v in reserved.items() if k != "schema_version"}
            summary = {"reservation_id": resume, **{k: reserved[k] for k in ("plan_id", "resolved_plan", "estimate") if k in reserved}, "resumed": True}
        elif held is None:
            # The reserve wrapper applies mcp.job_limits and releases a reservation over them.
            reserved = json.loads(tools["reserve"]({"plan": plan}))
            if "error" in reserved:
                return {}, {k: v for k, v in reserved.items() if k != "schema_version"}
            summary = {k: reserved[k] for k in ("reservation_id", "plan_id", "resolved_plan", "estimate", "limits") if k in reserved}
        else:
            summary = held["summary"]
        reservation_id = summary["reservation_id"]
        try:
            # The start goes through the approval gate like <device>_start; the operator sees the reserved plan.
            started = json.loads(tools["start"]({"reservation_id": reservation_id}))
        except Exception as exc:
            if isinstance(exc, ApprovalHeld):
                with self.held_lock:
                    self.held_runs[key] = {"reservation_id": reservation_id, "request_id": exc.request_id, "device": device,
                                           "created": time.monotonic(), "summary": summary, "why": exc.why,
                                           "call": {"device": device, **({"resume": resume} if resume else {"plan": plan})}}
                raise ToolError(self.held_message(device, exc.request_id, reservation_id, summary, exc.why)) from exc
            with self.held_lock:
                self.held_runs.pop(key, None)
            note = self._release_unless_resumed(device, summary)
            raise ToolError(f"{device} did not start ({describe_error(exc)}); {note}. Plan: {json.dumps(summary, default=str)[:1500]}") from exc
        with self.held_lock:
            self.held_runs.pop(key, None)
        if "error" in started:
            return summary, {**summary, **started, "note": self._release_unless_resumed(device, summary)}
        return summary, started

    def _release_unless_resumed(self, device: str, summary: dict) -> str:
        """Free a reservation run_job made for a start that failed; an aborted job being resumed keeps its own."""
        if summary.get("resumed"):
            return f"{device} reservation {summary['reservation_id']} is kept, so the job can still be resumed"
        return self.release(device, summary["reservation_id"])

    # ------------------------------------------------------------------ wait_for_job
    @tool()
    def wait_for_job(self, job_id: str, device: str | None = None, max_wait_s: float = JOB_WAIT_MAX_S) -> dict[str, Any]:
        """Wait for any background job (a map from run_job, or assemble_map / score_map on the analysis server) to finish, at most max_wait_s (25 s) per call, and return its state. When it is done the reply also has its results (result_keys, summary). device (e.g. "Mapper") is optional. Call again while the state is still active."""
        budget = min(max(float(max_wait_s), 0.0), JOB_WAIT_MAX_S)
        self.sweep_held()
        devices = self.devices()
        if device and device not in devices:
            raise ToolError(f"no job device {device!r}; job devices here: {devices}"
                            + ("; leave device out for analysis jobs" if self.analysis_url else ""))
        unreachable = []
        for candidate in [device] if device else devices:
            proxy = self.job_device(candidate)
            try:
                status = json.loads(proxy.job_status(json.dumps({"job_id": job_id})))
            except tango.DevFailed as exc:
                if device:  # the one device asked for is down: say so
                    raise ToolError(f"{candidate} did not answer ({describe_error(exc)}); try wait_for_job again shortly") from exc
                unreachable.append(candidate)  # one dead job device must not stop waiting on the others' jobs
                continue
            if status.get("error", {}).get("code") != "unknown_job":
                return self._wait_on_device(proxy, job_id, status, time.monotonic() + budget)
        if self.analysis_url:
            return self._wait_on_analysis(job_id, budget)
        raise ToolError(
            f"no job {job_id!r} on {device or 'any job device here'}"
            + (f" (not answering: {unreachable})" if unreachable else "")
            + ". Jobs started by assemble_map or score_map run on the analysis server; this endpoint can wait on them "
            "only when its MCP config sets mcp.analysis_url."
        )

    def _wait_on_device(self, proxy: DeviceProxy, job_id: str, status: dict, deadline: float) -> dict[str, Any]:
        while True:
            if "error" in status:
                return status
            state = status.get("state")
            if state not in RUNNING_JOB_STATES:
                if state == "done":  # every phase of the plan ran: nothing needs running or stitching again
                    results = json.loads(proxy.results(json.dumps({"job_id": job_id})))
                    status.update({k: results[k] for k in ("result_keys", "summary") if k in results}, next_actions=[], warnings=[{"code": "finished"}])
                return status
            if time.monotonic() >= deadline:  # still active: waiting again is the next step (a paused job says why in "suspended")
                return {**status, "next_actions": [wait_action(job_id)]}
            time.sleep(1.0)
            # Polls job_status (low risk) rather than blocking in the device, so abort stays immediate.
            status = json.loads(proxy.job_status(json.dumps({"job_id": job_id})))

    def _wait_on_analysis(self, job_id: str, budget: float) -> dict[str, Any]:
        """Wait on a stitching or scoring job of the analysis server; when it is done, add its results."""
        from fastmcp import Client

        def text(reply: Any) -> str:
            return "".join(getattr(b, "text", "") for b in getattr(reply, "content", []) or [])

        async def call() -> dict[str, Any]:
            async with Client(self.analysis_url, timeout=budget + 15) as client:
                reply = await client.call_tool("analysis_job_status", {"job_id": job_id, "wait_s": budget}, raise_on_error=False)
                if reply.is_error and "wait_s" in text(reply):  # an analysis server without server-side waits: poll it
                    deadline = time.monotonic() + budget
                    while True:
                        reply = await client.call_tool("analysis_job_status", {"job_id": job_id}, raise_on_error=False)
                        state = (reply.structured_content or {}).get("state")
                        if reply.is_error or state not in ("queued", "running") or time.monotonic() >= deadline:
                            break
                        await asyncio.sleep(1.0)
                if reply.is_error:
                    raise RuntimeError(text(reply)[:500] or "an error with no message")
                status = dict(reply.structured_content or {})
                if not status:
                    raise RuntimeError(f"an empty reply: {text(reply)[:300]!r}")
                if status.get("state") == "done":
                    results = await client.call_tool("analysis_job_results", {"job_id": job_id}, raise_on_error=False)
                    if results.is_error:
                        status["warnings"] = [{"code": "results_unreadable", "detail": text(results)[:300]}]
                    else:
                        status.update({k: v for k, v in (results.structured_content or {}).items() if k in ("result_keys", "summary")})
                return status

        try:
            # Sync tools run in FastMCP's worker threads, which have no event loop of their own.
            status = asyncio.run(call())
        except Exception as exc:
            raise ToolError(f"the analysis server at {self.analysis_url} did not answer ({type(exc).__name__}: {exc}); call wait_for_job again shortly") from exc
        if status.get("error", {}).get("code") == "unknown_job":
            raise ToolError(f"no job {job_id!r} on any job device here or on the analysis server")
        if status.get("state") in {"queued", "running"}:
            status["next_actions"] = [wait_action(job_id)]
        return status


# ---------------------------------------------------------------------- interceptors of job devices
def resume_with(device: str, reply: Any) -> Any:
    """An aborted (or aborting) job's reply names the call that continues it (models asked to resume started a
    new job instead); while aborting, that call is for once wait_for_job says aborted."""
    status = _json_object(reply)
    if status is None or status.get("state") not in ("aborting", "aborted") or not status.get("reservation_id"):
        return reply
    status["resume_with"] = {"tool": "run_job", "args": {"device": device, "resume": status["reservation_id"]}}
    return json.dumps(status)


def with_plan_id(request: Any, plan_id: str) -> Any:
    """The start request with the plan_id its approval was checked against."""
    body = _json_object(request)
    return json.dumps({**body, "plan_id": plan_id}) if body is not None else request


def with_grant(request: Any, gate: dict | None, caller: str | None, run: str | None = None, risk_overrides: dict | None = None) -> Any:
    """Tell a job who asked, from which agent run, and (at start, ``gate``) which approval covers it,
    so its trace links to the grant and the run. At reserve, ``risk_overrides`` are this endpoint's
    high patterns, which the job's dry run applies.

    Added after the approval check, so it never changes the call's fingerprint.
    """
    body = _json_object(request)
    if body is None:
        return request
    if gate is None and not caller and not run and not risk_overrides and "run" not in body:
        return request  # a reserve with nothing to add goes through as it came
    # Only the gate may say which approval covers a job; an agent-supplied grant_id is dropped.
    body.pop("grant_id", None)
    if gate and gate.get("grant_id"):
        body["grant_id"] = gate["grant_id"]
    if caller:
        body["caller"] = caller
    # Likewise the run: only the X-Asyncroscopy-Run header names it, never the request body.
    body.pop("run", None)
    if run:
        body["run"] = run
    if risk_overrides:  # the deployment's patterns, added to any the caller gave (they can only mark more high)
        body["risk_overrides"] = {**(body.get("risk_overrides") if isinstance(body.get("risk_overrides"), dict) else {}), **risk_overrides}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


class _EnvelopeOnly(Interceptor):
    def __init__(self, jobs: Jobs) -> None:
        self.jobs = jobs

    def applies(self, tool: ToolDescriptor) -> bool:
        info = self.jobs.server.devices.get(tool.device or "")
        return info is not None and info.envelope


class JobChecks(_EnvelopeOnly):
    """Before the gate: a start keeps to the job limits, never continues a person's pause, and is approved for
    the plan as reserved. After the call: a reserve over the limits is refused, an abort says how to resume."""

    order = 30

    def before_call(self, call: Call) -> Any:
        if call.tool.command != "start" or not call.call_args:
            return None
        server, device = self.jobs.server, call.tool.device or ""
        proxy = server.devices[device].proxy
        request = str(call.call_args[0])
        if self.jobs.job_limits and server.endpoint_role == "agent":
            # Whoever reserved it, a start on this endpoint keeps to its job limits (a resume too).
            refused = self.jobs.limit_start(device, call.call_args[0])
            if refused is not None:
                if server.trace is not None:
                    server.trace.record(call.tool.name, False, call.started, arguments=call.arguments, error="over_job_limits", result=refused,
                                        caller=call.caller, run=call.run)
                return refused
        body = _json_object(request)
        if server.endpoint_role != "operator" and body is not None and "continue" in body:
            # Continuing a paused job is the person's decision, made on the operator endpoint.
            raise ToolError("a person paused this job; only they continue it (from the operator endpoint or the GUI). Call wait_for_job, or abort if the user asked to stop.")
        if proxy is not None:
            call.context["describe"] = lambda: "Reserved plan: " + proxy.get_reservation(request)
            call.context["drives"] = lambda: (json.loads(proxy.get_reservation(request)).get("resolved_plan") or {}).get("drives")
        if server.endpoint_role != "operator" and server.gate.address is not None:
            # The approval is for the plan as shown: a plan revised after the request was filed (or a grant
            # given) has a new plan_id, so the old approval does not start it.
            try:
                plan_id = json.loads(proxy.get_reservation(request)).get("plan_id") if proxy is not None else None
            except Exception:
                plan_id = None
            call.context["plan_id"] = plan_id
            call.context["gated_arguments"] = {**call.arguments, "plan_id": plan_id}
        return None

    def after_call(self, call: Call, result: Any) -> Any:
        if call.tool.command == "reserve" and self.jobs.job_limits:
            result = call.reply = self.jobs.limit_reservation(call.tool.device or "", result)
        elif call.tool.command == "abort":
            result = call.reply = resume_with(call.tool.device or "", result)
        return result


class JobStamp(_EnvelopeOnly):
    """After the gate: a reserve or start tells the job who asked, the agent run, the approval that covers it
    (at start) and the endpoint's high risk patterns (at reserve, for a dry run)."""

    order = 50

    def before_call(self, call: Call) -> Any:
        command = call.tool.command
        if command not in ("start", "reserve") or not call.call_args:
            return None
        overrides = self.jobs.server.config.high_overrides() if command == "reserve" else None
        first = with_grant(call.call_args[0], call.trace if command == "start" else None, call.caller, call.run, overrides)
        if call.context.get("plan_id") is not None:  # the device refuses the start if the plan was revised after this check
            first = with_plan_id(first, call.context["plan_id"])
        call.call_args = (first, *call.call_args[1:])
        return None


def register(server: MCPServer, cfg: ServerConfig) -> Jobs:
    jobs = Jobs(server, cfg)
    server.add_interceptor(JobChecks(jobs))
    server.add_interceptor(JobStamp(jobs))
    return jobs
