"""Jobs: run_job, wait_for_job, job limits, and the interceptors of job (envelope) devices.

run_job reserves a plan, starts it and waits a little. One reserve-and-start at a time per job device.

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
from asyncroscopy.mcp.interceptors import Call, Interceptor, describe_error

if TYPE_CHECKING:
    from asyncroscopy.mcp.descriptors import ToolDescriptor
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

JOB_WAIT_MAX_S = 25.0  # stays under MCP client timeouts
RUN_JOB_WAIT_S = 15.0  # so a short job needs no wait_for_job
RUN_JOB_BUDGET_S = 25.0  # under the 30 s client timeout
DOSE_FIELDS = ("max_dose_e_per_a2", "beam_current_pa", "beam_current_source")


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
        self.analysis = cfg.urls["analysis"]
        self.job_limits = dict(cfg.job_limits)
        self._device_locks: dict[str, threading.RLock] = {}
        self._device_locks_lock = threading.Lock()
        self._via = threading.local()  # set while run_job calls a job device, for the trace

    def natives(self) -> list:
        if not self.devices():
            return []
        return [self.wait_for_job] if self.server.endpoint_role == "monitor" else [self.run_job, self.wait_for_job]

    def devices(self) -> list[str]:
        """The job devices' tool prefixes."""
        return sorted(p for p, d in self.server.devices.items() if d.envelope)

    def device_lock(self, device: str) -> threading.RLock:
        """The lock one job device's reserve-and-start runs under."""
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
            return None  # the start reports an unknown reservation
        violations = self.over_limits(reserved)
        if not violations:
            return None
        summary = {k: reserved[k] for k in ("reservation_id", "plan_id", "resolved_plan", "estimate") if k in reserved}
        return json.dumps({**summary, "error": self.limits_error(violations)})

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
        if "max_dose_e_per_a2" not in self.job_limits or not any(k in estimate for k in DOSE_FIELDS) or self.server.stack_simulated():
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

    @tool()
    def run_job(self, device: str, plan: dict[str, Any] | None = None, resume: str | None = None, wait: bool = True) -> dict[str, Any]:
        """Run a job on a job device in one call: the plan is checked (a refused plan comes back with the reason and an example), the job starts, and the call waits up to 15 s. device: the job device's tool prefix, e.g. "Mapper". Every reply has the estimate. If the reply's state is done it already carries the results (result_keys, summary); otherwise call wait_for_job with its job_id until it is done. A plan over this endpoint's limits is refused: tell the user the estimate and ask. To continue a job that was aborted, pass resume=<its reservation_id> and no plan: it goes on from the next step. wait=false returns as soon as the job runs (when the user wants the job id now, or to stop it later)."""
        began = time.monotonic()
        info = self.server.devices.get(device)
        tools = self.server.tools.get(device) if info is not None and info.envelope else None
        if not tools or "reserve" not in tools or "start" not in tools:
            raise ToolError(f"no job device {device!r}; job devices here: {self.devices()}")
        if (plan is None) == (resume is None):
            raise ToolError("give a plan for a new job, or resume=<reservation_id> to continue an aborted one (not both)")
        with self.device_lock(device):
            self._via.name = "run_job"
            try:
                summary, started = self._reserve_and_start(device, plan, tools, resume)
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
            if not wait and reply.get("state") in ("queued", "running"):  # runs in the background; stop or wait later
                reply["next_actions"] = [wait_action(started["job_id"]), {"tool": f"{device}_abort", "args": {"job_id": started["job_id"]}}]
        return reply

    def _reserve_and_start(self, device: str, plan: dict | None, tools: dict, resume: str | None = None) -> tuple[dict, dict]:
        """(plan summary, start reply or an error reply) for run_job; called under the device's lock."""
        if resume:
            try:
                reserved = json.loads(self.job_device(device).get_reservation(json.dumps({"reservation_id": resume})))
            except Exception as exc:
                raise ToolError(f"no reservation {resume!r} on {device} to resume ({describe_error(exc)})") from exc
            if "error" in reserved:
                return {}, {k: v for k, v in reserved.items() if k != "schema_version"}
            summary = {"reservation_id": resume, **{k: reserved[k] for k in ("plan_id", "resolved_plan", "estimate") if k in reserved}, "resumed": True}
        else:
            reserved = json.loads(tools["reserve"]({"plan": plan}))
            if "error" in reserved:
                return {}, {k: v for k, v in reserved.items() if k != "schema_version"}
            summary = {k: reserved[k] for k in ("reservation_id", "plan_id", "resolved_plan", "estimate", "limits") if k in reserved}
        reservation_id = summary["reservation_id"]
        try:
            started = json.loads(tools["start"]({"reservation_id": reservation_id}))
        except Exception as exc:
            note = self._release_unless_resumed(device, summary)
            raise ToolError(f"{device} did not start ({describe_error(exc)}); {note}. Plan: {json.dumps(summary, default=str)[:1500]}") from exc
        if "error" in started:
            return summary, {**summary, **started, "note": self._release_unless_resumed(device, summary)}
        return summary, started

    def _release_unless_resumed(self, device: str, summary: dict) -> str:
        """Free a reservation run_job made for a start that failed; an aborted job being resumed keeps its own."""
        if summary.get("resumed"):
            return f"{device} reservation {summary['reservation_id']} is kept, so the job can still be resumed"
        return self.release(device, summary["reservation_id"])

    @tool()
    def wait_for_job(self, job_id: str, device: str | None = None, max_wait_s: float = JOB_WAIT_MAX_S) -> dict[str, Any]:
        """Wait for any background job (a map from run_job, or assemble_map / score_map on the analysis server) to finish, at most max_wait_s (25 s) per call, and return its state. When it is done the reply also has its results (result_keys, summary). device (e.g. "Mapper") is optional. Call again while the state is still active."""
        budget = min(max(float(max_wait_s), 0.0), JOB_WAIT_MAX_S)
        devices = self.devices()
        if device and device not in devices:
            raise ToolError(f"no job device {device!r}; job devices here: {devices}"
                            + ("; leave device out for analysis jobs" if self.analysis else ""))
        unreachable = []
        for candidate in [device] if device else devices:
            proxy = self.job_device(candidate)
            try:
                status = json.loads(proxy.job_status(json.dumps({"job_id": job_id})))
            except tango.DevFailed as exc:
                if device:  # the one device asked for is down
                    raise ToolError(f"{candidate} did not answer ({describe_error(exc)}); try wait_for_job again shortly") from exc
                unreachable.append(candidate)  # one dead job device must not stop the others' waits
                continue
            if status.get("error", {}).get("code") != "unknown_job":
                return self._wait_on_device(proxy, job_id, status, time.monotonic() + budget)
        if self.analysis:
            return self._wait_on_analysis(job_id, budget)
        raise ToolError(
            f"no job {job_id!r} on {device or 'any job device here'}"
            + (f" (not answering: {unreachable})" if unreachable else "")
            + ". Jobs started by assemble_map or score_map run on the analysis server; this endpoint can wait on them "
            "only when its MCP config sets mcp.urls.analysis."
        )

    def _wait_on_device(self, proxy: DeviceProxy, job_id: str, status: dict, deadline: float) -> dict[str, Any]:
        while True:
            if "error" in status:
                return status
            state = status.get("state")
            if state not in RUNNING_JOB_STATES:
                if state == "done":  # every phase ran
                    results = json.loads(proxy.results(json.dumps({"job_id": job_id})))
                    status.update({k: results[k] for k in ("result_keys", "summary") if k in results}, next_actions=[], warnings=[{"code": "finished"}])
                return status
            if time.monotonic() >= deadline:  # still active: wait again
                return {**status, "next_actions": [wait_action(job_id)]}
            time.sleep(1.0)
            # Poll job_status and do not block in the device, so abort stays immediate.
            status = json.loads(proxy.job_status(json.dumps({"job_id": job_id})))

    def _wait_on_analysis(self, job_id: str, budget: float) -> dict[str, Any]:
        """Wait on a stitching or scoring job of the analysis server; when it is done, add its results."""
        from fastmcp import Client

        def text(reply: Any) -> str:
            return "".join(getattr(b, "text", "") for b in getattr(reply, "content", []) or [])

        async def call() -> dict[str, Any]:
            async with Client(self.analysis, timeout=budget + 15) as client:
                reply = await client.call_tool("analysis_job_status", {"job_id": job_id, "wait_s": budget}, raise_on_error=False)
                if reply.is_error and "wait_s" in text(reply):  # this analysis server cannot wait: poll it
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
            raise ToolError(f"the analysis server at {self.analysis} did not answer ({type(exc).__name__}: {exc}); call wait_for_job again shortly") from exc
        if status.get("error", {}).get("code") == "unknown_job":
            raise ToolError(f"no job {job_id!r} on any job device here or on the analysis server")
        if status.get("state") in {"queued", "running"}:
            status["next_actions"] = [wait_action(job_id)]
        return status


def resume_with(device: str, reply: Any) -> Any:
    """An aborted (or aborting) job's reply names the call that continues it (models asked to resume started a
    new job instead); while aborting, that call is for once wait_for_job says aborted."""
    status = _json_object(reply)
    if status is None or status.get("state") not in ("aborting", "aborted") or not status.get("reservation_id"):
        return reply
    status["resume_with"] = {"tool": "run_job", "args": {"device": device, "resume": status["reservation_id"]}}
    return json.dumps(status)


def with_caller(request: Any, caller: str | None, run: str | None = None, risk_overrides: dict | None = None) -> Any:
    """Tell a job who asked and from which agent run, so its trace links to the run. At reserve,
    ``risk_overrides`` are this endpoint's high patterns, which the job's dry run applies."""
    body = _json_object(request)
    if body is None:
        return request
    if not caller and not run and not risk_overrides and "run" not in body:
        return request  # nothing to add: pass it through
    if caller:
        body["caller"] = caller
    # Only the request header may name the run, so one in the request body is dropped.
    body.pop("run", None)
    if run:
        body["run"] = run
    if risk_overrides:  # deployment patterns only add high-risk commands
        body["risk_overrides"] = {**(body.get("risk_overrides") if isinstance(body.get("risk_overrides"), dict) else {}), **risk_overrides}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


class _EnvelopeOnly(Interceptor):
    def __init__(self, jobs: Jobs) -> None:
        self.jobs = jobs

    def applies(self, tool: ToolDescriptor) -> bool:
        info = self.jobs.server.devices.get(tool.device or "")
        return info is not None and info.envelope


class JobChecks(_EnvelopeOnly):
    """Before the call: a start keeps to the job limits and never continues a person's pause.
    After the call: a reserve over the limits is refused, an abort says how to resume."""

    order = 30

    def before_call(self, call: Call) -> Any:
        if call.tool.command != "start" or not call.call_args:
            return None
        server, device = self.jobs.server, call.tool.device or ""
        request = str(call.call_args[0])
        if self.jobs.job_limits and server.endpoint_role == "agent":
            # Whoever reserved it, a start or resume on this endpoint keeps to its job limits.
            refused = self.jobs.limit_start(device, call.call_args[0])
            if refused is not None:
                if server.trace is not None:
                    server.trace.record(call.tool.name, False, call.started, arguments=call.arguments, error="over_job_limits", result=refused,
                                        caller=call.caller, run=call.run)
                return refused
        body = _json_object(request)
        if server.endpoint_role != "operator" and body is not None and "continue" in body:
            raise ToolError("a person paused this job; only they continue it (from the operator endpoint or the GUI). Call wait_for_job, or abort if the user asked to stop.")
        return None

    def after_call(self, call: Call, result: Any) -> Any:
        if call.tool.command == "reserve" and self.jobs.job_limits:
            result = call.reply = self.jobs.limit_reservation(call.tool.device or "", result)
        elif call.tool.command == "abort":
            result = call.reply = resume_with(call.tool.device or "", result)
        return result


class JobStamp(_EnvelopeOnly):
    """A reserve or start tells the job who asked, the agent run, and (at reserve, for a dry run) the
    endpoint's high risk patterns."""

    order = 50

    def before_call(self, call: Call) -> Any:
        command = call.tool.command
        if command not in ("start", "reserve") or not call.call_args:
            return None
        overrides = self.jobs.server.config.high_overrides() if command == "reserve" else None
        first = with_caller(call.call_args[0], call.caller, call.run, overrides)
        call.call_args = (first, *call.call_args[1:])
        return None


def register(server: MCPServer, cfg: ServerConfig) -> Jobs:
    jobs = Jobs(server, cfg)
    server.add_interceptor(JobChecks(jobs))
    server.add_interceptor(JobStamp(jobs))
    return jobs
