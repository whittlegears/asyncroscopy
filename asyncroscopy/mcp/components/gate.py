"""The approval gate: every device call passes it, and approval_status reports on a held one.

The decision itself is ``asyncroscopy.safety.decide``; the Approvals device keeps the mode, the
switches and the grants, and ``check`` applies the same decision in auto mode.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Callable

import tango
from fastmcp.exceptions import ToolError
from fastmcp.tools import tool
from tango import DeviceProxy

from asyncroscopy.mcp import endpoint_auth
from asyncroscopy.mcp.interceptors import Call, Interceptor, describe_error
from asyncroscopy.safety.decide import decide

if TYPE_CHECKING:
    from asyncroscopy.mcp.descriptors import ToolDescriptor
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig


class ApprovalHeld(ToolError):
    """A high-risk call held for an operator; ``request_id`` is the approval request to wait on, ``why`` why
    auto mode held it (None in approve mode)."""

    def __init__(self, message: str, request_id: str, why: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.why = why


class Gate:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.address = cfg.approvals_device_address

    def natives(self) -> list:
        return [self.approval_status] if self.address is not None else []

    def approvals(self) -> DeviceProxy:
        proxy = DeviceProxy(self.address)
        proxy.set_timeout_millis(5_000)
        return proxy

    def mode_and_switches(self) -> tuple[str | None, list[str]]:
        """The approval mode and the commands switched on for auto mode; (None, []) without an Approvals device,
        ("unknown", []) when it does not answer (the gate then refuses high-risk calls)."""
        if self.address is None:
            return None, []
        try:
            approvals = self.approvals()
            mode, switched = approvals.get_mode(), json.loads(approvals.get_auto_allowed()).get("switched_on", [])
        except Exception:
            return "unknown", []
        return mode, [p for p in switched if isinstance(p, str)] if isinstance(switched, list) else []

    def check(self, tool: ToolDescriptor, arguments: dict, caller: str | None, run: str | None,
              describe: Callable[[], str] | None = None, drives: Callable[[], dict] | None = None) -> dict:
        """Apply the approval mode to one call. Returns trace fields, or raises ToolError to hold the call.

        Calls that would run even in approve mode (low risk, the operator endpoint, no Approvals device)
        pass here. Anything else is asked of the Approvals device: in auto mode a routine or switched-on
        call passes, and a job start whose plan drives only such commands (``drives``: the reserved plan's
        {device address: commands}); otherwise only an operator's grant for this exact call lets it run.
        A held call files a request and is refused with its id, so the agent can wait and retry it.
        """
        if self.server.endpoint_role == "operator":
            # The operator endpoint is the person at the GUI; nothing they do is held.
            return {"mode": None, "approved_by": "operator"}
        if decide(tool, endpoint_role=self.server.endpoint_role, mode=None if self.address is None else "approve") == "runs":
            return {"mode": None, "approved_by": None}
        try:
            approvals = self.approvals()
            mode = approvals.get_mode()
            body = {"action": tool.name, "arguments": arguments, "caller": caller, "run": run, "routine": tool.routine}
            if drives is not None:  # a job start: auto mode runs it when it runs every command the plan drives
                body.update(job_start=True, drives=self.driven(drives))
            decision = json.loads(approvals.check(json.dumps(body)))
            if decision.get("allowed"):
                return {"mode": mode, "approved_by": decision.get("approved_by"), "grant_id": decision.get("grant_id"), "request_id": decision.get("request_id")}
            held_in_auto = decision.get("held_in_auto")
            reason = None
            if describe is not None:
                try:  # what the operator is approving; outside the fingerprint, so the retry still matches
                    reason = describe()[:2000]
                except Exception as exc:
                    reason = f"(could not read what this call will do: {describe_error(exc)})"
            filed = json.loads(approvals.request(json.dumps({"action": tool.name, "arguments": arguments, "caller": caller, "run": run, "reason": reason})))
        except tango.DevFailed as exc:
            # Fail closed: without the Approvals device we cannot tell whether a person must approve.
            raise ToolError(f"{tool.name} is high risk and the approval service is unreachable ({describe_error(exc)}); not run.") from exc
        if self.server.trace is not None:
            self.server.trace.record(tool.name, False, time.time(), arguments=arguments, error="approval_required", caller=caller, run=run, mode=mode, request_id=filed["request_id"])
        raise ApprovalHeld(
            f"APPROVAL REQUIRED: {tool.name} is held for an operator (request_id={filed['request_id']}"
            + (f"; {held_in_auto}" if held_in_auto else "") + "). "
            "Call approval_status with this request_id until it is approved, then repeat exactly the same call. "
            "If it is denied, do not retry; report to the user.",
            filed["request_id"],
            held_in_auto,
        )

    def driven(self, drives: Callable[[], dict]) -> list[dict] | None:
        """The high-risk tools a reserved plan drives ({device address: commands} from ``drives``), each as
        {"tool", "routine"}; a command on a device this endpoint does not offer keeps its address and is never
        routine. None when the job does not say, or the reservation cannot be read."""
        try:
            driven = drives()
        except Exception:
            return None
        if not isinstance(driven, dict):
            return None
        prefixes: dict[str, str] = {}
        for prefix, device in self.server.devices.items():
            try:
                prefixes.setdefault(str(device.proxy.dev_name()).lower(), prefix)
            except Exception:
                continue
        found = []
        for address, commands in driven.items():
            prefix = prefixes.get(str(address).lower())
            for command in dict.fromkeys(commands or []):
                name = f"{prefix}_{command}" if prefix else f"{address}:{command}"
                tool = self.server.descriptors.get(name) if prefix else None
                if tool is None or tool.risk != "low":
                    found.append({"tool": name, "routine": bool(tool is not None and tool.routine)})
        return found

    @tool()
    def approval_status(self, request_id: str) -> dict[str, Any]:
        """State of an approval request: pending, approved, denied or expired. After 'approved', repeat the held call unchanged."""
        if self.address is None:
            return {"request_id": request_id, "state": "approved", "note": "this server has no approval mode"}
        status = json.loads(self.approvals().request_status(request_id))
        status = self.server.jobs.settle(request_id, status)
        self.server.jobs.sweep_held()
        return status


class GateInterceptor(Interceptor):
    """Holds a high-risk call for a person, or lets it run; its fields go into the call's trace events."""

    order = 40

    def __init__(self, gate: Gate) -> None:
        self.gate = gate

    def before_call(self, call: Call) -> Any:
        arguments = call.context.get("gated_arguments", call.arguments)
        call.trace.update(self.gate.check(call.tool, arguments, call.caller, call.run, call.context.get("describe"), call.context.get("drives")))
        return None


class WhoStamp(Interceptor):
    """With mcp.auth, an approve, deny or set_mode of an Approvals device records the token's subject as who."""

    order = 60

    def __init__(self, server: MCPServer) -> None:
        self.server = server

    def applies(self, tool: ToolDescriptor) -> bool:
        return self.server.auth is not None and (tool.dev_class or "").lower() == "approvals" and tool.command in endpoint_auth.DECIDING_COMMANDS

    def before_call(self, call: Call) -> Any:
        call.call_args = endpoint_auth.stamp_who(call.tool.dev_class or "", call.tool.command or "", call.call_args, call.caller)
        return None


def register(server: MCPServer, cfg: ServerConfig) -> Gate:
    gate = Gate(server, cfg)
    server.add_interceptor(GateInterceptor(gate))
    server.add_interceptor(WhoStamp(server))
    return gate
