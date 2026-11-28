"""The permission gate on the agent endpoint and the person's permission tools on the operator endpoint;
``docs/MCP/permissions.md`` and ``docs/MCP/twin_to_instrument.md`` are the contract."""

from __future__ import annotations

import inspect
import json
import time
from contextvars import ContextVar
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import without_injected_parameters
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult, tool
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from asyncroscopy.core.names import RUN_KEY_HEADER, STOP_ROLES
from asyncroscopy.mcp.interceptors import Call, Interceptor
from asyncroscopy.safety.permissions import (
    PermissionStore,
    arguments_sha256,
    effective_permission,
    run_key_sha256,
)
from asyncroscopy.safety.policy import hard_blocked

if TYPE_CHECKING:
    from asyncroscopy.mcp.descriptors import ToolDescriptor
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

always_run = frozenset({"endpoint_info", "action_space", "find_tools", "call_tool", "stop_all"})


def always_runs(t: ToolDescriptor) -> bool:
    """Never held: contract tools, stop_all, and commands that declare a stop; a stop role inferred from a name is not enough."""
    return t.name in always_run or t.stop or (t.role_source == "annotation" and t.action_role in STOP_ROLES)


def listed_in(rules: dict[str, list[str]], dev_class: str | None, command: str) -> bool:
    """Whether ``mcp.needs_approval`` names this command; a native tool (no class) only through "*" entries."""
    if dev_class is None:
        return any(fnmatchcase(command, e) for e in rules.get("*", []))
    qualified = f"{dev_class}.{command}"
    entries = [*rules.get("*", []), *rules.get(dev_class, [])]
    return any(fnmatchcase(command, e) or fnmatchcase(qualified, e) for e in entries)


def plan_of(arguments: dict[str, Any]) -> Any:
    if "plan" in arguments:
        return arguments["plan"]
    try:
        return json.loads(str(arguments.get("request") or "{}")).get("plan")
    except (ValueError, AttributeError):
        return None


def arguments_model(name: str, fn: Any) -> type[BaseModel]:
    signature = inspect.signature(without_injected_parameters(fn), eval_str=True)
    empty = inspect.Parameter.empty
    fields = {p.name: (Any if p.annotation is empty else p.annotation, ... if p.default is empty else p.default)
              for p in signature.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    return create_model(f"{name}_arguments", __config__=ConfigDict(extra="forbid"), **fields)


signed_off: ContextVar[dict[str, Any] | None] = ContextVar("signed_off", default=None)


def refusal(text: str, structured: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=structured, is_error=True)


class Gate(Middleware):
    """Holds or refuses a call by its tool's permission; a caller who may not run the tool at all is refused downstream."""

    def __init__(self, owner: Permissions) -> None:
        self.owner = owner

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        name = context.message.name
        arguments = dict(context.message.arguments or {})
        if name == "call_tool":
            name, arguments = str(arguments.get("name", "")), dict(arguments.get("arguments") or {})
        if not self.owner.server.may_call(name):
            return await call_next(context)
        held, signoff = self.owner.decide(name, arguments)
        if held is not None:
            return held
        token = signed_off.set(signoff)
        try:
            return await call_next(context)
        finally:
            signed_off.reset(token)


class SignoffStamp(Interceptor):

    order = 15

    def before_call(self, call: Call) -> Any:
        found = signed_off.get()
        if found:
            call.trace.update(approved_by=found["approved_by"], approver=found["approver"])
        return None


class Permissions:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.needs_approval = cfg.needs_approval
        self.store = PermissionStore(cfg.permissions_file, server.stack_id, cfg.approval_ttl_s)
        self.models: dict[str, type[BaseModel]] = {}
        self.covered_by_grant: dict[tuple[str, str], dict[str, dict[str, int]]] = {}

    def attach(self) -> None:
        if self.server.endpoint_role in ("agent", "monitor"):
            self.server.mcp.add_middleware(Gate(self))

    def natives(self) -> list:
        if self.server.endpoint_role != "operator":
            return []
        return [self.list_pending_approvals, self.approve_call, self.deny_call, self.get_permissions, self.set_permission,
                self.set_group_permission, self.preapprove_run, self.end_run_preapproval]

    def entry(self, name: str) -> dict[str, Any]:
        t = self.server.descriptors[name]
        registered = self.server.registered.get(name)
        description = ((registered.description if registered is not None else "") or "").strip()
        device_command = not t.native and t.dev_class is not None and t.command is not None
        listed = listed_in(self.needs_approval, t.dev_class, t.command) if device_command else t.native and listed_in(self.needs_approval, None, name)
        return {"acts": t.acts, "risk": t.risk, "description": description.splitlines()[0][:300] if description else "",
                "needs_approval": listed,
                "never": device_command and hard_blocked(t.dev_class, t.command),
                "fixed": always_runs(t)}

    def catalog(self) -> dict[str, dict[str, Any]]:
        return {name: self.entry(name) for name in sorted(self.server.descriptors) if name in self.server.registered}

    def unmatched(self) -> list[str]:
        """``needs_approval`` entries matching nothing; classes this stack does not run are skipped, as a shared config
        names the twin's and the hardware's alike."""
        commands: dict[str, set[str]] = {}
        for device in self.server.devices.values():
            commands.setdefault(device.dev_class, set()).update(device.commands)
        natives = {name for name, t in self.server.descriptors.items() if t.native}
        found = []
        for dev_class, entries in self.needs_approval.items():
            if dev_class != "*" and dev_class not in commands:
                continue
            classes = commands if dev_class == "*" else {dev_class: commands[dev_class]}
            for e in entries:
                hit = any(listed_in({dev_class: [e]}, c, command) for c, names in classes.items() for command in names)
                if not hit and not (dev_class == "*" and any(listed_in({"*": [e]}, None, n) for n in natives)):
                    found.append(f"{dev_class}: {e}")
        return found

    def publish(self) -> None:
        if self.server.endpoint_role != "agent":
            return
        unmatched = self.unmatched()
        if unmatched:
            raise ValueError(f"mcp.needs_approval names no command of its class and no tool of this endpoint: {', '.join(unmatched)}")
        self.store.publish(self.catalog())

    def plan_needs_approval(self, name: str, arguments: dict[str, Any]) -> bool:
        """Whether this call submits a job whose plan matches its device's ``approval_if`` (``JobDevice.approval_plans``)."""
        t = self.server.descriptors[name]
        if t.native and name == "run_job":
            device = self.server.devices.get(str(arguments.get("device")))
        elif not t.native and t.command == "reserve":
            device = self.server.devices.get(str(t.device))
        else:
            return False
        if device is None or not device.envelope:
            return False
        rules = (device.hints.get("reserve") or {}).get("approval_if") or {}
        plan = plan_of(arguments)
        plan = plan if isinstance(plan, dict) else {}
        return any(plan.get(field, rule.get("default")) in rule.get("values", []) for field, rule in rules.items())

    def effective(self) -> tuple[dict[str, str], dict[str, str]]:
        settings = self.store.settings()
        return {name: effective_permission(name, entry, settings) for name, entry in self.catalog().items()}, settings["groups"]

    def validated(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The arguments as FastMCP will run them: validated, defaults filled in, as JSON."""
        if name not in self.models:
            self.models[name] = arguments_model(name, self.server.registered[name].fn)
        return self.models[name].model_validate(arguments).model_dump(mode="json")

    def rehearsed(self, grant: dict[str, Any]) -> dict[str, dict[str, int]]:
        """{tool: {digest: count}} of a grant's calls, validated once per grant; an invalid one covers nothing."""
        key = (grant["run_id"], grant["granted_at"])
        if key not in self.covered_by_grant:
            covered: dict[str, dict[str, int]] = {}
            for call in grant["calls"]:
                tool = call["tool"]
                if tool not in self.server.registered:
                    continue
                try:
                    digest = arguments_sha256(self.validated(tool, call["arguments"]))
                except ValidationError:
                    continue
                counts = covered.setdefault(tool, {})
                counts[digest] = counts.get(digest, 0) + int(call.get("count") or 1)
            self.covered_by_grant = {k: v for k, v in self.covered_by_grant.items() if k[0] != grant["run_id"]}
            self.covered_by_grant[key] = covered
        return self.covered_by_grant[key]

    def decide(self, name: str, arguments: dict[str, Any]) -> tuple[ToolResult | None, dict[str, Any] | None]:
        """(None or the refusal to return instead, the sign-off that let the call run)."""
        if name not in self.server.descriptors or name not in self.server.registered:
            return None, None
        run_id, run_key = self.server.run_id(), self.server._header(RUN_KEY_HEADER)
        grant = self.store.live_grant(run_id, run_key_sha256(run_key)) if run_id and run_key else None
        entry = self.entry(name)
        entry["needs_approval"] = entry["needs_approval"] or self.plan_needs_approval(name, arguments)
        permission = effective_permission(name, entry, self.store.settings())
        if permission == "allow":
            return None, None
        if permission == "never":
            return refusal(f"{name} is never run from this endpoint.", {"status": "never", "tool": name}), None
        try:
            arguments = self.validated(name, arguments)
        except ValidationError as exc:
            return refusal(f"{name} was not run or held: its arguments do not validate: {exc}", {"status": "invalid_arguments", "tool": name}), None
        signoff = run = None
        if grant is not None:
            run = {"run_id": grant["run_id"], "run_key_sha256": grant["run_key_sha256"]}
            covered = self.rehearsed(grant)
            digest = arguments_sha256(arguments)
            allowed = covered.get(name, {}).get(digest, 0)
            if allowed and self.store.use_run_call(grant["run_id"], grant["run_key_sha256"], f"{name} {digest}", allowed):
                return None, {"approved_by": grant["approved_by"], "approver": grant["approver"], "run_id": grant["run_id"]}
            why = ("the rehearsal made this call fewer times" if allowed else
                   "the rehearsal called this tool with other arguments" if name in covered else "the rehearsal never called this tool")
            signoff = {"approved_by": grant["approved_by"], "approver": grant["approver"], "run_id": grant["run_id"], "why_held": why}
        record = self.store.request(name, arguments, entry["acts"], entry["risk"], self.server.caller(), signoff=signoff, run=run)
        if record["status"] == "approved":
            return None, None
        if record["status"] == "denied":
            reason = record.get("reason") or ""
            return refusal(f"{name} was denied by a person: {reason}" if reason else f"{name} was denied by a person.",
                           {"status": "denied", "request_id": record["request_id"], "reason": reason}), None
        held = {k: record[k] for k in ("request_id", "tool", "arguments", "arguments_sha256", "acts", "risk", "expires_at", "requested_by")}
        if record.get("signoff"):
            held["signoff"] = record["signoff"]
        return refusal(f"{name} is waiting for a person to approve it (request {record['request_id']}). "
                       "Send the same call again once it is approved.", {"status": "approval_required", **held}), None

    @tool()
    def list_pending_approvals(self) -> list[dict[str, Any]]:
        """The agent endpoint's calls waiting for a person to approve or deny them, oldest first."""
        return self.store.pending()

    @tool()
    def approve_call(self, request_id: str) -> dict[str, Any]:
        """Approve one held call. The agent's next identical call runs, once."""
        try:
            return self.store.approve(request_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from None

    @tool()
    def deny_call(self, request_id: str, reason: str) -> dict[str, Any]:
        """Deny one held call, with the reason the agent is told."""
        try:
            return self.store.deny(request_id, reason)
        except ValueError as exc:
            raise ToolError(str(exc)) from None

    @tool()
    def preapprove_run(self, run_id: str, run_key_sha256: str, calls: list[dict[str, Any]], approved_by: str, approver: str,
                       workflow: str, hash: str, rehearsal_run_id: str | None = None) -> dict[str, Any]:
        """For a workflow server, not a model: pre-approve, for one hardware run of a signed-off workflow, every call its
        passing rehearsal made ({"tool", "arguments", "count"} each; count defaults to 1). Only calls of that run (run id
        and run key) of the same tool with the same validated arguments pass the gate, each up to its count; any other
        call is held, never tools stay refused. Kept alive by the run's calls for approval_ttl_s, for at most 12 hours,
        or until end_run_preapproval."""
        try:
            grant = self.store.grant_run(run_id, run_key_sha256, list(calls or []), approved_by, approver, workflow, hash, rehearsal_run_id)
        except (ValueError, KeyError, TypeError) as exc:
            raise ToolError(str(exc)) from None
        self.audit("preapprove_run", grant)
        return grant

    @tool()
    def end_run_preapproval(self, run_id: str) -> dict[str, Any]:
        """For a workflow server, not a model: remove a hardware run's pre-approval when the run ends or is stopped."""
        removed = self.store.end_run(run_id)
        if removed is not None:
            self.audit("end_run_preapproval", removed)
        return {"run_id": str(run_id), "removed": removed}

    def audit(self, action: str, grant: dict[str, Any]) -> None:
        if self.server.trace is not None:
            self.server.trace.record(action, True, time.time(), run=grant["run_id"], caller=self.server.caller(),
                                     approved_by=grant["approved_by"], approver=grant["approver"], workflow=grant["workflow"],
                                     workflow_hash=grant["hash"], calls=grant["calls"], expires_at=grant["expires_at"])

    @tool()
    def get_permissions(self) -> dict[str, Any]:
        """The group defaults (reads, acts) and each agent-endpoint tool's permission: allow, approve or never."""
        return self.store.permissions()

    @tool()
    def set_permission(self, tool: str, permission: str) -> dict[str, Any]:
        """Set one agent-endpoint tool to allow or approve, or to default to clear the setting."""
        try:
            self.store.set_tool(tool, permission)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        return self.store.permissions()

    @tool()
    def set_group_permission(self, group: str, permission: str) -> dict[str, Any]:
        """Set the default of a group (reads or acts) to allow or approve."""
        try:
            self.store.set_group(group, permission)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        return self.store.permissions()


def register(server: MCPServer, cfg: ServerConfig) -> Permissions:
    permissions = Permissions(server, cfg)
    server.add_interceptor(SignoffStamp())
    return permissions
