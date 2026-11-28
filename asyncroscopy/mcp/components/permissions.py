"""Permissions: the agent endpoint holds calls a person must approve; the operator endpoint lists and decides them.

On the agent endpoint a middleware checks every tool call (and the tool ``call_tool`` runs) against the tool's
effective permission (``asyncroscopy.safety.permissions``) before it runs. On the operator endpoint the person's
tools: list_pending_approvals, approve_call, deny_call, get_permissions, set_permission, set_group_permission.
``docs/MCP/permissions.md`` is the contract.
"""

from __future__ import annotations

import inspect
import json
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import without_injected_parameters
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult, tool
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from asyncroscopy.core.names import STOP_ROLES
from asyncroscopy.safety.permissions import PermissionStore, effective_permission
from asyncroscopy.safety.policy import hard_blocked

if TYPE_CHECKING:
    from asyncroscopy.mcp.descriptors import ToolDescriptor
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

always_run = frozenset({"endpoint_info", "action_space", "find_tools", "call_tool", "stop_all"})


def always_runs(t: ToolDescriptor) -> bool:
    """Whether a tool is never held: the endpoint's contract tools, stop_all, and a command that declares it is a
    stop (``stop``) or declares a stop role. A role the name rules inferred (``gun_off`` reads as a stop) is not enough."""
    return t.name in always_run or t.stop or (t.role_source == "annotation" and t.action_role in STOP_ROLES)


def listed_in(rules: dict[str, list[str]], dev_class: str | None, command: str) -> bool:
    """Whether a {class or "*": [command or "Class.command"]} mapping (``mcp.needs_approval``) names this command.
    A native tool (``run_job``) has no class (None): only the "*" entries name it, by its tool name."""
    if dev_class is None:
        return any(fnmatchcase(command, e) for e in rules.get("*", []))
    qualified = f"{dev_class}.{command}"
    entries = [*rules.get("*", []), *rules.get(dev_class, [])]
    return any(fnmatchcase(command, e) or fnmatchcase(qualified, e) for e in entries)


def plan_of(arguments: dict[str, Any]) -> Any:
    """The plan of a job submission: run_job's ``plan``, or the ``plan`` in a reserve's JSON ``request``."""
    if "plan" in arguments:
        return arguments["plan"]
    try:
        return json.loads(str(arguments.get("request") or "{}")).get("plan")
    except (ValueError, AttributeError):
        return None


def arguments_model(name: str, fn: Any) -> type[BaseModel]:
    """The model of a tool function's arguments, as FastMCP validates them: their types and defaults, nothing else."""
    signature = inspect.signature(without_injected_parameters(fn), eval_str=True)
    empty = inspect.Parameter.empty
    fields = {p.name: (Any if p.annotation is empty else p.annotation, ... if p.default is empty else p.default)
              for p in signature.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    return create_model(f"{name}_arguments", __config__=ConfigDict(extra="forbid"), **fields)


def refusal(text: str, structured: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=structured, is_error=True)


class Gate(Middleware):
    """Holds or refuses a call by its tool's permission. A caller whose token may not run the tool at all is passed
    on to be refused there (the token gate, or ``call_tool``), so nobody queues a request for a tool they could never run."""

    def __init__(self, owner: Permissions) -> None:
        self.owner = owner

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        name = context.message.name
        arguments = dict(context.message.arguments or {})
        if name == "call_tool":
            name, arguments = str(arguments.get("name", "")), dict(arguments.get("arguments") or {})
        if not self.owner.server.may_call(name):
            return await call_next(context)
        held = self.owner.check(name, arguments)
        if held is not None:
            return held
        return await call_next(context)


class Permissions:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.needs_approval = cfg.needs_approval
        self.store = PermissionStore(cfg.permissions_file, server.stack_id, cfg.approval_ttl_s)
        self.models: dict[str, type[BaseModel]] = {}

    def attach(self) -> None:
        """Put the gate on the agent endpoint, inside the token check so a refused caller holds nothing."""
        if self.server.endpoint_role == "agent":
            self.server.mcp.add_middleware(Gate(self))

    def natives(self) -> list:
        if self.server.endpoint_role != "operator":
            return []
        return [self.list_pending_approvals, self.approve_call, self.deny_call, self.get_permissions, self.set_permission,
                self.set_group_permission]

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
        """The ``needs_approval`` entries that name no command of their class and no native tool: of "*", or of a class
        this stack runs (every command its devices have, offered here or not). A class the stack does not run cannot be
        checked; a shared config names the twin's and the hardware's classes alike."""
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
        """Whether this call submits a job whose plan the job device says a person approves (its reserve's
        ``approval_if``, ``JobDevice.approval_plans``): a Tuner plan whose strategy writes the corrector."""
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
        """(each tool's effective permission, the group defaults), for endpoint_info."""
        settings = self.store.settings()
        return {name: effective_permission(name, entry, settings) for name, entry in self.catalog().items()}, settings["groups"]

    def validated(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The arguments as the tool will run with them: validated, defaults filled in, each value in its declared type
        (``1`` for a float is ``1.0``), as JSON. Raises pydantic's ValidationError."""
        if name not in self.models:
            self.models[name] = arguments_model(name, self.server.registered[name].fn)
        return self.models[name].model_validate(arguments).model_dump(mode="json")

    def check(self, name: str, arguments: dict[str, Any]) -> ToolResult | None:
        """None when the call may run now, else the refusal or the held call to return in its place."""
        if name not in self.server.descriptors or name not in self.server.registered:
            return None
        entry = self.entry(name)
        entry["needs_approval"] = entry["needs_approval"] or self.plan_needs_approval(name, arguments)
        permission = effective_permission(name, entry, self.store.settings())
        if permission == "allow":
            return None
        if permission == "never":
            return refusal(f"{name} is never run from this endpoint.", {"status": "never", "tool": name})
        try:
            arguments = self.validated(name, arguments)
        except ValidationError as exc:
            return refusal(f"{name} was not run or held: its arguments do not validate: {exc}", {"status": "invalid_arguments", "tool": name})
        record = self.store.request(name, arguments, entry["acts"], entry["risk"], self.server.caller())
        if record["status"] == "approved":
            return None
        if record["status"] == "denied":
            reason = record.get("reason") or ""
            return refusal(f"{name} was denied by a person: {reason}" if reason else f"{name} was denied by a person.",
                           {"status": "denied", "request_id": record["request_id"], "reason": reason})
        held = {k: record[k] for k in ("request_id", "tool", "arguments", "arguments_sha256", "acts", "risk", "expires_at", "requested_by")}
        return refusal(f"{name} is waiting for a person to approve it (request {record['request_id']}). "
                       "Send the same call again once it is approved.", {"status": "approval_required", **held})

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
    return Permissions(server, cfg)
