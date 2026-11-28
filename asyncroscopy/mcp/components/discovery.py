"""What this endpoint offers, for models and for agent programs: find_tools, call_tool, endpoint_info, action_space.

All four read the tool descriptors built at discovery (``asyncroscopy.mcp.descriptors``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult, tool

if TYPE_CHECKING:
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig


class Discovery:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.urls = cfg.urls
        self.agent_profile = cfg.agent_profile  # the profile whose tool selection endpoint_info reports. None means the default profile.

    def natives(self) -> list:
        return [self.endpoint_info, self.action_space, self.find_tools, self.call_tool]

    def agent_tool_globs(self) -> list[str]:
        """The globs of the tools a model is given here (the agent profile's selection), by what the discovered devices are, not their names."""
        from asyncroscopy.mcp.agent_tools import agent_tool_patterns, tool_lists

        devices = self.server.devices.values()
        return agent_tool_patterns(self.server.tool_names(), bool(self.urls["analysis"]), lists=tool_lists(self.agent_profile),
                                   stem=any(d.stem for d in devices), maps=any(d.envelope and d.dev_class == "Mapper" for d in devices))

    def device_kinds(self) -> dict[str, list[str]]:
        """Each device's kinds, as discovery found them: stem, probe, envelope (runs jobs), map (a Mapper), stage (moves one)."""
        server = self.server
        moves = {t.device for t in server.descriptors.values() if not t.native and t.action_role == "stage.move"}
        found = {}
        for prefix, device in sorted(server.devices.items()):
            kinds = device.kinds()
            if device.envelope and device.dev_class == "Mapper":
                kinds.add("map")
            if prefix in moves:
                kinds.add("stage")
            found[prefix] = sorted(kinds)
        return found

    @tool()
    def endpoint_info(self) -> dict[str, Any]:
        """For agent programs, not models: each tool's risk and what it is (tools: its
        device, command, canonical role, risk, whether it acts and, on the agent endpoint, its permission), the job
        devices, each device's kinds, the other endpoints (urls), and agent_tools: globs of the tools (here and on the
        analysis server) to give a model."""
        server = self.server
        tools = server.descriptors
        simulated = server.stack_simulated()
        agent = server.endpoint_role == "agent"
        permissions, groups = server.permissions.effective() if agent else ({}, {})
        return {
            "role": server.endpoint_role,
            "stack_id": server.stack_id,
            # Draft workflows only run where this is true.
            "simulated": simulated,
            "urls": dict(self.urls),
            "command_timeout_s": server.command_timeout_s,
            "job_devices": server.jobs.devices(),
            "device_kinds": self.device_kinds(),
            "job_limits": server.jobs.job_limits,
            "risks": {name: t.risk for name, t in sorted(tools.items())},
            **({"permission_groups": groups} if agent else {}),
            # What each tool is, so an agent program never guesses from a tool's name.
            "tools": {name: {"device": t.device, "command": t.command, "role": t.action_role, "risk": t.risk, "acts": t.acts,
                             **({"permission": permissions[name]} if name in permissions else {})}
                      for name, t in sorted(tools.items())},
            # The general agent's own tool selection, so other harnesses can match it.
            "agent_tools": self.agent_tool_globs(),
            # Devices whose declarations could not be read: only the operator endpoint offers more than State and Status.
            "devices_without_hints": sorted(p for p, d in server.devices.items() if d.hints_unreadable),
            "discovery_warnings": list(server.discovery_warnings),
        }

    @tool()
    def action_space(self) -> dict[str, Any]:
        """For agent programs, not models: what can be controlled and observed here. Every tool with its
        canonical role (stage.move, acquire.image, ...; the same on every instrument whatever the command is
        called), arguments with units and ranges, risk and whether it acts or only observes; the readable
        state; each job device's plan schema and example; and which instruments are STEM, probe or job devices."""
        from asyncroscopy.mcp.action_space import build

        return build(self.server)

    @tool()
    def find_tools(self, query: str = "", limit: int = 40) -> dict[str, Any]:
        """Every tool this instrument offers, including ones not in your tool list. query: words to look for in a
        tool's name or description (empty lists them all). Each match has its description and its arguments.
        Run one with call_tool."""
        server = self.server
        words = [w for w in re.split(r"\W+", query.lower()) if w]
        matches = []
        for name in sorted(server.registered):
            if name in ("find_tools", "call_tool") or not server.may_call(name):
                continue
            registered = server.registered[name]
            description = (registered.description or "").strip()
            text = f"{name.lower()} {description.lower()}"
            if words and not all(w in text for w in words):
                continue
            matches.append({"name": name, "description": description.splitlines()[0][:300] if description else "",
                            "arguments": sorted((registered.parameters or {}).get("properties", {}))})
        return {"count": len(matches), "tools": matches[:max(1, int(limit))],
                **({"note": f"{len(matches) - int(limit)} more: narrow the query"} if len(matches) > int(limit) else {})}

    @tool()
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Run any tool this instrument offers by name (find one with find_tools), with its arguments as an object."""
        if name in ("find_tools", "call_tool"):
            raise ToolError(f"{name} cannot be run through call_tool; call it directly")
        registered = self.server.registered.get(name)
        if registered is None:
            raise ToolError(f"no tool {name!r} here; find_tools lists what this instrument offers")
        if not self.server.may_call(name):
            raise ToolError(f"{name} refused: your token does not grant the role it needs")
        return await registered.run(dict(arguments or {}))


def register(server: MCPServer, cfg: ServerConfig) -> Discovery:
    return Discovery(server, cfg)
