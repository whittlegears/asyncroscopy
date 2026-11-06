"""What this endpoint offers, for models and for agent programs: find_tools, call_tool, endpoint_info, action_space.

All four read the tool descriptors built at discovery (``asyncroscopy.mcp.descriptors``) and the one
approval decision (``asyncroscopy.safety.decide``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult, tool

from asyncroscopy.safety.decide import decide

if TYPE_CHECKING:
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

# What find_tools says for each outcome of the approval decision.
WHEN = {
    "runs": "runs",
    "job": "runs when every command the job drives runs",
    "refused": "refused: the approval service is not answering",
    "held": "held",
}
# What endpoint_info's "auto" says for each outcome in auto mode with nothing switched on.
AUTO = {"runs": "routine", "job": "job", "held": "held"}


class Discovery:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.analysis_url = cfg.analysis_url

    def natives(self) -> list:
        return [self.endpoint_info, self.action_space, self.find_tools, self.call_tool]

    @tool()
    def endpoint_info(self) -> dict[str, Any]:
        """For agent programs, not models: each tool's risk as the approval gate sees it, the job devices, the
        analysis server, agent_tools: globs of the tools (here and on the analysis server) to give a model, and
        auto: for each high-risk tool an agent may call, how auto mode treats it before any is switched on
        ("routine" runs, "held" waits for a person, "job" runs when every command the job drives runs)."""
        from asyncroscopy.mcp.agent_tools import agent_tool_patterns

        server = self.server
        tools = server.descriptors
        return {
            "role": server.endpoint_role,
            "analysis_url": self.analysis_url,
            "job_devices": server.jobs.devices(),
            "stem_devices": sorted(p for p, d in server.devices.items() if d.stem),
            "job_limits": server.jobs.job_limits,
            "risks": {name: t.risk for name, t in sorted(tools.items())},
            # The general agent's own selection (asyncroscopy.mcp.agent_tools), so other harnesses match it.
            "agent_tools": agent_tool_patterns(server.tool_names(), bool(self.analysis_url)),
            "auto": {
                name: AUTO[decide(t, endpoint_role="agent", mode="auto")]
                for name, t in sorted(tools.items()) if t.risk == "high" and not t.operator_only
            },
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
        tool's name or description (empty lists them all). Each match has its description, its arguments, and
        "when": "runs" (no approval needed now) or "held" (it waits for a person to approve). Run one with call_tool.
        mode "unknown" means the approval service is not answering: high-risk tools are refused until it does."""
        server = self.server
        mode, switched = server.gate.mode_and_switches()
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
            outcome = decide(server.descriptors[name], endpoint_role=server.endpoint_role, mode=mode, switched=switched)
            matches.append({"name": name, "description": description.splitlines()[0][:300] if description else "",
                            "arguments": sorted((registered.parameters or {}).get("properties", {})), "when": WHEN[outcome]})
        return {"mode": mode, "count": len(matches), "tools": matches[:max(1, int(limit))],
                **({"note": f"{len(matches) - int(limit)} more: narrow the query"} if len(matches) > int(limit) else {})}

    @tool()
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Run any tool this instrument offers by name (find one with find_tools), with its arguments as an object.
        It goes through the same approval rules as calling the tool directly."""
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
