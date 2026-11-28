"""The ranges and units each device declares, kept current in its tools' input schemas.

A device restarted with new limit properties declares new ranges in its action_hints. Each time a client lists the
tools, the hints are read again (at most every ``REREAD_AFTER_S``) and the schemas rebuilt from them; a device that
cannot be read keeps the last hints it gave. Only the schemas change: what a command is (risk, role, permission) is
still decided at discovery.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import Tool

from asyncroscopy.mcp.descriptors import read_hints, with_declared_limits

if TYPE_CHECKING:
    from asyncroscopy.mcp.mcp_server import MCPServer

REREAD_AFTER_S = 30.0


class Limits(Middleware):
    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self.plain: dict[str, tuple[str, str, dict]] = {}
        self.hints = {prefix: device.hints for prefix, device in server.devices.items()}
        self.read_at = time.monotonic()

    def add(self, tool: Tool, prefix: str, command: str) -> Tool:
        """``tool`` with its device's declared limits in its schema, remembered so a later listing can refresh them."""
        self.plain[tool.name] = (prefix, command, tool.parameters)
        return self.limited(tool)

    def limited(self, tool: Tool) -> Tool:
        prefix, command, plain = self.plain[tool.name]
        return tool.model_copy(update={"parameters": with_declared_limits(plain, self.hints.get(prefix, {}).get(command, {}))})

    def reread(self) -> None:
        if time.monotonic() - self.read_at < REREAD_AFTER_S:
            return
        self.read_at = time.monotonic()
        for prefix, device in self.server.devices.items():
            hints, _problem = read_hints(device.proxy) if device.proxy is not None else (None, None)
            if hints is not None:
                self.hints[prefix] = hints

    async def on_list_tools(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        tools = await call_next(context)
        await asyncio.to_thread(self.reread)
        return [self.limited(t) if t.name in self.plain else t for t in tools]
