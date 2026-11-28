"""Tango device commands as workflow tools, named and called the way the MCP bridge exposes them.

The bridge names a command ``{Class}_{command}`` and takes one keyword argument
named after ``:param name:`` in the command's input description, or ``arg``
when there is none (``asyncroscopy/mcp/mcp_server.py``). Calling a tool with
any other keyword fails here as it would over MCP.
"""

from __future__ import annotations

import json
import re

import numpy as np
import tango


class TangoTools:
    def __init__(self, proxy, cls: str, fail_on: dict | None = None):
        """``fail_on``: {command: n} raises on the n-th call of that command (1-based)."""
        self.proxy, self.cls = proxy, cls
        self.fail_on = dict(fail_on or {})
        self.calls: list[tuple[str, object]] = []
        self.counts: dict[str, int] = {}

    def tools(self) -> dict:
        out = {}
        for info in self.proxy.command_list_query():
            match = re.search(r"(?::param|@param)\s+(\w+):", info.in_type_desc or "")
            param = match.group(1) if match else "arg"
            out[f"{self.cls}_{info.cmd_name}"] = self._tool(info.cmd_name, param, info.in_type == tango.CmdArgType.DevVoid)
        return out

    def _tool(self, name: str, param: str, void: bool):
        async def call(**kwargs):
            if set(kwargs) != (set() if void else {param}):
                raise TypeError(f"{name} takes {'no arguments' if void else param!r}, got {sorted(kwargs)}")
            self.counts[name] = self.counts.get(name, 0) + 1
            self.calls.append((name, kwargs.get(param)))
            if self.fail_on.get(name) == self.counts[name]:
                raise RuntimeError(f"injected failure in {name}")
            result = self.proxy.command_inout(name) if void else self.proxy.command_inout(name, kwargs[param])
            if isinstance(result, np.ndarray):
                result = result.tolist()
            # The bridge returns JSON text; ToolBox decodes it. (A bare list would be read as content blocks.)
            return result if isinstance(result, str) else json.dumps(result)

        return call

    def called(self, *names: str) -> list:
        return [c for c in self.calls if c[0] in names]
