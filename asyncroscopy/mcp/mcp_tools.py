"""MCP server tools as LangChain tools, through the fastmcp client.

The agent swarm used ``langchain_mcp_adapters`` for this, but its released
versions need ``mcp`` 1.x while the instrument server's ``fastmcp`` needs
``mcp`` 2.x, so no environment satisfies both. This loader uses only
``fastmcp.Client`` (already a core dependency) and ``langchain_core``.

Each MCP tool becomes a ``StructuredTool`` whose argument schema is the
tool's own JSON schema. A call returns the tool's structured data (as JSON
text when it is not already a string), or its text content. Image content
(the acquisition previews) is reported as a placeholder line, since chat
models here receive text. A tool error comes back as text starting with
``Error:``, so the agent sees it and the graph keeps running, as the adapter
behaved.
"""

from __future__ import annotations

import json
from typing import Any


def _result_text(result: Any) -> str:
    if getattr(result, "is_error", False):
        texts = [getattr(c, "text", "") for c in getattr(result, "content", []) or []]
        return "Error: " + (" ".join(t for t in texts if t) or "tool failed")
    data = getattr(result, "data", None)
    parts = []
    if data is not None:
        parts.append(data if isinstance(data, str) else json.dumps(data, default=str))
    for block in getattr(result, "content", []) or []:
        kind = getattr(block, "type", "")
        if kind == "text" and data is None:
            parts.append(block.text)
        elif kind == "image":
            parts.append(f"[image: {getattr(block, 'mime_type', None) or 'image'}]")
    return "\n".join(parts) if parts else ""


class McpToolSource:
    """One open fastmcp client and the LangChain tools built from its tool list."""

    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url
        self.timeout = timeout
        self.client = None
        self.tools: list = []

    async def open(self) -> list:
        from fastmcp import Client
        from langchain_core.tools import StructuredTool

        self.client = Client(self.url, timeout=self.timeout)
        await self.client.__aenter__()
        client = self.client
        tools = []
        for spec in await client.list_tools():
            async def run(_tool_name: str = spec.name, **kwargs) -> str:
                return _result_text(await client.call_tool(_tool_name, kwargs, raise_on_error=False))

            schema = dict(spec.inputSchema or {"type": "object", "properties": {}})
            schema.setdefault("type", "object")
            schema.setdefault("properties", {})
            tools.append(StructuredTool.from_function(coroutine=run, name=spec.name,
                                                      description=spec.description or spec.name,
                                                      args_schema=schema))
        self.tools = tools
        return tools

    async def close(self) -> None:
        if self.client is not None:
            try:
                await self.client.__aexit__(None, None, None)
            finally:
                self.client = None
