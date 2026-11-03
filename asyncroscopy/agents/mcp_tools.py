"""MCP tools as LangChain tools, using fastmcp's client.

langchain-mcp-adapters pins an older ``mcp`` package than the fastmcp
version this project serves with, and the two cannot be installed together.
This loader needs only fastmcp (already a dependency) and langchain-core.
Each call opens a short-lived session, so tools keep working across server
restarts.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport
from langchain_core.tools import StructuredTool, ToolException

if TYPE_CHECKING:
    from mcp.types import Tool

# A bearer token for endpoints that verify tokens (the MCP config's mcp.auth; docs/Operation/remote_access.md).
TOKEN_ENV = "ASYNCROSCOPY_MCP_TOKEN"


def bearer_header(token: str | None) -> dict[str, str]:
    """{"Authorization": "Bearer <token>"}, or {} without a token."""
    token = (token or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def result_text(result: CallToolResult) -> str:
    """Text of a fastmcp CallToolResult: the structured result when it is a string, then any other text blocks (an
    acquisition's summary: "taken with the beam blanked", a spectrum's peaks), else the text blocks."""
    structured = getattr(result, "structured_content", None)
    texts = [getattr(block, "text", None) for block in (getattr(result, "content", None) or [])]
    if isinstance(structured, dict) and isinstance(structured.get("result"), str):
        return "\n".join([structured["result"], *(t for t in texts if t and t != structured["result"])])
    joined = "".join(t for t in texts if t)
    if joined:
        return joined
    return json.dumps(structured) if structured is not None else ""


class McpServer:
    """One MCP endpoint: list its tools and call them, with a caller header for the trace.

    ``ledger``: a list every call is appended to as soon as its reply arrives
    ({"name", "args", "result", "error"}), so a run that is stopped part way still knows
    what it started (the goal graph reads it; see ``AgentConfig.ledger``).
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout_s: float = 120.0, ledger: list | None = None) -> None:
        self.url = url
        self.headers = dict(headers or {})
        if not any(k.lower() == "authorization" for k in self.headers):  # e.g. the swarm, which builds its own headers
            self.headers.update(bearer_header(os.environ.get(TOKEN_ENV)))
        self.timeout_s = timeout_s
        self.ledger = ledger

    def _client(self) -> Client:
        return Client(StreamableHttpTransport(self.url, headers=self.headers), timeout=self.timeout_s)

    async def list_tools(self) -> list[Tool]:
        async with self._client() as client:
            return await client.list_tools()

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        """Call a tool; returns the fastmcp result (check ``is_error``) without raising on tool errors."""
        async with self._client() as client:
            result = await client.call_tool(name, arguments or {}, raise_on_error=False)
        if self.ledger is not None:
            self.ledger.append({"name": name, "args": dict(arguments or {}), "result": result_text(result)[:4000], "error": bool(result.is_error)})
        return result

    async def call_text(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = await self.call(name, arguments)
        if result.is_error:
            raise ToolException(result_text(result))
        return result_text(result)

    async def call_json(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        text = await self.call_text(name, arguments)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
        return value if isinstance(value, dict) else {"result": value}

    async def langchain_tools(self) -> list[StructuredTool]:
        tools = []
        for spec in await self.list_tools():
            tools.append(self._wrap(spec))
        return tools

    def _wrap(self, spec) -> StructuredTool:
        server, name = self, spec.name

        async def run(**kwargs) -> str:
            return await server.call_text(name, kwargs)

        schema = dict(getattr(spec, "input_schema", None) or getattr(spec, "inputSchema", None) or {"type": "object", "properties": {}})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return StructuredTool(
            name=name,
            description=(spec.description or name)[:1024],
            args_schema=schema,
            coroutine=run,
            handle_tool_error=True,  # a failed call goes back to the model as text, not a crash
        )
