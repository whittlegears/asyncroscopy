"""MCP tools as LangChain tools, using fastmcp's client.

langchain-mcp-adapters pins an older ``mcp`` package than the fastmcp
version this project serves with, and the two cannot be installed together.
This loader needs only fastmcp (already a dependency) and langchain-core.
Each call opens a short-lived session, so tools keep working across server
restarts.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport
from langchain_core.tools import StructuredTool, ToolException

from asyncroscopy import settings
from asyncroscopy.agents.guidance import render

if TYPE_CHECKING:
    from mcp.types import Tool


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


def result_images(result: CallToolResult) -> list[dict]:
    """The images of a fastmcp CallToolResult, each {"mime_type", "base64"}: a segmentation, the atoms found."""
    images = []
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", None) == "image" and getattr(block, "data", None):
            mime = getattr(block, "mime_type", None) or getattr(block, "mimeType", None)  # renamed in MCP SDK v2
            images.append({"mime_type": mime or "image/png", "base64": block.data})
    return images


IMAGE_NOTE = ("The reply has {count} image(s), which go to the user with your answer. You cannot see them: say what the numbers "
              "show, and that the user can check the picture.")
# The images tools returned during one request, for the answer to carry (collect_images). None outside one.
_COLLECTED: ContextVar[list | None] = ContextVar("collected_tool_images", default=None)
MAX_COLLECTED = 8  # an answer carries at most the latest this many


@contextmanager
def collect_images() -> Iterator[list[dict]]:
    """Collect, into the list it yields, the images every McpServer call returns inside this block, each
    {"tool", "mime_type", "base64"}. The agent server attaches them to its answer so the chat shows them."""
    bucket: list[dict] = []
    token = _COLLECTED.set(bucket)
    try:
        yield bucket
    finally:
        _COLLECTED.reset(token)


def tool_output(text: str, images: list[dict], show_model: bool) -> tuple[Any, dict | None]:
    """A LangChain tool's (content, artifact) for a reply's text and images.

    The artifact keeps the images for the chat to show, whatever the model. The content is the text, with the images
    as content blocks when the model reads images (``show_model``), else with a note saying the user has them.
    """
    if not images:
        return text, None
    artifact = {"images": images}
    if not show_model:
        return f"{text}\n\n{IMAGE_NOTE.format(count=len(images))}", artifact
    blocks = [{"type": "image", "base64": image["base64"], "mime_type": image["mime_type"]} for image in images]
    return [{"type": "text", "text": text}, *blocks], artifact


class McpServer:
    """One MCP endpoint: list its tools and call them, with a caller header for the trace.

    ``ledger``: a list every call is appended to as soon as its reply arrives
    ({"name", "args", "result", "error"}), so a run that is stopped part way still knows
    what it started (the goal graph reads it; see ``AgentConfig.ledger``).
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout_s: float = 120.0, ledger: list | None = None,
                 replies: str = "notes", images: bool = False) -> None:
        self.url = url
        # Whether langchain_tools shows a tool's images to the model (the profile's images); the chat gets them anyway.
        self.images = images
        # How structured replies reach a model through langchain_tools. It is the replies setting of the profile.
        # See asyncroscopy.agents.guidance.
        self.replies = replies
        self.headers = dict(headers or {})
        if not any(k.lower() == "authorization" for k in self.headers):  # a caller such as the swarm builds its own headers
            self.headers.update(bearer_header(settings.get().mcp_token))
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
        bucket = _COLLECTED.get()
        if bucket is not None:
            bucket.extend({"tool": name, **image} for image in result_images(result))
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

        async def run(**kwargs) -> tuple[Any, dict | None]:
            # The server replies with facts. render adds sentences saying what to do, or puts them in place of the facts.
            result = await server.call(name, kwargs)
            if result.is_error:
                raise ToolException(render(name, kwargs, result_text(result), error=True, replies=server.replies))
            text = render(name, kwargs, result_text(result), replies=server.replies)
            return tool_output(text, result_images(result), server.images)

        schema = dict(getattr(spec, "input_schema", None) or getattr(spec, "inputSchema", None) or {"type": "object", "properties": {}})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return StructuredTool(
            name=name,
            description=(spec.description or name)[:1024],
            args_schema=schema,
            coroutine=run,
            response_format="content_and_artifact",  # the artifact carries a reply's images for the chat
            handle_tool_error=True,  # a failed call goes back to the model as text, not a crash
        )
