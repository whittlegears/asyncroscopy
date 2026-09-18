"""MCP tool loading (via ``fastmcp.Client``) and glob-based filtering.

``fastmcp`` is a core dependency and its client can talk to the asyncroscopy
MCP server over streamable HTTP, or to a ``FastMCP`` instance in memory (handy
for tests). Each MCP tool becomes a LangChain ``StructuredTool`` whose argument
schema is the tool's JSON input schema.

``langchain-mcp-adapters`` is deliberately not used: every release pins
``mcp<2`` while fastmcp 4 (which the MCP server runs on) needs ``mcp>=2``.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
import threading
from typing import Any, Awaitable, Callable, Sequence

from langchain_core.tools import StructuredTool, ToolException

DEFAULT_CALL_TIMEOUT = 180.0  # seconds; acquisitions can take a while on real hardware
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def filter_tools(tools: Sequence[Any], patterns: Sequence[str]) -> list[Any]:
    """Return the tools whose ``name`` matches any glob in ``patterns``.

    ``"*"`` returns the original list object unchanged.
    """
    if "*" in patterns:
        return tools  # type: ignore[return-value]
    return [t for t in tools if any(fnmatch.fnmatch(t.name, pat) for pat in patterns)]


def find_tool(tools: Sequence[Any], pattern: str) -> Any:
    """Return the first tool whose name matches ``pattern`` (exact match wins, then glob).

    Raises ``KeyError`` listing the available names when nothing matches.
    """
    for tool in tools:
        if tool.name == pattern:
            return tool
    for tool in tools:
        if fnmatch.fnmatch(tool.name, pattern):
            return tool
    available = ", ".join(sorted(t.name for t in tools)) or "<none>"
    raise KeyError(f"No tool matches {pattern!r}. Available tools: {available}")


# --------------------------------------------------------------------- results
def format_call_result(result: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Turn a fastmcp ``CallToolResult`` into ``(content, artifacts)``.

    ``content`` is the structured ``data`` when the server returned some
    (dicts/lists are JSON-encoded so it can be sent back to the model), else
    the text blocks joined together. Image blocks become artifacts of the form
    ``{"type": "image", "mime_type": ..., "data": <base64>}``.
    Raises ``ToolException`` when the server flagged the call as an error.
    """
    texts: list[str] = []
    artifacts: list[dict[str, Any]] = []
    for block in getattr(result, "content", None) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text" or hasattr(block, "text"):
            text = getattr(block, "text", "")
            if text:
                texts.append(str(text))
        elif block_type == "image" or hasattr(block, "data"):
            artifacts.append(
                {
                    "type": "image",
                    "mime_type": getattr(block, "mime_type", None) or getattr(block, "mimeType", None),
                    "data": getattr(block, "data", None),
                }
            )
    text = "\n".join(texts).strip()

    if getattr(result, "is_error", False):
        raise ToolException(text or "MCP tool call failed")

    data = getattr(result, "data", None)
    if isinstance(data, dict) and set(data) == {"result"}:
        data = data["result"]  # fastmcp wraps scalar results as {"result": value}
    if data is not None and not (isinstance(data, str) and not data.strip()):
        if isinstance(data, (dict, list)):
            return json.dumps(data, default=str), artifacts
        if isinstance(data, (str, int, float, bool)):
            return str(data), artifacts
    structured = getattr(result, "structured_content", None)
    if not text and isinstance(structured, dict):
        inner = structured.get("result", structured)
        return (json.dumps(inner, default=str) if isinstance(inner, (dict, list)) else str(inner)), artifacts
    return text, artifacts


def tool_result_text(result: Any) -> str:
    """Extract the primary text from a tool result.

    Handles a plain string, a ``(content, artifact)`` tuple, a list of content
    blocks, or a dict. Acquisition tools return a Tiled key string.
    """
    if isinstance(result, tuple) and result:
        return tool_result_text(result[0])
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        if "result" in result:
            return tool_result_text(result["result"])
        if "text" in result:
            return str(result["text"]).strip()
        return json.dumps(result, default=str)
    if isinstance(result, list):
        for item in result:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict) and item.get("type") in (None, "text") and item.get("text"):
                return str(item["text"]).strip()
        return ""
    content = getattr(result, "content", None)
    if content is not None and not isinstance(result, (int, float, bool)):
        return tool_result_text(content)
    return str(result).strip()


def tool_result_dict(result: Any) -> dict[str, Any]:
    """Coerce a structured tool result into a dict (parsing JSON text if needed)."""
    if isinstance(result, tuple) and result:
        return tool_result_dict(result[0])
    if isinstance(result, dict):
        return result
    text = tool_result_text(result)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {"text": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


async def invoke_tool_strict(tool: Any, args: dict[str, Any]) -> Any:
    """Invoke a LangChain tool and *raise* on tool errors.

    ``BaseTool.ainvoke`` swallows ``ToolException`` into an error string when
    ``handle_tool_error`` is set (which is what the ReAct agent wants). The
    deterministic workflows need real exceptions, so this calls the underlying
    coroutine/function directly.
    """
    coroutine = getattr(tool, "coroutine", None)
    if coroutine is not None:
        return await coroutine(**args)
    func = getattr(tool, "func", None)
    if func is not None:
        return await asyncio.to_thread(func, **args)
    return await tool.ainvoke(args)


# ---------------------------------------------------------------------- loading
def _run_coroutine_in_thread(factory: Callable[[], Awaitable[Any]]) -> Any:
    """Run a coroutine to completion on a private event loop in a worker thread."""
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=runner, name="asyncroscopy-mcp-call", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def make_mcp_tool(target: Any, tool_info: Any, call_timeout: float = DEFAULT_CALL_TIMEOUT) -> StructuredTool:
    """Wrap one MCP tool description as a LangChain ``StructuredTool``.

    ``target`` is whatever ``fastmcp.Client`` accepts: a streamable-HTTP URL,
    a ``FastMCP`` server instance (in-memory), or a transport object. A new
    client session is opened per call, so tools stay valid across notebook
    cells and graph runs without holding a connection open.
    """
    from fastmcp import Client

    name = tool_info.name
    description = (getattr(tool_info, "description", None) or f"MCP tool {name}").strip()
    schema = getattr(tool_info, "input_schema", None) or getattr(tool_info, "inputSchema", None) or _EMPTY_SCHEMA
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        schema = _EMPTY_SCHEMA
    schema = {**schema, "properties": dict(schema.get("properties") or {})}

    async def _call_async(**kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        async with Client(target, timeout=call_timeout) as client:
            result = await client.call_tool(name, kwargs or None, raise_on_error=False)
        return format_call_result(result)

    def _call_sync(**kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        return _run_coroutine_in_thread(lambda: _call_async(**kwargs))

    return StructuredTool.from_function(
        func=_call_sync,
        coroutine=_call_async,
        name=name,
        description=description,
        args_schema=schema,
        response_format="content_and_artifact",
        handle_tool_error=True,
        metadata={"mcp_target": str(target)},
    )


async def load_mcp_tools(
    target: Any,
    transport: str = "streamable_http",
    timeout: float = 15.0,
    call_timeout: float = DEFAULT_CALL_TIMEOUT,
) -> list[StructuredTool]:
    """Connect to an MCP server and return its tools as LangChain tools.

    Args:
        target: server URL (e.g. ``http://127.0.0.1:8000/mcp``) or a ``FastMCP``
            instance for in-memory use.
        transport: kept for backwards compatibility with the LLM device's
            ``ConnectMCP`` JSON; only ``streamable_http``/``http`` are meaningful
            (fastmcp infers the transport from the URL).
        timeout: seconds to wait for the initial connection + tool listing.
        call_timeout: per-call timeout applied to every tool invocation.
    """
    if transport not in ("streamable_http", "streamable-http", "http", "sse", "stdio", "memory", None):
        raise ValueError(f"Unsupported MCP transport {transport!r}")
    from fastmcp import Client

    async def _list() -> list[Any]:
        async with Client(target, timeout=timeout, init_timeout=timeout) as client:
            return await client.list_tools()

    infos = await asyncio.wait_for(_list(), timeout=timeout + 1.0)
    return [make_mcp_tool(target, info, call_timeout=call_timeout) for info in infos]
