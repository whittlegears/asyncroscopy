"""Find and call tools by command name, whatever server or instrument class they come from.

The instrument MCP bridge names a Tango command ``{Class}_{command}`` (for
example ``DigitalTwin_acquire_map_grid``); the mapping server's tools are bare
(``start_map``). ``ToolBox.get("acquire_map_grid")`` finds either form.

Tools may be LangChain tools (anything with ``ainvoke``), async functions or
plain functions, so graphs can be tested with in-process fakes.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any, Iterable, Mapping

from asyncroscopy.mcp.approvals import parse_held

DEFAULT_APPROVAL_WAIT_S = 1800.0
APPROVAL_POLL_S = 2.0


class ToolNotFound(LookupError):
    pass


class ToolCallError(RuntimeError):
    """A tool failed. ``tool`` names it; the message is the tool's own error text."""

    def __init__(self, tool: str, message: str):
        super().__init__(f"{tool}: {message}")
        self.tool = tool


def _name(tool: Any) -> str:
    return getattr(tool, "name", None) or getattr(tool, "__name__", "")


def _is_text_block(block: Any) -> bool:
    """An MCP/LangChain text content block, as opposed to a value that happens to be in a list."""
    if isinstance(block, dict):
        return block.get("type") == "text" and "text" in block
    return getattr(block, "type", None) == "text" and hasattr(block, "text")


def normalise_result(result: Any) -> Any:
    """MCP results arrive as JSON text, content blocks, or (content, artifact); return the Python value."""
    if isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
        if isinstance(artifact, dict):
            structured = artifact.get("structured_content")
        else:
            structured = getattr(artifact, "structured_content", None)
        if structured is not None:
            return structured.get("result", structured) if isinstance(structured, dict) else structured
        result = content
    content = getattr(result, "content", result)  # ToolMessage
    if isinstance(content, list) and content and all(_is_text_block(b) for b in content):
        texts = [b.get("text", "") if isinstance(b, dict) else str(b.text) for b in content]
        content = "".join(texts) if len(texts) != 1 else texts[0]
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content
    return content


class ToolBox:
    """Tools by command name. A call the instrument bridge holds for approval is waited on.

    When the bridge answers ``HELD <id>`` (see asyncroscopy.mcp.approvals), ``call``
    polls ``call_status`` until an operator approves or denies it, up to
    ``approval_wait_s``, and returns the approved call's result.
    """

    def __init__(self, tools: Iterable[Any] | Mapping[str, Any], prefer: Iterable[str] = (),
                 approval_wait_s: float = DEFAULT_APPROVAL_WAIT_S, poll_s: float = APPROVAL_POLL_S):
        items = tools.items() if isinstance(tools, Mapping) else ((_name(t), t) for t in tools)
        self.tools = {str(name): tool for name, tool in items}
        self.prefer = list(prefer)
        self.approval_wait_s = float(approval_wait_s)
        self.poll_s = float(poll_s)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def find(self, command: str) -> str | None:
        if command in self.tools:
            return command
        matches = sorted(name for name in self.tools if name.endswith("_" + command))
        for prefix in self.prefer:
            for name in matches:
                if name.startswith(prefix + "_"):
                    return name
        return matches[0] if matches else None

    def has(self, command: str) -> bool:
        return self.find(command) is not None

    def get(self, command: str) -> Any:
        name = self.find(command)
        if name is None:
            raise ToolNotFound(f"No tool for {command!r}; available: {', '.join(self.names()) or 'none'}")
        return self.tools[name]

    async def call(self, command: str, **arguments: Any) -> Any:
        """Call a tool and return its normalised result; raise ToolCallError when it fails."""
        tool = self.get(command)
        try:
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(arguments)
            else:
                result = tool(**arguments)
                if inspect.isawaitable(result):
                    result = await result
        except ToolNotFound:
            raise
        except Exception as exc:  # MCP and Tango errors come in many types
            call_id = parse_held(str(exc))
            if call_id and self.has("call_status"):
                return await self._wait_for_approval(command, call_id)
            raise ToolCallError(self.find(command) or command, str(exc)) from exc
        return normalise_result(result)

    async def _wait_for_approval(self, command: str, call_id: str) -> Any:
        name = self.find(command) or command
        deadline = time.monotonic() + self.approval_wait_s
        while True:
            status = await self.call("call_status", call_id=call_id)
            state = status.get("status") if isinstance(status, dict) else None
            if state == "done":
                return status.get("result")
            if state in ("denied", "failed"):
                reason = status.get("note") if state == "denied" else status.get("error")
                raise ToolCallError(name, f"held call {call_id} was {state}: {reason or 'no reason given'}")
            if time.monotonic() > deadline:
                raise ToolCallError(name, f"held call {call_id} still {state} after {self.approval_wait_s:.0f} s")
            await asyncio.sleep(self.poll_s)
