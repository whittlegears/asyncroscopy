"""Stream a LangGraph agent run to a text sink (stdout by default)."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Sequence


def _text_of(content: Any) -> str:
    """Flatten a streamed chunk's content (str or content blocks) to text."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _tool_output_text(output: Any, limit: int) -> str:
    """Render a tool result for the log: message content only, truncated."""
    content = getattr(output, "content", output)
    if isinstance(content, tuple) and content:
        content = content[0]
    text = content if isinstance(content, str) else _text_of(content) or str(content)
    text = text.replace("\n", " ")
    if limit and len(text) > limit:
        return f"{text[:limit]}... [{len(text) - limit} more chars]"
    return text


async def stream_agent(
    agent_executor: Any,
    messages: Sequence[Any],
    label: str = "",
    sink: Callable[[str], None] | None = None,
    tool_output_limit: int = 500,
) -> str:
    """Run a compiled agent graph, printing tokens and tool calls as they arrive.

    Returns the final assistant text (the last model turn that produced no
    tool calls). Works with any graph that accepts ``{"messages": [...]}``.
    Tool results are logged truncated to ``tool_output_limit`` characters
    (0 disables truncation); image artifacts are never printed.
    """
    write = sink or (lambda text: print(text, end="", flush=True))
    prefix = f"[{label}] " if label else ""
    start_time = time.time()
    first_token_received = False
    final_content = ""
    streamed_any = False

    async for event in agent_executor.astream_events({"messages": list(messages)}, version="v2"):
        kind = event.get("event")

        if kind == "on_chat_model_start":
            start_time = time.time()
            first_token_received = False
            streamed_any = False

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if not first_token_received:
                ttft = time.time() - start_time
                write(f"\n{prefix}[DIAGNOSTIC]: Time to first token: {ttft:.2f}s\n")
                write(f"{prefix}[GENERATION]: ")
                first_token_received = True
            text = _text_of(getattr(chunk, "content", None))
            if text:
                streamed_any = True
                write(text)

        elif kind == "on_chat_model_end":
            output = event["data"]["output"]
            tool_calls = getattr(output, "tool_calls", None) or []
            text = _text_of(getattr(output, "content", None)).strip()
            if not streamed_any and text:
                # Models without token streaming (e.g. TangoChatModel) only report here.
                write(f"\n{prefix}[GENERATION]: {text}")
            if not tool_calls:
                final_content = text
            write("\n")

        elif kind == "on_tool_start":
            tool_input = event["data"].get("input")
            write(f"{prefix}[EXECUTING TOOL]: {event.get('name')}({tool_input})\n")

        elif kind == "on_tool_end":
            output = event["data"].get("output")
            write(f"{prefix}[TOOL RESULT]: {_tool_output_text(output, tool_output_limit)}\n")

    if final_content:
        write(f"{prefix}[FINAL ANSWER RETURNED]:\n{final_content}\n{'=' * 50}\n")
    sys.stdout.flush()
    return final_content
