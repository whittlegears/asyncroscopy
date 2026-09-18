"""Conversions between OpenAI-style chat message dicts and LangChain messages.

Used in both directions: the LLM Tango device's ``Complete`` command accepts
OpenAI-style requests (from SciAgentGUI or from ``TangoChatModel``), and
``TangoChatModel`` turns LangChain messages back into that shape.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def openai_messages_to_langchain(messages: list[dict]) -> list[BaseMessage]:
    """Convert OpenAI-style chat messages into LangChain message objects."""
    converted: list[BaseMessage] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if role == "system":
            converted.append(SystemMessage(content=content or ""))
        elif role == "assistant":
            tool_calls = [
                {
                    "name": call["function"]["name"],
                    "args": json.loads(call["function"].get("arguments") or "{}"),
                    "id": call.get("id", ""),
                }
                for call in (message.get("tool_calls") or [])
            ]
            converted.append(AIMessage(content=content or "", tool_calls=tool_calls))
        elif role == "tool":
            converted.append(
                ToolMessage(content=content or "", tool_call_id=message.get("tool_call_id", ""))
            )
        else:
            converted.append(HumanMessage(content=content or ""))
    return converted


def langchain_message_to_openai(message: BaseMessage) -> dict:
    """Convert a LangChain AIMessage into an OpenAI-style assistant message dict."""
    result: dict = {"role": "assistant", "content": _text_content(message.content)}
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        result["tool_calls"] = [
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("args") or {}),
                },
            }
            for index, call in enumerate(tool_calls)
        ]
    return result


def langchain_messages_to_openai(messages: list[BaseMessage]) -> list[dict]:
    """Convert a LangChain conversation into OpenAI-style message dicts."""
    converted: list[dict] = []
    for message in messages:
        msg_type = getattr(message, "type", None)
        if msg_type == "system" or isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": _text_content(message.content)})
        elif msg_type == "ai" or isinstance(message, AIMessage):
            converted.append(langchain_message_to_openai(message))
        elif msg_type == "tool" or isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "content": _text_content(message.content),
                    "tool_call_id": getattr(message, "tool_call_id", "") or "",
                }
            )
        else:
            converted.append({"role": "user", "content": _text_content(message.content)})
    return converted


def openai_message_to_langchain(message: dict) -> AIMessage:
    """Convert a single OpenAI-style assistant message dict into an AIMessage."""
    converted = openai_messages_to_langchain([{**message, "role": "assistant"}])
    return converted[0]  # type: ignore[return-value]


def _text_content(content: Any) -> str:
    """Flatten LangChain content (str or list of content blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in (None, "text") and "text" in block:
                    parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)
