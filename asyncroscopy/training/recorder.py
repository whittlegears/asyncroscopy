"""Record what the agent is asked and what it answers, as training data.

When the agent server has a ``training_log_dir``, every exchange is appended
to two JSON-lines files in that directory, the way Tiled collects every image:

    llm_calls.jsonl  One record per model call. "messages" holds what the model
                     was given, with its reply as the last message. "tools" holds
                     the tool definitions it was offered. This is the chat format
                     that fine-tuning tools read.
    runs.jsonl       One record per user request: the prompt, each tool call with
                     a short result, and the final answer. Runs are what get graded
                     (see dataset.py).

Each record is written as soon as it exists, so nothing is lost if the server
stops. A failed write gives a warning and never stops the agent.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Long tool results (images as text, big JSON) are cut to this length in run records.
RESULT_PREVIEW_CHARS = 2000

# LangChain message class -> OpenAI chat role.
_ROLES = {
    "SystemMessage": "system",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "AIMessageChunk": "assistant",
    "ToolMessage": "tool",
}


def _text(content: Any) -> str:
    """Message content as plain text (LangChain content can be a list of blocks)."""
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return "" if content is None else str(content)


def to_openai_message(message: Any) -> dict:
    """Convert a LangChain message, or pass through an OpenAI-style dict."""
    if isinstance(message, dict):
        return dict(message)
    role = _ROLES.get(type(message).__name__)
    if role is None:
        raise ValueError(f"Cannot record a message of type {type(message).__name__}")
    result: dict[str, Any] = {"role": role, "content": _text(getattr(message, "content", ""))}
    if role == "assistant" and getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call.get("args") or {})},
            }
            for index, call in enumerate(message.tool_calls)
        ]
    if role == "tool":
        result["tool_call_id"] = getattr(message, "tool_call_id", "") or ""
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TrainingRecorder:
    """Append agent exchanges to JSON-lines files."""

    def __init__(self, directory: str | Path, model: str = ""):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._lock = threading.Lock()
        self._run: dict | None = None  # the request in progress; requests run one at a time

    def _append(self, kind: str, record: dict) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock, (self.directory / f"{kind}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception as exc:  # never let recording stop the agent
            warnings.warn(f"training recorder could not write a record: {exc}", RuntimeWarning, stacklevel=2)

    def start_run(self, prompt: str) -> None:
        self._run = {"id": uuid.uuid4().hex, "prompt": prompt, "time": _now(), "started": time.monotonic(), "tool_calls": []}

    def record_call(self, messages: list, output: Any, agent: str = "", tools: list | None = None,
                    latency_s: float | None = None) -> None:
        """One model call: the messages it was given and its reply."""
        self._append(
            "llm_calls",
            {
                "id": uuid.uuid4().hex,
                "run_id": self._run["id"] if self._run else None,
                "time": _now(),
                "model": self.model,
                "agent": agent,
                "latency_s": latency_s,
                "tools": tools or [],
                "messages": [to_openai_message(m) for m in messages] + [to_openai_message(output)],
            },
        )

    def record_tool(self, name: str, arguments: Any, result: Any, agent: str = "") -> None:
        """One tool call in the current run; written with the run when it ends."""
        if self._run is not None:
            self._run["tool_calls"].append(
                {"agent": agent, "name": name, "arguments": arguments, "result": _text(result)[:RESULT_PREVIEW_CHARS]}
            )

    def end_run(self, final_answer: str, error: str | None = None) -> str | None:
        """Write the run record and return its id (None if no run was started)."""
        run, self._run = self._run, None
        if run is None:
            return None
        self._append(
            "runs",
            {
                "id": run["id"],
                "time": run["time"],
                "model": self.model,
                "duration_s": round(time.monotonic() - run["started"], 3),
                "prompt": run["prompt"],
                "tool_calls": run["tool_calls"],
                "final_answer": final_answer,
                "error": error,
            },
        )
        return run["id"]
