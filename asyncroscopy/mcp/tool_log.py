"""Append-only JSONL log of MCP tool calls."""

from __future__ import annotations

import json
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

RESULT_PREVIEW_CHARS = 500


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class ToolCallLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        tool: str,
        arguments: dict[str, Any],
        started: float,
        duration_s: float,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        # Never let a logging failure fail the device call.
        try:
            row = {
                "ts": started,
                "iso": datetime.fromtimestamp(started).isoformat(),
                "tool": tool,
                "arguments": _json_safe(arguments),
                "duration_s": round(duration_s, 6),
                "ok": error is None,
                "error": f"{type(error).__name__}: {error}" if error is not None else None,
                "result": None if result is None else str(result)[:RESULT_PREVIEW_CHARS],
            }
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except Exception as exc:
            warnings.warn(f"tool log {self.path} not written: {exc}", RuntimeWarning, stacklevel=2)


def read_tool_log(path: str | Path, start: float | None = None, end: float | None = None) -> list[dict[str, Any]]:
    """Rows whose start time falls in [start, end]; a missing file reads as empty, a damaged line is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a line cut short by a crash or a concurrent write
            continue
        if start is not None and row["ts"] < start:
            continue
        if end is not None and row["ts"] > end:
            continue
        rows.append(row)
    return rows
