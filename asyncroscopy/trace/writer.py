"""Append-only JSONL logs: the per-call tool log and the Instrument Trace.

Every line gets the file's next ``seq`` and the Unix time it was ``written``
(``asyncroscopy.core.fileio.SequencedAppender``), so a reader can follow several files at once
and merge them in order (``asyncroscopy.trace.stream``). A line is never written without its
``seq``: when the file's lock cannot be had after a few attempts the line is dropped with a
RuntimeWarning instead.
"""

from __future__ import annotations

import json
import os
import socket
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from asyncroscopy.contracts import ENVELOPE_VERSION
from asyncroscopy.core.fileio import SequencedAppender

RESULT_PREVIEW_CHARS = 500


def _plain(value: Any) -> Any:
    """numpy scalars and arrays as plain numbers and lists; anything else as text."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_plain))


class ToolCallLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._out = SequencedAppender(self.path)

    def record(
        self,
        tool: str,
        arguments: dict[str, Any],
        started: float,
        duration_s: float,
        result: Any = None,
        error: BaseException | None = None,
        **context: Any,
    ) -> None:
        """One line per call. ``context``: e.g. caller and run, written when not None."""
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
                "result": None if result is None else (result if isinstance(result, str) else json.dumps(_json_safe(result), default=str))[:RESULT_PREVIEW_CHARS],
                **{k: _json_safe(v) for k, v in context.items() if v is not None},
            }
            self._out.append(row)
        except Exception as exc:
            warnings.warn(f"tool log {self.path} not written: {exc}", RuntimeWarning, stacklevel=2)


class TraceWriter:
    """Instrument Trace: one ``trace_event.schema.json`` line per action.

    Writes go to a local JSONL file first, so the trace survives a Tiled outage;
    jobs copy their slice of it into the run file in Tiled when they finish.

    ``writer`` names who writes (a Tango device name, an MCP server); ``defaults`` are fields
    every event carries unless the call gives its own (``simulated``); ``clock`` returns the
    instrument's ``clock_s`` (None when it cannot be read), else ``clock_s`` is this computer's
    time and ``clock`` says "wall".
    """

    def __init__(
        self,
        path: str | Path,
        source: str,
        writer: str | None = None,
        defaults: dict[str, Any] | None = None,
        clock: Callable[[], float | None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._out = SequencedAppender(self.path)
        self.source = source
        self.writer = writer or f"{source}@{socket.gethostname()}:{os.getpid()}"
        self.defaults = dict(defaults or {})
        self.clock = clock

    def _instrument_clock(self) -> float | None:
        if self.clock is None:
            return None
        try:
            value = self.clock()
            return None if value is None else float(value)
        except Exception:
            return None

    def record(self, action: str, ok: bool, started: float, **fields: Any) -> dict[str, Any]:
        event = {
            "schema_version": ENVELOPE_VERSION,
            "ts": started,
            "iso": datetime.fromtimestamp(started).isoformat(),
            "source": self.source,
            "writer": self.writer,
            "action": action,
            "ok": bool(ok),
        }
        for key, value in {**self.defaults, **fields}.items():
            if key == "result" and value is not None:
                # JSON, not Python repr, so analysis code can parse previews that were not cut short.
                value = (value if isinstance(value, str) else json.dumps(_json_safe(value), default=str))[:RESULT_PREVIEW_CHARS]
            if key == "run" and not value:
                continue  # no agent run known: leave it out rather than write null
            event[key] = _json_safe(value)
        if "clock_s" in event:
            event.setdefault("clock", "instrument")
        else:
            clock = self._instrument_clock()
            event["clock_s"], event["clock"] = (clock, "instrument") if clock is not None else (time.time(), "wall")
        try:
            self._out.append(event)
        except Exception as exc:
            warnings.warn(f"trace {self.path} not written: {exc}", RuntimeWarning, stacklevel=2)
        return event


def read_tool_log(path: str | Path, start: float | None = None, end: float | None = None) -> list[dict[str, Any]]:
    """Rows whose start time falls in [start, end]; a missing file reads as empty, a damaged line is skipped.

    One file, as written. To follow several trace files at once, in order, use
    ``asyncroscopy.trace.stream``.
    """
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
