"""Append-only JSONL logs: the per-call tool log and the Instrument Trace.

Every line gets the file's next ``seq`` and the Unix time it was ``written``, under a lock
shared by every process appending to that file, so a reader can follow several files at once
and merge them in order (``asyncroscopy.trace.stream``).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from asyncroscopy.contracts import ENVELOPE_VERSION

RESULT_PREVIEW_CHARS = 500
LOCK_TIMEOUT_S = 5.0  # longest a writer waits for another process's append before writing the line without a seq
_LOCK_BYTE = 1 << 20  # Windows locks this byte of the .seq file, past its data, so its data stays readable


def _plain(value: Any) -> Any:
    """numpy scalars and arrays as plain numbers and lists; anything else as text."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_plain))


def seq_path(path: str | Path) -> Path:
    """The sidecar holding a log's last ``seq`` and ``written`` time: ``<name>.seq`` next to it.

    Every process appending to the log shares it, and it outlives a rotation of the log, so
    ``seq`` keeps rising when ``jobs.jsonl`` is renamed to ``jobs.jsonl.1`` and a new one begun.
    """
    path = Path(path)
    return path.with_name(path.name + ".seq")


@contextmanager
def file_lock(handle, deadline: float) -> Iterator[None]:
    """An exclusive lock between processes on ``handle`` (the .seq file, opened "a+b"), waited for until ``deadline``."""
    if os.name == "nt":
        import msvcrt

        def attempt():
            handle.seek(_LOCK_BYTE)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

        def release():
            handle.seek(_LOCK_BYTE)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        def attempt():
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def release():
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    while True:
        try:
            attempt()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("another process held the log's lock too long") from None
            time.sleep(0.002)
    try:
        yield
    finally:
        release()


def _last_seq_in(path: Path) -> tuple[int, float]:
    """(highest seq, latest written) already in ``path``: how a log without a .seq sidecar is continued."""
    seq, written = 0, 0.0
    if not path.exists():
        return seq, written
    with path.open("rb") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict):
                if isinstance(row.get("seq"), int):
                    seq = max(seq, row["seq"])
                if isinstance(row.get("written"), (int, float)):
                    written = max(written, float(row["written"]))
    return seq, written


class SequencedAppender:
    """Appends JSON lines to one file, each with the file's next ``seq`` and its ``written`` time.

    ``seq`` rises by one per line of the file whichever thread or process wrote it: the sidecar
    (``seq_path``) is locked while a line is numbered and appended, so lines land in ``seq``
    order. ``written`` is the Unix time of the append, never earlier than the line before it;
    the merging reader orders the lines of several files by it. The file is opened for each
    line and closed again, so it can be renamed (rotated) between lines, on Windows too.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        """Number ``row``, append it as one line and return it. Raises OSError if the file cannot be written."""
        with self._lock:
            try:
                with seq_path(self.path).open("a+b") as counter, file_lock(counter, time.monotonic() + LOCK_TIMEOUT_S):
                    counter.seek(0)
                    stored = counter.read(100).decode("ascii", errors="replace").split()
                    try:
                        last_seq, last_written = int(stored[0]), float(stored[1])
                    except (IndexError, ValueError):  # a new sidecar (or a damaged one): continue from the log itself
                        last_seq, last_written = _last_seq_in(self.path)
                    row["seq"] = last_seq + 1
                    row["written"] = max(time.time(), last_written)
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    counter.seek(0)
                    counter.truncate()
                    counter.write(f"{row['seq']} {row['written']!r}".encode("ascii"))
                return row
            except TimeoutError as exc:
                warnings.warn(f"{self.path}: {exc}; this line has no seq", RuntimeWarning, stacklevel=3)
            row.pop("seq", None)
            row["written"] = time.time()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            return row


class ToolCallLog:
    def __init__(self, path: str | Path):
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
    ):
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
