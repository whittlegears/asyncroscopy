"""The record of every call: the per-call tool log, the Instrument Trace, and trace_since for the operator's GUI.

``Recorder`` is the interceptor that writes one tool-log line and one trace event per device call (with
the approval gate's fields), and tells a model to stop after the same call failed the same way twice.
"""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.tools import tool

from asyncroscopy.mcp.interceptors import Call, Interceptor, describe_error
from asyncroscopy.trace import stream as trace_stream

if TYPE_CHECKING:
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

# From this many identical failures in a row, tell the model to stop and report.
REPEAT_FAILURE_WARN_AT = 2
# Identical failures further apart than this don't count as a loop.
REPEAT_FAILURE_WINDOW_S = 120.0


class TraceTools:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.follow = list(cfg.trace_follow)

    def natives(self) -> list:
        # The GUI follows the trace here; the agent endpoint does not offer it (it records arguments of every caller).
        return [self.trace_since] if self.server.endpoint_role == "operator" and self.follow else []

    @tool()
    def trace_since(
        self, cursor: str = "", run: str | None = None, caller: str | None = None, device: str | None = None,
        job: str | None = None, limit: int = 200, from_end: bool = True,
    ) -> dict[str, Any]:
        """Instrument Trace events written since `cursor`, in order, from every trace file of the stack (operator
        endpoint only). Returns {"events": [...], "cursor": "...", "more": bool}: pass the cursor back to get the next
        events, none twice. An empty cursor starts at the current end (from_end: false, at the first event). Filters: run
        (the agent run id), caller, device (Tango name or tool prefix, e.g. Mapper), job (job_id or reservation id)."""
        if not self.follow:
            raise ToolError("this endpoint has no trace to follow (set mcp.trace or mcp.trace_follow)")
        try:
            return trace_stream.trace_since(
                self.follow, cursor=cursor, limit=max(1, min(int(limit), 2000)), from_end=from_end,
                run=run or None, caller=caller or None, device=device or None, job=job or None,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc


class Recorder(Interceptor):
    """One tool-log line and one trace event per device call; a repeated identical failure says so."""

    order = 20

    def __init__(self, server: MCPServer) -> None:
        self.server = server
        # tool name -> (arguments + error signature, consecutive count, time of last failure)
        self.failures: dict[str, tuple[str, int, float]] = {}
        self._lock = threading.Lock()

    def _via(self) -> dict[str, Any]:
        name = self.server.jobs.via_name()  # run_job's own reserve and start
        return {"via": name} if name else {}

    def after_call(self, call: Call, result: Any) -> Any:
        name = call.tool.name
        with self._lock:
            self.failures.pop(name, None)
        if self.server.tool_log is not None:
            self.server.tool_log.record(name, call.arguments, call.started, call.duration_s(), result=call.reply, caller=call.caller, run=call.run)
        if self.server.trace is not None:
            key = call.reply if isinstance(call.reply, str) and (call.tool.command or "").startswith("acquire") else None
            self.server.trace.record(name, True, call.started, arguments=call.arguments, duration_s=call.duration_s(), result=call.reply,
                                     result_key=key, caller=call.caller, run=call.run, **call.trace, **self._via())
        return result

    def on_error(self, call: Call, exc: Exception) -> Exception:
        name = call.tool.name
        if self.server.tool_log is not None:
            self.server.tool_log.record(name, call.arguments, call.started, call.duration_s(), error=exc, caller=call.caller, run=call.run)
        if self.server.trace is not None:
            self.server.trace.record(name, False, call.started, arguments=call.arguments, duration_s=call.duration_s(), error=describe_error(exc),
                                     caller=call.caller, run=call.run, **call.trace, **self._via())
        message = describe_error(exc)
        repeats = self.note_failure(name, call.arguments, message)
        if repeats >= REPEAT_FAILURE_WARN_AT:
            message += f"\nThis call has failed {repeats} times in a row. Do not call it again with the same arguments; report the error to the user."
        return RuntimeError(message)

    def note_failure(self, tool_name: str, arguments: dict, error: str) -> int:
        """Count consecutive identical failures of a tool; a success, new error or long pause resets it."""
        signature = json.dumps(arguments, sort_keys=True, default=str) + "\n" + error
        now = time.monotonic()
        with self._lock:
            previous, count, last = self.failures.get(tool_name, ("", 0, 0.0))
            repeat = previous == signature and now - last <= REPEAT_FAILURE_WINDOW_S
            count = count + 1 if repeat else 1
            self.failures[tool_name] = (signature, count, now)
        return count


def register(server: MCPServer, cfg: ServerConfig) -> TraceTools:
    traces = TraceTools(server, cfg)
    traces.recorder = Recorder(server)
    server.add_interceptor(traces.recorder)
    return traces
