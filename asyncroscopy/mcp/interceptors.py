"""Interceptors: the steps every device tool call passes through, in a list instead of one closure.

The MCP server wraps each Tango command as a tool (``MCPServer._create_wrapper``). A call of that tool
builds a ``Call`` and runs the interceptors that apply to the tool (``applies``), in order:

1. ``before_call`` of each, first to last. One may change ``call.call_args`` (what the device gets),
   add ``call.trace`` fields, raise to refuse the call, or return a reply that ends it there (the
   device is not called and no ``after_call`` runs).
2. The device command, its reply made JSON-safe (``call.reply``).
3. ``after_call`` of each, last to first. One may return a new result (a picture added to an
   acquisition's key) and, when it changes the reply itself, set ``call.reply`` (what is logged).
4. ``on_error`` of each, last to first, when the command or an ``after_call`` raised: one may return
   another exception to raise in its place.

The components add theirs (``asyncroscopy.mcp.components``); envelope-only ones apply only to job devices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from asyncroscopy.mcp.descriptors import ToolDescriptor


def describe_error(exc: Exception) -> str:
    """One-line summary of a failure; for DevFailed, the first (cause) and last (device) errors, no stack."""
    import tango

    if isinstance(exc, tango.DevFailed) and exc.args:
        errors = [exc.args[0]] + ([exc.args[-1]] if len(exc.args) > 1 else [])
        return " | ".join(f"{e.reason}: {' '.join(str(e.desc).split())}" for e in errors)
    return f"{type(exc).__name__}: {exc}".strip()


@dataclass
class Call:
    """One call of a device tool, as the interceptors see it."""

    tool: ToolDescriptor
    arguments: dict[str, Any]  # as the caller gave them (logged, and fingerprinted by the approval gate)
    call_args: tuple  # what the device command gets
    caller: str | None = None
    run: str | None = None
    started: float = field(default_factory=time.time)
    t0: float = field(default_factory=time.perf_counter)
    trace: dict[str, Any] = field(default_factory=dict)  # fields every trace event of this call carries (the gate's)
    context: dict[str, Any] = field(default_factory=dict)  # what one interceptor leaves for a later one
    reply: Any = None  # the device's reply, JSON-safe, as logged

    def duration_s(self) -> float:
        return time.perf_counter() - self.t0


class Interceptor:
    """Base class: does nothing, applies to every tool. ``order`` places it in the list (lowest first):
    10 the preview, 20 logging and the trace, 30 job checks before the gate, 40 the approval gate,
    50 job stamping after it, 60 the token subject."""

    order = 100

    def applies(self, tool: ToolDescriptor) -> bool:
        return True

    def before_call(self, call: Call) -> Any:
        return None

    def after_call(self, call: Call, result: Any) -> Any:
        return result

    def on_error(self, call: Call, exc: Exception) -> Exception | None:
        return None


def run_call(call: Call, interceptors: list[Interceptor], invoke: Callable[[tuple], Any]) -> Any:
    """Run one call through ``interceptors`` around ``invoke`` (the device command, JSON-safe reply)."""
    for interceptor in interceptors:
        reply = interceptor.before_call(call)
        if reply is not None:
            return reply
    try:
        result = call.reply = invoke(call.call_args)
        for interceptor in reversed(interceptors):
            result = interceptor.after_call(call, result)
    except Exception as exc:
        raised: Exception = exc
        for interceptor in reversed(interceptors):
            raised = interceptor.on_error(call, raised) or raised
        if raised is exc:
            raise
        raise raised from exc
    return result
