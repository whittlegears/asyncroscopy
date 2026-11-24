"""take_turn, give_turn, turn_status: one run at a time acts on the instrument, and the others queue.

Two workflow runs on one instrument undo each other: each blanks the beam, sets the image shift and the field of
view for its own tile, and the other's next image is taken with them. The runs may be in different processes (the
agent server, a command line), so the turn is kept here, on the endpoint every one of them calls.

* ``take_turn`` asks for the instrument. The reply is ``held`` (it is this run's until it is given back or the lease
  runs out) or ``queued`` with the run's place in line. A queued run calls it again every few seconds: that keeps
  its place, and gets the turn when the runs ahead are done. The holder calls it again to renew the lease.
* ``give_turn`` gives the instrument back, or leaves the queue.
* While a run holds the turn, a tool that acts on the instrument is refused to every other caller with
  ``instrument_busy``. Tools that only read are never refused, and neither is a stop (blanking the beam, an abort):
  anyone may make the instrument safe at any time.
* A holder that stops renewing (its process died) loses the turn when its lease ends, and a queued run that stops
  asking loses its place, so neither blocks the line for long.

A run is known by the run id its calls carry (the ``X-Asyncroscopy-Run`` header). With no turn held, nothing here
changes: an agent calls tools as before. The turn lives in this endpoint's process, so the operator endpoint, a
separate process for a person, is not bound by it.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.tools import tool

from asyncroscopy.mcp.interceptors import Call, Interceptor

if TYPE_CHECKING:
    from asyncroscopy.mcp.descriptors import ToolDescriptor
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

LEASE_S = 60.0  # how long a turn lasts without being renewed
MIN_LEASE_S, MAX_LEASE_S = 5.0, 600.0
QUEUE_PATIENCE_S = 30.0  # a queued run that has not asked again for this long loses its place


class TurnQueue:
    """Who holds the instrument and who waits, in the order they asked. Safe to call from several threads."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, patience_s: float = QUEUE_PATIENCE_S) -> None:
        self._clock, self._patience_s = clock, patience_s
        self._lock = threading.Lock()
        self._holder: dict | None = None
        self._queue: list[dict] = []

    def _tidy(self) -> None:
        now = self._clock()
        if self._holder is not None and now >= self._holder["expires"]:
            self._holder = None
        self._queue = [w for w in self._queue if now - w["seen"] < self._patience_s]

    @staticmethod
    def _public(entry: dict, now: float) -> dict:
        out = {"run": entry["run"], "caller": entry["caller"], "what": entry["what"]}
        if "expires" in entry:
            out["held_s"], out["expires_in_s"] = round(now - entry["since"], 1), round(entry["expires"] - now, 1)
        else:
            out["waited_s"] = round(now - entry["asked"], 1)
        return out

    def take(self, run: str, caller: str | None = None, what: str = "", lease_s: float = LEASE_S) -> dict:
        """Ask for the turn, renew it, or keep a place in the queue. Returns {"state": "held" | "queued", ...};
        a turn just taken (not renewed) also has "new": True."""
        lease_s = min(max(float(lease_s), MIN_LEASE_S), MAX_LEASE_S)
        with self._lock:
            self._tidy()
            now = self._clock()
            if self._holder is not None and self._holder["run"] == run:
                self._holder["expires"] = now + lease_s
                return {"state": "held", "run": run, "expires_in_s": lease_s, "waiting": len(self._queue)}
            waiting = next((w for w in self._queue if w["run"] == run), None)
            first = waiting is self._queue[0] if waiting is not None and self._queue else not self._queue
            if self._holder is None and first:
                self._queue = [w for w in self._queue if w["run"] != run]
                self._holder = {"run": run, "caller": caller, "what": what or (waiting or {}).get("what", ""), "since": now, "expires": now + lease_s}
                return {"state": "held", "run": run, "expires_in_s": lease_s, "waiting": len(self._queue), "new": True}
            if waiting is None:
                waiting = {"run": run, "caller": caller, "what": what, "asked": now, "seen": now}
                self._queue.append(waiting)
            waiting["seen"] = now
            position = self._queue.index(waiting) + 1
            ahead = ([self._holder] if self._holder else []) + self._queue[: position - 1]
            return {"state": "queued", "run": run, "position": position, "ahead": [self._public(e, now) for e in ahead]}

    def give(self, run: str) -> dict:
        """Give the turn back, or leave the queue. Returns what was given up: "turn", "place" or "nothing"."""
        with self._lock:
            self._tidy()
            if self._holder is not None and self._holder["run"] == run:
                self._holder = None
                return {"gave": "turn", "waiting": len(self._queue)}
            before = len(self._queue)
            self._queue = [w for w in self._queue if w["run"] != run]
            return {"gave": "place" if len(self._queue) < before else "nothing", "waiting": len(self._queue)}

    def holder(self) -> dict | None:
        with self._lock:
            self._tidy()
            return dict(self._holder) if self._holder else None

    def status(self) -> dict:
        with self._lock:
            self._tidy()
            now = self._clock()
            return {"holder": self._public(self._holder, now) if self._holder else None,
                    "queue": [self._public(w, now) for w in self._queue]}


class TurnGuard(Interceptor):
    """Refuses a tool that acts on the instrument to every run but the one holding the turn. Stops always pass."""

    order = 15  # before the trace and the job checks: a refused call never reaches the device

    def __init__(self, queue: TurnQueue) -> None:
        self.queue = queue

    def applies(self, tool: ToolDescriptor) -> bool:
        return bool(tool.acts) and not tool.stop and tool.action_role != "beam.blank"

    def before_call(self, call: Call) -> Any:
        holder = self.queue.holder()
        if holder is None or (call.run and call.run == holder["run"]):
            return None
        what = f" ({holder['what']})" if holder.get("what") else ""
        raise ToolError(json.dumps({
            "error": {"code": "instrument_busy",
                      "message": f"{call.tool.name} was not called: run {holder['run']}{what} holds the instrument. Tools that only read "
                                 "still work. Wait for it to finish (turn_status), or queue behind it with take_turn."},
            "holder": {"run": holder["run"], "caller": holder.get("caller"), "what": holder.get("what")},
        }))


class Turns:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.queue = TurnQueue()

    def natives(self) -> list:
        return [] if self.server.endpoint_role == "monitor" else [self.take_turn, self.give_turn, self.turn_status]

    def _run(self) -> str:
        run = self.server.run_id()
        if not run:
            raise ToolError("take_turn and give_turn need a run id: send the X-Asyncroscopy-Run header with every call of the run "
                            "(asyncroscopy.agents.config.AgentConfig does)")
        return run

    def _trace(self, action: str, started: float, **fields: Any) -> None:
        if self.server.trace is None:
            return
        try:
            self.server.trace.record(action, True, started, caller=self.server.caller(), run=self.server.run_id() or None, **fields)
        except Exception as exc:  # the turn itself must not fail because a trace line could not be written
            warnings.warn(f"{action}: trace event not written: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

    @tool()
    def take_turn(self, what: str = "", lease_s: float = LEASE_S) -> dict[str, Any]:
        """For agent programs, not models: ask for the instrument for this run, so no other run acts on it meanwhile.

        Returns {"state": "held", "expires_in_s"} or {"state": "queued", "position", "ahead"}. While queued, call it
        again every few seconds: that keeps the place and takes the turn when it comes. While holding, call it again
        before ``lease_s`` seconds pass to keep the turn. ``what`` says what the run is, for whoever waits behind it."""
        started = time.time()
        reply = self.queue.take(self._run(), self.server.caller(), str(what)[:200], lease_s)
        if reply.pop("new", False):  # renewals and queue polls are not traced: there is one every few seconds
            self._trace("take_turn", started, arguments={"what": str(what)[:200]}, result="held")
        return reply

    @tool()
    def give_turn(self) -> dict[str, Any]:
        """For agent programs, not models: give the instrument back when this run has finished acting on it, or leave
        the queue. Returns {"gave": "turn" | "place" | "nothing", "waiting"}."""
        started = time.time()
        reply = self.queue.give(self._run())
        if reply["gave"] == "turn":
            self._trace("give_turn", started, arguments={}, result=json.dumps(reply))
        return reply

    @tool()
    def turn_status(self) -> dict[str, Any]:
        """Which run holds the instrument now and which runs wait for it, in order: {"holder", "queue"}. Reads only."""
        return self.queue.status()


def register(server: MCPServer, cfg: ServerConfig) -> Turns:
    turns = Turns(server, cfg)
    server.add_interceptor(TurnGuard(turns.queue))
    return turns
