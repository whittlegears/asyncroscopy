"""One action queue for the operator and the agent.

Every instrument action, whoever asked for it, becomes an :class:`Action` in a
single FIFO queue and runs one at a time. Each action records who requested it
(``agent``, ``operator`` or ``recipe``), which atlas site it targets, and its
status. Composite actions (return to a site, run a recipe) record each
instrument operation they make as a child action.

Operator controls take effect *between* actions, including between the child
steps of a composite one:

* **pause / resume / step**: hold before the next action; step lets exactly one through.
* **skip**: drop an action that has not started.
* **abort**: end the running composite action at its next step boundary.
* **take over**: the operator holds control. Agent requests are recorded as
  ``rejected`` and not run; operator requests still run. **release** hands back.
* **approve / deny**: actions the policy marks ``awaiting_approval`` wait at the
  head of the queue until an operator decides. The agent cannot approve.

The instrument's **stop** is deliberately *not* a queue operation: it calls the
stop handler immediately on the caller's thread, without taking the queue's
lock or waiting for the running action, and then pauses the queue so nothing
new starts until the operator resumes.

Run history is append-only JSONL (one line per state change), so it survives
take-overs, restarts and crashes, and is what recipes are built from.
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from asyncroscopy.supervision.policy import ControlPolicy, PolicyViolation

ORIGINS = ("agent", "operator", "recipe")
FINAL = ("done", "error", "skipped", "denied", "aborted", "rejected", "refused")
PREVIEW_CHARS = 600


class Aborted(RuntimeError):
    """The operator aborted this action at a step boundary."""


class OperatorHasControl(RuntimeError):
    """The operator has taken over; agent actions are recorded but not run."""


def preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    if isinstance(value, np.ndarray):
        return f"<array shape={list(value.shape)} dtype={value.dtype}>"
    try:
        text = json.dumps(value, default=lambda o: f"<array shape={list(o.shape)}>" if isinstance(o, np.ndarray)
                          else str(o))
    except Exception:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _safe_args(args: dict) -> dict:
    return json.loads(preview(args, limit=10_000_000)) if args else {}


@dataclass
class Action:
    id: str
    tool: str
    origin: str
    label: str
    args: dict = field(default_factory=dict)
    site_id: str | None = None
    run_id: str | None = None
    parent_id: str | None = None
    status: str = "queued"
    requires_approval: bool = False
    approval_reason: str = ""
    decided_by: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: str | None = None
    error: str | None = None
    children: list[str] = field(default_factory=list)
    note: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


class _Item:
    def __init__(self, action: Action, fn: Callable[[], Any]):
        self.action = action
        self.fn = fn
        self.done = threading.Event()
        self.value: Any = None
        self.exc: BaseException | None = None


class ActionQueue:
    def __init__(self, *, policy: ControlPolicy | None = None, history_path: str | Path | None = None,
                 stop_handler: Callable[[], Any] | None = None,
                 current_stage: Callable[[], list[float]] | None = None,
                 passthrough: list[str] | None = None, max_events: int = 5000):
        self.policy = policy or ControlPolicy()
        self.history_path = Path(history_path) if history_path else None
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_handler = stop_handler
        self.current_stage = current_stage
        self.passthrough = list(passthrough or [])
        self._cv = threading.Condition()
        self._pending: deque[_Item] = deque()
        self.actions: dict[str, Action] = {}
        self.order: list[str] = []
        self._ids = itertools.count(1)
        self._session = uuid.uuid4().hex[:4]
        self._paused = False
        self._step_tokens = 0
        self._abort = False
        self.controller = "shared"            # "shared" | "operator"
        self.controller_note = ""
        self.current: str | None = None
        self._events: deque[dict] = deque(maxlen=max_events)
        self._event_seq = itertools.count(1)
        self._local = threading.local()
        self._history_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_forever, name="action-queue", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, data: Any) -> None:
        event = {"seq": next(self._event_seq), "t": time.time(), "kind": kind, "data": data}
        self._events.append(event)
        if self.history_path:
            try:
                with self._history_lock, self.history_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")
            except Exception:
                pass      # history must never break the instrument path

    def events_since(self, seq: int = 0) -> list[dict]:
        return [e for e in list(self._events) if e["seq"] > seq]

    def _publish(self, action: Action) -> None:
        self._emit("action", action.to_json())

    def _set(self, action: Action, **changes) -> None:
        for key, value in changes.items():
            setattr(action, key, value)
        self._publish(action)

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        with self._cv:
            return {"paused": self._paused, "controller": self.controller, "controller_note": self.controller_note,
                    "current": self.current, "queued": sum(1 for i in self._pending if i.action.status in
                                                           ("queued", "awaiting_approval")),
                    "awaiting_approval": [i.action.id for i in self._pending
                                          if i.action.status == "awaiting_approval"],
                    "last_event": self._events[-1]["seq"] if self._events else 0}

    def snapshot(self, limit: int = 200) -> dict:
        rows = []
        for aid in self.order[-limit:]:
            top = self.actions.get(aid)
            if top is None:
                continue
            rows.append(top.to_json())
            rows.extend(self.actions[c].to_json() for c in top.children if c in self.actions)
        return {"state": self.state(), "actions": rows}

    def get(self, action_id: str) -> Action:
        try:
            return self.actions[action_id]
        except KeyError:
            raise KeyError(f"no action {action_id!r}") from None

    # ------------------------------------------------------------------ submission
    def is_passthrough(self, names: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(n, p) for p in self.passthrough for n in names)

    def in_worker(self) -> bool:
        return getattr(self._local, "action", None) is not None

    def _new_action(self, tool: str, origin: str, label: str | None, args: dict | None, **kw) -> Action:
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")
        action = Action(id=f"a{self._session}-{next(self._ids)}", tool=tool, origin=origin,
                        label=label or tool, args=_safe_args(dict(args or {})), **kw)
        self.actions[action.id] = action
        return action

    def submit(self, tool: str, fn: Callable[[], Any], *, args: dict | None = None, origin: str = "agent",
               label: str | None = None, site_id: str | None = None, run_id: str | None = None,
               aliases: list[str] | None = None) -> _Item:
        """Queue *fn* as one top-level action. Returns a handle; see :meth:`wait`.

        Hard-limit violations are refused here, before queueing, and recorded.
        """
        names = [tool, *(aliases or [])]
        action = self._new_action(tool, origin, label, args, site_id=site_id, run_id=run_id)
        item = _Item(action, fn)
        with self._cv:
            self.order.append(action.id)
            if origin == "agent" and self.controller == "operator":
                action.status, action.finished = "rejected", time.time()
                action.error = ("the operator has taken control of the instrument; this request was "
                                "recorded but not run. Wait for the operator to release control.")
                item.exc = OperatorHasControl(action.error)
                item.done.set()
                self._publish(action)
                return item
            decision = self.policy.check(names, dict(args or {}), self.current_stage)
            if not decision.allowed:
                action.status, action.finished, action.error = "refused", time.time(), decision.reason
                item.exc = PolicyViolation(decision.reason)
                item.done.set()
                self._publish(action)
                return item
            if decision.needs_approval:
                action.status = "awaiting_approval"
                action.requires_approval = True
                action.approval_reason = decision.reason
            self._pending.append(item)
            self._publish(action)
            self._cv.notify_all()
        return item

    @staticmethod
    def wait(item: _Item, timeout: float | None = None) -> Any:
        """Block until the action finishes; raise its error. ``TimeoutError`` if still pending."""
        if not item.done.wait(timeout):
            raise TimeoutError(item.action.id)
        if item.exc is not None:
            raise item.exc
        return item.value

    def call(self, tool: str, fn: Callable[[], Any], **kw) -> Any:
        """Submit and wait. Inside a running action, runs *fn* as a child step instead."""
        if self.in_worker():
            return self.child(tool, kw.get("args") or {}, fn, aliases=kw.get("aliases"))
        return self.wait(self.submit(tool, fn, **kw))

    # ------------------------------------------------------------------ child steps
    def child(self, tool: str, args: dict, fn: Callable[[], Any], label: str | None = None,
              aliases: list[str] | None = None) -> Any:
        """Run one step of the running composite action, recorded as its child.

        The pause/step gate and abort apply before every child, and so do the
        policy's hard limits. Approval-gated tools are refused inside a
        composite action (approve the composite instead, or run them alone).
        """
        parent: Action = getattr(self._local, "action", None)
        if parent is None:
            return fn()
        self._gate()
        names = [tool, *(aliases or [])]
        action = self._new_action(tool, parent.origin, label, args, site_id=parent.site_id,
                                  run_id=parent.run_id, parent_id=parent.id)
        parent.children.append(action.id)
        decision = self.policy.check(names, dict(args or {}), self.current_stage)
        if not decision.allowed or decision.needs_approval:
            reason = decision.reason if not decision.allowed else \
                f"{decision.reason}; it cannot run inside {parent.label!r} without its own approval"
            self._set(action, status="refused", finished=time.time(), error=reason)
            raise PolicyViolation(reason)
        self._set(action, status="running", started=time.time())
        try:
            value = fn()
        except BaseException as exc:
            self._set(action, status="error", finished=time.time(), error=f"{type(exc).__name__}: {exc}")
            raise
        self._set(action, status="done", finished=time.time(), result=preview(value))
        return value

    # Adapter with the atlas' step(name, args, fn) signature.
    def step_fn(self, aliases_for: Callable[[str], list[str]] | None = None):
        def step(name: str, args: dict, fn: Callable[[], Any]) -> Any:
            return self.child(name, args, fn, aliases=aliases_for(name) if aliases_for else None)
        return step

    # ------------------------------------------------------------------ control
    def pause(self, note: str = "") -> None:
        with self._cv:
            self._paused, self._step_tokens = True, 0
            self._cv.notify_all()
        self._emit("control", {"op": "pause", "note": note})

    def resume(self) -> None:
        with self._cv:
            self._paused, self._step_tokens = False, 0
            self._cv.notify_all()
        self._emit("control", {"op": "resume"})

    def step(self) -> None:
        with self._cv:
            self._paused = True
            self._step_tokens += 1
            self._cv.notify_all()
        self._emit("control", {"op": "step"})

    def abort(self) -> bool:
        with self._cv:
            if self.current is None:
                return False
            self._abort = True
            self._cv.notify_all()
        self._emit("control", {"op": "abort", "action": self.current})
        return True

    def skip(self, action_id: str, by: str = "operator") -> bool:
        with self._cv:
            for item in list(self._pending):
                if item.action.id == action_id and item.action.status in ("queued", "awaiting_approval"):
                    self._pending.remove(item)
                    self._set(item.action, status="skipped", finished=time.time(), decided_by=by,
                              error=f"skipped by {by} before it ran")
                    item.exc = RuntimeError(f"action {action_id} was skipped by the {by} before it ran")
                    item.done.set()
                    self._cv.notify_all()
                    return True
        return False

    def approve(self, action_id: str, by: str = "operator") -> bool:
        with self._cv:
            for item in self._pending:
                if item.action.id == action_id and item.action.status == "awaiting_approval":
                    self._set(item.action, status="queued", decided_by=by, note=f"approved by {by}")
                    self._cv.notify_all()
                    return True
        return False

    def deny(self, action_id: str, by: str = "operator") -> bool:
        with self._cv:
            for item in list(self._pending):
                if item.action.id == action_id and item.action.status == "awaiting_approval":
                    self._pending.remove(item)
                    self._set(item.action, status="denied", finished=time.time(), decided_by=by,
                              error=f"denied by {by}")
                    item.exc = PolicyViolation(f"the {by} denied action {action_id}")
                    item.done.set()
                    self._cv.notify_all()
                    return True
        return False

    def take_over(self, who: str = "operator", note: str = "") -> None:
        """Operator holds control: pause, and reject new agent requests. History is kept."""
        with self._cv:
            self.controller, self.controller_note = "operator", note or f"taken over by {who}"
            self._paused, self._step_tokens = True, 0
            for item in list(self._pending):        # queued agent actions go back to the agent, unrun
                if item.action.origin == "agent":
                    self._pending.remove(item)
                    self._set(item.action, status="rejected", finished=time.time(),
                              error="the operator took over before this ran")
                    item.exc = OperatorHasControl(item.action.error)
                    item.done.set()
            self._cv.notify_all()
        self._emit("control", {"op": "take_over", "who": who, "note": note})

    def release(self, who: str = "operator") -> None:
        with self._cv:
            self.controller, self.controller_note = "shared", ""
            self._cv.notify_all()
        self._emit("control", {"op": "release", "who": who})

    def stop(self, who: str = "operator") -> dict:
        """Instrument stop. Runs NOW on this thread; does not wait for or lock the queue."""
        t0 = time.time()
        outcome: dict[str, Any] = {"op": "stop", "who": who, "t": t0, "running": self.current}
        try:
            outcome["result"] = preview(self.stop_handler()) if self.stop_handler else "no stop handler configured"
            outcome["ok"] = self.stop_handler is not None
        except Exception as exc:
            outcome["ok"], outcome["error"] = False, f"{type(exc).__name__}: {exc}"
        # Afterwards, make sure nothing new starts by itself.
        self._paused, self._step_tokens = True, 0
        if self.current is not None:
            self._abort = True
        with self._cv:
            self._cv.notify_all()
        self._emit("stop", outcome)
        return outcome

    # ------------------------------------------------------------------ execution
    def _gate(self) -> None:
        """Wait while paused (step lets one through); raise Aborted if aborted."""
        with self._cv:
            while True:
                if self._abort:
                    self._abort = False
                    raise Aborted("aborted by operator at a step boundary")
                if not self._paused:
                    return
                if self._step_tokens > 0:
                    self._step_tokens -= 1
                    return
                self._cv.wait()

    def _next(self) -> _Item:
        with self._cv:
            while True:
                head = self._pending[0] if self._pending else None
                if head is not None and head.action.status == "queued":
                    ready = (not self._paused) or self._step_tokens > 0
                    if ready:
                        if self._paused:
                            self._step_tokens -= 1
                        self._pending.popleft()
                        self._abort = False
                        return head
                self._cv.wait()     # empty, paused, or the head awaits approval

    def _run_forever(self) -> None:
        while True:
            item = self._next()
            action = item.action
            self.current = action.id
            self._local.action = action
            self._set(action, status="running", started=time.time())
            try:
                item.value = item.fn()
                self._set(action, status="done", finished=time.time(), result=preview(item.value))
            except Aborted as exc:
                item.exc = exc
                self._set(action, status="aborted", finished=time.time(), error=str(exc))
            except BaseException as exc:
                item.exc = exc
                self._set(action, status="error", finished=time.time(), error=f"{type(exc).__name__}: {exc}")
            finally:
                self._local.action = None
                with self._cv:
                    self.current = None
                    self._abort = False
                    self._cv.notify_all()
                item.done.set()


def load_history(path: str | Path) -> dict[str, dict]:
    """Final state of every action in a history file (later lines win)."""
    path = Path(path)
    actions: dict[str, dict] = {}
    if not path.exists():
        return actions
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") == "action":
            data = event["data"]
            actions[data["id"]] = data
    return actions
