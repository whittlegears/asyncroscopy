"""Instrument commands that wait for a person before they run.

Commands listed under ``mcp.needs_approval`` (by Tango class, or ``"*"``, with
shell-style patterns such as ``move_*``) are not run when an agent calls them on
the agent endpoint. The call is *held*: its tool name and arguments go into a
small SQLite file, and the agent gets back an error that starts with ``HELD``
and names the call id. It can check the call with ``call_status``.

A person approves or denies held calls on the *operator* endpoint, a second
instance of the bridge started with ``mcp.role: operator`` and the same
``mcp.approvals_db``. Approving runs the command there and stores its result,
which the agent then reads with ``call_status``. The agent endpoint never offers
``approve_call``, so an agent cannot approve its own call.

The store is a file so the two endpoints can be separate processes, and so held
calls survive a restart of either.
"""

from __future__ import annotations

import fnmatch
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

HELD_PREFIX = "HELD"
STATUSES = ("pending", "approved", "denied", "done", "failed")


def needs_approval(rules: Mapping[str, Sequence[str]], dev_class: str, command: str) -> bool:
    """True when ``command`` of ``dev_class`` matches a pattern under its class or ``"*"``.

    A pattern may also be written ``Class.command``.
    """
    patterns = list(rules.get("*", [])) + list(rules.get(dev_class, []))
    for pattern in patterns:
        if "." in pattern:
            cls, _, cmd = pattern.partition(".")
            if fnmatch.fnmatchcase(dev_class, cls) and fnmatch.fnmatchcase(command, cmd):
                return True
        elif fnmatch.fnmatchcase(command, pattern):
            return True
    return False


def held_message(call_id: str, tool: str) -> str:
    return (
        f"{HELD_PREFIX} {call_id}: {tool} needs an operator's approval before it runs. "
        f"It has not run. Check it with call_status(call_id='{call_id}'); tell the user it is waiting."
    )


def parse_held(message: str) -> Optional[str]:
    """The call id in a held-call message, or None."""
    text = str(message)
    index = text.find(HELD_PREFIX + " ")
    if index < 0:
        return None
    rest = text[index + len(HELD_PREFIX) + 1 :]
    call_id = rest.split(":", 1)[0].strip()
    return call_id or None


class ApprovalStore:
    """Held calls in one SQLite file, safe to share between processes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS calls (
                    id TEXT PRIMARY KEY, tool TEXT NOT NULL, arguments TEXT NOT NULL, status TEXT NOT NULL,
                    requested_at REAL NOT NULL, decided_at REAL, decided_by TEXT, note TEXT,
                    result TEXT, error TEXT, finished_at REAL)"""
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    def hold(self, tool: str, arguments: dict) -> str:
        call_id = uuid.uuid4().hex[:10]
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO calls (id, tool, arguments, status, requested_at) VALUES (?, ?, ?, 'pending', ?)",
                (call_id, tool, json.dumps(arguments, default=str), time.time()),
            )
        return call_id

    def get(self, call_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        if row is None:
            raise ValueError(f"No held call {call_id!r}")
        return self._row(row)

    def pending(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM calls WHERE status = 'pending' ORDER BY requested_at").fetchall()
        return [self._row(r) for r in rows]

    def decide(self, call_id: str, status: str, actor: str, note: str = "") -> dict[str, Any]:
        """Mark a pending call approved or denied. Refuses a call that was already decided."""
        if status not in ("approved", "denied"):
            raise ValueError("status must be approved or denied")
        with self._lock, self._connect() as db:
            changed = db.execute(
                "UPDATE calls SET status = ?, decided_at = ?, decided_by = ?, note = ? WHERE id = ? AND status = 'pending'",
                (status, time.time(), actor, note, call_id),
            ).rowcount
        if not changed:
            current = self.get(call_id)
            raise ValueError(f"Call {call_id} is {current['status']}, not pending")
        return self.get(call_id)

    def finish(self, call_id: str, result: Any = None, error: str | None = None) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE calls SET status = ?, result = ?, error = ?, finished_at = ? WHERE id = ?",
                ("failed" if error else "done", None if error else json.dumps(result, default=str), error, time.time(), call_id),
            )
        return self.get(call_id)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["arguments"] = json.loads(out["arguments"])
        if out.get("result") is not None:
            out["result"] = json.loads(out["result"])
        return out
