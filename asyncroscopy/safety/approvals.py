"""Approvals: the instrument's approval mode, pending requests and grants.

``auto`` (the default) lets every agent call through and the trace records
that no person approved it. ``approve`` holds high-risk agent calls: the MCP
server files a request here and refuses the call; once an operator approves,
the same call (same tool, same arguments) succeeds once. Notebooks and
scripts talk to Tango directly and are never held.

Every mode change, request and decision is written to the Instrument Trace.
The mode is also saved to ``mode_file`` and restored on restart, so a crashed
or re-initialised device does not silently fall back to ``auto``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

from tango import DevState
from tango.server import Device, command, device_property

from asyncroscopy.mcp.tool_log import TraceWriter

MODES = ("auto", "approve")
PRUNE_AFTER_S = 86400.0


def fingerprint(action: str, arguments) -> str:
    canonical = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(f"{action}\n{canonical}".encode()).hexdigest()[:16]


class Approvals(Device):
    default_mode = device_property(dtype=str, default_value="auto", doc="Mode at start-up: auto or approve.")
    request_timeout_s = device_property(dtype=float, default_value=1800.0, doc="A request nobody answers expires after this long.")
    grant_ttl_s = device_property(dtype=float, default_value=900.0, doc="An approval must be used within this long.")
    trace_path = device_property(dtype=str, default_value="outputs/trace/approvals.jsonl", doc="Instrument Trace JSONL file.")
    mode_file = device_property(dtype=str, default_value="outputs/approvals/mode.json", doc="Where the mode is kept, so a restart does not fall back to default_mode.")

    def init_device(self) -> None:
        Device.init_device(self)
        # Set before any early return, so a FAULT device still answers, and answers "approve".
        self._lock = threading.Lock()
        self._requests: dict[str, dict] = {}
        self._grants: dict[str, dict] = {}
        self._trace = TraceWriter(self.trace_path, source="approvals")
        self._mode = "approve"
        mode = str(self.default_mode).strip().lower()
        if mode not in MODES:
            self.set_state(DevState.FAULT)
            self.set_status(f"default_mode must be one of {MODES}, got {self.default_mode!r}; holding every high-risk call")
            return
        self._mode = self._restore_mode(mode)
        self.set_state(DevState.ON)

    def _restore_mode(self, default: str) -> str:
        """The last mode an operator set, or ``default`` if none was ever saved.

        A saved file that cannot be read gives "approve": after a crash the safe
        assumption is that someone wanted calls held.
        """
        path = Path(self.mode_file)
        if not path.exists():
            return default
        try:
            mode = str(json.loads(path.read_text(encoding="utf-8"))["mode"])
        except (OSError, ValueError, KeyError, TypeError):
            return "approve"
        return mode if mode in MODES else "approve"

    def _save_mode(self, who) -> None:
        path = Path(self.mode_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mode": self._mode, "who": who, "at": time.time()}), encoding="utf-8")

    def _expire(self) -> None:
        now = time.time()
        for request in self._requests.values():
            if request["state"] == "pending" and now - request["created"] > float(self.request_timeout_s):
                request["state"] = "expired"
                self._trace.record("Approvals.expire", True, now, request_id=request["request_id"], arguments=request["action"])
        for grant in self._grants.values():
            if not grant["used"] and now - grant["created"] > float(self.grant_ttl_s):
                grant["used"] = True
                grant["expired"] = True
        # Forget settled requests and used grants after a day (the trace keeps the record),
        # so a long session or a retrying agent cannot grow these without bound.
        for table, settled in ((self._requests, lambda r: r["state"] != "pending"), (self._grants, lambda g: g["used"])):
            for key in [k for k, v in table.items() if settled(v) and now - v["created"] > PRUNE_AFTER_S]:
                del table[key]

    @staticmethod
    def _load(text: str) -> dict:
        try:
            body = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"expected a JSON object: {exc}") from exc
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        return body

    @command(dtype_out=str)
    def get_mode(self) -> str:
        """Current approval mode: 'auto' or 'approve'."""
        return self._mode

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"mode": "auto" or "approve", "who": "operator name"}.')
    def set_mode(self, request: str) -> str:
        """Switch between auto and approve. Operator only; recorded in the trace."""
        body = self._load(request)
        mode = str(body.get("mode", "")).strip().lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            previous, self._mode = self._mode, mode
            self._save_mode(body.get("who"))
        self._trace.record("Approvals.set_mode", True, time.time(), caller=body.get("who"), arguments={"from": previous, "to": mode}, mode=mode)
        return json.dumps({"mode": mode, "previous": previous})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"action": "Class_command", "arguments": {...}, "caller": "...", "reason": "..."}.')
    def request(self, request: str) -> str:
        """File a request for a held call; the same pending call returns the same request id."""
        body = self._load(request)
        action = str(body.get("action", ""))
        if not action:
            raise ValueError("'action' is required")
        key = fingerprint(action, body.get("arguments"))
        with self._lock:
            self._expire()
            for existing in self._requests.values():
                if existing["fingerprint"] == key and existing["state"] == "pending":
                    return json.dumps({"request_id": existing["request_id"], "state": "pending"})
            request_id = uuid.uuid4().hex[:12]
            self._requests[request_id] = {
                "request_id": request_id,
                "action": action,
                "arguments": body.get("arguments"),
                "caller": body.get("caller"),
                "reason": body.get("reason"),
                "fingerprint": key,
                "state": "pending",
                "created": time.time(),
            }
        self._trace.record("Approvals.request", True, time.time(), request_id=request_id, caller=body.get("caller"), arguments={"action": action, "arguments": body.get("arguments")}, mode=self._mode)
        return json.dumps({"request_id": request_id, "state": "pending"})

    @command(dtype_out=str)
    def list_pending(self) -> str:
        """Requests waiting for an operator, oldest first."""
        with self._lock:
            self._expire()
            pending = [r for r in self._requests.values() if r["state"] == "pending"]
        return json.dumps(sorted(pending, key=lambda r: r["created"]), default=str)

    @command(dtype_in=str, dtype_out=str, doc_in=':param request_id: The request id returned when the call was held.')
    def request_status(self, request_id: str) -> str:
        """State of one request: pending, approved, denied or expired."""
        with self._lock:
            self._expire()
            request = self._requests.get(request_id.strip())
        if request is None:
            raise ValueError(f"no request {request_id!r}")
        return json.dumps({k: request.get(k) for k in ("request_id", "action", "state", "grant_id", "reason", "decided_by")})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"request_id": "...", "who": "operator name"}.')
    def approve(self, request: str) -> str:
        """Approve a pending request. Operator only. The call it covers may then run once."""
        body = self._load(request)
        with self._lock:
            self._expire()
            req = self._requests.get(str(body.get("request_id")))
            if req is None or req["state"] != "pending":
                raise ValueError(f"request {body.get('request_id')!r} is not pending")
            grant_id = uuid.uuid4().hex[:12]
            self._grants[grant_id] = {"grant_id": grant_id, "fingerprint": req["fingerprint"], "action": req["action"], "request_id": req["request_id"], "who": body.get("who"), "created": time.time(), "used": False}
            req.update(state="approved", grant_id=grant_id, decided_by=body.get("who"))
        self._trace.record("Approvals.approve", True, time.time(), request_id=req["request_id"], grant_id=grant_id, caller=body.get("who"), arguments=req["action"])
        return json.dumps({"request_id": req["request_id"], "grant_id": grant_id, "state": "approved"})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"request_id": "...", "who": "operator name", "reason": "..."}.')
    def deny(self, request: str) -> str:
        """Deny a pending request. Operator only."""
        body = self._load(request)
        with self._lock:
            self._expire()
            req = self._requests.get(str(body.get("request_id")))
            if req is None or req["state"] != "pending":
                raise ValueError(f"request {body.get('request_id')!r} is not pending")
            req.update(state="denied", decided_by=body.get("who"), reason=body.get("reason") or req.get("reason"))
        self._trace.record("Approvals.deny", True, time.time(), request_id=req["request_id"], caller=body.get("who"), arguments=req["action"])
        return json.dumps({"request_id": req["request_id"], "state": "denied"})

    @command(dtype_in=str, dtype_out=str, doc_in=':param request: JSON {"action": "Class_command", "arguments": {...}}.')
    def check(self, request: str) -> str:
        """Use an unused grant for exactly this call, if one exists. Called by the MCP server."""
        body = self._load(request)
        key = fingerprint(str(body.get("action", "")), body.get("arguments"))
        with self._lock:
            self._expire()
            for grant in self._grants.values():
                if grant["fingerprint"] == key and not grant["used"]:
                    grant["used"] = True
                    grant["used_at"] = time.time()
                    return json.dumps({"allowed": True, "grant_id": grant["grant_id"], "approved_by": grant["who"], "request_id": grant["request_id"]})
        return json.dumps({"allowed": False})

    @command(dtype_out=str)
    def list_grants(self) -> str:
        """Grants issued so far, used or not."""
        with self._lock:
            self._expire()
            return json.dumps(list(self._grants.values()), default=str)


if __name__ == "__main__":
    Approvals.run_server()
