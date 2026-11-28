"""Which tools the agent endpoint runs at once, which it holds for a person, and the calls it holds.

A tool's permission is ``allow`` (runs at once), ``approve`` (held until a person approves that exact call) or
``never`` (refused; not settable). ``docs/MCP/permissions.md`` is the contract.

Two files, both keyed by the stack's ``stack_id``:

- the permissions file (``mcp.permissions_file``, else ``$ASYNCROSCOPY_PERMISSIONS_FILE``, else
  ``outputs/permissions/permissions.json``): what people set, ``{"groups": {...}, "tools": {name: permission}}``;
- beside it, ``<name>.state.json``: the agent endpoint's tools as it last published them, and the held calls.

The agent endpoint and the operator endpoint are separate processes, so every read and write takes a lock
shared between processes (``<name>.lock``).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from asyncroscopy.core.fileio import file_lock, write_atomic

default_approval_ttl_s = 600.0
settable_permissions = ("allow", "approve")
permission_groups = ("reads", "acts")
lock_wait_s = 5.0


def group_of(acts: bool) -> str:
    return "acts" if acts else "reads"


def canonical_json(arguments: dict[str, Any] | None) -> str:
    return json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"), default=str)


def arguments_sha256(arguments: dict[str, Any] | None) -> str:
    return hashlib.sha256(canonical_json(arguments).encode("utf-8")).hexdigest()


def iso_utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seconds_of(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def permission_and_source(name: str, entry: dict[str, Any], settings: dict[str, Any]) -> tuple[str, str]:
    """(permission, where it came from), first match wins: the fixed hard-block set ("fixed"), a person's per-tool
    setting ("tool"), the config's needs_approval ("config"), the group default ("group"). A ``fixed`` tool (the
    endpoint's own contract tools, stop_all and the stops) always runs, also with source "fixed"."""
    if entry.get("never"):
        return "never", "fixed"
    if entry.get("fixed"):
        return "allow", "fixed"
    saved = settings["tools"].get(name)
    if saved in settable_permissions:
        return saved, "tool"
    if entry.get("needs_approval"):
        return "approve", "config"
    return settings["groups"][group_of(bool(entry.get("acts")))], "group"


def effective_permission(name: str, entry: dict[str, Any], settings: dict[str, Any]) -> str:
    return permission_and_source(name, entry, settings)[0]


class PermissionStore:
    """The permission settings and held calls of one stack."""

    def __init__(self, path: str | Path, stack_id: str, approval_ttl_s: float = default_approval_ttl_s,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.state_path = self.path.with_name(f"{self.path.stem}.state.json")
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.stack_id = stack_id
        self.approval_ttl_s = float(approval_ttl_s)
        self.clock = clock

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle, file_lock(handle, time.monotonic() + lock_wait_s):
            yield

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        found = json.loads(path.read_text(encoding="utf-8") or "{}")
        if not isinstance(found, dict):
            raise ValueError(f"{path} does not hold a JSON object")
        return found

    def _write(self, path: Path, everything: dict[str, Any]) -> None:
        write_atomic(path, json.dumps(everything, indent=1, sort_keys=True))

    def _settings_in(self, everything: dict[str, Any]) -> dict[str, Any]:
        mine = everything.get(self.stack_id) or {}
        groups = {group: "allow" for group in permission_groups}
        groups.update({g: p for g, p in (mine.get("groups") or {}).items() if g in permission_groups and p in settable_permissions})
        tools = {t: p for t, p in (mine.get("tools") or {}).items() if p in settable_permissions}
        return {"groups": groups, "tools": tools}

    def settings(self) -> dict[str, Any]:
        with self._locked():
            return self._settings_in(self._read(self.path))

    def set_tool(self, tool: str, permission: str) -> dict[str, Any]:
        if permission not in (*settable_permissions, "default"):
            raise ValueError(f"permission must be allow, approve or default, not {permission!r}")
        with self._locked():
            catalog = self._read(self.state_path).get(self.stack_id, {}).get("catalog") or {}
            if tool not in catalog:
                raise ValueError(f"the agent endpoint of {self.stack_id} offers no tool {tool!r}")
            if catalog[tool].get("never") or catalog[tool].get("fixed"):
                raise ValueError(f"{tool} is {'never run' if catalog[tool].get('never') else 'always run'}; its permission cannot be set")
            everything = self._read(self.path)
            tools = everything.setdefault(self.stack_id, {}).setdefault("tools", {})
            if permission == "default":
                tools.pop(tool, None)
            else:
                tools[tool] = permission
            self._write(self.path, everything)
            return self._settings_in(everything)

    def set_group(self, group: str, permission: str) -> dict[str, Any]:
        if group not in permission_groups:
            raise ValueError(f"group must be reads or acts, not {group!r}")
        if permission not in settable_permissions:
            raise ValueError(f"a group's permission must be allow or approve, not {permission!r}")
        with self._locked():
            everything = self._read(self.path)
            everything.setdefault(self.stack_id, {}).setdefault("groups", {})[group] = permission
            self._write(self.path, everything)
            return self._settings_in(everything)

    def publish(self, catalog: dict[str, dict[str, Any]]) -> None:
        """Record the agent endpoint's tools, each {"acts", "risk", "description", "needs_approval", "never", "fixed"}."""
        with self._locked():
            state = self._read(self.state_path)
            mine = state.setdefault(self.stack_id, {})
            mine["catalog"] = catalog
            mine["approval_ttl_s"] = self.approval_ttl_s
            self._write(self.state_path, state)

    def permissions(self) -> dict[str, Any]:
        """What get_permissions returns: the group defaults and each published tool's effective permission."""
        with self._locked():
            settings = self._settings_in(self._read(self.path))
            catalog = self._read(self.state_path).get(self.stack_id, {}).get("catalog") or {}
        tools = {name: {**dict(zip(("permission", "source"), permission_and_source(name, entry, settings))), "acts": bool(entry.get("acts")),
                        "risk": entry.get("risk"), "description": entry.get("description", ""),
                        "settable": not (entry.get("never") or entry.get("fixed"))}
                 for name, entry in sorted(catalog.items())}
        return {"groups": settings["groups"], "tools": tools}

    def _ttl(self, mine: dict[str, Any]) -> float:
        return float(mine.get("approval_ttl_s") or self.approval_ttl_s)

    def _live_requests(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        requests = state.setdefault(self.stack_id, {}).setdefault("requests", {})
        now = self.clock()
        for request_id in [r for r, record in requests.items() if seconds_of(record["expires_at"]) <= now]:
            del requests[request_id]
        return requests

    def request(self, tool: str, arguments: dict[str, Any], acts: bool, risk: str, requested_by: str | None) -> dict[str, Any]:
        """A call of an ``approve`` tool: the request it makes or matches, with ``status`` pending, approved or denied.

        A request matches the same tool, the same canonical arguments and the same caller (``requested_by``), so another
        caller's identical call cannot use an approval meant for someone else. An approved request is consumed here, so
        the call it approved runs once.
        """
        digest = arguments_sha256(arguments)
        with self._locked():
            state = self._read(self.state_path)
            requests = self._live_requests(state)
            found = next((r for r in requests.values()
                          if r["tool"] == tool and r["arguments_sha256"] == digest and r.get("requested_by") == requested_by), None)
            if found is not None and found["status"] == "approved":
                del requests[found["request_id"]]
            elif found is None:
                now = self.clock()
                found = {"request_id": str(uuid.uuid4()), "tool": tool, "arguments": json.loads(canonical_json(arguments)),
                         "arguments_sha256": digest, "acts": bool(acts), "risk": risk, "requested_at": iso_utc(now),
                         "expires_at": iso_utc(now + self._ttl(state[self.stack_id])), "requested_by": requested_by,
                         "status": "pending"}
                requests[found["request_id"]] = found
            self._write(self.state_path, state)
            return dict(found)

    def pending(self) -> list[dict[str, Any]]:
        with self._locked():
            state = self._read(self.state_path)
            requests = self._live_requests(state)
            self._write(self.state_path, state)
        shown = ("request_id", "tool", "arguments", "acts", "risk", "requested_at", "expires_at", "requested_by")
        return [{k: r.get(k) for k in shown} for r in sorted(requests.values(), key=lambda r: r["requested_at"]) if r["status"] == "pending"]

    def _decide(self, request_id: str, status: str, reason: str | None = None) -> dict[str, Any]:
        with self._locked():
            state = self._read(self.state_path)
            requests = self._live_requests(state)
            record = requests.get(request_id)
            if record is None:
                self._write(self.state_path, state)
                raise ValueError(f"no pending request {request_id!r}; it was never made, has expired, or was already used")
            if record["status"] != "pending":
                raise ValueError(f"request {request_id} is already {record['status']}")
            record["status"] = status
            record["decided_at"] = iso_utc(self.clock())
            record["expires_at"] = iso_utc(self.clock() + self._ttl(state[self.stack_id]))
            if reason is not None:
                record["reason"] = reason
            self._write(self.state_path, state)
            return dict(record)

    def approve(self, request_id: str) -> dict[str, Any]:
        return self._decide(request_id, "approved")

    def deny(self, request_id: str, reason: str) -> dict[str, Any]:
        return self._decide(request_id, "denied", str(reason))
