"""Tool permissions (allow, approve, never), held calls and run pre-approvals of one stack, in files shared by the agent
and operator endpoints under a lock; ``docs/MCP/permissions.md`` is the contract."""

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
max_run_preapproval_s = 12 * 3600.0
settable_permissions = ("allow", "approve")
permission_groups = ("reads", "acts")
lock_wait_s = 5.0


def group_of(acts: bool) -> str:
    return "acts" if acts else "reads"


def canonical_json(value: Any) -> str:
    return json.dumps({} if value is None else value, sort_keys=True, separators=(",", ":"), default=str)


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def run_key_sha256(run_key: str) -> str:
    return hashlib.sha256(str(run_key).encode("utf-8")).hexdigest()


def iso_utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seconds_of(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def permission_and_source(name: str, entry: dict[str, Any], settings: dict[str, Any]) -> tuple[str, str]:
    """(permission, its source); the order is the table in ``docs/MCP/permissions.md``."""
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

    def request(self, tool: str, arguments: dict[str, Any], acts: bool, risk: str, requested_by: str | None,
                signoff: dict[str, Any] | None = None, run: dict[str, str] | None = None) -> dict[str, Any]:
        """The request a call of an ``approve`` tool makes or matches (same tool, arguments, caller and run); an
        approved one is used up here."""
        digest = json_sha256(arguments)
        bound = {"run_id": (run or {}).get("run_id"), "run_key_sha256": (run or {}).get("run_key_sha256")}
        with self._locked():
            state = self._read(self.state_path)
            requests = self._live_requests(state)
            found = next((r for r in requests.values()
                          if r["tool"] == tool and r["arguments_sha256"] == digest and r.get("requested_by") == requested_by
                          and all(r.get(k) == v for k, v in bound.items())), None)
            if found is not None and found["status"] == "approved":
                del requests[found["request_id"]]
            elif found is None:
                now = self.clock()
                found = {"request_id": str(uuid.uuid4()), "tool": tool, "arguments": json.loads(canonical_json(arguments)),
                         "arguments_sha256": digest, "acts": bool(acts), "risk": risk, "requested_at": iso_utc(now),
                         "expires_at": iso_utc(now + self._ttl(state[self.stack_id])), "requested_by": requested_by,
                         "status": "pending", **({"signoff": dict(signoff)} if signoff else {}),
                         **({k: v for k, v in bound.items()} if run else {})}
                requests[found["request_id"]] = found
            self._write(self.state_path, state)
            return dict(found)

    def pending(self) -> list[dict[str, Any]]:
        with self._locked():
            state = self._read(self.state_path)
            requests = self._live_requests(state)
            self._write(self.state_path, state)
        shown = ("request_id", "tool", "arguments", "acts", "risk", "requested_at", "expires_at", "requested_by")
        return [{**{k: r.get(k) for k in shown}, **({"signoff": r["signoff"]} if r.get("signoff") else {})}
                for r in sorted(requests.values(), key=lambda r: r["requested_at"]) if r["status"] == "pending"]

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

    def _live_runs(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        mine = state.setdefault(self.stack_id, {})
        runs = mine.setdefault("runs", {})
        now = self.clock()
        for run_id in [r for r, grant in runs.items() if seconds_of(grant["expires_at"]) <= now]:
            del runs[run_id]
        uses = mine.setdefault("run_uses", {})
        for run_id in [r for r, kept in uses.items() if r not in runs and seconds_of(kept["at"]) + max_run_preapproval_s <= now]:
            del uses[run_id]
        return runs

    def _kept_alive(self, grant: dict[str, Any], mine: dict[str, Any]) -> bool:
        """Extend a grant's expiry; only once half the TTL is gone, so a run's many calls seldom write the file."""
        now = self.clock()
        ttl = self._ttl(mine)
        if seconds_of(grant["expires_at"]) - now >= ttl / 2:
            return False
        later = min(now + ttl, seconds_of(grant["granted_at"]) + max_run_preapproval_s)
        if later <= seconds_of(grant["expires_at"]):
            return False
        grant["expires_at"] = iso_utc(later)
        return True

    def live_grant(self, run_id: str, run_key_digest: str) -> dict[str, Any] | None:
        """The live pre-approval of ``run_id`` for the right key, kept alive by this call."""
        with self._locked():
            state = self._read(self.state_path)
            grant = self._live_runs(state).get(str(run_id))
            if grant is None or grant["run_key_sha256"] != str(run_key_digest):
                return None
            if self._kept_alive(grant, state[self.stack_id]):
                self._write(self.state_path, state)
        return dict(grant)

    def use_run_call(self, run_id: str, run_key_digest: str, call: str, allowed: int) -> bool:
        """Use one of the ``allowed`` passes of ``call``; the count outlives ``end_run`` so a resumed run keeps it."""
        with self._locked():
            state = self._read(self.state_path)
            grant = self._live_runs(state).get(str(run_id))
            if grant is None or grant["run_key_sha256"] != str(run_key_digest):
                return False
            kept = state[self.stack_id]["run_uses"].setdefault(str(run_id), {"used": {}})
            if int(kept["used"].get(call, 0)) >= int(allowed):
                return False
            kept["used"][call] = int(kept["used"].get(call, 0)) + 1
            kept["at"] = iso_utc(self.clock())
            self._kept_alive(grant, state[self.stack_id])
            self._write(self.state_path, state)
        return True

    def grant_run(self, run_id: str, run_key_digest: str, calls: list[dict[str, Any]], approved_by: str, approver: str,
                  workflow: str, hash_: str, rehearsal_run_id: str | None = None) -> dict[str, Any]:
        """Pre-approve the rehearsal's calls for the hardware run ``run_id``; a new grant for a resumed run replaces the
        old one and moves its held requests to the new key."""
        if not str(run_id).strip() or len(str(run_key_digest)) != 64:
            raise ValueError("a run pre-approval needs the run id and the SHA-256 of the run key")
        if not str(approver or "").strip() or not str(approved_by).startswith("workflow_signoff:"):
            raise ValueError("a run pre-approval names the sign-off (approved_by workflow_signoff:<workflow>@<hash>) and the approver")
        kept = [{"tool": str(c["tool"]), "arguments": json.loads(canonical_json(c.get("arguments") or {})), "count": int(c.get("count") or 1)}
                for c in calls]
        if any(c["count"] < 1 for c in kept):
            raise ValueError("a pre-approved call's count is 1 or more")
        with self._locked():
            state = self._read(self.state_path)
            runs = self._live_runs(state)
            now = self.clock()
            grant = {"run_id": str(run_id), "run_key_sha256": str(run_key_digest), "approved_by": str(approved_by), "approver": str(approver).strip(),
                     "workflow": str(workflow), "hash": str(hash_), "rehearsal_run_id": rehearsal_run_id, "granted_at": iso_utc(now),
                     "expires_at": iso_utc(now + self._ttl(state[self.stack_id])), "calls": kept}
            runs[str(run_id)] = grant
            for request in self._live_requests(state).values():
                if request.get("run_id") == str(run_id):
                    request["run_key_sha256"] = str(run_key_digest)
            self._write(self.state_path, state)
        return {k: v for k, v in grant.items() if k != "calls"} | {"calls": sum(c["count"] for c in kept)}

    def run_grant(self, run_id: str) -> dict[str, Any] | None:
        if not run_id:
            return None
        with self._locked():
            state = self._read(self.state_path)
            found = self._live_runs(state).get(str(run_id))
        return dict(found) if found else None

    def end_run(self, run_id: str) -> dict[str, Any] | None:
        with self._locked():
            state = self._read(self.state_path)
            removed = self._live_runs(state).pop(str(run_id), None)
            self._write(self.state_path, state)
        return None if removed is None else {k: v for k, v in removed.items() if k != "calls"} | {"calls": sum(int(c.get("count") or 1) for c in removed["calls"])}
