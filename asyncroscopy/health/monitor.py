"""Device health, failure diagnosis, and a recovery ladder.

An autonomous session lives or dies on what happens when a link drops: the
corrector PC reboots, a vendor server hangs, a device server process exits.
This module lets the agent, the operator page and the ``asyncroscopy-doctor``
command all answer the same questions the same way:

* :meth:`HealthMonitor.check` — is the device reachable, what state is it in
  (ON / FAULT / ALARM / ...), how slow is it, which server hosts it?
* :func:`diagnose` — turn an exception from a device call into a kind
  (``unreachable``, ``timeout``, ``fault``, ``device_error``) and a next step.
* :func:`recover` — climb a ladder, re-checking after every rung:

  1. ``reconnect``: the device's own reconnect command, when it has one
     (re-opens the vendor link without touching anything else);
  2. ``init``: Tango ``Init`` (re-runs the device's initialisation);
  3. ``restart_server``: restart the whole device server through its admin
     device. Disruptive: every device in that server restarts.

  Rungs not in *allowed* are not run; the result says which rung is next and
  that it needs approval, so the operator decides on the disruptive ones.

Nothing here is specific to one vendor: it only uses standard Tango calls and
an optional ``reconnect`` command.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import tango
from tango import DevState

LADDER = ("reconnect", "init", "restart_server")
DEFAULT_ALLOWED = ("reconnect", "init")
BAD_STATES = {"FAULT": "fault", "ALARM": "alarm", "UNKNOWN": "unknown_state", "DISABLE": "disabled"}

UNREACHABLE_REASONS = ("API_DeviceNotExported", "API_CantConnectToDevice", "API_CorbaException",
                       "API_ServerNotRunning", "API_DeviceNotDefined", "API_CommunicationFailed",
                       "API_ConnectionFailed", "TRANSIENT", "API_CantConnectToDatabase")
TIMEOUT_REASONS = ("API_DeviceTimedOut", "API_CommandTimedOut", "TIMEOUT")


@dataclass
class DeviceHealth:
    device: str
    reachable: bool
    state: str | None = None
    status: str | None = None
    dev_class: str | None = None
    server: str | None = None
    latency_ms: float | None = None
    problem: str | None = None           # None | unreachable | fault | alarm | slow | unknown_state | disabled
    detail: str = ""
    has_reconnect: bool = False
    checked_at: float = field(default_factory=time.time)

    @property
    def healthy(self) -> bool:
        return self.problem is None

    def to_json(self) -> dict:
        return {**asdict(self), "healthy": self.healthy}


def _reasons(exc: BaseException) -> list[str]:
    if isinstance(exc, tango.DevFailed) and exc.args:
        return [str(getattr(e, "reason", "")) for e in exc.args]
    return []


def _desc(exc: BaseException) -> str:
    if isinstance(exc, tango.DevFailed) and exc.args:
        first = exc.args[0]
        return f"{first.reason}: {' '.join(str(first.desc).split())}"
    return f"{type(exc).__name__}: {exc}"


def diagnose(exc: BaseException) -> str:
    """Classify a failed device call: ``unreachable``, ``timeout`` or ``device_error``."""
    if isinstance(exc, (tango.ConnectionFailed, tango.CommunicationFailed)):
        reasons = " ".join(_reasons(exc))
        return "timeout" if any(r in reasons for r in TIMEOUT_REASONS) else "unreachable"
    reasons = " ".join(_reasons(exc))
    if any(r in reasons for r in TIMEOUT_REASONS):
        return "timeout"
    if any(r in reasons for r in UNREACHABLE_REASONS):
        return "unreachable"
    text = str(exc)
    if "ConnectionError" in text or "link" in text.lower() and "down" in text.lower():
        return "link_down"
    return "device_error"


class HealthMonitor:
    """Checks Tango devices. *devices* fixes the list; otherwise it is read from the Tango DB."""

    def __init__(self, devices: list[str] | None = None, *, database: Any = None,
                 proxy: Callable[[str], Any] = tango.DeviceProxy, slow_ms: float = 1000.0,
                 timeout_ms: int = 3000):
        self._devices = list(devices) if devices else None
        self._database = database
        self._proxy = proxy
        self.slow_ms = slow_ms
        self.timeout_ms = timeout_ms

    def devices(self) -> list[str]:
        if self._devices is not None:
            return list(self._devices)
        db = self._database if self._database is not None else tango.Database()
        # Every device the database *defines*, not only the exported ones: a server that
        # stopped unexports its devices, and a check that only looks at exported devices
        # would report a crashed server as "all OK".
        names: list[str] = []
        try:
            for server in db.get_server_list():
                if server.split("/")[0] in ("DataBaseds", "TangoAccessControl", "TangoTest", "Starter"):
                    continue
                entries = list(db.get_device_class_list(server))
                names.extend(entries[0::2])
        except Exception:
            names = list(db.get_device_exported("*").value_string)
        seen = dict.fromkeys(n for n in names if not n.lower().startswith(("dserver/", "sys/")))
        return list(seen)

    def _open(self, device: str):
        proxy = self._proxy(device)
        try:
            proxy.set_timeout_millis(self.timeout_ms)
        except Exception:
            pass
        return proxy

    def check(self, device: str) -> DeviceHealth:
        try:
            proxy = self._open(device)
            t0 = time.perf_counter()
            proxy.ping()
            latency = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            return DeviceHealth(device, False, problem="unreachable", detail=_desc(exc),
                                server=self._server_from_db(device))
        h = DeviceHealth(device, True, latency_ms=round(latency, 2))
        try:
            info = proxy.info()
            h.dev_class, h.server = info.dev_class, info.server_id
        except Exception:
            h.server = self._server_from_db(device)
        try:
            state = proxy.state()
            h.state = str(state)
            h.status = " ".join(str(proxy.status()).split())
        except Exception as exc:
            h.problem, h.detail = "unreachable", f"state could not be read: {_desc(exc)}"
            return h
        try:
            h.has_reconnect = any(c.cmd_name == "reconnect" for c in proxy.command_list_query())
        except Exception:
            pass
        if h.state in BAD_STATES:
            h.problem, h.detail = BAD_STATES[h.state], h.status or ""
        elif latency > self.slow_ms:
            h.problem, h.detail = "slow", f"ping took {latency:.0f} ms (> {self.slow_ms:.0f} ms)"
        return h

    def check_all(self) -> list[DeviceHealth]:
        return [self.check(d) for d in self.devices()]

    def _server_from_db(self, device: str) -> str | None:
        try:
            db = self._database if self._database is not None else tango.Database()
            return db.get_device_info(device).ds_full_name
        except Exception:
            return None

    # ------------------------------------------------------------------ recovery
    def _run_rung(self, device: str, rung: str) -> str:
        proxy = self._open(device)
        if rung == "reconnect":
            proxy.command_inout("reconnect")
            return "reconnect sent"
        if rung == "init":
            proxy.command_inout("Init")
            return "Init sent"
        if rung == "restart_server":
            admin = self._proxy(proxy.adm_name())
            admin.command_inout("RestartServer")
            return f"RestartServer sent to {proxy.adm_name()}"
        raise ValueError(f"unknown recovery rung {rung!r}")

    def admin_alive(self, server: str) -> bool:
        """Does the device server process still answer (through its admin device)?"""
        try:
            admin = self._proxy(f"dserver/{server}")
            try:
                admin.set_timeout_millis(self.timeout_ms)
            except Exception:
                pass
            admin.ping()
            return True
        except Exception:
            return False

    def wait_healthy(self, device: str, timeout_s: float = 10.0, poll_s: float = 0.25) -> DeviceHealth:
        deadline = time.monotonic() + timeout_s
        h = self.check(device)
        while not h.healthy and time.monotonic() < deadline:
            time.sleep(poll_s)
            h = self.check(device)
        return h


def recover(monitor: HealthMonitor, device: str, *, allowed: tuple[str, ...] = DEFAULT_ALLOWED,
            step: Callable[[str, dict, Callable[[], Any]], Any] | None = None,
            settle_s: float = 10.0, start_at: str | None = None) -> dict:
    """Try the recovery ladder on *device*; stop at the first rung that leaves it healthy.

    *step(name, args, fn)* runs each rung (the action queue passes its own, so
    each rung is a recorded child action under the policy). *start_at* skips
    the rungs before it (used once an operator approves a later rung).
    """
    run = step or (lambda name, args, fn: fn())
    before = monitor.check(device)
    out: dict[str, Any] = {"device": device, "before": before.to_json(), "attempts": []}
    if before.healthy:
        out.update(status="healthy", after=before.to_json())
        return out
    if not before.reachable and before.server and not monitor.admin_alive(before.server):
        # The process is gone: Init/RestartServer go through that process, so they cannot help.
        # The launcher's watchdog relaunches exited servers; give it a moment.
        after = monitor.wait_healthy(device, settle_s)
        if after.healthy:
            out.update(status="recovered", by="watchdog_restart", after=after.to_json())
        else:
            out.update(status="server_down", after=after.to_json(), message=_unrecoverable_hint(after))
        return out
    ladder = LADDER[LADDER.index(start_at):] if start_at else LADDER
    for rung in ladder:
        if rung == "reconnect" and not before.has_reconnect:
            continue
        if rung != "restart_server" and not before.reachable:
            continue                 # Init/reconnect need the device to answer
        if rung not in allowed:
            out.update(status="needs_approval", next_rung=rung,
                       message=f"{device} is still {before.problem}; the next step, {rung!r}, needs operator "
                               f"approval" + (" (it restarts every device in that server)"
                                             if rung == "restart_server" else ""))
            out["after"] = monitor.check(device).to_json()
            return out
        attempt: dict[str, Any] = {"rung": rung, "t": time.time()}
        try:
            attempt["result"] = run(f"recovery.{rung}", {"device": device}, lambda r=rung: monitor._run_rung(device, r))
        except Exception as exc:
            attempt["error"] = _desc(exc)
        after = monitor.wait_healthy(device, settle_s)
        attempt["after"] = after.to_json()
        out["attempts"].append(attempt)
        if after.healthy:
            out.update(status="recovered", by=rung, after=after.to_json())
            return out
    final = monitor.check(device)
    out.update(status="failed", after=final.to_json(),
               message=_unrecoverable_hint(final))
    return out


def _unrecoverable_hint(h: DeviceHealth) -> str:
    if not h.reachable:
        return (f"{h.device} does not answer and its server process{' (' + h.server + ')' if h.server else ''} "
                f"has exited. run_servers.py restarts exited servers automatically unless --no-watchdog is set; "
                f"if it gave up, check that server's log, then restart the stack on the instrument PC.")
    return (f"{h.device} is reachable but still {h.problem} ({h.detail}); check the vendor software or "
            f"hardware it connects to.")


def hint_for(kind: str, device: str) -> str:
    """One line the agent can act on after a failed call."""
    return {
        "unreachable": f"Device {device} is not answering. Call system_health, then recover_device('{device}'); "
                       f"do not retry the command first.",
        "timeout": f"Device {device} timed out; the command may or may not have taken effect (see readback). "
                   f"Call system_health before retrying.",
        "link_down": f"The link behind {device} is down. Call recover_device('{device}').",
    }.get(kind, "")


def is_bad_state(state: Any) -> bool:
    return str(state) in BAD_STATES or state in (DevState.FAULT, DevState.ALARM)
