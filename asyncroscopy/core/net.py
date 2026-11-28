"""Which address the MCP endpoint may bind: on hardware, loopback only unless the config allows remote.

A host is remote unless it is a loopback address (127.0.0.0/8, ::1, an IPv4-mapped
loopback) or the name ``localhost``. The wildcards (``0.0.0.0``, ``::``, ``""``), a
LAN address such as ``192.168.1.20`` and any other host name all count as remote:
each lets another machine reach the endpoint. Names are not resolved; a name other
than ``localhost`` counts as remote, the safe answer.
"""

from __future__ import annotations

import ipaddress
import socket

LOOPBACK_FALLBACK = "127.0.0.1"
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})  # bind addresses that mean every interface, reached locally on loopback


def is_loopback_host(host: str) -> bool:
    """True only for an address (or ``localhost``) that other machines cannot reach."""
    text = str(host or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.split("%", 1)[0]  # drop an IPv6 zone id, which does not change the address
    if text.lower().rstrip(".") == "localhost":
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(address.is_loopback or (mapped is not None and mapped.is_loopback))


def is_remote_host(host: str) -> bool:
    """True when binding ``host`` would accept connections from other machines (or might)."""
    return not is_loopback_host(host)


def bind_host(requested: str, allow_remote_bind: bool, simulated: bool) -> tuple[str, str | None]:
    """(host to bind, note): a stack that is not simulated binds a remote host only when its MCP config says
    allow_remote_bind: true; otherwise 127.0.0.1, so only this machine can drive the hardware."""
    if is_loopback_host(requested) or allow_remote_bind or simulated:
        return requested, None
    return LOOPBACK_FALLBACK, (
        f"[safety] http_host {requested!r} would accept connections from other machines, and this stack is not simulated "
        "(or its devices could not say). Binding 127.0.0.1 instead; to accept other machines on real hardware, "
        "set mcp.allow_remote_bind: true in the MCP config."
    )


def check_port_free(host: str, port: int) -> str | None:
    """An error string if (host, port) cannot be bound, else None: a previous MCP server may still hold the port."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        return str(exc)
    finally:
        probe.close()
    return None


def report_busy_port(host: str, port: int, server_kind: str) -> bool:
    """Print the "MCP ERROR:" line the startup GUIs parse and return True when (host, port) is taken."""
    bind_error = check_port_free(host, port)
    if bind_error:
        print(f"MCP ERROR: http://{host}:{port} is already in use - another {server_kind} is likely still running "
              f"with a different config. Stop it or choose a different port. ({bind_error})", flush=True)
    return bool(bind_error)
