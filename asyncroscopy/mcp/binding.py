"""Which address the MCP endpoint may bind: on hardware, loopback only unless the config allows remote.

A host is remote unless it is a loopback address (127.0.0.0/8, ::1, an IPv4-mapped
loopback) or the name ``localhost``. The wildcards (``0.0.0.0``, ``::``, ``""``), a
LAN address such as ``192.168.1.20`` and any other host name all count as remote:
each lets another machine reach the endpoint. Names are not resolved; a name other
than ``localhost`` counts as remote, the safe answer.
"""

from __future__ import annotations

import ipaddress

LOOPBACK_FALLBACK = "127.0.0.1"
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})  # bind addresses that mean "every interface"; reached locally on loopback


def is_loopback_host(host: str) -> bool:
    """True only for an address (or ``localhost``) that other machines cannot reach."""
    text = str(host or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.split("%", 1)[0]  # an IPv6 zone id (fe80::1%eth0) does not change what the address is
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
        "(or its Approvals device could not say). Binding 127.0.0.1 instead; to accept other machines on real hardware, "
        "set mcp.allow_remote_bind: true in the MCP config."
    )
