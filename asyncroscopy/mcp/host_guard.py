"""Which Host and Origin an MCP HTTP endpoint answers: a guard against DNS rebinding.

A web page on another site can make the browser of someone on this machine send requests to 127.0.0.1:8000 once
its own name resolves there (DNS rebinding). Such a request carries that site's name as its Host, and its Origin.
The guard answers 403 to any request whose Host is not one this endpoint is reached by, and to any request whose
Origin names another host, before the MCP handler sees it.

The hosts allowed are the bind host, ``localhost``, ``127.0.0.1`` and ``::1``; when the endpoint binds every
interface (``0.0.0.0``, ``::``), also this computer's own name and addresses; and the config's ``allowed_hosts``
(``mcp.allowed_hosts`` for run_mcp, ``agent.allowed_hosts`` for run_llm), for a name a proxy or another machine
reaches it by. A port in Host or Origin is not checked: it is already the port the endpoint listens on.
"""

from __future__ import annotations

import socket
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse

from asyncroscopy.core.net import WILDCARD_HOSTS

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def _bare(host: str) -> str:
    """A host without port or brackets, lower case."""
    text = str(host or "").strip().lower().rstrip(".")
    if text.startswith("["):
        return text[1:text.index("]")] if "]" in text else text[1:]
    if text.count(":") == 1:
        return text.split(":", 1)[0]
    return text


def _this_computer() -> set[str]:
    names = {socket.gethostname(), socket.getfqdn()}
    found = {n for n in names if n}
    for name in names:
        try:
            found |= {str(info[4][0]) for info in socket.getaddrinfo(name, None)}
        except OSError:
            continue
    return found


def allowed_hosts(bind_host: str, extra: Iterable[str] = ()) -> frozenset[str]:
    """The hosts an endpoint bound to ``bind_host`` answers (see the module docstring)."""
    hosts = set(LOCAL_HOSTS) | {str(h) for h in extra}
    bound = str(bind_host or "").strip()
    if bound in WILDCARD_HOSTS:
        hosts |= _this_computer()
    else:
        hosts.add(bound)
    return frozenset(_bare(h) for h in hosts if _bare(h))


class HostGuard:
    """ASGI middleware: 403 for a foreign Host or Origin (see the module docstring)."""

    def __init__(self, app: Any, hosts: frozenset[str]) -> None:
        self.app = app
        self.hosts = hosts

    def refusal(self, headers: dict[str, str]) -> str | None:
        host = _bare(headers.get("host", ""))
        if host not in self.hosts:
            return f"Host {headers.get('host', '')!r} is not one this endpoint answers; add it to allowed_hosts in the MCP config"
        origin = headers.get("origin")
        if origin is not None and _bare(urlsplit(origin).netloc) not in self.hosts:
            return f"Origin {origin!r} is not one this endpoint answers; add its host to allowed_hosts in the MCP config"
        return None

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            reason = self.refusal(headers)
            if reason is not None:
                await PlainTextResponse(reason, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def middleware(bind_host: str, extra: Iterable[str] = ()) -> list[Middleware]:
    """The guard, for FastMCP's ``run(..., middleware=...)`` or ``http_app(middleware=...)``."""
    return [Middleware(HostGuard, hosts=allowed_hosts(bind_host, extra))]
