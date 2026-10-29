"""The one way asyncroscopy code reaches Tiled: bounded timeouts, retries, and errors an agent can act on.

Everything that reads or registers data goes through ``connect`` (or ``node_for_key``),
never ``tiled.client.from_uri`` directly, so a Tiled outage gives the same clear message
everywhere instead of a raw HTTP error or a hang.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import httpx

from asyncroscopy.mcp.binding import is_loopback_host

TIMEOUT = httpx.Timeout(connect=3.0, read=15.0, write=15.0, pool=3.0)
RETRY_DELAYS_S = (0.5,)  # on top of the Tiled client's own two attempts
URI_CACHE_S = 30.0
DEFAULT_URI = "http://127.0.0.1:9091"

# The Tiled client retries every request on a connection error for up to 45 s
# (10 attempts). A dead Tiled would then hang each call past the 12-30 s the
# instrument devices and the MCP bridge wait, so keep its retries short. An
# explicit TILED_RETRY_* in the environment still wins. The values are read at
# import by tiled, and at call time from tiled.client.utils, so both are set.
os.environ.setdefault("TILED_RETRY_ATTEMPTS", "2")
os.environ.setdefault("TILED_RETRY_TIMEOUT", "3")


def _limit_tiled_retries() -> None:
    try:
        from tiled.client import utils as tiled_utils
    except ImportError:
        return
    tiled_utils.TILED_RETRY_ATTEMPTS = int(os.environ["TILED_RETRY_ATTEMPTS"])
    tiled_utils.TILED_RETRY_TIMEOUT = float(os.environ["TILED_RETRY_TIMEOUT"])


_limit_tiled_retries()


class TiledUnavailable(RuntimeError):
    """Tiled is not answering (down, restarting, or the address is wrong)."""


class DataKeyNotFound(LookupError):
    """Tiled answered, but has no entry with this key."""

    def __str__(self) -> str:  # LookupError would quote the message
        return str(self.args[0]) if self.args else ""


_clients: dict[str, Any] = {}
_uris: dict[str, tuple[str, float]] = {}
_lock = threading.Lock()


def api_key() -> str:
    """The Tiled API key: ASYNCROSCOPY_TILED_API_KEY, default "secret" (the loopback default SciAgentGUI uses)."""
    return os.environ.get("ASYNCROSCOPY_TILED_API_KEY", "secret")


def _nothing_listening(uri: str) -> bool:
    """True when ``uri`` is on this machine and no process listens on its port.

    Windows takes about 2 s to refuse a connection to a closed local port; asking
    the OS for its listening sockets answers the common "Tiled is down" case at once.
    """
    parts = urlsplit(uri)
    if not is_loopback_host(parts.hostname or ""):
        return False
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        import psutil

        return not any(
            c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port for c in psutil.net_connections(kind="tcp")
        )
    except Exception:  # psutil missing or not permitted: fall back to connecting
        return False


def about(uri: str, timeout_s: float = 2.0) -> dict | None:
    """Tiled's about document at ``uri``, or None if nothing answers there or what answers is not Tiled."""
    if _nothing_listening(uri):
        return None
    try:
        request = Request(uri.rstrip("/") + "/api/v1/", headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_s) as response:
            document = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError):
        return None
    if isinstance(document, dict) and "api_version" in document and "library_version" in document:
        return document
    return None


# Failures that mean Tiled cannot be reached. Not OSError in general: FileExistsError,
# FileNotFoundError and TimeoutError are OSErrors too, and mean something else.
CONNECTION_ERRORS = (httpx.TransportError, ConnectionError)


def unavailable(uri: str, exc: BaseException) -> TiledUnavailable:
    """Forget the cached client for ``uri`` and describe the failure as Tiled being unreachable."""
    forget(uri)
    return _unavailable(uri, exc)


def _unavailable(uri: str, exc: BaseException) -> TiledUnavailable:
    return TiledUnavailable(
        f"Tiled at {uri} is not answering ({type(exc).__name__}: {exc}). The DATA device restarts the Tiled "
        "it manages; its get_config shows tiled_server_status and tiled_log. Try again in a few seconds."
    )


def connect(uri: str) -> Any:
    """A cached Tiled client for ``uri``; retries twice on connection errors, then raises TiledUnavailable."""
    with _lock:
        client = _clients.get(uri)
    if client is not None:
        return client
    if _nothing_listening(uri):
        raise _unavailable(uri, ConnectionRefusedError("nothing is listening on that port"))
    from tiled.client import from_uri

    last: BaseException | None = None
    for delay in (0.0, *RETRY_DELAYS_S):
        if delay:
            time.sleep(delay)
        try:
            client = from_uri(uri, api_key=api_key(), timeout=TIMEOUT)
        except CONNECTION_ERRORS as exc:
            last = exc
            continue
        with _lock:
            _clients[uri] = client
        return client
    raise _unavailable(uri, last)


def forget(uri: str) -> None:
    """Drop the cached client for ``uri`` (after Tiled restarted, or a call through it failed)."""
    with _lock:
        _clients.pop(uri, None)


def uri_from_data_device(address: str) -> str:
    """The Tiled URI the DATA device at ``address`` serves; cached for 30 s, so a busy DATA device does not block reads."""
    with _lock:
        cached = _uris.get(address)
    if cached is not None and time.monotonic() - cached[1] < URI_CACHE_S:
        return cached[0]
    try:
        from tango import DeviceProxy

        proxy = DeviceProxy(address)
        proxy.set_timeout_millis(5000)
        uri = json.loads(proxy.get_config()).get("uri")
    except Exception as exc:  # Tango raises DevFailed; a busy DATA device times out
        if cached is not None:
            return cached[0]
        fallback = os.environ.get("ASYNCROSCOPY_TILED_URI")
        if fallback:
            return fallback
        raise TiledUnavailable(f"could not ask the DATA device {address!r} for the Tiled address ({exc})") from exc
    if not uri:
        raise TiledUnavailable(f"the DATA device {address!r} reports no Tiled uri")
    with _lock:
        _uris[address] = (uri, time.monotonic())
    return uri


def node_for_key(key: str, uri: str) -> Any:
    """The Tiled node for ``key``; DataKeyNotFound names close keys, TiledUnavailable if Tiled does not answer."""
    client = connect(uri)
    try:
        return client[key]
    except KeyError:
        pass
    except CONNECTION_ERRORS as exc:
        raise unavailable(uri, exc) from exc
    try:
        close = [k for k in client.keys() if k.startswith(key)][:10]
    except Exception:
        close = []
    hint = f" Keys that start with it: {close}." if close else ""
    raise DataKeyNotFound(
        f"Tiled at {uri} has no key {key!r}.{hint} If it came from an acquisition just now while Tiled was down, "
        "it is registered when Tiled is back; list_data shows the newest keys."
    )
