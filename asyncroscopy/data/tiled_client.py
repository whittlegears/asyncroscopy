"""Single place that knows how to reach the Tiled HTTP data server.

Every reader and writer (DATA device, MCP bridge, instrument devices) builds
its client here so the URI and API key are configured once.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from tiled.client import from_uri

TILED_URI_ENV = "ASYNCROSCOPY_TILED_URI"
TILED_API_KEY_ENV = "ASYNCROSCOPY_TILED_API_KEY"
TILED_ALLOW_ORIGINS_ENV = "ASYNCROSCOPY_TILED_ALLOW_ORIGINS"
DEFAULT_TILED_URI = "http://10.46.217.241:9091"
DEFAULT_TILED_API_KEY = "secret"
# Browser GUIs (SciAgentGUI vite dev/preview) that may read Tiled and the MCP
# bridge cross-origin. Deliberately loopback only, never "*".
DEFAULT_BROWSER_ORIGINS = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://localhost:1421",
    "http://127.0.0.1:1421",
]


def allowed_origins() -> list[str]:
    """Origins allowed to read Tiled/MCP from a browser; env overrides the default (space or comma separated)."""
    raw = os.environ.get(TILED_ALLOW_ORIGINS_ENV, "").strip()
    if not raw:
        return list(DEFAULT_BROWSER_ORIGINS)
    return [item for item in re.split(r"[\s,]+", raw) if item]

# Acquisition keys are "<acquisition_type>_<detector>_<YYYYmmddTHHMMSSffffff>.<ext>"
# (data_writer.acquisition_filename); the stamp is the only chronological handle.
_KEY_STAMP = re.compile(r"_(\d{8}T\d{6}\d{0,6})(?:_[^_]+)?\.[A-Za-z0-9]+$")


def default_api_key() -> str:
    return os.environ.get(TILED_API_KEY_ENV, DEFAULT_TILED_API_KEY)


def default_uri() -> str:
    return os.environ.get(TILED_URI_ENV, DEFAULT_TILED_URI)


def open_client(uri: str, api_key: str | None = None) -> Any:
    """Open a Tiled client for ``uri`` with the configured API key."""
    return from_uri(uri, api_key=api_key or default_api_key())


def uri_from_data_proxy(data_proxy: Any) -> str:
    """Read the Tiled URI the DATA device is configured for."""
    config = json.loads(data_proxy.get_config())
    uri = config.get("uri")
    if not uri:
        raise RuntimeError("the DATA device's config carries no Tiled uri")
    return uri


def key_timestamp(key: str) -> datetime | None:
    """Parse the acquisition timestamp embedded in a Tiled key, if any."""
    match = _KEY_STAMP.search(key)
    if not match:
        return None
    stamp = match.group(1)
    for fmt in ("%Y%m%dT%H%M%S%f", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(stamp, fmt)
        except ValueError:
            continue
    return None


def list_acquisitions(
    client: Any,
    acquisition_type: str | None = None,
    since: str | None = None,
    limit: int = 20,
    with_metadata: bool = True,
) -> list[dict[str, Any]]:
    """Most recent acquisition keys first, optionally filtered by type and time.

    ``since`` is an ISO-8601 timestamp. Keys without a parseable stamp sort last.
    """
    since_dt = datetime.fromisoformat(since) if since else None
    entries: list[tuple[datetime | None, str]] = []
    for key in client.keys():
        key = str(key)
        if acquisition_type and not key.startswith(f"{acquisition_type}_"):
            continue
        stamp = key_timestamp(key)
        if since_dt is not None and (stamp is None or stamp < since_dt):
            continue
        entries.append((stamp, key))
    entries.sort(key=lambda item: (item[0] is None, -(item[0].timestamp()) if item[0] else 0.0, item[1]))

    result: list[dict[str, Any]] = []
    for stamp, key in entries[: max(0, int(limit))]:
        item: dict[str, Any] = {"key": key, "timestamp": stamp.isoformat() if stamp else None}
        if with_metadata:
            try:
                item["metadata"] = _plain(dict(getattr(client[key], "metadata", {}) or {}))
            except Exception as exc:
                item["metadata"] = {"error": f"{type(exc).__name__}: {exc}"}
        result.append(item)
    return result


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _plain(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj
