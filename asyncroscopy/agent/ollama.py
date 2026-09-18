"""Helpers for a local Ollama server."""

from __future__ import annotations

import asyncio
import subprocess
import time
import urllib.error
import urllib.request

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _tags_reachable(tags_url: str) -> bool:
    try:
        with urllib.request.urlopen(tags_url, timeout=1):
            return True
    except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError):
        return False


def ensure_ollama_running_sync(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 10.0) -> None:
    """Start ``ollama serve`` if the API at ``host`` is not answering, then wait for it."""
    tags_url = f"{host.rstrip('/')}/api/tags"
    if _tags_reachable(tags_url):
        return

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama binary not found on PATH.") from exc

    start_time = time.time()
    while time.time() - start_time < timeout:
        if _tags_reachable(tags_url):
            return
        time.sleep(0.5)

    raise RuntimeError(f"Ollama endpoint '{tags_url}' did not respond.")


async def ensure_ollama_running(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 10.0) -> None:
    """Async wrapper around :func:`ensure_ollama_running_sync` (runs in a thread)."""
    await asyncio.to_thread(ensure_ollama_running_sync, host, timeout)
