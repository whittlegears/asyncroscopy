"""The analysis MCP server as a ``StitchService``: the one module in ``jobs/`` that knows about MCP.

Loaded the first time a Mapper's ``stitch_service`` is ``mcp`` (stitch_service.py), and ``fastmcp`` only at the first call,
so a job device with another backend, or with no ``analysis_url``, never imports it.
"""

from __future__ import annotations

import asyncio
import json
import threading

from .stitch_service import register_stitch_service

ANALYSIS_CALL_S = 60.0  # longest one analysis-server call may take


class AnalysisSession:
    """One MCP session to the analysis server, used from any thread; reopened after a failure."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="mapper-analysis", daemon=True).start()
                self._loop = loop
            return self._loop

    async def _call(self, tool: str, arguments: dict, timeout: float) -> dict:
        from fastmcp import Client

        if self._client is None:
            client = Client(self.url, timeout=timeout)
            await client.__aenter__()
            self._client = client
        try:
            result = await asyncio.wait_for(self._client.call_tool(tool, arguments, raise_on_error=False), timeout)
        except BaseException:
            await self._drop()
            raise
        if result.is_error:
            raise RuntimeError(" ".join(getattr(b, "text", "") for b in result.content) or f"{tool} failed")
        if result.structured_content is not None:
            return result.structured_content
        return json.loads("".join(getattr(b, "text", "") for b in result.content))

    async def _drop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except BaseException:
                pass

    def call(self, tool: str, arguments: dict, timeout: float = ANALYSIS_CALL_S) -> dict:
        future = asyncio.run_coroutine_threadsafe(self._call(tool, arguments, timeout), self._ensure_loop())
        return future.result(timeout + 10.0)

    def close(self) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._drop(), self._loop).result(10.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None


@register_stitch_service
class McpStitchService:
    """Calls the tools of the analysis server at ``url`` over one session."""

    name = "mcp"

    def __init__(self, url: str) -> None:
        self.url = url
        self._session = AnalysisSession(url)

    def call(self, tool: str, arguments: dict) -> dict:
        return self._session.call(tool, arguments)

    def close(self) -> None:
        self._session.close()
