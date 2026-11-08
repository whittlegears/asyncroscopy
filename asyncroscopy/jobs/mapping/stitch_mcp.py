"""The analysis MCP server as a ``StitchService``: the one module in ``jobs/`` that knows about MCP.

Loaded the first time a Mapper's ``stitch_service`` is ``mcp`` (stitch_service.py), and ``fastmcp`` only at the first call,
so a job device with another backend, or with no ``analysis_url``, never imports it.
"""

from __future__ import annotations

import asyncio
import json
import threading

from .stitch_service import register_stitch_service

ANALYSIS_CALL_S = 60.0  # longest one call to the analysis server may take


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
    """Each operation is the tool of the same name on the analysis server at ``url``, over one session."""

    name = "mcp"

    def __init__(self, url: str) -> None:
        self.url = url
        self._session = AnalysisSession(url)

    def assemble_map(self, run: str, stage_sigma_m: float, max_drift_m_per_s: float, eds_pois: bool) -> dict:
        return self._session.call("assemble_map", {"run": run, "stage_sigma_m": stage_sigma_m, "max_drift_m_per_s": max_drift_m_per_s, "eds_pois": eds_pois})

    def score_map(self, map: str, top_k: int, tag: str) -> dict:
        return self._session.call("score_map", {"map": map, "top_k": top_k, "tag": tag})

    def analysis_job_status(self, job_id: str, wait_s: float) -> dict:
        return self._session.call("analysis_job_status", {"job_id": job_id, "wait_s": wait_s})

    def analysis_job_results(self, job_id: str) -> dict:
        return self._session.call("analysis_job_results", {"job_id": job_id})

    def analysis_job_abort(self, job_id: str) -> dict:
        return self._session.call("analysis_job_abort", {"job_id": job_id})

    def map_info(self, map: str) -> dict:
        return self._session.call("map_info", {"map": map})

    def list_pois(self, map: str, kind: str, limit: int) -> dict:
        return self._session.call("list_pois", {"map": map, "kind": kind, "limit": limit})

    def add_poi(self, map: str, x_m: float, y_m: float, kind: str, reason: str, links: list[str], id: str) -> dict:
        return self._session.call("add_poi", {"map": map, "x_m": x_m, "y_m": y_m, "kind": kind, "reason": reason, "links": links, "id": id})

    def find_features(self, image: str, n: int, min_size_m: float | None = None, max_size_m: float | None = None) -> dict:
        extras = {k: v for k, v in (("min_size_m", min_size_m), ("max_size_m", max_size_m)) if v is not None}
        return self._session.call("find_features", {"image": image, "n": n, **extras})

    def place_image(self, image: str, parent: str, centre_xy_m: list[float], window_m: float) -> dict:
        return self._session.call("place_image", {"image": image, "parent": parent, "centre_xy_m": centre_xy_m, "window_m": window_m})

    def close(self) -> None:
        self._session.close()
