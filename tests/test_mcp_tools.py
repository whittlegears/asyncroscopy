"""MCP tools as LangChain tools through the fastmcp client (replaces langchain-mcp-adapters)."""

import asyncio
import importlib
import json
import sys

import pytest


@pytest.fixture()
def real_langchain(monkeypatch):
    """The swarm tests stub langchain/langgraph; this needs the real langchain_core."""
    for name in list(sys.modules):
        if name.split(".")[0] in ("langgraph", "langchain", "langchain_core"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        importlib.import_module("langchain_core.tools")
    except Exception as exc:
        pytest.skip(f"langchain_core not available: {exc}")
    import asyncroscopy.mcp.mcp_tools as module
    return importlib.reload(module)


def test_tools_round_trip_through_a_real_mcp_server(real_langchain):
    from fastmcp import FastMCP
    from fastmcp.utilities.types import Image

    server = FastMCP("t")

    @server.tool
    def move_stage(position: list[float]) -> dict:
        """Move the stage."""
        return {"moved_to": position}

    @server.tool
    def get_fov() -> float:
        """Field of view."""
        return 2e-9

    @server.tool
    def acquire() -> list:
        """An acquisition with a preview."""
        return ["key-1", Image(data=b"\x89PNG\r\n\x1a\n", format="png")]

    @server.tool
    def broken() -> str:
        """Always fails."""
        raise RuntimeError("link to the corrector is down")

    async def run():
        source = real_langchain.McpToolSource(server)          # fastmcp clients also accept a server object
        tools = {t.name: t for t in await source.open()}
        try:
            assert set(tools) == {"move_stage", "get_fov", "acquire", "broken"}
            assert "position" in tools["move_stage"].args
            moved = json.loads(await tools["move_stage"].ainvoke({"position": [1e-9, 0, 0, 0, 0]}))
            assert moved == {"moved_to": [1e-9, 0, 0, 0, 0]}
            assert float(await tools["get_fov"].ainvoke({})) == 2e-9
            acquired = await tools["acquire"].ainvoke({})
            assert "key-1" in acquired and "[image" in acquired
            failed = await tools["broken"].ainvoke({})
            assert failed.startswith("Error:") and "link to the corrector is down" in failed
        finally:
            await source.close()

    asyncio.new_event_loop().run_until_complete(run())
