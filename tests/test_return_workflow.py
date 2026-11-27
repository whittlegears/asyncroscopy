"""The return-to-site LangGraph workflow, end to end over (fake) MCP tool calls on the synthetic microscope."""

import asyncio
import importlib
import sys

import numpy as np
import pytest

from asyncroscopy.imaging.synthetic import SyntheticMicroscope

FOV, N = 2e-6, 128


@pytest.fixture()
def real_langgraph(monkeypatch):
    """The swarm tests stub langchain/langgraph; this test needs the real ones."""
    for name in list(sys.modules):
        if name.split(".")[0] in ("langgraph", "langchain", "langchain_core"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        importlib.import_module("langgraph.graph")
    except Exception as exc:  # the agent extra is not installed
        pytest.skip(f"langgraph not available: {exc}")
    import asyncroscopy.mcp.workflows.return_to_site as module
    return importlib.reload(module)


class FakeMcp:
    """MCP tool calls answered by a synthetic microscope and the server's real register_images."""

    def __init__(self, monkeypatch, scope):
        from asyncroscopy.mcp.mcp_server import MCPServer

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        self.server = MCPServer("t", "localhost", 1, blocked_classes=[], blocked_functions={},
                                data_device_address="", verbose=False)
        self.images: dict[str, np.ndarray] = {}
        monkeypatch.setattr(self.server, "_fetch_image_array", lambda key: self.images[key])
        self.scope = scope
        self.fov = FOV
        self.calls: list[str] = []
        self.tools = ["Twin_get_stage", "Twin_move_stage", "Twin_get_fov", "Twin_set_fov",
                      "Twin_acquire_scanned_image", "register_images", "Corrector_get_info"]

    async def call(self, tool: str, args: dict):
        self.calls.append(tool)
        if tool == "register_images":
            return self.server.register_images(**args)
        cmd = tool.split("_", 1)[1]
        if cmd == "get_stage":
            return self.scope.get_stage()
        if cmd == "move_stage":
            return self.scope.move_stage(args["position"])
        if cmd == "get_fov":
            return self.fov
        if cmd == "set_fov":
            self.fov = float(args["fov"])
            return None
        if cmd == "acquire_scanned_image":
            key = f"k{len(self.images)}"
            size = N if self.fov <= FOV * 1.01 else 2 * N
            self.images[key] = self.scope.acquire(self.fov, size)
            return key
        raise AssertionError(tool)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_save_calibrate_remount_return(real_langgraph, monkeypatch, tmp_path):
    wf = real_langgraph
    mcp = FakeMcp(monkeypatch, SyntheticMicroscope(seed=11, fov_m=FOV, imsize=N))
    inst = wf.McpInstrument(mcp.call, mcp.tools)
    assert inst.prefix == "Twin"
    book = wf.SiteBook(tmp_path / "sites.json")

    j = run(wf.calibrate(inst, book))
    assert np.allclose(j, mcp.scope.true_stage_to_image(), atol=0.2)
    mcp.scope.move_stage([4e-6, -3e-6, 0, 0, 0])
    site = run(wf.save_site(inst, book, "grain boundary"))
    mcp.scope.remount(0.0, (1.2e-6, -0.8e-6))                   # unload / reload, shifted
    mcp.scope.move_stage([0, 0, 0, 0, 0])

    out = run(wf.return_to_site(inst, book, "grain boundary"))
    assert out["status"] == "match", out["reasons"]
    assert mcp.scope.true_error_nm(site["stage"][:2]) < 3 * FOV / N * 1e9
    reopened = wf.SiteBook(tmp_path / "sites.json")               # a teammate reads the notes later
    visit = reopened.site("grain boundary")["visits"][-1]
    assert visit["status"] == "match" and visit["keys"]["native"] in mcp.images


def test_refuses_without_calibration_and_on_the_wrong_specimen(real_langgraph, monkeypatch, tmp_path):
    wf = real_langgraph
    mcp = FakeMcp(monkeypatch, SyntheticMicroscope(seed=11, fov_m=FOV, imsize=N))
    inst = wf.McpInstrument(mcp.call, mcp.tools)
    book = wf.SiteBook(tmp_path / "sites.json")
    run(wf.save_site(inst, book, "here"))
    out = run(wf.return_to_site(inst, book, "here"))
    assert out["status"] == "not_calibrated"                     # never centres on an unmeasured mapping

    run(wf.calibrate(inst, book))
    mcp.scope = SyntheticMicroscope(seed=999, fov_m=FOV, imsize=N)  # a different specimen in the holder
    out = run(wf.return_to_site(inst, book, "here"))
    assert out["status"] in ("no_match", "ambiguous")
    assert [v["status"] for v in wf.SiteBook(tmp_path / "sites.json").site("here")["visits"]] == \
        ["not_calibrated", out["status"]]
