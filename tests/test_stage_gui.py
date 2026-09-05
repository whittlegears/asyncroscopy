"""The stage GUI reaches the instrument only through MCP tools."""

import pytest

from startup_guis import stage


def test_resolve_instrument_tools_picks_class_exposing_get_stage():
    names = [
        "list_devices", "SCAN_get_dwell_time", "STAGE_get_x", "DigitalTwin_get_stage", "DigitalTwin_move_stage",
        "DigitalTwin_get_defocus", "DigitalTwin_set_defocus", "DigitalTwin_get_image_shift",
        "DigitalTwin_set_image_shift", "DigitalTwin_get_beam_tilt", "DigitalTwin_set_beam_tilt",
    ]
    tools = stage.resolve_instrument_tools(names)

    assert tools["get_stage"] == "DigitalTwin_get_stage"
    assert tools["set_beam_tilt"] == "DigitalTwin_set_beam_tilt"
    assert "auto_focus" not in tools


def test_resolve_instrument_tools_requires_a_single_instrument():
    with pytest.raises(RuntimeError, match="no \\*_get_stage"):
        stage.resolve_instrument_tools(["SCAN_get_dwell_time"])
    with pytest.raises(RuntimeError, match="Several instrument classes"):
        stage.resolve_instrument_tools(["DigitalTwin_get_stage", "AutoScriptMicroscope_get_stage"])
    assert stage.resolve_instrument_tools(
        ["DigitalTwin_get_stage", "AutoScriptMicroscope_get_stage"], instrument_class="AutoScriptMicroscope"
    ) == {"get_stage": "AutoScriptMicroscope_get_stage"}


def test_mcp_microscope_calls_named_tool_with_arg(monkeypatch):
    calls = []

    class FakeResult:
        def __init__(self, data):
            self.data = data
            self.structured_content = {"result": data}

    class FakeClient:
        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def list_tools(self):
            return [type("T", (), {"name": name})() for name in ("DigitalTwin_get_stage", "DigitalTwin_move_stage")]

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return FakeResult([1.0, 2.0, 3.0, 0.0, 0.0])

    monkeypatch.setattr(stage, "Client", FakeClient)
    microscope = stage.MCPMicroscope("http://bridge/mcp")
    microscope.connect()

    assert microscope.call("get_stage") == [1.0, 2.0, 3.0, 0.0, 0.0]
    microscope.call("move_stage", [0.0, 0.0, 0.0, 0.0, 0.0])
    assert calls == [
        ("DigitalTwin_get_stage", {}),
        ("DigitalTwin_move_stage", {"arg": [0.0, 0.0, 0.0, 0.0, 0.0]}),
    ]
    assert not microscope.has("auto_focus")
    with pytest.raises(RuntimeError, match="no 'auto_focus' tool"):
        microscope.call("auto_focus")


def test_default_bridge_is_local_not_the_real_instrument():
    assert "127.0.0.1" in stage.DEFAULT_MCP_URL or "localhost" in stage.DEFAULT_MCP_URL
