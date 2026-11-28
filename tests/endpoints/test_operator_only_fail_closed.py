"""Operator-only commands stay off agent endpoints when a device's action_hints cannot be read."""

import tango

from asyncroscopy.mcp import mcp_server as mcp_server_module
from asyncroscopy.mcp.descriptors import read_hints
from asyncroscopy.mcp.mcp_server import MCPServer

AUTOSCRIPT = "asyncroscopy/autoscriptmicroscope/default"


class _Exported:
    def __init__(self, names: list[str]) -> None:
        self.value_string = names

    def get_device_exported(self, pattern: str) -> "_Exported":
        return self


class BusyProxy:
    def get_attribute_list(self):
        return ["State", "Status", "action_hints"]

    def read_attribute(self, name):
        raise TimeoutError("API_DeviceTimedOut: device busy (3000 ms)")


def _server(role: str, names: list[str] | None = None) -> MCPServer:
    return MCPServer("t", "localhost", 0, needs_approval={}, blocked_classes=[], data_device_address="",
                     endpoint_role=role, database=_Exported(names or []), verbose=False)


def test_unreadable_hints_offer_agents_only_state_and_status():
    hints, problem = read_hints(BusyProxy())
    assert hints is None and "could not read" in problem
    agent = _server("agent")
    for command in ("set_beam_stopper", "run_stem_auto_focus", "move_relative", "export_ground_truth", "get_parameters"):
        assert not agent._offers_command("AutoScriptMicroscope", "AutoScriptMicroscope", command, hints)
    assert agent._offers_command("AutoScriptMicroscope", "AutoScriptMicroscope", "State", hints)
    assert not _server("monitor")._offers_command("X", "X", "get_parameters", hints)
    assert _server("operator")._offers_command("AutoScriptMicroscope", "AutoScriptMicroscope", "set_beam_stopper", hints)


def test_operator_only_commands_carry_the_expert_display_level(auto_script_proxy):
    levels = {c.cmd_name: c.disp_level for c in auto_script_proxy.command_list_query()}
    for command in ("set_beam_stopper", "run_stem_auto_focus", "set_projector_mode", "acquire_camera_series"):
        assert levels[command] == tango.DispLevel.EXPERT, command
    assert levels["get_parameters"] == tango.DispLevel.OPERATOR


def test_expert_commands_stay_off_agents_even_when_the_hints_say_nothing(auto_script_proxy):
    levels = {c.cmd_name: c.disp_level for c in auto_script_proxy.command_list_query()}
    agent = _server("agent")
    assert not agent._offers_command("AutoScriptMicroscope", "AutoScriptMicroscope", "set_beam_stopper", {}, levels["set_beam_stopper"])
    assert _server("operator")._offers_command("AutoScriptMicroscope", "AutoScriptMicroscope", "set_beam_stopper", {}, levels["set_beam_stopper"])


def test_discovery_holds_a_device_whose_hints_cannot_be_read(tango_ctx, monkeypatch):
    monkeypatch.setattr(mcp_server_module, "read_hints", lambda proxy: (None, "could not read its action_hints (TimeoutError: busy)"))
    agent = _server("agent", [tango_ctx.get_device_access(AUTOSCRIPT)])
    agent.setup(print_summary=False)
    offered = {name for name, t in agent.descriptors.items() if not t.native}
    assert offered == {"AutoScriptMicroscope_State", "AutoScriptMicroscope_Status"}
    info = agent.discovery.endpoint_info()
    assert info["devices_without_hints"] == ["AutoScriptMicroscope"]
    assert any("could not read" in w and "only its State and Status" in w for w in info["discovery_warnings"])
