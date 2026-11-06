"""Devices declare what their commands are; the name rules decide only what a device leaves out."""

import importlib
from pathlib import Path

import pytest
import yaml

from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.mcp.descriptors import DeviceInfo, describe_command
from asyncroscopy.mcp.mcp_server import MCPServer
from asyncroscopy.safety.approvals import Approvals
from asyncroscopy.utils.commands import declared_hints
from startup_scripts import run_mcp

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
EXAMPLES = ROOT / "asyncroscopy" / "instruments" / "yaml_sim" / "examples"


def described(device, command, overrides=None):
    return describe_command(device, command, overrides=overrides or {}, routine=None, warnings=[])


def test_the_approvals_device_declares_its_commands():
    hints = declared_hints(Approvals)
    assert all(hints[c]["operator_only"] for c in ("approve", "deny", "set_mode", "set_auto_allowed", "check", "request", "revoke_grants"))
    assert hints["get_mode"]["risk"] == "low" and hints["request_status"]["risk"] == "low"


def test_a_declaration_beats_the_name_rules_and_an_override_beats_both():
    stage = DeviceInfo("Stage", "Stage", hints={"park": {"risk": "low", "stop": True, "role": "safe_stop"}, "get_tip": {"risk": "high"}})
    park = described(stage, "park")
    assert park.risk == "low" and park.risk_source == "declared" and park.stop and park.action_role == "safe_stop"
    assert described(stage, "get_tip").risk == "high"  # a get_* that acts, declared so
    assert described(stage, "move").risk_source == "rule" and described(stage, "move").risk == "high"  # nothing declared: unknown is high
    held = described(stage, "park", overrides={"Stage.park": "high"})
    assert held.risk == "high" and held.risk_source == "override"


def test_declared_operator_only_and_routine():
    robot = DeviceInfo("Robot", "Robot", hints={"calibrate": {"operator_only": True}, "pick": {"routine": True}})
    assert described(robot, "calibrate").operator_only
    pick = described(robot, "pick")
    assert pick.routine and pick.risk == "high"
    assert not described(robot, "pick", overrides={"Robot.pick": "high"}).routine  # a deployment that marks it high holds it


def test_a_bad_declared_risk_falls_back_to_the_rules_with_a_warning():
    warnings: list[str] = []
    tool = describe_command(DeviceInfo("X", "X", hints={"go": {"risk": "medium"}}), "go", overrides={}, routine=None, warnings=warnings)
    assert tool.risk == "high" and tool.risk_source == "rule" and "not low or high" in warnings[0]


# The twin stack (configs/digital_twin_mapping.yaml) as tests/conftest.py starts it: one device of each class.
TWIN_STACK = {
    "DigitalTwin": "asyncroscopy/digitaltwin/default",
    "CAMERA": "asyncroscopy/camera/default",
    "DATA": "asyncroscopy/data/default",
    "EDS": "asyncroscopy/eds/default",
    "SCAN": "asyncroscopy/scan/default",
    "Mapper": "asyncroscopy/mapper/default",
    "Tuner": "asyncroscopy/tuner/default",
    "Approvals": "asyncroscopy/approvals/default",
    "DigitalTwinSTAGE": "asyncroscopy/stage/drift",
    "DigitalTwinCorrector": "asyncroscopy/corrector/default",
}


def _stack_classes(path: Path) -> set[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    found = {raw["instrument"]["class_name"]}
    for name, device in raw["devices"].items():
        module = importlib.import_module(device["module_name"])
        found.add(device.get("class_name") or next(
            v.__name__ for v in vars(module).values() if isinstance(v, type) and v.__module__ == module.__name__ and v.__name__.lower() == name.lower()))
    return found


class _Exported:
    def __init__(self, names: list[str]) -> None:
        self.value_string = names

    def get_device_exported(self, pattern: str) -> "_Exported":
        return self


@pytest.mark.parametrize("mcp_config", ["mcp_dt.yaml", "mcp_operator.yaml"])
def test_the_twin_stack_declares_every_command(tango_ctx, mcp_config):
    """No command of the twin stack falls back to the name rules, on the agent or the operator endpoint."""
    assert _stack_classes(CONFIGS / "digital_twin_mapping.yaml") == set(TWIN_STACK)
    config = run_mcp.load_config(CONFIGS / mcp_config).mcp
    server = MCPServer(
        "twin", "localhost", 0, blocked_functions=config.blocked_functions, blocked_classes=config.blocked_classes,
        data_device_address=config.data_device_address, verbose=False, endpoint_role=config.endpoint_role,
        approvals_device_address=config.approvals_device_address or None, risk_overrides=config.risk,
        auto_routine=config.auto_routine, database=_Exported(list(TWIN_STACK.values())),
    )
    server.setup(print_summary=False)
    assert server.risk_fallbacks == {}
    devices = [t for t in server.descriptors.values() if not t.native]
    assert {t.dev_class for t in devices} >= {"DigitalTwin", "Mapper", "Tuner", "DATA", "DigitalTwinSTAGE", "DigitalTwinCorrector"}
    assert all(t.risk_source in ("declared", "tango", "override") for t in devices)
    tools = server.descriptors
    # The device declares what used to be guessed: routine imaging, the stops, reads, and its arguments' ranges.
    assert tools["DigitalTwin_move_stage"].routine and tools["DigitalTwin_move_stage"].risk == "high"
    assert tools["DigitalTwin_blank_beam"].stop and tools["DigitalTwin_blank_beam"].risk == "low"
    assert tools["Mapper_abort_all"].stop and not tools["Mapper_job_status"].acts
    assert not tools["DigitalTwin_set_column_valves"].routine
    assert server.devices["DigitalTwin"].hints["set_fov"]["args"][0]["max"] > 0
    if config.endpoint_role == "operator":
        assert tools["Approvals_approve"].operator_only and tools["DigitalTwin_export_ground_truth"].operator_only
    else:
        assert "Approvals_approve" not in tools and "DigitalTwin_export_ground_truth" not in tools


def test_a_config_auto_routine_replaces_the_declarations():
    declared = DeviceInfo("DigitalTwin", "DigitalTwin", stem=True, hints=declared_hints(DigitalTwin))
    assert describe_command(declared, "move_stage", overrides={}, routine=None, warnings=[]).routine
    assert not describe_command(declared, "move_stage", overrides={}, routine={"tools": ["DigitalTwin_place_beam"]}, warnings=[]).routine
    assert describe_command(declared, "place_beam", overrides={}, routine={"tools": ["DigitalTwin_place_beam"]}, warnings=[]).routine


def test_tango_builtins_need_no_declaration():
    bare = DeviceInfo("Thing", "Thing")
    state, init = described(bare, "State"), described(bare, "Init")
    assert state.risk == "low" and not state.acts and state.risk_source == "tango"
    assert init.risk == "high" and init.operator_only and init.risk_source == "tango"


def test_vendors_and_twins_inherit_the_base_declarations():
    from asyncroscopy.instruments.electron_microscope.auto_script import AutoScriptMicroscope
    from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope
    from asyncroscopy.instruments.electron_microscope.jeol import JeolMicroscope

    base = declared_hints(ElectronMicroscope)
    for vendor in (AutoScriptMicroscope, JeolMicroscope, DigitalTwin):
        hints = declared_hints(vendor)
        assert all(hints[c] == base[c] for c in base), vendor


def test_a_yaml_instrument_declares_from_its_file():
    from asyncroscopy.instruments.yaml_sim.spec import action_hints, load_spec

    hints = action_hints(load_spec(EXAMPLES / "afm.yaml"))
    assert hints["retract_tip"]["risk"] == "low" and hints["retract_tip"]["stop"]
    assert hints["approach_tip"]["risk"] == "high" and hints["move_xy"]["role"] == "stage.move"
    assert hints["get_status"]["risk"] == "low" and hints["set_setpoint_v"]["risk"] == "high"
    assert all(h.get("risk") in ("low", "high") for h in hints.values())
