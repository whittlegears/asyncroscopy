"""Devices declare what their commands are; the name rules decide only what a device leaves out."""

from asyncroscopy.mcp.descriptors import DeviceInfo, describe_command
from asyncroscopy.safety.approvals import Approvals
from asyncroscopy.utils.commands import declared_hints


def described(device, command, overrides=None):
    return describe_command(device, command, overrides=overrides or {}, routine={}, warnings=[])


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
    tool = describe_command(DeviceInfo("X", "X", hints={"go": {"risk": "medium"}}), "go", overrides={}, routine={}, warnings=warnings)
    assert tool.risk == "high" and tool.risk_source == "rule" and "not low or high" in warnings[0]
