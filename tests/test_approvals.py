"""Held instrument calls: an agent's call waits, an operator approves or denies it on another endpoint."""

from __future__ import annotations

import asyncio

import pytest
import tango

from asyncroscopy.mcp.approvals import ApprovalStore, needs_approval, parse_held
from asyncroscopy.mcp.mcp_server import MCPServer


def _info(in_type=tango.CmdArgType.DevString, desc=":param position: where to"):
    return type("CommandInfo", (), {"in_type": in_type, "out_type": tango.CmdArgType.DevString, "in_type_desc": desc, "out_type_desc": ""})


def _server(monkeypatch, db, role, rules=None):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    return MCPServer(
        "t", "localhost", 1234, blocked_functions={}, blocked_classes=[], data_device_address="d", verbose=False,
        needs_approval=rules if rules is not None else {"*": ["move_*"], "Scope": ["correct_aberration"]},
        approvals_db=str(db), role=role,
    )


def test_patterns_match_by_class_wildcard_and_dotted_name():
    rules = {"*": ["move_*", "Corrector.apply"], "Scope": ["unblank_beam"]}
    assert needs_approval(rules, "AnyClass", "move_stage")
    assert needs_approval(rules, "Scope", "unblank_beam")
    assert not needs_approval(rules, "Other", "unblank_beam")
    assert needs_approval(rules, "Corrector", "apply")
    assert not needs_approval(rules, "Scope", "get_stage")


def test_agent_call_is_held_then_approved_by_the_operator(monkeypatch, tmp_path):
    moved = []

    def move_stage(position):
        moved.append(position)
        return f"moved to {position}"

    agent = _server(monkeypatch, tmp_path / "held.sqlite", "agent")
    operator = _server(monkeypatch, tmp_path / "held.sqlite", "operator")
    agent_move = agent._create_wrapper(move_stage, _info(), "move_stage", "Scope")
    operator._create_wrapper(move_stage, _info(), "move_stage", "Scope")

    with pytest.raises(RuntimeError) as held:
        agent_move("x=1")
    call_id = parse_held(str(held.value))
    assert call_id and moved == []  # nothing ran
    assert agent.call_status(call_id)["status"] == "pending"
    assert [c["id"] for c in operator.list_pending()] == [call_id]

    done = operator.approve_call(call_id, note="looks safe", actor="human:operator")
    assert done["status"] == "done" and done["result"] == "moved to x=1"
    assert moved == ["x=1"]
    assert agent.call_status(call_id)["decided_by"] == "human:operator"
    with pytest.raises(ValueError, match="not pending"):
        operator.approve_call(call_id)


def test_denied_call_never_runs(monkeypatch, tmp_path):
    ran = []
    agent = _server(monkeypatch, tmp_path / "held.sqlite", "agent")
    operator = _server(monkeypatch, tmp_path / "held.sqlite", "operator")
    void = _info(tango.CmdArgType.DevVoid, "")
    wrapper = agent._create_wrapper(lambda: ran.append(1), void, "correct_aberration", "Scope")
    operator._create_wrapper(lambda: ran.append(1), void, "correct_aberration", "Scope")
    with pytest.raises(RuntimeError) as held:
        wrapper()
    call_id = parse_held(str(held.value))
    assert operator.deny_call(call_id, note="tableau is stale")["status"] == "denied"
    assert agent.call_status(call_id)["note"] == "tableau is stale"
    assert ran == []


def test_unlisted_commands_and_the_operator_run_directly(monkeypatch, tmp_path):
    agent = _server(monkeypatch, tmp_path / "held.sqlite", "agent")
    operator = _server(monkeypatch, tmp_path / "held.sqlite", "operator")
    assert agent._create_wrapper(lambda p: "[0, 0]", _info(), "get_stage", "Scope")("all") == "[0, 0]"
    assert operator._create_wrapper(lambda p: "moved", _info(), "move_stage", "Scope")("x") == "moved"


def test_operator_tools_are_only_on_the_operator_endpoint(monkeypatch, tmp_path):
    names = {}
    for role in ("agent", "operator"):
        server = _server(monkeypatch, tmp_path / "held.sqlite", role)
        registered = []
        monkeypatch.setattr(server.mcp, "add_tool", lambda tool, registered=registered: registered.append(getattr(tool, "__name__", getattr(tool, "name", ""))))
        monkeypatch.setattr(server, "_find_tools", lambda: {})
        server.setup(print_summary=False)
        names[role] = set(registered)
    assert "call_status" in names["agent"] and "approve_call" not in names["agent"]
    assert {"list_pending", "approve_call", "deny_call", "call_status"} <= names["operator"]


def test_rules_without_a_store_are_refused(monkeypatch):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    with pytest.raises(ValueError, match="approvals_db"):
        MCPServer("t", "h", 1, blocked_functions={}, blocked_classes=[], data_device_address="d", needs_approval={"*": ["move_*"]})


def test_store_survives_reopening(tmp_path):
    first = ApprovalStore(tmp_path / "held.sqlite")
    call_id = first.hold("Scope_move_stage", {"position": [0, 0, 0, 0, 0]})
    again = ApprovalStore(tmp_path / "held.sqlite")
    assert again.get(call_id)["arguments"] == {"position": [0, 0, 0, 0, 0]}


def test_toolbox_waits_for_an_approved_call():
    pytest.importorskip("langgraph.types")
    from asyncroscopy.agent_workflows.tools import ToolBox, ToolCallError

    answers = iter([{"status": "pending"}, {"status": "approved"}, {"status": "done", "result": "moved"}])

    async def Scope_move_stage(position):
        raise RuntimeError("HELD abc123: Scope_move_stage needs an operator's approval before it runs.")

    async def call_status(call_id):
        assert call_id == "abc123"
        return next(answers)

    box = ToolBox({"Scope_move_stage": Scope_move_stage, "call_status": call_status}, poll_s=0.0)
    assert asyncio.run(box.call("move_stage", position=[0] * 5)) == "moved"

    async def denied(call_id):
        return {"status": "denied", "note": "not now"}

    box = ToolBox({"Scope_move_stage": Scope_move_stage, "call_status": denied}, poll_s=0.0)
    with pytest.raises(ToolCallError, match="denied: not now"):
        asyncio.run(box.call("move_stage", position=[0] * 5))


def test_mcp_config_checks_the_approval_settings(tmp_path):
    import yaml

    from startup_scripts import run_mcp
    from startup_scripts.config_schema import ConfigError

    raw = yaml.safe_load((run_mcp.DEFAULT_CONFIG_PATH).read_text())
    raw["mcp"]["needs_approval"] = {"*": ["move_stage"]}
    path = tmp_path / "mcp.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="approvals_db"):
        run_mcp.load_config(path)
    raw["mcp"]["approvals_db"] = str(tmp_path / "held.sqlite")
    raw["mcp"]["role"] = "operator"
    path.write_text(yaml.safe_dump(raw))
    config = run_mcp.load_config(path)
    command = run_mcp.build_command(config)
    assert "--needs-approval-json" in command and "--approvals-db" in command
    assert command[command.index("--role") + 1] == "operator"
