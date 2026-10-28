"""Approval mode: the Approvals device, the MCP policy gate, endpoint roles and the trace."""

import json

import pytest
import tango
from fastmcp.exceptions import ToolError

from asyncroscopy.contracts import validate
from asyncroscopy.mcp.mcp_server import MCPServer
from asyncroscopy.safety.policy import operator_only, risk_of


def cmd_info(in_type=tango.CmdArgType.DevString, out_type=tango.CmdArgType.DevString):
    return type("CommandInfo", (), {"in_type": in_type, "out_type": out_type, "in_type_desc": "", "out_type_desc": ""})()


@pytest.fixture
def server_factory(monkeypatch, approvals_proxy, tmp_path):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

    def make(role="agent", approvals=True):
        server = MCPServer(
            "test",
            "localhost",
            1234,
            blocked_classes=[],
            blocked_functions={},
            data_device_address="asyncroscopy/data/default",
            verbose=False,
            endpoint_role=role,
            approvals_device_address="asyncroscopy/approvals/default" if approvals else None,
            trace_path=str(tmp_path / f"trace_{role}.jsonl"),
        )
        server._approvals = lambda: approvals_proxy
        return server

    return make


@pytest.fixture(autouse=True)
def auto_mode_after(approvals_proxy):
    yield
    approvals_proxy.set_mode(json.dumps({"mode": "auto", "who": "test-teardown"}))


class TestPolicy:
    @pytest.mark.parametrize(
        "dev_class, command, level",
        [
            ("DigitalTwin", "move_stage", "high"),
            ("DigitalTwin", "acquire_scanned_image", "high"),
            ("Mapper", "start", "high"),
            ("Mapper", "abort", "low"),
            ("Mapper", "job_status", "low"),
            ("DigitalTwin", "get_stage", "low"),
            ("DigitalTwin", "blank_beam", "low"),
            ("DigitalTwin", "unblank_beam", "high"),
            ("SomeNewDevice", "do_something_new", "high"),
        ],
    )
    def test_default_risk(self, dev_class, command, level):
        assert risk_of(dev_class, command) == level

    def test_overrides_come_first(self):
        assert risk_of("DigitalTwin", "move_stage", {"DigitalTwin.move_stage": "low"}) == "low"

    def test_operator_only(self):
        assert operator_only("Approvals", "approve")
        assert operator_only("DigitalTwin", "export_ground_truth")
        assert not operator_only("Approvals", "request")


class TestApprovalsDevice:
    def test_default_mode_is_auto(self, approvals_proxy):
        assert approvals_proxy.get_mode() == "auto"

    def test_request_approve_check_is_single_use(self, approvals_proxy):
        call = {"action": "DigitalTwin_move_stage", "arguments": {"position": [1, 2, 3, 0, 0]}}
        request_id = json.loads(approvals_proxy.request(json.dumps({**call, "caller": "agent-1"})))["request_id"]
        assert json.loads(approvals_proxy.request(json.dumps(call)))["request_id"] == request_id
        assert not json.loads(approvals_proxy.check(json.dumps(call)))["allowed"]
        json.loads(approvals_proxy.approve(json.dumps({"request_id": request_id, "who": "operator"})))
        first = json.loads(approvals_proxy.check(json.dumps(call)))
        assert first["allowed"] and first["approved_by"] == "operator"
        assert not json.loads(approvals_proxy.check(json.dumps(call)))["allowed"]

    def test_approval_covers_only_the_same_arguments(self, approvals_proxy):
        call = {"action": "DigitalTwin_move_stage", "arguments": {"position": [9, 9, 0, 0, 0]}}
        request_id = json.loads(approvals_proxy.request(json.dumps(call)))["request_id"]
        approvals_proxy.approve(json.dumps({"request_id": request_id, "who": "operator"}))
        other = {**call, "arguments": {"position": [10, 9, 0, 0, 0]}}
        assert not json.loads(approvals_proxy.check(json.dumps(other)))["allowed"]

    def test_denied_request_reports_denied(self, approvals_proxy):
        request_id = json.loads(approvals_proxy.request(json.dumps({"action": "X_acquire", "arguments": {}})))["request_id"]
        approvals_proxy.deny(json.dumps({"request_id": request_id, "who": "operator", "reason": "not now"}))
        status = json.loads(approvals_proxy.request_status(request_id))
        assert status["state"] == "denied" and status["reason"] == "not now"

    def test_bad_mode_rejected(self, approvals_proxy):
        with pytest.raises(tango.DevFailed):
            approvals_proxy.set_mode(json.dumps({"mode": "yolo"}))


class TestGate:
    def test_auto_mode_runs_high_risk_and_traces_auto(self, server_factory, tmp_path):
        server = server_factory()
        wrapper = server._create_wrapper(lambda arg: f"moved {arg}", cmd_info(), "move_stage", "DigitalTwin")
        assert wrapper("here") == "moved here"
        event = json.loads((tmp_path / "trace_agent.jsonl").read_text().splitlines()[-1])
        validate(event, "trace_event")
        assert event["approved_by"] == "auto" and event["mode"] == "auto"

    def test_approve_mode_holds_until_operator_approves(self, server_factory, approvals_proxy, tmp_path):
        server = server_factory()
        calls = []
        wrapper = server._create_wrapper(lambda arg: calls.append(arg) or "ok", cmd_info(), "move_stage", "DigitalTwin")
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        with pytest.raises(ToolError, match="APPROVAL REQUIRED") as held:
            wrapper("target-1")
        assert calls == []
        request_id = str(held.value).split("request_id=")[1].split(")")[0]
        assert server.approval_status(request_id)["state"] == "pending"
        approvals_proxy.approve(json.dumps({"request_id": request_id, "who": "alice"}))
        assert wrapper("target-1") == "ok"
        assert calls == ["target-1"]
        last = json.loads((tmp_path / "trace_agent.jsonl").read_text().splitlines()[-1])
        assert last["approved_by"] == "alice" and last["grant_id"]
        with pytest.raises(ToolError, match="APPROVAL REQUIRED"):
            wrapper("target-1")  # the grant was single use

    def test_low_risk_never_held(self, server_factory, approvals_proxy):
        server = server_factory()
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        wrapper = server._create_wrapper(lambda arg: "stage", cmd_info(), "get_stage_json", "DigitalTwin")
        assert wrapper("x") == "stage"

    def test_operator_endpoint_is_never_held(self, server_factory, approvals_proxy):
        server = server_factory(role="operator")
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        wrapper = server._create_wrapper(lambda arg: "ok", cmd_info(), "move_stage", "DigitalTwin")
        assert wrapper("x") == "ok"

    def test_unreachable_approvals_fails_closed_for_high_risk(self, server_factory):
        server = server_factory()

        def broken():
            raise tango.DevFailed()

        server._approvals = broken
        high = server._create_wrapper(lambda arg: "ran", cmd_info(), "move_stage", "DigitalTwin")
        low = server._create_wrapper(lambda arg: "ran", cmd_info(), "get_thing", "DigitalTwin")
        with pytest.raises(ToolError, match="unreachable"):
            high("x")
        assert low("x") == "ran"

    def test_no_approvals_device_means_no_gate(self, server_factory):
        server = server_factory(approvals=False)
        wrapper = server._create_wrapper(lambda arg: "ran", cmd_info(), "move_stage", "DigitalTwin")
        assert wrapper("x") == "ran"


class TestEndpointRoles:
    class FakeDb:
        class Exported:
            value_string = ["asyncroscopy/approvals/default", "asyncroscopy/instrument/default"]

        def get_device_exported(self, pattern):
            return self.Exported()

    class FakeProxy:
        def __init__(self, name):
            self.name = name

        def set_timeout_millis(self, ms):
            pass

        def info(self):
            return type("Info", (), {"dev_class": "Approvals" if "approvals" in self.name else "DigitalTwin"})()

        def command_list_query(self):
            names = ["approve", "deny", "set_mode", "request", "get_mode"] if "approvals" in self.name else ["move_stage", "export_ground_truth"]
            return [type("Cmd", (), {"cmd_name": n})() for n in names]

        def __getattr__(self, item):
            return lambda *a: None

    @pytest.mark.parametrize("role, hidden", [("agent", True), ("operator", False)])
    def test_operator_only_commands(self, monkeypatch, role, hidden):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: self.FakeDb())
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", self.FakeProxy)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False, endpoint_role=role)
        tools = server._find_tools()
        approvals = set(tools["Approvals"])
        twin = set(tools["DigitalTwin"])
        assert ("approve" not in approvals) is hidden
        assert ("set_mode" not in approvals) is hidden
        assert ("export_ground_truth" not in twin) is hidden
        assert {"request", "get_mode"} <= approvals and "move_stage" in twin

    def test_unknown_role_rejected(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        with pytest.raises(ValueError, match="endpoint_role"):
            MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", endpoint_role="admin")
