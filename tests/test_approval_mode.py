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
            ("Mapper", "job_status", "high"),  # envelope verbs are low only on envelope devices
            ("DigitalTwin", "get_stage", "low"),
            ("DigitalTwin", "blank_beam", "low"),
            ("DigitalTwin", "unblank_beam", "high"),
            ("SomeNewDevice", "do_something_new", "high"),
        ],
    )
    def test_default_risk(self, dev_class, command, level):
        assert risk_of(dev_class, command) == level

    def test_envelope_verbs_low_only_on_envelope_devices(self):
        assert risk_of("Mapper", "job_status", envelope=True) == "low"
        assert risk_of("Mapper", "release", envelope=True) == "low"
        assert risk_of("Mapper", "start", envelope=True) == "high"
        assert risk_of("Camera", "release") == "high"

    def test_safe_stop_verbs_always_low(self):
        assert risk_of("AnyStage", "stop") == "low"
        assert risk_of("Mapper", "abort") == "low"

    @pytest.mark.parametrize("name", [
        "interlock_off", "turn_off_safety", "switch_off_interlock", "emergency_unlock", "withdraw_and_move",
        "stop_and_approach", "move_to_stop", "drop_off_sample", "set_stop_height", "pause_then_resume", "limits_off", "feedback_off",
        "off_axis_tilt", "goto_stop", "jog_to_halt", "stop_at_position", "retract_to", "withdraw_volume", "vacuum_off", "pump_off",
        "cooling_off", "LN2_off", "stop_cooling", "stop_pumping", "pause_ventilation", "go_to_safe_state",
        "chiller_off", "z_piezo_off", "stop_return_home", "off_axis_illumination", "approach_stop",
        # Found by the final review: hardware holds, harmful subsystem stops, feedback loops,
        # motion until a stop, re-engage after a retreat, stopping a safety action, re-arming.
        "brake_off", "BrakeOff", "servo_off", "motor_off", "gripper_off", "clamp_off", "magnet_off", "turbopump_off",
        "ionpump_off", "cryocooler_off", "helium_off", "stop_flow", "stop_pid", "zfeedback_off", "z_loop_off",
        "seek_hard_stop", "descend_until_stop", "find_stop", "retract_approach", "withdraw_engage", "retract_tip_down",
        "stop_retract", "abort_withdraw", "stop_restart", "abort_retry", "estop_ack", "acknowledge_estop",
        "rearm_after_stop", "continue_after_pause", "kick_off", "lift_off", "back_off",
    ])
    def test_names_that_do_more_than_stop_are_high(self, name):
        assert risk_of("AnyDevice", name) == "high"

    @pytest.mark.parametrize("name", ["stop_scan", "AbortScan", "shutter_close", "blank_beam", "retract", "emergency_stop", "beam_off", "safe_state", "lamp_off", "halt", "abort_approach", "stop_move", "StopScan",
                                      "halt_xy", "stop_xyz", "stop_motors", "abort_autofocus", "stop_live", "disengage", "led_off", "illumination_off",
                                      "stop_indent", "abort_indent", "retract_indenter", "retract_head", "stop_imaging", "stop_exposure",
                                      "retract_objective", "xray_off", "stop_robot", "abort_protocol", "pause_protocol", "source_off",
                                      "stop_xray", "disengage_tip", "DisengageProbe"])
    def test_plain_stops_are_low(self, name):
        assert risk_of("AnyDevice", name) == "low"

    def test_stop_like_names_left_high_are_spotted(self):
        from asyncroscopy.safety.policy import looks_like_stop
        assert looks_like_stop("park_stage") and looks_like_stop("unload") and looks_like_stop("brake_off") and looks_like_stop("ParkStage")
        assert not looks_like_stop("move_stage") and not looks_like_stop("get_offset")

    @pytest.mark.parametrize("name", ["indent_retract", "IndentRetract", "indent_withdraw", "indent_stop", "exposure_stop", "imaging_pause", "protocol_halt",
                                      "Off", "off", "disengage_motor", "disengage_motors", "disengage_z", "disengage_stage", "DisengageAll"])
    def test_an_action_before_the_stop_verb_is_high(self, name):
        assert risk_of("AnyDevice", name) == "high"

    def test_place_beam_is_high_even_on_a_stem(self):
        assert risk_of("DigitalTwin", "place_beam", stem=True) == "high"

    def test_stem_beam_settings_are_low_only_on_a_stem(self):
        assert risk_of("OpticalScope", "set_defocus") == "high"
        assert risk_of("DigitalTwin", "set_defocus", stem=True) == "low"
        assert risk_of("DigitalTwin", "move_stage", stem=True) == "high"

    def test_overrides_come_first(self):
        assert risk_of("DigitalTwin", "move_stage", {"DigitalTwin.move_stage": "low"}) == "low"

    def test_operator_only(self):
        assert operator_only("Approvals", "approve")
        assert operator_only("DigitalTwin", "export_ground_truth")
        assert operator_only("Approvals", "request")
        assert operator_only("Approvals", "check")
        assert not operator_only("Approvals", "get_mode")


def test_a_configured_approve_beats_a_saved_auto():
    from asyncroscopy.safety.approvals import starting_mode
    assert starting_mode("approve", "auto") == "approve"
    assert starting_mode("auto", "approve") == "approve"
    assert starting_mode("auto", "auto") == "auto"


class TestApprovalsDevice:
    def test_default_mode_is_auto(self, approvals_proxy):
        assert approvals_proxy.get_mode() == "auto"

    def test_mode_survives_a_device_restart(self, approvals_proxy, job_dir):
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        try:
            assert json.loads((job_dir / "approvals_mode.json").read_text())["mode"] == "approve"
            approvals_proxy.Init()  # what a crash and restart, or an Init, would do
            assert approvals_proxy.get_mode() == "approve"
        finally:
            approvals_proxy.set_mode(json.dumps({"mode": "auto", "who": "operator"}))
        approvals_proxy.Init()
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
        assert ("request" not in approvals) is hidden
        assert "get_mode" in approvals and "move_stage" in twin

    def test_unknown_role_rejected(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        with pytest.raises(ValueError, match="endpoint_role"):
            MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", endpoint_role="admin")


class TestJsonArguments:
    """Envelope and approval commands take a JSON string; models pass objects. All forms must work."""

    @staticmethod
    def make(monkeypatch, seen):
        import asyncio

        from fastmcp import Client, FastMCP
        from fastmcp.tools import Tool

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)
        info = type("CommandInfo", (), {"in_type": tango.CmdArgType.DevString, "out_type": tango.CmdArgType.DevString, "in_type_desc": ':param request: JSON {"plan": {...}, "caller": "name"}.', "out_type_desc": ""})()
        wrapper = server._create_wrapper(lambda text: seen.append(json.loads(text)) or "ok", info, "reserve", "Mapper")
        app = FastMCP("t")
        app.add_tool(Tool.from_function(wrapper))

        def call(args):
            async def go():
                async with Client(app) as client:
                    return await client.call_tool("Mapper_reserve", args)
            return asyncio.run(go())

        return wrapper, call

    def test_object_string_and_flattened_fields(self, monkeypatch):
        seen = []
        wrapper, call = self.make(monkeypatch, seen)
        wrapper(request={"plan": {"fov_m": 1}})
        wrapper(request='{"plan": {"fov_m": 2}}')
        wrapper(plan={"fov_m": 3})
        call({"request": {"plan": {"fov_m": 4}}})
        call({"request": '{"plan": {"fov_m": 5}}'})
        call({"plan": {"fov_m": 6}, "caller": "x"})
        assert [s["plan"]["fov_m"] for s in seen] == [1, 2, 3, 4, 5, 6]
        assert seen[-1]["caller"] == "x"


class TestDevicePrefixes:
    """Two devices of one class each get their own tools; neither hides the other."""

    @staticmethod
    def server_for(monkeypatch, names):
        class Exported:
            value_string = names

        class FakeDb:
            def get_device_exported(self, pattern):
                return Exported()

        class FakeProxy:
            def __init__(self, name):
                self.name = name

            def set_timeout_millis(self, ms):
                pass

            def info(self):
                return type("Info", (), {"dev_class": "Mapper"})()

            def command_list_query(self):
                return [type("Cmd", (), {"cmd_name": n})() for n in ("reserve", "start", "job_status", "abort", "results", "release")]

            def __getattr__(self, item):
                return lambda *a, _name=self.name: _name

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: FakeDb())
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", FakeProxy)
        return MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)

    def test_same_family_devices_get_member_prefixes(self, monkeypatch):
        tools = self.server_for(monkeypatch, ["lab/mapper/1", "lab/mapper/2"])._find_tools()
        assert {prefix: commands["start"][0]() for prefix, commands in tools.items()} == {"Mapper_1": "lab/mapper/1", "Mapper_2": "lab/mapper/2"}

    def test_repeated_default_members_are_named_by_their_config_key(self, monkeypatch):
        tools = self.server_for(monkeypatch, ["asyncroscopy/grid1/default", "asyncroscopy/grid2/default"])._find_tools()
        assert set(tools) == {"Mapper_grid1", "Mapper_grid2"}

    def test_repeated_members_fall_back_to_the_device_name(self, monkeypatch):
        tools = self.server_for(monkeypatch, ["east/mapper/default", "west/mapper/default"])._find_tools()
        assert set(tools) == {"Mapper_east_mapper_default", "Mapper_west_mapper_default"}


class TestGateDetails:
    @staticmethod
    def server(monkeypatch, approvals_proxy):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False, approvals_device_address="asyncroscopy/approvals/default")
        monkeypatch.setattr(server, "_approvals", lambda: approvals_proxy)
        return server

    def test_a_held_start_shows_the_operator_what_it_will_do(self, monkeypatch, approvals_proxy):
        server = self.server(monkeypatch, approvals_proxy)
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "test"}))
        try:
            with pytest.raises(ToolError):
                server._gate("Mapper_start", "high", {"request": '{"reservation_id":"r1"}'}, "agent", describe=lambda: 'Reserved plan: {"size_m": [6e-7, 6e-7]}')
            pending = [p for p in json.loads(approvals_proxy.list_pending()) if p["action"] == "Mapper_start"]
            assert pending and "size_m" in pending[-1]["reason"]
            for p in pending:
                approvals_proxy.deny(json.dumps({"request_id": p["request_id"], "who": "test"}))
        finally:
            approvals_proxy.set_mode(json.dumps({"mode": "auto", "who": "test"}))

    def test_an_agent_cannot_name_its_own_grant(self):
        body = json.loads(MCPServer._with_grant('{"reservation_id": "r1", "grant_id": "forged"}', {"approved_by": "auto"}, "agent"))
        assert body.get("grant_id") != "forged"


class TestWaitForJob:
    def test_waits_until_the_job_finishes_and_refuses_unknown_devices(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)

        class Job:
            polls = 0

            def job_status(self, request):
                Job.polls += 1
                return json.dumps({"job_id": "j1", "state": "done" if Job.polls >= 2 else "running"})

        server._devices = {"Mapper": Job()}
        server._envelope_prefixes = {"Mapper"}
        assert server.wait_for_job("Mapper", "j1", max_wait_s=10)["state"] == "done"
        assert Job.polls == 2
        with pytest.raises(ToolError, match="Mapper"):
            server.wait_for_job("Camera", "j1")

    def test_an_analysis_job_id_is_pointed_at_the_analysis_server(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)

        class Job:
            def job_status(self, request):
                return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})

        server._devices = {"Mapper": Job()}
        server._envelope_prefixes = {"Mapper"}
        with pytest.raises(ToolError, match="analysis_job_status"):
            server.wait_for_job("Mapper", "assemble-job-1")
