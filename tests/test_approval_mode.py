"""Approval mode: the Approvals device, the MCP policy gate, endpoint roles and the trace."""

import json
import time

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
                                      "halt_xy", "stop_xyz", "stop_motors", "abort_autofocus", "stop_live", "withdraw_tip", "led_off", "illumination_off",
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
                                      "Off", "off", "disengage_motor", "disengage_motors", "disengage_z", "disengage_stage", "DisengageAll",
                                      "Disengage", "emergency_disengage", "retract_shutter", "withdraw_shutter", "ShutterRetract", "retract_x",
                                      "withdraw_y", "retract_xy", "retract_stage", "StageRetract", "retract_robot", "withdraw_motors",
                                      "retract_all", "retract_source", "retract_laser", "retract_gun", "retract_beam",
                                      "BeamStopMove", "beam_stop_move", "BeamStopX", "beam_stop_y", "BeamStop", "BeamStopScan"])
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

    def test_overrides_that_disagree_never_lower_a_command(self):
        assert risk_of("DigitalTwin", "move_stage", {"*": "low", "DigitalTwin.move_stage": "high"}) == "high"
        assert risk_of("DigitalTwin", "move_stage", {"DigitalTwin.move_stage": "high", "*": "low"}) == "high"
        assert risk_of("DigitalTwin", "get_stage", {"*": "low"}) == "low"

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

            def results(self, request):
                return json.dumps({"schema_version": "1.1.0", "job_id": "j1", "state": "done", "result_keys": {"run": "map_r1.h5"}, "summary": {"grid": {"rows": 3}}})

        server._devices = {"Mapper": Job()}
        server._envelope_prefixes = {"Mapper"}
        done = server.wait_for_job("j1", device="Mapper", max_wait_s=10)
        assert done["state"] == "done" and Job.polls == 2
        # A finished job's reply carries its results, so <device>_results is not a separate call.
        assert done["result_keys"] == {"run": "map_r1.h5"} and done["summary"]["grid"]["rows"] == 3
        with pytest.raises(ToolError, match="Mapper"):
            server.wait_for_job("j1", device="Camera")

    def test_the_device_can_be_left_out(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)

        class Other:
            def job_status(self, request):
                return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})

        class Mapper:
            def job_status(self, request):
                return json.dumps({"job_id": "j1", "state": "failed", "last_error": "stage refused"})

        server._devices = {"IndentGrid": Other(), "Mapper": Mapper()}
        server._envelope_prefixes = {"IndentGrid", "Mapper"}
        assert server.wait_for_job("j1")["last_error"] == "stage refused"

    def test_analysis_jobs_are_waited_on_through_the_analysis_server(self, monkeypatch):
        from fastmcp import FastMCP

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        analysis = FastMCP("analysis")
        asked = []

        @analysis.tool
        def analysis_job_status(job_id: str, wait_s: float = 0.0) -> dict:
            asked.append(wait_s)
            return {"job_id": job_id, "state": "done"} if job_id == "assemble-1" else {"error": {"code": "unknown_job", "message": "no job"}}

        @analysis.tool
        def analysis_job_results(job_id: str) -> dict:
            return {"job_id": job_id, "state": "done", "result_keys": {"map": "map_r1_map"}, "summary": {"tiles": 9}}

        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False, analysis_url=analysis)

        class Mapper:
            def job_status(self, request):
                return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})

        server._devices = {"Mapper": Mapper()}
        server._envelope_prefixes = {"Mapper"}
        done = server.wait_for_job("assemble-1", max_wait_s=5)
        assert done["result_keys"] == {"map": "map_r1_map"} and done["summary"]["tiles"] == 9 and asked == [5.0]
        with pytest.raises(ToolError, match="analysis server"):
            server.wait_for_job("nowhere")

    def test_one_dead_job_device_does_not_stop_waiting_on_the_others(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)

        class Dead:
            def job_status(self, request):
                raise tango.DevFailed()

        class Mapper:
            def job_status(self, request):
                if json.loads(request)["job_id"] != "j1":
                    return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})
                return json.dumps({"job_id": "j1", "state": "failed", "last_error": "x"})

        server._devices = {"Aaa": Dead(), "Mapper": Mapper()}
        server._envelope_prefixes = {"Aaa", "Mapper"}
        assert server.wait_for_job("j1")["state"] == "failed"
        with pytest.raises(ToolError, match="not answering: \\['Aaa'\\]"):
            server.wait_for_job("j-nowhere")

    def test_an_analysis_error_reply_is_said_not_swallowed_and_old_servers_are_polled(self, monkeypatch):
        from fastmcp import FastMCP
        from fastmcp.exceptions import ToolError as AnalysisError

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        old = FastMCP("old analysis")  # an analysis_job_status without wait_s
        polls = []

        @old.tool
        def analysis_job_status(job_id: str) -> dict:
            if job_id == "broken":
                raise AnalysisError("the job store is unreadable")
            polls.append(job_id)
            return {"job_id": job_id, "state": "done" if len(polls) > 1 else "running"}

        @old.tool
        def analysis_job_results(job_id: str) -> dict:
            return {"result_keys": {"map": "map_r2_map"}}

        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False, analysis_url=old)
        server._devices, server._envelope_prefixes = {}, set()
        done = server.wait_for_job("a1", max_wait_s=5)
        assert done["state"] == "done" and done["result_keys"] == {"map": "map_r2_map"} and len(polls) == 2
        with pytest.raises(ToolError, match="job store is unreadable"):
            server.wait_for_job("broken", max_wait_s=1)

    def test_a_job_still_running_at_the_deadline_says_to_wait_again(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)

        class Job:
            def job_status(self, request):
                return json.dumps({"job_id": "j1", "state": "running", "progress": 0.4})

        server._devices = {"Mapper": Job()}
        server._envelope_prefixes = {"Mapper"}
        status = server.wait_for_job("j1", device="Mapper", max_wait_s=0)
        assert status["state"] == "running" and status["progress"] == 0.4
        assert "wait_for_job again" in status["note"]

    def test_paused_jobs_are_waited_through_and_explained(self, monkeypatch):
        from asyncroscopy.jobs.job_device import RUNNING_STATES
        from asyncroscopy.mcp.mcp_server import WAIT_NOTES

        assert set(WAIT_NOTES) == RUNNING_STATES
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)

        class Job:
            def job_status(self, request):
                return json.dumps({"job_id": "j1", "state": "paused_suspended", "suspended": {"reason": "beam_current 2 < 10"}})

        server._devices = {"Mapper": Job()}
        server._envelope_prefixes = {"Mapper"}
        status = server.wait_for_job("j1", device="Mapper", max_wait_s=0)
        assert "beam_current 2 < 10" in status["note"] and "resumes by itself" in status["note"]

    def test_an_analysis_job_id_is_pointed_at_the_analysis_server(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)

        class Job:
            def job_status(self, request):
                return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})

        server._devices = {"Mapper": Job()}
        server._envelope_prefixes = {"Mapper"}
        with pytest.raises(ToolError, match="mcp.analysis_url"):
            server.wait_for_job("assemble-job-1", device="Mapper")


JSON_REQUEST = type("CommandInfo", (), {
    "in_type": tango.CmdArgType.DevString, "out_type": tango.CmdArgType.DevString,
    "in_type_desc": ':param request: JSON {"plan": {...}, "reservation_id": "..."}.', "out_type_desc": "",
})()


class TestRunJob:
    @staticmethod
    def mapper_server(server_factory):
        server = server_factory()
        calls = []

        def reserve(request):
            body = json.loads(request)
            calls.append(("reserve", body))
            if body["plan"].get("fov_m", 1) <= 0:
                return json.dumps({"schema_version": "1.1.0", "error": {"code": "plan_rejected", "message": "fov_m must be > 0"}})
            return json.dumps({"schema_version": "1.1.0", "reservation_id": f"r{len(calls)}", "estimate": {"steps": 9}})

        def start(request):
            body = json.loads(request)
            calls.append(("start", body["reservation_id"]))
            if server._devices["Mapper"].start_fails == "raise":
                raise RuntimeError("Tango: device not answering")
            if server._devices["Mapper"].start_fails == "error":
                return json.dumps({"schema_version": "1.1.0", "error": {"code": "device_error", "message": "a worker from before the device restarted"}})
            return json.dumps({"schema_version": "1.1.0", "job_id": "j-" + body["reservation_id"], "state": "queued"})

        class Device:
            finished = False  # True: the job is done by the time run_job looks
            start_fails = ""  # "raise" or "error": the start does not succeed

            def __init__(self):
                self.released = []

            def release(self, request):
                self.released.append(json.loads(request))
                return json.dumps({"state": "released"})

            def get_reservation(self, request):
                return '{"plan": {"fov_m": 2e-07}}'

            def job_status(self, request):
                job_id = json.loads(request)["job_id"]
                return json.dumps({"job_id": job_id, "state": "done" if self.finished else "running"})

            def results(self, request):
                return json.dumps({"job_id": json.loads(request)["job_id"], "result_keys": {"map": "map_r1_map"}, "summary": {"tiles": 9}})

        server._envelope_prefixes = {"Mapper"}
        server._devices = {"Mapper": Device()}
        server.tools = {"Mapper": {
            "reserve": server._create_wrapper(reserve, JSON_REQUEST, "reserve", "Mapper", tool_prefix="Mapper"),
            "start": server._create_wrapper(start, JSON_REQUEST, "start", "Mapper", tool_prefix="Mapper"),
        }}
        return server, calls

    def test_one_call_reserves_and_starts(self, server_factory, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.RUN_JOB_WAIT_S", 0.0)
        server, calls = self.mapper_server(server_factory)
        reply = server.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["job_id"] == "j-r1" and reply["estimate"] == {"steps": 9} and reply["reservation_id"] == "r1"
        assert reply["state"] == "running" and "wait_for_job again" in reply["note"]
        assert [c[0] for c in calls] == ["reserve", "start"]

    def test_wait_false_returns_at_once_with_the_job_id(self, server_factory, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.RUN_JOB_WAIT_S", 30.0)  # never reached: no wait
        server, calls = self.mapper_server(server_factory)
        began = time.monotonic()
        reply = server.run_job("Mapper", {"fov_m": 2e-7}, wait=False)
        assert time.monotonic() - began < 5.0
        assert reply["job_id"] == "j-r1" and reply["state"] == "running" and "Mapper_abort" in reply["note"]

    def test_a_short_job_comes_back_finished_with_its_results(self, server_factory):
        server, calls = self.mapper_server(server_factory)
        server._devices["Mapper"].finished = True
        reply = server.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["state"] == "done" and reply["result_keys"] == {"map": "map_r1_map"} and reply["summary"] == {"tiles": 9}

    def test_a_refused_plan_comes_back_without_a_start(self, server_factory):
        server, calls = self.mapper_server(server_factory)
        assert server.run_job("Mapper", {"fov_m": 0})["error"]["code"] == "plan_rejected"
        assert [c[0] for c in calls] == ["reserve"]
        with pytest.raises(ToolError, match="no job device"):
            server.run_job("Camera", {"fov_m": 2e-7})

    def test_a_held_start_is_approved_once_and_the_same_call_starts_that_reservation(self, server_factory, approvals_proxy):
        server, calls = self.mapper_server(server_factory)
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        with pytest.raises(ToolError, match="repeat this run_job call unchanged") as held:
            server.run_job("Mapper", {"fov_m": 2e-7})
        request_id = str(held.value).split("request_id=")[1].split(")")[0]
        pending = [p for p in json.loads(approvals_proxy.list_pending()) if p["request_id"] == request_id]
        assert pending and pending[0]["action"] == "Mapper_start" and "fov_m" in pending[0]["reason"]  # the operator sees the plan
        approvals_proxy.approve(json.dumps({"request_id": request_id, "who": "alice"}))
        reply = server.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["job_id"] == "j-r1"
        assert calls == [("reserve", {"plan": {"fov_m": 2e-7}}), ("start", "r1")]  # one reservation, started once
        with pytest.raises(ToolError, match="APPROVAL REQUIRED"):
            server.run_job("Mapper", {"fov_m": 2e-7})  # the grant was single use: a second map needs its own approval

    def test_a_denied_run_frees_the_instrument(self, server_factory, approvals_proxy):
        server, calls = self.mapper_server(server_factory)
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        with pytest.raises(ToolError) as held:
            server.run_job("Mapper", {"fov_m": 2e-7})
        request_id = str(held.value).split("request_id=")[1].split(")")[0]
        approvals_proxy.deny(json.dumps({"request_id": request_id, "who": "alice"}))
        status = server.approval_status(request_id)
        assert status["state"] == "denied" and "was released" in status["note"]
        assert server._devices["Mapper"].released == [{"reservation_id": "r1"}] and not server._held_runs

    def test_a_denied_run_is_freed_even_if_the_agent_never_asks(self, server_factory, approvals_proxy):
        server, calls = self.mapper_server(server_factory)
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        with pytest.raises(ToolError) as held:
            server.run_job("Mapper", {"fov_m": 2e-7})
        approvals_proxy.deny(json.dumps({"request_id": str(held.value).split("request_id=")[1].split(")")[0], "who": "alice"}))
        server._sweep_held(force=True)  # what any later run_job, wait_for_job or approval_status does
        assert server._devices["Mapper"].released == [{"reservation_id": "r1"}] and not server._held_runs

    def test_asking_again_before_the_operator_decides_files_no_second_request(self, server_factory, approvals_proxy):
        server, calls = self.mapper_server(server_factory)
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        before = len(json.loads(approvals_proxy.list_pending()))
        for _ in range(3):
            with pytest.raises(ToolError, match="APPROVAL REQUIRED") as held:
                server.run_job("Mapper", {"fov_m": 2e-7})
            assert "estimate" in str(held.value)  # what is held is said every time
        assert len(json.loads(approvals_proxy.list_pending())) == before + 1
        assert [c[0] for c in calls] == ["reserve"]  # one reservation; the gate held its start each time

    def test_the_held_retry_reply_carries_the_estimate(self, server_factory, approvals_proxy):
        server, calls = self.mapper_server(server_factory)
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        with pytest.raises(ToolError) as held:
            server.run_job("Mapper", {"fov_m": 2e-7})
        approvals_proxy.approve(json.dumps({"request_id": str(held.value).split("request_id=")[1].split(")")[0], "who": "alice"}))
        reply = server.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["job_id"] == "j-r1" and reply["estimate"] == {"steps": 9}  # the goal checker counts its dose

    @pytest.mark.parametrize("failure", ["raise", "error"])
    def test_a_start_that_fails_releases_the_reservation_it_made(self, server_factory, failure):
        server, calls = self.mapper_server(server_factory)
        server._devices["Mapper"].start_fails = failure
        if failure == "raise":
            with pytest.raises(ToolError, match="did not start.*was released.*r1"):
                server.run_job("Mapper", {"fov_m": 2e-7})
        else:
            reply = server.run_job("Mapper", {"fov_m": 2e-7})
            assert reply["error"]["code"] == "device_error" and reply["reservation_id"] == "r1" and "was released" in reply["note"]
        assert server._devices["Mapper"].released == [{"reservation_id": "r1"}]

    def test_a_plan_over_the_endpoints_limits_never_starts_by_either_path(self, server_factory):
        server, calls = self.mapper_server(server_factory)
        server.job_limits = {"max_duration_s": 3600.0}

        def long_reserve(request):
            calls.append(("reserve", json.loads(request)))
            return json.dumps({"schema_version": "1.1.0", "reservation_id": f"r{len(calls)}", "estimate": {"duration_s": 36000.0, "max_dose_e_per_a2": 10}})

        server.tools["Mapper"]["reserve"] = server._create_wrapper(long_reserve, JSON_REQUEST, "reserve", "Mapper", tool_prefix="Mapper")
        reply = server.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["error"]["code"] == "plan_rejected" and reply["error"]["violations"][0]["what"] == "duration_s"
        assert "ask" in reply["error"]["message"] and reply["estimate"]["duration_s"] == 36000.0
        direct = json.loads(server.tools["Mapper"]["reserve"]({"plan": {"fov_m": 2e-7}}))  # Mapper_reserve on the endpoint
        assert direct["error"]["code"] == "plan_rejected"
        assert [c[0] for c in calls] == ["reserve", "reserve"]  # never a start
        assert server._devices["Mapper"].released == [{"reservation_id": "r1"}, {"reservation_id": "r2"}]

    def test_an_approved_retry_through_a_new_mcp_session_starts_the_held_reservation(self, server_factory, approvals_proxy):
        """Agents open a new MCP session for every call (asyncroscopy.agents.mcp_tools): the retry must still match."""
        import asyncio

        from fastmcp import Client

        server, calls = self.mapper_server(server_factory)
        server.mcp.add_tool(server.run_job)
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))

        async def run_job_in_a_new_session():
            async with Client(server.mcp) as client:  # a fresh client and session each time, as the agents do
                return await client.call_tool("run_job", {"device": "Mapper", "plan": {"fov_m": 2e-7}}, raise_on_error=False)

        held = asyncio.run(run_job_in_a_new_session())
        text = held.content[0].text
        assert held.is_error and "APPROVAL REQUIRED" in text
        approvals_proxy.approve(json.dumps({"request_id": text.split("request_id=")[1].split(")")[0], "who": "alice"}))
        started = asyncio.run(run_job_in_a_new_session())
        assert not started.is_error and started.structured_content["job_id"] == "j-r1"
        assert calls == [("reserve", {"plan": {"fov_m": 2e-7}}), ("start", "r1")]  # no second reservation or request

    def test_each_call_in_its_own_http_session_still_finds_the_held_run(self, server_factory, approvals_proxy, monkeypatch):
        """Over HTTP every agent call has a new mcp-session-id; caller and run id are what stay the same."""
        import itertools

        server, calls = self.mapper_server(server_factory)
        sessions = itertools.count()
        headers = {"x-asyncroscopy-caller": "general-agent", "x-asyncroscopy-run": "run-1"}
        monkeypatch.setattr("fastmcp.server.dependencies.get_http_headers",
                            lambda include_all=False: {**headers, "mcp-session-id": f"s{next(sessions)}"})
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        with pytest.raises(ToolError) as held:
            server.run_job("Mapper", {"fov_m": 2e-7})
        approvals_proxy.approve(json.dumps({"request_id": str(held.value).split("request_id=")[1].split(")")[0], "who": "alice"}))
        assert server.run_job("Mapper", {"fov_m": 2e-7})["job_id"] == "j-r1"
        assert calls == [("reserve", {"plan": {"fov_m": 2e-7}}), ("start", "r1")]
        headers["x-asyncroscopy-run"] = "run-2"  # another agent run with the same caller name: its own reservation
        with pytest.raises(ToolError, match="APPROVAL REQUIRED"):
            server.run_job("Mapper", {"fov_m": 2e-7})
        assert [c[0] for c in calls].count("reserve") == 2

    def test_a_sweep_waits_for_a_run_job_that_is_starting(self, server_factory):
        import threading

        server, _calls = self.mapper_server(server_factory)
        server._held_runs[("c", "", "Mapper", "{}")] = {"reservation_id": "r9", "request_id": "", "device": "Mapper", "created": time.monotonic() - 3 * 3600, "summary": {}}
        server._run_job_lock.acquire()  # a run_job reserving or starting
        sweeper = threading.Thread(target=server._sweep_held, kwargs={"force": True})
        sweeper.start()
        sweeper.join(0.3)
        assert sweeper.is_alive() and server._devices["Mapper"].released == []  # it waits
        server._run_job_lock.release()
        sweeper.join(5)
        assert server._devices["Mapper"].released == [{"reservation_id": "r9"}]

    def test_two_callers_with_one_name_do_not_share_a_held_run(self, server_factory, approvals_proxy, monkeypatch):
        server, calls = self.mapper_server(server_factory)
        approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "operator"}))
        session = {"id": "s1"}
        monkeypatch.setattr(type(server), "_run_id", staticmethod(lambda: session["id"]))
        with pytest.raises(ToolError):
            server.run_job("Mapper", {"fov_m": 2e-7})
        session["id"] = "s2"
        with pytest.raises(ToolError):
            server.run_job("Mapper", {"fov_m": 2e-7})
        assert len([c for c in calls if c[0] == "reserve"]) == 2 and len(server._held_runs) == 2

    def test_stale_held_runs_are_bounded(self, server_factory, monkeypatch):
        server, _calls = self.mapper_server(server_factory)
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.MAX_HELD_RUNS", 2)
        for i in range(4):
            server._held_runs[("c", "", "Mapper", str(i))] = {"reservation_id": f"r{i}", "request_id": "", "device": "Mapper", "created": time.monotonic() - 10 + i, "summary": {}}
        server._sweep_held(force=True)
        assert sorted(h["reservation_id"] for h in server._held_runs.values()) == ["r2", "r3"]
        assert server._devices["Mapper"].released == [{"reservation_id": "r0"}, {"reservation_id": "r1"}]

    def test_endpoint_info_reports_the_gates_risks(self, server_factory):
        server, _calls = self.mapper_server(server_factory)
        server._native_names = {"run_job", "view_data"}
        info = server.endpoint_info()
        assert info["risks"]["Mapper_start"] == "high" and info["risks"]["Mapper_reserve"] == "low"
        assert info["risks"]["run_job"] == "high" and info["risks"]["view_data"] == "low" and info["job_devices"] == ["Mapper"]


class TestRunJobResume:
    def test_an_abort_reply_says_how_to_resume(self):
        reply = json.loads(MCPServer._resume_note("Mapper", json.dumps({"job_id": "j1", "reservation_id": "r1", "state": "aborted"})))
        assert reply["to_resume"].startswith('run_job with {"device": "Mapper", "resume": "r1"}')
        aborting = json.loads(MCPServer._resume_note("Mapper", json.dumps({"job_id": "j1", "reservation_id": "r1", "state": "aborting"})))
        assert aborting["to_resume"].startswith("once wait_for_job says aborted")
        done = json.dumps({"job_id": "j1", "reservation_id": "r1", "state": "done"})
        assert MCPServer._resume_note("Mapper", done) == done

    def test_resume_starts_the_aborted_reservation_without_a_new_one(self, server_factory):
        server, calls = TestRunJob.mapper_server(server_factory)
        server._devices["Mapper"].finished = True
        reply = server.run_job("Mapper", resume="r7")
        assert reply["state"] == "done" and calls == [("start", "r7")]  # no reserve: the reservation holds the plan

    def test_a_failed_resume_keeps_the_reservation(self, server_factory):
        server, calls = TestRunJob.mapper_server(server_factory)
        server._devices["Mapper"].start_fails = "raise"
        with pytest.raises(ToolError, match="is kept, so the job can still be resumed"):
            server.run_job("Mapper", resume="r7")
        assert server._devices["Mapper"].released == []

    def test_a_plan_and_a_resume_together_are_refused(self, server_factory):
        server, calls = TestRunJob.mapper_server(server_factory)
        with pytest.raises(ToolError, match="not both"):
            server.run_job("Mapper", {"fov_m": 2e-7}, resume="r7")
        with pytest.raises(ToolError, match="not both"):
            server.run_job("Mapper")
        assert calls == []
