"""The risk policy, endpoint roles, tool calls and the trace, run_job and the endpoint's own guards."""

import json
import time

import pytest
import tango
from fastmcp.exceptions import ToolError

from asyncroscopy.contracts import validate
from asyncroscopy.agents.guidance import render_reply
from asyncroscopy.mcp.components.jobs import resume_with
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.mcp.descriptors import describe_native
from asyncroscopy.utils.commands import declared_hints
from asyncroscopy.mcp.mcp_server import MCPServer
from asyncroscopy.safety.policy import operator_only, risk_of


def cmd_info(in_type=tango.CmdArgType.DevString, out_type=tango.CmdArgType.DevString):
    return type("CommandInfo", (), {"in_type": in_type, "out_type": out_type, "in_type_desc": "", "out_type_desc": ""})()


@pytest.fixture
def server_factory(monkeypatch, tmp_path):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

    def make(role="agent"):
        server = MCPServer(
            "test",
            "localhost",
            1234,
            blocked_classes=[],
            needs_approval={},
            data_device_address="asyncroscopy/data/default",
            verbose=False,
            endpoint_role=role,
            trace_path=str(tmp_path / f"trace_{role}.jsonl"),
        )
        server.add_device("DigitalTwin", stem=True, hints=declared_hints(DigitalTwin))
        return server

    return make


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
        "brake_off", "BrakeOff", "servo_off", "motor_off", "gripper_off", "clamp_off", "magnet_off", "turbopump_off",
        "ionpump_off", "cryocooler_off", "helium_off", "stop_flow", "stop_pid", "zfeedback_off", "z_loop_off",
        "seek_hard_stop", "descend_until_stop", "find_stop", "retract_approach", "withdraw_engage", "retract_tip_down",
        "stop_retract", "abort_withdraw", "stop_restart", "abort_retry", "estop_ack", "acknowledge_estop",
        "rearm_after_stop", "continue_after_pause", "kick_off", "lift_off", "back_off",
    ])
    def test_names_that_do_more_than_stop_are_high(self, name):
        assert risk_of("AnyDevice", name) == "high"

    @pytest.mark.parametrize("name", ["stop_scan", "AbortScan", "shutter_close", "blank_beam", "retract_tip", "withdraw_needle", "emergency_stop", "beam_off", "safe_state", "lamp_off", "halt", "abort_approach", "stop_move", "StopScan",
                                      "halt_xy", "stop_xyz", "stop_motors", "abort_autofocus", "stop_live", "withdraw_tip", "led_off", "illumination_off",
                                      "stop_indent", "abort_indent", "retract_indenter", "retract_head", "stop_imaging", "stop_exposure",
                                      "retract_objective", "xray_off", "stop_robot", "abort_protocol", "pause_protocol", "source_off",
                                      "stop_xray", "disengage_tip", "DisengageProbe"])
    def test_plain_stops_are_low(self, name):
        assert risk_of("AnyDevice", name) == "low"

    @pytest.mark.parametrize("name", ["indent_retract", "IndentRetract", "indent_withdraw", "indent_stop", "exposure_stop", "imaging_pause", "protocol_halt",
                                      "Off", "off", "disengage_motor", "disengage_motors", "disengage_z", "disengage_stage", "DisengageAll",
                                      "Disengage", "emergency_disengage", "retract_shutter", "withdraw_shutter", "ShutterRetract", "retract_x",
                                      "withdraw_y", "retract_xy", "retract_stage", "StageRetract", "retract_robot", "withdraw_motors",
                                      "retract_all", "retract_source", "retract_laser", "retract_gun", "retract_beam",
                                      "BeamStopMove", "beam_stop_move", "BeamStopX", "beam_stop_y", "BeamStop", "BeamStopScan"])
    def test_an_action_before_the_stop_verb_is_high(self, name):
        assert risk_of("AnyDevice", name) == "high"

    @pytest.mark.parametrize("name", ["withdraw", "retract", "Withdraw", "Retract", "withdraw_now", "retract_now"])
    def test_a_bare_retreat_verb_is_high_unless_the_device_brings_a_probe_to_the_sample(self, name):
        assert risk_of("SyringePump", name) == "high"
        assert risk_of("SimIndenter", name, probe=True) == "low"

    def test_probe_devices_are_recognised_by_their_approach_or_indent(self):
        from asyncroscopy.safety.policy import is_probe_device

        assert is_probe_device(["approach", "retract", "get_z"]) and is_probe_device(["indent", "retract"])
        assert is_probe_device(["approach_tip", "retract_tip", "move_xy"]) and is_probe_device(["ApproachSurface", "Withdraw"])
        assert not is_probe_device(["infuse", "withdraw", "get_volume"]) and not is_probe_device(["move_stage", "retract"])

    def test_the_twins_named_retreats_stay_low(self):
        for dev, command in [("AFMTwin", "retract_tip"), ("SimIndenter", "retract_indenter"), ("Fib", "withdraw_needle"), ("Spm", "retract_z")]:
            assert risk_of(dev, command) == "low", command
        assert risk_of("Microscope", "retract_screen") == "high"  # uncovers the detectors below it


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
        assert declared_hints(DigitalTwin)["export_ground_truth"]["operator_only"]
        assert not declared_hints(DigitalTwin)["move_stage"].get("operator_only")
        assert operator_only("AnyDevice", "Init") and not operator_only("DigitalTwin", "export_ground_truth")

    def test_a_bad_auto_routine_profile_is_refused(self):
        from asyncroscopy.safety.policy import check_auto_routine

        assert check_auto_routine({"stem": ["move_stage"], "tools": ["Robot.pick"]}) == {"stem": ["move_stage"], "tools": ["Robot.pick"]}
        for bad in ({"stm": ["move_stage"]}, {"stem": "move_stage"}, ["move_stage"], {"stem": [""]}):
            with pytest.raises(ValueError):
                check_auto_routine(bad)


class TestToolCalls:
    def test_a_high_risk_command_runs_and_is_traced(self, server_factory, tmp_path):
        server = server_factory()
        wrapper = server._create_wrapper(lambda arg: f"moved {arg}", cmd_info(), "move_stage", "DigitalTwin")
        assert wrapper("here") == "moved here"
        event = json.loads((tmp_path / "trace_agent.jsonl").read_text().splitlines()[-1])
        validate(event, "trace_event")
        assert event["action"] == "DigitalTwin_move_stage" and event["ok"]

    def test_every_tool_is_found_and_run_through_call_tool(self, server_factory):
        import asyncio
        from fastmcp.tools import Tool

        server = server_factory()
        calls = []
        for command in ("set_column_valves", "move_stage", "get_stage"):
            wrapper = server._create_wrapper(lambda arg, c=command: calls.append((c, arg)) or "ok", cmd_info(), command, "DigitalTwin")
            server.registered[f"DigitalTwin_{command}"] = Tool.from_function(wrapper)
        found = {t["name"]: t for t in server.discovery.find_tools()["tools"]}
        assert {"DigitalTwin_set_column_valves", "DigitalTwin_move_stage", "DigitalTwin_get_stage"} <= set(found)
        assert [t["name"] for t in server.discovery.find_tools("column valves")["tools"]] == ["DigitalTwin_set_column_valves"]
        arg = next(iter(found["DigitalTwin_move_stage"]["arguments"]))
        asyncio.run(server.discovery.call_tool("DigitalTwin_move_stage", {arg: "here"}))
        asyncio.run(server.discovery.call_tool("DigitalTwin_set_column_valves", {arg: "open"}))
        assert calls == [("move_stage", "here"), ("set_column_valves", "open")]
        with pytest.raises(ToolError, match="no tool"):
            asyncio.run(server.discovery.call_tool("DigitalTwin_nothing", {}))

    def test_only_a_person_continues_a_paused_job(self, server_factory):
        server = server_factory()

        class Device:
            def get_reservation(self, request):
                return json.dumps({"plan_id": "p1", "resolved_plan": {"drives": {}}})

        server.add_device("Mapper", Device(), envelope=True)
        start = server._create_wrapper(lambda arg: "started", cmd_info(), "start", "Mapper")
        assert start(json.dumps({"reservation_id": "r1"})) == "started"
        with pytest.raises(ToolError, match="only they continue it"):
            start(json.dumps({"reservation_id": "r1", "continue": True}))
        operator = server_factory(role="operator")
        operator.add_device("Mapper", Device(), envelope=True)
        resume = operator._create_wrapper(lambda arg: "continued", cmd_info(), "start", "Mapper")
        assert resume(json.dumps({"reservation_id": "r1", "continue": True})) == "continued"


class TestEndpointRoles:
    class FakeDb:
        class Exported:
            value_string = ["asyncroscopy/instrument/default"]

        def get_device_exported(self, pattern):
            return self.Exported()

    class FakeProxy:
        def __init__(self, name):
            self.name = name

        def set_timeout_millis(self, ms):
            pass

        def info(self):
            return type("Info", (), {"dev_class": "DigitalTwin"})()

        def command_list_query(self):
            return [type("Cmd", (), {"cmd_name": n})() for n in ("move_stage", "export_ground_truth", "get_stage")]

        def get_attribute_list(self):
            return ["action_hints"]

        def read_attribute(self, name):
            return type("Value", (), {"value": json.dumps(declared_hints(DigitalTwin))})()

        def __getattr__(self, item):
            return lambda *a: None

    @pytest.mark.parametrize("role, hidden", [("agent", True), ("operator", False)])
    def test_operator_only_commands(self, monkeypatch, role, hidden):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: self.FakeDb())
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", self.FakeProxy)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False, endpoint_role=role)
        twin = set(server._find_tools()["DigitalTwin"])
        assert ("export_ground_truth" not in twin) is hidden
        assert {"move_stage", "get_stage"} <= twin

    def test_the_monitor_role_offers_only_reads(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: self.FakeDb())
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", self.FakeProxy)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False, endpoint_role="monitor")
        assert set(server._find_tools()["DigitalTwin"]) == {"get_stage"}

    def test_unknown_role_rejected(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        with pytest.raises(ValueError, match="endpoint_role"):
            MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", endpoint_role="admin")


class TestJsonArguments:
    """Envelope commands take a JSON string; models pass objects. All forms must work."""

    @staticmethod
    def make(monkeypatch, seen):
        import asyncio

        from fastmcp import Client, FastMCP
        from fastmcp.tools import Tool

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)
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
        return MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)

    def test_same_family_devices_get_member_prefixes(self, monkeypatch):
        tools = self.server_for(monkeypatch, ["lab/mapper/1", "lab/mapper/2"])._find_tools()
        assert {prefix: commands["start"][0]() for prefix, commands in tools.items()} == {"Mapper_1": "lab/mapper/1", "Mapper_2": "lab/mapper/2"}

    def test_repeated_default_members_are_named_by_their_config_key(self, monkeypatch):
        tools = self.server_for(monkeypatch, ["asyncroscopy/grid1/default", "asyncroscopy/grid2/default"])._find_tools()
        assert set(tools) == {"Mapper_grid1", "Mapper_grid2"}

    def test_repeated_members_fall_back_to_the_device_name(self, monkeypatch):
        tools = self.server_for(monkeypatch, ["east/mapper/default", "west/mapper/default"])._find_tools()
        assert set(tools) == {"Mapper_east_mapper_default", "Mapper_west_mapper_default"}


class TestWaitForJob:
    def test_waits_until_the_job_finishes_and_refuses_unknown_devices(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)

        class Job:
            polls = 0

            def job_status(self, request):
                Job.polls += 1
                return json.dumps({"job_id": "j1", "state": "done" if Job.polls >= 2 else "running"})

            def results(self, request):
                return json.dumps({"schema_version": "1.1.0", "job_id": "j1", "state": "done", "result_keys": {"run": "map_r1.h5"}, "summary": {"grid": {"rows": 3}}})

        server.add_device("Mapper", Job(), envelope=True)
        done = server.jobs.wait_for_job("j1", device="Mapper", max_wait_s=10)
        assert done["state"] == "done" and Job.polls == 2
        assert done["result_keys"] == {"run": "map_r1.h5"} and done["summary"]["grid"]["rows"] == 3
        with pytest.raises(ToolError, match="Mapper"):
            server.jobs.wait_for_job("j1", device="Camera")

    def test_the_device_can_be_left_out(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)

        class Other:
            def job_status(self, request):
                return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})

        class Mapper:
            def job_status(self, request):
                return json.dumps({"job_id": "j1", "state": "failed", "last_error": "stage refused"})

        server.add_device("IndentGrid", Other(), envelope=True)
        server.add_device("Mapper", Mapper(), envelope=True)
        assert server.jobs.wait_for_job("j1")["last_error"] == "stage refused"

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

        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False, urls={"analysis": analysis})

        class Mapper:
            def job_status(self, request):
                return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})

        server.add_device("Mapper", Mapper(), envelope=True)
        done = server.jobs.wait_for_job("assemble-1", max_wait_s=5)
        assert done["result_keys"] == {"map": "map_r1_map"} and done["summary"]["tiles"] == 9 and asked == [5.0]
        with pytest.raises(ToolError, match="analysis server"):
            server.jobs.wait_for_job("nowhere")

    def test_one_dead_job_device_does_not_stop_waiting_on_the_others(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)

        class Dead:
            def job_status(self, request):
                raise tango.DevFailed()

        class Mapper:
            def job_status(self, request):
                if json.loads(request)["job_id"] != "j1":
                    return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})
                return json.dumps({"job_id": "j1", "state": "failed", "last_error": "x"})

        server.add_device("Aaa", Dead(), envelope=True)
        server.add_device("Mapper", Mapper(), envelope=True)
        assert server.jobs.wait_for_job("j1")["state"] == "failed"
        with pytest.raises(ToolError, match="not answering: \\['Aaa'\\]"):
            server.jobs.wait_for_job("j-nowhere")

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

        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False, urls={"analysis": old})
        server.devices.clear()
        done = server.jobs.wait_for_job("a1", max_wait_s=5)
        assert done["state"] == "done" and done["result_keys"] == {"map": "map_r2_map"} and len(polls) == 2
        with pytest.raises(ToolError, match="job store is unreadable"):
            server.jobs.wait_for_job("broken", max_wait_s=1)

    def test_a_job_still_running_at_the_deadline_says_to_wait_again(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)

        class Job:
            def job_status(self, request):
                return json.dumps({"job_id": "j1", "state": "running", "progress": 0.4})

        server.add_device("Mapper", Job(), envelope=True)
        status = server.jobs.wait_for_job("j1", device="Mapper", max_wait_s=0)
        assert status["state"] == "running" and status["progress"] == 0.4
        assert status["next_actions"] == [{"tool": "wait_for_job", "args": {"job_id": "j1"}}]
        assert "wait_for_job again" in render_reply("wait_for_job", {"job_id": "j1"}, status)["note"]

    def test_paused_jobs_are_waited_through_and_explained(self, monkeypatch):
        from asyncroscopy.agents.guidance import WAIT_NOTES
        from asyncroscopy.core.names import RUNNING_JOB_STATES

        assert set(WAIT_NOTES) == RUNNING_JOB_STATES
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)

        class Job:
            def job_status(self, request):
                return json.dumps({"job_id": "j1", "state": "paused_suspended", "suspended": {"reason": "beam_current 2 < 10"}})

        server.add_device("Mapper", Job(), envelope=True)
        status = server.jobs.wait_for_job("j1", device="Mapper", max_wait_s=0)
        assert status["suspended"]["reason"] == "beam_current 2 < 10" and status["next_actions"][0]["tool"] == "wait_for_job"
        note = render_reply("wait_for_job", {"job_id": "j1"}, status)["note"]
        assert "beam_current 2 < 10" in note and "resumes by itself" in note

    def test_an_analysis_job_id_is_pointed_at_the_analysis_server(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)

        class Job:
            def job_status(self, request):
                return json.dumps({"error": {"code": "unknown_job", "message": "no job"}})

        server.add_device("Mapper", Job(), envelope=True)
        with pytest.raises(ToolError, match="mcp.urls.analysis"):
            server.jobs.wait_for_job("assemble-job-1", device="Mapper")


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
            if server.devices["Mapper"].proxy.start_fails == "raise":
                raise RuntimeError("Tango: device not answering")
            if server.devices["Mapper"].proxy.start_fails == "error":
                return json.dumps({"schema_version": "1.1.0", "error": {"code": "device_error", "message": "a worker from before the device restarted"}})
            return json.dumps({"schema_version": "1.1.0", "job_id": "j-" + body["reservation_id"], "state": "queued"})

        class Device:
            finished = False  # set True when the job is done before run_job looks
            start_fails = ""  # set to raise or error to make the start fail

            def __init__(self):
                self.released = []

            def release(self, request):
                self.released.append(json.loads(request))
                return json.dumps({"state": "released"})

            def get_reservation(self, request):
                return '{"plan": {"fov_m": 2e-07}, "resolved_plan": {"drives": {}}}'

            def job_status(self, request):
                job_id = json.loads(request)["job_id"]
                return json.dumps({"job_id": job_id, "state": "done" if self.finished else "running"})

            def results(self, request):
                return json.dumps({"job_id": json.loads(request)["job_id"], "result_keys": {"map": "map_r1_map"}, "summary": {"tiles": 9}})

        server.add_device("Mapper", Device(), envelope=True)
        server.tools = {"Mapper": {
            "reserve": server._create_wrapper(reserve, JSON_REQUEST, "reserve", "Mapper", tool_prefix="Mapper"),
            "start": server._create_wrapper(start, JSON_REQUEST, "start", "Mapper", tool_prefix="Mapper"),
        }}
        return server, calls

    def test_one_call_reserves_and_starts(self, server_factory, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.components.jobs.RUN_JOB_WAIT_S", 0.0)
        server, calls = self.mapper_server(server_factory)
        reply = server.jobs.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["job_id"] == "j-r1" and reply["estimate"] == {"steps": 9} and reply["reservation_id"] == "r1"
        assert reply["state"] == "running" and reply["next_actions"] == [{"tool": "wait_for_job", "args": {"job_id": "j-r1"}}]
        note = render_reply("run_job", {"device": "Mapper"}, reply)["note"]
        assert 'call wait_for_job with {"job_id": "j-r1"} now' in note and "every phase" in note
        assert [c[0] for c in calls] == ["reserve", "start"]

    def test_wait_false_returns_at_once_with_the_job_id(self, server_factory, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.components.jobs.RUN_JOB_WAIT_S", 30.0)  # never reached because nothing waits
        server, calls = self.mapper_server(server_factory)
        began = time.monotonic()
        reply = server.jobs.run_job("Mapper", {"fov_m": 2e-7}, wait=False)
        assert time.monotonic() - began < 5.0
        assert reply["job_id"] == "j-r1" and reply["state"] == "running"
        assert [a["tool"] for a in reply["next_actions"]] == ["wait_for_job", "Mapper_abort"]
        assert "Mapper_abort stops it" in render_reply("run_job", {"wait": False}, reply)["note"]

    def test_a_short_job_comes_back_finished_with_its_results(self, server_factory):
        server, calls = self.mapper_server(server_factory)
        server.devices["Mapper"].proxy.finished = True
        reply = server.jobs.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["state"] == "done" and reply["result_keys"] == {"map": "map_r1_map"} and reply["summary"] == {"tiles": 9}
        assert reply["next_actions"] == [] and reply["warnings"] == [{"code": "finished"}]
        rendered = render_reply("run_job", {}, reply)
        assert "nothing needs running" in rendered["note"] and "warnings" not in rendered and "next_actions" not in rendered

    def test_a_refused_plan_comes_back_without_a_start(self, server_factory):
        server, calls = self.mapper_server(server_factory)
        assert server.jobs.run_job("Mapper", {"fov_m": 0})["error"]["code"] == "plan_rejected"
        assert [c[0] for c in calls] == ["reserve"]
        with pytest.raises(ToolError, match="no job device"):
            server.jobs.run_job("Camera", {"fov_m": 2e-7})

    @pytest.mark.parametrize("failure", ["raise", "error"])
    def test_a_start_that_fails_releases_the_reservation_it_made(self, server_factory, failure):
        server, calls = self.mapper_server(server_factory)
        server.devices["Mapper"].proxy.start_fails = failure
        if failure == "raise":
            with pytest.raises(ToolError, match="did not start.*was released.*r1"):
                server.jobs.run_job("Mapper", {"fov_m": 2e-7})
        else:
            reply = server.jobs.run_job("Mapper", {"fov_m": 2e-7})
            assert reply["error"]["code"] == "device_error" and reply["reservation_id"] == "r1" and "was released" in reply["note"]
        assert server.devices["Mapper"].proxy.released == [{"reservation_id": "r1"}]

    def test_a_plan_over_the_endpoints_limits_never_starts_by_either_path(self, server_factory):
        server, calls = self.mapper_server(server_factory)
        server.jobs.job_limits = {"max_duration_s": 3600.0}

        def long_reserve(request):
            calls.append(("reserve", json.loads(request)))
            return json.dumps({"schema_version": "1.1.0", "reservation_id": f"r{len(calls)}", "estimate": {"duration_s": 36000.0, "max_dose_e_per_a2": 10}})

        server.tools["Mapper"]["reserve"] = server._create_wrapper(long_reserve, JSON_REQUEST, "reserve", "Mapper", tool_prefix="Mapper")
        reply = server.jobs.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["error"]["code"] == "plan_rejected" and reply["error"]["violations"][0]["what"] == "duration_s"
        assert "ask" in reply["error"]["message"] and reply["estimate"]["duration_s"] == 36000.0
        direct = json.loads(server.tools["Mapper"]["reserve"]({"plan": {"fov_m": 2e-7}}))  # Mapper_reserve on the endpoint
        assert direct["error"]["code"] == "plan_rejected"
        assert [c[0] for c in calls] == ["reserve", "reserve"]  # never a start
        assert server.devices["Mapper"].proxy.released == [{"reservation_id": "r1"}, {"reservation_id": "r2"}]

    def test_endpoint_info_reports_the_risks(self, server_factory):
        server, _calls = self.mapper_server(server_factory)
        server.native_names = {"run_job", "view_data"}
        server.descriptors.update({name: describe_native(name) for name in server.native_names})
        info = server.discovery.endpoint_info()
        assert info["risks"]["Mapper_start"] == "high" and info["risks"]["Mapper_reserve"] == "low"
        assert info["risks"]["run_job"] == "high" and info["risks"]["view_data"] == "low" and info["job_devices"] == ["Mapper"]
        assert info["stack_id"] == server.stack_id and info["stack_id"].startswith("tango://")


class TestRunJobResume:
    def test_an_abort_reply_says_how_to_resume(self):
        reply = json.loads(resume_with("Mapper", json.dumps({"job_id": "j1", "reservation_id": "r1", "state": "aborted"})))
        assert reply["resume_with"] == {"tool": "run_job", "args": {"device": "Mapper", "resume": "r1"}}
        assert render_reply("Mapper_abort", {}, reply)["to_resume"].startswith('run_job with {"device": "Mapper", "resume": "r1"}')
        aborting = json.loads(resume_with("Mapper", json.dumps({"job_id": "j1", "reservation_id": "r1", "state": "aborting"})))
        assert render_reply("Mapper_abort", {}, aborting)["to_resume"].startswith("once wait_for_job says aborted")
        done = json.dumps({"job_id": "j1", "reservation_id": "r1", "state": "done"})
        assert resume_with("Mapper", done) == done

    def test_resume_starts_the_aborted_reservation_without_a_new_one(self, server_factory):
        server, calls = TestRunJob.mapper_server(server_factory)
        server.devices["Mapper"].proxy.finished = True
        reply = server.jobs.run_job("Mapper", resume="r7")
        assert reply["state"] == "done" and calls == [("start", "r7")]  # no reserve because the reservation holds the plan

    def test_a_failed_resume_keeps_the_reservation(self, server_factory):
        server, calls = TestRunJob.mapper_server(server_factory)
        server.devices["Mapper"].proxy.start_fails = "raise"
        with pytest.raises(ToolError, match="is kept, so the job can still be resumed"):
            server.jobs.run_job("Mapper", resume="r7")
        assert server.devices["Mapper"].proxy.released == []

    def test_a_plan_and_a_resume_together_are_refused(self, server_factory):
        server, calls = TestRunJob.mapper_server(server_factory)
        with pytest.raises(ToolError, match="not both"):
            server.jobs.run_job("Mapper", {"fov_m": 2e-7}, resume="r7")
        with pytest.raises(ToolError, match="not both"):
            server.jobs.run_job("Mapper")
        assert calls == []


MEASURED = {"duration_s": 60.0, "max_dose_e_per_a2": 5.0, "beam_current_pa": 50.0, "beam_current_source": "measured", "beam_current_age_s": 0.0}


class TestEndpointGuards:
    """What the agent endpoint checks on its own: job limits at every start, the dose cap's current, dry-run risks."""

    @staticmethod
    def server_with(server_factory, tango_ctx, estimate, hardware=True, limits=None):
        server, calls = TestRunJob.mapper_server(server_factory)
        server.jobs.job_limits = limits if limits is not None else {"max_dose_e_per_a2": 100.0}
        server.stack_simulated = lambda: not hardware  # what the devices' simulated property would say

        def reserve(request):
            calls.append(("reserve", json.loads(request)))
            return json.dumps({"schema_version": "1.1.0", "reservation_id": f"r{len(calls)}", "estimate": dict(estimate)})

        server.tools["Mapper"]["reserve"] = server._create_wrapper(reserve, JSON_REQUEST, "reserve", "Mapper", tool_prefix="Mapper")
        server.devices["Mapper"].proxy.get_reservation = lambda request: json.dumps({"reservation_id": json.loads(request)["reservation_id"], "estimate": dict(estimate), "resolved_plan": {"drives": {}}})
        return server, calls

    @pytest.mark.parametrize("estimate, why", [
        ({**MEASURED, "beam_current_source": "default", "beam_current_reason": "the microscope reports an assumed current"}, "assumed current"),
        ({**MEASURED, "beam_current_reason": "the screen-current measurement is 3600 s old"}, "s old"),
        ({k: v for k, v in MEASURED.items() if k != "max_dose_e_per_a2"}, "gives no dose"),
        ({**MEASURED, "max_dose_e_per_a2": None}, "gives no dose"),
    ])
    def test_on_hardware_the_dose_cap_refuses_a_dose_it_cannot_check(self, server_factory, tango_ctx, estimate, why):
        server, calls = self.server_with(server_factory, tango_ctx, estimate)
        reply = server.jobs.run_job("Mapper", {"fov_m": 2e-7})
        assert reply["error"]["code"] == "plan_rejected" and why in reply["error"]["message"], reply
        assert [c[0] for c in calls] == ["reserve"] and server.devices["Mapper"].proxy.released == [{"reservation_id": "r1"}]

    def test_a_measured_dose_and_a_job_without_a_dose_both_run_on_hardware(self, server_factory, tango_ctx, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.components.jobs.RUN_JOB_WAIT_S", 0.0)
        afm = {"duration_s": 60.0, "steps": 9, "budgets": {"tip_force_nn": {"value": 2.0, "budget": 5.0, "units": "nN"}}}
        for estimate in (MEASURED, afm):
            server, calls = self.server_with(server_factory, tango_ctx, estimate)
            assert server.jobs.run_job("Mapper", {"fov_m": 2e-7})["job_id"] == "j-r1", estimate

    def test_the_twin_runs_with_its_measured_current(self, server_factory, tango_ctx, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.components.jobs.RUN_JOB_WAIT_S", 0.0)
        assumed = {**MEASURED, "beam_current_source": "default", "beam_current_reason": "assumed"}
        server, _calls = self.server_with(server_factory, tango_ctx, assumed, hardware=False)
        assert server.jobs.run_job("Mapper", {"fov_m": 2e-7})["job_id"] == "j-r1"

    def test_a_start_of_a_reservation_over_the_limits_never_starts(self, server_factory, tango_ctx):
        long = {**MEASURED, "duration_s": 36000.0}
        server, calls = self.server_with(server_factory, tango_ctx, long, hardware=False, limits={"max_duration_s": 3600.0})
        direct = json.loads(server.tools["Mapper"]["start"]({"reservation_id": "r9"}))
        assert direct["error"]["code"] == "plan_rejected" and direct["error"]["violations"][0]["what"] == "duration_s"
        resumed = server.jobs.run_job("Mapper", resume="r9")
        assert resumed["error"]["code"] == "plan_rejected"
        assert calls == []
        server.devices["Mapper"].proxy.get_reservation = lambda request: (_ for _ in ()).throw(RuntimeError("device not answering"))
        unread = json.loads(server.tools["Mapper"]["start"]({"reservation_id": "r9"}))
        assert "could not read the reservation" in unread["error"]["message"] and calls == []

    def test_a_reserve_carries_the_endpoints_high_risk_overrides_for_a_dry_run(self, server_factory):
        server, calls = TestRunJob.mapper_server(server_factory)
        server.config.risk_overrides = {"AFMTwin.get_image": "high", "Mapper_reserve": "low"}
        server.tools["Mapper"]["reserve"]({"plan": {"fov_m": 2e-7, "dry_run": True}})
        assert calls[-1] == ("reserve", {"plan": {"fov_m": 2e-7, "dry_run": True}, "risk_overrides": {"AFMTwin.get_image": "high"}})

    def test_a_device_in_fault_is_named_in_a_safety_line(self, monkeypatch, capsys):
        class Faulty(TestEndpointRoles.FakeProxy):
            def state(self):
                return tango.DevState.FAULT if "instrument" in self.name else tango.DevState.ON

            def status(self):
                return "afm.yaml: settings.stage_x_m.max must be a number"

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: TestEndpointRoles.FakeDb())
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", Faulty)
        MCPServer("t", "h", 1, blocked_classes=[], needs_approval={}, data_device_address="d", verbose=False)._find_tools()
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[safety]")]
        assert len(lines) == 1 and "asyncroscopy/instrument/default is in FAULT" in lines[0] and "stage_x_m.max" in lines[0]


def test_a_plan_cannot_say_what_its_job_drives(mapper_proxy):
    reply = json.loads(mapper_proxy.reserve(json.dumps({"plan": {"size_m": [2e-7, 2e-7], "fov_m": 1e-7, "drives": {}}})))
    assert reply["error"]["code"] == "bad_request" and "drives" in reply["error"]["message"]


def test_an_spm_stage_move_is_refused_while_the_tip_is_engaged():
    from asyncroscopy.instruments.scanning_probe_microscope.scanning_probe_microscope import refuse_if_engaged

    class Approach:
        def __init__(self, engaged):
            self.engaged = engaged

        @property
        def approached(self):
            if isinstance(self.engaged, Exception):
                raise self.engaged
            return self.engaged

    refuse_if_engaged(Approach(False))
    with pytest.raises(RuntimeError, match="retract it first"):
        refuse_if_engaged(Approach(True))
    with pytest.raises(RuntimeError, match="could not be read"):
        refuse_if_engaged(Approach(tango.DevFailed()))


def test_stop_all_stops_every_device_and_reports_what_it_could_not(server_factory, monkeypatch):
    monkeypatch.setattr("asyncroscopy.safety.estop.STOP_ALL_WAIT_S", 1.0)
    server = server_factory()
    server.config.risk_overrides = {"*": "high"}  # a deployment that marks everything high must not hold up the e-stop
    done = []

    class Device:
        def __init__(self, prefix, commands, broken=(), hang=False):
            self.prefix, self.commands, self.broken, self.hang = prefix, commands, set(broken), hang

        def command_inout(self, name):
            if self.hang:
                time.sleep(3)
            if name in self.broken:
                raise RuntimeError(f"{self.prefix} not answering")
            done.append(f"{self.prefix}_{name}")
            return json.dumps({"aborted": [{"job_id": "j1", "state": "aborted"}]}) if name == "abort_all" else None

    void, text = tango.CmdArgType.DevVoid, tango.CmdArgType.DevString
    devices = {
        "Mapper": (Device("Mapper", ["abort_all"], broken=["abort_all"]), {"abort_all": void, "abort": text, "start": text}),
        "Tuner": (Device("Tuner", ["abort_all"]), {"abort_all": void}),
        "OldJob": (Device("OldJob", []), {"abort": text, "start": text}),
        "DigitalTwin": (Device("DigitalTwin", []), {"blank_beam": void, "move_stage": text, "unblank_beam": void}),
        "Raman": (Device("Raman", []), {"laser_off": void, "laser_on": void, "close_shutter": void}),
        "GunCtl": (Device("GunCtl", []), {"hv_off": void, "filament_off": void, "stop": void, "estop": void}),
        "AFMTwin": (Device("AFMTwin", []), {"approach_tip": void, "retract_tip": void}),
        "SlowCam": (Device("SlowCam", [], hang=True), {"close_shutter": void}),
        "BlindStem": (Device("BlindStem", []), {"acquire_scanned_image": text}),
    }
    kinds = {"Mapper": "envelope", "Tuner": "envelope", "OldJob": "envelope", "DigitalTwin": "stem", "BlindStem": "stem", "AFMTwin": "probe"}
    server.tools = {}
    for prefix, (device, types) in devices.items():
        server.add_device(prefix, device, **({kinds[prefix]: True} if prefix in kinds else {}))
        server.tools[prefix] = {c: server._create_wrapper(lambda *a: done.append("through the wrapper"), cmd_info(in_type=t), c, prefix) for c, t in types.items()}
    started = time.monotonic()
    reply = server.estop.stop_all()
    assert time.monotonic() - started < 3.0  # A hung device does not hold up the report. The two phases take 1 s each.
    assert "through the wrapper" not in done  # stop_all calls the devices directly
    assert not set(done) & {"GunCtl_hv_off", "GunCtl_filament_off", "GunCtl_stop", "GunCtl_estop", "DigitalTwin_unblank_beam", "Raman_laser_on", "AFMTwin_approach_tip"}
    assert done.count("DigitalTwin_blank_beam") == 2 and done.count("Raman_laser_off") == 2 and done.count("Raman_close_shutter") == 2
    assert {"Tuner_abort_all", "AFMTwin_retract_tip"} <= set(done)
    failed = {s["tool"] for s in reply["steps"] if not s["ok"]}
    assert failed == {"Mapper_abort_all", "OldJob_abort_all", "BlindStem_blank_beam", "SlowCam"} and not reply["stopped"]
    assert len(reply["not_stopped"]) == 1 and reply["not_stopped"][0].startswith("DigitalTwin: a stage move")


def test_stop_all_reports_a_blank_the_backend_does_not_support_as_a_failure(server_factory, monkeypatch):
    """A JEOL or an atomistic twin has no blanker: its blank_beam raises UnsupportedCommand, and the e-stop must say
    the beam was not blanked instead of reporting it stopped."""
    import threading

    from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope
    from asyncroscopy.instruments.electron_microscope.jeol import JeolMicroscope

    monkeypatch.setattr("asyncroscopy.safety.estop.STOP_ALL_WAIT_S", 1.0)
    jeol = JeolMicroscope.__new__(JeolMicroscope)
    jeol._command_lock = threading.RLock()
    jeol._check_connected = lambda name: None

    class Proxy:
        def command_inout(self, name):
            return getattr(ElectronMicroscope, name)(jeol)

    server = server_factory()
    server.tools = {}
    server.add_device("Jeol", Proxy(), stem=True)
    void = tango.CmdArgType.DevVoid
    server.tools["Jeol"] = {"blank_beam": server._create_wrapper(lambda *a: None, cmd_info(in_type=void), "blank_beam", "Jeol")}
    reply = server.estop.stop_all()
    blanks = [s for s in reply["steps"] if s["tool"] == "Jeol_blank_beam"]
    assert reply["stopped"] is False and blanks and all(not s["ok"] and "does not support blank_beam" in s["error"] for s in blanks)
