"""Permissions (docs/MCP/permissions.md): the agent endpoint holds calls of an ``approve`` tool, the operator endpoint
decides them, and the settings live in the permissions file. Real MCP servers over the session's Tango devices."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.mcp_tools import McpServer
from asyncroscopy.core.names import CALLER_HEADER, RUN_HEADER
from asyncroscopy.mcp.components.permissions import always_runs
from asyncroscopy.mcp.descriptors import DeviceInfo, describe_command, describe_native
from asyncroscopy.safety.permissions import PermissionStore, iso_utc, seconds_of
from asyncroscopy.trace.writer import read_tool_log
from asyncroscopy.workflows.blocks import Block, register_block
from asyncroscopy.workflows.graph import Workflow, WorkflowState
from asyncroscopy.workflows.runner import resume_workflow, run_workflow

DOSE = "SAMPLE_get_dose"
SAMPLE = "asyncroscopy/sample/mapping"
TWIN = "asyncroscopy/digitaltwin/default"


async def call(server, name, arguments=None):
    async with Client(server.mcp) as client:
        return await client.call_tool(name, arguments or {}, raise_on_error=False)


async def operator_call(server, name, arguments=None):
    async with Client(server.mcp) as client:
        return (await client.call_tool(name, arguments or {})).structured_content


def run(coro):
    return asyncio.run(coro)


def held_pair(stack):
    return stack("agent", SAMPLE, needs_approval={"SAMPLE": ["get_dose"]}), stack("operator", SAMPLE)


def test_a_held_call_returns_approval_required_and_resending_it_gives_the_same_request(stack):
    agent, operator = held_pair(stack)
    first = run(call(agent, DOSE, {"request": "{}"}))
    assert first.is_error
    held = first.structured_content
    assert held["status"] == "approval_required" and held["tool"] == DOSE and held["arguments"] == {"request": "{}", "x_m": None, "y_m": None}
    assert held["acts"] is False and held["risk"] == "low" and len(held["arguments_sha256"]) == 64
    assert held["expires_at"].endswith("Z") and "waiting for a person" in first.content[0].text
    again = run(call(agent, DOSE, {"request": "{}"}))
    assert again.structured_content["request_id"] == held["request_id"]
    through = run(call(agent, "call_tool", {"name": DOSE, "arguments": {"request": "{}"}}))
    assert through.structured_content["request_id"] == held["request_id"]
    pending = run(operator_call(operator, "list_pending_approvals"))["result"]
    assert [p["request_id"] for p in pending] == [held["request_id"]]
    assert set(pending[0]) == {"request_id", "tool", "arguments", "acts", "risk", "requested_at", "expires_at", "requested_by"}
    other = run(call(agent, DOSE, {"request": '{"x_m": 0.0, "y_m": 0.0}'}))
    assert other.structured_content["request_id"] != held["request_id"]


def test_an_approved_call_runs_once_and_the_next_one_is_held_again(stack):
    agent, operator = held_pair(stack)
    held = run(call(agent, DOSE, {"request": "{}"})).structured_content
    approved = run(operator_call(operator, "approve_call", {"request_id": held["request_id"]}))
    assert approved["status"] == "approved"
    ran = run(call(agent, DOSE, {"request": "{}"}))
    assert not ran.is_error and json.loads(ran.structured_content["result"])
    after = run(call(agent, DOSE, {"request": "{}"}))
    assert after.structured_content["status"] == "approval_required" and after.structured_content["request_id"] != held["request_id"]
    assert run(operator_call(operator, "list_pending_approvals"))["result"][0]["request_id"] == after.structured_content["request_id"]
    with pytest.raises(ToolError, match="no pending request"):
        run(operator_call(operator, "approve_call", {"request_id": held["request_id"]}))


def test_a_denied_call_says_denied_with_the_reason(stack):
    agent, operator = held_pair(stack)
    held = run(call(agent, DOSE, {"request": "{}"})).structured_content
    run(operator_call(operator, "deny_call", {"request_id": held["request_id"], "reason": "not on this sample"}))
    denied = run(call(agent, DOSE, {"request": "{}"}))
    assert denied.is_error
    assert denied.structured_content == {"status": "denied", "request_id": held["request_id"], "reason": "not on this sample"}
    assert "denied by a person: not on this sample" in denied.content[0].text
    assert run(operator_call(operator, "list_pending_approvals"))["result"] == []


def test_held_calls_and_approvals_expire(tmp_path):
    now = [1_800_000_000.0]
    store = PermissionStore(tmp_path / "permissions.json", "tango://h:1", approval_ttl_s=600, clock=lambda: now[0])
    held = store.request("T", {"a": 1}, True, "high", "me")
    assert seconds_of(held["expires_at"]) == now[0] + 600 and held["requested_at"] == iso_utc(now[0])
    now[0] += 601
    assert store.pending() == []
    renewed = store.request("T", {"a": 1}, True, "high", "me")
    assert renewed["request_id"] != held["request_id"] and renewed["status"] == "pending"
    store.approve(renewed["request_id"])
    now[0] += 601
    assert store.request("T", {"a": 1}, True, "high", "me")["status"] == "pending"
    with pytest.raises(ValueError, match="no pending request"):
        store.approve(renewed["request_id"])


def test_an_approval_is_used_only_by_the_caller_who_asked(tmp_path):
    store = PermissionStore(tmp_path / "permissions.json", "tango://h:1")
    alice = store.request("T", {"a": 1}, True, "high", "alice")
    store.approve(alice["request_id"])
    bob = store.request("T", {"a": 1}, True, "high", "bob")
    assert bob["status"] == "pending" and bob["request_id"] != alice["request_id"]
    assert store.request("T", {"a": 1}, True, "high", "alice")["status"] == "approved"
    assert store.request("T", {"a": 1}, True, "high", "bob")["request_id"] == bob["request_id"]


def test_a_held_call_is_matched_on_its_validated_arguments(stack):
    agent = stack("agent", TWIN, needs_approval={"DigitalTwin": ["set_defocus"]})
    agent.permissions.store.set_tool("list_data", "approve")
    held = run(call(agent, "DigitalTwin_set_defocus", {"defocus": 1})).structured_content
    assert held["status"] == "approval_required" and held["arguments"] == {"defocus": 1.0}
    assert run(call(agent, "DigitalTwin_set_defocus", {"defocus": 1.0})).structured_content["request_id"] == held["request_id"]
    listed = agent.permissions.check("list_data", {}).structured_content
    assert listed["arguments"] == {"kind": None, "search": None, "limit": 20, "offset": 0}
    assert agent.permissions.check("list_data", {"limit": 20}).structured_content["request_id"] == listed["request_id"]
    refused = run(call(agent, "DigitalTwin_set_defocus", {"defocus": "near"}))
    assert refused.is_error and refused.structured_content == {"status": "invalid_arguments", "tool": "DigitalTwin_set_defocus"}
    assert {p["request_id"] for p in agent.permissions.store.pending()} == {held["request_id"], listed["request_id"]}


def test_only_a_declared_stop_is_never_held():
    device = DeviceInfo("Gun", "Gun", hints={"blank_beam": {"role": "beam.blank"}, "halt": {"stop": True}})

    def described(command):
        return describe_command(device, command, overrides={}, routine=None, warnings=[])

    assert described("gun_off").action_role in ("beam.blank", "safe_stop")
    assert [always_runs(described(c)) for c in ("gun_off", "hv_off", "emission_off")] == [False, False, False]
    assert always_runs(described("blank_beam")) and always_runs(described("halt"))
    assert always_runs(describe_native("stop_all")) and not always_runs(describe_native("run_job"))


TUNER = "asyncroscopy/tuner/default"


def test_a_native_tool_can_need_approval_and_an_entry_naming_nothing_is_refused(stack):
    agent = stack("agent", SAMPLE, TUNER, needs_approval={"*": ["run_job"], "SAMPLE": ["load_sample"]})
    assert agent.discovery.endpoint_info()["tools"]["run_job"]["permission"] == "approve"
    plan = {"corrections": ["focus"], "strategy": "image_metric"}
    assert agent.permissions.check("run_job", {"device": "Tuner", "plan": plan}).structured_content["status"] == "approval_required"
    for wrong in ({"SAMPLE": ["get_doze"]}, {"*": ["no_such_tool"]}):
        with pytest.raises(ValueError, match="names no command of its class"):
            stack("agent", SAMPLE, needs_approval=wrong)
    stack("agent", SAMPLE, needs_approval={"CEOSCorrector": ["correct"]})


def test_a_job_that_would_write_the_corrector_waits_for_a_person(stack):
    agent = stack("agent", TUNER)
    operator = stack("operator", TUNER)
    plan = {"corrections": ["focus"], "strategy": "corrector", "dose_budget_e_per_a2": 3000.0}
    held = run(call(agent, "run_job", {"device": "Tuner", "plan": plan}))
    assert held.is_error and held.structured_content["status"] == "approval_required"
    assert held.structured_content["arguments"]["plan"] == plan
    reserve = run(call(agent, "Tuner_reserve", {"request": json.dumps({"plan": plan})}))
    assert reserve.structured_content["status"] == "approval_required"
    focus_only = {**plan, "strategy": "image_metric"}
    assert agent.permissions.check("run_job", {"device": "Tuner", "plan": focus_only}) is None
    assert agent.permissions.check("run_job", {"device": "Tuner", "plan": {"corrections": ["focus"]}}) is None
    run(operator_call(operator, "approve_call", {"request_id": held.structured_content["request_id"]}))
    assert agent.permissions.check("run_job", {"device": "Tuner", "plan": plan}) is None
    assert agent.permissions.check("run_job", {"device": "Tuner", "plan": plan}).structured_content["status"] == "approval_required"


def test_the_monitor_endpoint_holds_reads_as_the_agent_endpoint_does(stack):
    agent, operator = held_pair(stack)
    monitor = stack("monitor", SAMPLE)
    assert not run(call(monitor, "SAMPLE_get_sample")).is_error
    run(operator_call(operator, "set_group_permission", {"group": "reads", "permission": "approve"}))
    held = run(call(monitor, "SAMPLE_get_sample")).structured_content
    assert held["status"] == "approval_required"
    assert [p["request_id"] for p in run(operator_call(operator, "list_pending_approvals"))["result"]] == [held["request_id"]]
    assert not run(call(monitor, "endpoint_info")).is_error


def test_never_cannot_be_set_and_disconnect_is_never_run(stack):
    agent = stack("agent", TWIN)
    operator = stack("operator", TWIN)
    assert "DigitalTwin_Disconnect" not in operator.registered
    info = agent.discovery.endpoint_info()
    assert info["tools"]["DigitalTwin_Disconnect"]["permission"] == "never"
    refused = agent.permissions.check("DigitalTwin_Disconnect", {})
    assert refused.is_error and refused.structured_content == {"status": "never", "tool": "DigitalTwin_Disconnect"}
    with pytest.raises(ToolError, match="allow, approve or default"):
        run(operator_call(operator, "set_permission", {"tool": "DigitalTwin_get_stage", "permission": "never"}))
    with pytest.raises(ToolError, match="cannot be set"):
        run(operator_call(operator, "set_permission", {"tool": "DigitalTwin_Disconnect", "permission": "allow"}))
    with pytest.raises(ToolError, match="allow or approve"):
        run(operator_call(operator, "set_group_permission", {"group": "acts", "permission": "never"}))
    permissions = run(operator_call(operator, "get_permissions"))
    assert permissions["tools"]["DigitalTwin_Disconnect"] == {**permissions["tools"]["DigitalTwin_Disconnect"], "permission": "never", "settable": False}
    assert permissions["tools"]["stop_all"]["settable"] is False and permissions["tools"]["DigitalTwin_blank_beam"]["permission"] == "allow"


def test_settings_round_trip_through_the_permissions_file(stack):
    agent, operator = held_pair(stack)

    def saved():
        return json.loads(Path(stack.permissions_file).read_text(encoding="utf-8"))[agent.stack_id]

    before = run(operator_call(operator, "get_permissions"))["tools"]
    assert (before[DOSE]["permission"], before[DOSE]["source"]) == ("approve", "config")
    assert before["SAMPLE_get_sample"]["source"] == "group" and before["stop_all"]["source"] == "fixed"
    run(operator_call(operator, "set_permission", {"tool": DOSE, "permission": "allow"}))
    assert saved()["tools"] == {DOSE: "allow"}
    assert run(operator_call(operator, "get_permissions"))["tools"][DOSE]["source"] == "tool"
    assert not run(call(agent, DOSE, {"request": "{}"})).is_error
    run(operator_call(operator, "set_permission", {"tool": DOSE, "permission": "default"}))
    assert saved()["tools"] == {}
    assert run(call(agent, DOSE, {"request": "{}"})).structured_content["status"] == "approval_required"
    groups = run(operator_call(operator, "set_group_permission", {"group": "reads", "permission": "approve"}))["groups"]
    assert groups == {"reads": "approve", "acts": "allow"} and saved()["groups"] == {"reads": "approve"}
    assert agent.discovery.endpoint_info()["tools"]["SAMPLE_get_sample"]["permission"] == "approve"
    assert run(call(agent, "SAMPLE_get_sample")).structured_content["status"] == "approval_required"
    assert not run(call(agent, "endpoint_info")).is_error
    run(operator_call(operator, "set_group_permission", {"group": "reads", "permission": "allow"}))
    assert not run(call(agent, "SAMPLE_get_sample")).is_error
    controls = {a["tool"] for a in operator.discovery.action_space()["actions"] if a["device"] is None}
    assert {"list_pending_approvals", "approve_call", "deny_call", "get_permissions", "set_permission", "set_group_permission"} <= controls


def test_endpoint_info_reports_each_permission_and_every_tool_carries_annotations(stack):
    agent, _operator = held_pair(stack)
    info = agent.discovery.endpoint_info()
    assert info["permission_groups"] == {"reads": "allow", "acts": "allow"}
    assert info["tools"][DOSE]["permission"] == "approve" and info["tools"]["SAMPLE_get_sample"]["permission"] == "allow"
    assert all("permission" in entry for entry in info["tools"].values())

    async def listed():
        async with Client(agent.mcp) as client:
            return await client.list_tools()

    tools = {t.name: t for t in run(listed())}
    for name, tool in tools.items():
        acts = info["tools"][name]["acts"]
        assert tool.annotations.read_only_hint is (not acts) and tool.annotations.destructive_hint is acts, name


def test_urls_the_command_timeout_and_device_kinds_are_reported_and_the_old_keys_are_gone(stack):
    urls = {"operator": "http://127.0.0.1:8001/mcp", "analysis": "http://127.0.0.1:8003/mcp"}
    agent = stack("agent", TWIN, SAMPLE, urls=urls)
    info = agent.discovery.endpoint_info()
    assert info["urls"] == {"operator": urls["operator"], "agent_server": None, "workflow_operator": None, "tiled": None,
                            "analysis": urls["analysis"]}
    assert info["command_timeout_s"] == 30.0
    assert info["device_kinds"] == {"DigitalTwin": ["stage", "stem"], "SAMPLE": []}
    assert not {"analysis_url", "stem_devices", "map_devices"} & set(info)
    with pytest.raises(ValueError, match="unknown"):
        stack("agent", SAMPLE, urls={"analysis_url": "x"})


def dose_workflow() -> Workflow:
    def build(use):
        graph = StateGraph(WorkflowState)
        graph.add_node("dose", use.block("run_tool", tool=DOSE, arguments={"request": "{}"}))
        graph.add_edge(START, "dose")
        graph.add_edge("dose", END)
        return graph

    return Workflow(name="held_dose", summary="Read the dose once, through a tool a person must approve.", inputs={},
                    budget={"max_tool_calls": 20}, build=build)


def test_a_workflow_pauses_on_a_held_call_and_resumes_once_it_is_approved(stack, served, tmp_path):
    agent, operator = held_pair(stack)
    endpoint = McpServer(served(agent), {CALLER_HEADER: "workflow-test", RUN_HEADER: "held1"})
    workflow = dose_workflow()
    record = run(run_workflow(workflow, {}, instrument=endpoint, state_dir=tmp_path, run_id="held1"))
    assert record["state"] == "waiting", record
    [held] = record["approvals"]
    assert held["tool"] == DOSE and held["status"] == "approval_required"
    again = run(resume_workflow(record["run_id"], instrument=endpoint, state_dir=tmp_path, find=lambda name: workflow))
    assert again["state"] == "waiting" and again["approvals"][0]["request_id"] == held["request_id"]
    run(operator_call(operator, "approve_call", {"request_id": held["request_id"]}))
    done = run(resume_workflow(record["run_id"], instrument=endpoint, state_dir=tmp_path, find=lambda name: workflow))
    assert done["state"] == "done", done
    assert isinstance(done["outputs"]["dose"]["result"], dict)
    assert not done.get("approvals")


def test_the_agent_server_reports_its_role_and_every_workflow_tool_as_a_service():
    from asyncroscopy.mcp.agent_mcp_server import AgentMCPServer

    server = AgentMCPServer("agents", [], verbose=False, urls={"workflow_operator": "http://127.0.0.1:8004/mcp"})
    info = server.endpoint_info()
    assert info["role"] == "agent_server" and info["urls"]["workflow_operator"] == "http://127.0.0.1:8004/mcp"
    offered = {fn.__name__ for fn in server.workflows.tools()}
    assert set(info["services"]["workflows"].values()) == offered
    assert info["services"]["workflows"]["list_runs"] == "list_runs" and "skills" not in info["services"]


async def _three_reads(ctx, args, previous):
    sample = await ctx.call("SAMPLE_get_sample")
    here = await ctx.call(DOSE, {"request": "{}"})
    there = await ctx.call(DOSE, {"request": '{"x_m":0.0,"y_m":0.0}'})
    return {"sample": sample, "here": here, "there": there}


three_reads = register_block(Block(name="test_three_reads", summary="Three reads in one node, two of them held.", run=_three_reads,
                                   outputs={"sample": "any", "here": "any", "there": "any"}, tools=("SAMPLE_get_sample", DOSE)))


def three_reads_workflow() -> Workflow:
    def build(use):
        graph = StateGraph(WorkflowState)
        graph.add_node("reads", use.block("test_three_reads"))
        graph.add_edge(START, "reads")
        graph.add_edge("reads", END)
        return graph

    return Workflow(name="held_three_reads", summary="One node with an allowed call and two held ones.", inputs={},
                    budget={"max_tool_calls": 20}, build=build)


def test_a_resumed_node_sends_each_call_once_and_each_approval_is_used_once(stack, served, tmp_path):
    log = tmp_path / "tools.jsonl"
    agent = stack("agent", SAMPLE, needs_approval={"SAMPLE": ["get_dose"]}, tool_log_path=str(log))
    operator = stack("operator", SAMPLE)
    address = served(agent)
    workflow = three_reads_workflow()
    journal = tmp_path / "runs" / "three1.journal.json"

    def endpoint():
        return McpServer(address, {CALLER_HEADER: "workflow-test", RUN_HEADER: "three1"})

    record = run(run_workflow(workflow, {}, instrument=endpoint(), state_dir=tmp_path, run_id="three1"))
    held = []
    for _ in range(2):
        assert record["state"] == "waiting", record
        [waiting] = record["approvals"]
        held.append(waiting)
        run(operator_call(operator, "approve_call", {"request_id": waiting["request_id"]}))
        record = run(resume_workflow("three1", instrument=endpoint(), state_dir=tmp_path, find=lambda name: workflow))
    assert record["state"] == "done", record
    assert [h["arguments"]["request"] for h in held] == ["{}", '{"x_m":0.0,"y_m":0.0}']
    ran = [(row["tool"], json.dumps(row["arguments"], sort_keys=True)) for row in read_tool_log(log)]
    assert sorted(ran) == sorted([("SAMPLE_get_sample", "{}"), (DOSE, json.dumps({"request": "{}"})),
                                  (DOSE, json.dumps({"request": '{"x_m":0.0,"y_m":0.0}'}))]), ran
    assert run(operator_call(operator, "list_pending_approvals"))["result"] == []
    assert not journal.exists()


operator_process = """
import asyncio, json, sys
import tango
import asyncroscopy.mcp.mcp_server as mcp_server
from fastmcp import Client

access, permissions_file, request_id = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
mcp_server.DeviceProxy = lambda name: tango.DeviceProxy(access[name])
database = type("Db", (), {"get_device_exported": lambda self, p: type("E", (), {"value_string": list(access)})()})()
server = mcp_server.MCPServer("perm-test", "localhost", 1, blocked_classes=[], needs_approval={}, data_device_address="asyncroscopy/data/default",
                              verbose=False, endpoint_role="operator", database=database, permissions_file=permissions_file)
server.setup(print_summary=False)

async def main():
    async with Client(server.mcp) as client:
        pending = (await client.call_tool("list_pending_approvals", {})).structured_content["result"]
        approved = (await client.call_tool("approve_call", {"request_id": request_id})).structured_content
        print(json.dumps({"pending": [p["request_id"] for p in pending], "status": approved["status"]}))

asyncio.run(main())
"""


def test_an_operator_endpoint_in_another_process_approves_a_call_this_one_holds(stack, tango_ctx):
    agent = stack("agent", SAMPLE, needs_approval={"SAMPLE": ["get_dose"]})
    held = run(call(agent, DOSE, {"request": "{}"})).structured_content
    assert held["status"] == "approval_required"
    access = {name: tango_ctx.get_device_access(name) for name in (SAMPLE,)}
    done = subprocess.run([sys.executable, "-c", operator_process, json.dumps(access), str(stack.permissions_file), held["request_id"]],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    reply = json.loads(done.stdout.strip().splitlines()[-1])
    assert reply == {"pending": [held["request_id"]], "status": "approved"}
    ran = run(call(agent, DOSE, {"request": "{}"}))
    assert not ran.is_error, ran
    assert run(call(agent, DOSE, {"request": "{}"})).structured_content["status"] == "approval_required"
