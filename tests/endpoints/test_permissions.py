"""Permissions (docs/MCP/permissions.md): the agent endpoint holds calls of an ``approve`` tool, the operator endpoint
decides them, and the settings live in the permissions file. Real MCP servers over the session's Tango devices."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import pytest
import tango
import uvicorn
from fastmcp import Client
from fastmcp.exceptions import ToolError
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.mcp_tools import McpServer
from asyncroscopy.core.names import CALLER_HEADER, RUN_HEADER
from asyncroscopy.mcp.mcp_server import MCPServer
from asyncroscopy.safety.permissions import PermissionStore, iso_utc, seconds_of
from asyncroscopy.workflows.graph import Workflow, WorkflowState
from asyncroscopy.workflows.runner import resume_workflow, run_workflow

DOSE = "SAMPLE_get_dose"
SAMPLE = "asyncroscopy/sample/mapping"
TWIN = "asyncroscopy/digitaltwin/default"


@pytest.fixture
def stack(monkeypatch, tango_ctx, tmp_path):
    """make(role, *devices, needs_approval=...) -> an MCPServer on the session devices, sharing one permissions file."""
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", lambda name: tango.DeviceProxy(tango_ctx.get_device_access(name)))
    permissions_file = tmp_path / "permissions.json"

    def make(role, *names, needs_approval=None, urls=None):
        database = type("Db", (), {"get_device_exported": lambda self, p: type("E", (), {"value_string": list(names)})()})()
        server = MCPServer("perm-test", "localhost", 1, blocked_classes=[], needs_approval=needs_approval or {},
                           data_device_address="asyncroscopy/data/default", verbose=False, endpoint_role=role, database=database,
                           permissions_file=str(permissions_file), urls=urls)
        server.setup(print_summary=False)
        return server

    make.permissions_file = permissions_file
    return make


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
    assert held["status"] == "approval_required" and held["tool"] == DOSE and held["arguments"] == {"request": "{}"}
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


@pytest.fixture
def served():
    """serve(server) -> its HTTP address, from uvicorn in a thread on a free 127.0.0.1 port."""
    running = []

    def serve(server):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        web = uvicorn.Server(uvicorn.Config(server.mcp.http_app(), host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=web.run, daemon=True)
        thread.start()
        running.append((web, thread))
        deadline = time.monotonic() + 20
        while not web.started:
            assert time.monotonic() < deadline, "the test MCP server did not start"
            time.sleep(0.05)
        return f"http://127.0.0.1:{port}/mcp"

    yield serve
    for web, thread in running:
        web.should_exit = True
        thread.join(10)


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
