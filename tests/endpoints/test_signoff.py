"""Twin to instrument (docs/MCP/twin_to_instrument.md): a hardware run of a signed-off workflow gets the calls its
rehearsal made pre-approved, for that run only. Real MCP endpoints over the session's Tango devices."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastmcp import Client

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.mcp_tools import McpServer
from asyncroscopy.core.names import CALLER_HEADER, RUN_HEADER, RUN_KEY_HEADER
from asyncroscopy.safety.permissions import PermissionStore, iso_utc, max_run_preapproval_s, run_key_sha256, seconds_of
from asyncroscopy.workflows import Library
from asyncroscopy.workflows.server import Endpoints, WorkflowServer

DOSE = "SAMPLE_get_dose"
SAMPLE_READ = "SAMPLE_get_sample"
SAMPLE_READ_ALLOWED = "SAMPLE_State"
NEVER = "DigitalTwin_Disconnect"
SAMPLE = "asyncroscopy/sample/mapping"
TWIN = "asyncroscopy/digitaltwin/default"
HERE = {"request": "{}"}
THERE = {"request": '{"x_m":0.0,"y_m":0.0}'}
SIGNOFF = "workflow_signoff:two_doses@abc"


def run(coro):
    return asyncio.run(coro)


async def operator_call(server, name, arguments=None):
    async with Client(server.mcp) as client:
        return (await client.call_tool(name, arguments or {})).structured_content


def hardware(stack, served, **extra):
    """An instrument that does not say it is simulated: its agent endpoint holds both SAMPLE reads, and its operator endpoint."""
    agent = stack("agent", SAMPLE, TWIN, needs_approval={"SAMPLE": ["get_dose", "get_sample"]}, port=2, **extra)
    agent._simulated = False
    return agent, served(agent), stack("operator", SAMPLE, port=2)


def as_run(address, run_id, key):
    return McpServer(address, {CALLER_HEADER: "agent-workflow", RUN_HEADER: run_id, RUN_KEY_HEADER: key})


def grant(operator, run_id, key, calls):
    return run(operator_call(operator, "preapprove_run", {
        "run_id": run_id, "run_key_sha256": run_key_sha256(key), "calls": calls, "approved_by": SIGNOFF, "approver": "Sam Reed",
        "workflow": "two_doses", "hash": "abc", "rehearsal_run_id": "rehearsal1"}))


def test_a_rehearsed_call_runs_and_any_other_call_of_the_run_is_held_with_the_reason(stack, served, tmp_path):
    trace = tmp_path / "trace.jsonl"
    _, address, operator = hardware(stack, served, trace_path=str(trace))
    granted = grant(operator, "hw1", "key1", [{"tool": DOSE, "arguments": HERE, "count": 2}, {"tool": NEVER, "arguments": {}}])
    assert granted["calls"] == 3 and granted["approved_by"] == SIGNOFF and granted["approver"] == "Sam Reed" and granted["expires_at"]
    endpoint = as_run(address, "hw1", "key1")

    ran = run(endpoint.call(DOSE, HERE))
    assert not ran.is_error, ran
    again = run(endpoint.call(DOSE, dict(HERE, x_m=None)))
    assert not again.is_error, "the same call, validated, is the same call"

    other = run(endpoint.call(DOSE, THERE))
    assert other.is_error and other.structured_content["status"] == "approval_required"
    assert other.structured_content["signoff"] == {"approved_by": SIGNOFF, "approver": "Sam Reed", "run_id": "hw1",
                                                  "why_held": "the rehearsal called this tool with other arguments"}
    unrehearsed = run(endpoint.call(SAMPLE_READ, {}))
    assert unrehearsed.structured_content["status"] == "approval_required"
    assert unrehearsed.structured_content["signoff"]["why_held"] == "the rehearsal never called this tool"
    never = run(endpoint.call(NEVER, {}))
    assert never.is_error and never.structured_content == {"status": "never", "tool": NEVER}

    pending = run(operator_call(operator, "list_pending_approvals"))["result"]
    assert sorted(p["tool"] for p in pending) == sorted([DOSE, SAMPLE_READ])
    assert all(p["signoff"]["approved_by"] == SIGNOFF for p in pending)

    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    passed = [e for e in events if e["action"] == DOSE and e.get("ok")]
    assert len(passed) == 2 and all(e["approved_by"] == SIGNOFF and e["approver"] == "Sam Reed" and e["run"] == "hw1" for e in passed)


def test_a_pre_approval_is_the_runs_own_and_ends_with_it(stack, served):
    _, address, operator = hardware(stack, served)
    grant(operator, "hw1", "key1", [{"tool": DOSE, "arguments": HERE}])
    assert not run(as_run(address, "hw1", "key1").call(DOSE, HERE)).is_error
    held = {"second run": as_run(address, "hw2", "key2"), "the run id without its key": as_run(address, "hw1", "key2"),
            "no run": McpServer(address, {CALLER_HEADER: "agent"})}
    for why, endpoint in held.items():
        assert run(endpoint.call(DOSE, HERE)).structured_content["status"] == "approval_required", why

    removed = run(operator_call(operator, "end_run_preapproval", {"run_id": "hw1"}))
    assert removed["removed"]["approved_by"] == SIGNOFF and removed["removed"]["calls"] == 1
    assert run(as_run(address, "hw1", "key1").call(DOSE, HERE)).structured_content["status"] == "approval_required"
    assert run(operator_call(operator, "end_run_preapproval", {"run_id": "hw1"}))["removed"] is None


def test_a_pre_approval_expires_after_the_approval_ttl(stack, served):
    agent, address, operator = hardware(stack, served, approval_ttl_s=4.0)
    grant(operator, "hw1", "key1", [{"tool": DOSE, "arguments": HERE}])
    endpoint = as_run(address, "hw1", "key1")
    assert not run(endpoint.call(DOSE, HERE)).is_error
    time.sleep(4.2)
    assert run(endpoint.call(DOSE, HERE)).structured_content["status"] == "approval_required"
    assert agent.permissions.store.run_grant("hw1") is None


TWO_DOSES = '''"""Read the dose here, then at a point the inputs give."""

from langgraph.graph import END, START, StateGraph

from asyncroscopy.workflows import Steps, Workflow, WorkflowState

INPUTS = {
    "where": {"type": "string", "default": '{"x_m":0.0,"y_m":0.0}', "doc": "The point, as the dose tool's request."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)
    graph.add_node("here", use.block("run_tool", tool="SAMPLE_get_dose", arguments={"request": "{}"}))
    graph.add_node("there", use.block("run_tool", tool="SAMPLE_get_dose", arguments={"request": "{inputs.where}"}))
    graph.add_edge(START, "here")
    graph.add_edge("here", "there")
    graph.add_edge("there", END)
    return graph


WORKFLOW = Workflow(
    name="two_doses",
    summary="Read the sample's dose at the current position, then at a point the inputs give.",
    inputs=INPUTS,
    budget={"max_tool_calls": 20},
    build=build,
)
'''
ASKED = "read the dose here and then at the centre, the same way on the real microscope"


def test_propose_rehearse_review_approve_and_run_on_the_paired_instrument(stack, served, tmp_path):
    """The whole flow: the twin and the instrument are two stacks; the workflow server acts on both, by URL."""
    twin_operator = stack("operator", SAMPLE, TWIN, port=1)
    twin = stack("agent", SAMPLE, urls={"operator": served(twin_operator)}, port=1)
    twin._simulated = True
    twin_url = served(twin)
    scope_operator = stack("operator", SAMPLE, port=2)
    scope = stack("agent", SAMPLE, needs_approval={"SAMPLE": ["get_dose"]}, urls={"operator": served(scope_operator)}, port=2)
    scope._simulated = False
    scope_url = served(scope)

    library = Library(tmp_path / "library")
    agents = WorkflowServer(Endpoints(twin_url, caller="agent-workflow", instruments=[scope_url]), state_dir=tmp_path / "state",
                            library=library, hardware=True)
    person = WorkflowServer(Endpoints(twin_url, caller="workflow-operator", instruments=[scope_url]), state_dir=tmp_path / "state",
                            library=library, name="Workflow_Operator_MCP")
    review, approve, _ = person.operator_tools()

    async def as_the_agent(tool, **arguments):
        async with Client(agents.mcp) as client:
            result = await client.call_tool(tool, arguments)
            return result.structured_content

    proposed = run(as_the_agent("propose_workflow", user_request=ASKED, source=TWO_DOSES, context="The user wants the twin's way repeated."))
    assert proposed["ok"], proposed
    elsewhere = run(as_the_agent("run_workflow", name="two_doses", instrument_url="http://127.0.0.1:1/mcp"))
    assert elsewhere["state"] == "refused" and "not an instrument endpoint this workflow server acts on" in elsewhere["problems"][0]
    drafted = run(as_the_agent("run_workflow", name="two_doses", instrument_url=scope_url))
    assert drafted["state"] == "refused" and "nobody has approved it" in drafted["problems"][0]

    rehearsed = run(as_the_agent("run_workflow", name="two_doses", wait_s=60))
    assert rehearsed["state"] == "done", rehearsed
    assert rehearsed["instrument_url"] == twin_url and rehearsed["results"]["instrument_calls"] == 2
    assert "twin_score" not in json.dumps(rehearsed) and "dose" not in rehearsed["results"], "an agent never sees the twin's truth"

    card = run(review(name="two_doses", instrument_url=scope_url))
    assert card["ok"] and card["can_approve"], card.get("why_not")
    assert card["instrument_url"] == scope_url and card["stack_fingerprint"]["simulated"]["value"] is False
    assert card["context_as_reported_by_agent"] == {"text": "The user wants the twin's way repeated.", "this_code": True}
    assert card["last_approved"] is None and card["diff_from_last_approved"] is None and card["source"] == TWO_DOSES
    shown = card["rehearsal"]
    assert shown["run_id"] == rehearsed["run_id"] and shown["stack_fingerprint"]["simulated"]["value"] is True
    assert [(c["tool"], c["arguments"]) for c in shown["calls_made"]] == [(DOSE, HERE), (DOSE, THERE)]
    score = shown["results"]["twin_score"]
    assert score["label"].startswith("twin score") and score["known"] and score["images"] == 0 and score["particles_total"] > 0
    assert shown["results"]["instrument_s"] is not None and shown["results"]["dose"]["from"] == "the twin's dose map"

    approved = run(approve(name="two_doses", approver="Sam Reed", expected_hash=card["hash"], instrument_url=scope_url))
    assert approved["ok"] and approved["approval"]["rehearsed"]["run_id"] == rehearsed["run_id"], approved

    on_scope = run(as_the_agent("run_workflow", name="two_doses", instrument_url=scope_url, wait_s=60))
    assert on_scope["state"] == "done", on_scope
    signed = f"workflow_signoff:two_doses@{card['hash']}"
    assert on_scope["preapproval"]["granted"] and on_scope["preapproval"]["approved_by"] == signed and on_scope["preapproval"]["approver"] == "Sam Reed"
    assert on_scope["results"]["instrument_calls"] == 2 and not on_scope.get("approvals")
    assert scope.permissions.store.run_grant(on_scope["run_id"]) is None, "the pre-approval ends with the run"

    moved = run(as_the_agent("run_workflow", name="two_doses", instrument_url=scope_url, inputs={"where": '{"x_m":1e-07,"y_m":0.0}'}, wait_s=60))
    assert moved["state"] == "waiting", moved
    (held,) = moved["approvals"]
    assert held["arguments"]["request"] == '{"x_m":1e-07,"y_m":0.0}' and held["signoff"]["approved_by"] == signed
    assert held["signoff"]["why_held"] == "the rehearsal called this tool with other arguments"
    assert moved["tools_called"][DOSE]["ok"] == 1, "the rehearsed call ran before the changed one was held"
    assert scope.permissions.store.run_grant(moved["run_id"]) is not None, "a waiting run keeps its pre-approval"
    stopped = run(as_the_agent("stop_workflow", run_id=moved["run_id"]))
    assert stopped["state"] == "stopped" and scope.permissions.store.run_grant(moved["run_id"]) is None


def test_each_rehearsed_call_passes_as_many_times_as_the_rehearsal_made_it(stack, served):
    _, address, operator = hardware(stack, served)
    grant(operator, "hw1", "key1", [{"tool": DOSE, "arguments": HERE, "count": 2}, {"tool": DOSE, "arguments": dict(HERE, x_m=None)}])
    endpoint = as_run(address, "hw1", "key1")
    for _ in range(3):
        assert not run(endpoint.call(DOSE, HERE)).is_error, "two rehearsed calls that validate the same add up"
    fourth = run(endpoint.call(DOSE, HERE)).structured_content
    assert fourth["status"] == "approval_required" and fourth["signoff"]["why_held"] == "the rehearsal made this call fewer times"
    grant(operator, "hw1", "key2", [{"tool": DOSE, "arguments": HERE, "count": 3}])
    assert run(as_run(address, "hw1", "key2").call(DOSE, HERE)).structured_content["status"] == "approval_required", \
        "a resumed run's new grant does not give back what the run already used"


def test_a_persons_approval_of_a_held_signed_off_call_is_used_by_that_run_only(stack, served):
    _, address, operator = hardware(stack, served)
    grant(operator, "hw1", "key1", [{"tool": DOSE, "arguments": HERE}])
    held = run(as_run(address, "hw1", "key1").call(DOSE, THERE)).structured_content
    assert held["status"] == "approval_required"
    run(operator_call(operator, "approve_call", {"request_id": held["request_id"]}))
    for why, endpoint in {"another run": as_run(address, "hw2", "key1"), "the run id with another key": as_run(address, "hw1", "other"),
                          "no run": McpServer(address, {CALLER_HEADER: "agent-workflow"})}.items():
        assert run(endpoint.call(DOSE, THERE)).structured_content["status"] == "approval_required", why
    grant(operator, "hw1", "key2", [{"tool": DOSE, "arguments": HERE}])
    assert not run(as_run(address, "hw1", "key2").call(DOSE, THERE)).is_error, "the resumed run, with its new key, uses it"
    assert run(as_run(address, "hw1", "key2").call(DOSE, THERE)).structured_content["status"] == "approval_required", "once"


def test_a_pre_approval_stays_alive_while_the_run_makes_calls(stack, served):
    _, address, operator = hardware(stack, served, approval_ttl_s=4.0)
    grant(operator, "hw1", "key1", [{"tool": DOSE, "arguments": HERE, "count": 5}])
    endpoint = as_run(address, "hw1", "key1")
    for _ in range(3):
        assert not run(endpoint.call(SAMPLE_READ_ALLOWED, {})).is_error
        time.sleep(2.2)
    assert not run(endpoint.call(DOSE, HERE)).is_error, "6.6 s after a 4 s grant, kept alive by the run's calls"


def test_a_kept_alive_pre_approval_ends_at_the_overall_maximum(tmp_path):
    now = [1_800_000_000.0]
    store = PermissionStore(tmp_path / "permissions.json", "tango://h:2", approval_ttl_s=600, clock=lambda: now[0])
    store.grant_run("hw1", run_key_sha256("k"), [{"tool": DOSE, "arguments": {}}], SIGNOFF, "Sam", "two_doses", "abc")
    assert store.live_grant("hw1", run_key_sha256("other")) is None
    end = now[0] + max_run_preapproval_s
    while now[0] < end - 400:
        now[0] += 400
        assert store.live_grant("hw1", run_key_sha256("k")) is not None
        assert seconds_of(store.run_grant("hw1")["expires_at"]) <= end
    now[0] = end + 1
    assert store.live_grant("hw1", run_key_sha256("k")) is None


def test_the_store_keeps_one_pre_approval_per_run_until_it_expires(tmp_path):
    now = [1_800_000_000.0]
    store = PermissionStore(tmp_path / "permissions.json", "tango://h:2", approval_ttl_s=600, clock=lambda: now[0])
    with pytest.raises(ValueError, match="names the sign-off"):
        store.grant_run("hw1", run_key_sha256("k"), [], "a person", "Sam", "two_doses", "abc")
    first = store.grant_run("hw1", run_key_sha256("k"), [{"tool": DOSE, "arguments": {"b": 1, "a": 2}}], SIGNOFF, "Sam", "two_doses", "abc")
    assert first["calls"] == 1 and first["expires_at"] == iso_utc(now[0] + 600)
    assert store.run_grant("hw1")["calls"] == [{"tool": DOSE, "arguments": {"a": 2, "b": 1}, "count": 1}]
    store.grant_run("hw1", run_key_sha256("k2"), [], SIGNOFF, "Sam", "two_doses", "abc")
    assert store.run_grant("hw1")["calls"] == [] and store.run_grant("hw1")["run_key_sha256"] == run_key_sha256("k2")
    now[0] += 601
    assert store.run_grant("hw1") is None and store.end_run("hw1") is None


def test_the_run_key_goes_to_the_instrument_only_never_to_the_analysis_server():
    config = AgentConfig(mcp_url="http://127.0.0.1:1/mcp", analysis_url="http://127.0.0.1:2/mcp", run_key="secret")
    assert config.instrument().headers[RUN_KEY_HEADER] == "secret"
    analysis = config.analysis().headers
    assert RUN_KEY_HEADER not in analysis and analysis[RUN_HEADER] == config.run_id
