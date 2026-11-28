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


SET_FOV_AND_DOSE = '''"""Set the field of view the inputs give, read the dose, and set the field of view again."""

from langgraph.graph import END, START, StateGraph

from asyncroscopy.workflows import Steps, Workflow, WorkflowState

INPUTS = {
    "fov_m": {"type": "number", "default": 1e-06, "min": 1e-08, "max": 1e-05, "doc": "The field of view, m."},
}


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)
    graph.add_node("zoom", use.block("run_tool", tool="*_set_fov", arguments={"fov": "{inputs.fov_m}"}))
    graph.add_node("dose", use.block("run_tool", tool="SAMPLE_get_dose", arguments={"request": "{}"}))
    graph.add_node("again", use.block("run_tool", tool="*_set_fov", arguments={"fov": "{inputs.fov_m}"}))
    graph.add_edge(START, "zoom")
    graph.add_edge("zoom", "dose")
    graph.add_edge("dose", "again")
    graph.add_edge("again", END)
    return graph


WORKFLOW = Workflow(
    name="fov_and_dose",
    summary="Set the field of view, read the sample's dose, set the field of view again.",
    inputs=INPUTS,
    budget={"max_tool_calls": 20},
    build=build,
)
'''
ASKED = "set the field of view, read the dose, and do it again the same way on the real microscope"
TWIN_SCOPE = "asyncroscopy/digitaltwin/mapping"
OTHER_SCOPE = "asyncroscopy/digitaltwinparticles/default"


def test_propose_rehearse_review_approve_and_run_on_a_paired_instrument_of_another_device_class(stack, served, tmp_path):
    """The twin is a DigitalTwin, the instrument a DigitalTwinParticles: the same commands under other tool names."""
    twin_operator = stack("operator", SAMPLE, TWIN_SCOPE, port=1)
    twin = stack("agent", SAMPLE, TWIN_SCOPE, urls={"operator": served(twin_operator)}, port=1)
    twin._simulated = True
    twin_url = served(twin)
    scope_operator = stack("operator", SAMPLE, OTHER_SCOPE, port=2)
    scope = stack("agent", SAMPLE, OTHER_SCOPE, needs_approval={"SAMPLE": ["get_dose"], "DigitalTwinParticles": ["set_fov"]},
                  urls={"operator": served(scope_operator)}, port=2)
    scope._simulated = False
    scope_url = served(scope)
    twofold = stack("agent", "asyncroscopy/sample/mapping", "asyncroscopy/sample/sem", OTHER_SCOPE, port=3)
    twofold._simulated = False
    twofold_url = served(twofold)
    assert "DigitalTwin_set_fov" in twin.registered and "DigitalTwinParticles_set_fov" in scope.registered

    library = Library(tmp_path / "library")
    agents = WorkflowServer(Endpoints(twin_url, caller="agent-workflow", instruments=[scope_url, twofold_url]), state_dir=tmp_path / "state",
                            library=library, hardware=True)
    person = WorkflowServer(Endpoints(twin_url, caller="workflow-operator", instruments=[scope_url, twofold_url]), state_dir=tmp_path / "state",
                            library=library, name="Workflow_Operator_MCP")
    review, approve, _ = person.operator_tools()

    async def as_the_agent(tool, **arguments):
        async with Client(agents.mcp) as client:
            result = await client.call_tool(tool, arguments)
            return result.structured_content

    assert run(as_the_agent("propose_workflow", user_request=ASKED, source=SET_FOV_AND_DOSE))["ok"]
    first = run(as_the_agent("run_workflow", name="fov_and_dose", wait_s=60))
    assert first["state"] == "done", first

    card = run(review(name="fov_and_dose", instrument_url=scope_url))
    assert card["can_approve"] and card["rehearsal"]["run_id"] == first["run_id"], card.get("why_not")
    assert [(c["tool"], c["count"], c["instrument_tool"], c["held_because"]) for c in card["rehearsal"]["calls_made"]] == [
        ("DigitalTwin_set_fov", 2, "DigitalTwinParticles_set_fov", None), (DOSE, 1, DOSE, None)]
    elsewhere = run(review(name="fov_and_dose", instrument_url=twofold_url))["rehearsal"]["calls_made"]
    assert elsewhere[0]["instrument_tool"] == "DigitalTwinParticles_set_fov"
    assert elsewhere[1]["instrument_tool"] is None and elsewhere[1]["held_because"].startswith("no match: will be held (2 instrument tools match")

    second = run(as_the_agent("run_workflow", name="fov_and_dose", wait_s=60))
    assert second["state"] == "done", second
    stale = run(approve(name="fov_and_dose", approver="Sam Reed", expected_hash=card["hash"], rehearsal_run_id=first["run_id"], instrument_url=scope_url))
    assert not stale["ok"] and second["run_id"] in stale["error"], "a rehearsal the person did not see is not the one approved"
    card = run(review(name="fov_and_dose", instrument_url=scope_url))
    approved = run(approve(name="fov_and_dose", approver="Sam Reed", expected_hash=card["hash"], rehearsal_run_id=card["rehearsal"]["run_id"],
                           instrument_url=scope_url))
    assert approved["ok"], approved
    rehearsed = approved["approval"]["rehearsed"]
    assert rehearsed["run_id"] == second["run_id"] and rehearsed["calls_made"] == card["rehearsal"]["calls_made"]
    assert rehearsed["inputs"] == card["rehearsal"]["inputs"] == {"fov_m": 1e-06}

    (tmp_path / "library" / "rehearsals.json").write_text("{}", encoding="utf-8")
    on_scope = run(as_the_agent("run_workflow", name="fov_and_dose", instrument_url=scope_url, inputs=card["rehearsal"]["inputs"], wait_s=60))
    assert on_scope["state"] == "done", on_scope
    assert on_scope["preapproval"]["granted"] and on_scope["preapproval"]["calls"] == 3, "the approval keeps its own copy of the rehearsal"
    assert on_scope["tools_called"]["DigitalTwinParticles_set_fov"]["ok"] == 2 and not on_scope.get("approvals")

    other = run(as_the_agent("run_workflow", name="fov_and_dose", instrument_url=scope_url, inputs={"fov_m": 2e-06}, wait_s=60))
    assert other["state"] == "refused" and "signed off for the inputs its rehearsal ran with" in other["problems"][0], other

    approvals = json.loads((tmp_path / "library" / "approvals.json").read_text(encoding="utf-8"))
    approvals["fov_and_dose"]["rehearsed"]["calls_made"][1]["instrument_tool"] = "SAMPLE_get_sample"
    (tmp_path / "library" / "approvals.json").write_text(json.dumps(approvals), encoding="utf-8")
    changed = run(as_the_agent("run_workflow", name="fov_and_dose", instrument_url=scope_url, wait_s=60))
    assert changed["state"] == "waiting", changed
    (held,) = changed["approvals"]
    assert held["tool"] == DOSE and held["signoff"]["why_held"] == "the rehearsal never called this tool"
    assert changed["preapproval"]["calls"] == 2, "a call whose instrument tool is not the one the approval recorded is not pre-approved"
    assert scope.permissions.store.run_grant(changed["run_id"]) is not None, "a waiting run keeps its pre-approval"
    stopped = run(as_the_agent("stop_workflow", run_id=changed["run_id"]))
    assert stopped["state"] == "stopped" and scope.permissions.store.run_grant(changed["run_id"]) is None


def test_an_approval_binds_the_passing_entry_of_a_rehearsal_that_was_stopped_and_resumed(tmp_path):
    library = Library(tmp_path / "library")
    assert library.propose(SET_FOV_AND_DOSE, request=ASKED)["ok"]
    workflow = library.workflow("fov_and_dose")
    fingerprint = {"digest": "twin-1", "software": {"commit": "v1"}, "simulated": {"value": True}}
    info = {"stack_id": "tango://twin:1", "stack_fingerprint": fingerprint,
            "tools": {DOSE: {"device": "SAMPLE", "command": "get_dose", "role": "setting.read"}}, "device_kinds": {"SAMPLE": []}}
    partial = {"run_id": "r1", "hash": workflow.hash(), "state": "stopped", "inputs": {"fov_m": 1e-06},
               "calls_made": [{"server": "instrument", "tool": DOSE, "arguments": HERE, "count": 1}]}
    library.record_rehearsal(workflow, partial, info)
    library.record_rehearsal(workflow, {**partial, "state": "done", "calls_made": [{"server": "instrument", "tool": DOSE, "arguments": HERE, "count": 2}]}, info)
    record = library.approve("fov_and_dose", "Sam Reed", require_rehearsal=True, instrument_info=info, rehearsal_run_id="r1")
    assert record["rehearsed"]["calls_made"][0]["count"] == 2 and record["rehearsed"]["calls_made"][0]["instrument_tool"] == DOSE
    with pytest.raises(ValueError, match="rehearsal_run_id"):
        library.approve("fov_and_dose", "Sam Reed", require_rehearsal=True, instrument_info=info)


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
