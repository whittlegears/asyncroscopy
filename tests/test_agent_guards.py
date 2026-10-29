"""Context guard, repeat guard and in-tool job waits (no model, no stack)."""

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from asyncroscopy.agents.guards import looping, JobWaitMiddleware, context_chars, fit_messages, repeats
from asyncroscopy.agents.skill_middleware import SkillMiddleware

TOOLS = {"run_job", "wait_for_job", "view_data", "list_data", "Mapper_reserve", "Mapper_abort", "approval_status", "load_skill"}
REQUEST = "Map a 600 nm square around the current position and take an EDS spectrum with each image."


def round_(i, name="wait_for_job", args=None, reply=None):
    call_id = f"c{i}"
    return [AIMessage("", tool_calls=[{"name": name, "args": args if args is not None else {"job_id": "j1"}, "id": call_id}]),
            ToolMessage(reply if reply is not None else json.dumps({"state": "running", "i": i}), tool_call_id=call_id, name=name)]


def big_done(i):
    spectra = [{"index": k, "key": f"spectrum_eds_{i}_{k:05d}.h5", "stage_xy_m": [k * 1e-8, k * 2e-8]} for k in range(120)]
    return json.dumps({"job_id": "j1", "state": "done", "result_keys": {"run": f"map_r{i}.h5", "map": f"map_r{i}_map"}, "summary": {"spectra": spectra}})


def chars(messages):
    return sum(len(m.content if isinstance(m.content, str) else json.dumps(m.content)) + sum(len(json.dumps(c["args"])) for c in getattr(m, "tool_calls", None) or []) + 20 for m in messages)


def test_long_runs_with_big_replies_stay_inside_the_budget_and_keep_the_request():
    messages = [HumanMessage(REQUEST)]
    for i in range(40):
        messages += round_(i, reply=big_done(i))
    budget = 12_000
    fitted = fit_messages(messages, budget)
    assert chars(fitted) <= budget
    assert fitted[0].content == REQUEST  # the request is never dropped
    latest = fitted[-1].content  # the latest reply stays, up to its cap, and says it was cut
    assert latest.startswith(messages[-1].content[:2000]) and latest.endswith("(cut; ask again with less)")
    assert "map_r0_map" not in json.dumps([m.content for m in fitted]) or "earlier reply" in json.dumps([m.content for m in fitted])
    ids = {c["id"] for m in fitted for c in getattr(m, "tool_calls", None) or []}
    assert all(m.tool_call_id in ids for m in fitted if m.type == "tool")  # every reply still has its call


def test_older_replies_become_digests_that_keep_their_keys():
    messages = [HumanMessage(REQUEST), *round_(1, reply=big_done(1)), *round_(2, reply=big_done(2)), HumanMessage("and the map key?")]
    fitted = fit_messages(messages, 100_000)
    older = fitted[2].content
    assert older.startswith("[earlier reply, shortened") and "map_r1_map" in older
    assert fitted[-1].content == "and the map key?"


def test_the_whole_model_call_fits_the_context():
    middleware = SkillMiddleware("BASE " * 50, TOOLS, TOOLS, memory=lambda r: "notes " * 300, num_ctx=8192)
    messages = [HumanMessage(REQUEST)]
    for i in range(30):
        messages += round_(i, reply=big_done(i))
    state = {"messages": messages}
    update = middleware.before_model(state, None)
    tools = [SimpleNamespace(name=n, description="x" * 200, args_schema={"type": "object", "properties": {"a": {"type": "string"}}}) for n in sorted(TOOLS)]
    request = SimpleNamespace(state={**state, **(update or {})}, messages=messages, tools=tools, override=lambda **c: SimpleNamespace(**c))
    out = middleware._prompted(request)
    total = len(out.system_message.content) + chars(out.messages) + sum(len(t.description) + 80 for t in out.tools)
    assert total <= context_chars(8192)
    assert out.messages[0].content == REQUEST


def test_cycling_over_the_same_calls_is_a_loop_but_going_back_once_is_not():
    a, b = {"key": "one.h5"}, {"key": "two.h5"}
    back_once = [HumanMessage(REQUEST), *round_(1, args=a, reply="image one"), *round_(2, args=b, reply="image two"), *round_(3, args=a, reply="image one")]
    assert looping(back_once) == 0
    third = back_once + round_(4, args=b, reply="image two") + round_(5, args=a, reply="image one")
    assert looping(third) == 1
    assert looping(third + round_(6, args=b, reply="image two") + round_(7, args=a, reply="image one")) == 2
    assert looping(back_once + round_(4, args=a, reply="image one, changed")) == 0


def test_protocol_steps_approval_waits_and_earlier_requests_are_not_loops():
    blank = [HumanMessage(REQUEST)]
    for i in range(6):
        blank += round_(2 * i, name="DigitalTwin_blank_beam", args={}, reply="null") + round_(2 * i + 1, name="DigitalTwin_acquire_scanned_image", args={"detector_list": ["haadf"]}, reply=f"image_{i}.h5")
    blank += round_(99, name="DigitalTwin_blank_beam", args={}, reply="null")
    assert looping(blank) == 0
    pending = [HumanMessage(REQUEST)] + [m for i in range(5) for m in round_(i, name="approval_status", args={"request_id": "r1"}, reply="pending")]
    assert looping(pending) == 0
    a = {"key": "one.h5"}
    earlier = [HumanMessage(REQUEST), *round_(1, args=a, reply="image one"), *round_(2, args=a, reply="image one"), HumanMessage("look again")]
    assert looping(earlier + round_(3, args=a, reply="image one")) == 0


def test_a_repeated_call_is_pointed_out_then_the_run_stops_honestly():
    same = {"job_id": "j1"}
    once = [HumanMessage(REQUEST), *round_(1, args=same, reply="state running"), *round_(2, args=same, reply="state running")]
    twice = once + round_(3, args=same, reply="state running")
    assert repeats(once) == 1 and repeats(twice) == 2
    middleware = SkillMiddleware("BASE", TOOLS, TOOLS)
    called = []

    async def handler(request):
        called.append(request)
        return "model reply"

    def request_for(messages):
        return SimpleNamespace(state={"messages": messages}, messages=messages, tools=[], override=lambda **c: SimpleNamespace(**c))

    asyncio.run(middleware.awrap_model_call(request_for(once), handler))
    assert "Do not call it again" in called[-1].system_message.content
    stopped = asyncio.run(middleware.awrap_model_call(request_for(twice), handler))
    assert len(called) == 1 and "I stopped" in stopped.result[0].content and not stopped.result[0].tool_calls


class FakeToolRequest:
    def __init__(self, name, args):
        self.tool_call = {"name": name, "args": args, "id": "call-1", "type": "tool_call"}
        self.tool = SimpleNamespace(name=name)
        self.state, self.runtime = {}, None

    def override(self, **changes):
        copy = FakeToolRequest(changes.get("tool_call", self.tool_call)["name"], changes.get("tool_call", self.tool_call)["args"])
        copy.tool_call = changes.get("tool_call", self.tool_call)
        return copy


def test_a_long_job_costs_one_tool_round(monkeypatch):
    monkeypatch.setattr("asyncroscopy.agents.guards.asyncio.sleep", lambda s: asyncio.sleep(0))
    states = iter(["running"] * 40 + ["done"])
    seen = []

    async def handler(request):
        seen.append(request.tool_call["name"])
        if request.tool_call["name"] == "run_job":
            return ToolMessage(json.dumps({"job_id": "j1", "state": "running", "estimate": {"duration_s": 3600.0}}), tool_call_id="call-1", name="run_job")
        state = next(states)
        reply = {"job_id": "j1", "state": state, "progress": {"done": 3, "total": 9}}
        if state == "done":
            reply["result_keys"] = {"map": "map_r1_map"}
        else:
            reply["note"] = "Still working"
        return ToolMessage(json.dumps(reply), tool_call_id="call-1", name="wait_for_job")

    progress = []
    middleware = JobWaitMiddleware([SimpleNamespace(name="wait_for_job")], on_progress=lambda j, s, t: progress.append(s["state"]))
    final = asyncio.run(middleware.awrap_tool_call(FakeToolRequest("run_job", {"device": "Mapper", "plan": {}}), handler))
    body = json.loads(final.content)
    assert body["state"] == "done" and body["result_keys"] == {"map": "map_r1_map"} and body["estimate"] == {"duration_s": 3600.0}
    assert final.tool_call_id == "call-1" and seen.count("wait_for_job") == 41 and progress[-1] == "done"


def test_waiting_stops_at_its_cap_and_says_the_job_goes_on(monkeypatch):
    monkeypatch.setattr("asyncroscopy.agents.guards.asyncio.sleep", lambda s: asyncio.sleep(0))

    async def handler(request):
        return ToolMessage(json.dumps({"job_id": "j1", "state": "running", "note": "Still working"}), tool_call_id="call-1", name="wait_for_job")

    import time

    middleware = JobWaitMiddleware([SimpleNamespace(name="wait_for_job")], deadline=time.monotonic() + 0.05, on_progress=None)
    final = asyncio.run(middleware.awrap_tool_call(FakeToolRequest("wait_for_job", {"job_id": "j1"}), handler))
    body = json.loads(final.content)
    assert body["state"] == "running" and "the job goes on" in body["note"]


def test_a_wait_that_fails_says_the_job_may_still_run():
    replies = iter([
        ToolMessage(json.dumps({"job_id": "j1", "state": "running"}), tool_call_id="call-1", name="wait_for_job"),
        ToolMessage("Error: the device did not answer", tool_call_id="call-1", name="wait_for_job", status="error"),
    ])

    async def handler(request):
        return next(replies)

    final = asyncio.run(JobWaitMiddleware([SimpleNamespace(name="wait_for_job")], on_progress=None).awrap_tool_call(FakeToolRequest("wait_for_job", {"job_id": "j1"}), handler))
    assert "may still be running" in json.loads(final.content)["note"]
