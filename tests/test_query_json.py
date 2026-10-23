"""QueryJSON: in-order tool execution, step limits, per-call sampling and errors reported as errors."""

import asyncio
import json
from types import SimpleNamespace

from tests.test_llm_device import _make_agent, _make_llm

from langchain_core.messages import ToolMessage


class ScriptedModel:
    """Returns one pre-scripted response per astream call and remembers its sampling options."""

    def __init__(self, responses, temperature=0, seed=None):
        self.responses = responses
        self.temperature, self.seed = temperature, seed
        self.copies = []

    def bind_tools(self, tools):
        return self

    def model_copy(self, update=None):
        copy = ScriptedModel(self.responses, **{"temperature": self.temperature, "seed": self.seed, **(update or {})})
        self.copies.append(copy)
        return copy

    async def astream(self, messages):
        yield self.responses.pop(0)


def _response(content="", calls=()):
    return SimpleNamespace(content=content, tool_calls=list(calls), usage_metadata={"input_tokens": 10, "output_tokens": 2})


class OrderTool:
    def __init__(self, name, log, delay=0.0, fail=False):
        self.name, self.log, self.delay, self.fail = name, log, delay, fail

    async def ainvoke(self, call):
        self.log.append(("start", self.name))
        await asyncio.sleep(self.delay)
        self.log.append(("end", self.name))
        if self.fail:
            raise RuntimeError("device rejected the command")
        return ToolMessage(content=f"{self.name} ok", tool_call_id=call["id"])


def _query(device, **request):
    return json.loads(asyncio.run(device.QueryJSON(json.dumps({"prompt": "go", "sequential": True, **request}))))


def _device(responses, tools):
    return _make_llm(agents=[_make_agent(name="operator")], tools=tools, model=ScriptedModel(responses))


def test_tool_calls_run_one_at_a_time_in_issued_order():
    log = []
    tools = [OrderTool("move", log, delay=0.05), OrderTool("acquire", log)]
    calls = [{"name": "move", "args": {"x": 1}, "id": "a"}, {"name": "acquire", "args": {}, "id": "b"}]
    device = _device([_response(calls=calls), _response("done")], tools)
    result = _query(device, query_id="q1")
    assert result == {"query_id": "q1", "answer": "done", "error": None}
    assert log == [("start", "move"), ("end", "move"), ("start", "acquire"), ("end", "acquire")]
    metrics = json.loads(device.read_last_metrics())
    assert [s["tool"] for s in metrics["tool_steps"]] == ["move", "acquire"]
    assert [s["tool_calls"] for s in metrics["model_steps"]] == [["move", "acquire"], []]
    assert metrics["model_steps"][0]["input_tokens"] == 10


def test_step_limit_is_an_error_not_an_answer():
    looping = [_response(calls=[{"name": "move", "args": {}, "id": str(i)}]) for i in range(3)]
    device = _device(looping, [OrderTool("move", [])])
    result = _query(device, max_steps=3)
    assert result["answer"] is None and "step limit" in result["error"]
    assert len(json.loads(device.read_last_metrics())["model_steps"]) == 3


def test_tool_failure_is_reported_to_the_model_and_recorded():
    device = _device([_response(calls=[{"name": "move", "args": {}, "id": "a"}]), _response("could not move")],
                     [OrderTool("move", [], fail=True)])
    assert _query(device)["answer"] == "could not move"
    step = json.loads(device.read_last_metrics())["tool_steps"][0]
    assert step["ok"] is False and "device rejected" in step["output"]


def test_unknown_tool_is_reported_not_raised():
    device = _device([_response(calls=[{"name": "teleport", "args": {}, "id": "a"}]), _response("no such tool")], [])
    assert _query(device)["answer"] == "no such tool"
    assert json.loads(device.read_last_metrics())["tool_steps"][0]["ok"] is False


def test_seed_and_temperature_apply_to_this_call_only():
    model = ScriptedModel([_response("ok")])
    device = _make_llm(agents=[_make_agent()], tools=[], model=model)
    _query(device, seed=7, temperature=0.6)
    assert (model.copies[0].seed, model.copies[0].temperature) == (7, 0.6)
    assert device._model is model


def test_invalid_json_is_an_error():
    device = _device([], [])
    result = json.loads(asyncio.run(device.QueryJSON("{not json")))
    assert result["answer"] is None and "invalid JSON" in result["error"]


def test_disconnect_drops_tools():
    device = _device([], [OrderTool("move", [])])
    assert asyncio.run(device.DisconnectMCP()) is True
    assert device._tools == [] and device._mcp_clients == []


def test_block_list_content_is_returned_as_text():
    blocks = [{"type": "text", "text": "done "}, {"type": "reasoning", "text": "hidden"}]
    device = _device([_response(blocks)], [])
    assert _query(device)["answer"] == "done"


def test_requests_that_cannot_run_are_errors():
    two = _make_llm(agents=[_make_agent(name="a"), _make_agent(name="b")], tools=[], model=ScriptedModel([]))
    assert "exactly one agent" in _query(two)["error"]
    assert "no agents" in _query(_make_llm(agents=[], tools=[], model=ScriptedModel([])))["error"]
    device = _device([], [])
    assert "at least 1" in _query(device, max_steps=0)["error"]
    for bad in ('"just a string"', '{"prompt": "go", "max_steps": "ten"}'):
        result = json.loads(asyncio.run(device.QueryJSON(bad)))
        assert result["answer"] is None and result["error"]


def test_swarm_failure_is_an_error_and_max_steps_reaches_a_lone_agent(monkeypatch):
    device = _device([], [])
    seen = {}

    async def fake_swarm(prompt, limit_single_agent=False):
        seen["limit"], seen["steps"] = limit_single_agent, device._max_steps
        return "Swarm Error: No agent produced a response before routing finished."

    monkeypatch.setattr(device, "_run_swarm", fake_swarm)
    result = json.loads(asyncio.run(device.QueryJSON(json.dumps({"prompt": "go", "max_steps": 4}))))
    assert result["answer"] is None and "No agent produced" in result["error"]
    assert seen == {"limit": True, "steps": 4}
