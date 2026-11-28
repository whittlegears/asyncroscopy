import asyncio
import json
from unittest.mock import MagicMock

import pytest

from tests.test_agent import setup_llm_stubs

setup_llm_stubs()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from asyncroscopy.mcp.agent import Agent, AgentSwarm  # noqa: E402
from asyncroscopy.training.recorder import TrainingRecorder, to_openai_message  # noqa: E402


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_messages_convert_to_the_openai_chat_format():
    assert to_openai_message(SystemMessage(content="be brief")) == {"role": "system", "content": "be brief"}
    assert to_openai_message(HumanMessage(content="hi")) == {"role": "user", "content": "hi"}
    call = to_openai_message(AIMessage(content="", tool_calls=[{"name": "get_stage", "args": {"axis": "x"}, "id": "c1"}]))
    assert call["tool_calls"][0]["function"] == {"name": "get_stage", "arguments": '{"axis": "x"}'}
    assert to_openai_message(ToolMessage(content="1e-6", tool_call_id="c1")) == {
        "role": "tool", "content": "1e-6", "tool_call_id": "c1"
    }
    with pytest.raises(ValueError, match="Cannot record"):
        to_openai_message(object())


def test_each_model_call_is_one_prompt_output_record_with_its_tools(tmp_path):
    recorder = TrainingRecorder(tmp_path, model="test-model")
    tools = [{"type": "function", "function": {"name": "get_stage"}}]
    recorder.record_call([SystemMessage(content="sys"), HumanMessage(content="focus")], AIMessage(content="done"),
                         agent="w", tools=tools)

    [record] = _lines(tmp_path / "llm_calls.jsonl")
    assert record["model"] == "test-model"
    assert record["tools"] == tools
    assert record["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "focus"},
        {"role": "assistant", "content": "done"},
    ]


def test_a_run_collects_its_tool_calls_and_final_answer(tmp_path):
    recorder = TrainingRecorder(tmp_path)
    recorder.start_run("take an image")
    recorder.record_tool("acquire_scanned_image", {"detectors": ["haadf"]}, "key-1")
    run_id = recorder.end_run("Image saved as key-1.")

    [record] = _lines(tmp_path / "runs.jsonl")
    assert record["id"] == run_id
    assert record["tool_calls"] == [
        {"agent": "", "name": "acquire_scanned_image", "arguments": {"detectors": ["haadf"]}, "result": "key-1"}
    ]
    assert record["final_answer"] == "Image saved as key-1."
    assert recorder.end_run("again") is None  # no run in progress


def _swarm_with_events(tmp_path, events):
    swarm = AgentSwarm(model="mock-model", verbose=False, training_log_dir=str(tmp_path),
                       startup_agents=[Agent(name="worker", system_prompt="You run the microscope.", tools=["*"])])
    swarm._model = MagicMock()
    swarm._tools = []
    executor = MagicMock()
    executor.astream_events = events
    swarm._build_agent_executor = MagicMock(return_value=executor)
    return swarm


def test_agent_query_is_recorded(tmp_path):
    tool_round = AIMessage(content="", tool_calls=[{"name": "get_stage", "args": {}, "id": "c1"}])
    answer = AIMessage(content="The stage is at x = 0.")

    async def events(inputs, config=None, version=None):
        prompt = inputs["messages"]
        yield {"event": "on_chat_model_start", "run_id": "m1", "data": {"input": {"messages": [prompt]}}}
        yield {"event": "on_chat_model_end", "run_id": "m1", "data": {"output": tool_round}}
        yield {"event": "on_tool_start", "run_id": "t1", "name": "get_stage", "data": {"input": {}}}
        yield {"event": "on_tool_end", "run_id": "t1", "name": "get_stage",
               "data": {"output": ToolMessage(content="[0, 0]", tool_call_id="c1")}}
        yield {"event": "on_chat_model_start", "run_id": "m2", "data": {"input": {"messages": [prompt + [tool_round]]}}}
        yield {"event": "on_chat_model_end", "run_id": "m2", "data": {"output": answer}}

    swarm = _swarm_with_events(tmp_path, events)
    final = asyncio.run(swarm.query("Where is the stage?", include_transcript=False))

    assert final == "The stage is at x = 0."
    calls = _lines(tmp_path / "llm_calls.jsonl")
    assert calls[0]["messages"][0] == {"role": "user", "content": "Where is the stage?"}
    assert calls[0]["messages"][-1]["tool_calls"][0]["function"]["name"] == "get_stage"
    assert calls[1]["messages"][-1] == {"role": "assistant", "content": "The stage is at x = 0."}
    [run] = _lines(tmp_path / "runs.jsonl")
    assert run["tool_calls"][0]["result"] == "[0, 0]"
    assert {call["run_id"] for call in calls} == {run["id"]}


def test_a_failed_query_is_still_recorded(tmp_path):
    async def events(inputs, config=None, version=None):
        raise RuntimeError("model went away")
        yield

    swarm = _swarm_with_events(tmp_path, events)
    asyncio.run(swarm.query("Take an image", include_transcript=False))

    [run] = _lines(tmp_path / "runs.jsonl")
    assert run["error"] == "model went away"


def test_recording_is_off_without_a_directory():
    assert AgentSwarm(model="mock-model", verbose=False)._recorder is None
