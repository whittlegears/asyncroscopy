"""The agent swarm's graphs built and run with the real LangChain/LangGraph stack.

tests/test_agent.py replaces the stack with stubs where it is missing, so the real
graph is never compiled there. These tests skip unless it is installed
(uv sync --extra agent). A scripted chat model stands in for the LLM.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain.agents.middleware")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGenerationChunk  # noqa: E402
from langchain_core.tools import ToolException, tool  # noqa: E402

from asyncroscopy.mcp.agent import Agent, AgentSwarm  # noqa: E402

if not hasattr(AIMessage, "model_fields"):  # the stubs are loaded instead of the real stack
    pytest.skip("LangChain stubs are active", allow_module_level=True)


class ScriptedModel(GenericFakeChatModel):
    """Replies with a fixed script, streaming each reply as one chunk with its tool calls.

    GenericFakeChatModel streams only the text and drops tool calls, and the swarm
    streams every model call.
    """

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = next(self.messages)
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=reply.content,
                tool_call_chunks=[
                    {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                    for i, c in enumerate(reply.tool_calls)
                ],
            )
        )


def _swarm(replies, tools, agents):
    swarm = AgentSwarm(max_steps=12, startup_agents=agents, model="scripted", verbose=False)
    swarm._model = ScriptedModel(messages=iter(replies))
    swarm._tools = tools
    swarm._mcp_clients = []
    return swarm


@tool
async def Scope_move_stage(x: float) -> str:
    """Move the stage."""
    raise ToolException("DevFailed: x is outside the stage limits")


@tool
async def Scope_get_stage() -> str:
    """Read the stage."""
    return "[0, 0, 0, 0, 0]"


def test_a_failed_tool_reaches_the_model_instead_of_ending_the_run():
    replies = [
        AIMessage(content="", tool_calls=[{"name": "Scope_move_stage", "args": {"x": 1.0}, "id": "c1"}]),
        AIMessage(content="The move failed: x is outside the stage limits."),
    ]
    swarm = _swarm(replies, [Scope_move_stage], [Agent("operator", "You run the scope.", ["Scope_*"])])
    answer = asyncio.run(swarm.query("move the stage to x=1", include_transcript=False))
    assert "outside the stage limits" in answer


def test_supervisor_routes_to_a_worker_and_finishes():
    replies = [
        AIMessage(content=json.dumps({"next": "reader", "task": "read the stage"})),  # supervisor
        AIMessage(content="", tool_calls=[{"name": "Scope_get_stage", "args": {}, "id": "c1"}]),  # worker
        AIMessage(content="The stage is at the origin."),  # worker
        AIMessage(content=json.dumps({"next": "FINISH", "task": ""})),  # supervisor
    ]
    agents = [
        Agent("reader", "You read the scope.", ["Scope_get_*"], description="reads state"),
        Agent("mover", "You move the scope.", ["Scope_move_*"], description="moves"),
    ]
    swarm = _swarm(replies, [Scope_get_stage, Scope_move_stage], agents)
    answer = asyncio.run(swarm.query("where is the stage?", include_transcript=True))
    assert "reader" in answer
    assert "origin" in answer


def test_agent_server_runs_the_sample_map_workflow_to_its_first_pause():
    from asyncroscopy.mcp.agent_mcp_server import AgentMCPServer

    server = AgentMCPServer(name="t", mcp_urls=[], verbose=False)
    server.swarm._tools = []
    listed = json.loads(server.list_workflows())
    assert listed[0]["name"] == "sample_map" and "acquire_map_grid" in listed[0]["missing_tools"]

    started = json.loads(asyncio.run(server.start_workflow("sample_map", json.dumps({"request": {"rows": 2, "cols": 2}}))))
    assert started["waiting_for"]["kind"] == "approve_scan"
    assert started["waiting_for"]["plan"]["tiles"] == 4
    done = json.loads(asyncio.run(server.resume_workflow(started["thread_id"], json.dumps({"approved": False}))))
    assert done["finished"] and done["state"]["status"] == "declined"


def test_an_unreadable_routing_reply_is_shown_in_the_transcript():
    replies = [
        AIMessage(content="I think the reader should go"),  # supervisor: not JSON
        AIMessage(content="The stage is at the origin."),  # worker
        AIMessage(content=json.dumps({"next": "FINISH", "task": ""})),  # supervisor
    ]
    agents = [Agent("reader", "You read the scope.", ["Scope_get_*"]), Agent("mover", "You move.", ["Scope_move_*"])]
    swarm = _swarm(replies, [Scope_get_stage, Scope_move_stage], agents)
    answer = asyncio.run(swarm.query("where is the stage?", include_transcript=True))
    assert "was not JSON" in answer


def test_a_failed_query_keeps_the_partial_transcript():
    replies = [AIMessage(content="", tool_calls=[{"name": "Scope_get_stage", "args": {}, "id": f"c{i}"}]) for i in range(20)]
    swarm = _swarm(replies, [Scope_get_stage], [Agent("reader", "You read.", ["Scope_*"])])
    swarm.max_steps = 4  # the lone agent loops on tool calls until the step limit
    answer = asyncio.run(swarm.query("loop", include_transcript=True))
    assert "ERROR:" in answer and "Scope_get_stage" in answer


class StructuredRouter(ScriptedModel):
    """Answers routing with structured output; workers still get the scripted replies."""

    routes: list = []

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda

        allowed = schema["properties"]["next"]["enum"]
        routes = type(self).routes

        async def route(_messages):
            decision = routes.pop(0)
            assert decision["next"] in allowed
            return decision

        return RunnableLambda(lambda m: None, afunc=route)


def test_structured_routing_is_used_when_the_model_supports_it():
    StructuredRouter.routes = [{"next": "reader", "task": "read the stage"}, {"next": "FINISH", "task": ""}]
    replies = [AIMessage(content="The stage is at the origin.")]  # the worker's only model call
    agents = [Agent("reader", "You read.", ["Scope_get_*"]), Agent("mover", "You move.", ["Scope_move_*"])]
    swarm = AgentSwarm(max_steps=12, startup_agents=agents, model="scripted", verbose=False)
    swarm._model = StructuredRouter(messages=iter(replies))
    swarm._tools = [Scope_get_stage, Scope_move_stage]
    swarm._mcp_clients = []
    answer = asyncio.run(swarm.query("where is the stage?", include_transcript=True))
    assert "origin" in answer and "was not JSON" not in answer
    assert StructuredRouter.routes == []
