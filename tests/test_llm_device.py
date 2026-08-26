"""Tests for the LLM device shell and the LangGraph backend, with stubbed agent deps."""

import asyncio
import json
import sys
import types

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

def setup_llm_stubs():
    """Stub every langchain/langgraph import so backends.langgraph_backend loads without the real agent deps."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if "langgraph.graph" in sys.modules:
        return

    base_msg_cls = type("BaseMessage", (), {})
    human_msg_cls = type("HumanMessage", (base_msg_cls,), {
        "__init__": lambda self, content, name=None: (
            setattr(self, "content", content) or setattr(self, "name", name)
        ),
    })
    system_msg_cls = type("SystemMessage", (base_msg_cls,), {
        "__init__": lambda self, content: setattr(self, "content", content),
    })
    ai_msg_cls = type("AIMessage", (base_msg_cls,), {
        "__init__": lambda self, content, tool_calls=None: (
            setattr(self, "content", content) or setattr(self, "tool_calls", tool_calls or [])
        ),
    })
    tool_msg_cls = type("ToolMessage", (base_msg_cls,), {
        "__init__": lambda self, content, tool_call_id=None: (
            setattr(self, "content", content) or setattr(self, "tool_call_id", tool_call_id)
        ),
    })

    langchain_core = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.BaseTool = type("BaseTool", (), {})

    class _StubStructuredTool:
        def __init__(self, func):
            self.name = func.__name__
            self.func = func

        @staticmethod
        def from_function(func):
            return _StubStructuredTool(func)

    lc_tools.StructuredTool = _StubStructuredTool
    lc_messages = types.ModuleType("langchain_core.messages")
    lc_messages.BaseMessage = base_msg_cls
    lc_messages.HumanMessage = human_msg_cls
    lc_messages.SystemMessage = system_msg_cls
    lc_messages.AIMessage = ai_msg_cls
    lc_messages.ToolMessage = tool_msg_cls
    langchain_core.tools = lc_tools
    langchain_core.messages = lc_messages

    langchain = types.ModuleType("langchain")
    lc_cm = types.ModuleType("langchain.chat_models")
    lc_cm.init_chat_model = MagicMock()
    langchain.chat_models = lc_cm

    lc_agents = types.ModuleType("langchain.agents")
    lc_agents.create_agent = MagicMock()
    langchain.agents = lc_agents

    lc_mcp = types.ModuleType("langchain_mcp_adapters")
    lc_mcp_client = types.ModuleType("langchain_mcp_adapters.client")
    lc_mcp_client.MultiServerMCPClient = MagicMock()
    lc_mcp.client = lc_mcp_client

    lg = types.ModuleType("langgraph")
    lg_graph = types.ModuleType("langgraph.graph")
    lg_graph.END = "__end__"
    lg_graph.START = "__start__"
    lg_graph.StateGraph = MagicMock()
    lg.graph = lg_graph

    sys.modules.update({
        "langchain_core": langchain_core,
        "langchain_core.tools": lc_tools,
        "langchain_core.messages": lc_messages,
        "langchain": langchain,
        "langchain.chat_models": langchain.chat_models,
        "langchain.agents": langchain.agents,
        "langchain_mcp_adapters": lc_mcp,
        "langchain_mcp_adapters.client": lc_mcp_client,
        "langgraph": lg,
        "langgraph.graph": lg_graph,
    })


setup_llm_stubs()

from asyncroscopy.mcp.llm import Agent, LLM
from asyncroscopy.mcp.backends import create_backend
from asyncroscopy.mcp.backends.base import BackendConfig, BackendUnsupported
from asyncroscopy.mcp.backends.langgraph_backend import LangGraphBackend
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(**kwargs) -> LangGraphBackend:
    """Return a LangGraphBackend without running initialize()."""
    backend = LangGraphBackend(BackendConfig(), kwargs.get("agents", []))
    backend._tools = kwargs.get("tools", [])
    backend._model = kwargs.get("model", None)
    return backend


def _make_llm(**kwargs) -> LLM:
    """Return a bare LLM instance without touching Tango at all."""
    device = LLM.__new__(LLM)
    device._max_steps = kwargs.get("max_steps", 5)
    device._agents: list[Agent] = kwargs.get("agents", [])
    device._backend = kwargs.get("backend", None)

    device._tango_properties = {}
    device.ollama_model = "mock-model"
    device.model_name = ""
    device.model_provider = "ollama"
    device.local_model_path = ""
    device.ollama_host = "http://localhost:11434"
    device.api_key = ""
    device.use_init_chat_model = False
    device.agent_backend = "langgraph"
    device.mcp_url = kwargs.get("mcp_url", "")
    device.hermes_url = ""
    device.hermes_model = "hermes-agent"
    device.hermes_api_key = ""
    device.skills_dir = kwargs.get("skills_dir", "outputs/agent_skills")
    device.embedding_model = "stub-embed"
    device._skills_service = kwargs.get("skills_service", None)
    device._skill_sync_buffers = {}

    # Mock C++ Tango logging methods that would otherwise segfault
    # when called on an uninitialized C++ object
    device.info_stream = MagicMock()
    device.error_stream = MagicMock()
    device.debug_stream = MagicMock()
    device.set_state = MagicMock()
    device.set_status = MagicMock()

    return device


def _stub_backend(name="hermes", supports_connect_mcp=False) -> MagicMock:
    """Return a backend double for SetBackend switches, initialize() already stubbed."""
    backend = MagicMock()
    backend.name = name
    backend.supports_connect_mcp = supports_connect_mcp
    backend.initialize = AsyncMock()
    backend.connect_mcp = AsyncMock(return_value=2)
    return backend


def _make_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    return t


def _make_agent(name="worker", system_prompt="You are helpful.", tools=None) -> Agent:
    return Agent(name=name, system_prompt=system_prompt, tools=tools or ["*"])


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

class TestCreateBackend:
    def test_langgraph_by_name(self):
        backend = create_backend("langgraph", BackendConfig(), [])
        assert isinstance(backend, LangGraphBackend)
        assert backend.name == "langgraph"

    def test_empty_name_defaults_to_langgraph(self):
        backend = create_backend("", BackendConfig(), [])
        assert backend.name == "langgraph"

    def test_unknown_name_raises_with_valid_names(self):
        with pytest.raises(RuntimeError) as excinfo:
            create_backend("smolagents", BackendConfig(), [])
        assert "langgraph" in str(excinfo.value)
        assert "hermes" in str(excinfo.value)

    def test_shared_agents_list(self):
        agents: list[Agent] = []
        backend = create_backend("langgraph", BackendConfig(), agents)
        agents.append(_make_agent(name="LateSpawn"))
        assert backend.agents[0].name == "LateSpawn"

    def test_capabilities_reported(self):
        backend = create_backend("langgraph", BackendConfig(), [])
        assert backend.capabilities() == {"complete": True, "connect_mcp": True}


# ---------------------------------------------------------------------------
# LangGraph backend internals
# ---------------------------------------------------------------------------

class TestGetAgentTools:
    def test_wildcard_returns_all(self):
        tools = [_make_tool("math_add"), _make_tool("read_file"), _make_tool("write_file")]
        backend = _make_backend(tools=tools)
        assert backend._get_agent_tools(["*"]) is tools

    def test_exact_name_match(self):
        t_read = _make_tool("read_file")
        t_write = _make_tool("write_file")
        backend = _make_backend(tools=[t_read, t_write])
        result = backend._get_agent_tools(["read_file"])
        assert result == [t_read]

    def test_glob_prefix(self):
        tools = [_make_tool("math_add"), _make_tool("math_sub"), _make_tool("read_file")]
        backend = _make_backend(tools=tools)
        result = backend._get_agent_tools(["math_*"])
        assert len(result) == 2
        assert all(t.name.startswith("math_") for t in result)

    def test_multiple_patterns(self):
        tools = [_make_tool("math_add"), _make_tool("read_file"), _make_tool("write_file")]
        backend = _make_backend(tools=tools)
        result = backend._get_agent_tools(["math_*", "read_file"])
        assert len(result) == 2

    def test_no_match_returns_empty(self):
        backend = _make_backend(tools=[_make_tool("math_add")])
        assert backend._get_agent_tools(["nonexistent"]) == []

    def test_empty_tool_list(self):
        backend = _make_backend(tools=[])
        assert backend._get_agent_tools(["*"]) == []


class TestExtractJson:
    def test_bare_json_passthrough(self):
        backend = _make_backend()
        raw = '{"next": "worker", "task": "do something"}'
        assert backend._extract_json(raw) == raw

    def test_strips_json_code_fence(self):
        backend = _make_backend()
        fenced = '```json\n{"next": "worker"}\n```'
        assert backend._extract_json(fenced) == '{"next": "worker"}'

    def test_strips_plain_code_fence(self):
        backend = _make_backend()
        fenced = '```\n{"next": "FINISH"}\n```'
        assert backend._extract_json(fenced) == '{"next": "FINISH"}'

    def test_whitespace_stripped(self):
        backend = _make_backend()
        assert backend._extract_json('  {"a": 1}  ') == '{"a": 1}'

    def test_non_json_text_passthrough(self):
        backend = _make_backend()
        text = "Just some plain text"
        assert backend._extract_json(text) == text


class TestParseRoutingDecision:
    def test_valid_next_returned(self):
        backend = _make_backend()
        content = '{"next": "worker", "task": "scan the sample"}'
        next_agent, subtask = backend._parse_routing_decision(content, ["worker", "FINISH"], "FINISH")
        assert next_agent == "worker"
        assert subtask == "scan the sample"

    def test_invalid_next_falls_back(self):
        backend = _make_backend()
        content = '{"next": "nonexistent_agent", "task": "whatever"}'
        next_agent, _ = backend._parse_routing_decision(content, ["worker", "FINISH"], "FINISH")
        assert next_agent == "FINISH"

    def test_missing_next_key_falls_back(self):
        backend = _make_backend()
        content = '{"task": "do something"}'
        # decision.get("next", fallback) returns fallback when key is absent
        next_agent, _ = backend._parse_routing_decision(content, ["worker"], "worker")
        assert next_agent == "worker"

    def test_malformed_json_falls_back(self):
        backend = _make_backend()
        next_agent, subtask = backend._parse_routing_decision("{broken json!!}", ["worker"], "worker")
        assert next_agent == "worker"
        assert subtask == ""

    def test_fenced_json_parsed(self):
        backend = _make_backend()
        content = '```json\n{"next": "FINISH", "task": ""}\n```'
        next_agent, _ = backend._parse_routing_decision(content, ["worker", "FINISH"], "worker")
        assert next_agent == "FINISH"

    def test_missing_task_key_returns_empty_string(self):
        backend = _make_backend()
        content = '{"next": "worker"}'
        _, subtask = backend._parse_routing_decision(content, ["worker"], "worker")
        assert subtask == ""


class TestRunSwarm:
    def test_no_agents_returns_error_message(self):
        backend = _make_backend()
        result = asyncio.run(backend._run_swarm("hello", 5))
        assert "No agents available" in result

    def test_single_agent_calls_stream_agent(self):
        """Single-agent path skips the supervisor graph entirely."""
        agent = _make_agent(name="Solo")
        backend = _make_backend(agents=[agent])

        # Replace internal helpers so no LangChain objects are needed
        backend._build_agent_executor = MagicMock(return_value=MagicMock())
        backend._stream_agent = AsyncMock(return_value="42 is the answer.")

        result = asyncio.run(backend._run_swarm("What is the answer?", 5))

        assert result == "42 is the answer."
        backend._build_agent_executor.assert_called_once_with(agent)
        backend._stream_agent.assert_called_once()

    def test_single_agent_receives_prompt_in_message(self):
        """The prompt must be forwarded as the HumanMessage content."""
        agent = _make_agent()
        backend = _make_backend(agents=[agent])
        backend._build_agent_executor = MagicMock(return_value=MagicMock())

        captured_messages = []

        async def fake_stream(executor, messages, agent_label=""):
            captured_messages.extend(messages)
            return "done"

        backend._stream_agent = fake_stream

        asyncio.run(backend._run_swarm("scan now", 5))
        assert len(captured_messages) == 1
        assert captured_messages[0].content == "scan now"

    def test_query_delegates_to_run_swarm(self):
        backend = _make_backend()
        backend._run_swarm = AsyncMock(return_value="done")
        result = asyncio.run(backend.query("do a thing", 7))
        assert result == "done"
        backend._run_swarm.assert_called_once_with("do a thing", 7)


class TestOpenAIMessagesToLangchain:
    def test_user_message_becomes_human_message(self):
        [msg] = LangGraphBackend._openai_messages_to_langchain([{"role": "user", "content": "hi"}])
        assert isinstance(msg, HumanMessage)
        assert msg.content == "hi"

    def test_system_message_becomes_system_message(self):
        [msg] = LangGraphBackend._openai_messages_to_langchain([{"role": "system", "content": "be careful"}])
        assert isinstance(msg, SystemMessage)
        assert msg.content == "be careful"

    def test_assistant_message_without_tool_calls(self):
        [msg] = LangGraphBackend._openai_messages_to_langchain([{"role": "assistant", "content": "done"}])
        assert isinstance(msg, AIMessage)
        assert msg.content == "done"
        assert msg.tool_calls == []

    def test_assistant_message_with_tool_calls_parses_arguments(self):
        [msg] = LangGraphBackend._openai_messages_to_langchain([{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "acquire_image", "arguments": '{"detector": "haadf"}'},
            }],
        }])
        assert isinstance(msg, AIMessage)
        assert msg.tool_calls == [{"name": "acquire_image", "args": {"detector": "haadf"}, "id": "call_1"}]

    def test_tool_message_becomes_tool_message(self):
        [msg] = LangGraphBackend._openai_messages_to_langchain([{
            "role": "tool", "tool_call_id": "call_1", "content": "stem_image_HAADF_x.h5",
        }])
        assert isinstance(msg, ToolMessage)
        assert msg.content == "stem_image_HAADF_x.h5"
        assert msg.tool_call_id == "call_1"

    def test_unknown_role_falls_back_to_human_message(self):
        [msg] = LangGraphBackend._openai_messages_to_langchain([{"role": "weird", "content": "??"}])
        assert isinstance(msg, HumanMessage)

    def test_missing_content_defaults_to_empty_string(self):
        [msg] = LangGraphBackend._openai_messages_to_langchain([{"role": "user"}])
        assert msg.content == ""


class TestLangchainMessageToOpenAI:
    def test_plain_text_message(self):
        message = AIMessage(content="hello there")
        result = LangGraphBackend._langchain_message_to_openai(message)
        assert result == {"role": "assistant", "content": "hello there"}

    def test_message_with_tool_calls_encodes_arguments_as_json_string(self):
        message = AIMessage(content="", tool_calls=[{"name": "acquire_image", "args": {"n": 1}, "id": "call_9"}])
        result = LangGraphBackend._langchain_message_to_openai(message)
        assert result["tool_calls"] == [{
            "id": "call_9",
            "type": "function",
            "function": {"name": "acquire_image", "arguments": '{"n": 1}'},
        }]

    def test_tool_call_missing_id_gets_a_fallback(self):
        message = AIMessage(content="", tool_calls=[{"name": "acquire_image", "args": {}, "id": ""}])
        result = LangGraphBackend._langchain_message_to_openai(message)
        assert result["tool_calls"][0]["id"] == "call_0"


class TestBackendComplete:
    def test_returns_tool_call_decision(self):
        backend = _make_backend()
        response = AIMessage(content="", tool_calls=[{"name": "acquire_image", "args": {}, "id": "call_1"}])
        bound_model = AsyncMock()
        bound_model.ainvoke.return_value = response
        backend._model = MagicMock()
        backend._model.bind_tools.return_value = bound_model

        request = {
            "messages": [{"role": "user", "content": "acquire an image"}],
            "tools": [{"type": "function", "function": {"name": "acquire_image"}}],
        }
        result = asyncio.run(backend.complete(request))

        assert result["tool_calls"][0]["function"]["name"] == "acquire_image"
        backend._model.bind_tools.assert_called_once_with(request["tools"])

    def test_skips_bind_tools_when_no_tools_given(self):
        backend = _make_backend()
        response = AIMessage(content="hi there")
        backend._model = AsyncMock()
        backend._model.ainvoke.return_value = response

        request = {"messages": [{"role": "user", "content": "hi"}]}
        result = asyncio.run(backend.complete(request))

        assert result["content"] == "hi there"
        backend._model.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# Device shell
# ---------------------------------------------------------------------------

class TestSpawnAgent:
    def test_spawn_adds_agent(self):
        device = _make_llm()
        config = json.dumps({"name": "Alpha", "system_prompt": "You scan.", "tools": ["scan_*"]})
        result = device.SpawnAgent(config)
        assert result is True
        assert len(device._agents) == 1
        assert device._agents[0].name == "Alpha"
        assert device._agents[0].tools == ["scan_*"]

    def test_spawn_multiple_agents(self):
        device = _make_llm()
        for i in range(3):
            device.SpawnAgent(json.dumps({"name": f"Agent{i}", "system_prompt": "help", "tools": ["*"]}))
        assert len(device._agents) == 3

    def test_spawn_defaults_tools_to_wildcard(self):
        device = _make_llm()
        device.SpawnAgent(json.dumps({"name": "Beta", "system_prompt": "help"}))
        assert device._agents[0].tools == ["*"]

    def test_spawn_missing_required_field_returns_false(self):
        device = _make_llm()
        # "name" is required by Agent dataclass
        result = device.SpawnAgent(json.dumps({"system_prompt": "missing name"}))
        assert result is False
        assert device._agents == []

    def test_spawn_invalid_json_returns_false(self):
        device = _make_llm()
        result = device.SpawnAgent("{bad json")
        assert result is False

    def test_spawn_preserves_description(self):
        device = _make_llm()
        device.SpawnAgent(json.dumps({"name": "Gamma", "system_prompt": "help", "description": "does science"}))
        assert device._agents[0].description == "does science"

    def test_spawn_is_visible_to_the_backend(self):
        agents: list[Agent] = []
        backend = LangGraphBackend(BackendConfig(), agents)
        device = _make_llm(agents=agents, backend=backend)
        device.SpawnAgent(json.dumps({"name": "Delta", "system_prompt": "help"}))
        assert backend.agents[0].name == "Delta"


class TestMaxSteps:
    def test_read_default(self):
        device = _make_llm()
        assert device.read_max_steps() == 5

    def test_write_valid(self):
        device = _make_llm()
        device.write_max_steps(10)
        assert device.read_max_steps() == 10

    def test_write_zero_raises(self):
        device = _make_llm()
        with pytest.raises(ValueError):
            device.write_max_steps(0)

    def test_write_negative_raises(self):
        device = _make_llm()
        with pytest.raises(ValueError):
            device.write_max_steps(-3)

    def test_write_one_is_valid(self):
        device = _make_llm()
        device.write_max_steps(1)
        assert device.read_max_steps() == 1


class TestDeviceBackendAttributes:
    def test_backend_name_reported(self):
        device = _make_llm(backend=_make_backend())
        assert device.read_backend() == "langgraph"

    def test_backend_capabilities_reported_as_json(self):
        device = _make_llm(backend=_make_backend())
        assert json.loads(device.read_backend_capabilities()) == {"complete": True, "connect_mcp": True}

    def test_backend_raises_when_uninitialized(self):
        device = _make_llm()
        with pytest.raises(RuntimeError):
            device.read_backend()

    def test_tools_empty_when_uninitialized(self):
        device = _make_llm()
        assert json.loads(device.read_tools()) == []

    def test_tools_reports_backend_tool_names(self):
        backend = _make_backend(tools=[_make_tool("scan_image")])
        device = _make_llm(backend=backend)
        assert json.loads(device.read_tools()) == [{"name": "scan_image"}]

    def test_agents_reports_spawned_names(self):
        device = _make_llm(agents=[_make_agent(name="Solo")])
        assert device.read_agents() == ["Solo"]


class TestDeviceComplete:
    def test_wraps_backend_message(self):
        backend = _make_backend()
        backend.complete = AsyncMock(return_value={"role": "assistant", "content": "hi there"})
        device = _make_llm(backend=backend)

        request = {"messages": [{"role": "user", "content": "hi"}]}
        result = json.loads(asyncio.run(device.Complete(json.dumps(request))))

        assert result["message"]["content"] == "hi there"
        backend.complete.assert_called_once_with(request)

    def test_invalid_json_returns_error_payload(self):
        device = _make_llm(backend=_make_backend())
        result = json.loads(asyncio.run(device.Complete("not json")))
        assert "error" in result
        assert "message" in result["error"]

    def test_backend_exception_returns_error_payload(self):
        backend = _make_backend()
        backend.complete = AsyncMock(side_effect=RuntimeError("model unavailable"))
        device = _make_llm(backend=backend)

        request = {"messages": [{"role": "user", "content": "hi"}]}
        result = json.loads(asyncio.run(device.Complete(json.dumps(request))))

        assert result["error"]["message"] == "model unavailable"

    def test_unsupported_backend_returns_error_payload(self):
        backend = _make_backend()
        backend.complete = AsyncMock(side_effect=BackendUnsupported("no raw model turn"))
        device = _make_llm(backend=backend)

        request = {"messages": [{"role": "user", "content": "hi"}]}
        result = json.loads(asyncio.run(device.Complete(json.dumps(request))))

        assert result["error"]["message"] == "no raw model turn"


class TestDeviceConnectMCP:
    def test_success_returns_true(self):
        backend = _make_backend()
        backend.connect_mcp = AsyncMock(return_value=3)
        device = _make_llm(backend=backend)
        result = asyncio.run(device.ConnectMCP(json.dumps({"url": "http://127.0.0.1:8000/mcp"})))
        assert result is True
        backend.connect_mcp.assert_called_once_with("http://127.0.0.1:8000/mcp", "streamable_http")

    def test_unsupported_returns_false(self):
        backend = _make_backend()
        backend.connect_mcp = AsyncMock(side_effect=BackendUnsupported("not here"))
        device = _make_llm(backend=backend)
        result = asyncio.run(device.ConnectMCP(json.dumps({"url": "http://127.0.0.1:8000/mcp"})))
        assert result is False

    def test_failure_returns_false(self):
        backend = _make_backend()
        backend.connect_mcp = AsyncMock(side_effect=RuntimeError("boom"))
        device = _make_llm(backend=backend)
        result = asyncio.run(device.ConnectMCP(json.dumps({"url": "http://127.0.0.1:8000/mcp"})))
        assert result is False


class TestSetBackend:
    def test_same_name_is_a_no_op(self):
        current = _make_backend()
        device = _make_llm(backend=current)
        with patch("asyncroscopy.mcp.llm.create_backend") as factory:
            result = asyncio.run(device.SetBackend("langgraph"))
        assert result is True
        assert device._backend is current
        factory.assert_not_called()

    def test_successful_switch_replaces_the_backend(self):
        current = _make_backend()
        device = _make_llm(backend=current)
        replacement = _stub_backend()
        with patch("asyncroscopy.mcp.llm.create_backend", return_value=replacement):
            result = asyncio.run(device.SetBackend("hermes"))
        assert result is True
        assert device._backend is replacement
        replacement.initialize.assert_awaited_once()

    def test_failed_initialize_keeps_the_current_backend(self):
        current = _make_backend()
        device = _make_llm(backend=current)
        replacement = _stub_backend()
        replacement.initialize = AsyncMock(side_effect=RuntimeError("gateway down"))
        with patch("asyncroscopy.mcp.llm.create_backend", return_value=replacement):
            result = asyncio.run(device.SetBackend("hermes"))
        assert result is False
        assert device._backend is current
        status = device.set_status.call_args[0][0]
        assert "gateway down" in status

    def test_unknown_name_keeps_the_current_backend(self):
        current = _make_backend()
        device = _make_llm(backend=current)
        with patch(
            "asyncroscopy.mcp.llm.create_backend",
            side_effect=RuntimeError("Unknown agent_backend 'nope'"),
        ):
            result = asyncio.run(device.SetBackend("nope"))
        assert result is False
        assert device._backend is current

    def test_switch_reconnects_mcp_when_the_backend_supports_it(self):
        device = _make_llm(backend=_stub_backend(), mcp_url="http://127.0.0.1:8000/mcp")
        replacement = _stub_backend(name="langgraph", supports_connect_mcp=True)
        with patch("asyncroscopy.mcp.llm.create_backend", return_value=replacement):
            result = asyncio.run(device.SetBackend("LangGraph"))
        assert result is True
        replacement.connect_mcp.assert_awaited_once_with("http://127.0.0.1:8000/mcp", "streamable_http")


class TestSkillTools:
    def test_set_skills_service_adds_find_and_load_tools(self):
        backend = _make_backend()
        backend.set_skills_service(MagicMock())
        assert "find_skills" in backend.tool_names()
        assert "load_skill" in backend.tool_names()

    def test_skill_tools_are_absent_without_a_service(self):
        assert _make_backend().tool_names() == []

    def test_resetting_the_service_does_not_duplicate_tools(self):
        backend = _make_backend()
        backend.set_skills_service(MagicMock())
        backend.set_skills_service(MagicMock())
        assert backend.tool_names().count("find_skills") == 1

    def test_find_skills_tool_reports_unavailability_honestly(self):
        backend = _make_backend()
        service = MagicMock()
        service.find_skills.side_effect = RuntimeError("index empty")
        backend.set_skills_service(service)
        tool = next(t for t in backend._tools if t.name == "find_skills")
        assert "index empty" in tool.func("focus the probe")


class TestDeviceSkillCommands:
    def _service(self, **kwargs):
        service = MagicMock()
        service.sync_from_payload.return_value = kwargs.get(
            "report", {"written": 1, "removed": 0, "skills": 1, "chunks_embedded": 3}
        )
        service.find_skills.return_value = kwargs.get(
            "results", [{"id": "probe-alignment", "score": 0.5}]
        )
        return service

    def test_sync_skills_applies_a_single_chunk_payload(self):
        service = self._service()
        device = _make_llm(skills_service=service)
        body = json.dumps({"version": 1, "sync_id": "s1", "seq": 0, "total": 1, "skills": [{"id": "a"}]})
        result = json.loads(asyncio.run(device.SyncSkills(body)))
        assert result["status"] == "applied"
        assert result["chunks_embedded"] == 3
        service.sync_from_payload.assert_called_once_with([{"id": "a"}])

    def test_sync_skills_reassembles_a_chunked_envelope(self):
        service = self._service()
        device = _make_llm(skills_service=service)
        first = json.loads(asyncio.run(device.SyncSkills(
            json.dumps({"sync_id": "s2", "seq": 1, "total": 2, "skills": [{"id": "b"}]})
        )))
        assert first == {"status": "buffered", "received": 1, "of": 2}
        second = json.loads(asyncio.run(device.SyncSkills(
            json.dumps({"sync_id": "s2", "seq": 0, "total": 2, "skills": [{"id": "a"}]})
        )))
        assert second["status"] == "applied"
        service.sync_from_payload.assert_called_once_with([{"id": "a"}, {"id": "b"}])

    def test_sync_skills_reports_errors_as_json(self):
        service = self._service()
        service.sync_from_payload.side_effect = RuntimeError("disk full")
        device = _make_llm(skills_service=service)
        body = json.dumps({"sync_id": "s3", "seq": 0, "total": 1, "skills": []})
        result = json.loads(asyncio.run(device.SyncSkills(body)))
        assert "disk full" in result["error"]["message"]

    def test_sync_without_a_service_reports_unavailable(self):
        device = _make_llm()
        result = json.loads(asyncio.run(device.SyncSkills(json.dumps({"skills": []}))))
        assert "unavailable" in result["error"]["message"]

    def test_search_skills_returns_results(self):
        device = _make_llm(skills_service=self._service())
        result = json.loads(asyncio.run(device.SearchSkills(json.dumps({"query": "focus"}))))
        assert result["results"][0]["id"] == "probe-alignment"

    def test_search_skills_relays_the_index_refusal(self):
        service = self._service()
        service.find_skills.side_effect = RuntimeError("embedding model not reachable")
        device = _make_llm(skills_service=service)
        result = json.loads(asyncio.run(device.SearchSkills(json.dumps({"query": "focus"}))))
        assert "not reachable" in result["error"]["message"]

    def test_search_skills_without_a_query_is_an_error(self):
        device = _make_llm(skills_service=self._service())
        result = json.loads(asyncio.run(device.SearchSkills(json.dumps({}))))
        assert "query" in result["error"]["message"].lower()
