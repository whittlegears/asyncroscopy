"""Model factory tests: require the real LangChain packages (uv sync --extra agent)."""

import asyncio
import json

import pytest

from asyncroscopy.agent.config import AgentSettings, default_api_key_env, resolve_api_key
from tests._agent_deps import real_agent_deps_installed

pytestmark = pytest.mark.skipif(not real_agent_deps_installed(), reason="requires `uv sync --extra agent`")

if real_agent_deps_installed():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    import asyncroscopy.agent.models as models
    from asyncroscopy.agent.messages import langchain_messages_to_openai, openai_message_to_langchain, openai_messages_to_langchain
    from asyncroscopy.agent.models import TangoChatModel, build_chat_model


class TestSettings:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("ASYNCROSCOPY_AGENT_PROVIDER", "Anthropic")
        monkeypatch.setenv("ASYNCROSCOPY_AGENT_MODEL", "claude-sonnet-4-5")
        monkeypatch.setenv("ASYNCROSCOPY_AGENT_SKILLS_DIRS", "skills,extra/skills")
        monkeypatch.setenv("ASYNCROSCOPY_AGENT_MAX_STEPS", "25")
        settings = AgentSettings.from_env()
        assert settings.provider == "anthropic"
        assert settings.model == "claude-sonnet-4-5"
        assert settings.skills_dirs == ["skills", "extra/skills"]
        assert settings.max_steps == 25
        assert settings.resolved_skills_dirs[0].name == "skills"

    def test_from_mapping_accepts_legacy_keys_and_refuses_api_key(self):
        settings = AgentSettings.from_mapping({"model_provider": "openai", "model_name": "gpt-4o-mini", "tango": {}})
        assert (settings.provider, settings.model) == ("openai", "gpt-4o-mini")
        with pytest.raises(ValueError, match="api_key"):
            AgentSettings.from_mapping({"api_key": "sk-secret"})

    def test_api_key_resolution(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert default_api_key_env("openai") == "OPENAI_API_KEY"
        assert default_api_key_env("ollama") is None
        assert resolve_api_key(AgentSettings(provider="openai")) is None
        monkeypatch.setenv("MY_KEY", "abc")
        assert resolve_api_key(AgentSettings(provider="openai", api_key_env="MY_KEY")) == "abc"


class TestBuildChatModel:
    def test_api_provider_uses_init_chat_model(self, monkeypatch):
        captured = {}

        def fake_init(model, model_provider=None, **kwargs):
            captured.update(model=model, provider=model_provider, **kwargs)
            return "MODEL"

        monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = build_chat_model(AgentSettings(provider="openai", model="gpt-4o-mini", temperature=0.2))
        assert result == "MODEL"
        assert captured == {"model": "gpt-4o-mini", "provider": "openai", "api_key": "sk-test", "temperature": 0.2}

    def test_missing_key_raises_with_variable_name(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(models, "load_dotenv_if_present", lambda *a, **k: False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            build_chat_model(AgentSettings(provider="anthropic", model="claude-sonnet-4-5"))

    def test_ollama_provider(self, monkeypatch):
        pytest.importorskip("langchain_ollama")
        model = build_chat_model(AgentSettings(provider="ollama", model="qwen3:8b", base_url="http://ollama:11434"))
        assert type(model).__name__ == "ChatOllama"
        assert model.model == "qwen3:8b"

    def test_tango_provider(self):
        model = build_chat_model(AgentSettings(provider="tango", model="asyncroscopy/llm/default"))
        assert isinstance(model, TangoChatModel)
        assert model.device_name == "asyncroscopy/llm/default"


class FakeProxy:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def set_timeout_millis(self, value):
        self.timeout = value

    def Complete(self, request_json):
        self.requests.append(json.loads(request_json))
        return json.dumps(self.reply)


class TestTangoChatModel:
    def _model(self, reply):
        model = TangoChatModel(device_name="asyncroscopy/llm/test")
        model._proxy = FakeProxy(reply)
        return model

    def test_tool_call_round_trip(self):
        reply = {"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "DigitalTwin_acquire_spectrum", "arguments": "{\"detector_name\": \"eds\"}"}}]}}
        model = self._model(reply)
        from langchain_core.tools import tool

        @tool
        def DigitalTwin_acquire_spectrum(detector_name: str) -> str:
            """spec"""
            return "k"

        bound = model.bind_tools([DigitalTwin_acquire_spectrum])
        response = asyncio.run(bound.ainvoke([SystemMessage(content="sys"), HumanMessage(content="eds")]))
        assert isinstance(response, AIMessage)
        assert response.tool_calls[0]["name"] == "DigitalTwin_acquire_spectrum"
        assert response.tool_calls[0]["args"] == {"detector_name": "eds"}
        request = model._proxy.requests[0]
        assert request["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "eds"}]
        assert request["tools"][0]["function"]["name"] == "DigitalTwin_acquire_spectrum"

    def test_error_payload_raises(self):
        model = self._model({"error": {"message": "model unavailable"}})
        with pytest.raises(RuntimeError, match="model unavailable"):
            model.invoke([HumanMessage(content="hi")])

    def test_plain_text_and_stream(self):
        model = self._model({"message": {"role": "assistant", "content": "hello"}})
        assert model.invoke([HumanMessage(content="hi")]).content == "hello"
        chunks = list(model.stream([HumanMessage(content="hi")]))
        assert "".join(c.content for c in chunks) == "hello"


class TestMessageConversion:
    def test_round_trip_with_tool_message(self):
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="do it"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {"a": 1}, "id": "call_1"}]),
            ToolMessage(content="result", tool_call_id="call_1"),
            AIMessage(content=[{"type": "text", "text": "final"}]),
        ]
        openai = langchain_messages_to_openai(messages)
        assert openai[2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
        assert openai[3] == {"role": "tool", "content": "result", "tool_call_id": "call_1"}
        assert openai[4]["content"] == "final"
        back = openai_messages_to_langchain(openai)
        assert [m.type for m in back] == ["system", "human", "ai", "tool", "ai"]
        assert back[2].tool_calls[0]["args"] == {"a": 1}
        assert openai_message_to_langchain(openai[2]).tool_calls[0]["id"] == "call_1"
