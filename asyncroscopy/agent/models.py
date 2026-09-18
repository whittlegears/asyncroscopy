"""Chat-model factory: Ollama, API-key providers, or the LLM Tango device."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterator, Sequence

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from asyncroscopy.agent.config import AgentSettings, default_api_key_env, load_dotenv_if_present, resolve_api_key
from asyncroscopy.agent.messages import langchain_messages_to_openai, openai_message_to_langchain

DEFAULT_LLM_DEVICE = "asyncroscopy/llm/default"


class TangoChatModel(BaseChatModel):
    """LangChain chat model backed by the asyncroscopy LLM Tango device.

    Each generation is one call to the device's ``Complete`` command (an
    OpenAI-style single-step completion), so graphs can run in a notebook or in
    LangGraph Studio while the model itself runs on the LLM computer.

    Limitations: no token streaming (the whole turn arrives at once) and no
    ``with_structured_output``. Tool calls round-trip as OpenAI-style dicts.
    """

    device_name: str = DEFAULT_LLM_DEVICE
    timeout_ms: int = 300_000
    _proxy: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "asyncroscopy-tango"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"device_name": self.device_name}

    def _get_proxy(self) -> Any:
        if self._proxy is None:
            import tango

            proxy = tango.DeviceProxy(self.device_name)
            proxy.set_timeout_millis(int(self.timeout_ms))
            self._proxy = proxy
        return self._proxy

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)

    def _complete(self, messages: list[BaseMessage], tools: list[dict] | None) -> AIMessage:
        request = {"messages": langchain_messages_to_openai(messages), "tools": tools or []}
        raw = self._get_proxy().Complete(json.dumps(request))
        payload = json.loads(raw)
        if "error" in payload:
            raise RuntimeError(f"LLM device {self.device_name} failed: {payload['error'].get('message', payload['error'])}")
        return openai_message_to_langchain(payload["message"])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._complete(messages, kwargs.get("tools"))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = await asyncio.to_thread(self._complete, messages, kwargs.get("tools"))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._complete(messages, kwargs.get("tools"))
        chunk = AIMessageChunk(content=message.content, tool_calls=message.tool_calls)
        yield ChatGenerationChunk(message=chunk)


def build_chat_model(settings: AgentSettings) -> Any:
    """Build a LangChain chat model from ``settings``.

    - ``provider="tango"``: :class:`TangoChatModel` on device ``settings.model``.
    - ``provider="ollama"``: ``ChatOllama`` (needs ``uv sync --extra ollama``),
      unless ``use_init_chat_model`` is set.
    - anything else: ``init_chat_model`` with the API key read from the
      environment (``.env`` is loaded first if python-dotenv is installed).
      Raises ``RuntimeError`` naming the missing variable.
    """
    provider = settings.provider

    if provider == "tango":
        return TangoChatModel(device_name=settings.model or DEFAULT_LLM_DEVICE)

    if provider == "ollama" and not settings.use_init_chat_model:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("Ollama support needs `uv sync --extra agent --extra ollama`.") from exc
        kwargs: dict[str, Any] = {"model": settings.model, "temperature": settings.temperature, "reasoning": False}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        return ChatOllama(**kwargs)

    from langchain.chat_models import init_chat_model

    load_dotenv_if_present()
    api_key = resolve_api_key(settings)
    if provider != "ollama" and not api_key:
        env_name = settings.api_key_env or default_api_key_env(provider)
        raise RuntimeError(
            f"No API key for provider {provider!r}: set {env_name} in the environment or in a .env file "
            "(see .env.example). Never put keys in YAML configs."
        )
    kwargs = {"temperature": settings.temperature}
    if api_key:
        kwargs["api_key"] = api_key
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return init_chat_model(settings.model, model_provider=provider, **kwargs)
