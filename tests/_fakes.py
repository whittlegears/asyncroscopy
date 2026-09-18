"""Test doubles for the agent graphs (require the real LangChain packages)."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AIMessages in order and records what it was asked."""

    responses: list[AIMessage]
    prompts: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        self.bound_tools = list(tools)
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.prompts.append(list(messages))
        if not self.responses:
            return AIMessage(content="(no scripted response left)")
        return self.responses.pop(0)

    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content=message.content, tool_calls=message.tool_calls))


class FakeMicroscope:
    """Fake MCP tool set mirroring the DigitalTwin's acquisition tools."""

    def __init__(self, fail_image: bool = False, fail_spectrum: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_image = fail_image
        self.fail_spectrum = fail_spectrum
        self.data: dict[str, dict] = {}

    def tools(self) -> list[Any]:
        fake = self

        @tool
        def DigitalTwin_acquire_scanned_image(detector_list: list[str]) -> str:
            """Acquire a scanned image and return its Tiled key."""
            fake.calls.append(("image", {"detector_list": detector_list}))
            if fake.fail_image:
                raise RuntimeError("scan generator offline")
            key = f"stem_image_{detector_list[0].upper()}_1.h5"
            fake.data[key] = {
                "key": key,
                "datasets": [{"name": f"image/{detector_list[0].upper()}", "shape": [64, 64], "dtype": "float32", "attrs": {}, "preview": [0.1, 0.2]}],
            }
            return key

        @tool
        def DigitalTwin_acquire_spectrum(detector_name: str) -> str:
            """Acquire a spectrum and return its Tiled key."""
            fake.calls.append(("spectrum", {"detector_name": detector_name}))
            if fake.fail_spectrum:
                raise RuntimeError("no spectrum detector")
            key = f"spectrum_{detector_name}_1.h5"
            fake.data[key] = {
                "key": key,
                "datasets": [{"name": "spectrum", "shape": [2], "dtype": "float64", "attrs": {"elements": ["Si", "O"]}, "preview": [0.61, 0.39]}],
            }
            return key

        @tool
        def get_data_from_key(key: str, max_values: int = 64) -> dict:
            """Describe a saved acquisition."""
            fake.calls.append(("read", {"key": key, "max_values": max_values}))
            if key not in fake.data:
                raise FileNotFoundError(key)
            return fake.data[key]

        @tool
        def list_devices() -> list[str]:
            """List devices."""
            return ["asyncroscopy/instrument/default"]

        return [DigitalTwin_acquire_scanned_image, DigitalTwin_acquire_spectrum, get_data_from_key, list_devices]
