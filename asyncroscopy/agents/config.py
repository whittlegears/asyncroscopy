"""Agent configuration, the chat model, and MCP tool loading."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field

from .mcp_tools import McpServer

DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_ANALYSIS_URL = "http://127.0.0.1:8003/mcp"
CALLER_HEADER = "X-Asyncroscopy-Caller"


@dataclass
class AgentConfig:
    model: str = "qwen3:8b"
    model_provider: str = "ollama"
    num_ctx: int = 8192
    temperature: float = 0.0
    mcp_url: str = DEFAULT_MCP_URL
    # None: the analysis server the instrument endpoint is configured with (endpoint_info), so
    # it is set in one place, the MCP config's mcp.analysis_url; ASYNCROSCOPY_ANALYSIS_URL overrides.
    analysis_url: str | None = None
    caller: str = "agent"
    tool_allow: list[str] = field(default_factory=lambda: ["*"])
    _info: dict | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_env(cls, **overrides) -> AgentConfig:
        env = os.environ
        config = cls(
            model=env.get("ASYNCROSCOPY_MODEL", cls.model),
            model_provider=env.get("ASYNCROSCOPY_MODEL_PROVIDER", cls.model_provider),
            num_ctx=int(env.get("ASYNCROSCOPY_NUM_CTX", cls.num_ctx)),
            mcp_url=env.get("ASYNCROSCOPY_MCP_URL", DEFAULT_MCP_URL),
            analysis_url=env.get("ASYNCROSCOPY_ANALYSIS_URL") or None,
            caller=env.get("ASYNCROSCOPY_CALLER", cls.caller),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        return config

    def chat_model(self):
        """The chat model. Ollama always gets an explicit num_ctx: its default silently truncates tool descriptions."""
        if self.model_provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=self.model, temperature=self.temperature, num_ctx=self.num_ctx, reasoning=False)
        from langchain.chat_models import init_chat_model

        return init_chat_model(model=self.model, model_provider=self.model_provider, temperature=self.temperature)

    def instrument(self) -> McpServer:
        return McpServer(self.mcp_url, {CALLER_HEADER: self.caller})

    def analysis(self) -> McpServer | None:
        """The analysis server: the configured or endpoint-reported one, else the default; None when set to "" (none)."""
        if self.analysis_url == "":
            return None
        return McpServer(self.analysis_url or DEFAULT_ANALYSIS_URL, {CALLER_HEADER: self.caller})

    async def info(self) -> dict:
        """The instrument endpoint's endpoint_info (tool risks, job devices, analysis server); {} if it has none."""
        if self._info is None:
            try:
                result = await self.instrument().call("endpoint_info", {})
                payload = getattr(result, "structured_content", None) if result is not None and not result.is_error else None
                self._info = dict(payload or {})
            except Exception:
                self._info = {}
            if self.analysis_url is None and self._info.get("analysis_url"):
                self.analysis_url = self._info["analysis_url"]  # the MCP config's mcp.analysis_url
        return self._info

    async def tools(self, include_analysis: bool = True, allow: list[str] | None = None):
        """LangChain tools from the instrument endpoint (and the analysis server), filtered by a glob allow-list."""
        try:
            tools = await self.instrument().langchain_tools()
        except Exception as exc:
            raise ConnectionError(f"cannot reach the asyncroscopy MCP endpoint at {self.mcp_url}: {exc}") from exc
        await self.info()
        analysis = self.analysis() if include_analysis else None
        if analysis is not None:
            try:
                tools.extend(await analysis.langchain_tools())
            except Exception as exc:
                print(f"[agents] analysis server {self.analysis_url} unavailable ({exc}); continuing without it")
        patterns = allow or self.tool_allow
        return [t for t in tools if any(fnmatch.fnmatch(t.name, p) for p in patterns)]
