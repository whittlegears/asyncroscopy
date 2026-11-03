"""Agent configuration, the chat model, and MCP tool loading."""

from __future__ import annotations

import dataclasses
import fnmatch
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

from .mcp_tools import TOKEN_ENV, McpServer, bearer_header

DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_ANALYSIS_URL = "http://127.0.0.1:8003/mcp"
CALLER_HEADER = "X-Asyncroscopy-Caller"
RUN_HEADER = "X-Asyncroscopy-Run"  # one agent run: run_job finds a held start again by caller and run

log = logging.getLogger(__name__)


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
    # A bearer token from WG1's token service (or startup_scripts/dev_tokens.py), sent as Authorization:
    # Bearer to the instrument and analysis endpoints. With mcp.auth the endpoint takes the caller from the
    # token's subject and ignores the caller header. None: no token (an endpoint without mcp.auth).
    token: str | None = field(default=None, repr=False)
    tool_allow: list[str] = field(default_factory=lambda: ["*"])
    # One id per agent run, sent with every call: agents open a new MCP session per call, so the
    # server recognises a retried run_job by caller and run id.
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Every instrument call of this run as its reply arrives (see McpServer), so a run stopped part
    # way still knows which jobs it started.
    ledger: list = field(default_factory=list, repr=False, compare=False)
    # time.monotonic() after which an agent stops waiting on jobs (a goal's time budget); None: no limit.
    deadline: float | None = None
    _info: dict | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> AgentConfig:
        env = os.environ
        config = cls(
            model=env.get("ASYNCROSCOPY_MODEL", cls.model),
            model_provider=env.get("ASYNCROSCOPY_MODEL_PROVIDER", cls.model_provider),
            num_ctx=int(env.get("ASYNCROSCOPY_NUM_CTX", cls.num_ctx)),
            mcp_url=env.get("ASYNCROSCOPY_MCP_URL", DEFAULT_MCP_URL),
            analysis_url=env.get("ASYNCROSCOPY_ANALYSIS_URL") or None,
            caller=env.get("ASYNCROSCOPY_CALLER", cls.caller),
            token=env.get(TOKEN_ENV) or None,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        return config

    def chat_model(self) -> BaseChatModel:
        """The chat model. Ollama always gets an explicit num_ctx: its default silently truncates tool descriptions."""
        if self.model_provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=self.model, temperature=self.temperature, num_ctx=self.num_ctx, reasoning=False)
        from langchain.chat_models import init_chat_model

        return init_chat_model(model=self.model, model_provider=self.model_provider, temperature=self.temperature)

    def for_run(self, **changes: Any) -> AgentConfig:
        """A copy for another agent run: a new run id and ledger (and any other changes, e.g. caller)."""
        return dataclasses.replace(self, run_id=uuid.uuid4().hex[:12], ledger=[], **changes)

    def _headers(self) -> dict[str, str]:
        return {CALLER_HEADER: self.caller, RUN_HEADER: self.run_id, **bearer_header(self.token)}

    def instrument(self) -> McpServer:
        return McpServer(self.mcp_url, self._headers(), ledger=self.ledger)

    def analysis(self) -> McpServer | None:
        """The analysis server: the configured or endpoint-reported one, else the default; None when there is none ("")."""
        if self.analysis_url == "":
            return None
        return McpServer(self.analysis_url or DEFAULT_ANALYSIS_URL, self._headers(), ledger=self.ledger)

    async def info(self) -> dict:
        """The instrument endpoint's endpoint_info (tool risks, job devices, analysis server); {} if it has none."""
        if self._info is None:
            try:
                result = await McpServer(self.mcp_url, self._headers()).call("endpoint_info", {})
                payload = getattr(result, "structured_content", None) if result is not None and not result.is_error else None
                self._info = dict(payload or {})
            except Exception:  # not cached: one transient failure must not hide the risks for the whole run
                log.warning("endpoint_info from %s failed; going on without it", self.mcp_url, exc_info=True)
                return {}
            if self.analysis_url is None and "analysis_url" in self._info:
                # The MCP config's mcp.analysis_url; empty there means this instrument has no analysis server.
                self.analysis_url = self._info["analysis_url"] or ""
        return self._info

    async def tools(self, include_analysis: bool = True, allow: list[str] | None = None) -> list[StructuredTool]:
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
