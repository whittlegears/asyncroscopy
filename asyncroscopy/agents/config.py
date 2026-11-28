"""Agent configuration, the chat model, and MCP tool loading."""

from __future__ import annotations

import dataclasses
import fnmatch
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

from asyncroscopy.agents.catalog import ToolCatalog
from asyncroscopy import settings
from asyncroscopy.core.names import CALLER_HEADER, DEFAULT_ANALYSIS_URL, DEFAULT_MCP_URL, RUN_HEADER, RUN_KEY_HEADER

from asyncroscopy.mcp.agent_tools import ToolLists

from .mcp_tools import McpServer, bearer_header
from .profile import DEFAULT_PROFILE_NAME, AgentProfile, load_profile


log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    # The agent profile holds what is tuned to one model.
    # model, model_provider, num_ctx and middleware take its values when left as None.
    profile: str = DEFAULT_PROFILE_NAME
    model: str | None = None
    model_provider: str | None = None
    num_ctx: int | None = None
    middleware: list[str] | None = None
    temperature: float = 0.0
    mcp_url: str = DEFAULT_MCP_URL
    # None means use the analysis server the instrument endpoint reports, so it is set in one place.
    # ASYNCROSCOPY_ANALYSIS_URL overrides it.
    analysis_url: str | None = None
    caller: str = "agent"
    # A bearer token from the facility's token service or startup_scripts/dev_tokens.py, sent to both endpoints.
    # With mcp.auth the endpoint takes the caller from the token and ignores the caller header.
    # None means no token, for an endpoint without mcp.auth.
    token: str | None = field(default=None, repr=False)
    tool_allow: list[str] = field(default_factory=lambda: ["*"])
    # One id per agent run, sent with every call. Agents open a new MCP session per call,
    # so the server uses the caller and run id to spot a retried run_job.
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # A hardware run of a signed-off workflow sends this secret with every call, so only that run's calls use the
    # run's pre-approvals (``docs/MCP/twin_to_instrument.md``); the run id alone is in replies an agent reads.
    run_key: str | None = field(default=None, repr=False)
    # Every instrument call of this run, added as its reply arrives.
    # A run stopped part way still knows which jobs it started.
    ledger: list = field(default_factory=list, repr=False, compare=False)
    # The time.monotonic value after which an agent stops waiting on jobs. None means no limit.
    deadline: float | None = None
    _info: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        chosen = load_profile(self.profile)
        self.model = self.model or chosen.model
        self.model_provider = self.model_provider or chosen.model_provider
        self.num_ctx = int(self.num_ctx or chosen.num_ctx)
        self.middleware = list(chosen.middleware if self.middleware is None else self.middleware)

    def agent_profile(self) -> AgentProfile:
        """The agent profile: what depends on the model (asyncroscopy.agents.profile)."""
        return load_profile(self.profile)

    def tool_lists(self) -> ToolLists:
        """The profile's tool selection (asyncroscopy.mcp.agent_tools)."""
        return load_profile(self.profile).tools

    @classmethod
    def from_env(cls, **overrides: Any) -> AgentConfig:
        env = settings.get()
        config = cls(
            profile=overrides.pop("profile", None) or env.agent_profile,
            model=env.model,
            model_provider=env.model_provider,
            num_ctx=env.num_ctx,
            mcp_url=env.mcp_url or DEFAULT_MCP_URL,
            analysis_url=env.analysis_url,
            caller=env.caller or cls.caller,
            token=env.mcp_token,
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
        """The instrument endpoint. Only its calls carry the run key: the pre-approvals it unlocks are the instrument's."""
        profile = self.agent_profile()
        headers = {**self._headers(), **({RUN_KEY_HEADER: self.run_key} if self.run_key else {})}
        return McpServer(self.mcp_url, headers, ledger=self.ledger, replies=profile.replies, images=profile.images)

    def analysis(self) -> McpServer | None:
        """The analysis server: the configured or endpoint-reported one, else the default; None when there is none ("")."""
        if self.analysis_url == "":
            return None
        profile = self.agent_profile()
        return McpServer(self.analysis_url or DEFAULT_ANALYSIS_URL, self._headers(), ledger=self.ledger, replies=profile.replies,
                         images=profile.images)

    async def info(self) -> dict:
        """The instrument endpoint's endpoint_info (tool risks, job devices, analysis server); {} if it has none."""
        if self._info is None:
            try:
                result = await McpServer(self.mcp_url, self._headers()).call("endpoint_info", {})
                payload = getattr(result, "structured_content", None) if result is not None and not result.is_error else None
                self._info = dict(payload or {})
            except Exception:  # not cached, so one passing failure does not hide the risks for the whole run
                log.warning("endpoint_info from %s failed; going on without it", self.mcp_url, exc_info=True)
                return {}
            if self.analysis_url is None and "urls" in self._info:
                # An empty mcp.urls.analysis means this instrument has no analysis server.
                self.analysis_url = (self._info["urls"] or {}).get("analysis") or ""
        return self._info

    async def catalog(self) -> ToolCatalog:
        """What the instrument endpoint's tools are (endpoint_info's tools, asked once per run); unknown without it."""
        return ToolCatalog.from_info(await self.info())

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
