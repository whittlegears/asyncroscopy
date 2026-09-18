"""Graph factories for LangGraph Studio (``langgraph.json``).

Each factory receives the run's ``RunnableConfig``; values under
``config["configurable"]`` (``provider``, ``model``, ``mcp_url``,
``skills_dirs``, ``api_key_env``, ``base_url``, ``temperature``) override the
``ASYNCROSCOPY_AGENT_*`` environment variables from ``.env``.

If the MCP server is unreachable the factories still return a compiled graph
(with no tools, or with placeholder tools that explain the problem) so Studio
can render and inspect the graph structure without the microscope stack.

``langgraph dev`` runs these factories under a blocking-call detector, so all
file/sqlite/Tango work (settings, skills registry, model construction) is done
in worker threads via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any

from langchain_core.runnables import RunnableConfig

from asyncroscopy.agent.config import Agent, AgentSettings, load_dotenv_if_present
from asyncroscopy.agent.tools import load_mcp_tools

CONFIGURABLE_KEYS = ("provider", "model", "api_key_env", "base_url", "mcp_url", "mcp_transport", "skills_dirs", "temperature", "max_steps")
MCP_CONNECT_TIMEOUT = 5.0

DEFAULT_STUDIO_AGENTS = [
    Agent(
        name="imaging",
        system_prompt="You acquire scanned images. Use the *_acquire_scanned_image tool with detector_list=['haadf'].",
        tools=["*_acquire_scanned_image", "*_acquire_camera_image", "get_data_from_key", "list_devices"],
        description="Acquires HAADF/camera images.",
    ),
    Agent(
        name="spectroscopy",
        system_prompt="You acquire EDS spectra. Use the *_acquire_spectrum tool with detector_name='eds'.",
        tools=["*_acquire_spectrum", "get_data_from_key", "list_devices"],
        description="Acquires EDS spectra and reports composition.",
    ),
]


def settings_from_config(config: RunnableConfig | None) -> AgentSettings:
    load_dotenv_if_present()
    settings = AgentSettings.from_env()
    configurable = (config or {}).get("configurable") or {}
    overrides = {key: configurable.get(key) for key in CONFIGURABLE_KEYS if key in configurable}
    return settings.with_overrides(overrides)


async def tools_or_empty(settings: AgentSettings) -> list[Any]:
    """Load MCP tools, or return ``[]`` with a warning if the server is unreachable."""
    if not settings.mcp_url:
        return []
    try:
        return await load_mcp_tools(settings.mcp_url, settings.mcp_transport, timeout=MCP_CONNECT_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - Studio should still show the graph
        warnings.warn(f"MCP server {settings.mcp_url} unavailable ({type(exc).__name__}: {exc}); building graph without tools.", stacklevel=2)
        return []


def placeholder_tools() -> list[Any]:
    """Stand-ins for the acquisition tools so the survey graph renders without MCP."""
    from langchain_core.tools import StructuredTool

    def _unavailable(*_: Any, **__: Any) -> str:
        raise RuntimeError("MCP server not connected: start run_servers.py and run_mcp.py, then reload the graph.")

    def acquire_scanned_image(detector_list: list[str]) -> str:
        """Placeholder for <DeviceClass>_acquire_scanned_image."""
        return _unavailable()

    def acquire_spectrum(detector_name: str) -> str:
        """Placeholder for <DeviceClass>_acquire_spectrum."""
        return _unavailable()

    def get_data_from_key(key: str, max_values: int = 64) -> dict:
        """Placeholder for get_data_from_key."""
        return _unavailable()

    return [
        StructuredTool.from_function(acquire_scanned_image, name="Placeholder_acquire_scanned_image"),
        StructuredTool.from_function(acquire_spectrum, name="Placeholder_acquire_spectrum"),
        StructuredTool.from_function(get_data_from_key, name="get_data_from_key"),
    ]


def _registry(settings: AgentSettings) -> Any:
    from asyncroscopy.agent.skills.registry import SkillRegistry

    return SkillRegistry.from_dirs(settings.resolved_skills_dirs)


def _model(settings: AgentSettings) -> Any:
    from asyncroscopy.agent.models import build_chat_model

    return build_chat_model(settings)


async def make_react_graph(config: RunnableConfig | None = None) -> Any:
    """Skills-aware ReAct agent over all MCP tools."""
    from asyncroscopy.agent.graphs.react import build_react_graph

    settings = await asyncio.to_thread(settings_from_config, config)
    tools = await tools_or_empty(settings)

    def build() -> Any:
        return build_react_graph(_model(settings), tools, _registry(settings))

    return await asyncio.to_thread(build)


async def make_supervisor_graph(config: RunnableConfig | None = None) -> Any:
    """Supervisor swarm; agents come from ``ASYNCROSCOPY_AGENT_STARTUP_AGENTS`` (JSON list) or defaults."""
    import os

    from asyncroscopy.agent.graphs.react import build_react_graph
    from asyncroscopy.agent.graphs.supervisor import build_supervisor_graph
    from asyncroscopy.agent.streaming import stream_agent
    from asyncroscopy.agent.tools import filter_tools

    settings = await asyncio.to_thread(settings_from_config, config)
    tools = await tools_or_empty(settings)

    def build() -> Any:
        model = _model(settings)
        registry = _registry(settings)
        raw_agents = os.environ.get("ASYNCROSCOPY_AGENT_STARTUP_AGENTS")
        agents = [Agent(**item) for item in json.loads(raw_agents)] if raw_agents else DEFAULT_STUDIO_AGENTS

        def build_worker(agent: Agent) -> Any:
            return build_react_graph(
                model, filter_tools(tools, agent.tools), registry, system_prompt=agent.system_prompt, name=agent.name
            )

        return build_supervisor_graph(model, agents, build_worker, stream_agent)

    return await asyncio.to_thread(build)


async def make_image_eds_survey_graph(config: RunnableConfig | None = None) -> Any:
    """Deterministic image-then-EDS survey workflow."""
    from asyncroscopy.agent.graphs.workflows.image_eds_survey import build_image_eds_survey_graph
    from asyncroscopy.agent.tools import find_tool

    settings = await asyncio.to_thread(settings_from_config, config)
    tools = await tools_or_empty(settings)

    def build() -> Any:
        graph_tools = tools
        try:
            find_tool(tools, "*_acquire_scanned_image")
            find_tool(tools, "*_acquire_spectrum")
            find_tool(tools, "get_data_from_key")
        except KeyError:
            warnings.warn("Acquisition tools not found on the MCP server; using placeholder tools.", stacklevel=2)
            graph_tools = placeholder_tools()

        model = None
        try:
            model = _model(settings)
        except Exception as exc:  # noqa: BLE001 - summary falls back to the deterministic report
            warnings.warn(f"No chat model for the survey summary ({exc}); using the deterministic report.", stacklevel=2)
        return build_image_eds_survey_graph(graph_tools, model=model)

    return await asyncio.to_thread(build)
