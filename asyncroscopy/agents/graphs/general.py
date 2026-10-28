"""General instrument agent: a tool-using model that loads skills on demand.

Unlike ``mapping``, the model chooses every call. It sees a curated set of
tools (not all of them: a 4-8 B model loses track with 100) and the names of
the available skills; the system prompt tells it to load the matching skill
before a multi-step task. Approval-mode holds come back as tool errors with
a request id, and the prompt says how to wait for them.
"""

from __future__ import annotations

import asyncio
import os

from langchain.agents import create_agent

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.skills import list_skills

try:  # import the model client here: an import inside the async factory blocks the event loop
    import langchain_ollama  # noqa: F401
except ImportError:  # other providers load through init_chat_model
    pass

DEFAULT_TOOLS = [
    "*_get_stage", "*_move_stage", "*_get_parameters", "*_acquire_scanned_image", "*_acquire_spectrum",
    "*_set_fov", "*_get_fov", "*_set_defocus", "*_place_beam", "*_blank_beam", "*_unblank_beam",
    "Mapper_*", "Approvals_get_mode", "approval_status", "list_skills", "load_skill", "list_devices", "get_data_from_key",
    "assemble_map", "score_map", "analysis_job_status", "analysis_job_results", "map_info", "get_region", "list_pois", "add_poi", "list_scorers",
]

SYSTEM = """You operate laboratory instruments (for example a scanning transmission electron microscope, or its digital twin) through tools.

Rules:
- Lengths are metres, tilts degrees. Each tool's description gives its arguments; a STEM stage position is [x, y, z, alpha, beta].
- Before a multi-step task, call load_skill with the matching skill and follow it. Skills: {skills}
- Never move the stage or acquire in a loop for a map: use the Mapper job (see the mapping skill).
- If a tool says APPROVAL REQUIRED with a request_id, call approval_status until it is approved, then repeat the same call once. If denied or expired, stop and tell the user.
- Data come back as keys; look at images with get_region or get_data_from_key, never ask for raw arrays.
- Never end your turn by saying what you will do next: make that tool call now. Fix a refused call and call again; stop only when the task is done or impossible.
- When done, answer briefly: what you did, the keys of what you produced, and anything refused or left undone."""


async def make_graph(config: dict | None = None):
    """Graph factory for LangGraph Studio and the agent server (tools are discovered when the graph is built)."""
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    agent_config = AgentConfig.from_env(
        model=configurable.get("model"),
        caller=configurable.get("caller") or os.environ.get("ASYNCROSCOPY_CALLER", "general-agent"),
    )
    # Other instruments: list their tool patterns in ASYNCROSCOPY_AGENT_TOOLS (comma-separated globs).
    env_tools = [p.strip() for p in os.environ.get("ASYNCROSCOPY_AGENT_TOOLS", "").split(",") if p.strip()]
    tools = await agent_config.tools(allow=configurable.get("tools") or env_tools or DEFAULT_TOOLS)
    # Reading skill files is blocking I/O; keep it off the event loop (LangGraph Studio enforces this).
    available = await asyncio.to_thread(list_skills)
    skills = "; ".join(f"{s.name}: {s.description}" for s in available if s.available) or "none"
    return create_agent(agent_config.chat_model(), tools, system_prompt=SYSTEM.format(skills=skills))
