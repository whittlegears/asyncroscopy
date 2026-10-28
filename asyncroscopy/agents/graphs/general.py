"""General instrument agent: a tool-using model that loads skills on demand.

Unlike ``mapping``, the model chooses every call. It sees a curated set of
tools (not all of them: a 4-8 B model loses track with 100) and the names of
the available skills; the system prompt tells it to load the matching skill
before a multi-step task. Approval-mode holds come back as tool errors with
a request id, and the prompt says how to wait for them.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os

from langchain.agents import create_agent

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.skills import list_skills

try:  # import the model client here: an import inside the async factory blocks the event loop
    import langchain_ollama  # noqa: F401
except ImportError:  # other providers load through init_chat_model
    pass

# Tools every instrument endpoint and the pyTEMlib analysis server offer.
GENERIC_TOOLS = [
    "Approvals_get_mode", "approval_status", "wait_for_job", "list_skills", "load_skill", "list_devices", "get_data_from_key",
    "assemble_map", "score_map", "analysis_job_status", "analysis_job_results", "map_info", "get_region", "list_pois", "add_poi", "list_scorers",
]
# A curated STEM set: a 4-8 B model loses track with all ~80 microscope tools.
STEM_TOOLS = [
    "*_get_stage", "*_move_stage", "*_get_parameters", "*_acquire_scanned_image", "*_acquire_spectrum",
    "*_set_fov", "*_get_fov", "*_set_defocus", "*_place_beam", "*_blank_beam", "*_unblank_beam", "Mapper_*",
]
DEFAULT_TOOLS = STEM_TOOLS + GENERIC_TOOLS

SYSTEM = """You operate a laboratory instrument (or its digital twin) through tools.

Rules:
- Lengths are metres unless a tool says otherwise. Each tool's description gives its arguments.
- Before a multi-step task, call load_skill with the matching skill and follow it. Skills: {skills}
- For a long experiment use the instrument's job tools (*_reserve, *_start, *_job_status, *_results, *_release) instead of looping over single actions.
- If a tool says APPROVAL REQUIRED with a request_id, call approval_status until it is approved, then repeat the same call once. If denied or expired, stop and tell the user.
- Data come back as keys; look at them with get_region or get_data_from_key, never ask for raw arrays.
- Never end your turn by saying what you will do next: make that tool call now. Fix a refused call and call again; stop only when the task is done or impossible.
- When done, answer briefly: what you did, the keys of what you produced, and anything refused or left undone.{extra}"""

STEM_RULES = """
- STEM: a stage position is [x, y, z, alpha, beta] (metres, degrees). Never move the stage or acquire in a loop for a map: use the Mapper job (see the mapping skill)."""


async def make_graph(config: dict | None = None):
    """Graph factory for LangGraph Studio and the agent server (tools are discovered when the graph is built).

    Tools: ``configurable["tools"]``, else ASYNCROSCOPY_AGENT_TOOLS (comma-separated globs), else the
    STEM set; when that set matches no instrument tool (another kind of instrument), every tool on
    the endpoint. ASYNCROSCOPY_AGENT_PROMPT replaces the system prompt ({skills} is filled in).
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    agent_config = AgentConfig.from_env(
        model=configurable.get("model"),
        caller=configurable.get("caller") or os.environ.get("ASYNCROSCOPY_CALLER", "general-agent"),
    )
    env_tools = [p.strip() for p in os.environ.get("ASYNCROSCOPY_AGENT_TOOLS", "").split(",") if p.strip()]
    chosen = configurable.get("tools") or env_tools
    tools = await agent_config.tools(allow=chosen or DEFAULT_TOOLS)
    stem = any(fnmatch.fnmatch(t.name, p) for t in tools for p in STEM_TOOLS)
    if not chosen and not stem:
        tools = await agent_config.tools(allow=["*"])
        print(f"[agents] no STEM tools on {agent_config.mcp_url}; giving the agent all {len(tools)} tools. "
              "Set ASYNCROSCOPY_AGENT_TOOLS to choose fewer (small models lose track with many).")
    # Reading skill files is blocking I/O; keep it off the event loop (LangGraph Studio enforces this).
    available = await asyncio.to_thread(list_skills, {t.name for t in tools})
    skills = "; ".join(f"{s.name}: {s.description}" for s in available if s.available) or "none"
    template = os.environ.get("ASYNCROSCOPY_AGENT_PROMPT") or SYSTEM
    return create_agent(agent_config.chat_model(), tools, system_prompt=template.format(skills=skills, extra=STEM_RULES if stem else ""))
