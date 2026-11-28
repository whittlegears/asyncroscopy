"""General instrument agent: a tool-using model given the skills its task needs.

Unlike ``mapping``, the model chooses every call. It sees a curated set of
tools (not all of them: a 4-8 B model loses track with 100). It does not look
skills up: before every model call ``SkillMiddleware`` puts the skills that
match the request, and those the task has since needed (a failed call, data to
read), into the system prompt, with notes from the instrument's
memory (``asyncroscopy.agents.memory``).
"""

from __future__ import annotations

import asyncio

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from asyncroscopy import settings
from asyncroscopy.core.names import devices_of_kind
from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.graphs import map_rule
from asyncroscopy.agents.memory import Memory
from asyncroscopy.agents.middleware import build_middleware
from asyncroscopy.mcp.agent_tools import agent_tool_patterns, for_model

try:  # import the model client here because an import inside the async factory blocks the event loop
    import langchain_ollama  # noqa: F401
except ImportError:  # other providers load through init_chat_model
    pass

SYSTEM = """You operate a laboratory instrument (or its digital twin) through tools.

Rules:
- Lengths are metres unless a tool says otherwise. Each tool's description gives its arguments.
- For a long experiment use run_job and wait_for_job instead of looping over single actions.
- Data come back as keys; look at them with {readers}, never ask for raw arrays.
- Never end your turn by saying what you will do next: make that tool call now. Fix a refused call and call again; stop only when the task is done or impossible.
- Every value the user gives is theirs: if the instrument refuses it as out of range, tell the user the limit and stop; never set a different value in its place.
- Do every step the user asked for, in order, before answering; never say a step was done unless a tool call did it.
- When done, answer briefly: what you did, the keys of what you produced, and anything refused or left undone.{extra}"""

# Kept short: naming every other tool leaves a small model no room to answer.
SEARCH_RULE = """
- Your tools are a selection. If none fits the request, call find_tools with one or two words (e.g. "holder", "dose", "jobs") and then call_tool with the name it gives; never say the instrument cannot do something before searching."""

STEM_RULES = """
- STEM: a stage position is [x, y, z, alpha, beta] (metres, degrees).{map_rule}
- STEM: the beam is blanked when idle. Before you acquire an image or spectrum yourself, call <Microscope>_unblank_beam; after the last one, <Microscope>_blank_beam. Jobs do this themselves."""


async def make_graph(config: dict | None = None) -> CompiledStateGraph:
    """Graph factory for LangGraph Studio and the agent server (tools are discovered when the graph is built).

    Tools: ``configurable["tools"]``, else ASYNCROSCOPY_AGENT_TOOLS (comma-separated globs), else the
    STEM set when the endpoint has a STEM (a scanned-image or Mapper tool), otherwise every tool on
    the endpoint. ASYNCROSCOPY_AGENT_PROMPT replaces the system prompt; in it the literal text
    {readers} becomes the tools that look at data (view_data, get_region) and {extra} the STEM rules
    on a STEM endpoint (other braces, such as JSON examples, are left alone). The matching skills
    and memory notes are appended to it for each model call; a {skills} placeholder is ignored.
    A final answer goes through the claim check (``asyncroscopy.agents.claims``): ``configurable["claim_check"]``
    (True/False), else ASYNCROSCOPY_CLAIM_CHECK, turns it on or off (on by default).
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    agent_config = configurable.get("agent_config") or AgentConfig.from_env(
        model=configurable.get("model"),
        caller=configurable.get("caller") or settings.get().caller or "general-agent",
    )
    env_tools = list(settings.get().agent_tools)
    chosen = configurable.get("tools") or env_tools
    every = await agent_config.tools(allow=["*"])
    offered = {t.name for t in every}
    own = {t.name for t in await agent_config.tools(include_analysis=False, allow=["*"])}
    info = await agent_config.info()  # what discovery found, such as the microscopes, the Mappers and each tool's role
    stem = bool(devices_of_kind(info, "stem"))
    lists = agent_config.tool_lists()
    patterns = agent_tool_patterns(own, bool((info.get("urls") or {}).get("analysis")), chosen, lists, stem=stem, maps=bool(devices_of_kind(info, "map")))
    tools = [t for t in every if for_model(t.name, patterns, lists)]
    if not chosen and not stem:
        print(f"[agents] no STEM on {agent_config.mcp_url}; giving the agent every tool. "
              "Set ASYNCROSCOPY_AGENT_TOOLS to choose fewer (a small model loses track with many).")
    print(f"[agents] {len(tools)} tools: {', '.join(sorted(t.name for t in tools))}")
    template = settings.get().agent_prompt or SYSTEM
    names = {t.name for t in tools}
    readers = " or ".join(r for r in ("view_data", "get_region") if r in names) or "the instrument's own read tools"
    stem_rules = STEM_RULES.replace("{map_rule}", map_rule(bool(devices_of_kind(info, "map")), names))
    extra = (SEARCH_RULE if {"find_tools", "call_tool"} <= names else "") + (stem_rules if stem else "")
    prompt = template.replace("{skills}", "").replace("{extra}", extra).replace("{readers}", readers)
    memory = await asyncio.to_thread(Memory.from_env)
    catalog = await agent_config.catalog()
    await asyncio.to_thread(memory.seed_instrument, catalog, stem)
    middleware = build_middleware(agent_config.middleware, prompt=prompt, tool_names=names, offered=offered, tools=tools, memory=memory.notes_for,
                                  num_ctx=agent_config.num_ctx, deadline=agent_config.deadline, claim_check=configurable.get("claim_check"),
                                  catalog=catalog, profile=agent_config.agent_profile())
    return create_agent(agent_config.chat_model(), tools, system_prompt=prompt, middleware=middleware)
