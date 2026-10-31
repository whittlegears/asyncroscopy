"""General instrument agent: a tool-using model given the skills its task needs.

Unlike ``mapping``, the model chooses every call. It sees a curated set of
tools (not all of them: a 4-8 B model loses track with 100). It does not look
skills up: before every model call ``SkillMiddleware`` puts the skills that
match the request, and those the task has since needed (a failed call, a held
call, data to read), into the system prompt, with notes from the instrument's
memory (``asyncroscopy.agents.memory``). Approval-mode holds come back as tool
errors with a request id, and the prompt says how to wait for them.
"""

from __future__ import annotations

import asyncio
import os

from langchain.agents import create_agent

from asyncroscopy.agents.claims import ClaimCheckMiddleware
from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.guards import EmptyReplyMiddleware, JobWaitMiddleware
from asyncroscopy.agents.memory import Memory
from asyncroscopy.agents.skill_middleware import SkillMiddleware
# Which tools the model sees: one rule set, shared with the endpoint's endpoint_info.agent_tools.
from asyncroscopy.mcp.agent_tools import agent_tool_patterns, for_model, is_stem_endpoint

try:  # import the model client here: an import inside the async factory blocks the event loop
    import langchain_ollama  # noqa: F401
except ImportError:  # other providers load through init_chat_model
    pass

SYSTEM = """You operate a laboratory instrument (or its digital twin) through tools.

Rules:
- Lengths are metres unless a tool says otherwise. Each tool's description gives its arguments.
- For a long experiment use run_job and wait_for_job instead of looping over single actions.
- If a tool says APPROVAL REQUIRED with a request_id, call approval_status until it is approved, then repeat the same call once. If denied or expired, stop and tell the user.
- Data come back as keys; look at them with {readers}, never ask for raw arrays.
- Never end your turn by saying what you will do next: make that tool call now. Fix a refused call and call again; stop only when the task is done or impossible.
- Every value the user gives is theirs: if the instrument refuses it as out of range, tell the user the limit and stop; never set a different value in its place.
- Do every step the user asked for, in order, before answering; never say a step was done unless a tool call did it.
- When done, answer briefly: what you did, the keys of what you produced, and anything refused or left undone.{extra}"""

# Measured on qwen3:8b (2026-10-30, benchmarks/results/find_tools_qwen3_8b.md): it did not search with or without
# this rule, and naming every other tool here left it no room to answer; the curated list is what it can use.
SEARCH_RULE = """
- Your tools are a selection. If none fits the request, call find_tools with one or two words (e.g. "holder", "dose", "jobs") and then call_tool with the name it gives; never say the instrument cannot do something before searching."""

STEM_RULES = """
- STEM: a stage position is [x, y, z, alpha, beta] (metres, degrees). Never move the stage or acquire in a loop for a map: use the Mapper job (see the mapping skill).
- STEM: the beam is blanked when idle. Before you acquire an image or spectrum yourself, call <Microscope>_unblank_beam; after the last one, <Microscope>_blank_beam. Jobs do this themselves."""


async def make_graph(config: dict | None = None):
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
    # A caller (the goal graph) may hand over its own AgentConfig, so this agent's calls join its run ledger.
    agent_config = configurable.get("agent_config") or AgentConfig.from_env(
        model=configurable.get("model"),
        caller=configurable.get("caller") or os.environ.get("ASYNCROSCOPY_CALLER", "general-agent"),
    )
    env_tools = [p.strip() for p in os.environ.get("ASYNCROSCOPY_AGENT_TOOLS", "").split(",") if p.strip()]
    chosen = configurable.get("tools") or env_tools
    # Whether this is a STEM, and which skills apply, depend on the whole endpoint, not the curated subset.
    every = await agent_config.tools(allow=["*"])
    offered = {t.name for t in every}
    own = {t.name for t in await agent_config.tools(include_analysis=False, allow=["*"])}
    stem = is_stem_endpoint(own)
    # load_skill stays in, so the middleware can show it while a matched skill does not fit.
    patterns = agent_tool_patterns(own, bool((await agent_config.info()).get("analysis_url")), chosen)
    tools = [t for t in every if for_model(t.name, patterns)]
    if not chosen and not stem:
        print(f"[agents] no STEM on {agent_config.mcp_url}; giving the agent every tool. "
              "Set ASYNCROSCOPY_AGENT_TOOLS to choose fewer (small models lose track with many).")
    print(f"[agents] {len(tools)} tools: {', '.join(sorted(t.name for t in tools))}")
    template = os.environ.get("ASYNCROSCOPY_AGENT_PROMPT") or SYSTEM
    names = {t.name for t in tools}
    readers = " or ".join(r for r in ("view_data", "get_region") if r in names) or "the instrument's own read tools"
    extra = (SEARCH_RULE if {"find_tools", "call_tool"} <= names else "") + (STEM_RULES if stem else "")
    prompt = template.replace("{skills}", "").replace("{extra}", extra).replace("{readers}", readers)
    memory = await asyncio.to_thread(Memory.from_env)
    skills = SkillMiddleware(prompt, names, offered, memory=memory.notes_for, num_ctx=agent_config.num_ctx)
    waits = JobWaitMiddleware(tools, deadline=agent_config.deadline)  # a job of any length is one tool round
    # Last, so it wraps the model call itself: a final answer is checked against the run's tool calls.
    claims = ClaimCheckMiddleware(names, enabled=configurable.get("claim_check"))
    return create_agent(agent_config.chat_model(), tools, system_prompt=prompt, middleware=[skills, waits, EmptyReplyMiddleware(), claims])
