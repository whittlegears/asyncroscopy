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
import fnmatch
import os

from langchain.agents import create_agent

from asyncroscopy.agents.claims import ClaimCheckMiddleware
from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.guards import JobWaitMiddleware
from asyncroscopy.agents.memory import Memory
from asyncroscopy.agents.skill_middleware import SkillMiddleware

try:  # import the model client here: an import inside the async factory blocks the event loop
    import langchain_ollama  # noqa: F401
except ImportError:  # other providers load through init_chat_model
    pass

# Tools every instrument endpoint and the pyTEMlib analysis server offer. Left out, from the trial
# traces (benchmarks/results/tool_usage_analysis.md): calls that always come as a pair or are always
# made first. run_job is reserve+start, wait_for_job also waits on analysis jobs and returns results,
# and the skills are put in the prompt (load_skill is shown to the model only while one does not fit).
GENERIC_TOOLS = [
    "approval_status", "run_job", "wait_for_job", "list_devices", "list_data", "view_data",
    "assemble_map", "score_map", "map_info", "get_region", "list_pois", "add_poi", "list_scorers",
]
# A curated STEM set: a 4-8 B model loses track with all ~80 microscope tools.
STEM_TOOLS = [
    "*_get_stage", "*_move_stage", "*_get_parameters", "*_acquire_scanned_image", "*_acquire_spectrum",
    "*_set_fov", "*_get_fov", "*_set_defocus", "*_place_beam", "*_blank_beam", "*_unblank_beam", "Mapper*_abort",
]
DEFAULT_TOOLS = STEM_TOOLS + GENERIC_TOOLS
# Only when the instrument endpoint has no mcp.analysis_url (so wait_for_job cannot wait on stitching
# and scoring jobs) does an agent need the analysis server's own wait and results tools.
ANALYSIS_WAITS = ["analysis_job_status", "analysis_job_results"]
NOT_FOR_MODELS = ("list_skills", "endpoint_info")
# An endpoint is a STEM when one device both scans images and blanks a beam (an AFM scans too),
# or it has a Mapper: move_stage, set_fov and the like are common to many instruments.
def is_stem_endpoint(names: set[str]) -> bool:
    return any(fnmatch.fnmatch(n, "Mapper*_reserve") for n in names) or any(
        n.endswith("_acquire_scanned_image") and f"{n[: -len('_acquire_scanned_image')]}_blank_beam" in names for n in names
    )

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

STEM_RULES = """
- STEM: a stage position is [x, y, z, alpha, beta] (metres, degrees). Never move the stage or acquire in a loop for a map: use the Mapper job (see the mapping skill)."""


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
    offered = {t.name for t in await agent_config.tools(allow=["*"])}
    defaults = DEFAULT_TOOLS if (await agent_config.info()).get("analysis_url") else DEFAULT_TOOLS + ANALYSIS_WAITS
    tools = await agent_config.tools(allow=chosen or defaults)
    stem = is_stem_endpoint(offered)
    if not chosen and not stem:
        # The analysis server's tools stitch and score STEM maps; another instrument gets its own tools only.
        tools = [t for t in await agent_config.tools(include_analysis=False, allow=["*"]) if not t.name.endswith(("_State", "_Status"))]
        print(f"[agents] no STEM on {agent_config.mcp_url}; giving the agent every tool. "
              "Set ASYNCROSCOPY_AGENT_TOOLS to choose fewer (small models lose track with many).")
    # Skills come to the agent (it need not list them); endpoint_info is for agent programs, not models.
    tools = [t for t in tools if t.name not in NOT_FOR_MODELS]
    # load_skill stays registered so the middleware can show it while a matched skill does not fit.
    tools += [t for t in await agent_config.tools(allow=["load_skill"]) if t.name not in {u.name for u in tools}]
    print(f"[agents] {len(tools)} tools: {', '.join(sorted(t.name for t in tools))}")
    template = os.environ.get("ASYNCROSCOPY_AGENT_PROMPT") or SYSTEM
    names = {t.name for t in tools}
    readers = " or ".join(r for r in ("view_data", "get_region") if r in names) or "the instrument's own read tools"
    prompt = template.replace("{skills}", "").replace("{extra}", STEM_RULES if stem else "").replace("{readers}", readers)
    memory = await asyncio.to_thread(Memory.from_env)
    skills = SkillMiddleware(prompt, names, offered, memory=memory.notes_for, num_ctx=agent_config.num_ctx)
    waits = JobWaitMiddleware(tools, deadline=agent_config.deadline)  # a job of any length is one tool round
    # Last, so it wraps the model call itself: a final answer is checked against the run's tool calls.
    claims = ClaimCheckMiddleware(names, enabled=configurable.get("claim_check"))
    return create_agent(agent_config.chat_model(), tools, system_prompt=prompt, middleware=[skills, waits, claims])
