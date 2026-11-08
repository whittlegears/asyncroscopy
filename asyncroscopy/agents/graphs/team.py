"""A small team of agents: a planner splits the request, specialists do the steps, a reporter answers.

Small local models lose track with many tools and long instructions, so no
single agent here gets both. The planner (one model call, JSON out) splits
the request into steps for named specialists:

* ``operator``: the instrument (stage, beam, acquisitions, job devices such as the Mapper, approvals);
* ``analyst``: stored data and maps (list_data, view_data, stitching, scoring, points of interest).

Each specialist is a tool-using agent that holds only its role's tools,
discovered from the live endpoints by the role patterns of the agent profile
(its ``team`` section; so renamed or extra tools still land in the right role). Before each of its model calls the
skills matching its step, and those its work has since needed, are put in
its prompt with notes from the instrument's memory
(``asyncroscopy.agents.skill_middleware``, ``asyncroscopy.agents.memory``);
which skills each step applied is kept in ``skills_applied``. Steps pass
on what they produced as facts taken from the servers' replies, not from the
model's prose. A step that fails is retried once with the error as feedback.
The reporter writes the answer from those facts only.

``run_team(request)`` returns ``(answer, trace, skills_applied)``; the trace is
the list of tool calls with their replies, which ``asyncroscopy.agents.goals`` checks.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from asyncroscopy import settings
from asyncroscopy.agents.catalog import ToolCatalog
from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.goals import Call, Facts, trace_from_messages
from asyncroscopy.agents.graphs import run_keeping_partial
from asyncroscopy.agents.memory import Memory
from asyncroscopy.agents.middleware import build_middleware
from asyncroscopy.agents.profile import TeamTools, load_profile
from asyncroscopy.agents.skill_middleware import recursion_limit

try:  # import the model client here: an import inside the async factory blocks the event loop
    import langchain_ollama  # noqa: F401
except ImportError:
    pass

log = logging.getLogger(__name__)

MAX_STEPS = 6
STEP_TOOL_ROUNDS = 25  # tool calls one step may make (a long map is many wait_for_job calls)
STEP_RECURSION = recursion_limit(STEP_TOOL_ROUNDS)
ROLE_PROMPTS = {
    "operator": """You run the instrument for one step of a larger task. Tools act on the instrument or its digital twin.
- Lengths are metres. A map is one Mapper job: run_job, then wait_for_job until done (its reply has the keys). Never a loop of moves.
- If a call says APPROVAL REQUIRED, follow approval_status until approved, then repeat the call once; if denied, stop.
- Do only your step. Another agent stitches and analyses data: hand on the keys (run key, job id, spectrum keys).
- Fix a refused call from its message and call again; never repeat a failing call unchanged.
- Finish with one short paragraph: what you did and the keys you got.""",
    "analyst": """You analyse stored data for one step of a larger task. You cannot move or acquire anything.
- Data are keys; look with view_data (format text for numbers and peaks) and list_data; never ask for raw arrays.
- Stitching (assemble_map) and scoring (score_map) are jobs: call wait_for_job with their job_id until done (its reply has the results).
- Fix a refused call from its message and call again; never repeat a failing call unchanged.
- Finish with one short paragraph: what you found and the keys (a map key contains _map).""",
}
PLAN_PROMPT = """Split the user's request into steps for these agents. Reply with JSON only:
{{"steps": [{{"agent": "<name>", "task": "<what to do, with every number and unit from the request>"}}]}}
Agents:
{agents}
Rules: use as few steps as the request needs (at most {max_steps}); an analysis step comes after the step that records the data;
a map request is an operator step (run the map) followed by an analyst step (stitch it; find regions if asked)."""
REPORT_PROMPT = """Write the final answer to the user's request from the notes below. Use only keys and numbers that appear in the facts or notes; never invent a key.
Say briefly what was done, the keys of what was produced, and anything refused, denied or left undone."""
AGENT_LINES = {
    "operator": "operator: runs the instrument: moves, images, spectra, map jobs (Mapper), approvals",
    "analyst": "analyst: reads and analyses stored data: list/view data, stitch a map run, score regions, points of interest",
}


class TeamState(TypedDict, total=False):
    request: str
    plan: list
    step: int
    notes: list
    trace: list
    retried: bool
    answer: str
    skills_applied: list  # {"step", "agent", "skill", "reason"} for every skill a specialist was given


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object")
    return json.loads(match.group(0))


def role_of(name: str, team: TeamTools, catalog: ToolCatalog, stem: bool = True, analysis_waits: bool = True) -> str | None:
    """Which specialist a tool belongs to: a role, "shared" (both), or None (nobody).

    ``team``: the profile's split. On a STEM endpoint the operator gets a curated set (a small model loses
    track with ~80 microscope tools); on any other instrument it also gets every device command (``catalog``:
    what the endpoint's tools are), as ``general`` does. ``analysis_waits``: wait_for_job also waits on
    analysis jobs (the endpoint has mcp.analysis_url); when it does not, the analyst gets
    analysis_job_status and analysis_job_results instead.
    """
    if not analysis_waits and name in ("analysis_job_status", "analysis_job_results"):
        return "analyst"
    if any(fnmatch.fnmatchcase(name, p) for p in team.hidden):
        return None
    if any(fnmatch.fnmatchcase(name, p) for p in team.shared):
        return "shared"
    for role, patterns in team.roles.items():
        if any(fnmatch.fnmatchcase(name, p) for p in patterns):
            return role
    if not stem and "operator" in team.roles and catalog.is_device_tool(name):
        return "operator"
    return None


def heuristic_plan(request: str, roles: set[str]) -> list[dict]:
    """A plan without the model, from the request's words (used when the planner's reply is unusable)."""
    text = request.lower()
    acts = re.search(r"\bmap\b|\bmapping\b|acquire|take an?\b|capture|move|record|survey|image of|spectrum (?:at|of|here)", text)
    looks = re.search(r"\bmap\b|interesting|regions?|stitch|analy[sz]e|look at|inspect|what is in|peaks?|elements?|composition|latest|list", text)
    steps = []
    if acts and "operator" in roles:
        steps.append({"agent": "operator", "task": request})
    if looks and "analyst" in roles:
        steps.append({"agent": "analyst", "task": request if not steps else f"For this request: {request} -- stitch, inspect or score what the operator recorded."})
    return steps or [{"agent": sorted(roles)[0], "task": request}]


class TeamGraph:
    def __init__(self, config: AgentConfig | None = None, memory: Memory | None = None) -> None:
        self.config = config or AgentConfig.from_env(caller=settings.get().caller or "team")
        self.memory = memory or Memory.from_env()
        self.team = load_profile(self.config.profile).team
        self.catalog = ToolCatalog.unknown()  # what the endpoint's tools are (endpoint_info), read with the tools
        self._tools: dict[str, list] | None = None
        self._all_names: set[str] = set()

    async def tools_by_role(self) -> dict[str, list]:
        if self._tools is None:
            tools = await self.config.tools(allow=["*"])
            self._all_names = {t.name for t in tools}
            info = await self.config.info()
            self.catalog = ToolCatalog.from_info(info)
            stem = bool(info.get("stem_devices"))
            self.memory.seed_instrument(self.catalog, stem=stem)
            waits = bool(info.get("analysis_url"))  # else the analyst needs analysis_job_*
            role = lambda name: role_of(name, self.team, self.catalog, stem, waits)  # noqa: E731
            self._tools = {r: [t for t in tools if role(t.name) in (r, "shared")] for r in self.team.roles}
            self._tools = {r: ts for r, ts in self._tools.items() if any(role(t.name) == r for t in ts)}
        return self._tools

    def specialist(self, role: str, tools: list, task: str, notes: list[str]) -> CompiledStateGraph:
        """A tool-using agent for one step: its role's tools and rules, the notes so far, and (each model call) the skills and memory that apply."""
        names = {t.name for t in tools}
        known = ("\n\nWhat earlier steps produced:\n" + "\n".join(notes[-4:])) if notes else ""
        prompt = ROLE_PROMPTS[role] + known
        # The claim check matters here too: the step's answer becomes the next step's notes.
        middleware = build_middleware(self.config.middleware, prompt=prompt, tool_names=names, offered=self._all_names or names, tools=tools,
                                      memory=self.memory.notes_for, num_ctx=self.config.num_ctx, deadline=self.config.deadline,
                                      catalog=self.catalog)
        return create_agent(self.config.chat_model(), tools, system_prompt=prompt, middleware=middleware)

    # ------------------------------------------------------------------ nodes
    async def plan(self, state: TeamState) -> TeamState:
        roles = set(await self.tools_by_role())
        if not roles:
            return {"plan": [], "notes": ["no instrument or analysis tools were found on the endpoints"], "step": 0, "trace": [], "skills_applied": []}
        agents = "\n".join(line for r, line in AGENT_LINES.items() if r in roles)
        try:
            reply = await self.config.chat_model().ainvoke([SystemMessage(PLAN_PROMPT.format(agents=agents, max_steps=MAX_STEPS)), HumanMessage(state["request"])])
            steps = _parse_json(reply.content if isinstance(reply.content, str) else str(reply.content)).get("steps") or []
            steps = [{"agent": str(s["agent"]).lower(), "task": str(s["task"])} for s in steps if isinstance(s, dict) and s.get("task")]
            steps = [s for s in steps if s["agent"] in roles][:MAX_STEPS]
        except Exception:
            log.warning("the planner model failed; using the heuristic plan", exc_info=True)
            steps = []
        return {"plan": steps or heuristic_plan(state["request"], roles), "step": 0, "notes": [], "trace": [], "retried": False, "skills_applied": []}

    async def work(self, state: TeamState) -> TeamState:
        step = state["plan"][state["step"]]
        tools = (await self.tools_by_role())[step["agent"]]
        notes = list(state.get("notes", []))
        task = step["task"]
        if state.get("retried"):
            task += f"\n\nThe first try at this step failed: {notes[-1]}\nFix that and finish the step."
            running = Facts.from_calls([Call.from_dict(c) for c in state.get("trace", [])], self.catalog).active_jobs()
            if running:  # never a second job while the first try's is still on the instrument
                task += "\nStill running from the first try: " + ", ".join(f"{j} on {v['device']} ({v['state']})" for j, v in running.items()) + \
                        ". Wait for it with wait_for_job (its reply has the results); do not start another job."
        agent = self.specialist(step["agent"], tools, task, notes)
        # A recursion limit, a timeout or a model crash keeps the calls made so far: a job started
        # before the failure is still on the instrument and the retry must know about it.
        result, failed = await run_keeping_partial(agent, {"messages": [HumanMessage(task)]}, {"recursion_limit": STEP_RECURSION})
        messages, applied = result.get("messages", []), result.get("skills_applied") or []
        calls = trace_from_messages(messages)
        facts = Facts.from_calls(calls, self.catalog)
        said = messages[-1].content if messages and isinstance(messages[-1].content, str) else ""
        if not failed and calls and all(c.error for c in calls):
            failed = f"every call failed; last: {calls[-1].name}: {calls[-1].result[:300]}"
        note = f"[{step['agent']}] {step['task'][:120]} -> " + (f"FAILED: {failed}" if failed else said[:500]) + f" (facts: {facts.summary()})"
        trace = list(state.get("trace", [])) + [c.to_dict() for c in calls]
        skills = list(state.get("skills_applied", [])) + [{"step": state["step"], "agent": step["agent"], **a} for a in applied]
        if failed and not state.get("retried"):
            return {"notes": notes + [note], "trace": trace, "retried": True, "skills_applied": skills}
        return {"notes": notes + [note], "trace": trace, "retried": False, "step": state["step"] + 1, "skills_applied": skills}

    async def report(self, state: TeamState) -> TeamState:
        facts = Facts.from_calls([Call.from_dict(c) for c in state.get("trace", [])], self.catalog)
        notes = "\n".join(state.get("notes", [])) or "(no steps ran)"
        try:
            reply = await self.config.chat_model().ainvoke([
                SystemMessage(REPORT_PROMPT),
                HumanMessage(f"Request: {state['request']}\n\nNotes:\n{notes}\n\nFacts from the servers: {facts.summary()}"),
            ])
            answer = reply.content if isinstance(reply.content, str) else str(reply.content)
        except Exception:
            log.warning("the report model failed; answering with the step notes", exc_info=True)
            answer = ""
        return {"answer": answer.strip() or f"Steps run:\n{notes}\nFacts: {facts.summary()}"}

    # ------------------------------------------------------------------ graph
    def build(self) -> CompiledStateGraph:
        graph = StateGraph(TeamState)
        graph.add_node("plan", self.plan)
        graph.add_node("work", self.work)
        graph.add_node("report", self.report)
        more = lambda s: "work" if s.get("step", 0) < len(s.get("plan", [])) else "report"
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", more, ["work", "report"])
        graph.add_conditional_edges("work", more, ["work", "report"])
        graph.add_edge("report", END)
        return graph.compile()


async def run_team(request: str, config: AgentConfig | None = None) -> tuple[str, list[Call], list[dict]]:
    """(answer, tool calls with replies, skills applied) for one request."""
    state = await TeamGraph(config).build().ainvoke({"request": request}, {"recursion_limit": 4 * MAX_STEPS + 6})
    return state["answer"], [Call.from_dict(c) for c in state.get("trace", [])], state.get("skills_applied", [])


def make_graph(config: dict | None = None) -> CompiledStateGraph:
    """Graph factory for LangGraph Studio (input: {"request": "..."})."""
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    return TeamGraph(AgentConfig.from_env(model=configurable.get("model"), caller=configurable.get("caller") or "team")).build()
