"""A small team of agents: a planner splits the request, specialists do the steps, a reporter answers.

Small local models lose track with many tools and long instructions, so no
single agent here gets both. The planner (one model call, JSON out) splits
the request into steps for named specialists:

* ``operator``: the instrument (stage, beam, acquisitions, job devices such as the Mapper, approvals);
* ``analyst``: stored data and maps (list_data, view_data, stitching, scoring, points of interest).

Each specialist is a tool-using agent that holds only its role's tools,
discovered from the live endpoints by role patterns (so renamed or extra
tools still land in the right role). Before each of its model calls the
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
import os
import re
from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.goals import Call, Facts, trace_from_messages
from asyncroscopy.agents.graphs.general import is_stem_endpoint
from asyncroscopy.agents.memory import Memory
from asyncroscopy.agents.skill_middleware import SkillMiddleware

try:  # import the model client here: an import inside the async factory blocks the event loop
    import langchain_ollama  # noqa: F401
except ImportError:
    pass

# Jobs go through run_job (reserve + start) and wait_for_job (which waits on instrument and analysis jobs
# and returns their results), so the envelope's separate reserve/start/status/results/release tools and
# the analysis job tools stay on the endpoint but are not given to a specialist.
ROLES = {
    "operator": [
        "run_job", "*_abort", "*_get_stage", "*_move_stage", "*_get_parameters", "*_acquire_*", "*_set_fov", "*_get_fov",
        "*_set_defocus", "*_place_beam", "*_blank_beam", "*_unblank_beam",
    ],
    "analyst": ["list_data", "view_data", "assemble_map", "score_map", "map_info", "get_region", "list_pois", "add_poi", "list_scorers"],
}
# Both specialists get these. load_skill is shown to a model only while a matched skill did not fit its prompt.
SHARED = ["wait_for_job", "approval_status", "load_skill"]
# Never given to a specialist: covered by run_job / wait_for_job, skills come to the agent, and the operator side of approvals is the operator's.
HIDDEN = [
    "*_reserve", "*_start", "*_job_status", "*_results", "*_release", "analysis_job_*", "list_skills",
    "Approvals_*", "*_State", "*_Status", "*_Init",
]
MAX_STEPS = 6
STEP_RECURSION = 40
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
- Finish with one short paragraph: what you found and the keys (the map key ends in _map).""",
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


def role_of(name: str, stem: bool = True) -> str | None:
    """Which specialist a tool belongs to: a role, "shared" (both), or None (nobody).

    On a STEM endpoint the operator gets a curated set (a small model loses track with ~80
    microscope tools); on any other instrument it gets every device command, as ``general`` does.
    """
    if any(fnmatch.fnmatchcase(name, p) for p in HIDDEN):
        return None
    if any(fnmatch.fnmatchcase(name, p) for p in SHARED):
        return "shared"
    for role in ("analyst", "operator"):  # analyst first: its tools are never the instrument's
        if any(fnmatch.fnmatchcase(name, p) for p in ROLES[role]):
            return role
    device, _, command = name.partition("_")
    if not stem and command and device[:1].isupper():
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
    def __init__(self, config: AgentConfig | None = None, memory: Memory | None = None):
        self.config = config or AgentConfig.from_env(caller=os.environ.get("ASYNCROSCOPY_CALLER", "team"))
        self.memory = memory or Memory.from_env()
        self._tools: dict[str, list] | None = None
        self._all_names: set[str] = set()

    async def tools_by_role(self) -> dict[str, list]:
        if self._tools is None:
            tools = await self.config.tools(allow=["*"])
            self._all_names = {t.name for t in tools}
            self.memory.seed_instrument(self._all_names)
            stem = is_stem_endpoint(self._all_names)
            self._tools = {role: [t for t in tools if role_of(t.name, stem) in (role, "shared")] for role in ROLES}
            self._tools = {role: ts for role, ts in self._tools.items() if any(role_of(t.name, stem) == role for t in ts)}
        return self._tools

    def specialist(self, role: str, tools: list, task: str, notes: list[str]):
        """A tool-using agent for one step: its role's tools and rules, the notes so far, and (each model call) the skills and memory that apply."""
        names = {t.name for t in tools}
        known = ("\n\nWhat earlier steps produced:\n" + "\n".join(notes[-4:])) if notes else ""
        prompt = ROLE_PROMPTS[role] + known
        skills = SkillMiddleware(prompt, names, self._all_names or names, memory=self.memory.notes_for)
        return create_agent(self.config.chat_model(), tools, system_prompt=prompt, middleware=[skills])

    # ------------------------------------------------------------------ nodes
    async def plan(self, state: TeamState) -> TeamState:
        roles = set(await self.tools_by_role())
        if not roles:
            return {"plan": [], "notes": ["no instrument or analysis tools were found on the endpoints"], "step": 0, "trace": [], "skills_applied": []}
        agents = "\n".join(AGENT_LINES[r] for r in ROLES if r in roles)
        try:
            reply = await self.config.chat_model().ainvoke([SystemMessage(PLAN_PROMPT.format(agents=agents, max_steps=MAX_STEPS)), HumanMessage(state["request"])])
            steps = _parse_json(reply.content if isinstance(reply.content, str) else str(reply.content)).get("steps") or []
            steps = [{"agent": str(s["agent"]).lower(), "task": str(s["task"])} for s in steps if isinstance(s, dict) and s.get("task")]
            steps = [s for s in steps if s["agent"] in roles][:MAX_STEPS]
        except Exception:
            steps = []
        return {"plan": steps or heuristic_plan(state["request"], roles), "step": 0, "notes": [], "trace": [], "retried": False, "skills_applied": []}

    async def work(self, state: TeamState) -> TeamState:
        step = state["plan"][state["step"]]
        tools = (await self.tools_by_role())[step["agent"]]
        notes = list(state.get("notes", []))
        task = step["task"] + (f"\n\nThe first try at this step failed: {notes[-1]}\nFix that and finish the step." if state.get("retried") else "")
        agent = self.specialist(step["agent"], tools, task, notes)
        failed, applied = "", []
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(task)]}, {"recursion_limit": STEP_RECURSION})
            messages, applied = result["messages"], result.get("skills_applied") or []
        except Exception as exc:  # the recursion limit, a model crash: the step failed
            messages, failed = [], f"{type(exc).__name__}: {str(exc)[:300]}"
        calls = trace_from_messages(messages)
        facts = Facts.from_calls(calls)
        said = messages[-1].content if messages and isinstance(messages[-1].content, str) else ""
        if not failed and calls and all(c.error for c in calls):
            failed = f"every call failed; last: {calls[-1].name}: {calls[-1].result[:300]}"
        note = f"[{step['agent']}] {step['task'][:120]} -> " + (f"FAILED: {failed}" if failed else said[:500]) + f" (facts: {facts.summary()})"
        trace = list(state.get("trace", [])) + [c.__dict__ for c in calls]
        skills = list(state.get("skills_applied", [])) + [{"step": state["step"], "agent": step["agent"], **a} for a in applied]
        if failed and not state.get("retried"):
            return {"notes": notes + [note], "trace": trace, "retried": True, "skills_applied": skills}
        return {"notes": notes + [note], "trace": trace, "retried": False, "step": state["step"] + 1, "skills_applied": skills}

    async def report(self, state: TeamState) -> TeamState:
        facts = Facts.from_calls([Call(**c) for c in state.get("trace", [])])
        notes = "\n".join(state.get("notes", [])) or "(no steps ran)"
        try:
            reply = await self.config.chat_model().ainvoke([
                SystemMessage(REPORT_PROMPT),
                HumanMessage(f"Request: {state['request']}\n\nNotes:\n{notes}\n\nFacts from the servers: {facts.summary()}"),
            ])
            answer = reply.content if isinstance(reply.content, str) else str(reply.content)
        except Exception:
            answer = ""
        return {"answer": answer.strip() or f"Steps run:\n{notes}\nFacts: {facts.summary()}"}

    # ------------------------------------------------------------------ graph
    def build(self):
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
    return state["answer"], [Call(**c) for c in state.get("trace", [])], state.get("skills_applied", [])


def make_graph(config: dict | None = None):
    """Graph factory for LangGraph Studio (input: {"request": "..."})."""
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    return TeamGraph(AgentConfig.from_env(model=configurable.get("model"), caller=configurable.get("caller") or "team")).build()
