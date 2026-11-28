"""Skills an agent needs, found for it before every model call: from the request, then as the task develops.

Small models rarely look skills up themselves, so the graphs do it. Before
each model call this middleware reads the conversation so far and decides,
by fixed rules (no model call, so the same run always gets the same skills):

* from the request: the best few skills by keywords, name and description
  (``asyncroscopy.skills.select_skills``), after this endpoint's skill marked ``new_map`` when the request
  asks for a new map;
* from what has happened since, once each:

  ====================== ============================================================
  skill                  added when
  ====================== ============================================================
  recover_from_errors    a tool call failed
  inspect_data           the request asks about data
  explore_regions        a stitched map came back and the request asks for regions
  ====================== ============================================================

A procedure chosen this way is written into the system prompt whole, within a size budget
(``asyncroscopy.skills.skills_text``), as background on how to do the task rather
than as "skills" to name; ``load_skill`` is offered to the model only while a
chosen skill did not fit. The skills an event brings for advice rather than a procedure
(``PASSAGE_SKILLS``: what to do about a failed call, how to read data) are not written whole.

Then, before every model call, the skills are searched as passages (``asyncroscopy.skills.index``):
the request and the latest tool replies are the query, so the advice for this step comes with it (the
paragraph about the error that just came back, the section on reading a map once a map exists). Those
passages fill what the budget leaves, up to the profile's ``passages`` and ``passage_chars``, and are added
to the latest message (the tool reply or the request) rather than to the system prompt, so the prompt's start stays the same
and a server that caches it (Ollama) does not read the whole conversation again each step. The search
runs in process (BM25 over a few dozen passages, no model call), so it adds no noticeable time.

Which skills were applied, and why, is kept in the agent's state as ``skills_applied`` (a list of
``{"skill", "reason", "in_prompt"}``), and the passages of the latest step as ``guidance`` (their
labels), so traces and benchmarks can report them.

Optional ``memory``: a function from the request to Markdown notes (see
``asyncroscopy.agents.memory``) appended after the skills.

What depends on the model comes from the agent profile (``asyncroscopy.agents.profile``): how many skills and
how much text a prompt carries (``skills``) and which tools are held back unless the request asks for them
(``tool_gates``). A small model is helped by a short prompt and few tools; a capable one is not. These hold for
every model: a request that only asks about data
gets no skill that acts, and the whole call is kept inside the context.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from collections.abc import Awaitable, Sequence
from typing import Any, Callable, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from asyncroscopy.agents.goals import MAP_KEY, NEW_MAP
from asyncroscopy.agents.guards import REPEAT_NOTE, context_chars, is_request, fit_messages, looping, stop_message, tool_rounds, tools_chars
from asyncroscopy.agents.profile import SkillLimits, ToolGate, held_back
from asyncroscopy.skills import select_skills, skills_text
from asyncroscopy.skills.index import Passage, passages_text, search
from asyncroscopy.skills.registry import Skill, list_skills, read_only_request, usable_by

ASKS_ABOUT_DATA = re.compile(r"\b(?:what is in|look at|inspect|show me|analy[sz]e|peaks?|elements?|composition|latest (?:image|spectrum|data))\b", re.I)
ASKS_FOR_REGIONS = re.compile(r"\b(?:\d+|three|two|five|the) (?:most )?(?:interesting|notable|unusual) (?:regions?|areas?|features?|spots?)\b|\bregions? of interest\b|\bwhere to look\b", re.I)
# Below this, unused tools are dropped, largest first, before Ollama cuts the start of the prompt.
MIN_MESSAGES_CHARS = 6000
ALWAYS_OFFERED = ("view_data", "wait_for_job", "load_skill")
PASSAGE_SKILLS = frozenset({"recover_from_errors", "inspect_data"})
ADVICE_HEADING = ("Advice that may help with this step, from this instrument's skills (use what applies; "
                  "the request and the tool replies come first):\n")
EVENT_BOOST = 3.0  # what an event adds to its skill's passages in the search
REPLY_CHARS = 800
REPLY_WEIGHT = 0.6
MAX_APPLIED = SkillLimits().at_most  # the default profile's limit
# The middleware adds a node before every model call, so a tool round is three graph steps.
STEPS_PER_TOOL_ROUND = 3


def recursion_limit(tool_rounds: int) -> int:
    """The LangGraph recursion limit that lets an agent with this middleware make ``tool_rounds`` tool calls."""
    return STEPS_PER_TOOL_ROUND * tool_rounds + 3


class SkillState(AgentState):
    skills_applied: NotRequired[list[dict]]
    guidance: NotRequired[list[str]]


def _text(message) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _request(messages) -> str:
    """The latest human message: in a conversation each turn is a request of its own."""
    return next((_text(m) for m in reversed(messages) if is_request(m)), "")


def step_query(messages: Sequence[BaseMessage]) -> list[tuple[str, float]]:
    """What to search the skills with for the next model call: the request, and the tool replies since it
    (the latest two, a failed one counting as much as the request)."""
    request = _request(messages)
    replies = []
    for message in reversed(messages):
        if is_request(message):
            break
        if getattr(message, "type", "") == "tool":
            failed = getattr(message, "status", "") == "error" or _text(message).startswith(("Error", "error:"))
            replies.append((_text(message)[:REPLY_CHARS], 1.0 if failed else REPLY_WEIGHT))
            if len(replies) == 2:
                break
    return [(request, 1.0), *replies]


def with_advice(messages: list, advice: str) -> list:
    """``messages`` with ``advice`` added to a copy of the last one (the latest tool reply, or the request), so
    every provider takes it (some refuse a system message after the first) and the state keeps the original."""
    if messages and isinstance(getattr(messages[-1], "content", None), str):
        last = messages[-1]
        return [*messages[:-1], last.model_copy(update={"content": f"{last.content}\n\n{advice}"})]
    return [*messages, HumanMessage(advice)]


def usable_skills(messages: Sequence[BaseMessage], agent_tools: set[str], endpoint_tools: set[str] | None = None) -> list[Skill]:
    """The skills this agent may be given for this conversation: usable with its tools, and none that acts when
    the request only reads."""
    read_only = read_only_request(_request(messages))
    return [s for s in list_skills(endpoint_tools or agent_tools) if usable_by(s, agent_tools) and not (read_only and s.acts)]


def skills_for(messages: Sequence[BaseMessage], agent_tools: set[str], endpoint_tools: set[str] | None = None,
               limits: SkillLimits | None = None) -> list[dict]:
    """The skills that apply to a conversation so far, in the order they first applied, each with why.

    A request that only asks about data (``read_only_request``) never gets a skill that acts,
    whatever comes back later: a map key in a reply must not lead to a new map. ``limits`` are the
    profile's (the default profile's when None).
    """
    limits = limits or SkillLimits()
    request = _request(messages)
    known = usable_skills(messages, agent_tools, endpoint_tools)
    usable = {s.name for s in known}
    applied: list[dict] = []

    def add(name: str, reason: str) -> None:
        if name in usable and len(applied) < limits.at_most and all(a["skill"] != name for a in applied):
            applied.append({"skill": name, "reason": reason})

    if NEW_MAP.search(request):  # a new map gets this endpoint's mapping procedure first
        for skill in known:
            if skill.new_map:
                add(skill.name, "the request asks for a new map")
    for skill in select_skills(request, agent_tools, endpoint_tools, limit=limits.from_request):
        add(skill.name, "matches the request")
    if ASKS_ABOUT_DATA.search(request):
        add("inspect_data", "the request asks about data")
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        text, name = _text(message), getattr(message, "name", "") or "a tool"
        if getattr(message, "status", "") == "error" or text.startswith(("Error", "error:")):
            add("recover_from_errors", f"{name} failed")
        if MAP_KEY.search(text) and ASKS_FOR_REGIONS.search(request):
            add("explore_regions", f"a stitched map came back from {name} and the request asks for regions")
    return applied


class SkillMiddleware(AgentMiddleware):
    """Puts the skills that apply (and optional memory notes) into the system prompt before every model call,
    and keeps the whole call inside the model's context (``asyncroscopy.agents.guards``): the messages are
    fitted to what the prompt and tools leave, and a call repeated with the same result is pointed out
    once, then the run is stopped with an honest answer."""

    state_schema = SkillState

    def __init__(self, base_prompt: str, agent_tools: set[str], endpoint_tools: set[str] | None = None,
                 memory: Callable[[str], str] | None = None, num_ctx: int = 8192, limits: SkillLimits | None = None,
                 gates: tuple[ToolGate, ...] = ()) -> None:
        super().__init__()
        self.limits = limits or SkillLimits()
        self.gates = tuple(gates)  # the profile's tool_gates
        self.base_prompt = base_prompt
        self.agent_tools = set(agent_tools)
        self.endpoint_tools = set(endpoint_tools or agent_tools)
        known = list_skills(self.endpoint_tools)
        self.skill_tools = {s.name: list(s.tools) for s in known}
        self.memory = memory
        self.num_ctx = num_ctx
        self._notes: dict[str, str] = {}  # read once per run, not per model call

    def _notes_for(self, request: str) -> str:
        if self.memory is None:
            return ""
        if request not in self._notes:
            self._notes[request] = self.memory(request) or ""
        return self._notes[request]

    def _budget(self, messages) -> int:
        """Characters of skill text this prompt may carry: skills and memory share at most a quarter of the context."""
        share = min(self.limits.prompt_chars, context_chars(self.num_ctx) // 4)
        return max(1500, share - len(self._notes_for(_request(messages))))

    def skills_for(self, messages) -> list[dict]:
        """The skills that apply to ``messages`` under this middleware's tools and limits."""
        return skills_for(messages, self.agent_tools, self.endpoint_tools, self.limits)

    def guidance(self, messages, applied: list[dict]) -> tuple[str, list[str], str, list[Passage]]:
        """What the skills put in the prompt for this step: (whole procedures' text, the ones cut short,
        passages' text, the passages). Procedures come first; passages fill what the budget leaves."""
        budget = self._budget(messages)
        whole = [a["skill"] for a in applied if a["skill"] not in PASSAGE_SKILLS]
        text, cut = skills_text(whole, self.endpoint_tools, budget=budget)
        room = min(self.limits.passage_chars, budget - len(text))
        if self.limits.passages <= 0 or room <= 200:
            return text, cut, "", []
        boost = {a["skill"]: EVENT_BOOST for a in applied if a["skill"] in PASSAGE_SKILLS}
        found = search(step_query(messages), usable_skills(messages, self.agent_tools, self.endpoint_tools),
                       k=self.limits.passages, boost=boost, skip=set(whole))
        passages, fitted = passages_text([p for p, _score in found], room)
        return text, cut, passages, fitted

    def _update(self, state) -> dict[str, Any] | None:
        applied = self.skills_for(state["messages"])
        _text, cut, _passages, fitted = self.guidance(state["messages"], applied)
        shown = {p.skill for p in fitted}
        applied = [{**a, "in_prompt": a["skill"] in shown if a["skill"] in PASSAGE_SKILLS else a["skill"] not in cut} for a in applied]
        guidance = [p.label for p in fitted]
        if applied == state.get("skills_applied") and guidance == state.get("guidance", []):
            return None
        return {"skills_applied": applied, "guidance": guidance}

    def before_model(self, state: SkillState, runtime: Runtime) -> dict[str, Any] | None:
        return self._update(state)

    async def abefore_model(self, state: SkillState, runtime: Runtime) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._update, state)

    def _prompted(self, request):
        state = request.state
        applied = state.get("skills_applied") or self.skills_for(state["messages"])
        notes = self._notes_for(_request(state["messages"]))
        text, cut, passages, _fitted = self.guidance(state["messages"], applied)
        prompt = self.base_prompt
        if text:
            prompt += "\n\nHow to do this on this instrument:\n" + text
        if notes:
            prompt += "\n\nNotes from earlier work on this instrument (keys in them are from earlier runs, not this one):\n" + notes
        asked = _request(state["messages"])
        hidden = set() if cut else {"load_skill"}
        hidden |= held_back(self.gates, asked, [getattr(t, "name", "") for t in request.tools])
        tools = [t for t in request.tools if getattr(t, "name", None) not in hidden]
        messages = request.messages
        if looping(messages) == 1:
            name = tool_rounds(messages)[-1][0]
            prompt += REPEAT_NOTE.format(name=name)
        advice = ADVICE_HEADING + passages if passages else ""
        tools = self._fit_tools(tools, prompt + advice, applied, messages)
        room = context_chars(self.num_ctx) - len(prompt) - len(advice) - tools_chars(tools)
        fitted = fit_messages(messages, max(2000, room))
        if advice:
            # After the conversation, so the prompt's start stays the same and Ollama's cache reads only what is new.
            fitted = with_advice(fitted, advice)
        return request.override(system_message=SystemMessage(prompt), tools=tools, messages=fitted)

    def _fit_tools(self, tools: list, prompt: str, applied: list[dict], messages) -> list:
        """The tools, with those the task does not need dropped (largest first) until MIN_MESSAGES_CHARS fit."""
        spare = context_chars(self.num_ctx) - len(prompt) - MIN_MESSAGES_CHARS
        if tools_chars(tools) <= spare:
            return tools
        patterns = [p for a in applied for p in self.skill_tools.get(a["skill"], [])] + list(ALWAYS_OFFERED)
        called = {c.get("name") for m in messages for c in getattr(m, "tool_calls", None) or []}

        def needed(tool) -> bool:
            return tool.name in called or any(fnmatch.fnmatchcase(tool.name, p) for p in patterns)

        kept = list(tools)
        for tool in sorted((t for t in tools if not needed(t)), key=lambda t: tools_chars([t]), reverse=True):
            if tools_chars(kept) <= spare:
                break
            kept.remove(tool)
        return kept

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:
        if looping(request.messages) == 2:  # stop before a third identical round or a fourth in a cycle
            return ModelResponse(result=[stop_message(request.messages)])
        return handler(self._prompted(request))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelCallResult:
        if looping(request.messages) == 2:
            return ModelResponse(result=[stop_message(request.messages)])
        return await handler(await asyncio.to_thread(self._prompted, request))
