"""Skills an agent needs, found for it before every model call: from the request, then as the task develops.

Small models rarely look skills up themselves, so the graphs do it. Before
each model call this middleware reads the conversation so far and decides,
by fixed rules (no model call, so the same run always gets the same skills):

* from the request: the best one or two skills by keywords, name and description
  (``asyncroscopy.skills.select_skills``);
* from what has happened since, once each:

  ====================== ============================================================
  skill                  added when
  ====================== ============================================================
  recover_from_errors    a tool call failed
  approval_etiquette     a call was held (APPROVAL REQUIRED), or approval_status said
                         denied or expired
  inspect_data           a data key came back, or the request asks about data
  explore_regions        a stitched map came back and the request asks for regions
  ====================== ============================================================

The chosen skills are written into the system prompt within a size budget
(``asyncroscopy.skills.skills_text``); ``load_skill`` is offered to the model
only while a chosen skill did not fit. Which skills were applied, and why, is
kept in the agent's state as ``skills_applied`` (a list of
``{"skill", "reason"}``), so traces and benchmarks can report it.

Optional ``memory``: a function from the request to Markdown notes (see
``asyncroscopy.agents.memory``) appended after the skills.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import SystemMessage

from asyncroscopy.agents.goals import DATA_KEY, MAP_KEY, NEW_MAP
from asyncroscopy.agents.guards import REPEAT_NOTE, context_chars, is_request, fit_messages, looping, stop_message, tool_rounds, tools_chars
from asyncroscopy.skills import select_skills, skills_text
from asyncroscopy.skills.registry import PROMPT_BUDGET_CHARS, list_skills, read_only_request, usable_by

ASKS_ABOUT_DATA = re.compile(r"\b(?:what is in|look at|inspect|show me|analy[sz]e|peaks?|elements?|composition|latest (?:image|spectrum|data))\b", re.I)
ASKS_FOR_REGIONS = re.compile(r"\b(?:\d+|three|two|five|the) (?:most )?(?:interesting|notable|unusual) (?:regions?|areas?|features?|spots?)\b|\bregions? of interest\b|\bwhere to look\b", re.I)
MAX_APPLIED = 4  # skills one run may carry (the text budget applies on top)
# The middleware adds a node before every model call, so one tool round (model, tools, this node) is
# three graph steps: graphs size their recursion limits with recursion_limit(tool_rounds).
STEPS_PER_TOOL_ROUND = 3


def recursion_limit(tool_rounds: int) -> int:
    """The LangGraph recursion limit that lets an agent with this middleware make ``tool_rounds`` tool calls."""
    return STEPS_PER_TOOL_ROUND * tool_rounds + 3


class SkillState(AgentState):
    skills_applied: NotRequired[list[dict]]


def _text(message) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _request(messages) -> str:
    """The latest human message: in a conversation each turn is a request of its own."""
    return next((_text(m) for m in reversed(messages) if is_request(m)), "")


def skills_for(messages, agent_tools: set[str], endpoint_tools: set[str] | None = None) -> list[dict]:
    """The skills that apply to a conversation so far, in the order they first applied, each with why.

    A request that only asks about data (``read_only_request``) never gets a skill that acts,
    whatever comes back later: a map key in a reply must not lead to a new map.
    """
    request = _request(messages)
    read_only = read_only_request(request)
    usable = {s.name for s in list_skills(endpoint_tools or agent_tools) if usable_by(s, agent_tools) and not (read_only and s.acts)}
    applied: list[dict] = []

    def add(name: str, reason: str) -> None:
        if name in usable and len(applied) < MAX_APPLIED and all(a["skill"] != name for a in applied):
            applied.append({"skill": name, "reason": reason})

    if NEW_MAP.search(request):  # a request for a new map always gets the mapping procedure first
        add("mapping", "the request asks for a new map")
    for skill in select_skills(request, agent_tools, endpoint_tools):
        add(skill.name, "matches the request")
    if ASKS_ABOUT_DATA.search(request):
        add("inspect_data", "the request asks about data")
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        text, name = _text(message), getattr(message, "name", "") or "a tool"
        if "APPROVAL REQUIRED" in text or re.search(r'"state":\s*"(?:denied|expired)"', text):
            add("approval_etiquette", f"{name} was held for an operator or not approved")
        elif getattr(message, "status", "") == "error" or text.startswith(("Error", "error:")):
            add("recover_from_errors", f"{name} failed")
        if MAP_KEY.search(text) and ASKS_FOR_REGIONS.search(request):
            add("explore_regions", f"a stitched map came back from {name} and the request asks for regions")
        if DATA_KEY.search(text):
            add("inspect_data", f"a data key came back from {name}")
    return applied


class SkillMiddleware(AgentMiddleware):
    """Puts the skills that apply (and optional memory notes) into the system prompt before every model call,
    and keeps the whole call inside the model's context (``asyncroscopy.agents.guards``): the messages are
    fitted to what the prompt and tools leave, and a call repeated with the same result is pointed out
    once, then the run is stopped with an honest answer."""

    state_schema = SkillState

    def __init__(self, base_prompt: str, agent_tools: set[str], endpoint_tools: set[str] | None = None,
                 memory: Callable[[str], str] | None = None, num_ctx: int = 8192):
        super().__init__()
        self.base_prompt = base_prompt
        self.agent_tools = set(agent_tools)
        self.endpoint_tools = set(endpoint_tools or agent_tools)
        self.memory = memory
        self.num_ctx = num_ctx
        self._notes: dict[str, str] = {}  # memory notes per request: read once per run, not per model call

    def _notes_for(self, request: str) -> str:
        if self.memory is None:
            return ""
        if request not in self._notes:
            self._notes[request] = self.memory(request) or ""
        return self._notes[request]

    def _budget(self, messages) -> int:
        """Characters of skill text this prompt may carry: skills and memory notes share at most a quarter
        of the context (and PROMPT_BUDGET_CHARS), so the request and the replies keep the rest."""
        share = min(PROMPT_BUDGET_CHARS, context_chars(self.num_ctx) // 4)
        return max(1500, share - len(self._notes_for(_request(messages))))

    def _update(self, state) -> dict[str, Any] | None:
        applied = skills_for(state["messages"], self.agent_tools, self.endpoint_tools)
        # Which skills' instructions reach the prompt in full (a cut one is listed by its description only).
        _text, cut = skills_text([a["skill"] for a in applied], self.endpoint_tools, budget=self._budget(state["messages"]))
        applied = [{**a, "in_prompt": a["skill"] not in cut} for a in applied]
        return None if applied == state.get("skills_applied") else {"skills_applied": applied}

    def before_model(self, state, runtime):
        return self._update(state)

    async def abefore_model(self, state, runtime):
        # Skill and memory files are read from disk: keep that off the event loop (LangGraph Studio checks).
        return await asyncio.to_thread(self._update, state)

    def _prompted(self, request):
        state = request.state
        applied = state.get("skills_applied") or skills_for(state["messages"], self.agent_tools, self.endpoint_tools)
        notes = self._notes_for(_request(state["messages"]))
        # One budget for skills and notes together: the prompt, tools and history share an 8k context.
        text, cut = skills_text([a["skill"] for a in applied], self.endpoint_tools, budget=self._budget(state["messages"]))
        prompt = self.base_prompt
        if text:
            prompt += "\n\nSkills for this task (follow them):\n" + text
        if notes:
            prompt += "\n\nNotes from earlier work on this instrument (keys in them are from earlier runs, not this one):\n" + notes
        # load_skill stays offered unless nothing was cut and the request only reads data: an acting
        # request may need a skill the rules did not pick.
        hide = not cut and read_only_request(_request(state["messages"]))
        tools = [t for t in request.tools if getattr(t, "name", None) != "load_skill"] if hide else request.tools
        messages = request.messages
        if looping(messages) == 1:
            name = tool_rounds(messages)[-1][0]
            prompt += REPEAT_NOTE.format(name=name)
        # The messages get what the prompt and the tool schemas leave of the context.
        room = context_chars(self.num_ctx) - len(prompt) - tools_chars(tools)
        return request.override(system_message=SystemMessage(prompt), tools=tools, messages=fit_messages(messages, max(2000, room)))

    def wrap_model_call(self, request, handler):
        if looping(request.messages) == 2:  # a third identical round in a row, or a fourth in a cycle: stop rather than loop
            return ModelResponse(result=[stop_message(request.messages)])
        return handler(self._prompted(request))

    async def awrap_model_call(self, request, handler):
        if looping(request.messages) == 2:
            return ModelResponse(result=[stop_message(request.messages)])
        return await handler(await asyncio.to_thread(self._prompted, request))
