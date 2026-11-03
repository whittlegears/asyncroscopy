"""Work toward a goal: run an agent, check the goal from facts, retry with feedback, stop on budget.

    work ──► check ──► done (every criterion met)
      ▲        │
      └────────┘ retry with the unmet criteria as feedback, while attempts and budget remain;
                 otherwise report which criteria were not met and why

The agent doing the work is the ``team`` graph (default) or the ``general``
agent. The checker (``asyncroscopy.agents.goals.GoalChecker``) reads only the
tool calls and the servers' replies and asks the live servers whether the
keys exist; the model's own claims never count. The time budget bounds each
attempt; the tool-call and dose budgets are checked after each attempt (the
Mapper's own ``dose_budget_e_per_a2`` is what stops a single map mid-run).

An attempt that hits its time budget or the recursion limit keeps the calls it
made, so a job it started is known: the retry is told to wait for it rather
than start another, and when the goal ends any job it started that is still
running is aborted (the report says so). The retry is told only the keys this
goal's own attempts made, never data found by listing.

At the end the run is remembered (``asyncroscopy.agents.memory``): a page in
experiments/, and either a learned skill drafted from the calls that worked
(goal met; ``asyncroscopy.agents.learning``) or a lesson in LESSONS.md (not
met). The skills each attempt was given are collected in ``skills_applied``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.goals import Call, Facts, Goal, GoalChecker, trace_from_messages
from asyncroscopy.agents.graphs import run_keeping_partial
from asyncroscopy.agents.learning import distill, lesson
from asyncroscopy.agents.mcp_tools import McpServer
from asyncroscopy.agents.memory import Memory
from asyncroscopy.agents.skill_middleware import recursion_limit


class GoalState(TypedDict, total=False):
    goal: dict
    attempt: int
    trace: list
    answer: str
    feedback: str
    verdict: dict
    started: float
    report: str
    met: bool
    skills_applied: list
    learned: str
    run_id: str


class GoalGraph:
    def __init__(self, goal: Goal, worker: str = "team", config: AgentConfig | None = None, checker: GoalChecker | None = None,
                 run_worker: Callable[[str, float], Awaitable[tuple]] | None = None, memory: Memory | None = None,
                 learn: bool = True, offered: set[str] | None = None, namer: BaseChatModel | None = None) -> None:
        """``run_worker(request, remaining_s)`` returns (answer, calls) or (answer, calls, skills_applied);
        ``offered``: the endpoint's tool names (fetched when None) that a learned skill may use;
        ``namer``: the chat model that names a learned skill (default: the configured model)."""
        if worker not in ("team", "general"):
            raise ValueError(f"worker must be 'team' or 'general', not {worker!r}")
        self.goal = goal
        self.worker = worker
        self.config = config or AgentConfig.from_env(caller=f"goal-{worker}")
        self._checker = checker  # built on first use, after the endpoint said which analysis server it has
        self._run_worker = run_worker or self._default_worker
        self.memory = memory or Memory.from_env()
        self.learn = learn
        self.offered = offered
        self.namer = namer

    def request(self, feedback: str, facts: Facts | None = None) -> str:
        text = f"{self.goal.statement}\n\nThe goal is reached when: {self.goal.describe()}."
        b = self.goal.budget
        if b.max_dose_e_per_a2 is not None:
            text += f" Keep the total electron dose under {b.max_dose_e_per_a2} e/A^2 (put dose_budget_e_per_a2 in map plans)."
        if feedback:
            text += f"\n\nAn earlier attempt did not reach the goal:\n{feedback}"
            produced = facts.produced() if facts else []
            if produced:  # only what this goal's own attempts made, never older data
                text += f"\nAlready made by this goal: {', '.join(produced[:10])}. Use these; do not repeat work that succeeded."
            running = facts.active_jobs() if facts else {}
            if running:
                text += "\nStill running from the earlier attempt: " + ", ".join(f"job {j} on {v['device']}" for j, v in running.items()) + \
                        ". Wait for it with wait_for_job; do not start another job."
        return text

    async def checker(self) -> GoalChecker:
        """The checker: its own calls (view_data, map_info, aborts) go to the servers without entering the run's ledger."""
        if self._checker is None:
            await self.config.info()
            analysis = self.config.analysis()
            self._checker = GoalChecker(McpServer(self.config.mcp_url, self.config._headers()),
                                        McpServer(analysis.url, analysis.headers) if analysis is not None else None)
        return self._checker

    async def _default_worker(self, request: str, remaining_s: float) -> tuple[str, list[Call], list, str]:
        """(answer, calls, skills applied, what went wrong or ""): a failed or timed-out run keeps the calls it made."""
        if self.worker == "team":
            from asyncroscopy.agents.graphs.team import MAX_STEPS, TeamGraph

            team = TeamGraph(self.config, self.memory)
            state, failed = await run_keeping_partial(team.build(), {"request": request}, {"recursion_limit": 4 * MAX_STEPS + 6}, timeout=remaining_s)
            return state.get("answer", ""), [Call.from_dict(c) for c in state.get("trace", [])], state.get("skills_applied", []), failed
        from asyncroscopy.agents.graphs.general import make_graph

        graph = await make_graph({"configurable": {"agent_config": self.config}})  # its calls go into this goal's ledger
        state, failed = await run_keeping_partial(graph, {"messages": [HumanMessage(request)]},
                                                  {"recursion_limit": recursion_limit(self.goal.budget.max_tool_calls)}, timeout=remaining_s)
        messages = state.get("messages", [])
        answer = messages[-1].content if messages and isinstance(messages[-1].content, str) else ""
        return answer, trace_from_messages(messages), state.get("skills_applied", []), failed

    async def _facts(self, trace: list) -> Facts:
        return await (await self.checker()).facts([Call.from_dict(c) for c in trace])

    # ------------------------------------------------------------------ nodes
    async def work(self, state: GoalState) -> GoalState:
        started = state.get("started") or time.monotonic()
        remaining = self.goal.budget.max_seconds - (time.monotonic() - started)
        trace = list(state.get("trace", []))
        attempt = state.get("attempt", 0) + 1
        skills: list = []
        facts = await self._facts(trace) if trace else None
        ledger_start = len(self.config.ledger)
        self.config.deadline = started + self.goal.budget.max_seconds  # waits inside a tool call stop there too
        try:
            answer, calls, *rest = await self._run_worker(self.request(state.get("feedback", ""), facts), max(1.0, remaining))
            skills = rest[0] if rest else []
            if len(rest) > 1 and rest[1]:
                answer = (answer + "\n" if answer else "") + f"(the agent stopped: {rest[1]})"
        except asyncio.TimeoutError:
            answer, calls = f"stopped: the time budget of {self.goal.budget.max_seconds:.0f} s ran out", []
        except Exception as exc:
            answer, calls = f"the agent failed: {type(exc).__name__}: {str(exc)[:300]}", []
        # The ledger has every call whose reply arrived, in order, even from a step or attempt that was
        # stopped part way (a team's state only grows when a step ends): it knows every job started.
        recorded = [Call.from_dict(c) for c in self.config.ledger[ledger_start:]]
        if len(recorded) > len(calls):
            calls = recorded
        applied = list(state.get("skills_applied", [])) + [{"attempt": attempt, **s} for s in skills]
        return {"answer": answer, "trace": trace + [c.to_dict() for c in calls], "attempt": attempt, "started": started,
                "skills_applied": applied, "run_id": state.get("run_id") or uuid.uuid4().hex[:12]}

    async def check(self, state: GoalState) -> GoalState:
        facts = await self._facts(state.get("trace", []))
        verdict = await (await self.checker()).check(self.goal, facts, state.get("answer", ""), time.monotonic() - state["started"])
        return {"verdict": {"met": verdict.met, "results": verdict.results, "over_budget": verdict.over_budget}, "feedback": verdict.feedback(), "met": verdict.met}

    def route(self, state: GoalState) -> str:
        verdict = state["verdict"]
        if verdict["met"] or verdict["over_budget"]:
            return "report"
        if state["attempt"] >= self.goal.budget.max_attempts:
            return "report"
        if time.monotonic() - state["started"] >= self.goal.budget.max_seconds:
            return "report"
        return "work"

    async def report(self, state: GoalState) -> GoalState:
        verdict = state["verdict"]
        facts = await self._facts(state.get("trace", []))
        head = "Goal reached." if verdict["met"] else ("Goal stopped: " + verdict["over_budget"] if verdict["over_budget"] else "Goal not reached.")
        lines = [head, f"Goal: {self.goal.statement}", f"Attempts: {state['attempt']}; {facts.summary()}."]
        running = facts.active_jobs()
        if running and self.goal.leave_running:  # the request said to start it and not wait
            lines.append("Left running, as asked: " + ", ".join(f"job {j} on {v['device']}" for j, v in running.items()))
        elif running:  # the goal is over: nothing it started keeps running
            stopped = await (await self.checker()).stop_jobs(facts)
            lines.append("Stopped jobs this goal had left running: " + "; ".join(f"{c.name} {c.args.get('request', '')}{' (FAILED: ' + c.result[:120] + ')' if c.error else ''}" for c in stopped))
        lines += [f"{'met' if r['met'] else 'NOT met'}: {r['why']}" for r in verdict["results"]]
        skills = state.get("skills_applied", [])
        if skills:
            lines.append("Skills used: " + "; ".join(f"{s['skill']} ({s['reason']})" for s in skills[:8]))
        learned = await self.remember(state, facts) if self.learn else ""
        if learned:
            lines.append(learned)
        lines += ["", state.get("answer", "")]
        return {"report": "\n".join(lines).strip(), "learned": learned}

    async def remember(self, state: GoalState, facts: Facts) -> str:
        """Write the experiment page, then a learned-skill draft (met) or a lesson (not met). Never fails the run."""
        verdict = state["verdict"]
        calls = [Call.from_dict(c) for c in state.get("trace", [])]
        run_id = state.get("run_id") or uuid.uuid4().hex[:12]
        try:
            outcome = "met" if verdict["met"] else ("stopped: " + verdict["over_budget"] if verdict["over_budget"] else "not met")
            self.memory.record_experiment(
                run_id, self.goal.statement, self.goal.describe(), [f"{c.name}{' (failed)' if c.error else ''}" for c in calls],
                sorted(facts.map_keys | facts.data_keys), outcome, [r["why"] for r in verdict["results"] if not r["met"]],
            )
            if not verdict["met"]:
                outcome = self.memory.add_lesson(lesson(self.goal, verdict["results"], calls))
                return {"added": "Lesson recorded in LESSONS.md.", "duplicate": "This lesson is already in LESSONS.md."}.get(outcome, f"Lesson not recorded ({outcome}).")
            offered = self.offered
            if offered is None:
                offered = {t.name for t in await self.config.tools(allow=["*"])}
            namer = self.namer if self.namer is not None else self.config.chat_model()
            path, what = await distill(self.goal, calls, self.memory, offered, namer, self.config.model, run_id)
            note = " -- a draft until an operator promotes it." if what.startswith("created") else ""
            return f"Learned skill: {what}" + (f" ({path})" if path else "") + note
        except Exception as exc:  # memory is a help, never a reason to fail the run
            return f"Could not update the agent memory: {type(exc).__name__}: {exc}"

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(GoalState)
        graph.add_node("work", self.work)
        graph.add_node("check", self.check)
        graph.add_node("report", self.report)
        graph.add_edge(START, "work")
        graph.add_edge("work", "check")
        graph.add_conditional_edges("check", self.route, ["work", "report"])
        graph.add_edge("report", END)
        return graph.compile()


async def run_goal(goal: Goal, worker: str = "team", config: AgentConfig | None = None) -> tuple[str, bool]:
    """(report, whether the goal was reached)."""
    state = await GoalGraph(goal, worker, config).build().ainvoke({"goal": goal.to_dict()}, {"recursion_limit": 4 * goal.budget.max_attempts + 4})
    return state["report"], bool(state.get("met"))
