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

At the end the run is remembered (``asyncroscopy.agents.memory``): a page in
experiments/, and either a learned skill drafted from the calls that worked
(goal met; ``asyncroscopy.agents.learning``) or a lesson in LESSONS.md (not
met). The skills each attempt was given are collected in ``skills_applied``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.goals import Call, Facts, Goal, GoalChecker, trace_from_messages
from asyncroscopy.agents.learning import distill, lesson
from asyncroscopy.agents.memory import Memory


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
    def __init__(self, goal: Goal, worker: str = "team", config: AgentConfig | None = None, checker: GoalChecker | None = None, run_worker=None,
                 memory: Memory | None = None, learn: bool = True, offered: set[str] | None = None, namer=None):
        """``run_worker(request, remaining_s)`` returns (answer, calls) or (answer, calls, skills_applied);
        ``offered``: the endpoint's tool names (fetched when None) that a learned skill may use;
        ``namer``: the chat model that names a learned skill (default: the configured model)."""
        if worker not in ("team", "general"):
            raise ValueError(f"worker must be 'team' or 'general', not {worker!r}")
        self.goal = goal
        self.worker = worker
        self.config = config or AgentConfig.from_env(caller=f"goal-{worker}")
        self.checker = checker or GoalChecker(self.config.instrument(), self.config.analysis())
        self._run_worker = run_worker or self._default_worker
        self.memory = memory or Memory.from_env()
        self.learn = learn
        self.offered = offered
        self.namer = namer

    def request(self, feedback: str) -> str:
        text = f"{self.goal.statement}\n\nThe goal is reached when: {self.goal.describe()}."
        b = self.goal.budget
        if b.max_dose_e_per_a2 is not None:
            text += f" Keep the total electron dose under {b.max_dose_e_per_a2} e/A^2 (put dose_budget_e_per_a2 in map plans)."
        if feedback:
            text += f"\n\nAn earlier attempt did not reach the goal:\n{feedback}\nUse what it already produced where you can; do not repeat work that succeeded."
        return text

    async def _default_worker(self, request: str, remaining_s: float) -> tuple[str, list[Call]]:
        if self.worker == "team":
            from asyncroscopy.agents.graphs.team import MAX_STEPS, TeamGraph

            team = TeamGraph(self.config, self.memory)
            state = await asyncio.wait_for(team.build().ainvoke({"request": request}, {"recursion_limit": 4 * MAX_STEPS + 6}), timeout=remaining_s)
            return state["answer"], [Call(**c) for c in state.get("trace", [])], state.get("skills_applied", [])
        from asyncroscopy.agents.graphs.general import make_graph

        graph = await make_graph({"configurable": {"model": self.config.model, "caller": self.config.caller}})
        limit = max(10, 2 * self.goal.budget.max_tool_calls)
        state = await asyncio.wait_for(graph.ainvoke({"messages": [HumanMessage(request)]}, {"recursion_limit": limit}), timeout=remaining_s)
        return state["messages"][-1].content, trace_from_messages(state["messages"]), state.get("skills_applied", [])

    # ------------------------------------------------------------------ nodes
    async def work(self, state: GoalState) -> GoalState:
        started = state.get("started") or time.monotonic()
        remaining = self.goal.budget.max_seconds - (time.monotonic() - started)
        trace = list(state.get("trace", []))
        attempt = state.get("attempt", 0) + 1
        skills: list = []
        try:
            answer, calls, *rest = await self._run_worker(self.request(state.get("feedback", "")), max(1.0, remaining))
            skills = rest[0] if rest else []
        except asyncio.TimeoutError:
            answer, calls = f"stopped: the time budget of {self.goal.budget.max_seconds:.0f} s ran out", []
        except Exception as exc:
            answer, calls = f"the agent failed: {type(exc).__name__}: {str(exc)[:300]}", []
        applied = list(state.get("skills_applied", [])) + [{"attempt": attempt, **s} for s in skills]
        return {"answer": answer, "trace": trace + [c.__dict__ for c in calls], "attempt": attempt, "started": started,
                "skills_applied": applied, "run_id": state.get("run_id") or uuid.uuid4().hex[:12]}

    async def check(self, state: GoalState) -> GoalState:
        facts = Facts.from_calls([Call(**c) for c in state.get("trace", [])])
        verdict = await self.checker.check(self.goal, facts, state.get("answer", ""), time.monotonic() - state["started"])
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
        facts = Facts.from_calls([Call(**c) for c in state.get("trace", [])])
        head = "Goal reached." if verdict["met"] else ("Goal stopped: " + verdict["over_budget"] if verdict["over_budget"] else "Goal not reached.")
        lines = [head, f"Goal: {self.goal.statement}", f"Attempts: {state['attempt']}; {facts.summary()}."]
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
        calls = [Call(**c) for c in state.get("trace", [])]
        run_id = state.get("run_id") or uuid.uuid4().hex[:12]
        try:
            outcome = "met" if verdict["met"] else ("stopped: " + verdict["over_budget"] if verdict["over_budget"] else "not met")
            self.memory.record_experiment(
                run_id, self.goal.statement, self.goal.describe(), [f"{c.name}{' (failed)' if c.error else ''}" for c in calls],
                sorted(facts.map_keys | facts.data_keys), outcome, [r["why"] for r in verdict["results"] if not r["met"]],
            )
            if not verdict["met"]:
                self.memory.add_lesson(lesson(self.goal, verdict["results"], calls))
                return "Lesson recorded in LESSONS.md."
            offered = self.offered
            if offered is None:
                offered = {t.name for t in await self.config.tools(allow=["*"])}
            namer = self.namer if self.namer is not None else self.config.chat_model()
            path, what = await distill(self.goal, calls, self.memory, offered, namer, self.config.model, run_id)
            note = " -- a draft until an operator promotes it." if what.startswith("created") else ""
            return f"Learned skill: {what}" + (f" ({path})" if path else "") + note
        except Exception as exc:  # memory is a help, never a reason to fail the run
            return f"Could not update the agent memory: {type(exc).__name__}: {exc}"

    def build(self):
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
