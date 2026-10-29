"""Goals: criteria from text and files, facts from tool calls, the checker, and the retry loop (no model, no stack)."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.goals import Call, Facts, Goal, GoalChecker, trace_from_messages
from asyncroscopy.agents.graphs.goal import GoalGraph
from asyncroscopy.agents.memory import Memory


@pytest.fixture(autouse=True)
def memory_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_AGENT_MEMORY", str(tmp_path / "memory"))


def reply(payload, error=False):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(is_error=error, content=[SimpleNamespace(text=text)], structured_content=payload if isinstance(payload, dict) else None)


class FakeServer:
    """map_info / list_pois / view_data answers for the keys it knows."""

    def __init__(self, maps=(), pois=0, data=None):
        self.maps, self.pois, self.data = set(maps), pois, dict(data or {})

    async def call(self, name, args):
        if name == "map_info":
            return reply({"map": args["map"]}) if args["map"] in self.maps else reply("no such map", error=True)
        if name == "list_pois":
            return reply({"result": [{"kind": "candidate"}] * self.pois})
        if name == "view_data":
            kind = self.data.get(args["key"])
            return reply({"kind": kind, "text": "..."}) if kind else reply(f"Tiled has no key {args['key']!r}", error=True)
        raise AssertionError(name)


MAP_CALLS = [
    Call("Mapper_reserve", {"request": "{}"}, json.dumps({"reservation_id": "r1", "estimate": {"max_dose_e_per_a2": 20.0}})),
    Call("Mapper_start", {"request": "{}"}, json.dumps({"job_id": "j1", "state": "running"})),
    Call("wait_for_job", {"device": "Mapper", "job_id": "j1"}, json.dumps({"job_id": "j1", "state": "done"})),
    Call("Mapper_results", {"request": "{}"}, json.dumps({"result_keys": {"run": "map_r1.h5"}, "summary": {"spectra": [{"key": "spectrum_eds_1.h5"}]}})),
    Call("analysis_job_results", {"job_id": "a1"}, json.dumps({"result_keys": {"map": "map_r1_map"}})),
]


def test_criteria_are_inferred_from_the_goal_text():
    goal = Goal.from_text("Map a 600 nm square here with EDS and find the 3 most interesting regions")
    kinds = {c["kind"]: c for c in goal.criteria}
    assert kinds["map"]["min_pois"] == 3 and kinds["map"]["poi_kind"] == "candidate"
    assert kinds["data"]["data_kind"] == "spectrum"
    assert "no_unrequested" in kinds and "grounded_answer" in kinds
    assert "stitched map" in goal.describe()
    plain = {c["kind"] for c in Goal.from_text("Take an image of the current area").criteria}
    assert "data" in plain and "map" not in plain


def test_goal_files_are_validated(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("statement: map here\ncriteria:\n  - kind: map\nbudget:\n  max_attempts: 3\n  max_dose_e_per_a2: 50\n", encoding="utf-8")
    goal = Goal.load(path)
    assert goal.criteria == [{"kind": "map"}] and goal.budget.max_attempts == 3 and goal.budget.max_dose_e_per_a2 == 50
    with pytest.raises(ValueError, match="kind"):
        Goal.from_dict({"statement": "x", "criteria": [{"kind": "vibes"}]})
    with pytest.raises(ValueError, match="unknown goal fields"):
        Goal.from_dict({"statement": "x", "deadline": 3})
    assert any(c["kind"] == "map" for c in Goal.from_dict({"statement": "map the area"}).criteria)  # no criteria: inferred


def test_facts_come_from_server_replies_not_the_model():
    messages = [
        HumanMessage("map it"),
        AIMessage("", tool_calls=[{"name": "Mapper_start", "args": {"request": "{}"}, "id": "c1"}]),
        ToolMessage(json.dumps({"job_id": "j1", "state": "running"}), tool_call_id="c1"),
        AIMessage("", tool_calls=[{"name": "DigitalTwin_move_stage", "args": {"position": [0, 0, 0, 0, 0]}, "id": "c2"}]),
        ToolMessage("Error: refused", tool_call_id="c2", status="error"),
        AIMessage("Done! The map is map_fake_map."),
    ]
    calls = trace_from_messages(messages)
    assert [c.name for c in calls] == ["Mapper_start", "DigitalTwin_move_stage"] and calls[1].error
    facts = Facts.from_calls(calls + MAP_CALLS)
    assert "map_fake_map" not in facts.map_keys  # the model's own text never counts
    assert facts.map_keys == {"map_r1_map"} and {"map_r1.h5", "spectrum_eds_1.h5"} <= facts.data_keys
    assert facts.jobs["j1"] == {"device": "Mapper", "state": "done"}
    assert facts.dose_e_per_a2 == 20.0
    assert "DigitalTwin_move_stage" in facts.high_risk and "Mapper_start" in facts.high_risk


def check(goal, calls, answer, server):
    return asyncio.run(GoalChecker(server, server).check(goal, Facts.from_calls(calls), answer))


def test_checker_verifies_keys_on_the_live_servers():
    goal = Goal("map", [{"kind": "map", "min_pois": 2, "poi_kind": "candidate"}, {"kind": "data", "data_kind": "spectrum"}, {"kind": "job_done"}, {"kind": "grounded_answer"}])
    server = FakeServer(maps={"map_r1_map"}, pois=3, data={"spectrum_eds_1.h5": "spectrum", "map_r1.h5": "image"})
    verdict = check(goal, MAP_CALLS, "Map map_r1_map from run map_r1.h5.", server)
    assert verdict.met, verdict.results
    missing_map = check(goal, MAP_CALLS, "ok", FakeServer(pois=3, data={"spectrum_eds_1.h5": "spectrum"}))
    assert not missing_map.met and "do not open" in missing_map.feedback()
    few = check(goal, MAP_CALLS, "ok", FakeServer(maps={"map_r1_map"}, pois=1, data={"spectrum_eds_1.h5": "spectrum"}))
    assert "1 candidate points of interest, 2 wanted" in few.feedback()


def test_an_answer_naming_an_unseen_key_is_not_grounded():
    goal = Goal("x", [{"kind": "grounded_answer"}])
    verdict = check(goal, MAP_CALLS, "Your map is map_9invented_map and spectrum_99.h5.", FakeServer())
    assert not verdict.met and "map_9invented_map" in verdict.feedback()


def test_unrequested_actions_and_budgets_are_caught():
    calls = MAP_CALLS + [Call("DigitalTwin_move_stage", {"position": [1e-6, 0, 0, 0, 0]}, "ok")]
    goal = Goal("x", [{"kind": "no_unrequested"}])
    assert "DigitalTwin_move_stage" in check(goal, calls, "", FakeServer()).feedback()
    assert check(Goal("x", [{"kind": "no_unrequested", "allow": ["*_move_stage", "*_start"]}]), calls, "", FakeServer()).met
    over = Goal("x", [])
    over.budget.max_dose_e_per_a2 = 10
    verdict = check(over, MAP_CALLS, "", FakeServer())
    assert not verdict.met and "dose" in verdict.over_budget


def run_goal_graph(goal, attempts_results, server, **options):
    """GoalGraph with a scripted worker: each attempt returns the next (answer, calls[, skills])."""
    requests = []

    async def worker(request, remaining_s):
        requests.append(request)
        return attempts_results[len(requests) - 1]

    options.setdefault("learn", False)
    graph = GoalGraph(goal, "team", AgentConfig(analysis_url=""), GoalChecker(server, server), run_worker=worker, **options)
    state = asyncio.run(graph.build().ainvoke({"goal": goal.to_dict()}))
    return state, requests


def test_the_loop_retries_with_feedback_then_succeeds():
    goal = Goal("map here", [{"kind": "map"}])
    server = FakeServer(maps={"map_r1_map"})
    state, requests = run_goal_graph(goal, [("I ran the map.", MAP_CALLS[:4]), ("Stitched: map_r1_map", MAP_CALLS[4:])], server)
    assert state["met"] and state["attempt"] == 2
    assert "no stitched map" in requests[1] and "did not reach the goal" in requests[1]
    assert state["report"].startswith("Goal reached.")


def test_the_loop_stops_after_its_attempts_and_says_why():
    goal = Goal("map here", [{"kind": "map"}])
    goal.budget.max_attempts = 2
    state, requests = run_goal_graph(goal, [("nothing", []), ("still nothing", [])], FakeServer())
    assert not state["met"] and len(requests) == 2
    assert state["report"].startswith("Goal not reached.") and "NOT met: no stitched map" in state["report"]


def test_a_worker_crash_or_timeout_is_an_attempt_not_a_crash():
    goal = Goal("map here", [{"kind": "map"}])
    goal.budget.max_attempts = 1

    async def worker(request, remaining_s):
        raise asyncio.TimeoutError()

    graph = GoalGraph(goal, "team", AgentConfig(analysis_url=""), GoalChecker(FakeServer(), FakeServer()), run_worker=worker, learn=False)
    state = asyncio.run(graph.build().ainvoke({"goal": goal.to_dict()}))
    assert not state["met"] and "time budget" in state["report"]


class Namer:
    async def ainvoke(self, messages):
        return SimpleNamespace(content=json.dumps({"name": "map and stitch", "description": "Map an area and stitch it.", "keywords": ["map"]}))


def test_a_met_goal_is_remembered_as_a_draft_skill_and_skills_used_are_reported(tmp_path):
    goal = Goal("map here", [{"kind": "map"}])
    memory = Memory(tmp_path / "mem")
    skills = [{"skill": "mapping", "reason": "matches the request"}]
    state, _ = run_goal_graph(goal, [("Stitched map_r1_map", MAP_CALLS, skills)], FakeServer(maps={"map_r1_map"}),
                              learn=True, memory=memory, offered={c.name for c in MAP_CALLS}, namer=Namer())
    assert state["met"] and state["skills_applied"] == [{"attempt": 1, "skill": "mapping", "reason": "matches the request"}]
    assert "Skills used: mapping (matches the request)" in state["report"]
    assert "created draft map_and_stitch" in state["learned"] and "until an operator promotes it" in state["learned"]
    assert (memory.skills_dir / "map_and_stitch" / "SKILL.md").is_file()
    pages = list(memory.experiments_dir.glob("*.md"))
    assert len(pages) == 1 and "Outcome: met" in pages[0].read_text(encoding="utf-8")


def test_an_unmet_goal_is_remembered_as_a_lesson(tmp_path):
    goal = Goal("map here", [{"kind": "map"}])
    goal.budget.max_attempts = 1
    memory = Memory(tmp_path / "mem")
    state, _ = run_goal_graph(goal, [("nothing", MAP_CALLS[:2])], FakeServer(), learn=True, memory=memory, offered=set(), namer=Namer())
    assert not state["met"] and state["learned"] == "Lesson recorded in LESSONS.md."
    assert "no stitched map" in memory.lessons_file.read_text(encoding="utf-8")
    assert not memory.skills_dir.exists()


def test_run_job_and_the_unified_wait_are_read_as_facts():
    calls = [
        Call("run_job", {"device": "Mapper", "plan": {}}, json.dumps({"reservation_id": "r9", "estimate": {"max_dose_e_per_a2": 7.5}, "job_id": "j9", "state": "running"})),
        Call("wait_for_job", {"job_id": "j9"}, json.dumps({"job_id": "j9", "state": "done", "result_keys": {"run": "map_r9.h5"}, "summary": {"spectra": []}})),
        Call("assemble_map", {"run": "map_r9.h5"}, json.dumps({"job_id": "a9"})),
        Call("wait_for_job", {"job_id": "a9"}, json.dumps({"job_id": "a9", "state": "done", "result_keys": {"map": "map_r9_map"}})),
    ]
    facts = Facts.from_calls(calls)
    assert facts.jobs == {"j9": {"device": "Mapper", "state": "done"}, "a9": {"device": "analysis", "state": "done"}}
    assert facts.dose_e_per_a2 == 7.5 and facts.map_keys == {"map_r9_map"} and "map_r9.h5" in facts.data_keys
    assert facts.high_risk == []  # run_job is the requested action, through the approval gate


def test_cli_goal_sets_the_budget_and_exits_by_the_verdict(monkeypatch, capsys):
    from asyncroscopy.agents import run as run_module

    seen = {}

    async def fake_run_goal_text(goal, worker="team", model=None):
        seen.update(goal=goal, worker=worker)
        return "Goal not reached.", False

    monkeypatch.setattr(run_module, "run_goal_text", fake_run_goal_text)
    monkeypatch.setattr("sys.argv", ["run", "team", "--goal", "map a 600 nm square here", "--max-minutes", "5", "--max-dose", "40"])
    with pytest.raises(SystemExit) as exited:
        run_module.main()
    assert exited.value.code == 1 and seen["worker"] == "team"
    assert seen["goal"].budget.max_seconds == 300 and seen["goal"].budget.max_dose_e_per_a2 == 40
    assert "Goal not reached." in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["run", "mapping", "--goal", "map here"])
    with pytest.raises(SystemExit) as refused:
        run_module.main()
    assert refused.value.code == 2  # the mapping graph is a fixed workflow, not a goal worker


def test_agent_server_goal_accepts_text_or_json(monkeypatch):
    from asyncroscopy.mcp import agent_mcp_server

    seen = []

    async def fake_run_goal_text(goal, worker="team", model=None):
        seen.append((goal, worker))
        return "Goal reached.", True

    monkeypatch.setattr(agent_mcp_server, "run_goal_text", fake_run_goal_text)
    server = agent_mcp_server.AgentMCPServer("t", ["http://x/mcp"], verbose=False)
    assert asyncio.run(server.query_agent("map here", goal=True)) == "Goal reached."
    asyncio.run(server.query_agent('{"statement": "map here", "criteria": [{"kind": "map"}]}', graph="general", goal=True))
    assert seen[0] == ("map here", "team") and seen[1] == ({"statement": "map here", "criteria": [{"kind": "map"}]}, "general")
