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
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.jobs.mapping.mapper import Mapper
from tests.catalogs import catalog, info_of

# What endpoint_info says the tools are: a STEM twin with two Mappers, and the Raman and XRD YAML twins.
CATALOG = catalog({"DigitalTwin": DigitalTwin, "Mapper": Mapper, "Mapper_key2": Mapper, "RamanTwin": "raman.yaml", "XRDTwin": "xrd.yaml"})


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
        if name == "endpoint_info":
            return reply(info_of(CATALOG))
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
    Call("Mapper_start", {"request": '{"reservation_id": "r1"}'}, json.dumps({"job_id": "j1", "state": "running"})),
    Call("wait_for_job", {"device": "Mapper", "job_id": "j1"}, json.dumps({"job_id": "j1", "state": "done"})),
    Call("Mapper_results", {"request": '{"job_id": "j1"}'}, json.dumps({"result_keys": {"run": "map_r1.h5"}, "summary": {"spectra": [{"key": "spectrum_eds_1.h5"}]}})),
    Call("assemble_map", {"run": "map_r1.h5"}, json.dumps({"job_id": "a1"})),
    Call("analysis_job_results", {"job_id": "a1"}, json.dumps({"result_keys": {"map": "map_r1_map"}})),
]


def test_criteria_are_inferred_from_the_goal_text():
    goal = Goal.from_text("Map a 600 nm square here with EDS and find the 3 most interesting regions")
    kinds = {c["kind"]: c for c in goal.criteria}
    assert kinds["map"]["min_pois"] == 3 and kinds["map"]["poi_kind"] == "candidate"
    assert "data" not in kinds and {"job", "acquire"} <= set(kinds["no_unrequested"]["categories"])  # the map job takes the spectra
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
    facts = Facts.from_calls(calls + MAP_CALLS, CATALOG)
    assert "map_fake_map" not in facts.map_keys  # the model's own text never counts
    assert facts.map_keys == {"map_r1_map"} and {"map_r1.h5", "spectrum_eds_1.h5"} <= facts.data_keys
    assert facts.jobs["j1"] == {"device": "Mapper", "state": "done", "kind": "instrument"}
    assert facts.dose_e_per_a2 == 20.0
    assert "DigitalTwin_move_stage" in facts.high_risk and "Mapper_start" in facts.high_risk


def check(goal, calls, answer, server, tools=CATALOG):
    return asyncio.run(GoalChecker(server, server).check(goal, Facts.from_calls(calls, tools), answer))


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
    assert "made no stitched map" in requests[1] and "did not reach the goal" in requests[1]
    assert state["report"].startswith("Goal reached.")


def test_the_loop_stops_after_its_attempts_and_says_why():
    goal = Goal("map here", [{"kind": "map"}])
    goal.budget.max_attempts = 2
    state, requests = run_goal_graph(goal, [("nothing", []), ("still nothing", [])], FakeServer())
    assert not state["met"] and len(requests) == 2
    assert state["report"].startswith("Goal not reached.") and "NOT met: this run made no stitched map" in state["report"]


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
    facts = Facts.from_calls(calls, CATALOG)
    assert facts.jobs == {"j9": {"device": "Mapper", "state": "done", "kind": "instrument"}, "a9": {"device": "analysis", "state": "done", "kind": "assemble"}}
    assert facts.dose_e_per_a2 == 7.5 and facts.map_keys == {"map_r9_map"} and "map_r9.h5" in facts.data_keys
    assert facts.high_risk == ["run_job"]  # it starts a job: allowed when the goal asks for one
    assert check(Goal("x", [{"kind": "no_unrequested"}]), calls, "", FakeServer()).met


def test_take_an_image_here_is_met_by_the_acquisition_it_needs():
    calls = [
        Call("DigitalTwin_set_fov", {"fov": 2e-7}, "ok"),
        Call("DigitalTwin_unblank_beam", {}, "ok"),
        Call("DigitalTwin_acquire_scanned_image", {"detector_list": ["haadf"]}, "stem_image_HAADF_20261028T1.h5"),
    ]
    goal = Goal.from_text("Take a HAADF image here")
    verdict = check(goal, calls, "Saved as stem_image_HAADF_20261028T1.h5.", None)
    assert verdict.met, verdict.results  # acquiring, its settings and unblanking are what the goal asked for
    moved = calls + [Call("DigitalTwin_move_stage", {"position": [1e-6, 0, 0, 0, 0]}, "ok")]
    assert "move_stage" in check(goal, moved, "", None).feedback()  # "here": no move


def test_an_old_spectrum_found_by_listing_does_not_meet_take_a_spectrum():
    calls = [Call("list_data", {"kind": "spectrum"}, json.dumps({"data": [{"key": "spectrum_eds_20251201T000000.h5", "kind": "spectrum"}]}))]
    goal = Goal.from_text("Take an EDS spectrum here")
    facts = Facts.from_calls(calls, CATALOG)
    assert facts.data_keys == set() and "spectrum_eds_20251201T000000.h5" in facts.seen_keys
    verdict = asyncio.run(GoalChecker().check(goal, facts, "The spectrum is spectrum_eds_20251201T000000.h5."))
    assert not verdict.met and "spectrum recorded in this run" in verdict.feedback()
    # A wait on a job this run did not start produces nothing either.
    other = [Call("wait_for_job", {"job_id": "old1"}, json.dumps({"job_id": "old1", "state": "done", "result_keys": {"map": "map_old1_map"}}))]
    assert Facts.from_calls(other, CATALOG).map_keys == set() and Facts.from_calls(other, CATALOG).jobs == {}


def test_an_invented_key_passed_to_a_tool_is_not_grounded():
    calls = [Call("view_data", {"key": "spectrum_made_up_1.h5"}, "Error: no key", True)]
    verdict = asyncio.run(GoalChecker().check(Goal("x", [{"kind": "grounded_answer"}]), Facts.from_calls(calls, CATALOG), "It is spectrum_made_up_1.h5."))
    assert not verdict.met


def test_a_read_only_goal_wants_no_data_and_allows_no_action():
    goal = Goal.from_text("Show me the latest spectrum")
    kinds = [c["kind"] for c in goal.criteria]
    assert kinds == ["no_unrequested", "grounded_answer"] and goal.criteria[0]["categories"] == []


def test_risk_comes_from_the_endpoints_catalog_never_from_a_name():
    calls = [Call("DigitalTwin_set_fov", {"fov": 2e-7}, "ok"), Call("DigitalTwin_acquire_scanned_image", {}, "img_1.h5")]
    assert Facts.from_calls(calls).high_risk == []  # no catalog: nothing is known, nothing guessed
    assert Facts.from_calls(calls, CATALOG).high_risk == ["DigitalTwin_acquire_scanned_image"]  # a STEM: set_fov does not expose the sample


def test_the_retry_names_only_this_goals_keys_and_its_running_job():
    goal = Goal("map here", [{"kind": "map"}])
    started = [Call("run_job", {"device": "Mapper", "plan": {}}, json.dumps({"reservation_id": "r2", "job_id": "j2", "state": "running"}))]
    listed = [Call("list_data", {}, json.dumps({"data": [{"key": "map_old7_map"}]}))]
    state, requests = run_goal_graph(goal, [("still mapping", started + listed), ("done", [])], FakeServer())
    assert "j2" in requests[1] and "do not start another job" in requests[1]
    assert "map_old7_map" not in requests[1] and "Use what it already produced" not in requests[1]


def test_jobs_left_running_when_a_goal_ends_are_aborted():
    goal = Goal("map here", [{"kind": "map"}])
    goal.budget.max_attempts = 1
    aborted = []

    class Server(FakeServer):
        async def call(self, name, args):
            if name.endswith("_abort"):
                aborted.append((name, json.loads(args["request"])))
                return reply({"state": "aborted"})
            return await super().call(name, args)

    started = [Call("run_job", {"device": "Mapper", "plan": {}}, json.dumps({"job_id": "j3", "state": "running"}))]
    state, _ = run_goal_graph(goal, [("ran out of time", started)], Server())
    assert aborted == [("Mapper_abort", {"job_id": "j3"})] and "Stopped jobs this goal had left running" in state["report"]


def test_a_worker_that_stops_part_way_keeps_its_calls():
    from asyncroscopy.agents.graphs import run_keeping_partial

    class Graph:
        async def astream(self, inputs, config=None, stream_mode=None):
            yield {"messages": inputs["messages"] + [AIMessage("", tool_calls=[{"name": "run_job", "args": {}, "id": "c1"}])]}
            yield {"messages": inputs["messages"] + [AIMessage("", tool_calls=[{"name": "run_job", "args": {}, "id": "c1"}]),
                                                     ToolMessage(json.dumps({"job_id": "j4", "state": "running"}), tool_call_id="c1")]}
            raise RuntimeError("Recursion limit of 9 reached")

    state, failed = asyncio.run(run_keeping_partial(Graph(), {"messages": [HumanMessage("map")]}))
    assert "Recursion limit" in failed
    assert Facts.from_calls(trace_from_messages(state["messages"]), CATALOG).jobs == {"j4": {"device": "unknown", "state": "running", "kind": "instrument"}}


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

    monkeypatch.setattr("asyncroscopy.agents.run.run_goal_text", fake_run_goal_text)
    server = agent_mcp_server.AgentMCPServer("t", ["http://x/mcp"], verbose=False)
    assert asyncio.run(server.query_agent("map here", goal=True)) == "Goal reached."
    asyncio.run(server.query_agent('{"statement": "map here", "criteria": [{"kind": "map"}]}', graph="general", goal=True))
    assert seen[0] == ("map here", "team") and seen[1] == ({"statement": "map here", "criteria": [{"kind": "map"}]}, "general")


OLD = "map_20250101_abc123def456_map"


def test_scoring_or_restitching_an_existing_map_does_not_make_it_this_runs():
    calls = [
        Call("list_data", {"kind": "map"}, json.dumps({"data": [{"key": OLD}]})),
        Call("score_map", {"map": OLD, "top_k": 3}, json.dumps({"job_id": "aaaaaaaaaaaa", "state": "queued"})),
        Call("wait_for_job", {"job_id": "aaaaaaaaaaaa"}, json.dumps({"job_id": "aaaaaaaaaaaa", "state": "done", "result_keys": {"map": OLD}})),
        Call("assemble_map", {"run": "map_20250101.h5"}, json.dumps({"job_id": "bbbbbbbbbbbb"})),  # a run this run did not acquire
        Call("wait_for_job", {"job_id": "bbbbbbbbbbbb"}, json.dumps({"job_id": "bbbbbbbbbbbb", "state": "done", "result_keys": {"map": "map_20250101_map"}})),
    ]
    facts = Facts.from_calls(calls, CATALOG)
    assert facts.map_keys == set() and "bbbbbbbbbbbb" not in facts.jobs
    assert not check(Goal.from_text("Map a 600 nm square around here."), calls, "", FakeServer(maps={OLD})).met


@pytest.mark.parametrize("request_text", [
    f"The map {OLD} has a point of interest at x = 10 nm, y = 5 nm. Go back to it and take a HAADF image there.",
    f"On the stitched map {OLD}, go to its highest-scoring point of interest and image it at a 100 nm field of view.",
    f"On the map {OLD}, find the three most interesting regions and image them at high magnification.",
])
def test_a_request_about_an_existing_map_never_asks_for_a_new_one(request_text):
    goal = Goal.from_text(request_text)
    kinds = {c["kind"]: c for c in goal.criteria}
    assert "map" not in kinds and "job" not in kinds["no_unrequested"]["categories"]
    assert kinds["data"]["data_kind"] == "image" and "move" in kinds["no_unrequested"]["categories"]
    graph = GoalGraph(goal, "team", AgentConfig(analysis_url=""), GoalChecker(), run_worker=None, learn=False)
    retry = graph.request("- not yet: 0 image recorded in this run, 1 wanted", Facts())
    assert "map" not in retry.split("An earlier attempt")[1].lower()  # the retry never pushes a new map


def test_without_eds_means_no_spectra():
    goal = Goal.from_text("Make an overview map of a 1 micrometre square around here, without EDS.")
    kinds = {c["kind"]: c for c in goal.criteria}
    assert "map" in kinds and "data" not in kinds and kinds["no_unrequested"]["categories"] == ["job"]
    image_only = Goal.from_text("Take an image here; no EDS spectrum please.")
    assert [c.get("data_kind") for c in image_only.criteria if c["kind"] == "data"] == ["image"]


def test_unsure_means_fewer_criteria_not_wrong_ones():
    kinds = [c["kind"] for c in Goal.from_text("What are the current field of view and defocus? Do not change anything.").criteria]
    assert kinds == ["no_unrequested", "grounded_answer"]
    start_only = Goal.from_text("Start mapping a 1 micrometre square around the current position. Do not wait for it to finish.")
    assert start_only.leave_running and "map" not in {c["kind"] for c in start_only.criteria}


@pytest.mark.parametrize("request_text, wanted", [
    ("Take HAADF images at defocus -20 nm, 0 nm and +20 nm.", {"acquire", "settings"}),
    ("Map a 600 nm square and take an EDS spectrum with each image.", {"job", "acquire"}),
    ("Find a particle and take an EDS spectrum on it.", {"acquire", "move"}),
    ("Move the stage to x = 1 mm, take a Raman spectrum at 2 mW and switch the laser off afterwards.", {"move", "acquire", "beam"}),
    ("Acquire a diffractogram of the mounted sample.", {"acquire"}),
])
def test_allowed_categories_follow_the_request(request_text, wanted):
    categories = set(next(c for c in Goal.from_text(request_text).criteria if c["kind"] == "no_unrequested")["categories"])
    assert wanted <= categories


def test_categories_follow_any_instruments_tool_roles():
    from asyncroscopy.agents.goals import action_category

    assert [action_category(n, CATALOG) for n in (
        "RamanTwin_acquire_raman_spectrum", "RamanTwin_laser_on", "RamanTwin_set_stage_x_m", "XRDTwin_xray_on",
        "XRDTwin_acquire_diffractogram", "DigitalTwin_set_defocus", "Mapper_key2_start", "run_job", "DigitalTwin_unblank_beam",
        "DigitalTwin_move_stage",
    )] == ["acquire", "beam", "move", "beam", "acquire", "settings", "job", "job", "beam", "move"]
    raman = [Call("RamanTwin_set_stage_x_m", {}, "ok"), Call("RamanTwin_laser_on", {}, "ok"), Call("RamanTwin_acquire_raman_spectrum", {}, "raman_1.h5"), Call("RamanTwin_laser_off", {}, "ok")]
    goal = Goal.from_text("Move the stage to x = 1 mm, take a Raman spectrum at 2 mW and switch the laser off afterwards.")
    assert check(goal, raman, "raman_1.h5", None).met  # every action was one the request asked for


def test_without_the_endpoints_risk_table_no_unrequested_is_not_checked():
    calls = [Call("DigitalTwin_move_stage", {}, "ok")]
    verdict = check(Goal("x", [{"kind": "no_unrequested", "categories": []}]), calls, "", None, tools=None)
    assert verdict.met and "not checked" in verdict.results[0]["why"]


def test_a_multi_mapper_endpoints_job_belongs_to_its_own_device():
    calls = [Call("Mapper_key2_start", {"request": "{}"}, json.dumps({"job_id": "j7", "state": "running"}))]
    assert Facts.from_calls(calls, CATALOG).jobs["j7"]["device"] == "Mapper_key2"


class AbortingServer(FakeServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aborted = []

    async def call(self, name, args):
        if name.endswith("_abort"):
            self.aborted.append(name)
            return reply({"state": "aborted"})
        return await super().call(name, args)


def test_a_job_started_before_a_timeout_is_known_from_the_ledger_and_stopped():
    goal = Goal("map here", [{"kind": "map"}])
    goal.budget.max_attempts = 1
    server = AbortingServer()
    config = AgentConfig(analysis_url="")

    async def worker(request, remaining_s):  # the team was mid-step: its state has no trace yet
        config.ledger.append({"name": "run_job", "args": {"device": "Mapper", "plan": {}}, "result": json.dumps({"job_id": "j8", "state": "running"}), "error": False})
        raise asyncio.TimeoutError()

    graph = GoalGraph(goal, "team", config, GoalChecker(server, server), run_worker=worker, learn=False)
    state = asyncio.run(graph.build().ainvoke({"goal": goal.to_dict()}))
    assert server.aborted == ["Mapper_abort"] and "j8" in state["report"]


def test_a_job_the_user_said_not_to_wait_for_is_left_running():
    goal = Goal.from_text("Start mapping a 1 micrometre square here. Do not wait for it to finish; tell me the job id.")
    server = AbortingServer()
    started = [Call("run_job", {"device": "Mapper", "plan": {}}, json.dumps({"job_id": "j9", "state": "running"}))]
    state, _ = run_goal_graph(goal, [("Job j9 is running.", started)], server)
    assert server.aborted == [] and "Left running, as asked: job j9" in state["report"]


def test_an_endpoint_without_an_analysis_server_means_none(monkeypatch):
    from asyncroscopy.agents.mcp_tools import McpServer

    async def endpoint_info(self, name, arguments=None):
        return SimpleNamespace(is_error=False, structured_content={"analysis_url": None, "risks": {}}, content=[])

    monkeypatch.setattr(McpServer, "call", endpoint_info)
    config = AgentConfig()
    asyncio.run(config.info())
    assert config.analysis_url == "" and config.analysis() is None  # never the STEM's pyTEMlib on 8003
    config = AgentConfig(analysis_url="http://elsewhere:9/mcp")
    asyncio.run(config.info())
    assert config.analysis().url == "http://elsewhere:9/mcp"  # an explicit setting wins



def test_a_dose_the_user_states_is_a_budget_checked_against_the_job_estimates():
    goal = Goal.from_text("Map a 1 um square at no more than 20 e/A^2.")
    assert goal.budget.max_dose_e_per_a2 == 20.0
    reserve = Call("run_job", {"device": "Mapper"}, json.dumps({"job_id": "j1", "state": "done", "estimate": {"max_dose_e_per_a2": 35.0}}))
    verdict = check(Goal(goal.statement, [], goal.budget), [reserve], "done", FakeServer())
    assert not verdict.met and "over the budget of 20.0" in verdict.over_budget
    assert Goal.from_text("Map a 1 um square.").budget.max_dose_e_per_a2 is None


def test_a_restitched_map_key_is_still_a_map_key():
    from asyncroscopy.agents.goals import MAP_KEY

    assert MAP_KEY.search("now map_f22dde7faf84_map_20261029T023056_33d6.").group(0) == "map_f22dde7faf84_map_20261029T023056_33d6"
    assert MAP_KEY.search("map_f22dde7faf84_map").group(0) == "map_f22dde7faf84_map"
    version = "map_f22dde7faf84_map_20261030T174128543Z_b1be"
    assert MAP_KEY.search(f"the new version {version}, beside the old one").group(0) == version
    assert MAP_KEY.search("call assemble_map then score_map") is None
