"""The team graph's planning, routing, retry and reporting, with a fake model and fake specialists."""

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import pytest

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.graphs import team
from asyncroscopy.agents.graphs.team import TeamGraph, heuristic_plan, role_of
from asyncroscopy.agents.memory import Memory

TOOL_NAMES = [
    "run_job", "Mapper_reserve", "Mapper_start", "wait_for_job", "Mapper_results", "Mapper_release", "DigitalTwin_get_stage",
    "DigitalTwin_move_stage", "DigitalTwin_acquire_scanned_image", "approval_status", "Approvals_get_mode", "Approvals_set_mode",
    "list_skills", "load_skill", "list_data", "view_data", "assemble_map", "score_map", "analysis_job_status",
    "analysis_job_results", "map_info", "get_region", "list_pois", "add_poi", "list_devices", "DigitalTwin_State",
]


@pytest.fixture(autouse=True)
def memory_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_AGENT_MEMORY", str(tmp_path / "memory"))


def test_tools_are_split_by_role_and_redundant_or_operator_only_tools_are_hidden():
    roles = {n: role_of(n) for n in TOOL_NAMES}
    assert roles["run_job"] == roles["DigitalTwin_move_stage"] == roles["DigitalTwin_acquire_scanned_image"] == "operator"
    assert roles["wait_for_job"] == roles["approval_status"] == roles["load_skill"] == "shared"
    assert roles["view_data"] == roles["add_poi"] == roles["assemble_map"] == "analyst"
    # run_job and wait_for_job cover these; they stay on the endpoint but no specialist gets them.
    for hidden in ("Mapper_reserve", "Mapper_start", "Mapper_results", "Mapper_release", "analysis_job_status", "analysis_job_results",
                   "list_skills", "Approvals_get_mode", "Approvals_set_mode", "DigitalTwin_State"):
        assert roles[hidden] is None, hidden
    assert role_of("Mapper_abort") == "operator"  # stopping stays possible
    assert role_of("XRDTwin_acquire_diffractogram") == "operator"  # a YAML-defined instrument lands in a role too
    assert role_of("XRDTwin_set_start_deg") is None and role_of("DigitalTwin_set_scan_rotation") is None  # STEM: curated
    assert role_of("XRDTwin_set_start_deg", stem=False) == "operator"  # any other instrument: every device command
    assert role_of("XRDTwin_State", stem=False) is None and role_of("XRDTwin_reserve", stem=False) is None


def test_heuristic_plan_maps_then_analyses():
    steps = heuristic_plan("Map a 600 nm square here and find the 3 most interesting regions", {"operator", "analyst"})
    assert [s["agent"] for s in steps] == ["operator", "analyst"]
    assert [s["agent"] for s in heuristic_plan("What is in the latest spectrum?", {"operator", "analyst"})] == ["analyst"]
    assert heuristic_plan("hello", {"analyst"}) == [{"agent": "analyst", "task": "hello"}]


class FakeModel:
    """Planner and reporter replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def ainvoke(self, messages):
        self.seen.append(messages)
        return SimpleNamespace(content=self.replies.pop(0))


def tool(name):
    return SimpleNamespace(name=name)


class FakeAgent:
    def __init__(self, script):
        self.script = script

    async def ainvoke(self, inputs, config=None):
        result = self.script(inputs["messages"][0].content)
        if isinstance(result, Exception):
            raise result
        return {"messages": [inputs["messages"][0], *result]}


def tool_turn(name, args, result, call_id, status="success"):
    return [AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}]), ToolMessage(result, tool_call_id=call_id, status=status)]


def make_team(model_replies, scripts, monkeypatch):
    graph = TeamGraph(AgentConfig(analysis_url=""))
    model = FakeModel(model_replies)
    monkeypatch.setattr(graph.config, "chat_model", lambda: model)

    async def tools(allow=None, include_analysis=True):
        return [tool(n) for n in TOOL_NAMES]

    monkeypatch.setattr(graph.config, "tools", tools)

    async def info():  # never the live endpoint: a test must not pass or fail with whatever stack is running
        return {"analysis_url": "http://127.0.0.1:8003/mcp"}

    monkeypatch.setattr(graph.config, "info", info)
    built = []

    def specialist(role, tools_, task, notes):
        built.append((role, sorted(t.name for t in tools_), task, list(notes)))
        return FakeAgent(scripts[role].pop(0))

    monkeypatch.setattr(graph, "specialist", specialist)
    return graph, model, built


def test_a_planned_run_routes_steps_and_reports_from_facts(monkeypatch):
    plan = json.dumps({"steps": [{"agent": "operator", "task": "map 600 nm here"}, {"agent": "analyst", "task": "stitch map_r1.h5"}]})
    operator = lambda task: [*tool_turn("run_job", {"device": "Mapper", "plan": {}}, json.dumps({"job_id": "j1", "state": "running"}), "c1"),
                             *tool_turn("wait_for_job", {"job_id": "j1"}, json.dumps({"job_id": "j1", "state": "done", "result_keys": {"run": "map_r1.h5"}}), "c2"),
                             AIMessage("Ran the map; run key map_r1.h5.")]
    analyst = lambda task: [*tool_turn("wait_for_job", {"job_id": "a1"}, json.dumps({"job_id": "a1", "state": "done", "result_keys": {"map": "map_r1_map"}}), "c3"),
                            AIMessage("Stitched map_r1_map.")]
    graph, model, built = make_team([plan, "Mapped and stitched: map_r1_map."], {"operator": [operator], "analyst": [analyst]}, monkeypatch)
    state = asyncio.run(graph.build().ainvoke({"request": "map a 600 nm square and stitch it"}))
    assert [b[0] for b in built] == ["operator", "analyst"]
    operator_tools, analyst_tools = built[0][1], built[1][1]
    assert "run_job" in operator_tools and "view_data" not in operator_tools and "Mapper_start" not in operator_tools
    assert "view_data" in analyst_tools and "run_job" not in analyst_tools and "analysis_job_status" not in analyst_tools
    assert "wait_for_job" in operator_tools and "wait_for_job" in analyst_tools
    assert "list_skills" not in operator_tools + analyst_tools and "Approvals_set_mode" not in operator_tools
    assert "map_r1.h5" in built[1][3][0]  # the analyst sees what the operator produced
    assert state["answer"] == "Mapped and stitched: map_r1_map."
    report_prompt = model.seen[-1][1].content
    assert "map_r1_map" in report_prompt and "map_r1.h5" in report_prompt
    assert [c["name"] for c in state["trace"]] == ["run_job", "wait_for_job", "wait_for_job"]


def test_an_unusable_plan_falls_back_and_a_failed_step_is_retried_once(monkeypatch):
    attempts = []

    def operator(task):
        attempts.append(task)
        if len(attempts) == 1:
            return RuntimeError("Recursion limit of 40 reached")
        return [*tool_turn("run_job", {}, json.dumps({"job_id": "j2", "state": "running"}), "c1"), AIMessage("started")]

    graph, _model, built = make_team(["I think you should map it.", ""], {"operator": [operator, operator]}, monkeypatch)
    state = asyncio.run(graph.build().ainvoke({"request": "acquire an image here"}))
    assert [b[0] for b in built] == ["operator", "operator"]
    assert "first try at this step failed" in attempts[1] and "Recursion limit" in attempts[1]
    assert "Steps run" in state["answer"]  # an empty report from the model falls back to the notes


def test_a_specialist_gets_its_rules_the_notes_and_skills_found_for_each_call(monkeypatch, tmp_path):
    graph = TeamGraph(AgentConfig(analysis_url=""), Memory(tmp_path / "memory"))
    graph._all_names = set(TOOL_NAMES)
    captured = {}
    monkeypatch.setattr(team, "create_agent", lambda model, tools, system_prompt, middleware: captured.update(prompt=system_prompt, middleware=middleware))
    monkeypatch.setattr(graph.config, "chat_model", lambda: None)
    analyst_tools = [tool(n) for n in TOOL_NAMES if role_of(n) in ("analyst", "shared")]
    graph.specialist("analyst", analyst_tools, "find the 3 most interesting regions on map_r1_map", ["[operator] ran map -> map_r1.h5"])
    assert captured["prompt"].startswith(team.ROLE_PROMPTS["analyst"]) and "map_r1.h5" in captured["prompt"]
    skills = captured["middleware"][0]
    update = skills.before_model({"messages": [HumanMessage("find the 3 most interesting regions on map_r1_map")]}, None)
    assert update["skills_applied"][0]["skill"] == "explore_regions"


def test_skills_applied_by_each_step_are_kept(monkeypatch):
    applied = [{"skill": "mapping", "reason": "matches the request"}]

    class Agent:
        async def ainvoke(self, inputs, config=None):
            return {"messages": [inputs["messages"][0], AIMessage("done")], "skills_applied": applied}

    graph, _model, _built = make_team([json.dumps({"steps": [{"agent": "operator", "task": "map here"}]}), "ok"], {"operator": []}, monkeypatch)
    monkeypatch.setattr(graph, "specialist", lambda role, tools_, task, notes: Agent())
    state = asyncio.run(graph.build().ainvoke({"request": "map here"}))
    assert state["skills_applied"] == [{"step": 0, "agent": "operator", "skill": "mapping", "reason": "matches the request"}]
