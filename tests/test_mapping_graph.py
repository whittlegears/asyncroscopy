"""The mapping graph's control flow, with fake MCP servers (no model, no instrument)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.graphs import mapping
from asyncroscopy.agents.graphs.mapping import MappingGraph


def reply(payload, error=False):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(is_error=error, content=[SimpleNamespace(text=text)], structured_content=None)


class FakeInstrument:
    def __init__(self, hold_for_approval=False, approval="approved"):
        self.calls = []
        self.hold = hold_for_approval
        self.approval = approval
        self.status_polls = 0

    async def list_tools(self):
        return [SimpleNamespace(name=n) for n in ("DigitalTwin_move_stage", "DigitalTwin_acquire_scanned_image", "DigitalTwin_get_parameters", "Mapper_reserve")]

    async def call(self, name, arguments=None):
        self.calls.append(name)
        if name == "DigitalTwin_get_parameters":
            return reply({"stage_position": [0, 0, 0, 0, 0], "fov_m": 2e-7, "world_bounds_angstrom": {"x_min": -5000, "x_max": 5000, "y_min": -5000, "y_max": 5000}})
        if name == "Mapper_reserve":
            plan = json.loads(arguments["request"])["plan"]
            assert plan["box_m"] == [-5e-7, 5e-7, -5e-7, 5e-7]
            return reply({"reservation_id": "r1", "plan_id": "p1", "resolved_plan": plan, "estimate": {"steps": 25}})
        if name == "Mapper_start":
            if self.hold:
                self.hold = False
                return reply("APPROVAL REQUIRED: Mapper_start is held for an operator (request_id=abc123). Call approval_status ...", error=True)
            return reply({"job_id": "j1", "state": "running"})
        if name == "approval_status":
            return reply({"request_id": "abc123", "state": self.approval})
        if name == "Mapper_job_status":
            self.status_polls += 1
            return reply({"job_id": "j1", "state": "done" if self.status_polls > 1 else "running", "progress": {"done": 25, "total": 25}})
        if name == "Mapper_results":
            keys = {"run": "map_r1.h5", **({"map": "map_r1_map"} if self.stitched else {})}
            return reply({"state": "done", "result_keys": keys, "summary": {"spectra": {"taken": 1, "first_keys": ["spectrum_0.h5"]}}})
        if name == "Mapper_release":
            return reply({"state": "released"})
        if name == "Mapper_abort":
            self.status_polls = 99  # the job reports aborted from now on
            return reply({"job_id": "j1", "state": "aborting"})
        raise AssertionError(f"unexpected call {name}")

    async def call_json(self, name, arguments=None):
        return json.loads((await self.call(name, arguments)).content[0].text)


class FakeAnalysis:
    def __init__(self):
        self.calls = []

    async def call_json(self, name, arguments=None):
        self.calls.append(name)
        if name in ("assemble_map", "score_map"):
            return {"job_id": name}
        if name == "analysis_job_status":
            return {"state": "done"}
        if name == "analysis_job_results":
            if arguments["job_id"] == "assemble_map":
                return {"result_keys": {"map": "map_r1_map"}}
            return {"summary": {"added_pois": [{"kind": "candidate", "x_m": 2e-7, "y_m": -1e-7, "score": 0.8, "reason": "particles"}]}}
        if name == "list_pois":
            pois = [p for p in [{"id": "poi1", "kind": "eds_point", "x_m": 1e-7, "y_m": 1e-7, "links": ["spectrum_0.h5"]}] if p["kind"] == arguments.get("kind", p["kind"])]
            return {"total": len(pois), "pois": pois[: arguments.get("limit", 200)]}
        raise AssertionError(name)


def run(graph_obj, state):
    return asyncio.run(graph_obj.build().ainvoke(state))


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch):
    monkeypatch.setenv("ASYNCROSCOPY_POLL_S", "0")


def make(hold=False, approval="approved", stitched=True):
    g = MappingGraph(AgentConfig(analysis_url="x"))
    g.instrument, g.analysis = FakeInstrument(hold, approval), FakeAnalysis()
    g.instrument.stitched = stitched
    return g


PLAN = {"fov_m": 2e-7, "per_tile": ["eds_point"]}


def test_full_run_with_approval_hold():
    g = make(hold=True)
    state = run(g, {"request": "map the whole sample with EDS and find interesting regions", "plan": dict(PLAN)})
    assert not state.get("error"), state
    assert g.instrument.calls.count("Mapper_start") == 2  # held, then repeated once approved
    assert "approval_status" in g.instrument.calls and "Mapper_release" in g.instrument.calls
    assert "assemble_map" not in g.analysis.calls and "score_map" in g.analysis.calls  # the Mapper job stitched its own run
    assert state["eds_points"] == 1 and [p["kind"] for p in state["pois"]] == ["candidate"]
    assert "after operator approval" in state["report"] and "Interesting region" in state["report"]
    # State carries keys and coordinates, never arrays.
    assert len(json.dumps(state, default=str)) < 20_000


def test_a_run_the_job_did_not_stitch_is_stitched_here():
    g = make(stitched=False)
    state = run(g, {"request": "map the whole sample with EDS", "plan": dict(PLAN)})
    assert not state.get("error"), state
    assert g.analysis.calls[:2] == ["assemble_map", "analysis_job_status"] and state["map_key"] == "map_r1_map"
    assert state["eds_points"] == 1 and state["pois"] == []


def test_denied_approval_ends_with_reason_and_releases():
    g = make(hold=True, approval="denied")
    state = run(g, {"request": "map the sample", "plan": dict(PLAN)})
    assert "did not approve" in state["report"]
    assert "Mapper_release" in g.instrument.calls and "Mapper_job_status" not in g.instrument.calls


def test_a_node_that_raises_ends_in_a_report_and_releases():
    g = make()

    async def broken(state):
        raise RuntimeError("tiled is down")

    g.stitch = broken  # build() looks nodes up by name, so the patched one is used
    state = run(g, {"request": "map the sample", "plan": dict(PLAN)})
    assert "broken failed: RuntimeError" in state["report"]
    assert state["report"].startswith("Map not completed") and "tiled is down" in state["report"]
    assert "Mapper_release" in g.instrument.calls


def test_a_failure_while_the_map_runs_aborts_it_before_releasing():
    g = make()

    async def lost(state):
        raise ConnectionError("endpoint dropped")

    g.wait = lost
    state = run(g, {"request": "map the sample", "plan": dict(PLAN)})
    assert state["report"].startswith("Map not completed")
    calls = g.instrument.calls
    assert "Mapper_abort" in calls and "Mapper_release" in calls
    assert calls.index("Mapper_abort") < calls.index("Mapper_release")  # never left running unattended


@pytest.mark.parametrize(
    "model_plan, request_text, expect",
    [
        ({"box_m": [500, -500, -500, 500], "fov_m": 200}, "map a 1 um square", {"box_m": [-5e-7, 5e-7, -5e-7, 5e-7], "fov_m": 2e-7}),
        ({}, "map the whole sample", {"box_m": [-5e-7, 5e-7, -5e-7, 5e-7]}),
        ({"per_tile": ["eds_map"]}, "map the whole sample with an EDS spectrum at each image", {"per_tile": ["eds_point"]}),
        ({"per_tile": ["eds_map"]}, "acquire an EDS spectrum image of every tile of the whole sample", {"per_tile": ["eds_map"]}),
        ({"per_tile": ["eds_point"]}, "map a 600 nm square", {"per_tile": []}),  # EDS nobody asked for
        ({"box_m": [-0.5, 0.5, -0.5, 0.5], "fov_m": 0.2}, "map the sample", {"box_m": [-5e-7, 5e-7, -5e-7, 5e-7], "fov_m": 2e-7}),  # micrometres
        ({}, "map the entire sample and tell me the composition", {"per_tile": ["eds_point"]}),
        ({"box_m": [0, 1e-6, 0, 1e-8], "fov_m": 2e-7}, "map a thin strip", {"box_m": [0, 1e-6, pytest.approx(-9.5e-8), pytest.approx(1.05e-7)]}),
    ],
)
def test_plan_completion_repairs_small_model_slips(model_plan, request_text, expect):
    context = {"sample_bounds_m": [-5e-7, 5e-7, -5e-7, 5e-7], "stage_position_m": [0, 0, 0, 0, 0], "fov_m": 2e-7}
    plan = MappingGraph._complete_plan(model_plan, context, request_text)
    for key, value in expect.items():
        numeric = isinstance(value, (int, float)) or (isinstance(value, list) and all(isinstance(v, (int, float)) for v in value))
        assert plan.get(key, []) == (pytest.approx(value) if numeric else value)  # no per_tile field means images only


SEM_CONTEXT = {"stage_position_m": [0.025, 0.012, 0.004, 0, 0], "fov_m": 1e-3, "microscope": "SEM", "eds": False}


@pytest.mark.parametrize("context", [SEM_CONTEXT, {**SEM_CONTEXT, "fov_m": 1e-5}, {**SEM_CONTEXT, "fov_m": None}])
def test_an_sem_scale_plan_in_metres_is_not_rescaled(context):
    plan = {"box_m": [0.020, 0.030, 0.010, 0.014], "fov_m": 2e-3}
    got = MappingGraph._complete_plan(dict(plan), context, "map this 10 x 4 mm area", from_model=True)
    assert got["fov_m"] == pytest.approx(2e-3)  # 2 mm stays 2 mm, not 2 nm
    assert got["box_m"] == pytest.approx(plan["box_m"])  # a stage at x = 25 mm stays there, not at 25 nm
    centred = MappingGraph._complete_plan({"size_m": [0.01, 0.004], "fov_m": 2e-3}, context, "", from_model=False)
    assert centred["box_m"] == pytest.approx([0.020, 0.030, 0.010, 0.014])
    assert "per_tile" not in got and "per_tile" not in centred  # the sem profile refuses even an empty per_tile


def test_an_sem_without_eds_gets_no_eds_even_when_the_request_mentions_composition():
    got = MappingGraph._complete_plan({"box_m": [0.020, 0.030, 0.010, 0.014], "fov_m": 2e-3, "per_tile": ["eds_point"]}, SEM_CONTEXT, "map this area and show the composition")
    assert "per_tile" not in got


def test_the_stem_unit_repair_is_unchanged_by_the_instrument_scale():
    from asyncroscopy.agents.graphs.mapping import _plausible_limits

    stem = {"sample_bounds_m": [-5e-7, 5e-7, -5e-7, 5e-7], "stage_position_m": [0, 0, 0, 0, 0], "fov_m": 2e-7}
    assert _plausible_limits(stem) == (1e-3, 1e-2)  # the old cut-offs
    got = MappingGraph._complete_plan({"box_m": [-0.5, 0.5, -0.5, 0.5], "fov_m": 0.2}, stem, "map the sample")
    assert got["fov_m"] == pytest.approx(2e-7) and got["box_m"] == pytest.approx([-5e-7, 5e-7, -5e-7, 5e-7])


def test_the_plan_prompt_names_the_instrument_and_offers_eds_only_when_it_has_it():
    from asyncroscopy.agents.graphs.mapping import plan_prompt

    stem = plan_prompt({"microscope": "DigitalTwin", "eds": True})
    sem = plan_prompt(SEM_CONTEXT)
    assert "DigitalTwin" in stem and '"per_tile"' in stem and "eds_point" in stem
    assert "SEM" in sem and "per_tile" not in sem and "eds" not in sem.replace('"eds": false', "")
    assert "1 mm = 1e-3 m" in sem


def test_an_area_inside_one_tile_is_refused_as_an_image_not_a_map():
    context = {"sample_bounds_m": [-5e-7, 5e-7, -5e-7, 5e-7], "stage_position_m": [0, 0, 0, 0, 0], "fov_m": 2e-7}
    with pytest.raises(ValueError, match="one image, not a map"):
        MappingGraph._complete_plan({"box_m": [0, 1e-8, 0, 1e-8], "fov_m": 2e-7}, context, "map a tiny spot")


@pytest.mark.parametrize(
    "request_text, size",
    [
        ("Map a 600 nm square around the current stage position.", (6e-7, 6e-7)),
        ("map a 2 x 1 um area here", (2e-6, 1e-6)),
        ("Map a 1.5 µm region with EDS", (1.5e-6, 1.5e-6)),
        ("map 300nm by 500 nm patch", (3e-7, 5e-7)),
        ("map the whole sample", None),
        ("take 3 images of 500 nm fov", None),
    ],
)
def test_requested_size_is_read_from_the_request(request_text, size):
    got = MappingGraph._requested_size(request_text)
    assert got == (pytest.approx(size) if size else None)


CONTEXT = {"sample_bounds_m": [-2e-6, 2e-6, -2e-6, 2e-6], "stage_position_m": [0, 0, 0, 0, 0], "fov_m": 1.6e-7}


@pytest.mark.parametrize("plan, from_model", [({}, True), ({"fov_m": 1.6e-7}, True), ({"fov_m": 1.6e-7}, False)])
def test_a_plan_without_an_area_is_refused_unless_the_whole_sample_is_asked_for(plan, from_model):
    with pytest.raises(ValueError, match="no area"):
        MappingGraph._complete_plan(dict(plan), CONTEXT, "image the particle here", from_model=from_model)
    whole = MappingGraph._complete_plan(dict(plan), CONTEXT, "map the whole sample", from_model=from_model)
    assert whole["box_m"] == pytest.approx(CONTEXT["sample_bounds_m"])


@pytest.mark.parametrize("request_text", [
    "find a grain boundary on the sample and map around it",
    "image a small patch near the edge of the sample",
    "map the region of interest in the sample",
    "move to the sample holder and image",
    "check everything is ok then map a small region",
])
def test_mentioning_the_sample_is_not_a_whole_sample_request(request_text):
    with pytest.raises(ValueError, match="no area"):
        MappingGraph._complete_plan({}, CONTEXT, request_text)


def test_whole_sample_without_reported_bounds_is_refused():
    with pytest.raises(ValueError, match="no sample bounds"):
        MappingGraph._complete_plan({}, {"stage_position_m": [0, 0, 0, 0, 0]}, "map the whole sample")


def test_size_and_centre_are_both_unit_repaired():
    plan = MappingGraph._complete_plan({"size_m": [600, 600], "centre_m": [100, 0], "fov_m": 1.6e-7}, CONTEXT, "", from_model=False)
    assert plan["box_m"] == pytest.approx([-2e-7, 4e-7, -3e-7, 3e-7])
    assert "size_m" not in plan and "centre_m" not in plan


def test_a_plan_passed_in_keeps_its_eds_and_ignores_the_request_text():
    plan = MappingGraph._complete_plan({"box_m": [0, 1e-6, 0, 1e-6], "fov_m": 2e-7, "per_tile": ["eds_point"]}, CONTEXT, "Map a 600 nm square", from_model=False)
    assert plan["per_tile"] == ["eds_point"]
    assert plan["box_m"] == pytest.approx([0, 1e-6, 0, 1e-6])


class FakeModel:
    def __init__(self, text):
        self.text = text

    async def ainvoke(self, messages):
        return SimpleNamespace(content=self.text)


def test_a_prose_reply_with_no_area_in_the_request_ends_without_touching_the_mapper(monkeypatch):
    g = make()
    monkeypatch.setattr(g.config, "chat_model", lambda: FakeModel("I would image that particle now."))
    state = run(g, {"request": "image the particle here"})
    assert state["report"].startswith("Map not completed") and "no area" in state["report"]
    assert not any(c.startswith("Mapper") and c != "Mapper_release" for c in g.instrument.calls)


def test_a_prose_reply_still_maps_what_the_request_states(monkeypatch):
    g = make()
    monkeypatch.setattr(g.config, "chat_model", lambda: FakeModel("Sure, mapping it."))
    state = run(g, {"request": "map the whole sample"})
    assert not state.get("error"), state
    assert "Mapper_start" in g.instrument.calls


def test_the_mapper_prefix_is_discovered():
    g = make()

    async def tools():
        return [SimpleNamespace(name=n) for n in ("Mapper_lab2_reserve", "Mapper_lab2_start", "DigitalTwin_move_stage")]

    g.instrument.list_tools = tools
    assert asyncio.run(g._mapper()) == "Mapper_lab2"


def test_several_mappers_need_a_choice(monkeypatch):
    g = make()

    async def tools():
        return [SimpleNamespace(name=n) for n in ("Mapper_a_reserve", "Mapper_b_reserve")]

    g.instrument.list_tools = tools
    with pytest.raises(RuntimeError, match="several Mappers"):
        asyncio.run(g._mapper())
    monkeypatch.setenv("ASYNCROSCOPY_MAPPER", "Mapper_b")
    g._mapper_prefix = None
    assert asyncio.run(g._mapper()) == "Mapper_b"


def test_stated_size_overrides_a_whole_sample_box():
    context = {"sample_bounds_m": [-1e-6, 1e-6, -1e-6, 1e-6], "stage_position_m": [1e-7, -2e-7, 0, 0, 0], "fov_m": 1.6e-7}
    plan = MappingGraph._complete_plan({"box_m": [-1e-6, 1e-6, -1e-6, 1e-6], "fov_m": 1.6e-7}, context, "Map a 600 nm square around the current stage position.")
    assert plan["box_m"] == pytest.approx([-2e-7, 4e-7, -5e-7, 1e-7])
