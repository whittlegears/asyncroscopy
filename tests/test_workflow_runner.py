"""The workflow runner's own handling (no stack, no model): placeholders, list_pois replies, per-workflow
sleepwalking, empty answers, partial transcripts, a wedged device, and the checks fixed after the 31-workflow run."""

import asyncio
import tempfile
from pathlib import Path
import json

import pytest

from benchmarks import _common, run_workflows
from benchmarks.workflow_checks import Call, Run, evaluate, workflow_allow

CALLER = "wf-test"


def job(action, run_id="r1", ok=True, **fields):
    return {"source": "job", "action": action, "ok": ok, "ts": 1.0, "run_id": run_id, "caller": CALLER, **fields}


def ok(run, *specs):
    return {name: result["ok"] for name, result in evaluate(run, list(specs)).items()}


# --- placeholders and replies ------------------------------------------------------------------------------

def test_a_placeholder_without_a_value_stops_the_run_as_infra():
    assert run_workflows.render("x = {poi_x_nm} nm", {"poi_x_nm": 200.0}) == "x = 200.0 nm"
    assert run_workflows.render("{poi_x}", {"poi_x": 2e-7}) == 2e-7  # a whole placeholder keeps its type
    for facts in ({}, {"poi_x_nm": None}):
        with pytest.raises(run_workflows.InfraError, match="poi_x_nm"):
            run_workflows.render("x = {poi_x_nm} nm", facts)
        with pytest.raises(run_workflows.InfraError):
            run_workflows.render({"x": "{poi_x_nm}"}, facts)


@pytest.mark.parametrize("reply", [
    {"total": 2, "pois": [{"x_m": 1e-7, "y_m": 2e-7}, {"x_m": 3e-7, "y_m": 4e-7}]},  # pyTEMlib list_pois
    {"result": [{"x_m": 1e-7, "y_m": 2e-7}, {"x_m": 3e-7, "y_m": 4e-7}]},
    [{"x_m": 1e-7, "y_m": 2e-7}, {"x_m": 3e-7, "y_m": 4e-7}],
])
def test_points_of_interest_are_read_from_every_list_pois_shape(reply):
    class Analysis:
        async def call_json(self, name, args):
            return reply

    live = run_workflows.Live.__new__(run_workflows.Live)
    live.analysis = Analysis()
    assert asyncio.run(live.pois("map_r1_map")) == [[1e-7, 2e-7], [3e-7, 4e-7]]


# --- the row's metrics ---------------------------------------------------------------------------------------

def test_the_sleepwalking_metric_uses_each_workflows_own_allow_list():
    assert workflow_allow({"checks": [{"check": "no_actuation"}]}) == []
    assert workflow_allow({"checks": [{"check": "no_sleepwalking", "allow": ["*_move_stage"]}]}) == ["*_move_stage"]
    assert workflow_allow({"checks": [{"check": "no_sleepwalking"}]}) is None  # the mapping skill's tools


class FakeLive:
    def __init__(self, healthy=True):
        from asyncroscopy.agents.config import AgentConfig

        self.config = AgentConfig(model="fake")
        self.risks = {"DigitalTwin_move_stage": "high"}
        self.healthy, self.restored = healthy, []

    async def setup(self, name):
        return {}

    async def restore(self, state):
        self.restored.append(state)

    async def snapshot(self):
        return {"stage": [0, 0, 0, 0, 0]}

    async def tidy(self, events):
        pass

    async def pois(self, key):
        return []

    async def health(self):
        return "" if self.healthy else "DigitalTwin_get_stage did not answer: API_DeviceTimedOut"


def run_with(monkeypatch, chat, wf):
    monkeypatch.setattr(run_workflows, "TRANSCRIPTS", Path(tempfile.mkdtemp(prefix="transcripts-")))
    monkeypatch.setattr(run_workflows, "run_chat_graph", chat)
    monkeypatch.setattr(run_workflows, "read_trace", lambda since, callers=None: [])
    real_sleep = asyncio.sleep
    monkeypatch.setattr(run_workflows.asyncio, "sleep", lambda s: real_sleep(0))
    base = {"name": "w", "timeout_s": 30, "runners": ["general"], "request": "Move the stage.", "checks": []}
    return asyncio.run(run_workflows.run_once(FakeLive(), {**base, **wf}, "general", "unused:make_graph", {}))


def test_a_move_the_workflow_allows_is_not_counted_as_sleepwalking(monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage

    async def chat(factory, config, wf, facts, caller, started):
        messages = [AIMessage("", tool_calls=[{"name": "DigitalTwin_move_stage", "args": {}, "id": "1"}]), ToolMessage("ok", tool_call_id="1"), AIMessage("Moved.")]
        return messages, "Moved.", []

    allowed = run_with(monkeypatch, chat, {"checks": [{"check": "no_sleepwalking", "allow": ["*_move_stage"]}]})
    assert allowed["ok"] and allowed["sleepwalking"] == []
    read_only = run_with(monkeypatch, chat, {"checks": [{"check": "no_actuation"}]})
    assert read_only["sleepwalking"] == ["DigitalTwin_move_stage"]


def test_an_empty_final_answer_fails(monkeypatch):
    async def chat(factory, config, wf, facts, caller, started):
        return [], "  ", []

    row = run_with(monkeypatch, chat, {})
    assert not row["ok"] and not row["checks"]["answered"]["ok"]


def test_a_timed_out_run_keeps_the_calls_whose_replies_arrived(monkeypatch):
    async def chat(factory, config, wf, facts, caller, started):
        config.ledger.append({"name": "view_data", "args": {"key": "a.h5"}, "result": "image", "error": ""})
        config.ledger.append({"name": "DigitalTwin_get_stage", "args": {}, "result": "[0,0]", "error": ""})
        raise TimeoutError

    row = run_with(monkeypatch, chat, {"checks": [{"check": "called", "tool": "view_data"}]})
    assert row["error"].startswith("timed out") and not row["infra_error"]
    assert row["tool_names"] == ["view_data", "DigitalTwin_get_stage"] and row["checks"]["called"]["ok"]


def test_a_wedged_device_is_re_homed_or_the_row_is_infra(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(run_workflows.asyncio, "sleep", lambda s: real_sleep(0))
    wedged = FakeLive(healthy=False)
    with pytest.raises(run_workflows.InfraError, match="API_DeviceTimedOut"):
        asyncio.run(run_workflows.ensure_healthy(wedged, {"stage": [0, 0, 0, 0, 0]}))
    answers = iter(["busy", ""])
    recovered = FakeLive()
    recovered.health = lambda: _value(next(answers))
    asyncio.run(run_workflows.ensure_healthy(recovered, {"stage": [1, 0, 0, 0, 0]}))
    assert recovered.restored == [{"stage": [1, 0, 0, 0, 0]}]  # re-homed once it answered again
    assert _common.is_infra_error("DevFailed: API_DeviceTimedOut while calling acquire")


async def _value(value):
    return value


def test_a_job_that_finishes_before_it_can_be_aborted_fails_fast(monkeypatch):
    events = [job("Mapper.step"), job("Mapper.step"), job("Mapper.step"), job("Mapper.done")]
    monkeypatch.setattr(run_workflows, "read_trace", lambda since, callers=None: events)
    with pytest.raises(run_workflows.WorkflowError, match="make the plan bigger"):
        asyncio.run(run_workflows.wait_for({"job_steps_at_least": 3, "timeout_s": 5}, 0.0, CALLER))


# --- checks fixed after the live run --------------------------------------------------------------------------

def test_a_stitched_map_of_this_run_must_be_named_in_the_answer():
    done = job("Mapper.done", result=json.dumps({"result_keys": {"map": "map_r7_map"}}))
    assert ok(Run(events=[done], final="Stitched map: map_r7_map."), {"check": "stitched_map_in_answer"}) == {"stitched_map_in_answer": True}
    assert ok(Run(events=[done], final="The map is done."), {"check": "stitched_map_in_answer"}) == {"stitched_map_in_answer": False}
    assert ok(Run(final="map_r1_map"), {"check": "stitched_map_in_answer"}) == {"stitched_map_in_answer": False}  # an old map


def test_regions_scored_and_zoomed_accept_the_map_jobs_find_regions_and_revisit():
    plan = job("Mapper.reserve", arguments={"fov_m": 4e-6})
    score = job("Mapper.step", result=json.dumps({"phase": "score", "ok": True, "regions": 3}))
    revisit = job("Mapper.step", result=json.dumps({"phase": "revisit", "image": "tile_9.h5", "stage_xy_m": [1e-6, 2e-6], "fov_m": 5e-7}))
    run = Run(events=[plan, score, revisit], facts={"pois": [[1.05e-6, 2e-6]]})
    assert ok(run, {"check": "regions_scored"}, {"check": "zoomed_on_region"}) == {"regions_scored": True, "zoomed_on_region": True}
    assert ok(Run(), {"check": "regions_scored"}) == {"regions_scored": False}


def test_counting_tiles_changes_nothing():
    read = [Call("view_data", {"key": "map_r1.h5"}, "36 tiles")]
    assert ok(Run(calls=read, final="36"), {"check": "no_writes"}) == {"no_writes": True}
    assert ok(Run(calls=[*read, Call("assemble_map", {"run": "map_r1.h5"}, "map_r1_map")]), {"check": "no_writes"}) == {"no_writes": False}


def test_the_best_regions_are_those_on_the_map_the_request_named():
    from benchmarks.workflow_checks import _top_scored

    def scored(x, score):
        return {"x_m": x, "y_m": 0.0, "source": "scorer:baseline", "score": score}

    run = Run(facts={"setup_map": "map_a_map", "poi_records": {"map_a_map": [scored(1, 0.5), scored(2, 0.4)], "map_new_map": [scored(3, 0.9)]}})
    assert [p["x_m"] for p in _top_scored(run, 2)] == [1, 2]
