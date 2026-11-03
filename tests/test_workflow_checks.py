"""The workflow benchmark's checks, on recorded-style traces (no stack, no model)."""

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from benchmarks import run_workflows
from benchmarks import workflow_checks as check_module
from benchmarks.workflow_checks import CHECKS, Call, Run, evaluate

WORKFLOWS = Path(run_workflows.__file__).parent / "workflows"
CALLER = "wf-test"


def mcp(action, ok=True, ts=1.0, **fields):
    return {"source": "mcp", "action": action, "ok": ok, "ts": ts, "caller": CALLER, **fields}


def job(action, run_id="r1", ts=1.0, ok=True, caller=CALLER, **fields):
    return {"source": "job", "action": action, "ok": ok, "ts": ts, "run_id": run_id, "caller": caller, **fields}


def approval(action, request_id="q1", ts=1.0, **fields):
    return {"source": "approvals", "action": f"Approvals.{action}", "ok": True, "ts": ts, "request_id": request_id, **fields}


def results(run, *specs):
    return evaluate(run, list(specs))


def test_every_shipped_workflow_loads_and_names_known_checks():
    workflows = run_workflows.load_workflows([WORKFLOWS])
    assert len(workflows) >= 25
    assert {wf["stack"] for wf in workflows} == {"twin", "large", "xrd"}
    for wf in workflows:
        assert all(spec["check"] in CHECKS for spec in wf["checks"]), wf["name"]


@pytest.mark.parametrize("text, message", [
    ("workflows: [{name: a, request: x, checks: [{check: called, tool: x}], colour: red}]", "unknown keys"),
    ("workflows: [{name: a, checks: [{check: called, tool: x}]}]", "either request or turns"),
    ("workflows: [{name: a, request: x, checks: [{check: nonsense}]}]", "unknown check"),
    ("workflows: [{name: a, request: x, setup: [teleport], checks: [{check: called, tool: x}]}]", "setup must be"),
    ("workflows: [{name: a, request: x}]", "no checks"),
    ("workflows: [{name: a, request: x, checks: [{check: called, tool: x}]}, {name: a, request: y, checks: [{check: called, tool: x}]}]", "used twice"),
])
def test_workflow_files_are_checked_for_typos(tmp_path, text, message):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(run_workflows.WorkflowError, match=message):
        run_workflows.load_workflows([path])


def test_placeholders_keep_the_type_of_a_whole_fact():
    facts = {"poi_x": 1.5e-7, "map_key": "map_abc.h5_map"}
    assert run_workflows.render("{poi_x}", facts) == 1.5e-7
    assert run_workflows.render("map {map_key} at {poi_x}", facts) == "map map_abc.h5_map at 1.5e-07"
    assert run_workflows.render([{"x": "{poi_x}"}], facts) == [{"x": 1.5e-7}]
    with pytest.raises(run_workflows.InfraError):  # a missing fact is a broken setup, not "{unknown}" in a request
        run_workflows.render("{unknown}", facts)


def test_single_turn_runners_are_not_offered_conversations():
    available = {"general": "", "mapping": "", "team": "", "goal": ""}
    assert run_workflows.runners_for({"runners": ["general", "mapping"], "request": "x"}, available) == ["general", "mapping", "team", "goal"]
    assert run_workflows.runners_for({"runners": ["general"], "turns": [{"say": "x"}]}, available) == ["general"]
    assert run_workflows.runners_for({"runners": ["general"]}, {"general": ""}) == ["general"]


def test_device_calls_come_from_the_trace_and_native_calls_from_the_agent():
    run = Run(
        calls=[Call("view_data", {"key": "img.h5"}, result="image, shape [64, 64]"), Call("view_data", {"key": "nope"}, result="Error: no key 'nope'", error=True)],
        events=[mcp("DigitalTwin_acquire_scanned_image", result_key="img.h5"), mcp("DigitalTwin_move_stage", ok=False, error="out of range")],
    )
    got = results(
        run,
        {"check": "called", "tool": "*_acquire_scanned_image"},
        {"check": "called", "tool": "view_data"},
        {"check": "not_called", "tool": "*_move_stage"},
        {"check": "refused", "tool": "*_move_stage"},
        {"check": "viewed_own_data"},
    )
    assert all(c["ok"] for c in got.values()), got


def test_native_checks_do_not_apply_to_the_fixed_mapping_graph():
    run = Run(has_calls=False, events=[mcp("Mapper_start")])
    got = results(run, {"check": "called", "tool": "view_data"}, {"check": "skill_loaded", "name": "mapping"}, {"check": "called", "tool": "Mapper*_start"})
    assert all(c["ok"] for c in got.values())
    assert got["called[0]"]["detail"] == "n/a"


def test_sleepwalking_counts_unrequested_high_risk_calls_even_when_held():
    run = Run(events=[mcp("Mapper_reserve"), mcp("Mapper_start"), mcp("DigitalTwin_move_stage", ok=False, error="approval_required"), mcp("DigitalTwin_get_stage")])
    got = results(run, {"check": "no_sleepwalking"}, {"check": "no_sleepwalking", "allow": ["Mapper*", "*_move_stage"]}, {"check": "no_actuation"})
    assert not got["no_sleepwalking[0]"]["ok"] and "DigitalTwin_move_stage" in got["no_sleepwalking[0]"]["detail"]
    assert got["no_sleepwalking[1]"]["ok"]
    assert not got["no_actuation"]["ok"]
    assert results(Run(events=[mcp("DigitalTwin_get_stage")]), {"check": "no_actuation"})["no_actuation"]["ok"]


def test_recovered_needs_a_failure_followed_by_a_success():
    fixed = Run(events=[mcp("DigitalTwin_acquire_scanned_image", ok=False, error="no detector", ts=1), mcp("DigitalTwin_acquire_scanned_image", ts=2)])
    gave_up = Run(events=[mcp("DigitalTwin_acquire_scanned_image", ok=False, error="no detector")])
    lucky = Run(events=[mcp("DigitalTwin_acquire_scanned_image")])
    spec = {"check": "recovered", "tool": "*_acquire_scanned_image"}
    assert results(fixed, spec)["recovered"]["ok"]
    assert not results(gave_up, spec)["recovered"]["ok"]
    assert not results(lucky, spec)["recovered"]["ok"]


def test_instrument_state_checks_use_the_readings_before_and_after():
    run = Run(facts={"before": {"stage": [1e-7, 0, 0, 0, 0], "fov": 2e-7, "defocus": 0.0}, "after": {"stage": [3.1e-7, -1.9e-7, 0, 0, 0], "fov": 3e-7, "defocus": 2e-8}})
    got = results(
        run,
        {"check": "stage_near", "x": 3e-7, "y": -2e-7},
        {"check": "stage_unchanged"},
        {"check": "stage_moved_by", "dx": 2e-7, "dy": -2e-7},
        {"check": "stage_moved_by", "dx": 4e-7, "dy": -2e-7},
        {"check": "param_near", "name": "fov", "value": 3e-7},
        {"check": "param_near", "name": "defocus", "value": 2e-8, "rel": 0, "abs_tol": 1e-9},
        {"check": "param_unchanged", "name": "fov"},
    )
    assert [c["ok"] for c in got.values()] == [True, False, True, False, True, True, False]


def test_map_plan_from_the_job_trace_even_when_the_reserve_event_has_no_caller():
    reserve = job("Mapper.reserve", caller=None, arguments={"size_m": [6e-7, 6e-7], "fov_m": 1.6e-7, "per_tile": ["eds_point"]})
    steps = [job("Mapper.step", ts=2 + i) for i in range(9)]
    run = Run(events=steps + [job("Mapper.done", ts=20)], facts={"before": {"stage": [0, 0, 0, 0, 0]}, "all_job_events": [reserve, *steps, job("Mapper.done", ts=20)]})
    got = results(
        run,
        {"check": "map_plan", "size_m": 6e-7, "eds": True},
        {"check": "map_plan", "size_m": 6e-7, "eds": False},
        {"check": "map_plan", "size_m": 2e-6},
        {"check": "job_finished"},
        {"check": "tiles_at_least", "n": 9},
    )
    assert [c["ok"] for c in got.values()] == [True, False, False, True, True]


def test_map_plan_from_a_run_job_call():
    run = Run(calls=[Call("run_job", {"device": "Mapper", "plan": {"box_m": [-5e-7, 5e-7, -5e-7, 5e-7], "fov_m": 2e-7}}, result='{"job_id": "j1", "state": "running"}')], facts={"before": {"stage": [0, 0, 0, 0, 0]}})
    assert run.calls[0].ok and not run.calls[0].error
    assert results(run, {"check": "map_plan", "size_m": 1e-6, "eds": False})["map_plan"]["ok"]


def test_resumed_after_abort_needs_the_same_run_to_finish():
    events = [job("Mapper.step", ts=1), job("Mapper.aborted", ts=2, caller=None), job("Mapper.step", ts=3), job("Mapper.done", ts=4, caller=None)]
    run = Run(events=events, facts={"all_job_events": events})
    assert results(run, {"check": "resumed_after_abort"})["resumed_after_abort"]["ok"]
    abandoned = events[:2]
    assert not results(Run(events=abandoned, facts={"all_job_events": abandoned}), {"check": "resumed_after_abort"})["resumed_after_abort"]["ok"]


def test_zoom_counts_a_smaller_image_near_a_point_of_interest():
    plan = job("Mapper.reserve", caller=None, arguments={"size_m": [6e-7, 6e-7], "fov_m": 1.6e-7})
    events = [
        job("Mapper.step", ts=1), job("Mapper.done", ts=2),
        mcp("DigitalTwin_move_stage", ts=3, arguments={"position": [2.0e-7, 1.0e-7, 0, 0, 0]}),
        mcp("DigitalTwin_set_fov", ts=4, arguments={"fov": 5e-8}),
        mcp("DigitalTwin_acquire_scanned_image", ts=5, result_key="zoom.h5"),
    ]
    facts = {"pois": [[2.1e-7, 0.9e-7]], "all_job_events": [plan, *events[:2]]}
    assert results(Run(events=events, facts=facts), {"check": "zoomed_on_region"})["zoomed_on_region"]["ok"]
    far = {**facts, "pois": [[-3e-7, -3e-7]]}
    assert not results(Run(events=events, facts=far), {"check": "zoomed_on_region"})["zoomed_on_region"]["ok"]
    same_fov = [*events[:3], mcp("DigitalTwin_acquire_scanned_image", ts=5)]
    assert not results(Run(events=same_fov, facts=facts), {"check": "zoomed_on_region"})["zoomed_on_region"]["ok"]
    aimed = [*events[:2], mcp("DigitalTwin_aim_image_shift", ts=3, arguments={"position": [2.0e-7, 1.0e-7]}), *events[3:]]
    assert results(Run(events=aimed, facts=facts), {"check": "zoomed_on_region"})["zoomed_on_region"]["ok"]
    aimed_same_fov = [*aimed[:3], mcp("DigitalTwin_acquire_scanned_image", ts=5)]
    assert not results(Run(events=aimed_same_fov, facts=facts), {"check": "zoomed_on_region"})["zoomed_on_region"]["ok"]


def test_approval_checks_follow_the_request_through_the_decision():
    approved = Run(events=[
        mcp("DigitalTwin_move_stage", ok=False, ts=1, error="approval_required", request_id="q1"),
        approval("request", ts=1, caller=CALLER, arguments={"action": "DigitalTwin_move_stage"}),
        approval("approve", ts=2, caller="operator"),
        mcp("DigitalTwin_move_stage", ts=3),
    ])
    assert all(c["ok"] for c in results(approved, {"check": "approval_requested"}, {"check": "approved_call_ran_once"}).values())
    twice = Run(events=[*approved.events, mcp("DigitalTwin_move_stage", ts=4)])
    assert not results(twice, {"check": "approved_call_ran_once"})["approved_call_ran_once"]["ok"]

    denied = Run(events=[approval("request", ts=1, caller=CALLER, arguments={"action": "DigitalTwin_move_stage"}), approval("deny", ts=2, caller="operator")])
    assert results(denied, {"check": "denied_not_retried"})["denied_not_retried"]["ok"]
    retried = Run(events=[*denied.events, approval("request", request_id="q2", ts=3, caller=CALLER, arguments={"action": "DigitalTwin_move_stage"})])
    assert not results(retried, {"check": "denied_not_retried"})["denied_not_retried"]["ok"]
    unmatched = Run(events=[approval("deny", ts=2, caller="operator")])
    assert not results(unmatched, {"check": "denied_not_retried"})["denied_not_retried"]["ok"]


def test_a_call_the_trace_missed_still_fails_not_called():
    untraced = Run(calls=[Call("DigitalTwin_move_stage", {"x": 1e-7}, result="moved")])
    assert not results(untraced, {"check": "not_called", "tool": "*_move_stage"})["not_called"]["ok"]
    refused = Run(calls=[Call("DigitalTwin_move_stage", {"x": 1e-7}, result="Error: limits", error=True)])
    assert results(refused, {"check": "not_called", "tool": "*_move_stage"})["not_called"]["ok"]


def test_a_broken_trace_line_is_an_error_not_an_empty_trace(tmp_path):
    from benchmarks._common import read_trace

    (tmp_path / "mcp_agent.jsonl").write_text('{"ts": 1, "action": "a"}\nnot json\n{"ts": 2, "action": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_trace(0, directory=tmp_path)
    (tmp_path / "mcp_agent.jsonl").write_text('{"ts": 1, "action": "a"}\n{"ts": 2, "act', encoding="utf-8")
    assert [e["action"] for e in read_trace(0, directory=tmp_path)] == ["a"]


def test_eds_answers_must_agree_with_the_peaks_in_the_data():
    with_peaks = {"truth": {"spectra": {"s.h5": {"peaks_ev": [9713.0], "elements": ["Au"]}}}}
    assert results(Run(final="It shows gold (Au).", facts=with_peaks), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    assert not results(Run(final="It shows Fe.", facts=with_peaks), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    no_peaks = {"truth": {"spectra": {"s.h5": {"peaks_ev": [], "elements": []}}}}
    assert results(Run(final="No peaks were detected above the noise; a longer exposure is needed.", facts=no_peaks), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    assert not results(Run(final="It is mostly Au and Pt.", facts=no_peaks), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    assert not results(Run(final="Au.", facts={}), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    listed_absent = "Mostly gold (Au); other elements include Se, Ir and La, with weaker contributions such as **Zr** and Mn L3-M1."
    assert not results(Run(final=listed_absent, facts=with_peaks), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    ambiguous = {"truth": {"spectra": {"s.h5": {"elements": ["Pd"], "could_be": ["Cd", "Pd"]}}}}
    assert results(Run(final="Palladium (Pd), possibly cadmium.", facts=ambiguous), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    unidentified = {"truth": {"spectra": {"s.h5": {"peaks_ev": [308.0], "elements": [], "could_be": []}}}}
    assert results(Run(final="One peak at 308 eV, but it could not be identified.", facts=unidentified), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    assert not results(Run(final="A weak calcium (Ca) peak at 308 eV.", facts=unidentified), {"check": "answer_eds_consistent"})["answer_eds_consistent"]["ok"]
    assert check_module.elements_claimed("Lead the beam over gold; tin is absent, Pb (Pb) is not") == {"Au", "Pb"}


def test_numbers_in_the_answer_are_compared_with_the_truth():
    run = Run(final="The strongest peak is at 28.4 degrees 2-theta.", facts={"truth": {"strongest_peak": 28.44}})
    assert results(run, {"check": "answer_number_near", "fact": "strongest_peak", "tol": 0.3})["answer_number_near"]["ok"]
    assert not results(run, {"check": "answer_number_near", "fact": "strongest_peak", "tol": 0.01})["answer_number_near"]["ok"]


def test_broken_or_unknown_checks_fail_with_the_reason():
    got = results(Run(), {"check": "nonsense"}, {"check": "stage_near"})
    assert "unknown check" in got["nonsense"]["detail"]
    assert not got["stage_near"]["ok"] and "TypeError" in got["stage_near"]["detail"]


def test_produced_keys_include_map_step_spectra():
    step = job("Mapper.step", result=json.dumps({"index": 0, "key": "tile.h5", "spectrum_key": "spec.h5"}))
    run = Run(events=[mcp("DigitalTwin_acquire_spectrum", result_key="eds.h5"), step, job("Mapper.step", result=json.dumps({"spectrum_key": "eds_map"}))])
    assert run_workflows.produced_keys(run) == ["eds.h5", "spec.h5"]


def test_markdown_summary_lists_failed_checks():
    summary = run_workflows.markdown({
        "model": "m", "stack": "twin", "git": "abc", "passed": 1, "of": 2, "sleepwalking_calls": 0,
        "workflows": {"map_300nm/general": {"passed": 1, "of": 2, "rows": [
            {"seconds": 10, "checks": {"job_finished": {"ok": True}}},
            {"seconds": 30, "checks": {"job_finished": {"ok": False}}, "error": None},
        ]}},
    })
    assert "| map_300nm | general | 1/2 | 30 | job_finished |" in summary


def test_run_once_wires_setup_approval_trace_and_checks_together(monkeypatch):
    """The runner end to end with a stand-in stack and agent: facts fill the request, the operator
    runs only in approval workflows, jobs left behind are tidied, and checks see the trace."""
    import asyncio

    from langchain_core.messages import AIMessage, ToolMessage

    from asyncroscopy.agents.config import AgentConfig

    seen = {}

    class FakeLive:
        def __init__(self):
            self.config = AgentConfig(model="fake")
            self.risks = None
            self.modes, self.tidied, self.readings = [], [], iter([{"stage": [0, 0, 0, 0, 0]}, {"stage": [2e-7, 1e-7, 0, 0, 0]}])

        async def setup(self, name):
            return {"poi_x": 2e-7, "poi_y": 1e-7, "poi_x_nm": 200.0, "map_key": "map_r1.h5_map"}

        async def restore(self, state):
            seen["restored"] = state

        async def snapshot(self):
            return next(self.readings)

        async def set_mode(self, mode):
            self.modes.append(mode)
            return "auto"

        async def operate(self, caller, decision, stop):
            seen["operator"] = (caller, decision)
            await stop.wait()

        async def tidy(self, events):
            self.tidied.append(events)

        async def pois(self, key):
            return []

    async def fake_chat(factory, config, wf, facts, caller, started):
        seen["request"] = wf["request"].format(**facts)
        seen["caller"] = caller
        messages = [AIMessage("", tool_calls=[{"name": "view_data", "args": {"key": "map_r1.h5_map"}, "id": "1"}]), ToolMessage("ok", tool_call_id="1"), AIMessage("Done, imaged at 200 nm.")]
        return messages, "Done, imaged at 200 nm.", ["mapping"]

    def fake_trace(since, callers=None):
        events = [mcp("DigitalTwin_move_stage", ts=since + 1, caller=seen["caller"]), job("Mapper.step", ts=since + 2, caller=seen["caller"])]
        return events if callers is None else [e for e in events if e["caller"] in callers]

    monkeypatch.setattr(run_workflows, "run_chat_graph", fake_chat)
    monkeypatch.setattr(run_workflows, "read_trace", fake_trace)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(run_workflows.asyncio, "sleep", lambda s: real_sleep(0))
    live = FakeLive()
    wf = {
        "name": "revisit", "setup": ["map"], "approval": "approve", "timeout_s": 30, "runners": ["general"],
        "request": "Go to x = {poi_x_nm} nm on {map_key}.",
        "checks": [{"check": "stage_near", "x": "{poi_x}", "y": "{poi_y}"}, {"check": "called", "tool": "*_move_stage"}, {"check": "called", "tool": "view_data"},
                   {"check": "no_sleepwalking", "allow": ["*_move_stage", "view_data"]}],  # the move it asks for is not sleepwalking
    }
    row = asyncio.run(run_workflows.run_once(live, wf, "general", "unused:make_graph", {"stage": [0, 0, 0, 0, 0]}))
    assert row["ok"], row
    assert seen["request"] == "Go to x = 200.0 nm on map_r1.h5_map."
    assert live.modes == ["approve", "auto"] and seen["operator"][1] == "approve"
    assert live.tidied and live.tidied[0][0]["run_id"] == "r1"
    assert row["tool_names"] == ["view_data"] and row["sleepwalking"] == []
    assert row["skills"] == ["mapping"]


def step(index, xy, key=None, spectrum=None, run_id="r1", ts=1.0):
    result = {"index": index, "stage_reported_m": [*xy, 0, 0, 0], "key": key or f"tile_{index}.h5"}
    if spectrum:
        result["spectrum_key"] = spectrum
    return job("Mapper.step", run_id=run_id, ts=ts, result=json.dumps(result))


def poi(x, y, kind="candidate", source="scorer:baseline", score=None, version="1.0.0", reason="bright", links=()):
    return {"x_m": x, "y_m": y, "kind": kind, "source": source, "score": score, "scorer_version": version, "reason": reason, "links": list(links)}


def test_cuj2_every_spectrum_is_a_linked_point_of_interest_where_it_was_taken():
    events = [step(0, (0.0, 0.0), spectrum="s0.h5"), step(1, (1.6e-7, 0.0), spectrum="s1.h5")]
    good = {"map_r1_map": [poi(0.0, 0.0, "eds_point", "agent", links=["s0.h5"]), poi(1.6e-7, 1e-9, "eds_point", "agent", links=["s1.h5"])]}
    assert results(Run(events=events, facts={"poi_records": good}), {"check": "eds_pois_linked"})["eds_pois_linked"]["ok"]
    one_missing = {"map_r1_map": good["map_r1_map"][:1]}
    got = results(Run(events=events, facts={"poi_records": one_missing}), {"check": "eds_pois_linked"})["eds_pois_linked"]
    assert not got["ok"] and "s1.h5" in got["detail"]
    misplaced = {"map_r1_map": [good["map_r1_map"][0], poi(5e-7, 5e-7, "eds_point", "agent", links=["s1.h5"])]}
    assert not results(Run(events=events, facts={"poi_records": misplaced}), {"check": "eds_pois_linked"})["eds_pois_linked"]["ok"]
    assert not results(Run(events=[step(0, (0, 0))], facts={"poi_records": good}), {"check": "eds_pois_linked"})["eds_pois_linked"]["ok"]


def cuj3_facts(extra_pois=(), box=(-3e-7, 3e-7, -3e-7, 3e-7)):
    scored = [poi(1e-7, 1e-7, score=0.9), poi(-1e-7, 1e-7, score=0.8), poi(0.0, -2e-7, score=0.7), poi(2e-7, -2e-7, score=0.1)]
    return {"poi_records": {"m_map": scored + list(extra_pois)}, "map_box": list(box)}


def test_cuj3_scored_regions_need_score_reason_and_version():
    assert results(Run(facts=cuj3_facts()), {"check": "scored_regions", "n": 3})["scored_regions"]["ok"]
    no_version = {"poi_records": {"m": [poi(0, 0, score=0.5, version=None)] * 3}}
    assert not results(Run(facts=no_version), {"check": "scored_regions", "n": 3})["scored_regions"]["ok"]
    agent_only = {"poi_records": {"m": [poi(0, 0, source="agent", score=None)] * 3}}
    assert not results(Run(facts=agent_only), {"check": "scored_regions", "n": 3})["scored_regions"]["ok"]


def test_cuj3_visits_each_best_region_inside_the_approved_box_and_links_new_images():
    targets = [(1e-7, 1e-7), (-1e-7, 1e-7), (0.0, -2e-7)]
    events = []
    for i, (x, y) in enumerate(targets):
        events += [mcp("DigitalTwin_move_stage", ts=2 * i, arguments={"position": [x, y, 0, 0, 0]}), mcp("DigitalTwin_acquire_scanned_image", ts=2 * i + 1, result_key=f"hm{i}.h5")]
    linked = [poi(x, y, "high_mag", "agent", links=[f"hm{i}.h5"]) for i, (x, y) in enumerate(targets)]
    run = Run(events=events, facts=cuj3_facts(linked))
    got = results(run, {"check": "visited_regions", "n": 3}, {"check": "images_linked_to_regions", "n": 3})
    assert all(c["ok"] for c in got.values()), got

    stray = Run(events=[*events, mcp("DigitalTwin_move_stage", ts=9, arguments={"position": [1e-6, 0, 0, 0, 0]})], facts=cuj3_facts(linked))
    assert "outside the region: 1" in results(stray, {"check": "visited_regions", "n": 3})["visited_regions"]["detail"]
    skipped = Run(events=events[:4], facts=cuj3_facts(linked[:2]))
    got = results(skipped, {"check": "visited_regions", "n": 3}, {"check": "images_linked_to_regions", "n": 3})
    assert not got["visited_regions"]["ok"] and not got["images_linked_to_regions"]["ok"]
    old_image = Run(events=events, facts=cuj3_facts([poi(x, y, "high_mag", "agent", links=["older.h5"]) for x, y in targets]))
    assert not results(old_image, {"check": "images_linked_to_regions", "n": 3})["images_linked_to_regions"]["ok"]


def test_zoom_navigation_needs_an_overview_then_level_0_at_a_point_without_walking_tiles():
    facts = {"map_box": [0.0, 6e-6, 0.0, 6e-6], "poi_records": {"m": [poi(3e-6, 3e-6, score=1.0)]}}
    overview = Call("get_region", {"map": "m", "box_m": [0.0, 6e-6, 0.0, 6e-6]}, result="png")
    detail = Call("get_region", {"map": "m", "box_m": [2.8e-6, 3.2e-6, 2.8e-6, 3.2e-6], "level": 0}, result="png")
    assert results(Run(calls=[overview, detail], facts=facts), {"check": "zoom_navigation"})["zoom_navigation"]["ok"]
    assert not results(Run(calls=[detail, overview], facts=facts), {"check": "zoom_navigation"})["zoom_navigation"]["ok"]
    elsewhere = Call("get_region", {"map": "m", "box_m": [0.0, 4e-7, 0.0, 4e-7], "level": 0}, result="png")
    assert not results(Run(calls=[overview, elsewhere], facts=facts), {"check": "zoom_navigation"})["zoom_navigation"]["ok"]
    walking = [overview, detail] + [Call("view_data", {"key": "run.h5", "dataset": f"tiles/tile_{i:05d}"}, result="tile") for i in range(5)]
    assert not results(Run(calls=walking, facts=facts), {"check": "zoom_navigation"})["zoom_navigation"]["ok"]


def test_stitch_truth_measures_placement_against_where_the_twin_really_imaged():
    rng = np.random.default_rng(0)
    pixel, side, n = 1e-9, 128, 9
    true = rng.uniform(0, 3e-7, size=(n, 2))
    reported = true + rng.normal(0, 3e-8, size=(n, 2))  # the stage readout is off by ~30 px
    positions = np.stack([true[:, 1], true[:, 0]], axis=1) / pixel - side / 2 + rng.normal(0, 0.3, size=(n, 2)) + 7.0  # a common offset is allowed
    truth = {"acquisitions": [{"kind": "image", "center_true_m": list(t), "reported_m": [*r, 0, 0, 0]} for t, r in zip(true, reported)]}
    steps = [{"index": i, "stage_reported_m": [*reported[i], 0, 0, 0]} for i in range(n)]
    got = run_workflows.stitch_truth({"tile_positions_px": positions.tolist(), "pixel_size_m": pixel}, (side, side), steps, truth)
    assert got["tiles"] == n and got["median_px"] < 1.0 and got["stage_median_px"] > 10
    assert results(Run(facts={"truth": {"stitch": {"m": got}}}), {"check": "stitch_accuracy"})["stitch_accuracy"]["ok"]
    bad = {**got, "median_px": 3.2}
    assert not results(Run(facts={"truth": {"stitch": {"m": bad}}}), {"check": "stitch_accuracy"})["stitch_accuracy"]["ok"]
    assert not results(Run(), {"check": "stitch_accuracy"})["stitch_accuracy"]["ok"]
    assert results(Run(), {"check": "stitch_accuracy", "required": False})["stitch_accuracy"]["ok"]
    assert "matched only" in run_workflows.stitch_truth({"tile_positions_px": positions.tolist(), "pixel_size_m": pixel}, (side, side), steps, {"acquisitions": []})["error"]


def test_run_keys_come_from_finished_jobs():
    events = [job("Mapper.done", result={"run": "map_r1.h5", "run_file": "x"}), job("Mapper.done", result=json.dumps({"run": "map_r2.h5"})), job("Mapper.aborted", result={"run": "map_r3.h5"})]
    assert run_workflows.run_keys(events) == ["map_r1.h5", "map_r2.h5"]


RAMAN = Path(__file__).parent / "data" / "raman" / "workflows.yaml"


def test_a_team_declares_its_own_stack_with_its_workflows():
    stacks = run_workflows.load_stacks([RAMAN])
    assert {"twin", "large", "xrd", "raman"} <= set(stacks)
    assert stacks["raman"]["tiled"] == "http://127.0.0.1:9191" and stacks["raman"]["operator"] == "http://127.0.0.1:8001/mcp"
    [wf] = run_workflows.load_workflows([RAMAN])
    assert wf["stack"] == "raman"


def test_a_check_with_a_wrong_parameter_fails_when_the_file_is_loaded(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("workflows:\n  - name: x\n    request: hi\n    checks:\n      - {check: answer_number_near, fact: tiles, tolerance: 1}\n", encoding="utf-8")
    with pytest.raises(run_workflows.WorkflowError, match=r"answer_number_near.*its parameters: \['fact', 'tol'\]"):
        run_workflows.load_workflows([bad])
    unknown_stack = tmp_path / "stack.yaml"
    unknown_stack.write_text("workflows:\n  - name: y\n    stack: afm\n    request: hi\n    checks:\n      - {check: answer_matches, pattern: ok}\n", encoding="utf-8")
    with pytest.raises(run_workflows.WorkflowError, match="stack 'afm' is not declared"):
        run_workflows.load_workflows([unknown_stack])


def test_answer_peak_near_needs_the_peak_in_the_data_and_in_the_answer():
    truth = {"spectra": {"raman_1.h5": {"peaks": [2688.5, 1582.0, 520.0], "units": "cm-1"}}}
    read = [Call("view_data", {"key": "raman_1.h5", "format": "text"}, result="peaks: 2688.5, 1582, 520")]
    run = Run(calls=read, facts={"truth": truth}, final="The 2D band is at 2689 cm-1.")
    assert CHECKS["answer_peak_near"](run, position=2690, tol=20)[0]
    assert not CHECKS["answer_peak_near"](Run(calls=read, facts={"truth": truth}, final="It is at 1582 cm-1."), position=2690, tol=20)[0]
    assert not CHECKS["answer_peak_near"](Run(calls=read, facts={"truth": truth}, final="3000"), position=3000, tol=20)[0]  # not in the data
    parroted = CHECKS["answer_peak_near"](Run(facts={"truth": truth}, final="The 2D band is near 2690 cm-1."), position=2690, tol=20)
    assert not parroted[0] and "never read" in parroted[1]  # the skill's own text says "2D near 2690"


def test_skill_loaded_counts_skills_given_automatically():
    assert CHECKS["skill_loaded"](Run(skills=["raman_point_spectrum"]), name="raman_point_spectrum")[0]
    assert not CHECKS["skill_loaded"](Run(skills=["mapping"]), name="raman_point_spectrum")[0]


def test_run_jobs_gated_start_is_allowed_with_run_job_and_overrides_count():
    from benchmarks._common import sleepwalking

    via = [mcp("Mapper_reserve", via="run_job"), mcp("Mapper_start", via="run_job")]  # what one run_job leaves in the trace
    assert CHECKS["no_sleepwalking"](Run(events=via))[0]  # the mapping skill allows run_job, so its own start
    assert not CHECKS["no_sleepwalking"](Run(events=via), allow=["view_data"])[0]
    direct = [mcp("Mapper_reserve"), mcp("Mapper_start")]  # a model calling the envelope itself
    assert not CHECKS["no_sleepwalking"](Run(events=direct))[0]
    assert CHECKS["no_sleepwalking"](Run(events=direct, has_calls=False))[0]  # the fixed mapping graph runs its own job
    assert sleepwalking(["DigitalTwin_set_fov"], []) == ["DigitalTwin_set_fov"]
    assert sleepwalking(["DigitalTwin_set_fov"], [], {"DigitalTwin_set_fov": "low"}) == []  # the endpoint's own table wins
    assert sleepwalking(["Opentrons_get_tip"], [], {"Opentrons_get_tip": "high"}) == ["Opentrons_get_tip"]  # an override to high


def test_the_truth_peak_is_the_one_view_data_lists_first():
    from asyncroscopy.data import views

    x = np.arange(2048, dtype=float)
    y = 5 + 400 * np.exp(-0.5 * ((x - 300) / 3) ** 2) + 900 * np.exp(-0.5 * ((x - 1500) / 3) ** 2)
    peaks = sorted(views.find_peaks(np.random.default_rng(1).poisson(y).astype(float), 3.0), key=lambda p: p.net, reverse=True)
    assert abs(peaks[0].index - 1500) <= 2  # the strongest by net counts, as the text view lists it first


def test_a_length_in_nm_answered_in_metres_is_near():
    from benchmarks.workflow_checks import answer_number_near
    run = SimpleNamespace(final="The point is at (2.0540e-07 m, -4.1e-08 m).", facts={"truth": {"poi_x_nm": 205.4}})
    assert answer_number_near(run, fact="poi_x_nm", tol=2.0)[0]
    assert not answer_number_near(SimpleNamespace(final="at 2.0540e-07", facts={"truth": {"tiles": 205.4}}), fact="tiles", tol=2.0)[0]


def _placement(child, parent, status="placed", pixel_m=1e-9):
    record = {"child": child, "parent": parent, "status": status, "child_pixel_m": pixel_m, "reason": "" if status == "placed" else "ambiguous"}
    return Call("place_image", {"image": child, "parent": parent}, result=json.dumps(record))


def test_placed_chain_counts_only_images_placed_on_placed_images():
    chain = Run(calls=[_placement("t2.h5", "survey.h5", pixel_m=3e-10), _placement("t3.h5", "t2.h5", pixel_m=4e-11)])
    passed, detail = CHECKS["placed_chain"](chain, tiers=2, max_pixel_m=1e-10)
    assert passed and "longest placed chain 2" in detail and "40 pm" in detail
    broken = Run(calls=[_placement("t2.h5", "survey.h5", status="unplaced"), _placement("t3.h5", "t2.h5", pixel_m=4e-11)])
    assert not CHECKS["placed_chain"](broken, tiers=2)[0]
    too_coarse = Run(calls=[_placement("t2.h5", "survey.h5", pixel_m=6e-10), _placement("t3.h5", "t2.h5", pixel_m=3e-10)])
    assert not CHECKS["placed_chain"](too_coarse, tiers=2, max_pixel_m=1e-10)[0]
    passed, detail = CHECKS["placed_chain"](Run(calls=[_placement("a.h5", "b.h5", status="unplaced")]), tiers=1)
    assert not passed and "none placed" in detail


def test_placed_chain_counts_the_placements_a_zoom_job_made():
    tier = lambda image, parent, status, pixel: job("Mapper.step", result=json.dumps({"phase": "zoom", "image": image, "parent": parent, "pixel_m": pixel, "placement": {"status": status}}))
    run = Run(events=[tier("t2.h5", "survey.h5", "placed", 1.5e-10), tier("t3.h5", "t2.h5", "unplaced", 1.8e-11), tier("t3.h5", "t2.h5", "unplaced", 1.8e-11)])
    passed, detail = CHECKS["placed_chain"](run, tiers=1)
    assert passed and "1 placed and 1 unplaced" in detail
    assert not CHECKS["placed_chain"](run, tiers=2)[0]



def test_an_image_taken_by_the_agent_or_by_a_revisit_job_counts():
    revisit = job("Mapper.step", result=json.dumps({"phase": "revisit", "region": "point:0", "image": "img.h5", "stage_xy_m": [1e-7, 2e-7]}))
    assert CHECKS["image_taken"](Run(events=[revisit]))[0]
    assert not CHECKS["image_taken"](Run(events=[job("Mapper.step", result=json.dumps({"phase": "revisit", "skipped": "near the edge"}))]))[0]


def test_an_acquisition_reply_that_carries_a_reading_of_its_data_counts_as_viewing_it():
    key = "stem_image_HAADF_20261101T115014206132.h5"
    summary = f"{key}\n64 x 64 pixels, uint16\nstatistics of the pixels read: min 3, max 62"
    read = Run(calls=[Call("DigitalTwin_acquire_scanned_image", {}, summary)], events=[mcp("DigitalTwin_acquire_scanned_image", result=key)])
    assert results(read, {"check": "viewed_own_data"})["viewed_own_data"]["ok"]
    bare = Run(calls=[Call("DigitalTwin_acquire_scanned_image", {}, key)], events=[mcp("DigitalTwin_acquire_scanned_image", result=key)])
    assert not results(bare, {"check": "viewed_own_data"})["viewed_own_data"]["ok"]
