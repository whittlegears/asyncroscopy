"""The measurement workflows (once skill run lists): a spectrum at a point, on a feature, and at a map's best region."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from asyncroscopy.agents.goals import trace_from_messages
from asyncroscopy.agents.workflows import get_workflow
from asyncroscopy.agents.workflows.runner import run_workflow

PEAKS = ("peaks: Au M 2.12 keV\nelements most consistent with these peaks: Au 100%\ndata (sums of 11 channels):\n1,2,3")


class FakeServer:
    def __init__(self, tools: dict) -> None:
        self.tools, self.log = tools, []

    async def list_tools(self):
        return [SimpleNamespace(name=name) for name in self.tools]

    async def call(self, name, arguments=None):
        self.log.append((name, dict(arguments or {})))
        text = self.tools[name](dict(arguments or {}))
        text = text if isinstance(text, str) else json.dumps(text)
        return SimpleNamespace(is_error=text.startswith("Error"), structured_content=None, content=[SimpleNamespace(text=text)])

    def names(self):
        return [name for name, _ in self.log]


def scope(features=({"x_m": 4e-7, "y_m": -2e-7},), pois=({"x_m": 1e-6, "y_m": 2e-6, "score": 0.9, "reason": "bright"},)):
    state = {"stage": [0.0, 0.0, 1e-9, 0.1, 0.0], "fov": 5e-8, "blanked": True, "parked": None}

    def setter(key):
        def set_it(args):
            state[key] = args
            return "ok"
        return set_it

    def blank(value):
        def set_it(args):
            state["blanked"] = value
            return "ok"
        return set_it

    def move(args):
        state["stage"] = list(args["position"])
        return "ok"

    def set_fov(args):
        state["fov"] = float(args["fov"])
        return "ok"

    def park(args):
        state["parked"] = list(args["position"])
        return "ok"

    instrument = FakeServer({
        "Twin_get_stage": lambda a: json.dumps(state["stage"]),
        "Twin_move_stage": move,
        "Twin_get_fov": lambda a: json.dumps(state["fov"]),
        "Twin_set_fov": set_fov,
        "Twin_unblank_beam": blank(False),
        "Twin_blank_beam": blank(True),
        "Twin_acquire_scanned_image": lambda a: "stem_image_HAADF_0001.h5\nimage 256 x 256",
        "Twin_place_beam_at": park,
        "Twin_acquire_spectrum": lambda a: "spectrum_eds_0001.h5\n2048 channels" if not state["blanked"] else "Error: beam blanked",
        "view_data": lambda a: PEAKS,
    })
    analysis = FakeServer({
        "find_features": lambda a: {"features": list(features)} if features else {"features": [], "hint": "nothing distinct"},
        "list_pois": lambda a: {"pois": list(pois)},
    })
    return instrument, analysis, state


def run(name, inputs, instrument, analysis, tmp_path):
    return asyncio.run(run_workflow(get_workflow(name), inputs, instrument=instrument, analysis=analysis, state_dir=tmp_path))


def test_the_measurement_workflows_lint_clean():
    for name in ("spectrum_at_point", "spectrum_on_feature", "spectrum_at_best_region"):
        result = get_workflow(name).lint()
        assert result["ok"], (name, result["problems"])
        assert "*_blank_beam" in result["tools"] and "*_acquire_spectrum" in result["tools"]


def test_a_spectrum_on_a_feature_surveys_finds_measures_reads_and_puts_the_field_back(tmp_path):
    instrument, analysis, state = scope()
    record = run("spectrum_on_feature", {}, instrument, analysis, tmp_path)
    assert record["state"] == "done", record
    assert instrument.names()[:4] == ["Twin_get_fov", "Twin_set_fov", "Twin_unblank_beam", "Twin_acquire_scanned_image"]
    assert analysis.names() == ["find_features"]
    assert state["parked"] == [4e-7, -2e-7] and state["blanked"] and state["fov"] == 5e-8  # the field of view is put back
    outputs = record["outputs"]
    assert outputs["spectrum"]["key"] == "spectrum_eds_0001.h5" and outputs["peaks"]["elements"].endswith("Au 100%")
    assert "data (sums" not in outputs["peaks"]["peaks"]  # the channel table is left out
    assert record["tools_called"]["Twin_acquire_spectrum"]["ok"] == 1


def test_a_survey_with_no_feature_stops_before_the_beam_is_parked_and_leaves_it_blanked(tmp_path):
    instrument, analysis, state = scope(features=())
    record = run("spectrum_on_feature", {"fov_m": 6e-7}, instrument, analysis, tmp_path)
    assert record["state"] == "failed" and "no feature found" in record["error"] and "nothing distinct" in record["error"]
    assert "Twin_place_beam_at" not in instrument.names() and "Twin_acquire_spectrum" not in instrument.names()
    assert state["blanked"]


def test_a_spectrum_at_a_point_needs_the_point_and_moves_there_keeping_z_and_tilts(tmp_path):
    instrument, analysis, state = scope()
    refused = run("spectrum_at_point", {}, instrument, analysis, tmp_path)
    assert refused["state"] == "refused" and instrument.log == []
    record = run("spectrum_at_point", {"x_m": 3e-7, "y_m": -1e-7}, instrument, analysis, tmp_path)
    assert record["state"] == "done", record
    assert state["stage"] == [3e-7, -1e-7, 1e-9, 0.1, 0.0] and state["parked"] == [3e-7, -1e-7] and state["blanked"]
    assert record["outputs"]["peaks"]["elements"].endswith("Au 100%")


def test_a_spectrum_at_the_best_region_moves_to_the_maps_first_region_and_measures_on_the_feature_there(tmp_path):
    instrument, analysis, state = scope()
    record = run("spectrum_at_best_region", {"map": "map_r1_map"}, instrument, analysis, tmp_path)
    assert record["state"] == "done", record
    assert analysis.log[0] == ("list_pois", {"map": "map_r1_map", "kind": "candidate", "limit": 3})
    moved = [args for name, args in instrument.log if name == "Twin_move_stage"]
    assert moved == [{"position": [1e-6, 2e-6, 1e-9, 0.1, 0.0]}] and state["parked"] == [4e-7, -2e-7]


def test_a_map_without_regions_stops_the_run_before_the_stage_moves(tmp_path):
    instrument, analysis, state = scope(pois=())
    record = run("spectrum_at_best_region", {"map": "map_r1_map"}, instrument, analysis, tmp_path)
    assert record["state"] == "failed" and "no scored regions" in record["error"]
    assert "Twin_move_stage" not in instrument.names()


def test_a_workflow_runs_calls_count_as_the_agents_own_in_the_trace(tmp_path):
    instrument, analysis, _ = scope()
    record = run("spectrum_on_feature", {}, instrument, analysis, tmp_path)
    reply = json.dumps({"run_id": record["run_id"], "workflow": "spectrum_on_feature", "state": "done", "tools_called": record["tools_called"]})
    messages = [HumanMessage("find a particle and take a spectrum on it"),
                AIMessage("", tool_calls=[{"name": "run_workflow", "args": {"name": "spectrum_on_feature"}, "id": "c1"}]),
                ToolMessage(reply, tool_call_id="c1", name="run_workflow")]
    calls = {c.name: c for c in trace_from_messages(messages)}
    assert {"run_workflow", "Twin_acquire_spectrum", "Twin_place_beam_at", "view_data"} <= set(calls)
    assert not calls["Twin_acquire_spectrum"].error and calls["Twin_place_beam_at"].args == {"position": [4e-7, -2e-7]}


def test_a_skill_with_a_run_list_is_unavailable_and_says_to_write_a_workflow(tmp_path, monkeypatch):
    from asyncroscopy.skills import list_skills

    folder = tmp_path / "lab" / "old_macro"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: old_macro\ndescription: an old run list\nrun:\n  steps:\n    - {tool: x}\n---\n\n# Old\n",
                                     encoding="utf-8")
    monkeypatch.setenv("ASYNCROSCOPY_SKILLS_PATH", str(tmp_path / "lab"))
    skill = {s.name: s for s in list_skills()}["old_macro"]
    assert not skill.available and "LangGraph workflows" in skill.problem


def test_a_runs_instrument_calls_carry_the_caller_that_asked_for_it(tmp_path, monkeypatch):
    # The instrument's trace, the benchmarks and the evaluators credit the agent that started the run.
    from fastmcp import Client

    from asyncroscopy.agents.workflows import server as workflow_server

    instrument, analysis, _ = scope()
    opened = []

    class Endpoints:
        async def open(self, run_id=None, caller=None):
            opened.append(caller)
            return instrument, analysis, {"simulated": True, "stack_id": "tango://twin:1"}

    monkeypatch.setattr(workflow_server, "requester", lambda: "wf-find_particle-general-1")
    served = workflow_server.WorkflowServer(Endpoints(), state_dir=tmp_path / "state")

    async def go():
        async with Client(served.mcp) as client:
            reply = await client.call_tool("run_workflow", {"name": "spectrum_on_feature", "wait_s": 30})
            return reply.structured_content or json.loads(reply.content[0].text)

    reply = asyncio.run(go())
    assert reply["state"] == "done", reply
    assert opened[-1] == "wf-find_particle-general-1"
    assert reply["tools_called"]["Twin_acquire_spectrum"]["ok"] == 1


def test_a_run_that_fails_after_zooming_puts_the_field_back_and_a_resume_zooms_again(tmp_path):
    from asyncroscopy.agents.workflows.runner import resume_workflow

    instrument, analysis, state = scope()
    good = instrument.tools["Twin_acquire_spectrum"]
    instrument.tools["Twin_acquire_spectrum"] = lambda a: "Error: detector not ready"
    record = asyncio.run(run_workflow(get_workflow("spectrum_on_feature"), {}, instrument=instrument, analysis=analysis,
                                      state_dir=tmp_path, run_id="zoomed"))
    assert record["state"] == "failed" and state["blanked"]
    assert state["fov"] == 5e-8  # put back although fov_back never ran
    assert record["restores"]["fov"]["restore"]["arguments"] == {"fov": 5e-8}
    instrument.tools["Twin_acquire_spectrum"] = good
    instrument.log.clear()
    record = asyncio.run(resume_workflow("zoomed", instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done", record
    assert ("Twin_set_fov", {"fov": 3e-7}) in instrument.log  # set again before the spectrum
    assert instrument.names().index("Twin_set_fov") < instrument.names().index("Twin_acquire_spectrum")
    assert state["fov"] == 5e-8 and record["restores"] is None
