"""Tests for workflows built as graphs of blocks, run against a fake instrument."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from langgraph.graph import END, START, StateGraph

from asyncroscopy.workflows.blocks import Block, register_block
from asyncroscopy.workflows import Workflow, WorkflowState, all_workflows, describe_blocks, get_workflow, register_workflow
from asyncroscopy.workflows.runner import read_run, resume_workflow, run_workflow, stop_workflow
from asyncroscopy.workflows.server import Endpoints
from asyncroscopy.workflows.values import evaluate


class FakeServer:
    """An MCP endpoint in memory. ``tools`` maps a tool name to a function that returns the reply text."""

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


def twin(on_acquire=None):
    """A fake microscope and analysis server. ``on_acquire(n)`` may return a reply for acquisition n."""
    state = {"stage": [1e-6, 2e-6, 0.0, 0.0, 0.0], "shift": [0.0, 0.0], "shift_limit": 1e-6, "fov": 5.8e-9, "blanked": True, "taken": 0,
             "beam_on_at_acquire": []}

    def move(args):
        state["stage"] = list(args["position"])
        return "ok"

    def shift(args):
        state["shift"] = list(args["shift"])
        return "ok"

    def set_fov(args):
        state["fov"] = float(args["fov"])
        return "ok"

    def acquire(args):
        state["beam_on_at_acquire"].append(not state["blanked"])
        if on_acquire:
            reply = on_acquire(state["taken"])
            if reply:
                return reply
        state["taken"] += 1
        return f"stem_image_HAADF_{state['taken']:04d}.h5\nimage 256 x 256"

    def blank(value):
        def set_it(args):
            state["blanked"] = value
            return "ok"
        return set_it

    instrument = FakeServer({
        "Twin_get_stage": lambda a: json.dumps(state["stage"]),
        "Twin_move_stage": move,
        "Twin_get_image_shift": lambda a: json.dumps(state["shift"]),
        "Twin_get_image_shift_limit": lambda a: json.dumps(state["shift_limit"]),
        "Twin_set_image_shift": shift,
        "Twin_set_fov": set_fov,
        "Twin_get_fov": lambda a: json.dumps(state["fov"]),
        "Twin_unblank_beam": blank(False),
        "Twin_blank_beam": blank(True),
        "Twin_acquire_scanned_image": acquire,
    })
    analysis = FakeServer({
        "collect_map_run": lambda a: {"run": "map_abc", "tiles": len(a["image_keys"])},
        "assemble_map": lambda a: {"job_id": "j1", "state": "running"},
        "analysis_job_status": lambda a: {"job_id": "j1", "state": "done"},
        "analysis_job_results": lambda a: {"result_keys": {"map": "map_abc_map_v1"}, "summary": {"tiles": 4}},
    })
    return instrument, analysis, state


INPUTS = {"size_m": 5.0e-7, "fov_m": 3.0e-7}  # a 2 x 2 grid
SHIPPED = {"map_grid", "map_4x4", "map_sample", "spectrum_at_point", "spectrum_on_feature", "spectrum_at_best_region"}


def run(coro):
    return asyncio.run(coro)


# The graph and its checks
def test_the_map_is_a_langgraph_graph_built_from_blocks_and_lints_clean():
    workflow = get_workflow("map_grid")
    assert "map_grid" in all_workflows()
    # The module source is what people see and what gets hashed.
    assert "graph = StateGraph(WorkflowState)" in workflow.source() and "graph.add_conditional_edges(" in workflow.source()
    compiled = workflow.compile(None)
    assert {"plan", "tile", "home", "run", "stitch"} <= set(compiled.get_graph().nodes)
    blueprint = workflow.blueprint()
    assert blueprint["nodes"]["tile"]["block"] == "acquire_tile" and ["START", "plan"] in blueprint["edges"]
    assert {"from": "tile", "when": "tile.remaining > 0", "to": "tile", "otherwise": "home", "max_visits": 2000} in blueprint["edges"]
    result = workflow.lint()
    assert result["ok"], result["problems"]
    assert result["acts"] and "acquire_tile" in result["blocks"] and len(result["hash"]) == 64
    assert "*_move_stage" in result["tools"] and "assemble_map" in result["tools"]


def two_tiles(name: str, summary: str = "Two blocks from the shelf: plan a grid, then take its tiles.", **changes) -> Workflow:
    """A small workflow built from the shipped blocks in plain LangGraph."""
    def build(use):
        graph = StateGraph(WorkflowState)
        graph.add_node("plan", use.block(changes.get("plan_block", "plan_grid"), size_m="{inputs.size_m}", fov_m=3e-7, move_by="stage"))
        graph.add_node("tile", use.block("acquire_tile", **changes.get("tile", {"tiles": "{plan.tiles}", "fov_m": 3e-7, "stage_rest": "{plan.stage_rest}"})))
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "tile")
        bound = changes.get("max_visits", 400)
        graph.add_conditional_edges("tile", use.when(changes.get("when", "tile.remaining > 0"), then="tile", otherwise=END, max_visits=bound))
        return graph
    return Workflow(name=name, summary=summary, inputs={"size_m": {"type": "number", "default": 5e-7, "min": 1e-7, "max": 5e-6}},
                    budget={"max_tool_calls": 200}, build=build)


def test_a_new_workflow_is_more_langgraph_code_over_the_same_blocks(tmp_path):
    workflow = two_tiles("two_tiles")
    assert workflow.lint()["ok"], workflow.lint()["problems"]
    instrument, analysis, state = twin()
    record = run(run_workflow(workflow, {}, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done" and state["taken"] == 4 and state["blanked"], record
    assert record["blueprint"]["nodes"]["plan"]["block"] == "plan_grid"
    assert analysis.log == []  # this workflow has no stitch step
    # Changing a condition changes the hash.
    assert two_tiles("two_tiles").hash() == workflow.hash() != two_tiles("two_tiles", when="tile.remaining > 1").hash()


def test_lint_names_each_problem():
    problems = " | ".join(two_tiles("bad", tile={"tiles": "{plan.nothing}", "fov_m": "wide", "speed": 3}).lint()["problems"])
    assert "no output 'nothing'" in problems
    assert "'wide' is not a number" in problems
    assert "no input(s) ['speed']" in problems
    assert "needs input(s) ['stage_rest']" in problems
    unknown = two_tiles("bad", plan_block="no_such_block").lint()
    assert not unknown["ok"] and "no block 'no_such_block'" in unknown["problems"][0] and unknown["hash"] is None


def test_lint_refuses_an_unbounded_loop_python_in_a_condition_and_a_node_that_is_not_a_block():
    assert any("has no bound" in p for p in two_tiles("bad", max_visits=None).lint()["problems"])
    assert any("is not a condition" in p for p in two_tiles("bad", when="__import__('os').system('x')").lint()["problems"])

    def build(use):
        graph = StateGraph(WorkflowState)

        async def anything(state):
            return {}
        graph.add_node("plan", use.block("plan_grid", size_m=5e-7, fov_m=3e-7))
        graph.add_node("free", anything)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", lambda state: "free", ["free"])
        graph.add_edge("free", END)
        return graph
    problems = " | ".join(Workflow(name="loose", summary="A node and a router written by hand.", build=build).lint()["problems"])
    assert "node free is not a block" in problems and "a conditional edge not made with use.when" in problems


def test_a_node_knows_its_name_from_the_graph_and_langgraph_knows_where_a_condition_leads():
    # One block is made once and added under two names. Each node reads its own name when it runs.
    def build(use):
        graph = StateGraph(WorkflowState)
        read = use.block("run_tool", tool="*_get_stage")
        graph.add_node("first", read)
        graph.add_node("second", read)
        graph.add_edge(START, "first")
        graph.add_conditional_edges("first", use.when("first.result != 0", then="second", otherwise=END))
        graph.add_edge("second", END)
        return graph
    workflow = Workflow(name="twice", summary="Read the stage twice.", build=build)
    assert workflow.lint()["ok"], workflow.lint()["problems"]
    assert {"from": "first", "when": "first.result != 0", "to": "second", "otherwise": "END"} in workflow.blueprint()["edges"]
    drawn = workflow.compile(None).get_graph().draw_mermaid()
    assert "first -.-> second" in drawn and "first -.-> __end__" in drawn  # the targets come from the router, not a list

    def two_conditions(use):
        graph = build(use)
        graph.add_conditional_edges("first", use.when("first.result == 0", then="second", otherwise=END))
        return graph
    problems = " | ".join(Workflow(name="split", summary="Two conditions on one node.", build=two_conditions).lint()["problems"])
    assert "node first has a condition and 1 other edge(s) leaving it" in problems


def test_a_workflow_in_the_old_shape_is_told_the_new_one():
    from asyncroscopy.workflows.authoring import TEMPLATE, check_source

    old = TEMPLATE.replace('use.block("plan_grid", ', 'use.block("plan", "plan_grid", ').replace(
        'use.when("tile.remaining > 0", then="tile", otherwise="home", max_visits=400))',
        'use.when("tile", "tile.remaining > 0", then="tile", otherwise="home", max_visits=400), ["tile", "home"])')
    problems = " | ".join(check_source(old))
    assert 'use.block takes the block\'s name as text' in problems and 'add_conditional_edges("name", use.when("condition"' in problems


def test_lint_reports_what_langgraph_refuses_and_the_shapes_it_lets_through():
    def shaped(*edges):
        def build(use):
            graph = StateGraph(WorkflowState)
            graph.add_node("first", use.block("run_tool", tool="*_get_stage"))
            graph.add_node("second", use.block("run_tool", tool="*_get_fov"))
            for edge in edges:
                graph.add_edge(*edge)
            return graph
        return Workflow(name="shaped", summary="Two reads.", build=build).lint()

    unknown = shaped((START, "first"), ("first", "nowhere"))
    assert unknown["hash"] is None and "unknown node" in unknown["problems"][0]
    assert "entrypoint" in shaped(("first", "second"), ("second", END))["problems"][0]
    assert shaped((START, "first"), ("first", END), ("second", END))["problems"] == ["node second is never reached from START"]
    assert "node second has no edge leaving it" in shaped((START, "first"), ("first", "second"))["problems"][0]


def test_lint_against_the_endpoints_tools():
    instrument, analysis, _ = twin()
    names = [n for n in instrument.tools if n != "Twin_move_stage"]
    problems = get_workflow("map_grid").lint(names, list(analysis.tools))["problems"]
    assert problems == ["the instrument endpoint offers no tool matching *_move_stage, which a block of this workflow calls"]



def test_a_node_switched_off_by_the_runs_inputs_needs_no_tool():
    from asyncroscopy.workflows.checks import switched_off_nodes

    instrument, analysis, _ = twin()
    names = [n for n in instrument.tools if not n.endswith("_acquire_eels_spectrum")]
    workflow = get_workflow("map_sample")
    blueprint = workflow.blueprint()
    missing = "the instrument endpoint offers no tool matching *_acquire_eels_spectrum, which a block of this workflow calls"
    assert missing in workflow.lint(names, list(analysis.tools))["problems"]  # without inputs every node counts
    off = switched_off_nodes(blueprint, workflow.check_inputs({"size_m": 1e-6, "fov_m": 3e-7}))
    assert "eels" in off and "eds" in off
    assert missing not in workflow.lint(names, list(analysis.tools), switched_off=off)["problems"]
    on = switched_off_nodes(blueprint, workflow.check_inputs({"size_m": 1e-6, "fov_m": 3e-7, "eels": True}))
    assert "eels" not in on and missing in workflow.lint(names, list(analysis.tools), switched_off=on)["problems"]


def test_inputs_outside_their_range_are_refused():
    workflow = get_workflow("map_grid")
    assert workflow.check_inputs({"size_m": "4e-7"})["size_m"] == 4e-7
    with pytest.raises(ValueError, match="outside its range"):
        workflow.check_inputs({"size_m": 1.0})
    with pytest.raises(ValueError, match="not \\['dwell'\\]"):
        workflow.check_inputs({"dwell": 1e-6})


def test_conditions():
    assert evaluate("tile.remaining > 0 and not inputs.dry", {"dry": False}, {"tile": {"remaining": 2}}) is True
    assert evaluate("tile.remaining > 0", {}, {"tile": {"remaining": 0}}) is False


def test_blocks_describe_their_ports():
    blocks = {b["name"]: b for b in describe_blocks()}
    assert {"plan_grid", "acquire_tile", "collect_run", "stitch_map", "return_stage"} <= set(blocks)
    assert blocks["acquire_tile"]["outputs"]["remaining"] == "integer"


# Running
def test_map_workflow_runs_tile_by_tile_and_stitches(tmp_path):
    instrument, analysis, state = twin()
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done", record
    assert record["outputs"]["plan"]["count"] == 4
    assert len(record["outputs"]["tile"]["keys"]) == 4 and record["visits"]["tile"] == 4
    assert record["outputs"]["stitch"]["map"] == "map_abc_map_v1"
    assert state["beam_on_at_acquire"] == [True] * 4 and state["blanked"]
    # The run puts back the field of view it found.
    fovs = [args["fov"] for name, args in instrument.log if name == "Twin_set_fov"]
    assert fovs == [3e-7] * 4 + [5.8e-9] and state["fov"] == 5.8e-9
    # With image shift the stage never moves.
    assert "Twin_move_stage" not in instrument.names() and state["stage"][:2] == [1e-6, 2e-6] and state["shift"] == [0.0, 0.0]
    every = [args["shift"] for name, args in instrument.log if name == "Twin_set_image_shift"]
    assert every[1:8:2] == [[0.0, 0.0]] * 4 and every[8] == [0.0, 0.0]  # the shift is reset after every tile
    shifts = every[0:8:2]
    assert shifts[0][0] < shifts[1][0] and shifts[2][0] > shifts[3][0]  # serpentine
    assert max(abs(v) for xy in shifts for v in xy) == pytest.approx(1.05e-7)
    # Each tile keeps its centre so images with no recorded position can still be mapped.
    read_at = record["outputs"]["tile"]["stage_xy_m"]
    assert read_at == [[1e-6 + dx, 2e-6 + dy] for dx, dy in shifts]
    assert analysis.log[0] == ("collect_map_run", {"image_keys": record["outputs"]["tile"]["keys"], "stage_xy_m": read_at, "fov_m": 3e-7,
                                                   "plan": {"workflow": "map_grid", "size_m": 5e-7, "fov_m": 3e-7, "overlap": 0.3}})
    assert "stage_sigma_m" not in analysis.log[1][1]  # 0 means the analysis server default


def test_the_map_can_move_the_stage_instead_and_a_square_beyond_the_beams_reach_is_refused(tmp_path):
    instrument, analysis, state = twin()
    record = run(run_workflow("map_grid", {**INPUTS, "move_by": "stage"}, instrument=instrument, analysis=analysis, state_dir=tmp_path / "a"))
    assert record["state"] == "done", record
    moves = [args["position"][:2] for name, args in instrument.log if name == "Twin_move_stage"]
    assert len(moves) == 5 and moves[0][0] < moves[1][0] and moves[2][0] > moves[3][0] and moves[4] == [1e-6, 2e-6]
    assert "Twin_set_image_shift" not in instrument.names() and record["outputs"]["tile"]["stage_xy_m"] == moves[:4]

    instrument, analysis, state = twin()
    far = run(run_workflow("map_grid", {"size_m": 4e-6, "fov_m": 5e-7}, instrument=instrument, analysis=analysis, state_dir=tmp_path / "b"))
    assert far["state"] == "failed" and "over the limit of 1e-06 m" in far["error"] and "move_by stage" in far["error"]
    assert state["taken"] == 0 and not {"Twin_set_image_shift", "Twin_move_stage"} & set(instrument.names())
    bad = run(run_workflow("map_grid", {**INPUTS, "move_by": "teleport"}, instrument=instrument, analysis=analysis, state_dir=tmp_path / "c"))
    assert bad["state"] == "failed" and "move_by must be one of" in bad["error"]


def test_the_stitch_is_given_the_stage_error_when_the_caller_states_it(tmp_path):
    instrument, analysis, _ = twin()
    record = run(run_workflow("map_grid", {**INPUTS, "stage_sigma_m": 2e-7}, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done", record
    assert analysis.log[1] == ("assemble_map", {"run": "map_abc", "refine": True, "stage_sigma_m": 2e-7})


def test_a_refused_run_makes_no_call(tmp_path):
    instrument, analysis, _ = twin()
    record = run(run_workflow("map_grid", {"size_m": 1.0}, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "refused" and "outside its range" in record["problems"][0]
    assert instrument.log == []


def test_a_failed_tile_blanks_the_beam_and_resumes_at_that_tile(tmp_path):
    failing = {"on": True}
    instrument, analysis, state = twin(lambda n: "Error: detector timeout" if n == 2 and failing["on"] else None)
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "failed" and "detector timeout" in record["error"]
    assert state["blanked"] and record["cleanup"] == [{"tool": "*_blank_beam", "ok": True}]
    assert len(record["outputs"]["tile"]["keys"]) == 2 and record["at_node"] == ["tile"]
    # A failed run leaves the instrument as it found it.
    assert state["shift"] == [0.0, 0.0] and state["fov"] == 5.8e-9

    failing["on"] = False
    before = len(instrument.log)
    record = run(resume_workflow(record["run_id"], instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done" and len(record["outputs"]["tile"]["keys"]) == 4
    assert state["taken"] == 4  # the two finished tiles were not taken again
    assert [a["fov"] for n, a in instrument.log[before:] if n == "Twin_set_fov"] == [3e-7, 3e-7, 5.8e-9]  # set again, then put back
    assert state["shift"] == [0.0, 0.0] and state["fov"] == 5.8e-9
    assert read_run(record["run_id"], tmp_path)["state"] == "done"


def test_a_scan_running_at_the_console_is_named_and_the_run_puts_everything_back(tmp_path):
    busy = {"on": True}
    reply = "Error calling tool 'Twin_acquire_scanned_image': ApplicationServerException: Another scantask is already active"
    instrument, analysis, state = twin(lambda n: reply if busy["on"] else None)
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    # The run waits for a person to stop the live scan, with everything it changed put back meanwhile.
    assert record["state"] == "waiting" and "live scan" in record["questions"][0]["question"], record
    assert state["taken"] == 0 and state["blanked"] and state["shift"] == [0.0, 0.0] and state["fov"] == 5.8e-9
    busy["on"] = False
    record = run(resume_workflow(record["run_id"], instrument=instrument, analysis=analysis, state_dir=tmp_path,
                                 answers={record["questions"][0]["id"]: "continue"}))
    assert record["state"] == "done" and state["taken"] == 4 and state["shift"] == [0.0, 0.0] and state["fov"] == 5.8e-9


def test_stop_is_seen_before_the_next_call_and_the_run_resumes(tmp_path):
    instrument, analysis, state = twin()

    def stop_after_first(n):
        if n == 1:
            stop_workflow("stopme", tmp_path)
        return None

    instrument.tools["Twin_acquire_scanned_image"] = (lambda inner: lambda a: (stop_after_first(state["taken"]), inner(a))[1])(instrument.tools["Twin_acquire_scanned_image"])
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path, run_id="stopme"))
    assert record["state"] == "stopped" and state["blanked"]
    assert state["taken"] == 2 and len(record["outputs"]["tile"]["keys"]) == 2
    record = run(resume_workflow("stopme", instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done" and state["taken"] == 4


def test_the_tool_call_budget_stops_a_run(tmp_path):
    instrument, analysis, state = twin()
    import dataclasses

    tight = dataclasses.replace(get_workflow("map_grid"), budget={"max_tool_calls": 8})
    record = run(run_workflow(tight, INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "stopped" and "budget" in record["error"] and state["blanked"]


def test_a_resumed_run_has_only_the_budget_its_earlier_attempts_left(tmp_path):
    import dataclasses

    instrument, analysis, state = twin()
    tight = dataclasses.replace(get_workflow("map_grid"), budget={"max_tool_calls": 15})
    record = run(run_workflow(tight, INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "stopped" and "budget" in record["error"] and record["calls"] >= 15 and record["seconds"] >= 0
    taken = state["taken"]
    record = run(resume_workflow(record["run_id"], instrument=instrument, analysis=analysis, state_dir=tmp_path, find=lambda name: tight))
    assert record["state"] == "stopped" and "budget" in record["error"] and state["taken"] == taken


def test_a_number_that_is_not_finite_is_refused_before_anything_runs(tmp_path):
    workflow = get_workflow("map_grid")
    for bad in ({"fov_m": float("nan")}, {"fov_m": "nan"}, {"size_m": float("inf")}, {"tiles_per_side": float("inf")}, {"tiles_per_side": "nan"}):
        with pytest.raises(ValueError, match="must be a"):
            workflow.check_inputs(bad)
    instrument, analysis, _ = twin()
    record = run(run_workflow("map_grid", {"fov_m": "nan"}, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "refused" and instrument.log == []


def test_the_grid_is_held_to_the_instruments_own_image_shift_limit(tmp_path):
    instrument, analysis, state = twin()
    state["shift_limit"] = 1e-7  # the 2 x 2 grid needs 1.05e-7 m
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path / "a"))
    assert record["state"] == "failed" and "over the limit of 1e-07 m" in record["error"]
    assert state["taken"] == 0 and "Twin_set_image_shift" not in instrument.names()

    instrument, analysis, state = twin()
    state["shift_limit"] = 5e-6  # a wider instrument but the workflow asks for less
    record = run(run_workflow("map_grid", {**INPUTS, "max_shift_m": 1e-7}, instrument=instrument, analysis=analysis, state_dir=tmp_path / "b"))
    assert record["state"] == "failed" and "over the limit of 1e-07 m" in record["error"]
    record = run(run_workflow("map_grid", {"size_m": 4e-6, "fov_m": 5e-7}, instrument=instrument, analysis=analysis, state_dir=tmp_path / "c"))
    assert record["state"] == "done", record  # within this instrument's reach


def test_a_stage_map_that_fails_puts_the_stage_back(tmp_path):
    instrument, analysis, state = twin(lambda n: "Error: detector timeout" if n == 1 else None)
    record = run(run_workflow("map_grid", {**INPUTS, "move_by": "stage"}, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "failed" and "detector timeout" in record["error"]
    assert state["stage"][:2] == [1e-6, 2e-6] and state["fov"] == 5.8e-9 and state["blanked"]


def test_a_blank_that_fails_never_hides_why_the_tile_failed(tmp_path):
    instrument, analysis, state = twin(lambda n: "Error: detector timeout" if n == 0 else None)
    instrument.tools["Twin_blank_beam"] = lambda a: "Error: blanker offline"
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path / "a"))
    assert record["state"] == "failed" and "detector timeout" in record["error"] and "could not be blanked" in record["error"]

    instrument, analysis, state = twin()
    instrument.tools["Twin_blank_beam"] = lambda a: "Error: blanker offline"
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path / "b"))
    assert record["state"] == "failed" and "stem_image_HAADF_0001.h5" in record["error"] and "could not be blanked" in record["error"]


def test_a_restore_that_fails_is_named_in_the_record(tmp_path):
    instrument, analysis, state = twin()
    shift = instrument.tools["Twin_set_image_shift"]
    instrument.tools["Twin_set_image_shift"] = lambda a: "Error: shift refused" if a["shift"] == [0.0, 0.0] else shift(a)
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["unrestored"] and record["unrestored"][0]["tool"] == "*_set_image_shift" and "shift refused" in record["unrestored"][0]["error"]
    assert record["state"] == "failed" and state["blanked"]  # The next shift back to the centre fails the run and the beam is still blanked.


def test_the_hash_covers_the_helpers_a_block_works_through(monkeypatch):
    import inspect

    from asyncroscopy.workflows import graph as graph_module

    before = get_workflow("map_grid").hash()
    real = inspect.getsource
    monkeypatch.setattr(graph_module.inspect, "getsource",
                        lambda obj: real(obj) + ("\n# _expose changed" if getattr(obj, "__name__", "") == "asyncroscopy.workflows.shipped_blocks" else ""))
    assert get_workflow("map_grid").hash() != before


# The MCP server an agent uses
class FakeEndpoints(Endpoints):
    def __init__(self, instrument, analysis, simulated=True, stack_id="tango://twin:1", fingerprint="twin-1", software="v1"):
        super().__init__("fake://instrument", caller="workflow-test")
        self.instrument, self.analysis, self.simulated, self.stack_id = instrument, analysis, simulated, stack_id
        self.fingerprint, self.software = fingerprint, software

    async def open(self, run_id=None, caller=None, instrument_url=None, run_key=None):
        self.url(instrument_url)
        return self.instrument, self.analysis, {"simulated": self.simulated, "stack_id": self.stack_id,
                                                "stack_fingerprint": {"digest": self.fingerprint, "software": {"commit": self.software},
                                                                      "simulated": {"value": self.simulated}}}


def served(tmp_path, simulated=True, **kwargs):
    from asyncroscopy.workflows.server import WorkflowServer

    instrument, analysis, state = twin(**kwargs)
    server = WorkflowServer(FakeEndpoints(instrument, analysis, simulated), state_dir=tmp_path / "state")
    return server, instrument, state


async def call(client, tool, **arguments):
    result = await client.call_tool(tool, arguments)
    return result.structured_content if result.structured_content is not None else json.loads(result.content[0].text)


def test_an_agent_lists_runs_and_reads_a_workflow_over_mcp(tmp_path):
    from fastmcp import Client

    server, instrument, state = served(tmp_path)

    async def go():
        async with Client(server.mcp) as client:
            names = {t.name for t in await client.list_tools()}
            assert {"list_workflows", "run_workflow", "workflow_status", "stop_workflow", "resume_workflow", "list_blocks"} <= names
            assert "propose_workflow" in names
            listed = await call(client, "list_workflows")
            maps = {w["name"]: w for w in listed["workflows"]}["map_grid"]
            assert "size_m" in maps["inputs"]
            shown = await call(client, "get_workflow", name="map_grid")
            assert "StateGraph(WorkflowState)" in shown["source"] and shown["graph"]["nodes"]["stitch"]["block"] == "stitch_map"
            done = await call(client, "run_workflow", name="map_grid", inputs=INPUTS)
            assert done["state"] == "done" and done["outputs"]["stitch"]["map"] == "map_abc_map_v1"
            assert len(done["outputs"]["plan"]["tiles"]) == 4
            again = await call(client, "workflow_status", run_id=done["run_id"])
            assert again["state"] == "done"
            refused = await call(client, "run_workflow", name="map_grid", inputs={"size_m": 1.0})
            assert refused["state"] == "refused" and "outside its range" in refused["problems"][0]
            assert (await call(client, "run_workflow", name="nope"))["state"] == "refused"
    run(go())
    assert state["taken"] == 4 and state["blanked"]


def test_the_server_runs_nothing_on_a_stack_that_is_not_simulated(tmp_path):
    from fastmcp import Client

    for flag in (False, None):
        server, instrument, state = served(tmp_path / str(flag), simulated=flag)

        async def go():
            async with Client(server.mcp) as client:
                return await call(client, "run_workflow", name="map_grid", inputs=INPUTS)
        reply = run(go())
        assert reply["state"] == "refused" and "simulated" in reply["problems"][0]
        assert instrument.log == []


def test_a_failed_run_is_resumed_over_mcp_and_a_changed_workflow_is_not(tmp_path):
    from fastmcp import Client

    failing = {"on": True}
    server, instrument, state = served(tmp_path, on_acquire=lambda n: "Error: detector timeout" if n == 1 and failing["on"] else None)

    async def go():
        async with Client(server.mcp) as client:
            failed = await call(client, "run_workflow", name="map_grid", inputs=INPUTS)
            assert failed["state"] == "failed" and "resume_workflow" in failed["next"]
            failing["on"] = False
            resumed = await call(client, "resume_workflow", run_id=failed["run_id"])
            assert resumed["state"] == "done"
            assert "plan_grid" in {b["name"] for b in (await call(client, "list_blocks"))["blocks"]}
    run(go())
    assert state["taken"] == 4

    # A run only resumes with the code it started with.
    first = register_workflow(two_tiles("changing"))
    instrument, analysis, state = twin(lambda n: "Error: detector timeout" if n == 1 else None)
    record = run(run_workflow("changing", {}, instrument=instrument, analysis=analysis, state_dir=tmp_path / "c"))
    assert record["state"] == "failed"
    from asyncroscopy.workflows import graph as registry

    registry._WORKFLOWS["changing"] = two_tiles("changing", when="tile.remaining > 1")
    try:
        with pytest.raises(ValueError, match="has changed since run"):
            run(resume_workflow(record["run_id"], instrument=instrument, analysis=analysis, state_dir=tmp_path / "c"))
    finally:
        registry._WORKFLOWS.pop("changing", None)
    assert first.name == "changing"


def test_a_stack_that_is_not_simulated_runs_only_when_hardware_is_allowed_on_purpose(tmp_path):
    from fastmcp import Client

    from asyncroscopy.workflows.runner import twin_only_problem
    from asyncroscopy.workflows.server import WorkflowServer

    assert twin_only_problem({"simulated": None}) and twin_only_problem({}) and not twin_only_problem({"simulated": True})
    assert twin_only_problem({"simulated": False}, hardware=True) is None
    instrument, analysis, state = twin()
    server = WorkflowServer(FakeEndpoints(instrument, analysis, None), state_dir=tmp_path / "state", hardware=True)

    async def go():
        async with Client(server.mcp) as client:
            return await call(client, "run_workflow", name="map_grid", inputs=INPUTS)
    reply = run(go())
    assert reply["state"] != "refused" and instrument.log


# Workflows inside the agent process
def test_the_agent_server_runs_workflows_itself_so_no_workflow_server_is_needed(tmp_path):
    pytest.importorskip("langchain_core")
    from asyncroscopy.mcp.agent_mcp_server import AgentMCPServer, main

    off = AgentMCPServer("agent", ["http://127.0.0.1:8000/mcp"], verbose=False, workflows=False)
    assert off.workflows is None

    server = AgentMCPServer("agent", ["http://127.0.0.1:8000/mcp"], verbose=False)  # on by default
    assert server.workflows.endpoints.mcp_url == "http://127.0.0.1:8000/mcp" and not server.workflows.hardware
    instrument, analysis, state = twin()
    server.workflows.endpoints = FakeEndpoints(instrument, analysis, simulated=True)
    server.workflows.state_dir = tmp_path
    tools = {t.name: t for t in server.workflow_tools()}
    assert {"list_workflows", "run_workflow", "workflow_status", "stop_workflow", "resume_workflow"} <= set(tools)
    done = run(tools["run_workflow"].ainvoke({"name": "map_grid", "inputs": INPUTS}))
    assert done["state"] == "done" and state["taken"] == 4 and state["blanked"]

    # Real hardware needs the hardware flag and a loopback host.
    server.workflows.endpoints = FakeEndpoints(*twin()[:2], simulated=None)
    assert run(tools["run_workflow"].ainvoke({"name": "map_grid", "inputs": INPUTS}))["state"] == "refused"
    assert AgentMCPServer("agent", [], verbose=False, workflows_hardware=True).workflows.hardware
    assert main(["--name", "a", "--transport", "streamable-http", "--http-host", "0.0.0.0", "--http-port", "8002", "--workflows-hardware"]) == 1


# Workflows written by an agent and their approval
ASKED = "Please create a workflow that takes the tiles of a square and stops, without stitching."


def test_an_agent_written_workflow_is_held_to_graph_wiring_before_any_of_it_runs(tmp_path):
    from asyncroscopy.workflows.authoring import TEMPLATE, NotAWorkflow, check_source, load_source

    assert check_source(TEMPLATE) == []
    workflow = load_source(TEMPLATE)  # the example shown to models must itself pass
    assert workflow.name == "tiles_only" and workflow.lint()["ok"] and workflow.source() == TEMPLATE

    marker = tmp_path / "ran"
    hostile = {
        "an import": TEMPLATE.replace("from langgraph.graph import", "import os\nfrom langgraph.graph import"),
        "a call in build": TEMPLATE.replace("    return graph", f"    open({str(marker)!r}, 'w')\n    return graph"),
        "a call as an input": TEMPLATE.replace('move_by="stage"', f"move_by=open({str(marker)!r}, 'w')"),
        "a call at the top": TEMPLATE.replace("INPUTS = {", f"X = open({str(marker)!r}, 'w')\nINPUTS = {{"),
        "a second function": TEMPLATE.replace("def build(", "def other():\n    pass\n\n\ndef build("),
        "a hand-made node": TEMPLATE.replace('graph.add_edge(START, "plan")', 'graph.add_node("x", lambda s: s)\n    graph.add_edge(START, "plan")'),
        "a hand-made router": TEMPLATE.replace('use.when("tile.remaining > 0", then="tile", otherwise="home", max_visits=400)', "lambda s: 'home'"),
        "an attribute": TEMPLATE.replace("return graph", "return graph.__class__"),
        "a dunder name": TEMPLATE.replace("inputs=INPUTS", "inputs=__builtins__"),
        "a comprehension": TEMPLATE.replace('"min": 1.0e-7', '"min": [x for x in (1,)]'),
    }
    for what, source in hostile.items():
        assert source != TEMPLATE, what
        problems = check_source(source)
        assert problems and all(p.startswith("line ") or "exactly one" in p for p in problems), (what, problems)
        with pytest.raises(NotAWorkflow):
            load_source(source)
    assert not marker.exists()  # no refused module ran any code
    assert "not valid Python" in check_source("def build(:")[0] and check_source("")


def test_a_draft_runs_on_the_twin_and_on_an_instrument_only_after_a_person_approves_it_once(tmp_path):
    from fastmcp import Client

    from asyncroscopy.workflows import Library
    from asyncroscopy.workflows.authoring import TEMPLATE
    from asyncroscopy.workflows.server import WorkflowServer

    library = Library(tmp_path / "library")
    instrument, analysis, state = twin()
    on_twin = WorkflowServer(FakeEndpoints(instrument, analysis, True), state_dir=tmp_path / "twin", library=library)
    hardware_instrument, hardware_analysis, hardware_state = twin()
    on_hardware = WorkflowServer(FakeEndpoints(hardware_instrument, hardware_analysis, None, "tango://scope-a:9094"), state_dir=tmp_path / "hw",
                                 library=library, hardware=True)
    inputs = {"size_m": 5e-7}

    async def as_the_agent():
        async with Client(on_twin.mcp) as client:
            shown = await call(client, "list_blocks")
            assert "plan_grid" in {b["name"] for b in shown["blocks"]} and "StateGraph(WorkflowState)" in shown["example"] and shown["rules"]
            # A proposal without the user's request is refused.
            for missing in ("", "   ", "a map"):
                unasked = await call(client, "propose_workflow", user_request=missing, source=TEMPLATE)
                assert not unasked["ok"] and "written for a task the user gave" in unasked["problems"][0] and unasked["draft"] is None
            assert {w["name"] for w in (await call(client, "list_workflows"))["workflows"]} == SHIPPED
            bad = await call(client, "propose_workflow", user_request=ASKED, source=TEMPLATE.replace('"plan_grid"', '"plan_gird"'))
            assert not bad["ok"] and "no block 'plan_gird'" in " ".join(bad["problems"]) and bad["draft"] is None
            unsafe = await call(client, "propose_workflow", user_request=ASKED, source=TEMPLATE.replace("    return graph", "    exit()\n    return graph"))
            assert not unsafe["ok"] and "inside build only" in unsafe["problems"][0]
            taken = await call(client, "propose_workflow", user_request=ASKED, source=TEMPLATE.replace('name="tiles_only"', 'name="map_grid"'))
            assert not taken["ok"] and "shipped workflow's name" in taken["problems"][0]
            kept = await call(client, "propose_workflow", user_request=ASKED, source=TEMPLATE)
            assert kept["ok"] and kept["status"] == "draft" and "approve tiles_only" in kept["next"] and "waiting for a person" in kept["next"]
            waiting = {w["name"]: w for w in (await call(client, "list_workflows"))["workflows"]}["tiles_only"]
            assert waiting["requested"] == ASKED
            assert {w["name"]: w["status"] for w in (await call(client, "list_workflows"))["workflows"]} == {**{name: "shipped" for name in SHIPPED}, "tiles_only": "draft"}
            assert (await call(client, "get_workflow", name="tiles_only"))["source"] == TEMPLATE
            return await call(client, "run_workflow", name="tiles_only", inputs=inputs)
    rehearsed = run(as_the_agent())
    assert rehearsed["state"] == "done" and state["taken"] == 4 and state["blanked"]  # a draft runs on the twin without approval

    async def on_the_instrument(tool="run_workflow", **arguments):
        async with Client(on_hardware.mcp) as client:
            return await call(client, tool, **arguments)
    refused = run(on_the_instrument(name="tiles_only", inputs=inputs))
    assert refused["state"] == "refused" and "nobody has approved it" in refused["problems"][0] and hardware_instrument.log == []
    assert run(on_the_instrument(name="map_grid", inputs=INPUTS))["state"] == "done"  # a shipped workflow needs no approval
    taken_by_shipped = hardware_state["taken"]

    with pytest.raises(ValueError, match="names the person"):
        library.approve("tiles_only", "")
    record = library.approve("tiles_only", "sam")  # only a person approves and no tool can
    assert record["by"] == "sam" and record["hash"] == library.workflow("tiles_only").hash() and library.find("tiles_only")[1] == "approved"
    # The approval records the rehearsal run on the twin.
    assert record["rehearsed"]["run_id"] == rehearsed["run_id"] and record["instrument"] == "any" and record["expires"] is None
    assert record["request"] == ASKED
    assert "approve" not in " ".join(t.__name__ for t in on_hardware.tools())
    for _ in range(2):  # one approval covers later runs
        assert run(on_the_instrument(name="tiles_only", inputs=inputs))["state"] == "done"
    assert hardware_state["taken"] == taken_by_shipped + 8
    listed = {w["name"]: w for w in run(on_the_instrument("list_workflows"))["workflows"]}
    assert listed["tiles_only"]["status"] == "approved" and listed["tiles_only"]["approved_by"] == "sam"

    # Changed code or a revoked approval makes it a draft again.
    approved_file = tmp_path / "library" / "approved" / "tiles_only.py"
    approved_file.write_text(TEMPLATE.replace("max_visits=400", "max_visits=399"), encoding="utf-8")
    assert library.find("tiles_only")[1] == "draft"
    assert run(on_the_instrument(name="tiles_only", inputs=inputs))["state"] == "refused"
    approved_file.write_text(TEMPLATE, encoding="utf-8")
    assert library.find("tiles_only")[1] == "approved"
    assert library.revoke("tiles_only") == {"name": "tiles_only", "status": "draft"}
    assert run(on_the_instrument(name="tiles_only", inputs=inputs))["state"] == "refused"
    assert (tmp_path / "library" / "drafts" / "tiles_only.py").exists()  # revoking keeps the text

    # An approval can be tied to one instrument and can expire.
    library.approve("tiles_only", "sam", instrument="tango://scope-b:9094")
    elsewhere = run(on_the_instrument(name="tiles_only", inputs=inputs))
    assert elsewhere["state"] == "refused" and "approved for the instrument tango://scope-b:9094, not this one (tango://scope-a:9094)" in elsewhere["problems"][0]
    library.approve("tiles_only", "sam", instrument="tango://scope-a:9094", days=30)
    assert run(on_the_instrument(name="tiles_only", inputs=inputs))["state"] == "done"
    listed = {w["name"]: w for w in run(on_the_instrument("list_workflows"))["workflows"]}["tiles_only"]
    assert listed["approved_for"] == "tango://scope-a:9094" and listed["approval_ends"]
    import time as clock

    assert library.hardware_problem("tiles_only", "approved", "tango://scope-a:9094", now=clock.time() + 29 * 86400) is None
    assert "ended on" in library.hardware_problem("tiles_only", "approved", "tango://scope-a:9094", now=clock.time() + 31 * 86400)
    with pytest.raises(ValueError, match="days must be above 0"):
        library.approve("tiles_only", "sam", days=0)


def test_a_rehearsal_counts_for_approval_only_on_this_stacks_fingerprint_and_only_when_it_passed(tmp_path):
    from fastmcp import Client

    from asyncroscopy.workflows import Library
    from asyncroscopy.workflows.authoring import TEMPLATE
    from asyncroscopy.workflows.server import WorkflowServer

    library = Library(tmp_path / "library")
    assert library.propose(TEMPLATE, request=ASKED)["ok"]
    code = library.workflow("tiles_only").hash()

    async def rehearse(fingerprint, on_acquire=None):
        instrument, analysis, _ = twin(on_acquire)
        server = WorkflowServer(FakeEndpoints(instrument, analysis, True, fingerprint=fingerprint), state_dir=tmp_path / fingerprint, library=library)
        async with Client(server.mcp) as client:
            return await call(client, "run_workflow", name="tiles_only", inputs={})

    async def approve_on(fingerprint):
        instrument, analysis, _ = twin()
        server = WorkflowServer(FakeEndpoints(instrument, analysis, True, fingerprint=fingerprint), state_dir=tmp_path / "operator", library=library)
        _, approve, _ = server.operator_tools()
        return await approve(name="tiles_only", approver="sam", expected_hash=code)

    assert run(rehearse("twin-a"))["state"] == "done"
    (kept,) = library.rehearsals("tiles_only", code)
    assert kept["stack_id"] == "tango://twin:1" and kept["stack_fingerprint"]["digest"] == "twin-a" and kept["cleanup_failed"] == []

    elsewhere = run(approve_on("twin-b"))
    assert not elsewhere["ok"] and "passed only on another stack" in elsewhere["error"] and "twin-a" in elsewhere["error"] and "twin-b" in elsewhere["error"]
    with pytest.raises(ValueError, match="no stack_fingerprint"):
        library.approve("tiles_only", "sam", require_rehearsal=True)

    assert run(rehearse("twin-b", lambda n: "Error: detector timeout" if n == 0 else None))["state"] == "failed"
    assert "passed only on another stack" in run(approve_on("twin-b"))["error"]

    library.record_rehearsal(library.workflow("tiles_only"), {"hash": code, "run_id": "r-cleanup", "state": "done",
                                                              "cleanup": [{"tool": "*_blank_beam", "ok": False, "error": "timed out"}]},
                             {"stack_id": "tango://twin:1", "stack_fingerprint": {"digest": "twin-b"}})
    assert library.rehearsals("tiles_only", code)[-1]["cleanup_failed"] == ["*_blank_beam"]
    assert "passed only on another stack" in run(approve_on("twin-b"))["error"]

    assert run(rehearse("twin-b"))["state"] == "done"
    approved = run(approve_on("twin-b"))
    assert approved["ok"] and approved["approval"]["rehearsed"]["stack_fingerprint"] == "twin-b"

    async def approve_on_hardware(software):
        instrument, analysis, _ = twin()
        endpoints = FakeEndpoints(instrument, analysis, False, stack_id="tango://column:1", fingerprint="column-1", software=software)
        server = WorkflowServer(endpoints, state_dir=tmp_path / "hardware", library=library)
        _, approve, _ = server.operator_tools()
        return await approve(name="tiles_only", approver="sam", expected_hash=code)

    library.revoke("tiles_only")
    other = run(approve_on_hardware("v2"))
    assert not other["ok"] and "same software" in other["error"] and "v2" in other["error"]
    on_column = run(approve_on_hardware("v1"))
    assert on_column["ok"] and on_column["approval"]["rehearsed"]["stack_fingerprint"] in ("twin-a", "twin-b")


def test_rehearsals_are_kept_for_the_person_approving_and_an_approvers_list_is_honoured(tmp_path):
    from fastmcp import Client

    from asyncroscopy.workflows import Library
    from asyncroscopy.workflows.authoring import TEMPLATE
    from asyncroscopy.workflows.server import WorkflowServer

    library = Library(tmp_path / "library")
    assert library.propose(TEMPLATE, request=ASKED)["ok"] and library.request("tiles_only")["request"] == ASKED

    # Every rehearsal is recorded for the approver, including failed ones.
    instrument, analysis, state = twin(lambda n: "Error: detector timeout" if n == 1 else None)
    failing = WorkflowServer(FakeEndpoints(instrument, analysis, True), state_dir=tmp_path / "a", library=library)

    async def rehearse(server, **inputs):
        async with Client(server.mcp) as client:
            return await call(client, "run_workflow", name="tiles_only", inputs=inputs)
    assert run(rehearse(failing))["state"] == "failed"
    kept = library.rehearsals("tiles_only")
    assert [r["state"] for r in kept] == ["failed"] and "detector timeout" in kept[0]["error"]
    instrument, analysis, state = twin()
    assert run(rehearse(WorkflowServer(FakeEndpoints(instrument, analysis, True), state_dir=tmp_path / "b", library=library)))["state"] == "done"
    assert [r["state"] for r in library.rehearsals("tiles_only", library.workflow("tiles_only").hash())] == ["failed", "done"]
    # An edited draft has no rehearsals but can still be approved.
    assert library.propose(TEMPLATE.replace("max_visits=400", "max_visits=300"))["ok"]
    assert library.rehearsals("tiles_only", library.workflow("tiles_only").hash()) == [] and library.request("tiles_only") is None
    assert library.approve("tiles_only", "sam")["rehearsed"] is None
    library.revoke("tiles_only")
    assert library.propose(TEMPLATE, request=ASKED)["ok"]

    # With an approvers list only those logins can approve.
    (tmp_path / "library" / "approvers.json").write_text('["sam.r"]', encoding="utf-8")
    with pytest.raises(PermissionError, match="approvers.json"):
        library.approve("tiles_only", "Somebody Else", login="guest")
    with pytest.raises(PermissionError):
        library.approve("tiles_only", "sam")  # no login is also refused
    record = library.approve("tiles_only", "Sam Reed", login="sam.r")
    assert (record["by"], record["login"]) == ("Sam Reed", "sam.r") and library.find("tiles_only")[1] == "approved"


def test_a_person_sees_every_workflow_drawn_on_one_page(tmp_path, capsys):
    from asyncroscopy.workflows import Library
    from asyncroscopy.workflows.__main__ import main
    from asyncroscopy.workflows.authoring import TEMPLATE
    from asyncroscopy.workflows.view import graph_svg, render_html

    library = Library(tmp_path / "library")
    # Text written by a model is escaped on the page.
    nasty = TEMPLATE.replace("Take every tile of a square grid around the current position", "<script>alert(1)</script> Take every tile")
    assert library.propose(nasty, request="<img src=x onerror=alert(2)> please create a tiles-only workflow")["ok"]
    page = render_html(library)
    assert page.startswith("<!doctype html>") and "http://" not in page and "https://" not in page  # one file that fetches nothing
    assert "<script" not in page and "<img" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "&lt;img src=x" in page
    waiting, shipped = page.index("Waiting for a person's approval (1)"), page.index(f"Shipped with the software ({len(SHIPPED)})")
    assert waiting < page.index('id="tiles_only"') < page.index("Approved (0)") < shipped < page.index('id="map_grid"')
    assert "python -m asyncroscopy.workflows approve tiles_only" in page and "it has never run to the end anywhere" in page
    assert "graph = StateGraph(WorkflowState)" in page and page.count("<svg") == len(SHIPPED) + 1  # each shipped workflow and the draft
    drawing = graph_svg(get_workflow("map_grid").blueprint())
    for text in ("plan_grid", "acquire_tile", "return_start", "collect_run", "stitch_map", "tile.remaining &gt; 0<tspan", ">at most 2000 times<", "otherwise", ">start<", ">end<"):
        assert text in drawing, text

    library.approve("tiles_only", "sam", days=30)
    page = render_html(library)
    assert "Waiting for a person's approval (0)" in page and "Approved (1)" in page and "by sam" in page and "please create a tiles-only workflow" in page

    out = tmp_path / "page.html"
    assert main(["view", "--library", str(tmp_path / "library"), "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip() == str(out.resolve()) and "tiles_only" in out.read_text(encoding="utf-8")


def test_the_command_line_shows_a_person_what_they_approve(tmp_path, capsys):
    from asyncroscopy.workflows.__main__ import main
    from asyncroscopy.workflows.authoring import TEMPLATE

    source = tmp_path / "mine.py"
    source.write_text(TEMPLATE, encoding="utf-8")
    library = ["--library", str(tmp_path / "library")]
    assert main(["propose", str(source), "--request", ASKED, *library]) == 0
    assert main(["approve", "tiles_only", *library]) == 1  # without a terminal or --yes nothing is approved
    out = capsys.readouterr()
    assert "graph = StateGraph(WorkflowState)" in out.out and '"hash"' in out.out and ASKED in out.out
    assert "never run to the end on a digital twin" in out.out and "instrument any, until revoked" in out.out and "not approved" in out.err
    here = ["--instrument", "tango://scope-a:9094", *library]
    assert main(["approve", "tiles_only", "--yes", "--by", "sam", "--days", "90", *here]) == 0
    approved = capsys.readouterr().out
    assert '"by": "sam"' in approved and '"instrument": "tango://scope-a:9094"' in approved and "for 90 days" in approved
    assert main(["list", *library]) == 0
    listed = {w["name"]: w for w in json.loads(capsys.readouterr().out)}
    assert listed["tiles_only"]["status"] == "approved" and listed["tiles_only"]["approved_by"] == "sam" and listed["map_grid"]["status"] == "shipped"
    assert listed["tiles_only"]["approved_for"] == "tango://scope-a:9094" and listed["tiles_only"]["approval_ends"]
    assert main(["revoke", "tiles_only", *library]) == 0 and main(["revoke", "tiles_only", *library]) == 1
    assert main(["approve", "nope", *here]) == 1


# Parallel steps, any tool as a step and questions for a person
class SlowServer(FakeServer):
    """A fake endpoint whose calls take a moment. It counts how many ran at once."""

    def __init__(self, tools: dict, delay_s: float = 0.05) -> None:
        super().__init__(tools)
        self.delay_s, self.going, self.most_at_once = delay_s, 0, 0

    async def call(self, name, arguments=None):
        self.going += 1
        self.most_at_once = max(self.most_at_once, self.going)
        try:
            await asyncio.sleep(self.delay_s)
            return await super().call(name, arguments)
        finally:
            self.going -= 1


def spectra_endpoint(**replies):
    taken = {"n": 0}

    def image(args):
        taken["n"] += 1
        return f"stem_image_HAADF_{taken['n']:04d}.h5\nimage 256 x 256"

    return SlowServer({
        "Twin_acquire_scanned_image": image,
        "Twin_set_fov": lambda a: "ok",
        "Twin_acquire_spectrum": replies.get("eds", lambda a: "spectrum_eds_0001.h5\npeaks at 8.04 keV"),
        "EELS_acquire_eels_spectrum": replies.get("eels", lambda a: "spectrum_eels_0001.h5"),
    })


def test_parallel_branches_run_at_the_same_time_and_a_question_waits_for_a_person(tmp_path):
    from asyncroscopy.workflows.authoring import TEMPLATE_PARALLEL, check_source, load_source
    from asyncroscopy.workflows.runner import check_answers

    assert check_source(TEMPLATE_PARALLEL) == []
    workflow = load_source(TEMPLATE_PARALLEL)  # the second example shown to models must also pass
    blueprint, checked = workflow.blueprint(), workflow.lint()
    assert checked["ok"], checked["problems"]
    assert {"*_acquire_scanned_image", "*_set_fov", "*_acquire_spectrum", "*_acquire_eels_spectrum"} == set(checked["tools"])
    assert ["zoom", "eds"] in blueprint["edges"] and ["zoom", "eels"] in blueprint["edges"]  # two plain edges make parallel branches
    assert {"from": "eds", "to": "look", "after": ["eds", "eels"]} in blueprint["edges"]  # the join

    def find(name):
        return workflow

    instrument = spectra_endpoint()
    record = run(run_workflow(workflow, {}, instrument=instrument, state_dir=tmp_path))
    # It took the image and now waits for an answer.
    assert record["state"] == "waiting" and instrument.names() == ["Twin_acquire_scanned_image"], record
    question = record["questions"][0]
    assert question["node"] == "ok" and question["options"] == ["yes", "no"] and question["show"] == "stem_image_HAADF_0001.h5"
    with pytest.raises(ValueError, match="waiting for a person's answer"):
        run(resume_workflow(record["run_id"], instrument=instrument, state_dir=tmp_path, find=find))
    with pytest.raises(ValueError, match="must be one of"):
        check_answers(record, {question["id"]: "maybe"})
    with pytest.raises(ValueError, match="has no question"):
        check_answers(record, {"nothing": "yes"})
    assert instrument.names() == ["Twin_acquire_scanned_image"]

    record = run(resume_workflow(record["run_id"], instrument=instrument, state_dir=tmp_path, find=find, answers={question["id"]: "Yes"}))
    assert record["state"] == "done" and not record.get("questions"), record
    assert record["outputs"]["ok"]["answer"] == "yes" and record["answered"][0]["answer"] == "yes"
    names = instrument.names()
    assert names[:2] == ["Twin_acquire_scanned_image", "Twin_set_fov"] and names[-1] == "Twin_acquire_scanned_image"  # the first image is not retaken
    assert set(names[2:4]) == {"Twin_acquire_spectrum", "EELS_acquire_eels_spectrum"} and len(names) == 5
    assert instrument.most_at_once == 2  # the two spectra ran together
    assert record["outputs"]["eds"]["key"] == "spectrum_eds_0001.h5" and record["outputs"]["eels"]["key"] == "spectrum_eels_0001.h5"

    # Answering no ends the run with no more calls.
    instrument = spectra_endpoint()
    record = run(run_workflow(workflow, {}, instrument=instrument, state_dir=tmp_path / "no"))
    record = run(resume_workflow(record["run_id"], instrument=instrument, state_dir=tmp_path / "no", find=find,
                                 answers={record["questions"][0]["id"]: "no"}))
    assert record["state"] == "done" and instrument.names() == ["Twin_acquire_scanned_image"]


def test_a_branch_that_fails_is_the_only_one_repeated_on_resume(tmp_path):
    from asyncroscopy.workflows.authoring import TEMPLATE_PARALLEL, load_source

    workflow = load_source(TEMPLATE_PARALLEL)

    def find(name):
        return workflow

    failing = {"on": True}
    instrument = spectra_endpoint(eels=lambda a: "Error: spectrometer busy" if failing["on"] else "spectrum_eels_0001.h5")
    record = run(run_workflow(workflow, {}, instrument=instrument, state_dir=tmp_path))
    record = run(resume_workflow(record["run_id"], instrument=instrument, state_dir=tmp_path, find=find, answers={record["questions"][0]["id"]: "yes"}))
    assert record["state"] == "failed" and "spectrometer busy" in record["error"] and record["at_node"] == ["eels"], record
    failing["on"] = False
    record = run(resume_workflow(record["run_id"], instrument=instrument, state_dir=tmp_path, find=find))
    assert record["state"] == "done"
    assert instrument.names().count("Twin_acquire_spectrum") == 1 and instrument.names().count("EELS_acquire_eels_spectrum") == 2


def any_tool(name: str, both_ways: bool = False, **step) -> Workflow:
    def build(use):
        graph = StateGraph(WorkflowState)
        graph.add_node("first", use.block("run_tool", **{"tool": "*_get_stage", **step}))
        graph.add_node("second", use.block("run_tool", tool="*_get_fov"))
        graph.add_edge(START, "first")
        if both_ways:
            graph.add_conditional_edges("first", use.when("first.result > 0", then="second", otherwise=END))
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        return graph
    return Workflow(name=name, summary="Read two values.", inputs={"which": {"type": "string", "default": "*_get_stage"}}, build=build)


def test_lint_knows_the_tools_a_workflow_names_and_refuses_what_it_cannot_check():
    fine = any_tool("reads").lint(["Twin_get_stage", "Twin_get_fov"])
    assert fine["ok"] and fine["tools"] == ["*_get_fov", "*_get_stage"], fine
    missing = any_tool("reads").lint(["Twin_get_stage"])
    assert any("offers no tool matching *_get_fov" in p for p in missing["problems"])
    # The tool name must be written in the workflow because a person approves it.
    hidden = any_tool("reads", tool="{inputs.which}").lint()
    assert any("must be written out as text" in p for p in hidden["problems"])
    on_analysis = any_tool("reads", tool="map_info", server="analysis").lint(["Twin_get_fov"], ["map_info"])
    assert on_analysis["ok"] and on_analysis["needs_analysis"], on_analysis
    # A node with a condition may have no other edges out.
    both = any_tool("reads", both_ways=True).lint()
    assert any("a node with a condition has only that one way out" in p for p in both["problems"])


def test_the_agent_relays_a_workflows_question_and_the_users_answer(tmp_path):
    from fastmcp import Client

    from asyncroscopy.workflows import Library
    from asyncroscopy.workflows.authoring import TEMPLATE_PARALLEL
    from asyncroscopy.workflows.server import WorkflowServer

    instrument = spectra_endpoint()
    server = WorkflowServer(FakeEndpoints(instrument, None, True), state_dir=tmp_path / "state", library=Library(tmp_path / "library"))

    async def go():
        async with Client(server.mcp) as client:
            shown = await call(client, "list_blocks")
            assert {"run_tool", "ask_person"} <= {b["name"] for b in shown["blocks"]} and shown["example_parallel"] == TEMPLATE_PARALLEL
            kept = await call(client, "propose_workflow", user_request="Please make a workflow that takes EDS and EELS together.", source=TEMPLATE_PARALLEL)
            assert kept["ok"], kept
            asked = await call(client, "run_workflow", name="spectra_together")
            assert asked["state"] == "waiting" and "answer_workflow" in asked["next"] and "Never answer it yourself" in asked["next"]
            assert asked["questions"][0]["question"].startswith("Zoom in here")
            again = await call(client, "resume_workflow", run_id=asked["run_id"])  # resuming is not answering
            assert again["state"] == "waiting" and "waiting for a person's answer" in again["error"]
            wrong = await call(client, "answer_workflow", run_id=asked["run_id"], answer="perhaps")
            assert wrong["state"] == "waiting" and "must be one of" in wrong["error"]
            done = await call(client, "answer_workflow", run_id=asked["run_id"], answer="yes")
            assert done["state"] == "done" and done["outputs"]["eels"]["key"] == "spectrum_eels_0001.h5", done
            late = await call(client, "answer_workflow", run_id=asked["run_id"], answer="yes")
            assert "not waiting for an answer" in late["error"]
    run(go())
    assert instrument.most_at_once == 2


def test_a_loop_of_three_nodes_runs_up_to_its_bound_without_hitting_the_recursion_limit(tmp_path):
    from asyncroscopy.workflows.blocks import Block, register_block, all_blocks

    async def count(ctx, args, previous):
        return {"n": (previous or {}).get("n", 0) + 1}

    if "test_count" not in all_blocks():
        register_block(Block(name="test_count", summary="Count visits.", run=count, outputs={"n": "integer"}))

    def build(use):
        graph = StateGraph(WorkflowState)
        for name in ("a", "b", "c"):
            graph.add_node(name, use.block("test_count"))
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_conditional_edges("c", use.when("c.n < 150", then="a", otherwise=END, max_visits=150))
        return graph

    workflow = Workflow(name="three_node_loop", summary="A loop of three nodes.", build=build)
    record = run(run_workflow(workflow, {}, instrument=FakeServer({}), state_dir=tmp_path))
    assert record["state"] == "done", record
    assert record["outputs"]["c"]["n"] == 150


def test_a_run_being_driven_is_not_resumed_a_second_time(tmp_path):
    from asyncroscopy.workflows.runner import _run_lock

    instrument, analysis, state = twin()
    instrument.tools["Twin_acquire_scanned_image"] = lambda a: "Error: no beam"
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path, run_id="busy"))
    assert record["state"] == "failed"
    with _run_lock(tmp_path, "busy"):  # as if another process were driving it now
        with pytest.raises(ValueError, match="being run now"):
            run(resume_workflow("busy", instrument=instrument, analysis=analysis, state_dir=tmp_path))


def test_a_running_map_reports_its_step_its_tiles_done_of_total_and_its_newest_image(tmp_path):
    from asyncroscopy.workflows import runner

    seen = []

    def watch(n):
        seen.append(runner.progress("watched"))  # before acquisition n (0-based) is taken

    instrument, analysis, _ = twin(on_acquire=watch)
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, run_id="watched", state_dir=tmp_path))
    assert record["state"] == "done"
    assert seen[0]["step"] == "tile" and "loop" not in seen[0] and "latest_image" not in seen[0]
    assert seen[2]["loop"] == {"node": "tile", "done": 2, "total": 4}  # two tiles taken, the third under way
    assert seen[2]["latest_image"] == "stem_image_HAADF_0002.h5"
    assert runner.progress("watched") is None  # a finished run is no longer live


def test_loop_progress_and_latest_image_read_only_what_is_there():
    from asyncroscopy.workflows.graph import loop_progress
    from asyncroscopy.workflows.runner import latest_image

    assert loop_progress("tile", {"index": 4, "remaining": 20, "level": 1}) == {"node": "tile", "done": 5, "total": 25, "level": 1}
    assert loop_progress("tile", {"index": True, "remaining": 1}) is None and loop_progress("plan", {"tiles": []}) is None
    calls = [{"tool": "Scope_acquire_scanned_image", "result": "a.h5\nimage", "error": False},
             {"tool": "Scope_acquire_spectrum", "result": "s.h5\nspectrum", "error": False},
             {"tool": "Scope_acquire_scanned_image", "result": "Error: beam blanked", "error": True}]
    assert latest_image(calls) == "a.h5" and latest_image([]) is None


ONE_IMAGE = '''"""Take one image."""

from langgraph.graph import END, START, StateGraph

from asyncroscopy.workflows import Steps, Workflow, WorkflowState


def build(use: Steps) -> StateGraph:
    graph = StateGraph(WorkflowState)
    graph.add_node("image", use.block("run_tool", tool="*_acquire_scanned_image", arguments={"detector_list": ["haadf"]}))
    graph.add_edge(START, "image")
    graph.add_edge("image", END)
    return graph


WORKFLOW = Workflow(name="one_image", summary="Take one image.", inputs={}, budget={"max_seconds": 60, "max_tool_calls": 10}, build=build)
'''


def test_the_microscopes_live_scan_pauses_the_run_for_a_person(tmp_path):
    """AutoScript refuses an acquisition while its own live scan runs; the run asks a person to stop it, then retries."""
    from asyncroscopy.workflows.authoring import load_source

    workflow = load_source(ONE_IMAGE)
    busy = {"on": True}

    def acquire(args):
        if busy["on"]:
            return ("Error: PyDs_PythonError: autoscript_core.common.ApplicationServerException: An unexpected error occurred in the "
                    "application server. Another scantask is already active")
        return "stem_image_HAADF_0001.h5"

    instrument = FakeServer({"Scope_acquire_scanned_image": acquire})
    record = run(run_workflow(workflow, {}, instrument=instrument, state_dir=tmp_path))
    assert record["state"] == "waiting", record
    question = record["questions"][0]
    assert "live scan" in question["question"] and question["options"] == ["continue"]

    # Stopping the waiting run ends it at once; resuming it asks again while the live scan still runs.
    assert stop_workflow(record["run_id"], tmp_path)["state"] == "stopped"
    assert read_run(record["run_id"], tmp_path)["state"] == "stopped"
    record = run(resume_workflow(record["run_id"], instrument=instrument, state_dir=tmp_path, find=lambda name: workflow))
    assert record["state"] == "waiting", record

    # The person stops the live scan and continues: the image is taken.
    busy["on"] = False
    record = run(resume_workflow(record["run_id"], instrument=instrument, state_dir=tmp_path, find=lambda name: workflow,
                                 answers={record["questions"][0]["id"]: "continue"}))
    assert record["state"] == "done", record
    assert instrument.names()[-1] == "Scope_acquire_scanned_image"


def test_a_second_run_that_drives_the_instrument_is_refused_while_one_is_running(tmp_path):
    from asyncroscopy.workflows.blocks import BlockContext
    from asyncroscopy.workflows.runner import LIVE

    instrument, analysis, _ = twin()
    other = BlockContext(instrument)
    other.acts = True
    LIVE["mapping-now"] = other
    try:
        record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path))
        assert record["state"] == "refused" and "mapping-now is driving the instrument" in record["problems"][0]
        assert instrument.names() == []  # nothing was sent
        other.acts = False  # a run that only reads does not hold the instrument
        assert run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path))["state"] == "done"
    finally:
        LIVE.pop("mapping-now", None)


# A call that never reached the endpoint is tried again; nothing else is
def test_a_call_that_never_reached_the_endpoint_is_retried_and_an_instrument_error_is_not(tmp_path, monkeypatch):
    import httpx
    from langgraph.types import RetryPolicy

    from asyncroscopy.workflows import graph as graph_module
    from asyncroscopy.workflows.blocks import Unreachable, undelivered

    monkeypatch.setattr(graph_module, "TRANSPORT_RETRY", RetryPolicy(max_attempts=3, initial_interval=0.01, jitter=False,
                                                                     retry_on=graph_module.TRANSPORT_RETRY.retry_on))
    try:
        raise RuntimeError("Client failed to connect") from httpx.ConnectError("refused")
    except RuntimeError as exc:
        assert undelivered(exc)
    assert not undelivered(RuntimeError("Error: the stage refused")) and not undelivered(httpx.ReadTimeout("no answer"))

    instrument, analysis, state = twin()
    real, dropped = instrument.call, []

    async def flaky(name, arguments=None):
        if name == "Twin_acquire_scanned_image" and not dropped:
            dropped.append(name)
            raise RuntimeError("Client failed to connect: All connection attempts failed") from httpx.ConnectError("refused")
        return await real(name, arguments)

    instrument.call = flaky
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path / "a"))
    assert dropped and record["state"] == "done" and state["taken"] == 4 and state["blanked"]

    instrument, analysis, state = twin(on_acquire=lambda n: "Error: detector not inserted")
    record = run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path / "b"))
    acquisitions = [n for n in instrument.names() if n == "Twin_acquire_scanned_image"]
    assert record["state"] == "failed" and len(acquisitions) == 1 and state["blanked"]
    from asyncroscopy.workflows import BlockFailed
    assert issubclass(Unreachable, BlockFailed)


def test_run_workflow_refuses_an_input_it_does_not_take_before_opening_the_instrument(tmp_path):
    from fastmcp import Client

    server, instrument, _ = served(tmp_path)

    async def go():
        async with Client(server.mcp) as client:
            return await call(client, "run_workflow", name="map_grid", inputs={**INPUTS, "dwell_s": 1e-6})
    reply = run(go())
    assert reply["state"] == "refused" and "not ['dwell_s']" in reply["problems"][0] and instrument.log == []


def test_a_person_reviews_approves_and_rejects_through_the_operator_endpoint_only(tmp_path):
    from fastmcp import Client

    from asyncroscopy.mcp.agent_mcp_server import AgentMCPServer
    from asyncroscopy.workflows import Library
    from asyncroscopy.workflows.authoring import TEMPLATE
    from asyncroscopy.workflows.server import WorkflowServer

    library = Library(tmp_path / "library")
    instrument, analysis, state = twin()
    on_twin = WorkflowServer(FakeEndpoints(instrument, analysis, True), state_dir=tmp_path / "state", library=library)
    assert not {"review_workflow", "approve_workflow", "reject_workflow"} & {fn.__name__ for fn in on_twin.tools()}

    operator = AgentMCPServer(name="Workflow_Operator_MCP", mcp_urls=[], verbose=False, endpoint_role="operator")
    assert operator.swarm is None
    operator.workflows = WorkflowServer(FakeEndpoints(instrument, analysis, True), state_dir=tmp_path / "state", library=library,
                                        name="Workflow_Operator_MCP")
    operator.setup()

    async def as_the_agent(tool, **arguments):
        async with Client(on_twin.mcp) as client:
            return await call(client, tool, **arguments)

    async def as_the_person(tool, **arguments):
        async with Client(operator.mcp) as client:
            if tool == "tools":
                return sorted(t.name for t in await client.list_tools())
            return await call(client, tool, **arguments)

    assert run(as_the_person("tools")) == ["action_space", "approve_workflow", "endpoint_info", "reject_workflow", "review_workflow"]
    info = run(as_the_person("endpoint_info"))
    assert info["role"] == "operator" and info["serves"] == "workflow_approvals" and info["tools"] == ["review_workflow", "approve_workflow", "reject_workflow"]
    space = run(as_the_person("action_space"))
    assert space["endpoint"] == {"name": "Workflow_Operator_MCP", "role": "operator"}
    assert {(a["tool"], a["risk"], a["effect"], a["device"]) for a in space["actions"]} == {
        ("review_workflow", "low", "observe", None), ("approve_workflow", "high", "act", None), ("reject_workflow", "high", "act", None)}

    assert run(as_the_agent("propose_workflow", user_request=ASKED, source=TEMPLATE))["ok"]
    review = run(as_the_person("review_workflow", name="tiles_only"))
    asked = review["request_as_reported_by_agent"]  # the proposing agent's words, not a record the person made
    assert review["ok"] and review["status"] == "draft" and review["source"] == TEMPLATE and asked["text"] == ASKED and "request" not in review
    assert asked["at"] > 0 and asked["this_code"] and set(review["graph"]) == {"nodes", "edges"}
    assert review["rehearsals"] == [] and not review["rehearsed"] and not review["can_approve"] and "never run to the end" in review["why_not"]
    assert review["approval"] is None and review["changed_since_approval"] is False and review["inputs"] == dict(library.workflow("tiles_only").inputs)

    refused = run(as_the_person("approve_workflow", name="tiles_only", approver="sam", expected_hash=review["hash"]))
    assert not refused["ok"] and "never run to the end on a digital twin" in refused["error"] and library.find("tiles_only")[1] == "draft"
    assert not run(as_the_person("approve_workflow", name="map_grid", approver="sam", expected_hash="x"))["ok"]  # shipped: nothing to approve
    assert "no workflow named" in run(as_the_person("review_workflow", name="nothing_here"))["error"]

    assert run(as_the_agent("run_workflow", name="tiles_only", inputs={"size_m": 5e-7}))["state"] == "done"
    review = run(as_the_person("review_workflow", name="tiles_only"))
    (rehearsal,) = review["rehearsals"]
    assert rehearsal["state"] == "done" and rehearsal["this_code"] and rehearsal["at"] > 0 and rehearsal["inputs"]["size_m"] == 5e-7
    assert rehearsal["outputs"] and rehearsal["problems"] == [] and review["can_approve"] and review["why_not"] is None
    reviewed = review["hash"]

    # The agent replaces the draft after the person opened it: what they read is not what would be approved.
    swapped = TEMPLATE.replace("max_visits=400", "max_visits=300")
    assert run(as_the_agent("propose_workflow", user_request=ASKED, source=swapped))["ok"]
    changed = run(as_the_person("approve_workflow", name="tiles_only", approver="Sam Reed", expected_hash=reviewed))
    assert not changed["ok"] and "changed since it was reviewed" in changed["error"] and library.find("tiles_only")[1] == "draft"
    changed = run(as_the_person("reject_workflow", name="tiles_only", reason="too big", expected_hash=reviewed))
    assert not changed["ok"] and "changed since it was reviewed" in changed["error"] and library.rejection("tiles_only") is None
    assert run(as_the_agent("propose_workflow", user_request=ASKED, source=TEMPLATE))["ok"]
    assert run(as_the_person("review_workflow", name="tiles_only"))["hash"] == reviewed

    approved = run(as_the_person("approve_workflow", name="tiles_only", approver="Sam Reed", instrument="tango://scope-a:9094", days=30,
                                 expected_hash=reviewed))
    assert approved["ok"] and approved["status"] == "approved"
    record = approved["approval"]
    assert record["by"] == "Sam Reed" and record["via"] == "Workflow_Operator_MCP" and record["instrument"] == "tango://scope-a:9094"
    assert record["rehearsed"]["run_id"] == rehearsal["run_id"] and record["expires"] and record["login"] and record["hash"] == reviewed
    assert library.find("tiles_only")[1] == "approved" and library.approval("tiles_only") == {k: v for k, v in record.items() if k != "name"}
    review = run(as_the_person("review_workflow", name="tiles_only"))
    assert review["status"] == "approved" and not review["changed_since_approval"] and "already approved" in review["why_not"]

    approved_file = tmp_path / "library" / "approved" / "tiles_only.py"
    approved_file.write_text(TEMPLATE.replace("max_visits=400", "max_visits=399"), encoding="utf-8")
    review = run(as_the_person("review_workflow", name="tiles_only"))
    assert review["status"] == "draft" and review["changed_since_approval"] and not review["rehearsed"] and not review["rehearsals"][0]["this_code"]
    approved_file.write_text(TEMPLATE, encoding="utf-8")

    assert "gives the reason" in run(as_the_person("reject_workflow", name="tiles_only", reason="  ", expected_hash=reviewed))["error"]
    rejected = run(as_the_person("reject_workflow", name="tiles_only", reason="the square is too large for this sample", expected_hash=reviewed))
    assert rejected["ok"] and rejected["rejection"]["status"] == "draft" and rejected["rejection"]["reason"] == "the square is too large for this sample"
    assert library.find("tiles_only")[1] == "draft" and library.approval("tiles_only") is None
    listed = {w["name"]: w for w in run(as_the_agent("list_workflows"))["workflows"]}["tiles_only"]
    assert listed["status"] == "draft" and listed["rejected"] == "the square is too large for this sample"
    review = run(as_the_person("review_workflow", name="tiles_only"))
    assert review["rejection"]["this_code"] and review["rejection"]["via"] == "Workflow_Operator_MCP" and review["rejection"]["hash"] == reviewed
    assert not run(as_the_person("reject_workflow", name="map_grid", reason="not wanted", expected_hash="x"))["ok"]
    # An approval after a rejection replaces it.
    assert run(as_the_person("approve_workflow", name="tiles_only", approver="sam", expected_hash=reviewed))["ok"] and library.rejection("tiles_only") is None

    # An approvers list holds here too: the login is the account the operator endpoint runs under.
    library.revoke("tiles_only")
    (tmp_path / "library" / "approvers.json").write_text('["somebody-else"]', encoding="utf-8")
    assert "approvers.json" in run(as_the_person("approve_workflow", name="tiles_only", approver="sam", expected_hash=reviewed))["error"]
    assert "approvers.json" in run(as_the_person("reject_workflow", name="tiles_only", reason="no", expected_hash=reviewed))["error"]


def test_the_workflow_operator_endpoint_binds_to_loopback_only(capsys):
    from asyncroscopy.mcp.agent_mcp_server import main

    assert main(["--name", "W", "--transport", "streamable-http", "--http-host", "0.0.0.0", "--http-port", "8004", "--endpoint-role", "operator"]) == 1
    assert "needs --http-host 127.0.0.1" in capsys.readouterr().out


def test_review_and_approve_agree_on_an_expired_approval(tmp_path, monkeypatch):
    from asyncroscopy.workflows import Library
    from asyncroscopy.workflows import server as workflow_server
    from asyncroscopy.workflows.authoring import TEMPLATE
    from asyncroscopy.workflows.server import WorkflowServer

    library = Library(tmp_path / "library")
    instrument, analysis, _state = twin()
    workflows = WorkflowServer(FakeEndpoints(instrument, analysis, True), state_dir=tmp_path / "state", library=library)
    tools = {fn.__name__: fn for fn in workflows.tools() + workflows.operator_tools()}
    assert run(tools["propose_workflow"](user_request=ASKED, source=TEMPLATE))["ok"]
    assert run(tools["run_workflow"](name="tiles_only", inputs={"size_m": 5e-7}))["state"] == "done"
    reviewed = workflows.review("tiles_only")["hash"]
    first = run(tools["approve_workflow"](name="tiles_only", approver="sam", expected_hash=reviewed, days=1))
    assert first["ok"] and first["approval"]["rehearsed"]["inputs"]["size_m"] == 5e-7

    review = workflows.review("tiles_only")
    again = run(tools["approve_workflow"](name="tiles_only", approver="sam", expected_hash=reviewed))
    assert not review["approval_expired"] and not review["can_approve"]
    assert not again["ok"] and again["error"] == review["why_not"] and library.approval("tiles_only")["expires"] == first["approval"]["expires"]

    later = time.time() + 2 * 86400.0
    monkeypatch.setattr(workflow_server.time, "time", lambda: later)
    review = workflows.review("tiles_only", run(workflows.stack_fingerprint()))
    assert review["approval_expired"] and review["can_approve"] and review["why_not"] is None
    renewed = run(tools["approve_workflow"](name="tiles_only", approver="sam", expected_hash=reviewed))
    assert renewed["ok"] and renewed["approval"]["expires"] is None


def _ask_between(ctx, args, previous):
    async def run_it():
        first = await ctx.call("Twin_get_stage")
        answer = ctx.ask({"id": "go", "question": "Go on?", "options": ["yes"]})
        return {"stage": first, "answer": answer, "again": await ctx.call("Twin_get_stage")}
    return run_it()


def _journal_workflow(name: str, always: tuple) -> Workflow:
    block = register_block(Block(name=f"test_{name}", summary="A read, a question, a read.", run=_ask_between,
                                 outputs={"stage": "any", "answer": "any", "again": "any"}, tools=("Twin_get_stage",), always=always))

    def build(use):
        graph = StateGraph(WorkflowState)
        graph.add_node("ask", use.block(block.name))
        graph.add_edge(START, "ask")
        graph.add_edge("ask", END)
        return graph

    return Workflow(name=name, summary="One node that asks between two reads.", inputs={}, budget={"max_tool_calls": 10}, build=build)


@pytest.mark.parametrize(("always", "sent"), [((), 2), (("Twin_blank_beam",), 3)])
def test_a_resumed_node_replays_its_completed_calls_unless_a_cleanup_ran_in_between(tmp_path, always, sent):
    instrument, _analysis, _ = twin()
    workflow = _journal_workflow(f"journal_{len(always)}", always)
    record = run(run_workflow(workflow, {}, instrument=instrument, state_dir=tmp_path))
    assert record["state"] == "waiting" and instrument.names().count("Twin_get_stage") == 1
    question = record["questions"][0]["id"]
    record = run(resume_workflow(record["run_id"], instrument=instrument, state_dir=tmp_path, find=lambda name: workflow,
                                 answers={question: "yes"}))
    assert record["state"] == "done", record
    assert instrument.names().count("Twin_get_stage") == sent
    assert not (tmp_path / "runs" / f"{record['run_id']}.journal.json").exists()


def test_the_review_shows_the_rehearsals_results_its_calls_and_what_changed_since_the_last_approval(tmp_path):
    from asyncroscopy.workflows import Library
    from asyncroscopy.workflows.authoring import TEMPLATE
    from asyncroscopy.workflows.server import WorkflowServer
    from asyncroscopy.workflows.view import render_html

    library = Library(tmp_path / "library")
    instrument, analysis, _ = twin()
    server = WorkflowServer(FakeEndpoints(instrument, analysis, True), state_dir=tmp_path / "state", library=library)
    review, _, _ = server.operator_tools()
    assert library.propose(TEMPLATE, request=ASKED, context="the sample is gold on carbon")["ok"]
    run_workflow_tool = {fn.__name__: fn for fn in server.tools()}["run_workflow"]
    record = run(run_workflow_tool("tiles_only", {"size_m": 5e-7}, 30.0))
    assert record["state"] == "done" and record["results"]["data_keys"] == [f"stem_image_HAADF_{n:04d}.h5" for n in range(1, 5)]
    card = run(review(name="tiles_only"))
    assert card["context_as_reported_by_agent"] == {"text": "the sample is gold on carbon", "this_code": True}
    rehearsal = card["rehearsal"]
    assert rehearsal["run_id"] == record["run_id"] and rehearsal["stack_fingerprint"]["digest"] == "twin-1"
    assert rehearsal["results"]["data_keys"] == record["results"]["data_keys"] and rehearsal["results"]["map"] is None
    score = rehearsal["results"]["twin_score"]
    assert score["known"] is False and "no operator endpoint" in score["why"] and rehearsal["results"]["dose"] is None
    moves = [c for c in rehearsal["calls_made"] if c["tool"] == "Twin_move_stage"]
    assert len(moves) == 5 and all(c["server"] == "instrument" and c["count"] >= 1 for c in moves)
    assert sum(c["count"] for c in rehearsal["calls_made"]) == record["calls"]
    assert card["rehearsals"][-1]["results"] == rehearsal["results"]
    page = render_html(library)
    assert "<b>Last rehearsal</b>: 4 data key(s)" in page and "no twin score: the stack reports no operator endpoint" in page

    library.approve("tiles_only", "sam")
    library.revoke("tiles_only")
    assert library.propose(TEMPLATE.replace("max_visits=400", "max_visits=300"), request=ASKED)["ok"]
    card = run(review(name="tiles_only"))
    assert card["last_approved"]["by"] == "sam" and card["last_approved"]["hash"] != card["hash"]
    assert "-    graph.add_conditional_edges(\"tile\", use.when(\"tile.remaining > 0\", then=\"tile\", otherwise=\"home\", max_visits=400))" in card["diff_from_last_approved"]
    assert "+    graph.add_conditional_edges(\"tile\", use.when(\"tile.remaining > 0\", then=\"tile\", otherwise=\"home\", max_visits=300))" in card["diff_from_last_approved"]
    assert card["rehearsal"] is None and card["context_as_reported_by_agent"] is None


def test_a_workflow_server_acts_only_on_the_instrument_endpoints_it_was_configured_with():
    endpoints = Endpoints("http://127.0.0.1:8000/mcp", instruments=["http://127.0.0.1:9000/mcp", "http://127.0.0.1:8000/mcp"])
    assert endpoints.instruments == ["http://127.0.0.1:8000/mcp", "http://127.0.0.1:9000/mcp"]
    assert endpoints.url(None) == endpoints.url("") == "http://127.0.0.1:8000/mcp"
    assert endpoints.url("http://127.0.0.1:9000/mcp") == "http://127.0.0.1:9000/mcp"
    with pytest.raises(LookupError, match="not an instrument endpoint this workflow server acts on"):
        endpoints.url("http://evil.example/mcp")
    assert endpoints.operator({"urls": {}}) is None
    assert endpoints.operator({"urls": {"operator": "http://127.0.0.1:8001/mcp"}}).url == "http://127.0.0.1:8001/mcp"


def test_a_twin_marker_is_not_a_run_and_goes_when_the_rehearsal_raises(tmp_path):
    from asyncroscopy.workflows import runner, signoff

    server, _, _ = served(tmp_path)
    run(signoff.mark_rehearsal(server.state_dir, "r1", None))
    marker = server.state_dir / "runs" / "r1.twin_marker.json"
    assert marker.exists()
    assert all(r["run_id"] for r in runner.list_runs(server.state_dir))

    async def raises():
        raise RuntimeError("the twin went away")

    with pytest.raises(RuntimeError):
        run(server._rehearsed("r1", {"simulated": True}, get_workflow("map_grid"), raises(), None, False))
    assert not marker.exists()
