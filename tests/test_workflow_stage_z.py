"""The stage-Z focus LangGraph workflow, through the real instrument MCP server over the digital twin.

``get_stage`` and ``move_stage`` are the twin's real commands reached through
the MCP tool wrappers. The image and its score are synthetic (see
tests/workflow_twin.py): sharpness peaks at ``Z_BEST`` as read back from the
stage device, so the climb must find it by moving the stage.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

pytest.importorskip("langgraph.types")  # the stubs provide langgraph but not this

from fastmcp import Client  # noqa: E402

from asyncroscopy.agent_workflows import climb as maths  # noqa: E402
from asyncroscopy.agent_workflows import stage_z_focus  # noqa: E402
from tests.workflow_twin import SyntheticImages, client_tools, mcp_server, run_workflow  # noqa: E402

START = [1e-6, -2e-6, 0.0, 0.0, 0.0]
Z_BEST = 0.35e-6
WIDTH = 0.4e-6  # of the sharpness peak


@pytest.fixture
def stage(twin_proxy):
    twin_proxy.move_stage(START)
    yield twin_proxy
    twin_proxy.move_stage([0.0] * 5)


@pytest.fixture
def server(tango_ctx, monkeypatch):
    return mcp_server(tango_ctx, monkeypatch, ["asyncroscopy/digitaltwin/default"])


def _z(stage):
    return float(stage.get_stage()[2])


def _run(server, stage, answer, request=None, fail_on=None, read_image=None, z_best=Z_BEST):
    """Run the workflow over MCP; return (final step, pauses, logged MCP calls, synthetic images)."""
    images = SyntheticImages(lambda: (_z(stage) - z_best) / WIDTH, fail_on=fail_on)
    log: list = []

    async def go():
        async with Client(server.mcp) as client:
            names = [t.name for t in await client.list_tools()]
            names = [n for n in names if not n.endswith(("_acquire_scanned_image", "_resolution"))]
            tools = {**client_tools(client, names, log), **images.tools()}
            assert stage_z_focus.missing_tools(tools) == []
            build = lambda cp: stage_z_focus.build_graph(tools, checkpointer=cp, read_image=read_image)  # noqa: E731
            return await run_workflow("stage_z_focus", build, {"request": request or {}}, answer)

    step, pauses = asyncio.run(go())
    return step, pauses, log, images


def _moves(log):
    return [args["position"] for name, args in log if name.endswith("_move_stage")]


def test_climb_finds_the_sharpest_z_and_stays_there(server, stage):
    step, pauses, log, _ = _run(server, stage, lambda pause: {"approved": True}, {"max_iterations": 12})

    report = stage_z_focus.describe(step["state"])
    assert step["finished"] and report["status"] == "done", report["errors"]
    assert len(pauses) == 1 and pauses[0]["kind"] == "approve_stage_z_climb"
    plan = pauses[0]["plan"]
    assert plan["start_z_m"] == 0.0 and plan["z_limits_m"] == pytest.approx([-2e-6, 2e-6])
    assert plan["max_measurements"] == 3 + 12
    assert report["final_z_m"] == pytest.approx(Z_BEST, abs=0.05e-6)
    assert _z(stage) == pytest.approx(report["final_z_m"])
    assert report["improvement"] > 1.4  # the start scores exp(-0.5 * (0.35 / 0.4) ** 2)
    # Only z ever changes: every move is the starting vector with a new z.
    for position in _moves(log):
        assert position[:2] == pytest.approx(START[:2]) and position[3:] == pytest.approx(START[3:])
    assert [h["trial"] for h in report["history"][:3]] == ["baseline", "initial -", "initial +"]


def test_first_model_step_is_the_shared_maths(server, stage):
    step, _, _, _ = _run(server, stage, lambda pause: True, {"max_iterations": 1})

    history = step["state"]["report"]["history"]
    explored = history[:3]
    expected = maths.propose_next_z([h["z_m"] for h in explored], [h["score"] for h in explored], 0.1e-6, 0.0, 2e-6)
    assert history[3]["z_m"] == pytest.approx(expected)
    assert step["state"]["report"]["stop_reason"] == "max_iterations"


def test_declined_plan_never_moves_the_stage(server, stage):
    step, pauses, log, images = _run(server, stage, lambda pause: {"approved": False})

    assert step["state"]["report"]["status"] == "declined"
    assert len(pauses) == 1 and _moves(log) == [] and images.keys == {}
    assert stage.get_stage().tolist() == pytest.approx(START)


def test_hard_limit_holds_when_the_peak_is_outside_it(server, stage):
    request = {"step_m": 0.1e-6, "max_offset_m": 0.3e-6, "max_iterations": 10}
    step, _, log, _ = _run(server, stage, lambda pause: True, request, z_best=1.5e-6)

    report = step["state"]["report"]
    assert report["status"] == "done"
    assert all(abs(p[2]) <= 0.3e-6 + 1e-15 for p in _moves(log))
    assert report["final_z_m"] == pytest.approx(0.3e-6, abs=0.02e-6)  # as close to the peak as allowed


def test_failure_mid_climb_moves_back_to_the_start(server, stage):
    step, _, log, _ = _run(server, stage, lambda pause: True, {"max_iterations": 6}, fail_on=5)

    report = step["state"]["report"]
    assert report["status"] == "failed" and "scan timed out" in report["errors"][0]
    assert report["restored_to_start"] and report["final_z_m"] == pytest.approx(0.0)
    assert _moves(log)[-1] == pytest.approx(START)
    assert stage.get_stage().tolist() == pytest.approx(START)
    assert report["measurements"] == 4


def test_bad_request_is_refused_before_anything_moves(server, stage):
    step, pauses, log, _ = _run(server, stage, lambda pause: True, {"step_m": 1e-6, "max_offset_m": 0.5e-6})

    report = step["state"]["report"]
    assert report["status"] == "failed" and pauses == []
    assert "step_m must not exceed max_offset_m" in report["errors"][0]
    assert [name.split("_", 1)[1] for name, _ in log] == ["get_stage"]


def test_pixel_score_with_read_image(server, stage):
    pattern = np.indices((16, 16)).sum(axis=0) % 2  # a small checkerboard

    def read_image(key):
        z = float(stage.get_stage()[2])  # read at scoring time, which follows the acquisition
        return 1.0 + pattern * math.exp(-0.5 * ((z - Z_BEST) / WIDTH) ** 2)

    step, pauses, _, _ = _run(server, stage, lambda pause: True, {"max_iterations": 10}, read_image=read_image)

    report = step["state"]["report"]
    assert pauses[0]["plan"]["metric"] == "contrast" == report["metric"]
    assert report["status"] == "done"
    assert report["final_z_m"] == pytest.approx(Z_BEST, abs=0.05e-6)


def test_pixel_metric_without_read_image_is_refused():
    plan = stage_z_focus.plan_request({"metric": "edge_sharpness"}, [0.0] * 5)
    assert "needs the pixels" in plan["problems"][0]
    assert stage_z_focus.missing_tools({"X_get_stage": None}) == ["move_stage", "acquire_scanned_image", "resolution"]
    assert set(stage_z_focus.WORKFLOW["inputs"]["request"]) == set(stage_z_focus.StageZRequest.__annotations__)
