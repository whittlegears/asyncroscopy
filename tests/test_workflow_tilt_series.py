"""The alpha-tilt diffraction series graph, run against the tilt digital twin over its Tango commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("langgraph.types")  # the stubs provide langgraph but not this

from asyncroscopy.agent_workflows import tilt_series  # noqa: E402
from asyncroscopy.agent_workflows.runner import WorkflowRunner  # noqa: E402
from asyncroscopy.instruments.electron_microscope.digital_twin_tilt import DigitalTwinTilt  # noqa: E402
from tests.workflow_tango_tools import TangoTools  # noqa: E402

START_STAGE = [2e-9, -1e-9, 0.0, 0.0, 0.0]
START_SHIFT = [3e-9, -4e-9]
START_FOV = 1100e-9
REQUEST = {"alpha_step_deg": 15.0, "max_alpha_deg": 30.0, "fov_m": 700e-9}
MOVES = ("move_stage", "set_image_shift", "set_fov", "acquire_scanned_image", "acquire_camera_image")


@pytest.fixture
def twin(monkeypatch, tilt_twin_proxy, scan_proxy, camera_proxy):
    """The tilt twin at a known start, with tiny images and no abTEM."""
    monkeypatch.setattr(DigitalTwinTilt, "_simulate_diffraction_from_atoms",
                        lambda self, atoms, imsize: np.ones((imsize, imsize), dtype=np.float32))
    scan_proxy.imsize = 64
    scan_proxy.dwell_time = 1e-6
    camera_proxy.imsize = 16
    tilt_twin_proxy.move_stage(START_STAGE)
    tilt_twin_proxy.set_image_shift(START_SHIFT)
    tilt_twin_proxy.set_fov(START_FOV)
    return tilt_twin_proxy


def _run(instrument: TangoTools, answer, request=None):
    async def go():
        runner = WorkflowRunner()
        runner.register("tilt_series", lambda cp: tilt_series.build_graph(instrument.tools(), checkpointer=cp))
        step = await runner.start("tilt_series", {"request": {**REQUEST, **(request or {})}})
        pauses = []
        while step["waiting_for"] is not None:
            pauses.append(step["waiting_for"])
            step = await runner.resume(step["thread_id"], answer)
        return step, pauses

    return asyncio.run(go())


def _assert_restored(twin):
    np.testing.assert_allclose(twin.get_stage(), START_STAGE, rtol=1e-6, atol=1e-15)
    np.testing.assert_allclose(twin.get_image_shift(), START_SHIFT, rtol=1e-6, atol=1e-15)
    assert twin.get_fov() == pytest.approx(START_FOV)


def test_plan_lists_pretilt_then_series_angles():
    plan = tilt_series.plan_request({"alpha_step_deg": 6.0, "max_alpha_deg": 30.0})
    assert plan["pretilt_deg"] == [-6.0, -12.0, -18.0, -24.0, -30.0]
    assert plan["series_deg"] == [-30.0 + 6.0 * i for i in range(11)]
    assert plan["patterns"] == 11 and plan["estimated_minutes"] > 0
    assert tilt_series.plan_request({"max_alpha_deg": 80})["problems"]
    assert tilt_series.plan_request({"alpha_step_deg": 0.2, "max_alpha_deg": 30})["problems"]  # too many steps


def test_full_run_returns_one_camera_key_per_angle_and_restores(twin):
    instrument = TangoTools(twin, "DigitalTwinTilt")
    assert tilt_series.missing_tools(instrument.tools(), set_fov=True) == []
    step, pauses = _run(instrument, {"approved": True})

    assert [p["kind"] for p in pauses] == ["approve_tilt_series"], step["state"].get("errors")
    assert pauses[0]["plan"]["series_deg"] == [-30.0, -15.0, 0.0, 15.0, 30.0]
    np.testing.assert_allclose(pauses[0]["start"]["stage"], START_STAGE, rtol=1e-6, atol=1e-15)
    report = step["state"]["report"]
    assert step["finished"] and report["status"] == "done", report["errors"]
    assert report["alpha_deg"] == [-30.0, -15.0, 0.0, 15.0, 30.0]
    assert len(set(report["camera_keys"])) == 5 and all(Path(k).is_file() for k in report["camera_keys"])
    assert report["restored"] is True
    # Tracking ran at every angle and moved the image shift away from the start.
    assert len(report["tracking"]) == 7 and all(t["shift_px"] is not None for t in report["tracking"])
    shifts = [arg for name, arg in instrument.called("set_image_shift")]
    assert any(not np.allclose(s, START_SHIFT) for s in shifts[:-1])
    # Without tracking the lamella drifts about 8 px by +30 deg at this size; tracked, the residual is one step's motion.
    assert max(abs(t["shift_px"][0]) for t in report["tracking"]) < 4.0
    # Each stage move only changes alpha; the last one restores the start.
    moves = [arg for name, arg in instrument.called("move_stage")]
    assert [m[3] for m in moves[:-1]] == [-15.0, -30.0, -30.0, -15.0, 0.0, 15.0, 30.0]
    assert all(np.allclose(np.delete(m, 3), np.delete(START_STAGE, 3)) for m in moves)
    _assert_restored(twin)


def test_declined_moves_nothing(twin):
    instrument = TangoTools(twin, "DigitalTwinTilt")
    step, pauses = _run(instrument, {"approved": False})

    assert len(pauses) == 1
    assert step["state"]["report"]["status"] == "declined"
    assert instrument.called(*MOVES) == []
    _assert_restored(twin)


def test_stage_not_at_alpha_zero_stops_before_asking(twin):
    twin.move_stage([0.0, 0.0, 0.0, 5.0, 0.0])
    instrument = TangoTools(twin, "DigitalTwinTilt")
    step, pauses = _run(instrument, {"approved": True})

    assert pauses == []
    report = step["state"]["report"]
    assert report["status"] == "failed" and "alpha to 0" in report["errors"][0]
    assert instrument.called(*MOVES) == []


def test_failure_mid_series_restores_stage_shift_and_fov(twin):
    instrument = TangoTools(twin, "DigitalTwinTilt", fail_on={"acquire_camera_image": 3})
    step, _ = _run(instrument, {"approved": True})

    report = step["state"]["report"]
    assert report["status"] == "failed"
    assert any("injected failure" in e for e in report["errors"])
    assert len(report["camera_keys"]) == 2 and report["restored"] is True
    assert report["tracking"][-1]["error"]
    _assert_restored(twin)


def test_lost_feature_stops_and_restores(twin):
    instrument = TangoTools(twin, "DigitalTwinTilt")
    step, _ = _run(instrument, {"approved": True}, request={"max_shift_fraction": 1e-6})

    report = step["state"]["report"]
    assert report["status"] == "failed" and "feature was lost" in report["errors"][0]
    assert report["camera_keys"] == []
    _assert_restored(twin)
