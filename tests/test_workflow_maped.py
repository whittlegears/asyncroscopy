"""The MAPED beam-tilt series graph, run against the tilt digital twin over its Tango commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("langgraph.types")  # the stubs provide langgraph but not this

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from asyncroscopy.agent_workflows import maped  # noqa: E402
from asyncroscopy.agent_workflows.runner import WorkflowRunner  # noqa: E402
from asyncroscopy.instruments.electron_microscope.digital_twin_tilt import DigitalTwinTilt  # noqa: E402
from tests.workflow_tango_tools import TangoTools  # noqa: E402

START_TILT = [1e-3, -2e-3]
REQUEST = {"radius_rad": 20e-3}
TILTS = [[0.0, 0.0], [20e-3, 0.0], [0.0, 20e-3], [-20e-3, 0.0], [0.0, -20e-3]]


@pytest.fixture
def twin(monkeypatch, tilt_twin_proxy, scan_proxy):
    """The tilt twin with a 4 x 4 scan, 16 x 16 patterns and no abTEM."""
    monkeypatch.setattr(DigitalTwinTilt, "_simulate_diffraction_from_atoms",
                        lambda self, atoms, imsize: np.ones((imsize, imsize), dtype=np.float32))
    scan_proxy.imsize = 4
    scan_proxy.dwell_time = 1e-6
    tilt_twin_proxy.move_stage([0.0] * 5)
    tilt_twin_proxy.set_beam_tilt(START_TILT)
    return tilt_twin_proxy


def _run(tools, answer, request=None, analyse=None):
    async def go():
        runner = WorkflowRunner()
        runner.register("maped", lambda cp: maped.build_graph(tools, checkpointer=cp, analyse=analyse))
        step = await runner.start("maped", {"request": {**REQUEST, **(request or {})}})
        pauses = []
        while step["waiting_for"] is not None:
            pauses.append(step["waiting_for"])
            step = await runner.resume(step["thread_id"], answer)
        return step, pauses

    return asyncio.run(go())


def test_plan_has_the_notebook_tilts_and_refuses_large_ones():
    plan = maped.plan_request({})
    assert plan["tilts_mrad"] == [[0, 0], [30, 0], [0, 30], [-30, 0], [0, -30]] and plan["scans"] == 5
    assert maped.plan_request({"radius_rad": 0.2})["problems"]
    assert maped.plan_request({"tilts_rad": [[0.0]]})["problems"]
    assert maped.plan_request({"tilts_rad": [[1e-3, 2e-3]]})["tilts_rad"] == [[1e-3, 2e-3]]


def test_full_run_returns_one_key_per_tilt_and_restores(twin):
    instrument = TangoTools(twin, "DigitalTwinTilt")
    assert maped.missing_tools(instrument.tools()) == []
    step, pauses = _run(instrument.tools(), {"approved": True})

    assert [p["kind"] for p in pauses] == ["approve_maped"]
    assert pauses[0]["plan"]["tilts_mrad"] == [[0, 0], [20, 0], [0, 20], [-20, 0], [0, -20]]
    assert pauses[0]["start_tilt_mrad"] == pytest.approx([1.0, -2.0])
    report = step["state"]["report"]
    assert step["finished"] and report["status"] == "done", report["errors"]
    assert len(set(report["keys"])) == 5 and report["restored"] is True
    for key, tilt in zip(report["keys"], TILTS):
        with h5py.File(key, "r") as h5:
            assert h5["stem_data"].shape == (4, 4, 16, 16)
            assert np.allclose(np.asarray(json.loads(h5["stem_data"].attrs["beam_tilt_mrad"])), np.asarray(tilt) * 1e3)
    sets = [arg for _name, arg in instrument.called("set_beam_tilt")]
    assert np.allclose(sets[:-1], TILTS) and np.allclose(sets[-1], START_TILT)
    np.testing.assert_allclose(twin.get_beam_tilt(), START_TILT)


def test_declined_changes_nothing(twin):
    instrument = TangoTools(twin, "DigitalTwinTilt")
    step, pauses = _run(instrument.tools(), {"approved": False})

    assert len(pauses) == 1 and step["state"]["report"]["status"] == "declined"
    assert instrument.called("set_beam_tilt", "acquire_scanned_data_advanced") == []
    np.testing.assert_allclose(twin.get_beam_tilt(), START_TILT)


def test_failure_mid_series_restores_the_tilt(twin):
    instrument = TangoTools(twin, "DigitalTwinTilt", fail_on={"acquire_scanned_data_advanced": 3})
    step, _ = _run(instrument.tools(), {"approved": True})

    report = step["state"]["report"]
    assert report["status"] == "failed" and any("injected failure" in e for e in report["errors"])
    assert len(report["keys"]) == 2 and report["restored"] is True
    assert report["scans"][-1]["error"]
    np.testing.assert_allclose(twin.get_beam_tilt(), START_TILT)


def test_optional_analysis_gets_the_keys(twin):
    def mean_pattern(keys, tilts):
        cubes = [np.asarray(h5py.File(k, "r")["stem_data"][()]) for k in keys]
        return {"n": len(cubes), "tilts": tilts, "mean_dp_max": float(np.mean(cubes, axis=0).mean(axis=(0, 1)).max())}

    step, _ = _run(TangoTools(twin, "DigitalTwinTilt").tools(), {"approved": True}, analyse=mean_pattern)

    report = step["state"]["report"]
    assert report["status"] == "done" and report["analysis"]["n"] == 5
    assert report["analysis"]["tilts"] == TILTS


class Crash(BaseException):
    """Stands in for the process dying: not an Exception, so no node catches it."""


def test_checkpointer_resumes_after_a_crash_without_repeating_scans(twin):
    instrument = TangoTools(twin, "DigitalTwinTilt")
    tools = instrument.tools()
    acquire = tools["DigitalTwinTilt_acquire_scanned_data_advanced"]
    crashed = []

    async def acquire_once_crashing():
        if len(instrument.called("acquire_scanned_data_advanced")) == 2 and not crashed:
            crashed.append(True)
            raise Crash()
        return await acquire()

    tools["DigitalTwinTilt_acquire_scanned_data_advanced"] = acquire_once_crashing
    graph = maped.build_graph(tools, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "crash"}}

    async def go():
        await graph.ainvoke({"request": REQUEST}, config)  # stops at the approval
        with pytest.raises(Crash):
            await graph.ainvoke(Command(resume={"approved": True}), config)
        return await graph.ainvoke(None, config)  # resume from the last checkpoint

    result = asyncio.run(go())
    report = maped.describe(result)
    assert report["status"] == "done" and len(report["keys"]) == 5
    assert len(instrument.called("acquire_scanned_data_advanced")) == 5  # the two before the crash were kept
    np.testing.assert_allclose(twin.get_beam_tilt(), START_TILT)
    assert all(Path(k).is_file() for k in report["keys"])
