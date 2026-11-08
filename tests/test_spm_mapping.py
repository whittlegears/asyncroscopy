"""One Mapper maps a real ``SPMMicroscope``: the tip is retracted for every stage move, approached after it, and every
frame goes through the SCAN device. The devices are the session's simulated SPM (instruments/scanning_probe_microscope/simulated.py).
"""

import json
import time

import h5py
import numpy as np
import pytest
import tango

from asyncroscopy.jobs.mapping.mapper import Mapper
from asyncroscopy.jobs.mapping.pipeline import Pipeline
from asyncroscopy.jobs.mapping.profile import get_profile
from asyncroscopy.instruments.scanning_probe_microscope.map_profile import SPMProfile
from tests.test_map_pipeline import InProcessAnalysis, ToolService

PLAN = {"size_m": [1.6e-6, 1.6e-6], "centre_m": [0.0, 0.0], "fov_m": 1e-6, "overlap": 0.25, "imsize": 64, "scan_rate_hz": 50.0}


def device(name):
    return tango.Util.instance().get_device_by_name(name)


@pytest.fixture(scope="module")
def mapper(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/spm"))


@pytest.fixture(scope="module")
def spm(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/spm/default"))


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def run_job(proxy, plan):
    reserved = call(proxy, "reserve", {"plan": plan})
    assert "reservation_id" in reserved, reserved
    started = call(proxy, "start", {"reservation_id": reserved["reservation_id"]})
    deadline = time.monotonic() + 120
    while True:
        status = call(proxy, "job_status", {"job_id": started["job_id"]})
        if status["state"] in {"done", "failed", "aborted"}:
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.05)
    results = call(proxy, "results", {"job_id": started["job_id"]})
    call(proxy, "release", {"reservation_id": reserved["reservation_id"]})
    return reserved, results


def test_the_spm_profile_comes_from_the_entry_point():
    assert get_profile("spm") is SPMProfile


def test_an_spm_maps_with_the_tip_lifted_for_every_move_and_stitches(mapper, spm, monkeypatch):
    servers = []

    def fake(_pipeline, tool, arguments):
        if not servers:
            from pathlib import Path

            servers.append(InProcessAnalysis([Path(arguments["run"]).parent]))
        return servers[0](None, tool, arguments)

    monkeypatch.setattr(Mapper, "analysis_url", "in-process")
    monkeypatch.setattr(Pipeline, "service", lambda self: ToolService(fake, self))
    approach = device("asyncroscopy/approach/spm")
    before = (approach.approaches, approach.retracts)
    reserved, results = run_job(mapper, PLAN)
    assert results["state"] == "done", results
    tiles = reserved["estimate"]["tiles"]
    assert tiles == 4
    # move_stage refuses with the tip down and a frame refuses with it up, so a finished map proves the order.
    assert approach.approaches - before[0] == tiles and approach.retracts - before[1] >= 2 * tiles
    assert not approach._approached
    with h5py.File(results["summary"]["run_file"], "r") as h5:
        assert h5.attrs["instrument_profile"] == "spm" and h5["tiles"].shape == (4, 64, 64)
        assert h5["tiles"].attrs["pixel_size_m"] == pytest.approx(1e-6 / 64) and h5["tiles"].attrs["detector"] == "HEIGHT"
        assert np.allclose(h5["commanded_xy_m"][()], h5["stage_reported_m"][()][:, :2]) and float(np.ptp(h5["tiles"][0])) > 0
    assert "stitch_error" not in results["summary"] and results["result_keys"]["map"]


def test_a_dry_run_rehearses_the_moves_and_really_retracts(mapper):
    approach = device("asyncroscopy/approach/spm")
    before = approach.approaches
    stage = device("asyncroscopy/stage/spm")
    where = list(stage._xy)
    _, results = run_job(mapper, {**PLAN, "dry_run": True})
    assert results["state"] == "done", results
    assert approach.approaches == before and stage._xy == where  # recorded, not performed
    assert not approach._approached


def test_the_spm_profile_refuses_another_instrument(mapper, monkeypatch):
    target = device("asyncroscopy/mapper/spm")
    monkeypatch.setattr(target, "microscope_device_address", "asyncroscopy/digitaltwin/mapping")
    target._map_profile = None
    try:
        refused = call(mapper, "reserve", {"plan": PLAN})
    finally:
        monkeypatch.undo()
        target._map_profile = None
    assert refused["error"]["code"] == "plan_rejected" and "drives SPM instruments" in refused["error"]["message"]


def test_a_stage_move_with_the_tip_down_is_still_refused_by_the_instrument(spm):
    spm.approach()
    try:
        with pytest.raises(tango.DevFailed, match="retract it first"):
            spm.move_stage([0.0, 0.0])
    finally:
        spm.retract()
