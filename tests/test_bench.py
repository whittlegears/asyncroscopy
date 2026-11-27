"""Benchmark tasks: the twin referees, and our baseline routines pass every task."""

import json

import numpy as np
import pytest

from asyncroscopy.bench.runner import leaderboard_markdown, run_baseline
from asyncroscopy.bench.tasks import TASKS, Env
from tests.test_autonomy import read_camera, read_h5


@pytest.fixture()
def env(tango_ctx, tilt_twin_proxy, scan_proxy, camera_proxy, corrector_proxy):
    scan_proxy.imsize = 64
    scan_proxy.dwell_time = 2e-6
    camera_proxy.imsize = 128
    corrector_proxy.command_inout("reconnect")
    yield Env(tilt_twin_proxy, corrector_proxy, fetch_image=read_h5, fetch_pattern=read_camera,
              corrector_name=tango_ctx.get_device_access("asyncroscopy/corrector/default"))
    corrector_proxy.command_inout("reconnect")
    corrector_proxy.set_aberrations_coeff_sim(json.dumps({}))
    tilt_twin_proxy.move_stage([0.0] * 5)
    tilt_twin_proxy.set_defocus(0.0)


def test_ground_truth_referee(env):
    t = env.truth()
    assert t["material_under_beam"] == "silicon"
    assert t["on_zone_alpha_deg"] == pytest.approx(-np.degrees(3.0173e-3), abs=0.01)


def test_doing_nothing_fails_every_task(env):
    for name in ("focus", "aberrations", "zone_axis", "recovery"):
        task = TASKS[name]
        state = task.setup(env, np.random.default_rng(1))
        assert not task.score(env, state)["pass"], name
        if name == "recovery":
            env.corrector.command_inout("reconnect")


@pytest.mark.parametrize("name", ["focus", "aberrations", "recovery"])
def test_baseline_passes(env, name):
    rows = run_baseline(env, [name], [3], log=lambda *_: None)
    assert rows and all(r["pass"] for r in rows), rows


def test_baseline_passes_zone_axis(env):
    pytest.importorskip("abtem")
    rows = run_baseline(env, ["zone_axis"], [3], log=lambda *_: None)
    assert rows[0]["pass"], rows
    assert "asyncroscopy baseline" in leaderboard_markdown(rows)
