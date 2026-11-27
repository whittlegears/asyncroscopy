"""Autonomous routines: the optimizer's safety net, focus, corrector workflow, whole session."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from asyncroscopy.autonomy.optimize import SearchConfig, search
from asyncroscopy.autonomy.routines import Imager, Resilient, find_regions, focus, measure_and_correct
from asyncroscopy.autonomy.session import SessionContext, SessionPlan, run_session
from asyncroscopy.health.monitor import HealthMonitor

# -- optimizer -----------------------------------------------------------------------


class Knob:
    """A setting with a smooth optimum and measurement noise."""

    def __init__(self, optimum, noise=0.0, flat=False, seed=0, fail_after=None):
        self.optimum = np.atleast_1d(np.asarray(optimum, dtype=float))
        self.value = np.zeros_like(self.optimum)
        self.noise, self.flat = noise, flat
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.fail_after = fail_after
        self.applied = []

    def apply(self, offset):
        self.value = np.asarray(offset, dtype=float)
        self.applied.append(self.value.copy())

    def measure(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ConnectionError("link lost")
        base = 1.0 if self.flat else 1.0 - float(np.sum((self.value - self.optimum) ** 2))
        return {"contrast": base + self.rng.normal(0, self.noise)}


def test_finds_1d_optimum_and_leaves_it_applied():
    k = Knob([0.6])
    res = search(k.apply, k.measure, "contrast", SearchConfig(step=0.3, components=1, max_iterations=10))
    assert res.status == "improved"
    assert abs(res.best_offset[0] - 0.6) < 0.05 and np.allclose(k.value, res.best_offset)


def test_finds_2d_optimum():
    k = Knob([0.2, -0.3], noise=0.001)
    res = search(k.apply, k.measure, "contrast", SearchConfig(step=0.2, components=2, max_iterations=12))
    assert res.status == "improved" and np.linalg.norm(np.array(res.best_offset) - [0.2, -0.3]) < 0.08


def test_noise_is_not_reported_as_progress():
    k = Knob([0.0], noise=0.01, flat=True, seed=4)
    res = search(k.apply, k.measure, "contrast", SearchConfig(step=0.3, components=1, min_gain=0.05))
    assert res.status == "no_improvement" and np.allclose(k.value, 0.0)


def test_hard_limit_is_never_exceeded():
    k = Knob([5.0])
    res = search(k.apply, k.measure, "contrast", SearchConfig(step=0.3, components=1, max_offset=0.5,
                                                              max_iterations=15))
    assert max(abs(a[0]) for a in k.applied) <= 0.5 + 1e-12
    assert res.best_offset[0] == pytest.approx(0.5, abs=0.01)


def test_failure_mid_search_restores_start():
    k = Knob([0.6], fail_after=4)
    res = search(k.apply, k.measure, "contrast", SearchConfig(step=0.3, components=1))
    assert res.status == "failed" and np.allclose(k.value, 0.0)


def test_find_regions_prefers_structure():
    img = np.random.default_rng(0).normal(0.5, 0.01, (128, 128))
    yy, xx = np.mgrid[:128, :128]
    img += 0.5 * np.exp(-((yy - 90) ** 2 + (xx - 30) ** 2) / 50)
    best = find_regions(img, n=2)[0]
    assert abs(best["row"] - 90) <= 16 and abs(best["col"] - 30) <= 16


# -- on the twins ----------------------------------------------------------------------


def read_h5(path):
    with h5py.File(Path(path), "r") as h5:
        group = h5["image"]
        return group[next(iter(group))][()]


@pytest.fixture()
def tilt(tilt_twin_proxy, scan_proxy):
    scan_proxy.imsize = 64
    scan_proxy.dwell_time = 2e-6
    tilt_twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    tilt_twin_proxy.set_defocus(0.0)
    yield tilt_twin_proxy
    tilt_twin_proxy.set_defocus(0.0)


def test_focus_on_tilt_twin_finds_true_focus(tilt):
    tilt.set_defocus(45e-9)                               # start well out of focus
    imager = Imager(tilt, read_h5, Resilient())
    res = focus(imager, mode="defocus", step_m=15e-9, max_offset_m=90e-9, score="edge_sharpness")
    assert res["status"] == "improved", res["reasons"]
    assert abs(tilt.get_defocus()) < 12e-9               # true focus is stage z = 0


@pytest.fixture()
def corrector(tango_ctx, corrector_proxy):
    corrector_proxy.command_inout("reconnect")
    corrector_proxy.set_aberrations_coeff_sim(json.dumps({"C1": [40e-9], "A1": [12e-9, -8e-9]}))
    yield tango_ctx.get_device_access("asyncroscopy/corrector/default"), corrector_proxy
    corrector_proxy.command_inout("reconnect")
    corrector_proxy.set_aberrations_coeff_sim(json.dumps({}))


def test_measure_and_correct_brings_aberrations_below_threshold(corrector):
    _, proxy = corrector
    res = measure_and_correct(proxy, thresholds_m={"C1": 2e-9, "A1": 2e-9})
    assert res["status"] == "corrected", res
    final = res["final"]
    assert abs(final["C1"][0]) <= 2e-9 and np.hypot(*final["A1"]) <= 2e-9


def test_measure_and_correct_survives_link_drop(corrector):
    access, proxy = corrector
    import tango
    fresh = tango.DeviceProxy(access)
    fresh.command_inout("inject_link_drop")
    resilient = Resilient(HealthMonitor([access]))
    res = measure_and_correct(fresh, thresholds_m={"C1": 2e-9}, resilient=resilient)
    assert res["status"] == "corrected"
    assert res["recoveries"] and res["recoveries"][0]["recovery"] == "recovered"


class WrongSignCorrector:
    """A backend whose correct_aberration adds instead of subtracting."""

    def __init__(self):
        self.ab = {"C1": [20e-9]}

    def command_inout(self, cmd, arg=None):
        if cmd == "acquire_tableau":
            return json.dumps({"result": {"aberrations": self.ab}})
        name, *vals = arg.split()
        self.ab[name] = [a + float(v) for a, v in zip(self.ab[name], vals)]
        return "{}"


def test_correction_that_makes_things_worse_is_undone_and_frozen():
    c = WrongSignCorrector()
    res = measure_and_correct(c, thresholds_m={"C1": 2e-9})
    assert res["frozen"] == ["C1"] and res["status"] == "partial"
    assert c.ab["C1"][0] == pytest.approx(20e-9)          # left where it started


def test_whole_session_on_twin(tilt, corrector, tmp_path):
    from asyncroscopy.atlas.store import Atlas
    from asyncroscopy.atlas.tango_backend import TangoMicroscope

    access, proxy = corrector
    tilt.set_defocus(40e-9)
    mic = TangoMicroscope(tilt, None, read_h5)
    imager = Imager(tilt, read_h5, Resilient(HealthMonitor([access])))
    atlas = Atlas(tmp_path)
    plan = SessionPlan(name="lamella check", find_region=False, focus_step_m=15e-9, focus_max_offset_m=90e-9,
                       score="edge_sharpness", thresholds_m={"C1": 2e-9, "A1": 2e-9})
    report = run_session(SessionContext(imager, mic=mic, corrector=proxy, monitor=HealthMonitor([access]),
                                        atlas=atlas), plan)
    assert report["status"] == "completed", report["failed_stages"]
    statuses = {s["stage"]: s["status"] for s in report["stages"]}
    assert statuses["focus"] == "improved" and statuses["aberrations"] == "corrected"
    assert report["final"]["relative_change"]["edge_sharpness"] > 0.2
    assert report["site_id"] in atlas.sites
