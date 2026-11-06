"""One Mapper, many instruments: the instrument profiles (asyncroscopy/jobs/mapping/profile.py).

* an AFM (the YAML twin's afm.yaml, profile ``yaml``) maps a few tiles that stitch, score and
  revisit on pyTEMlib's map tools, called in this process, with the tip lifted for every move;
* plans over budget are refused at reserve before anything moves: dose on the STEM, time and
  tip force on the AFM;
* the mapping twin driven as an SEM (profile ``sem``: the beam commands, no EDS);
* a profile written and registered in this file maps the STEM twin as a plain camera.

The devices are the session's in-process Tango devices from conftest.
"""

import importlib.util
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango

from asyncroscopy.jobs.mapping.mapper import Mapper
from asyncroscopy.jobs.mapping.pipeline import Pipeline
from asyncroscopy.instruments.yaml_sim.spec import SimInstrument, load_spec, map_description, sample_spots
from asyncroscopy.jobs.mapping.profile import Cost, InstrumentProfile, MapContext, MapSettings, Setting, get_profile, register_profile

EXAMPLES = Path(__file__).resolve().parents[1] / "asyncroscopy" / "instruments" / "yaml_sim" / "examples"
needs_map_tools = pytest.mark.skipif(importlib.util.find_spec("pyTEMlib.mcp_server.map_tools") is None, reason="needs the pyTEMlib fork with map tools")
# 2 x 2 tiles of 1 um (128 px of 7.8 nm) over a 1.6 um square: neighbours overlap by 40 %.
AFM_PLAN = {"size_m": [1.6e-6, 1.6e-6], "centre_m": [0.0, 0.0], "fov_m": 1e-6, "overlap": 0.25}


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def run_job(proxy, plan, deadline_s=240):
    reserved = call(proxy, "reserve", {"plan": plan})
    assert "reservation_id" in reserved, reserved
    started = call(proxy, "start", {"reservation_id": reserved["reservation_id"]})
    deadline = time.monotonic() + deadline_s
    while True:
        status = call(proxy, "job_status", {"job_id": started["job_id"]})
        if status["state"] in {"done", "failed", "aborted"}:
            break
        assert time.monotonic() < deadline, f"the map did not finish within {deadline_s} s: {status}"
        time.sleep(0.05)
    results = call(proxy, "results", {"job_id": started["job_id"]})
    call(proxy, "release", {"reservation_id": reserved["reservation_id"]})
    return reserved, results


def device(name):
    return tango.Util.instance().get_device_by_name(name)


@pytest.fixture
def afm_log(monkeypatch, afm_twin_proxy):
    """Every setting change and command on the AFM twin, with whether the tip was engaged at that moment."""
    afm_twin_proxy.retract_tip()
    sim = device("asyncroscopy/afm/default").sim
    log = []
    real_set, real_run = sim.set, sim.run_command

    def logged_set(name, value):
        log.append(("set", name, value, bool(sim.values["tip_engaged"])))
        return real_set(name, value)

    def logged_run(name, argument=None):
        log.append(("run", name, None, bool(sim.values["tip_engaged"])))
        return real_run(name, argument)

    monkeypatch.setattr(sim, "set", logged_set)
    monkeypatch.setattr(sim, "run_command", logged_run)
    return log


@pytest.fixture
def analysis(monkeypatch):
    from tests.test_map_pipeline import InProcessAnalysis

    servers = []

    def fake(self, tool, arguments):
        if not servers:  # pyTEMlib may read local run files only under its roots: the run's folder
            servers.append(InProcessAnalysis([Path(arguments["run"]).parent]))
        return servers[0](self, tool, arguments)

    monkeypatch.setattr(Mapper, "analysis_url", "in-process")
    monkeypatch.setattr(Pipeline, "call", fake)
    return servers


@needs_map_tools
def test_an_afm_maps_stitches_and_revisits_with_the_tip_lifted_for_every_move(afm_mapper_proxy, afm_twin_proxy, afm_log, analysis):
    plan = {**AFM_PLAN, "settings": {"setpoint_v": 0.4}, "find_regions": 1, "revisit": {"fov_m": 5e-7}, "budgets": {"tip_force_nn": 10, "time_s": 600}}
    reserved, results = run_job(afm_mapper_proxy, plan)
    assert results["state"] == "done", results
    estimate = reserved["estimate"]
    assert estimate["tiles"] == 4 and estimate["phases"] == ["tiles", "stitch", "score", "revisit"]
    assert estimate["budgets"]["tip_force_nn"] == {"value": 8.0, "budget": 10, "units": "nN"}

    # The tip was lifted for every stage move and is lifted at the end; every tile approached once.
    moves = [entry for entry in afm_log if entry[0] == "set" and entry[1] in ("stage_x_m", "stage_y_m")]
    assert len(moves) >= 2 * 4 and not any(engaged for *_rest, engaged in moves)
    assert sum(1 for entry in afm_log if entry[:2] == ("run", "approach_tip")) >= 4
    assert ("set", "setpoint_v", 0.4, False) in afm_log
    assert afm_twin_proxy.tip_engaged is False

    # A stitchable height map: the run file's tiles carry their pixel size and stage position, and they stitch.
    summary = results["summary"]
    assert "stitch_error" not in summary and results["result_keys"]["map"]
    assert summary["stitching"]["tiles"] == 4 and summary["stitching"]["median_residual_px"] < 1.0
    with h5py.File(results["result_keys"]["run_file"], "r") as h5:
        assert h5.attrs["instrument_profile"] == "yaml" and h5["tiles"].shape == (4, 128, 128)
        assert h5["tiles"].attrs["pixel_size_m"] == pytest.approx(1e-6 / 128)
        commanded, reported = h5["commanded_xy_m"][()], h5["stage_reported_m"][()][:, :2]
        assert np.allclose(commanded, reported) and np.allclose(np.abs(commanded), 3e-7)
        assert 1e-8 < float(np.max(h5["tiles"][()])) < 1e-7  # heights in metres: the 20 nm bumps
    assert summary["regions"]["count"] == 1
    assert summary["revisits"]["imaged"] + summary["revisits"]["skipped"] == 1
    assert afm_twin_proxy.scan_size_m == pytest.approx(1e-6)  # the revisit's smaller scan was put back


def test_an_afm_plan_over_its_time_or_tip_force_budget_is_refused_before_anything_moves(afm_mapper_proxy, afm_twin_proxy, afm_log):
    before = (afm_twin_proxy.stage_x_m, afm_twin_proxy.stage_y_m, afm_twin_proxy.tip_engaged)
    slow = call(afm_mapper_proxy, "reserve", {"plan": {**AFM_PLAN, "stitch": False, "budgets": {"time_s": 5}}})
    assert slow["error"]["code"] == "plan_rejected" and "above the budget of 5 s" in slow["error"]["message"]
    hard = call(afm_mapper_proxy, "reserve", {"plan": {**AFM_PLAN, "stitch": False, "settings": {"setpoint_v": 1.0}, "budgets": {"tip_force_nn": 15}}})
    assert hard["error"]["code"] == "plan_rejected" and "up to 20 nN" in hard["error"]["message"] and "above the budget of 15 nN" in hard["error"]["message"]
    dose = call(afm_mapper_proxy, "reserve", {"plan": {**AFM_PLAN, "stitch": False, "dose_budget_e_per_a2": 10}})
    assert dose["error"]["code"] == "plan_rejected" and "'dose_budget_e_per_a2' not used" in dose["error"]["message"]
    assert afm_log == []  # nothing set, no command run: refused at reserve
    assert (afm_twin_proxy.stage_x_m, afm_twin_proxy.stage_y_m, afm_twin_proxy.tip_engaged) == before


def test_a_stem_plan_over_its_dose_budget_is_refused_before_anything_moves(mapper_proxy, mapping_twin_proxy):
    before = list(mapping_twin_proxy.get_stage())
    plan = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": 1e-7, "imsize": 64, "dwell_s": 2e-6, "stitch": False}
    for budget in ({"dose_budget_e_per_a2": 1e-3}, {"budgets": {"dose_e_per_a2": 1e-3}}):
        reply = call(mapper_proxy, "reserve", {"plan": {**plan, **budget}})
        assert reply["error"]["code"] == "plan_rejected" and "above the budget of 0.001 e/A^2" in reply["error"]["message"]
    force = call(mapper_proxy, "reserve", {"plan": {**plan, "budgets": {"tip_force_nn": 5}}})
    assert "unknown budget(s) ['tip_force_nn']" in force["error"]["message"] and "dose_e_per_a2" in force["error"]["message"]
    assert list(mapping_twin_proxy.get_stage()) == before


def test_the_stem_twin_maps_as_an_sem_without_eds(sem_mapper_proxy, mapping_twin_proxy):
    plan = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": 1e-7, "overlap": 0.0, "imsize": 32, "dwell_s": 1e-6, "detectors": ["haadf"]}
    eds = call(sem_mapper_proxy, "reserve", {"plan": {**plan, "per_tile": ["eds_point"]}})
    assert eds["error"]["code"] == "plan_rejected" and "not used when mapping with the 'sem' instrument profile" in eds["error"]["message"]
    reserved, results = run_job(sem_mapper_proxy, plan)
    assert results["state"] == "done", results
    assert reserved["estimate"]["max_dose_e_per_a2"] > 0 and "eds_dose_e_per_a2" not in reserved["estimate"]
    with h5py.File(results["summary"]["run_file"], "r") as h5:
        assert h5.attrs["instrument_profile"] == "sem" and h5["tiles"].shape[0] == reserved["estimate"]["tiles"] == 9
        assert "spectra" not in h5
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True


@register_profile
class PinholeCamera(InstrumentProfile):
    """A test's own instrument: a camera on a stage, driven through three commands, with a frame budget."""

    name = "test_pinhole_camera"
    doc = "a test camera: one frame per tile, a frames budget"
    plan_fields = frozenset({"imsize"})
    plan_defaults = {"imsize": 32}
    settings = {"scan_device_address": Setting(str, "asyncroscopy/scan/default", "SCAN settings device.")}
    needs = {"microscope_device_address": ("move_stage", "get_stage", "acquire_scanned_image")}

    def current_xy(self):
        return tuple(float(v) for v in list(self.instrument().get_stage())[:2])

    def tile_seconds(self, p):
        return 0.01

    def costs(self, p, grid, revisit, revisits, budgets):
        return {"frames": Cost(grid["rows"] * grid["cols"], "frames", why="one per tile")}, {}

    def setup(self, job):
        self.ctx.proxy(self.setting("scan_device_address")).imsize = int(job["resolved_plan"]["imsize"])
        job.setdefault("z", float(list(self.instrument().get_stage())[2]))

    def move(self, job, x, y):
        self.instrument().move_stage([x, y, job["z"], 0.0, 0.0])
        return True

    def stage_reported(self, job):
        return [float(v) for v in self.instrument().get_stage()]

    def acquire_tile(self, job, tile):
        return {"image_key": self.instrument().acquire_scanned_image(["haadf"]), "acquired": time.time()}


def test_a_profile_registered_by_a_test_maps_the_twin(monkeypatch, sem_mapper_proxy):
    assert get_profile("test_pinhole_camera") is PinholeCamera
    monkeypatch.setattr(device("asyncroscopy/mapper/sem"), "instrument_profile", "test_pinhole_camera")
    plan = {"box_m": [-1e-7, 1e-7, -1e-7, 1e-7], "fov_m": 1e-7, "overlap": 0.0, "stitch": False}
    refused = call(sem_mapper_proxy, "reserve", {"plan": {**plan, "budgets": {"frames": 3}}})
    assert "plan would give up to 4 frames (one per tile), above the budget of 3 frames" in refused["error"]["message"]
    reserved, results = run_job(sem_mapper_proxy, {**plan, "budgets": {"frames": 4}})
    assert results["state"] == "done", results
    with h5py.File(results["summary"]["run_file"], "r") as h5:
        assert h5.attrs["instrument_profile"] == "test_pinhole_camera" and h5["tiles"].shape == (4, 32, 32)


def test_a_yaml_instrument_is_described_for_the_mapper_from_its_file():
    afm = load_spec(EXAMPLES / "afm.yaml")
    described = map_description(afm, SimInstrument(afm), {"scan_size_m": 2e-6, "setpoint_v": 1.0})
    assert described["image"] == "acquire_topography" and described["tile_m"] == [2e-6, 2e-6]
    assert described["before_move"] == ["retract_tip"] and described["after_move"] == ["approach_tip"] and described["safe"] == ["retract_tip"]
    assert described["budgets"]["tip_force_nn"]["per_tile"] == 20.0 and described["stage"]["x"]["max"] == 0.002
    # The Raman example has no map: section: inferred, with a fixed field of view (its brightfield camera).
    raman = load_spec(EXAMPLES / "raman.yaml")
    inferred = map_description(raman, SimInstrument(raman))
    assert inferred["image"] == "acquire_brightfield_image" and inferred["field_of_view"] is None
    assert inferred["shape"] == [480, 640] and inferred["safe"] == ["laser_off"]


def test_spots_fixed_on_the_sample_agree_where_two_images_overlap():
    pixel = 1e-8
    left = sample_spots(7, 64, 64, pixel, 0.0, 0.0, 30.0, 2e-8, 1.0)
    right = sample_spots(7, 64, 64, pixel, 32 * pixel, 0.0, 30.0, 2e-8, 1.0)  # 32 pixels to the right
    assert np.allclose(left[:, 32:], right[:, :32]) and left.max() > 0.5


def test_a_beam_profile_waits_for_a_move_already_under_way_before_its_own():
    events = []

    class Microscope:
        def move_stage(self, position):
            events.append(("move", list(position)))

    class Context(MapContext):
        def wait_for_stage(self, job):
            events.append("wait")
            return True

    profile = get_profile("stem")(Context(MapSettings("m"), lambda address, timeout_s=None: Microscope(), lambda job: False))
    job = {"pose": {"z": 0.0, "alpha": 0.0, "beta": 0.0}}
    assert profile.move(job, 1e-7, 2e-7)
    assert events == ["wait", ("move", [1e-7, 2e-7, 0.0, 0.0, 0.0]), "wait"]
