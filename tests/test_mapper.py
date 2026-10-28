"""Mapper: the job envelope, plan checks, box and dose limits, abort, resume, and the run file."""

import json
import time

import h5py
import numpy as np
import pytest

from asyncroscopy.contracts import validate
from asyncroscopy.instruments.electron_microscope.mapper import tile_centres


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def wait(mapper, job_id, states=("done", "failed", "aborted"), timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = call(mapper, "job_status", {"job_id": job_id})
        if status["state"] in states:
            return status
        time.sleep(0.05)
    raise TimeoutError(status)


def release_all(mapper, reservation):
    call(mapper, "release", {"reservation_id": reservation})


N_TILES = len(tile_centres([-1.2e-7, 1.2e-7, -8e-8, 8e-8], 8e-8, 0.2, "serpentine"))
PLAN = {"box_m": [-1.2e-7, 1.2e-7, -8e-8, 8e-8], "fov_m": 8e-8, "overlap": 0.2, "imsize": 32, "dwell_s": 1e-6}


class TestTileCentres:
    def test_tiles_cover_box_and_stay_inside(self):
        tiles = tile_centres([-1e-6, 1e-6, -5e-7, 5e-7], 2e-7, 0.15, "raster")
        xs = sorted({t["x"] for t in tiles})
        ys = sorted({t["y"] for t in tiles})
        assert xs[0] == pytest.approx(-1e-6 + 1e-7) and xs[-1] == pytest.approx(1e-6 - 1e-7)
        assert ys[0] == pytest.approx(-5e-7 + 1e-7) and ys[-1] == pytest.approx(5e-7 - 1e-7)
        assert max(np.diff(xs)) <= 2e-7 * 0.85 + 1e-15
        assert len(tiles) == len(xs) * len(ys)

    def test_serpentine_reverses_every_other_row(self):
        tiles = tile_centres([0, 6e-7, 0, 4e-7], 2e-7, 0.0, "serpentine")
        rows = {}
        for t in tiles:
            rows.setdefault(t["row"], []).append(t["col"])
        assert rows[0] == sorted(rows[0])
        assert rows[1] == sorted(rows[1], reverse=True)

    def test_too_many_tiles_is_refused_before_any_is_built(self):
        started = time.perf_counter()
        with pytest.raises(ValueError, match="at most 5000"):
            tile_centres([0, 1e-4, 0, 1e-4], 1e-9, 0.1, "raster", limit=5000)  # about 1e10 tiles
        assert time.perf_counter() - started < 1.0

    def test_box_smaller_than_fov_is_refused(self):
        with pytest.raises(ValueError, match="smaller than the field of view"):
            tile_centres([0, 1e-7, 0, 1e-7], 2e-7, 0.1, "raster")


class TestEnvelope:
    def test_reserve_returns_estimate_and_validates(self, mapper_proxy):
        reply = call(mapper_proxy, "reserve", {"plan": PLAN, "caller": "test"})
        validate(reply, "envelope", "reserve_response")
        assert reply["estimate"]["steps"] == N_TILES == 4 * 3
        assert reply["estimate"]["duration_s"] > 0
        release_all(mapper_proxy, reply["reservation_id"])

    def test_bad_plan_is_rejected_with_reason(self, mapper_proxy):
        reply = call(mapper_proxy, "reserve", {"plan": {"box_m": [0, 1], "fov_m": 1e-7}})
        validate(reply, "envelope", "error_response")
        assert reply["error"]["code"] == "plan_rejected"

    def test_refusal_carries_an_example_plan(self, mapper_proxy):
        message = call(mapper_proxy, "reserve", {"plan": {"box_m": [0, 1], "fov_m": 1e-7}})["error"]["message"]
        example = json.loads(message.split("Example plan: ", 1)[1])
        validate(example, "map_plan")

    def test_size_plan_is_centred_on_the_stage(self, mapper_proxy, mapping_twin_proxy):
        x, y = list(mapping_twin_proxy.get_stage())[:2]
        reply = call(mapper_proxy, "reserve", {"plan": {"size_m": [4e-7, 2e-7], "fov_m": 1e-7}})
        box = reply["resolved_plan"]["box_m"]
        assert box == pytest.approx([x - 2e-7, x + 2e-7, y - 1e-7, y + 1e-7])
        release_all(mapper_proxy, reply["reservation_id"])
        reply = call(mapper_proxy, "reserve", {"plan": {"size_m": [4e-7, 2e-7], "centre_m": [1e-6, 0.0], "fov_m": 1e-7}})
        assert reply["resolved_plan"]["box_m"] == pytest.approx([8e-7, 1.2e-6, -1e-7, 1e-7])
        release_all(mapper_proxy, reply["reservation_id"])

    def test_box_and_size_together_are_refused(self, mapper_proxy):
        reply = call(mapper_proxy, "reserve", {"plan": {**PLAN, "size_m": [4e-7, 4e-7]}})
        assert reply["error"]["code"] == "plan_rejected" and "not both" in reply["error"]["message"]

    def test_dose_budget_rejects_plan(self, mapper_proxy):
        reply = call(mapper_proxy, "reserve", {"plan": {**PLAN, "dose_budget_e_per_a2": 1e-6}})
        assert reply["error"]["code"] == "plan_rejected"
        assert "budget" in reply["error"]["message"]

    def test_second_reservation_is_busy(self, mapper_proxy):
        first = call(mapper_proxy, "reserve", {"plan": PLAN})
        second = call(mapper_proxy, "reserve", {"plan": PLAN})
        assert second["error"]["code"] == "busy"
        release_all(mapper_proxy, first["reservation_id"])


class TestRun:
    def test_full_run_writes_every_tile_inside_the_box(self, mapper_proxy, mapping_twin_proxy):
        reserve = call(mapper_proxy, "reserve", {"plan": {**PLAN, "per_tile": ["eds_point"], "eds_exposure_s": 0.2}})
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        status = wait(mapper_proxy, started["job_id"])
        validate(status, "envelope", "status_response")
        assert status["state"] == "done", status
        results = call(mapper_proxy, "results", {"job_id": started["job_id"]})
        validate(results, "envelope", "results_response")
        with h5py.File(results["result_keys"]["run_file"], "r") as h5:
            tiles = h5["tiles"]
            assert len(tiles) == N_TILES
            for name in tiles:
                x, y = tiles[name].attrs["commanded_xy_m"]
                assert -1.2e-7 + 4e-8 - 1e-15 <= x <= 1.2e-7 - 4e-8 + 1e-15
                assert -8e-8 + 4e-8 - 1e-15 <= y <= 8e-8 - 4e-8 + 1e-15
                assert tiles[name].shape == (32, 32)
            assert len(h5["spectra"]) == N_TILES
            assert h5["spectra/spectrum_00000"].shape == (2048,)
            trace = [json.loads(line) for line in h5["trace"][()]]
            assert sum(1 for e in trace if e["action"] == "Mapper.step") == N_TILES
            assert json.loads(h5.attrs["plan"])["fov_m"] == pytest.approx(8e-8)
        assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True  # blanked when the job ends
        release_all(mapper_proxy, reserve["reservation_id"])

    def test_abort_stops_within_one_tile_and_blanks_beam_then_resume_finishes(self, mapper_proxy, mapping_twin_proxy, mapping_scan_proxy):
        plan = {**PLAN, "imsize": 256, "dwell_s": 2e-6}
        reserve = call(mapper_proxy, "reserve", {"plan": plan})
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        while call(mapper_proxy, "job_status", {"job_id": started["job_id"]})["progress"]["done"] < 1:
            time.sleep(0.01)
        aborted = call(mapper_proxy, "abort", {"job_id": started["job_id"]})
        assert aborted["state"] == "aborted"
        done_at_abort = aborted["progress"]["done"]
        assert done_at_abort < N_TILES
        assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True

        resumed = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        status = wait(mapper_proxy, resumed["job_id"])
        assert status["state"] == "done"
        assert status["progress"]["done"] == N_TILES
        with h5py.File(call(mapper_proxy, "results", {"job_id": resumed["job_id"]})["result_keys"]["run_file"], "r") as h5:
            assert len(h5["tiles"]) == N_TILES
        release_all(mapper_proxy, reserve["reservation_id"])

    def test_abort_cuts_a_long_eds_map_step_short(self, mapper_proxy, mapping_twin_proxy):
        # One tile with a 64 x 64 spectrum image: thousands of spectra in a single step.
        plan = {"size_m": [8e-8, 8e-8], "fov_m": 8e-8, "imsize": 32, "dwell_s": 1e-6, "per_tile": ["eds_map"], "eds_map_points": 64}
        reserve = call(mapper_proxy, "reserve", {"plan": plan})
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        time.sleep(1.0)
        call(mapper_proxy, "abort", {"job_id": started["job_id"]})
        status = wait(mapper_proxy, started["job_id"], states=("aborted", "done", "failed"), timeout=15)
        assert status["state"] == "aborted" and status["progress"]["done"] == 0
        assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True
        release_all(mapper_proxy, reserve["reservation_id"])

    def test_release_refuses_a_running_job(self, mapper_proxy):
        reserve = call(mapper_proxy, "reserve", {"plan": {**PLAN, "imsize": 256}})
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        refused = call(mapper_proxy, "release", {"reservation_id": reserve["reservation_id"]})
        assert refused["error"]["code"] == "bad_state"
        call(mapper_proxy, "abort", {"job_id": started["job_id"]})
        release_all(mapper_proxy, reserve["reservation_id"])
