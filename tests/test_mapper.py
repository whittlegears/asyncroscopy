"""Mapper: the job envelope, plan checks, box and dose limits, abort, resume, and the run file."""

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango

from types import SimpleNamespace

from asyncroscopy.contracts import validate
from asyncroscopy.jobs.mapping import mapper as mapper_module
from asyncroscopy.jobs.mapping.mapper import Mapper, tile_centres


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
            assert h5.attrs["run_layout"] == 2
            assert h5["tiles"].shape == (N_TILES, 32, 32) and h5["tiles"].chunks == (1, 32, 32)
            assert h5["written"][()].all()
            for x, y in h5["commanded_xy_m"][()]:
                assert -1.2e-7 + 4e-8 - 1e-15 <= x <= 1.2e-7 - 4e-8 + 1e-15
                assert -8e-8 + 4e-8 - 1e-15 <= y <= 8e-8 - 4e-8 + 1e-15
            assert np.isfinite(h5["acquired_s"][()]).all() and np.isfinite(h5["stage_reported_m"][()]).all()
            assert h5["spectra"].shape == (N_TILES, 2048)
            assert all(k for k in h5["spectrum_keys"][()])
            # The trace keeps the job's own events; each tile's position and time are in the arrays above.
            trace = [json.loads(line) for line in h5["trace"][()]]
            assert any(e["action"] == "Mapper.reserve" for e in trace)
            assert not any(e["action"] == "Mapper.step" for e in trace)
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
        # Until the job finishes the run file is .partial, so DATA's reconcile cannot register it half-written.
        working = Path(call(mapper_proxy, "results", {"job_id": started["job_id"]})["result_keys"]["run_file"])
        assert working.name.endswith(".h5.partial") and working.exists()

        resumed = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        status = wait(mapper_proxy, resumed["job_id"])
        assert status["state"] == "done"
        assert status["progress"]["done"] == N_TILES
        finished = Path(call(mapper_proxy, "results", {"job_id": resumed["job_id"]})["result_keys"]["run_file"])
        assert finished.suffix == ".h5" and not working.exists()
        with h5py.File(finished, "r") as h5:
            assert h5["written"][()].all() and len(h5["tiles"]) == N_TILES
        release_all(mapper_proxy, reserve["reservation_id"])

    def test_abort_cuts_a_long_eds_map_step_short(self, mapper_proxy, mapping_twin_proxy):
        # Tiles with a 64 x 64 spectrum image each: thousands of spectra in a single step.
        plan = {"size_m": [1.2e-7, 1.2e-7], "fov_m": 8e-8, "imsize": 32, "dwell_s": 1e-6, "per_tile": ["eds_map"], "eds_map_points": 64}
        reserve = call(mapper_proxy, "reserve", {"plan": plan})
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        time.sleep(1.0)
        call(mapper_proxy, "abort", {"job_id": started["job_id"]})
        status = wait(mapper_proxy, started["job_id"], states=("aborted", "done", "failed"), timeout=15)
        assert status["state"] == "aborted" and status["progress"]["done"] == 0
        assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True
        release_all(mapper_proxy, reserve["reservation_id"])

    def test_a_tile_as_large_as_the_area_is_refused_with_the_way_to_image_one_place(self, mapper_proxy):
        refused = call(mapper_proxy, "reserve", {"plan": {"size_m": [1e-6, 1e-6], "fov_m": 1e-6}})
        assert refused["error"]["code"] == "plan_rejected"
        assert "one tile" in refused["error"]["message"] and "revisit" in refused["error"]["message"]
        strip = call(mapper_proxy, "reserve", {"plan": {"box_m": [0, 6e-7, 0, 1.6e-7], "fov_m": 1.6e-7, "imsize": 64}})
        assert strip["estimate"]["steps"] > 1, strip  # a strip one tile high is a map
        release_all(mapper_proxy, strip["reservation_id"])

    def test_releasing_an_aborted_run_keeps_its_tiles_under_the_final_name(self, mapper_proxy, mapping_twin_proxy):
        reserve = call(mapper_proxy, "reserve", {"plan": {**PLAN, "imsize": 256, "dwell_s": 2e-6}})
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        while call(mapper_proxy, "job_status", {"job_id": started["job_id"]})["progress"]["done"] < 1:
            time.sleep(0.01)
        call(mapper_proxy, "abort", {"job_id": started["job_id"]})
        working = Path(call(mapper_proxy, "results", {"job_id": started["job_id"]})["result_keys"]["run_file"])
        assert working.name.endswith(".h5.partial")

        release_all(mapper_proxy, reserve["reservation_id"])

        final = working.with_name(working.name[: -len(".partial")])
        assert final.exists() and not working.exists()
        with h5py.File(final, "r") as h5:
            assert len(h5["tiles"]) >= 1

    def test_release_refuses_a_running_job(self, mapper_proxy):
        reserve = call(mapper_proxy, "reserve", {"plan": {**PLAN, "imsize": 256}})
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        refused = call(mapper_proxy, "release", {"reservation_id": reserve["reservation_id"]})
        assert refused["error"]["code"] == "bad_state"
        call(mapper_proxy, "abort", {"job_id": started["job_id"]})
        release_all(mapper_proxy, reserve["reservation_id"])


class TestRunFileName:
    """A finished run file gets its final name even when Windows keeps the .partial open."""

    def stub(self, tmp_path):
        stub = SimpleNamespace()
        stub._run_path = lambda job: Mapper._run_path(stub, job)
        stub._file = lambda job: Mapper._file(stub, job)
        return stub, {"run_path": str(tmp_path / "map_r.h5")}

    def test_a_partial_file_is_renamed(self, tmp_path):
        stub, job = self.stub(tmp_path)
        (tmp_path / "map_r.h5.partial").write_bytes(b"tiles")
        path, note = Mapper._finalize_run_file(stub, job)
        assert path == tmp_path / "map_r.h5" and note is None and path.read_bytes() == b"tiles"

    def test_a_file_that_cannot_be_renamed_is_copied_and_the_job_still_finishes(self, tmp_path, monkeypatch):
        stub, job = self.stub(tmp_path)
        (tmp_path / "map_r.h5.partial").write_bytes(b"tiles")
        monkeypatch.setattr(mapper_module, "RENAME_RETRY_DELAYS_S", (0.0, 0.0))

        def locked(self, target):
            raise PermissionError("the file is open in another process")

        monkeypatch.setattr(Path, "replace", locked)
        path, note = Mapper._finalize_run_file(stub, job)
        assert path == tmp_path / "map_r.h5" and path.read_bytes() == b"tiles"
        assert "copied" in note


def test_a_release_naming_the_reservation_bare_still_keeps_an_aborted_runs_tiles(mapper_proxy, mapping_twin_proxy):
    reserve = call(mapper_proxy, "reserve", {"plan": {**PLAN, "imsize": 256, "dwell_s": 2e-6}})
    started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    while call(mapper_proxy, "job_status", {"job_id": started["job_id"]})["progress"]["done"] < 1:
        time.sleep(0.01)
    call(mapper_proxy, "abort", {"job_id": started["job_id"]})
    working = Path(call(mapper_proxy, "results", {"job_id": started["job_id"]})["result_keys"]["run_file"])
    assert json.loads(mapper_proxy.release(reserve["reservation_id"]))["state"] == "released"  # the id alone, as JobDevice accepts
    final = working.with_name(working.name[: -len(".partial")])
    assert final.exists() and not working.exists()


def test_a_dose_budgeted_map_checks_the_current_again_at_start_and_on_resume(mapper_proxy, mapping_twin_proxy, monkeypatch):
    from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin

    reserve = call(mapper_proxy, "reserve", {"plan": {**PLAN, "dose_budget_e_per_a2": 1e6}})
    budgeted = reserve["estimate"]["beam_current_pa"]
    original = DigitalTwin._get_parameters

    def raised(self):  # someone turned the current up after the plan was approved
        parameters = json.loads(original(self))
        parameters["beam_current_pa"] = budgeted * 2
        return json.dumps(parameters)

    monkeypatch.setattr(DigitalTwin, "_get_parameters", raised)
    moves = len(json.loads(mapping_twin_proxy.export_ground_truth())["moves"])
    for _attempt in ("start", "resume"):
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        status = wait(mapper_proxy, started["job_id"])
        assert status["state"] == "failed" and "dose budget no longer holds" in status["last_error"]["message"], status
        assert "above the" in status["last_error"]["message"] and status["progress"]["done"] == 0
    assert len(json.loads(mapping_twin_proxy.export_ground_truth())["moves"]) == moves  # nothing moved
    monkeypatch.setattr(DigitalTwin, "_get_parameters", original)
    resumed = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    assert wait(mapper_proxy, resumed["job_id"])["state"] == "done"
    release_all(mapper_proxy, reserve["reservation_id"])


def test_the_e_stop_aborts_whatever_is_running_without_naming_it(mapper_proxy, mapping_twin_proxy):
    assert json.loads(mapper_proxy.abort_all())["aborted"] == []
    plan = {**PLAN, "imsize": 256, "dwell_s": 2e-6}
    reserve = call(mapper_proxy, "reserve", {"plan": plan})
    started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    while call(mapper_proxy, "job_status", {"job_id": started["job_id"]})["progress"]["done"] < 1:
        time.sleep(0.01)
    stopped = json.loads(mapper_proxy.abort_all())["aborted"]
    assert [s["job_id"] for s in stopped] == [started["job_id"]]
    assert wait(mapper_proxy, started["job_id"])["state"] == "aborted"
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True
    release_all(mapper_proxy, reserve["reservation_id"])


def test_a_job_nobody_checks_on_is_aborted_when_its_lease_lapses(tango_ctx, mapping_twin_proxy):
    leased = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/lease"))
    leased.set_timeout_millis(60_000)
    plan = {**PLAN, "box_m": [-4e-7, 4e-7, -4e-7, 4e-7], "imsize": 512, "dwell_s": 4e-6}
    reserve = call(leased, "reserve", {"plan": plan})
    started = call(leased, "start", {"reservation_id": reserve["reservation_id"]})
    for _ in range(6):  # watched: each status call renews the lease
        time.sleep(0.5)
        assert call(leased, "job_status", {"job_id": started["job_id"]})["state"] == "running"
    time.sleep(3.0)  # unwatched for twice the lease
    status = wait(leased, started["job_id"])
    assert status["state"] == "aborted" and status["last_error"]["code"] == "lease_lapsed"
    assert status["progress"]["done"] < status["progress"]["total"]
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True
    release_all(leased, reserve["reservation_id"])


def test_a_paused_map_holds_the_device_with_the_beam_blanked_and_start_continues_it(mapper_proxy, mapping_twin_proxy):
    plan = {**PLAN, "imsize": 256, "dwell_s": 2e-6}
    reserve = call(mapper_proxy, "reserve", {"plan": plan})
    started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    while call(mapper_proxy, "job_status", {"job_id": started["job_id"]})["progress"]["done"] < 1:
        time.sleep(0.01)
    assert call(mapper_proxy, "pause", {"job_id": started["job_id"]})["pause_requested"] is True
    paused = wait(mapper_proxy, started["job_id"], states=("paused",))
    done = paused["progress"]["done"]
    time.sleep(1.0)
    assert call(mapper_proxy, "job_status", {"job_id": started["job_id"]})["progress"]["done"] == done < N_TILES
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True
    jobs = json.loads(mapper_proxy.get_jobs())["jobs"]
    assert [(j["job_id"], j["state"]) for j in jobs] == [(started["job_id"], "paused")]
    assert call(mapper_proxy, "reserve", {"plan": PLAN})["error"]["code"] == "busy"

    assert call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})["error"]["code"] == "paused"  # an agent's start or resume
    continued = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"], "continue": True})
    assert continued == {**continued, "job_id": started["job_id"], "state": "running", "continued": True}
    status = wait(mapper_proxy, started["job_id"])
    assert status["state"] == "done" and status["progress"]["done"] == N_TILES
    assert json.loads(mapper_proxy.get_jobs())["jobs"] == [] or json.loads(mapper_proxy.get_jobs())["jobs"][0]["state"] != "paused"
    release_all(mapper_proxy, reserve["reservation_id"])


def test_a_paused_map_can_be_aborted_and_the_e_stop_reaches_it(mapper_proxy, mapping_twin_proxy):
    plan = {**PLAN, "imsize": 256, "dwell_s": 2e-6}
    reserve = call(mapper_proxy, "reserve", {"plan": plan})
    started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    call(mapper_proxy, "pause", {"job_id": started["job_id"]})
    wait(mapper_proxy, started["job_id"], states=("paused",))
    stopped = json.loads(mapper_proxy.abort_all())["aborted"]
    assert [s["job_id"] for s in stopped] == [started["job_id"]]
    assert wait(mapper_proxy, started["job_id"])["state"] == "aborted"
    assert call(mapper_proxy, "pause", {"job_id": started["job_id"]})["error"]["code"] == "bad_state"
    resumed = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    assert wait(mapper_proxy, resumed["job_id"])["state"] == "done"
    release_all(mapper_proxy, reserve["reservation_id"])


def test_a_plan_is_revised_before_it_starts_and_checked_like_a_reserve(mapper_proxy):
    reserve = call(mapper_proxy, "reserve", {"plan": PLAN})
    smaller = {**PLAN, "box_m": [-1.2e-7, 4e-8, -8e-8, 8e-8]}
    revised = call(mapper_proxy, "revise", {"reservation_id": reserve["reservation_id"], "plan": smaller})
    assert "error" not in revised, revised
    assert revised["plan_id"] != reserve["plan_id"] and revised["replaces_plan_id"] == reserve["plan_id"]
    shown = call(mapper_proxy, "get_reservation", {"reservation_id": reserve["reservation_id"]})
    assert shown["plan"]["box_m"] == smaller["box_m"] and shown["plan_id"] == revised["plan_id"]
    refused = call(mapper_proxy, "revise", {"reservation_id": reserve["reservation_id"], "plan": {**PLAN, "fov_m": -1}})
    assert "error" in refused
    assert call(mapper_proxy, "get_reservation", {"reservation_id": reserve["reservation_id"]})["plan_id"] == revised["plan_id"]
    assert call(mapper_proxy, "revise", {"reservation_id": reserve["reservation_id"], "plan": {**PLAN, "drives": []}})["error"]["code"] == "bad_request"
    started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    status = wait(mapper_proxy, started["job_id"])
    assert status["state"] == "done" and status["progress"]["total"] == len(tile_centres(smaller["box_m"], 8e-8, 0.2, "serpentine"))
    assert call(mapper_proxy, "revise", {"reservation_id": reserve["reservation_id"], "plan": PLAN})["error"]["code"] == "bad_state"
    release_all(mapper_proxy, reserve["reservation_id"])


def test_a_stale_continue_never_resumes_a_job_aborted_meanwhile(mapper_proxy):
    plan = {**PLAN, "imsize": 256, "dwell_s": 2e-6}
    reserve = call(mapper_proxy, "reserve", {"plan": plan})
    started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
    call(mapper_proxy, "pause", {"job_id": started["job_id"]})
    wait(mapper_proxy, started["job_id"], states=("paused",))
    json.loads(mapper_proxy.abort_all())
    assert wait(mapper_proxy, started["job_id"])["state"] == "aborted"
    stale = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"], "continue": True})
    assert stale["error"]["code"] == "bad_state" and "not paused" in stale["error"]["message"]
    assert call(mapper_proxy, "job_status", {"job_id": started["job_id"]})["state"] == "aborted"
    release_all(mapper_proxy, reserve["reservation_id"])


def test_a_start_approved_for_one_plan_does_not_start_a_revised_one(mapper_proxy):
    reserve = call(mapper_proxy, "reserve", {"plan": PLAN})
    revised = call(mapper_proxy, "revise", {"reservation_id": reserve["reservation_id"], "plan": {**PLAN, "box_m": [-1.2e-7, 4e-8, -8e-8, 8e-8]}})
    stale = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"], "plan_id": reserve["plan_id"]})
    assert stale["error"]["code"] == "plan_changed"
    started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"], "plan_id": revised["plan_id"]})
    assert wait(mapper_proxy, started["job_id"])["state"] == "done"
    release_all(mapper_proxy, reserve["reservation_id"])


def test_a_forgotten_pause_is_aborted_after_pause_limit_s_and_not_by_the_lease(tango_ctx, mapping_twin_proxy):
    leased = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/lease"))
    leased.set_timeout_millis(60_000)
    plan = {**PLAN, "box_m": [-4e-7, 4e-7, -4e-7, 4e-7], "imsize": 256, "dwell_s": 2e-6}
    reserve = call(leased, "reserve", {"plan": plan})
    started = call(leased, "start", {"reservation_id": reserve["reservation_id"]})
    call(leased, "pause", {"job_id": started["job_id"]})
    paused = wait(leased, started["job_id"], states=("paused",))
    time.sleep(2.0)  # longer than lease_s with nobody checking: a paused job's lease does not lapse
    assert json.loads(leased.get_jobs())["jobs"][0]["state"] == "paused"
    status = wait(leased, started["job_id"], timeout=30)
    assert status["state"] == "aborted" and status["last_error"]["code"] == "pause_lapsed"
    assert status["progress"]["done"] == paused["progress"]["done"]
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True
    release_all(leased, reserve["reservation_id"])


def test_a_plan_whose_overlap_is_smaller_than_the_stage_error_is_warned_about():
    from types import SimpleNamespace

    from asyncroscopy.jobs.mapping.mapper import Mapper

    twin_stage = SimpleNamespace(stage_sigma_m=2.9e-7)
    warning = Mapper._overlap_warning(twin_stage, 0.3, 5.6e-7)  # 560 nm tiles: neighbours often do not even overlap
    assert warning and "left unplaced" in warning and "560 nm" in warning
    assert Mapper._overlap_warning(twin_stage, 0.5, 1.2e-5) is None  # 12 um tiles at half overlap: every pair is found
    assert Mapper._overlap_warning(SimpleNamespace(stage_sigma_m=1e-9), 0.15, 2e-7) is None  # an accurate stage
