"""Live drift tracking during a map (item 8 of docs/designs/concepts_brief.md).

Done when: at 5e-11 m/s the placement error stays below 0.2 field of view without
stitching, every correction is traced, and the stage never leaves the plan box.

The twin (asyncroscopy/digitaltwin/livedrift, tests/conftest.py) drifts steadily at
5e-11 m/s on its simulated clock (time_scale 0), with a 1 nm stage and a 10 s move per
tile. A tile's placement error is measured from the twin's ground truth, not from a
stitch: where its frame was really centred on the sample (``center_true_m``) minus the
tile centre the plan asked for, relative to the map's first tile. The proof is
circular: the tracker recovers the twin's own drift model.
"""

import json
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango

from asyncroscopy.routines.drift import LinearTracker, bounded_correction, drift_bench, get_tracker
from asyncroscopy.routines.grid import index_at, inside_box, tile_at

FOV = 5e-8
# 6 x 6 tiles of 50 nm at 25 % overlap: a 237.5 nm box, about 430 s of simulated time (21 nm of drift).
HALF = (FOV + 5 * FOV * 0.75) / 2
PLAN = {"box_m": [-HALF, HALF, -HALF, HALF], "fov_m": FOV, "overlap": 0.25, "imsize": 96, "dwell_s": 2e-4, "stitch": False}
TRACK = {"track": True, "max_shift_m": 5e-9}


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def run(mapper, plan, timeout=600):
    reserved = call(mapper, "reserve", {"plan": plan})
    assert "error" not in reserved, reserved
    started = call(mapper, "start", {"reservation_id": reserved["reservation_id"]})
    deadline = time.time() + timeout
    while (status := call(mapper, "job_status", {"job_id": started["job_id"]}))["state"] not in {"done", "failed", "aborted"}:
        assert time.time() < deadline, status
        time.sleep(0.05)
    results = call(mapper, "results", {"job_id": started["job_id"]})
    call(mapper, "release", {"reservation_id": reserved["reservation_id"]})
    assert status["state"] == "done", (status, results)
    return reserved, results


@pytest.fixture(scope="module")
def twin(tango_ctx):
    return tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/livedrift"))


@pytest.fixture(scope="module")
def mapper(tango_ctx):
    proxy = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/livedrift"))
    proxy.set_timeout_millis(60_000)
    return proxy


def mapped(twin, mapper, plan):
    """Run ``plan``; its reservation, results, tile images' ground truth and stage moves."""
    before = json.loads(twin.export_ground_truth())
    reserved, results = run(mapper, plan)
    after = json.loads(twin.export_ground_truth())
    images = [a for a in after["acquisitions"][len(before["acquisitions"]):] if a["kind"] == "image"]
    moves = after["moves"][len(before["moves"]):]
    assert len(images) == reserved["estimate"]["tiles"]
    return reserved, results, images, moves


def placement_errors(images) -> np.ndarray:
    """|where each tile's frame really was - where the plan put it|, relative to the first tile, in fields of view.

    The plan's tile centre is the stage position the Mapper commanded for it (tracking never changes it).
    """
    planned = np.asarray([image["commanded_m"][:2] for image in images])
    true = np.asarray([image["center_true_m"] for image in images])
    error = (true - planned) - (true[0] - planned[0])
    return np.hypot(error[:, 0], error[:, 1]) / FOV


@pytest.fixture(scope="module")
def untracked(twin, mapper):
    reserved, results, images, moves = mapped(twin, mapper, PLAN)
    return reserved, results, images, moves


@pytest.fixture(scope="module")
def tracked(twin, mapper, job_dir):
    base = [float(v) for v in twin.get_image_shift()]
    reserved, results, images, moves = mapped(twin, mapper, {**PLAN, "drift": TRACK})
    trace = Path(job_dir / "livedrift" / "trace.jsonl")
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [e for e in events if e.get("run_id") == reserved["reservation_id"]]
    return reserved, results, images, moves, events, base


def test_untracked_placement_error_grows_with_the_drift(untracked):
    reserved, results, images, _moves = untracked
    errors = placement_errors(images)
    print(f"\nuntracked: max placement error {errors.max():.3f} fov, last tile {errors[-1]:.3f} fov")
    assert errors.max() > 0.3  # the baseline: 21 nm of drift on 50 nm tiles
    assert errors[-1] > errors[len(errors) // 2] > errors[1]  # it grows with time
    assert "drift_tracking" not in results["summary"]  # off by default


def test_tracking_keeps_every_tile_within_a_fifth_of_a_field_of_view(tracked, untracked):
    reserved, results, images, _moves, _events, _base = tracked
    errors = placement_errors(images)
    baseline = placement_errors(untracked[2])
    summary = results["summary"]["drift_tracking"]
    print(f"\ntracked: max placement error {errors.max():.3f} fov (untracked {baseline.max():.3f}); {summary}")
    assert errors.max() < 0.2
    assert errors.max() < 0.5 * baseline.max()
    tiles = reserved["estimate"]["tiles"]
    assert summary["corrections"] == tiles and summary["measurements"] >= tiles - 3
    # The tracker's rate is the twin's (5e-11 m/s), measured on the twin's clock.
    assert np.hypot(*summary["rate_m_per_s"]) == pytest.approx(5e-11, rel=0.25)
    assert summary["max_step_m"] <= TRACK["max_shift_m"] + 1e-15
    assert summary["restored"] is True


def test_every_correction_is_traced_and_is_the_shift_the_twin_applied(tracked, twin):
    reserved, results, images, _moves, events, base = tracked
    corrections = [e for e in events if e["action"] == "Mapper.drift_correction"]
    assert len(corrections) == reserved["estimate"]["tiles"] and all(e["ok"] for e in corrections)
    assert [e["step"] for e in corrections] == list(range(len(corrections)))
    for event, image in zip(corrections, images):
        arguments = event["arguments"]
        assert max(abs(v) for v in arguments["step_m"]) <= TRACK["max_shift_m"] + 1e-15
        # The shift the twin applied to this frame: where it was centred minus the true stage and its drift.
        drift = (np.array(image["drift_start_m"]) + np.array(image["drift_end_m"])) / 2
        applied = np.array(image["center_true_m"]) - np.array(image["true_m"][:2]) - drift
        assert applied == pytest.approx(np.array(arguments["image_shift_m"]), abs=2e-11)
        assert arguments["image_shift_m"] == pytest.approx(np.array(base) + np.array(arguments["correction_m"]), abs=1e-15)
    # Afterwards the shift is back where it was before the map.
    assert [float(v) for v in twin.get_image_shift()] == pytest.approx(base, abs=1e-15)


def test_the_stage_never_leaves_the_plan_box(tracked):
    reserved, _results, images, moves, events, _base = tracked
    box = PLAN["box_m"]
    assert len(moves) == reserved["estimate"]["tiles"]
    for move in moves:
        assert inside_box(box, move["commanded"][0], move["commanded"][1], FOV / 2)
    # And only to the planned tile centres: tracking moves the beam, never the stage.
    plan = reserved
    steps = [e for e in events if e["action"] == "Mapper.step"]
    assert len(steps) == len(moves)
    for move, image in zip(moves, images):
        assert move["commanded"][:2] == image["commanded_m"][:2]
    del plan


def test_the_run_file_keeps_each_correction_and_measurement(tracked):
    _reserved, results, _images, _moves, events, _base = tracked
    with h5py.File(results["result_keys"]["run_file"], "r") as h5:
        corrections = h5["drift_correction_m"][:]
        errors = h5["drift_error_m"][:]
        traced = [e for e in h5["trace"].asstr()[:] if "drift_correction" in e]
    assert corrections.shape == (36, 2) and np.isfinite(corrections).all()
    assert np.isfinite(errors[:, 0]).sum() >= 33 and errors[0].tolist() == [0.0, 0.0]
    assert len(traced) == 36


def test_a_dry_run_with_tracking_records_every_shift_and_refuses_none(twin, mapper, job_dir):
    before = json.loads(twin.export_ground_truth())
    shift = [float(v) for v in twin.get_image_shift()]
    reserved, results = run(mapper, {**PLAN, "drift": TRACK, "dry_run": True})
    after = json.loads(twin.export_ground_truth())
    assert len(after["acquisitions"]) == len(before["acquisitions"]) and len(after["moves"]) == len(before["moves"])
    assert [float(v) for v in twin.get_image_shift()] == shift
    assert results["dry_run"]["failed_checks"] == 0
    trace = Path(job_dir / "livedrift" / "trace.jsonl")
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [e for e in events if e.get("run_id") == reserved["reservation_id"]]
    shifts = [e for e in events if e["action"] == "set_image_shift"]
    # One per tile, plus the restore at the end; each recorded (not performed) and checked.
    assert len(shifts) == reserved["estimate"]["tiles"] + 1
    assert all(e["performed"] is False and e["ok"] is True and e["check"]["ok"] is True for e in shifts)
    assert len([e for e in events if e["action"] == "Mapper.drift_correction"]) == reserved["estimate"]["tiles"]


def test_tracking_is_refused_by_name_where_it_cannot_run(tango_ctx, mapper):
    afm = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/mapper/afm"))
    refused = call(afm, "reserve", {"plan": {"size_m": [4e-6, 4e-6], "fov_m": 2e-6, "drift": {"track": True}}})
    assert refused["error"]["code"] in {"bad_request", "plan_rejected"}
    assert "drift.track" in refused["error"]["message"] and "'yaml' instrument profile" in refused["error"]["message"]
    for drift, words in (
        ({"track": True, "max_shift_m": 1e-5}, "max_shift_m"),
        ({"track": True, "max_total_m": 1e-3}, "image shift limit"),
        ({"track": True, "tracker": "kalman"}, "unknown drift tracker"),
        ({"track": True, "gain": 1}, "gain"),
    ):
        refused = call(mapper, "reserve", {"plan": {**PLAN, "drift": drift}})
        assert refused["error"]["code"] in {"bad_request", "plan_rejected"} and words in refused["error"]["message"], refused
    refused = call(mapper, "reserve", {"plan": {**PLAN, "overlap": 0.0, "drift": TRACK}})
    assert "overlap" in refused["error"]["message"]
    # Off is accepted and changes nothing.
    reserved = call(mapper, "reserve", {"plan": {**PLAN, "drift": {"track": False}}})
    assert "error" not in reserved
    call(mapper, "release", {"reservation_id": reserved["reservation_id"]})


# ----------------------------------------------------------------------
# The pure parts: no stack
# ----------------------------------------------------------------------
def test_the_linear_tracker_and_the_bounded_correction():
    tracker = LinearTracker({"window": 4})
    assert tracker.predict(10.0) == (0.0, 0.0)
    for t in range(6):
        tracker.observe(t * 10.0, [1e-9 * t, -2e-9 * t])
    assert tracker.predict(100.0) == pytest.approx((1e-8, -2e-8))
    assert len(tracker.state()["observations"]) == 4  # the window bounds the saved state
    assert get_tracker("linear") is LinearTracker
    shift, limited = bounded_correction([5e-9, -1e-9], [0.0, 0.0], [1e-7, 1e-7], 2e-9, 1e-6)
    assert shift == pytest.approx([2e-9, -1e-9]) and limited
    shift, limited = bounded_correction([1e-7 + 5e-9, 1e-7], [1e-7 + 2e-9, 1e-7], [1e-7, 1e-7], 2e-9, 3e-9)
    assert shift == pytest.approx([1e-7 + 3e-9, 1e-7]) and limited


def test_index_at_inverts_tile_at():
    grid = {"rows": 3, "cols": 4, "x0": 0.0, "dx": 1.0, "y0": 0.0, "dy": 1.0}
    for order in ("serpentine", "raster"):
        for k in range(12):
            tile = tile_at(grid, order, k)
            assert index_at(grid, order, tile["row"], tile["col"]) == k


def test_the_drift_bench_scores_a_tracker_without_a_stack():
    score = drift_bench("linear", seeds=(0,), grid=4)
    assert score["max_error_fov"] < 0.2 < score["untracked_max_error_fov"] or score["max_error_fov"] < 0.5 * score["untracked_max_error_fov"]
