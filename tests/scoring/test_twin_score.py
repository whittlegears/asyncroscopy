"""The twin score of a run, from the twin's export_ground_truth before and after it (asyncroscopy.sim.truth_score)."""

from __future__ import annotations

import pytest

from asyncroscopy.sim.truth_score import (
    median_error,
    points_on_particles,
    truth_marker,
    twin_score,
    twin_time_and_dose,
)


def truth(acquisitions, total_electrons, clock_s):
    return {"particles": [{"id": 7, "center_m": [0.0, 0.0], "semi_axes_m": [2e-9, 3e-9]},
                          {"id": 8, "center_m": [5e-8, 0.0], "semi_axes_m": [2e-9, 2e-9]},
                          {"id": 9, "center_m": [1e-6, 1e-6], "semi_axes_m": [2e-9, 2e-9]}],
            "acquisitions": acquisitions, "simulated_time_s": clock_s,
            "dose": {"max_dose_e_per_a2": 120.0, "total_electrons": total_electrons}}


def image(index, centre, reported, true):
    return {"index": index, "kind": "image", "center_true_m": centre, "fov_m": 2e-8, "imsize": 100, "reported_m": reported, "true_m": true}


def test_only_the_runs_own_acquisitions_count_and_every_number_says_what_it_measured():
    before = truth([image(0, [1e-6, 1e-6], [1e-6, 1e-6, 0], [1e-6, 1e-6, 0])], 1000.0, 10.0)
    marker = truth_marker(before)
    assert marker == {"last_acquisition": 0, "simulated_time_s": 10.0, "total_electrons": 1000.0}
    after = truth([*before["acquisitions"],
                   image(1, [1e-9, 0.0], [0.0, 0.0, 0], [1e-9, 0.0, 0]),
                   image(2, [5e-8, 0.0], [5.3e-8, 0.0, 0], [5e-8, 4e-9, 0])], 1600.0, 14.5)
    record = {"blueprint": {"nodes": {"find": {"block": "find_feature"}, "tile": {"block": "acquire_tile"}}},
              "outputs": {"find": {"features": [{"x_m": 1e-9, "y_m": 1e-9}, {"x_m": 3e-8, "y_m": 0.0}]},
                          "tile": {"features": [{"x_m": 5e-8, "y_m": 0.0}]}}}
    score = twin_score(marker, after, record)
    assert score["label"].startswith("twin score") and score["known"]
    assert score["acquisitions"] == 2 and score["images"] == 2
    assert score["particles_total"] == 3 and score["particles_imaged"] == 2, "particle 9 was imaged before this run"
    assert score["features_reported"] == 2 and score["features_on_particles"] == 1 and score["feature_precision"] == 0.5
    assert score["stage_error_m"]["images"] == 2
    assert score["stage_error_m"]["median"] == pytest.approx((1e-9 + 5e-9) / 2) and score["stage_error_m"]["max"] == pytest.approx(5e-9)
    clock, dose = twin_time_and_dose(marker, after)
    assert clock == pytest.approx(4.5) and dose == {"total_electrons": 600.0, "max_dose_e_per_a2": 120.0}


def test_a_twin_without_particles_or_a_dose_map_says_so():
    score = twin_score({"last_acquisition": None}, {"acquisitions": []}, {})
    assert not score["known"] and score["particles_total"] == 0 and score["feature_precision"] is None and score["stage_error_m"] is None
    assert twin_time_and_dose({}, {"acquisitions": []}) == (None, None)


def test_the_shared_geometry():
    centres = [[0.0, 0.0], [1e-8, 0.0]]
    assert list(points_on_particles([[0.5e-9, 0.0]], centres, [1e-9, 1e-9])) == [True, False]
    assert list(points_on_particles([[3e-9, 0.0]], centres, [1e-9, 1e-9], tolerance_m=4e-9)) == [True, False]
    assert list(points_on_particles([], centres, [1e-9, 1e-9])) == [False, False]
    assert median_error([[1.0, 1.0], [2.0, 2.0]], [[0.0, 0.0], [1.0, 1.0]]) == 0.0
    assert median_error([[3.0, 4.0]], [[0.0, 0.0]], remove_offset=False) == 5.0
