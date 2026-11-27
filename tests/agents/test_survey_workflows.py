"""Tests for the 4 by 4 map and the whole sample map, run against a fake instrument."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from asyncroscopy.workflows import get_workflow
from asyncroscopy.workflows.runner import resume_workflow, run_workflow
from pyTEMlib.analysis_server.map_tools import MapTools

from test_workflows import run, twin


def survey_twin(fits=None):
    """The fake microscope with beam placement and spectra added. ``fits(n, request)`` answers tile check n."""
    instrument, analysis, state = twin()
    checks = []

    def match(args):
        checks.append(args)
        return {"matched": True if fits is None else bool(fits(len(checks) - 1, args)), "score": 0.9}

    def spectrum(kind):
        def take(args):
            state.setdefault(kind, []).append({"shift": list(state["shift"]), "beam_on": not state["blanked"], "args": args})
            return f"spectrum_{kind}_{len(state[kind]):04d}.h5"
        return take

    instrument.tools.update({
        "Twin_place_beam": lambda a: "ok",
        "Twin_acquire_spectrum": spectrum("eds"),
        "EELS_acquire_eels_spectrum": spectrum("eels"),
    })
    analysis.tools["match_tiles"] = match
    return instrument, analysis, state, checks


def test_the_4_by_4_map_needs_no_settings_and_checks_each_tile_as_it_is_taken(tmp_path):
    instrument, analysis, state, checks = survey_twin()
    state["fov"] = 3.0e-7
    record = run(run_workflow("map_4x4", {}, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done", record
    tile = record["outputs"]["tile"]
    assert record["outputs"]["plan"]["count"] == 16 and len(tile["keys"]) == 16 and record["visits"]["tile"] == 1
    # Sixteen tiles are within image shift reach so the stage never moves.
    assert "Twin_move_stage" not in instrument.names() and state["stage"][:2] == [1e-6, 2e-6]
    assert state["shift"] == [0.0, 0.0] and state["fov"] == 3.0e-7 and state["blanked"]
    # Every tile after the first is checked against the one before it.
    assert len(checks) == 15 and tile["matched"] == 15 and tile["unchecked"] == 1 and tile["extra"] == 0
    assert checks[0]["image"] == tile["keys"][1] and checks[0]["neighbour"] == tile["keys"][0]
    assert checks[0]["offset_xy_m"] == pytest.approx([2.1e-7, 0.0]) and checks[0]["fov_m"] == 3.0e-7
    assert record["outputs"]["stitch"]["map"] == "map_abc_map_v1"
    assert analysis.log[-4][0] == "collect_map_run" and analysis.log[-4][1]["fov_m"] == 3.0e-7


def test_a_tile_that_does_not_fit_gets_a_bridging_image_closer_to_the_last(tmp_path):
    instrument, analysis, state, checks = survey_twin(fits=lambda n, request: n != 1)  # the third tile does not fit the second
    state["fov"] = 3.0e-7
    record = run(run_workflow("map_4x4", {}, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done", record
    tile = record["outputs"]["tile"]
    assert len(tile["keys"]) == 17 and tile["extra"] == 1 and tile["matched"] == 15 and tile["unmatched"] == 0
    # The bridge sits halfway between the second and third tiles.
    second, bridge, third = tile["stage_xy_m"][1:4]
    assert bridge == pytest.approx([(second[0] + third[0]) / 2, second[1]])
    assert [c["neighbour"] for c in checks[1:4]] == [tile["keys"][1], tile["keys"][1], tile["keys"][2]]
    assert checks[3]["image"] == tile["keys"][3] and checks[3]["offset_xy_m"][0] == pytest.approx(1.05e-7)

    # On a blank area nothing fits but the run still ends.
    instrument, analysis, state, checks = survey_twin(fits=lambda n, request: False)
    state["fov"] = 3.0e-7
    record = run(run_workflow("map_4x4", {}, instrument=instrument, analysis=analysis, state_dir=tmp_path / "blank"))
    tile = record["outputs"]["tile"]
    assert record["state"] == "done" and tile["unmatched"] == 15 and tile["extra"] == 15 and len(tile["keys"]) == 31


SAMPLE = {"size_m": 5.0e-6, "fov_m": 1.0e-6, "depth": 1, "zoom": 2.0, "spectra_every": 100, "eds": True}


def test_the_sample_map_asks_first_then_maps_every_level_with_spectra_on_the_way(tmp_path):
    workflow = get_workflow("map_sample")
    instrument, analysis, state, checks = survey_twin()
    record = run(run_workflow("map_sample", SAMPLE, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    # 81 images at 1 um plus 225 at 500 nm is over ask_over so a person is asked first.
    assert record["state"] == "waiting", record
    question = record["questions"][0]
    assert "306 images" in question["question"] and question["options"] == ["yes", "no"]
    assert not {"Twin_move_stage", "Twin_set_image_shift", "Twin_set_fov", "Twin_acquire_scanned_image"} & set(instrument.names())

    record = run(resume_workflow(record["run_id"], instrument=instrument, analysis=analysis, state_dir=tmp_path, find=lambda name: workflow,
                                 answers={question["id"]: "yes"}))
    assert record["state"] == "done", record
    plan, tile = record["outputs"]["plan"], record["outputs"]["tile"]
    assert [(level["fields_per_side"], level["tiles_per_field_side"], level["count"]) for level in plan["levels"]] == [(3, 3, 81), (3, 5, 225)]
    assert tile["taken"] == 306 and state["taken"] == 306 and tile["level"] == 1 and tile["left"] == 0
    gathered = [args for name, args in analysis.log if name == "collect_map_run"]
    assert [(len(g["image_keys"]), g["fov_m"]) for g in gathered] == [(81, 1.0e-6), (225, 5.0e-7)]
    assert record["outputs"]["maps"]["count"] == 2
    # The stage stays inside the square and every image shift stays within reach.
    moves = [args["position"] for name, args in instrument.log if name == "Twin_move_stage"]
    assert len({(round(p[0], 12), round(p[1], 12)) for p in moves[:-1]}) == 9 + 9 - 1  # the middle group is where the stage began
    assert max(abs(p[0] - 1e-6) for p in moves) < 2.5e-6 and all(p[2:] == [0.0, 0.0, 0.0] for p in moves)
    assert max(abs(v) for name, args in instrument.log if name == "Twin_set_image_shift" for v in args["shift"]) <= 1.0e-6
    # EDS spectra are taken at every hundredth tile with the beam on. EELS is off.
    assert len(state["eds"]) == 3 and "eels" not in state and all(s["beam_on"] for s in state["eds"])
    assert all(s["args"] == {"detector_name": "eds"} for s in state["eds"])
    assert record["outputs"]["eds"]["keys"] == ["spectrum_eds_0001.h5", "spectrum_eds_0002.h5", "spectrum_eds_0003.h5"]
    assert all(place.startswith("stem_image_HAADF_") for place in record["outputs"]["eds"]["at"])
    assert ("Twin_place_beam", {"position": [0.5, 0.5]}) in instrument.log
    # The instrument is left as it was found.
    assert state["stage"] == [1e-6, 2e-6, 0.0, 0.0, 0.0] and state["shift"] == [0.0, 0.0] and state["fov"] == 5.8e-9 and state["blanked"]


def test_the_sample_map_stops_when_the_person_says_no_and_refuses_what_is_too_large(tmp_path):
    workflow = get_workflow("map_sample")
    instrument, analysis, state, checks = survey_twin()
    record = run(run_workflow("map_sample", SAMPLE, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    record = run(resume_workflow(record["run_id"], instrument=instrument, analysis=analysis, state_dir=tmp_path, find=lambda name: workflow,
                                 answers={record["questions"][0]["id"]: "no"}))
    assert record["state"] == "done" and state["taken"] == 0 and "Twin_move_stage" not in instrument.names()

    for changed, said in (({"depth": 3, "zoom": 4.0}, "over the limit of 2000"), ({"size_m": 2.0e-3, "fov_m": 2.0e-4}, "beyond stage_limit_m")):
        instrument, analysis, state, checks = survey_twin()
        record = run(run_workflow("map_sample", {**SAMPLE, **changed}, instrument=instrument, analysis=analysis, state_dir=tmp_path / said[:4]))
        assert record["state"] == "failed" and said in str(record), record
        assert state["taken"] == 0 and "Twin_move_stage" not in instrument.names()


def test_two_overlapping_images_match_where_they_are_and_not_where_they_are_not():
    rng = np.random.default_rng(3)
    scene = rng.normal(size=(400, 400))
    scene = (scene + np.roll(scene, 1, 0) + np.roll(scene, 1, 1) + np.roll(scene, (1, 1), (0, 1))) / 4.0
    # b is 120 columns to the right of a and 5 rows down
    images = {"a": scene[40:296, 20:276], "b": scene[45:301, 140:396], "far": rng.normal(size=(256, 256))}
    store = SimpleNamespace(client={key: SimpleNamespace(structure_family="array", read=lambda value=value: value) for key, value in images.items()})
    tools = SimpleNamespace(store=store)
    pixel = 1.0e-9
    found = MapTools.match_tiles(tools, "b", "a", [118 * pixel, 3 * pixel], 256 * pixel)  # the guess is two pixels off on each axis
    assert found["matched"] and found["score"] > 0.9
    assert found["offset_xy_m"] == pytest.approx([120 * pixel, 5 * pixel], abs=0.3 * pixel)
    assert not MapTools.match_tiles(tools, "far", "a", [118 * pixel, 3 * pixel], 256 * pixel)["matched"]
    assert MapTools.match_tiles(tools, "b", "a", [300 * pixel, 0.0], 256 * pixel)["reason"] == "the two images do not overlap at that offset"


def test_a_spectrum_due_at_a_tile_that_needed_a_bridge_is_taken_on_that_tile_not_the_bridge(tmp_path):
    # Spectra at every third tile, and the third tile does not fit the second, so a bridge is taken right before the stop.
    instrument, analysis, state, checks = survey_twin(fits=lambda n, request: n != 1)
    inputs = {"size_m": 5.0e-6, "fov_m": 1.0e-6, "spectra_every": 3, "eds": True}
    record = run(run_workflow("map_sample", inputs, instrument=instrument, analysis=analysis, state_dir=tmp_path))
    assert record["state"] == "done", record
    shifts, at_image = [0.0, 0.0], []
    for name, args in instrument.log:
        if name == "Twin_set_image_shift":
            shifts = list(args["shift"])
        elif name == "Twin_acquire_scanned_image":
            at_image.append(shifts)
    third_tile, bridge = at_image[2], at_image[3]
    assert third_tile != bridge
    assert state["eds"][0]["shift"] == third_tile
