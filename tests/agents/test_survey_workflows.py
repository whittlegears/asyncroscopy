"""Tests for the 4 by 4 map and the whole sample map, run against a fake instrument."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastmcp import Client
from tiled.client import from_uri
from tiled.server import SimpleTiledServer

from asyncroscopy.workflows import get_workflow
from asyncroscopy.workflows.runner import resume_workflow, run_workflow
from pytemlib_mcp.analysis_server import create_server
from pytemlib_mcp.analysis_server.map_store import pixel_to_stage
from pytemlib_mcp.analysis_server.map_tools import MapTools

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
    assert "Twin_move_stage" not in instrument.names() and state["stage"][:2] == [1e-6, 2e-6]
    assert state["shift"] == [0.0, 0.0] and state["fov"] == 3.0e-7 and state["blanked"]
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
    second, bridge, third = tile["stage_xy_m"][1:4]
    assert bridge == pytest.approx([(second[0] + third[0]) / 2, second[1]])
    assert [c["neighbour"] for c in checks[1:4]] == [tile["keys"][1], tile["keys"][1], tile["keys"][2]]
    assert checks[3]["image"] == tile["keys"][3] and checks[3]["offset_xy_m"][0] == pytest.approx(1.05e-7)

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
    moves = [args["position"] for name, args in instrument.log if name == "Twin_move_stage"]
    assert len({(round(p[0], 12), round(p[1], 12)) for p in moves[:-1]}) == 9 + 9 - 1  # the middle group is where the stage began
    assert max(abs(p[0] - 1e-6) for p in moves) < 2.5e-6 and all(p[2:] == [0.0, 0.0, 0.0] for p in moves)
    assert max(abs(v) for name, args in instrument.log if name == "Twin_set_image_shift" for v in args["shift"]) <= 1.0e-6
    assert len(state["eds"]) == 3 and "eels" not in state and all(s["beam_on"] for s in state["eds"])
    assert all(s["args"] == {"detector_name": "eds"} for s in state["eds"])
    assert record["outputs"]["eds"]["keys"] == ["spectrum_eds_0001.h5", "spectrum_eds_0002.h5", "spectrum_eds_0003.h5"]
    assert all(place.startswith("stem_image_HAADF_") for place in record["outputs"]["eds"]["at"])
    eds = record["outputs"]["eds"]
    assert "spectrum_keys" not in gathered[0]
    by_tile = dict(zip(eds["at"], eds["keys"]))
    assert gathered[1]["spectrum_keys"] == [by_tile.get(key, "") for key in gathered[1]["image_keys"]]
    assert sum(1 for key in gathered[1]["spectrum_keys"] if key) == 3 and gathered[1]["spectra_at"] == [0.5, 0.5]
    assert ("Twin_place_beam", {"position": [0.5, 0.5]}) in instrument.log
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


class AnalysisServer:
    """The real analysis server (pyTEMlib's FastMCP app) reached in memory, the way a workflow reaches it over HTTP."""

    def __init__(self, mcp):
        self.mcp, self.log = mcp, []

    async def list_tools(self):
        async with Client(self.mcp) as client:
            return await client.list_tools()

    async def call(self, name, arguments=None):
        self.log.append((name, dict(arguments or {})))
        async with Client(self.mcp) as client:
            return await client.call_tool(name, dict(arguments or {}), raise_on_error=False)


@pytest.fixture
def tiled(tmp_path):
    try:
        server = SimpleTiledServer(directory=tmp_path / "tiled")
    except Exception as exc:
        print(f"skipping: a local Tiled server could not start ({exc!r})")
        pytest.skip(f"a local Tiled server could not start: {exc!r}")
    yield server
    server.close()


def test_a_sample_map_with_eds_gives_its_stitched_map_an_eds_point_where_each_spectrum_was_taken(tmp_path, tiled):
    client = from_uri(tiled.uri)
    instrument, _fake, state, _checks = survey_twin()
    acquire = instrument.tools["Twin_acquire_scanned_image"]
    rng = np.random.default_rng(7)

    def acquire_and_store(args):
        reply = acquire(args)
        client.create_container(reply.split("\n")[0]).create_container("image").write_array(
            rng.integers(0, 1000, (64, 64)).astype(np.uint16), key="HAADF")
        return reply

    instrument.tools["Twin_acquire_scanned_image"] = acquire_and_store
    mcp, tools = create_server(tiled_uri=tiled.uri, include_basic=False)
    analysis = AnalysisServer(mcp)
    inputs = {"size_m": 2.0e-6, "fov_m": 1.0e-6, "spectra_every": 2, "eds": True, "spectra_at": [0.25, 0.75], "check": False}
    record = run(run_workflow("map_sample", inputs, instrument=instrument, analysis=analysis, state_dir=tmp_path / "runs"))
    assert record["state"] == "done", record

    eds = record["outputs"]["eds"]
    taken = [key for key in eds["keys"] if key]
    assert len(taken) >= 2
    assert len(taken) == len(record["outputs"]["tile"]["keys"]) // 2 and record["outputs"]["run"]["spectra"] == len(taken)
    key = record["outputs"]["stitch"]["map"]
    assert record["outputs"]["stitch"]["summary"]["eds_pois"] == len(taken)
    assert "eds_point" in [k["kind"] for k in tools.map_info(key)["poi_kinds"]]

    meta = tools.store.map_metadata(key)
    inverse = np.linalg.inv(np.asarray(meta["scan_linear_applied"]))
    images = record["outputs"]["tile"]["keys"]
    points = {p["links"][0]: p for p in tools.list_pois(key, kind="eds_point")["pois"]}
    assert sorted(points) == sorted(taken)
    for spectrum, tile in zip(eds["keys"], eds["at"]):
        row, col = np.asarray(meta["tile_positions_px"][images.index(tile)]) + (64 - 1) / 2 + inverse @ [(0.75 - 0.5) * 64, (0.25 - 0.5) * 64]
        assert (points[spectrum]["x_m"], points[spectrum]["y_m"]) == pytest.approx(pixel_to_stage(meta, row, col), abs=1e-11)
    assert ("Twin_place_beam", {"position": [0.25, 0.75]}) in instrument.log
