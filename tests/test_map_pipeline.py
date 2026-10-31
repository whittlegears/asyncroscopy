"""The Mapper's pipeline: tiles, then stitch (with EDS points of interest), score and revisit, as one job.

The analysis server is pyTEMlib's real MapTools (jobs, stitching, scoring,
POI layer), called in this process instead of over HTTP; the twin, Mapper and
DATA are the in-process Tango devices from conftest.
"""

import importlib.util
import json
import time
from pathlib import Path

import pytest

from asyncroscopy.instruments.electron_microscope.mapper import Mapper, REVISIT_EDGE_SHIFT

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("pyTEMlib.mcp_server.map_tools") is None, reason="needs the pyTEMlib fork with map tools"),
]

PLAN = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": 1.0e-7, "overlap": 0.25, "imsize": 64, "dwell_s": 2e-6}


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


class InProcessAnalysis:
    """pyTEMlib's map tools behind the same calls the Mapper makes to the analysis server."""

    def __init__(self, roots):
        from pyTEMlib.mcp_server.map_store import MapStore
        from pyTEMlib.mcp_server.map_tools import MapTools

        self.tools = MapTools(MapStore(local_roots=roots))
        self.calls = []

    def __call__(self, mapper, tool, arguments):
        self.calls.append(tool)
        jobs = self.tools.jobs
        if tool in ("assemble_map", "score_map"):
            reserved = jobs.reserve({"task": tool, **arguments})
            return reserved if "error" in reserved else jobs.start(reserved["reservation_id"])
        if tool == "analysis_job_status":
            return jobs.wait(arguments["job_id"], timeout=arguments.get("wait_s", 5))
        if tool == "analysis_job_results":
            return jobs.results(arguments["job_id"])
        if tool == "analysis_job_abort":
            return jobs.abort(arguments["job_id"])
        if tool == "add_poi":
            poi = {"id": f"hm{len(self.calls)}", "source": "agent", "score": None, "scorer_version": None, "links": [], **arguments}
            poi.pop("map")
            self.tools.store.add_pois(arguments["map"], [poi])
            return poi
        if tool == "map_info":
            return self.tools.map_info(arguments["map"])
        if tool == "list_pois":
            return self.tools.list_pois(arguments["map"], kind=arguments.get("kind"), box_m=arguments.get("box_m"), limit=arguments.get("limit", 200))
        raise AssertionError(f"unexpected analysis call {tool}")


def run_job(mapper_proxy, plan):
    reserved = call(mapper_proxy, "reserve", {"plan": plan})
    assert "reservation_id" in reserved, reserved
    started = call(mapper_proxy, "start", {"reservation_id": reserved["reservation_id"]})
    phases = set()
    deadline = time.time() + 300
    while True:
        status = call(mapper_proxy, "job_status", {"job_id": started["job_id"]})
        phases.add(status.get("phase"))
        if status["state"] in {"done", "failed", "aborted"}:
            break
        assert time.time() < deadline
        time.sleep(0.05)
    results = call(mapper_proxy, "results", {"job_id": started["job_id"]})
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    return results, phases


@pytest.fixture
def analysis(monkeypatch, tmp_path_factory):
    holder = {}

    def fake(self, tool, arguments):
        return holder["server"](self, tool, arguments)

    monkeypatch.setattr(Mapper, "analysis_url", "in-process")
    monkeypatch.setattr(Mapper, "_analysis", fake)
    return holder


def test_a_map_job_stitches_scores_and_revisits_with_every_spectrum_on_the_map(mapper_proxy, mapping_twin_proxy, analysis):
    mapping_twin_proxy.set_defocus(0.0)
    plan = {**PLAN, "per_tile": ["eds_point"], "eds_exposure_s": 0.05, "find_regions": 2, "revisit": {"fov_m": 3e-8}}
    reserved = call(mapper_proxy, "reserve", {"plan": plan})
    assert reserved["estimate"]["phases"] == ["tiles", "stitch", "score", "revisit"]
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})

    servers = []

    def server(mapper, tool, arguments):
        if not servers:  # pyTEMlib may read local run files only under its roots: the run's folder
            servers.append(InProcessAnalysis([Path(arguments["run"]).parent]))
        return servers[0](mapper, tool, arguments)

    analysis["server"] = server
    results, phases = run_job(mapper_proxy, plan)
    assert results["state"] == "done", results
    summary = results["summary"]
    tiles = summary["steps_total"] - 1 - 1 - 2
    assert summary["steps_done"] == summary["steps_total"]
    assert "map" in results["result_keys"] and "stitch_error" not in summary
    stitching = summary["stitching"]
    assert stitching["eds_pois"] == tiles
    assert stitching["median_residual_px"] < 1.0
    assert any(p and p.startswith("acquiring tiles") for p in phases)

    store = servers[0].tools.store
    map_key = results["result_keys"]["map"]
    eds = store.list_pois(map_key, kind="eds_point")
    assert len(eds) == tiles and all(p["links"] and p["links"][0].endswith(".h5") for p in eds)
    assert summary["regions"]["count"] == 2 and all(r["scorer_version"] for r in store.list_pois(map_key, kind="candidate"))
    visited = [v for v in summary["revisits"]["first"] if "image" in v]
    for visit in visited:
        high_mag = [p for p in store.list_pois(map_key, kind="high_mag") if visit["image"] in p["links"]]
        assert len(high_mag) == 1 and f"poi:{visit['region']}" in high_mag[0]["links"]
    box, half = PLAN["box_m"], 1.5e-8
    margin = half * (1 - REVISIT_EDGE_SHIFT)  # a region this close to the edge is imaged off centre, the image kept in the box
    for visit in summary["revisits"]["first"]:
        region = next(r for r in summary["regions"]["first"] if r["id"] == visit["region"])
        inside = box[0] + margin <= region["x_m"] <= box[1] - margin and box[2] + margin <= region["y_m"] <= box[3] - margin
        assert ("image" in visit) == inside
        if "image" in visit:
            x, y = visit["stage_xy_m"]
            assert box[0] + half - 1e-12 <= x <= box[1] - half + 1e-12 and box[2] + half - 1e-12 <= y <= box[3] - half + 1e-12
            assert abs(x - region["x_m"]) <= REVISIT_EDGE_SHIFT * half + 1e-12 and abs(y - region["y_m"]) <= REVISIT_EDGE_SHIFT * half + 1e-12
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True


def test_an_analysis_server_that_is_down_leaves_the_tiles_safe_and_says_how_to_retry(mapper_proxy, mapping_twin_proxy, analysis):
    def down(mapper, tool, arguments):
        raise ConnectionError("analysis server at in-process is not answering")

    analysis["server"] = down
    results, _phases = run_job(mapper_proxy, {**PLAN, "find_regions": 2})
    assert results["state"] == "done", results
    summary = results["summary"]
    assert "map" not in results["result_keys"]
    assert "assemble_map(run=" in summary["stitch_error"] and "not answering" in summary["stitch_error"]
    run_file = Path(results["result_keys"]["run_file"])
    assert run_file.suffix == ".h5" and run_file.exists()


def test_revisit_without_regions_is_refused(mapper_proxy, analysis):
    reply = call(mapper_proxy, "reserve", {"plan": {**PLAN, "revisit": {"fov_m": 3e-8}}})
    assert reply["error"]["code"] == "plan_rejected" and "find_regions" in reply["error"]["message"]


def test_a_zoom_job_aims_images_and_places_them_tier_by_tier_until_one_is_not_placed(mapper_proxy, mapping_twin_proxy, analysis):
    stage = [float(v) for v in list(mapping_twin_proxy.get_stage())[:2]]
    mapping_twin_proxy.set_image_shift([1e-9, -2e-9])
    parent = mapping_twin_proxy.acquire_scanned_image(["haadf"])
    offsets = [(4e-8, -3e-8), (5e-9, 2e-9), (1e-9, 1e-9)]
    calls = []

    def server(mapper, tool, arguments):
        if tool == "find_features" and arguments["n"] == 0:  # the parent's frame, read at reserve
            return {"features": [], "pixel_size_m": 6e-7 / 512, "shape": [512, 512]}
        calls.append((tool, arguments))
        tier = sum(1 for t, _ in calls if t == "find_features") - 1
        if tool == "find_features":
            dx, dy = offsets[tier]
            return {"features": [{"x_m": stage[0] + dx, "y_m": stage[1] + dy, "diameter_m": arguments["min_size_m"] * 2}], "pixel_size_m": 6e-7 / 512 / 8 ** tier, "shape": [512, 512]}
        if tool == "place_image":
            if tier == 0:
                return {"status": "placed", "score": 0.99, "ambiguity": 0.4, "centre_xy_m": arguments["centre_xy_m"]}
            return {"status": "unplaced", "reason": "another place matches almost as well (0.97 of the best): a guess between places, not a placement"}
        raise AssertionError(f"unexpected analysis call {tool}")

    analysis["server"] = server
    plan = {"zoom": {"parent": parent, "fovs_m": [7.5e-8, 9.4e-9, 6e-9]}, "imsize": 64, "dwell_s": 2e-6}
    reserved = call(mapper_proxy, "reserve", {"plan": plan})
    assert reserved["estimate"]["phases"] == ["zoom"] and reserved["estimate"]["zoom_tiers"] == 3
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    results, _phases = run_job(mapper_proxy, plan)
    assert results["state"] == "done", results
    zoom = results["summary"]["zoom"]
    assert [t["tier"] for t in zoom] == [0, 1, 2]
    assert zoom[0]["parent"] == parent and zoom[0]["placement"]["status"] == "placed" and zoom[0]["image_shift_m"] == pytest.approx(list(offsets[0]))
    assert zoom[1]["parent"] == zoom[0]["image"] and zoom[1]["placement"]["status"] == "unplaced"
    assert zoom[2]["stopped"] == "tier 1 ended the zoom" and "image" not in zoom[2]
    assert results["result_keys"]["zoom_placed"] == [zoom[0]["image"]] and len(results["result_keys"]["zoom_images"]) == 2
    places = [a for t, a in calls if t == "place_image"]
    assert places[0]["window_m"] == pytest.approx(6e-7 / 16) and places[0]["centre_xy_m"] == pytest.approx([stage[0] + 4e-8, stage[1] - 3e-8])
    assert list(mapping_twin_proxy.get_image_shift())[:2] == pytest.approx([1e-9, -2e-9])  # put back as it was
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True
    assert [float(v) for v in list(mapping_twin_proxy.get_stage())[:2]] == pytest.approx(stage)  # the stage never moved
    mapping_twin_proxy.set_image_shift([0.0, 0.0])


@pytest.mark.parametrize("zoom, words", [
    ({"parent": "x.h5", "fovs_m": [7.5e-8, 9.4e-8]}, "smaller at every tier"),
    ({"parent": "x.h5", "fovs_m": [7.5e-8, 1e-9]}, "narrower than"),
])
def test_a_zoom_plan_that_widens_or_goes_too_fine_is_refused(mapper_proxy, analysis, zoom, words):
    analysis["server"] = lambda mapper, tool, arguments: {"features": [], "pixel_size_m": 6e-7 / 512, "shape": [512, 512]}
    reply = call(mapper_proxy, "reserve", {"plan": {"zoom": zoom}})
    assert reply["error"]["code"] == "plan_rejected" and words in reply["error"]["message"]
    reply = call(mapper_proxy, "reserve", {"plan": {"zoom": {"parent": "x.h5", "fovs_m": [7.5e-8]}, "fov_m": 1e-7}})
    assert "takes no area" in reply["error"]["message"]



def test_a_zoom_plan_without_fovs_steps_down_8x_from_the_parent_and_says_what_is_wrong_with_a_bad_plan(mapper_proxy, analysis):
    frames = {"survey.h5": {"features": [], "pixel_size_m": 6e-7 / 512, "shape": [512, 512]},
              "atomic.h5": {"features": [], "pixel_size_m": 1e-8 / 512, "shape": [512, 512]}}
    analysis["server"] = lambda mapper, tool, arguments: frames.get(arguments["image"]) or {"error": {"code": "not_found", "message": f"no image {arguments['image']}"}}
    reserved = call(mapper_proxy, "reserve", {"plan": {"zoom": {"parent": "survey.h5"}}})
    assert reserved["resolved_plan"]["zoom"]["fovs_m"] == pytest.approx([7.5e-8, 9.375e-9])
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    reply = call(mapper_proxy, "reserve", {"plan": {"zoom": {"parent": "atomic.h5"}}})
    assert "nothing finer to zoom to" in reply["error"]["message"] and "size_m" not in reply["error"]["message"]
    reply = call(mapper_proxy, "reserve", {"plan": {"zoom": {"parent": "missing.h5"}}})
    assert "no image missing.h5" in reply["error"]["message"]
    reply = call(mapper_proxy, "reserve", {"plan": {"zoom": {"parent": "survey.h5", "fovs_m": [7e-7]}}})
    assert "start below the parent" in reply["error"]["message"]
