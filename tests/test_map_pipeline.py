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

from asyncroscopy.instruments.electron_microscope.mapper import Mapper

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
    assert len(summary["regions"]) == 2 and all(r["scorer_version"] for r in summary["regions"])
    visited = [v for v in summary["revisits"] if "image" in v]
    for visit in visited:
        high_mag = [p for p in store.list_pois(map_key, kind="high_mag") if visit["image"] in p["links"]]
        assert len(high_mag) == 1 and f"poi:{visit['region']}" in high_mag[0]["links"]
    box = PLAN["box_m"]
    for visit in summary["revisits"]:
        region = next(r for r in summary["regions"] if r["id"] == visit["region"])
        inside = box[0] + 1.5e-8 <= region["x_m"] <= box[1] - 1.5e-8 and box[2] + 1.5e-8 <= region["y_m"] <= box[3] - 1.5e-8
        assert ("image" in visit) == inside  # only regions a 30 nm image fits around, inside the box, are visited
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
