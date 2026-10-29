"""The Mapper's pipeline when things go wrong: dose of revisits, regions at the box edge, an analysis
server that forgets jobs or never finishes, resumes mid-stitch and mid-revisit, and the real wire format.

Most tests drive the Mapper against pyTEMlib's real MapTools in this process (``InProcessAnalysis``),
with hooks that hold or break a call; the last test talks to a pyTEMlib MCP server over HTTP.
"""

import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from asyncroscopy.instruments.electron_microscope import mapper as mapper_module
from asyncroscopy.instruments.electron_microscope.mapper import Mapper, drift_at

from test_map_pipeline import PLAN, InProcessAnalysis, call

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("pyTEMlib.mcp_server.map_tools") is None, reason="needs the pyTEMlib fork with map tools"),
]


class Analysis:
    """The real map tools in this process, plus hooks: ``before[tool](arguments)`` may block, raise or answer instead."""

    def __init__(self):
        self.server = None
        self.before = {}
        self.calls = []

    def __call__(self, mapper, tool, arguments):
        self.calls.append(tool)
        if self.server is None and "run" in arguments:
            self.server = InProcessAnalysis([Path(arguments["run"]).parent])
        hook = self.before.get(tool)
        if hook is not None:
            answer = hook(arguments)
            if answer is not None:
                return answer
        return self.server(mapper, tool, arguments)


@pytest.fixture
def analysis(monkeypatch):
    holder = Analysis()
    monkeypatch.setattr(Mapper, "analysis_url", "in-process")
    monkeypatch.setattr(Mapper, "_analysis", lambda self, tool, arguments: holder(self, tool, arguments))
    monkeypatch.setattr(mapper_module, "ANALYSIS_BACKOFF_S", (0.05, 0.05))
    return holder


def start(mapper_proxy, plan):
    reserved = call(mapper_proxy, "reserve", {"plan": plan})
    assert "reservation_id" in reserved, reserved
    return reserved, call(mapper_proxy, "start", {"reservation_id": reserved["reservation_id"]})


def wait(mapper_proxy, job_id, until=("done", "failed", "aborted"), timeout=300):
    deadline = time.time() + timeout
    while True:
        status = call(mapper_proxy, "job_status", {"job_id": job_id})
        if status["state"] in until:
            return status
        assert time.time() < deadline, status
        time.sleep(0.05)


def images_taken(twin, since):
    truth = json.loads(twin.export_ground_truth())
    return [a for a in truth["acquisitions"] if a.get("kind") == "image"][since:]


def image_count(twin):
    return sum(1 for a in json.loads(twin.export_ground_truth())["acquisitions"] if a.get("kind") == "image")


def test_revisits_count_in_the_estimate_and_against_the_dose_budget(mapper_proxy, analysis):
    plan = {**PLAN, "find_regions": 2, "revisit": {"fov_m": 2e-8, "imsize": 128, "dwell_s": 4e-6}}
    reserved = call(mapper_proxy, "reserve", {"plan": plan})
    estimate = reserved["estimate"]
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    tiles_only = call(mapper_proxy, "reserve", {"plan": {**PLAN, "stitch": False}})
    call(mapper_proxy, "release", {"reservation_id": tiles_only["reservation_id"]})
    assert estimate["revisit_dose_e_per_a2"] > 0
    assert estimate["max_dose_e_per_a2"] == pytest.approx(tiles_only["estimate"]["max_dose_e_per_a2"] + estimate["revisit_dose_e_per_a2"], rel=1e-3)
    assert estimate["duration_s"] > tiles_only["estimate"]["duration_s"]
    refused = call(mapper_proxy, "reserve", {"plan": {**plan, "dose_budget_e_per_a2": tiles_only["estimate"]["max_dose_e_per_a2"] * 1.01}})
    assert refused["error"]["code"] == "plan_rejected" and "revisit image" in refused["error"]["message"]
    too_small = call(mapper_proxy, "reserve", {"plan": {**PLAN, "find_regions": 1, "revisit": {"fov_m": PLAN["fov_m"] / 50}}})
    assert too_small["error"]["code"] == "plan_rejected" and "twentieth" in too_small["error"]["message"]


def test_a_region_at_the_edge_of_the_box_is_not_visited(mapper_proxy, mapping_twin_proxy, analysis):
    x_max, y_max = PLAN["box_m"][1], PLAN["box_m"][3]

    def edge_region(arguments):
        jobs = analysis.server.tools.jobs
        reserved = jobs.reserve({"task": "score_map", **arguments, "top_k": 1})
        started = jobs.start(reserved["reservation_id"])
        jobs.wait(started["job_id"])
        results = jobs._find(started["job_id"])
        results["summary"]["added_pois"] = [
            {"id": "edge", "x_m": x_max - 1e-9, "y_m": y_max - 1e-9, "score": 1.0, "reason": "at the corner", "scorer_version": "test", "window_m": 1e-8},
            {"id": "inside", "x_m": 0.0, "y_m": 0.0, "score": 0.9, "reason": "in the middle", "scorer_version": "test", "window_m": 1e-8},
        ]
        return started

    analysis.before["score_map"] = edge_region
    before = image_count(mapping_twin_proxy)
    reserved, started = start(mapper_proxy, {**PLAN, "find_regions": 2, "revisit": {"fov_m": 3e-8}})
    assert wait(mapper_proxy, started["job_id"])["state"] == "done"
    summary = call(mapper_proxy, "results", {"job_id": started["job_id"]})["summary"]
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    by_region = {v["region"]: v for v in summary["revisits"]["first"]}
    assert "skipped" in by_region["edge"] and "image" in by_region["inside"]
    tiles = summary["steps_total"] - 4
    assert image_count(mapping_twin_proxy) - before == tiles + 1  # the edge region cost no image


def test_an_analysis_server_that_forgets_the_job_ends_the_stitch_with_a_clear_error(mapper_proxy, analysis):
    analysis.before["analysis_job_status"] = lambda arguments: {"schema_version": "1.1.0", "error": {"code": "unknown_job", "message": "no analysis job"}}
    reserved, started = start(mapper_proxy, {**PLAN, "find_regions": 1})
    t0 = time.time()
    assert wait(mapper_proxy, started["job_id"])["state"] == "done"
    summary = call(mapper_proxy, "results", {"job_id": started["job_id"]})["summary"]
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    assert "restarted" in summary["stitch_error"] and "assemble_map(run=" in summary["stitch_error"]
    assert summary.get("score_error") is None and "regions" not in summary
    assert time.time() - t0 < 60


def test_a_stitch_that_never_finishes_fails_at_its_deadline(mapper_proxy, analysis, monkeypatch):
    monkeypatch.setattr(mapper_module, "STITCH_DEADLINE_S", (2.0, 0.0))
    analysis.before["analysis_job_status"] = lambda arguments: (time.sleep(0.2), {"schema_version": "1.1.0", "job_id": arguments["job_id"], "state": "running", "progress": {"done": 1, "total": 1000}})[1]
    reserved, started = start(mapper_proxy, PLAN)
    assert wait(mapper_proxy, started["job_id"])["state"] == "done"
    summary = call(mapper_proxy, "results", {"job_id": started["job_id"]})["summary"]
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    assert "did not finish within 2 s" in summary["stitch_error"]


def test_resuming_mid_stitch_waits_for_the_same_stitch_and_mid_revisit_links_without_a_second_image(mapper_proxy, mapping_twin_proxy, analysis):
    hold = threading.Event()
    seen_status = threading.Event()

    def held_status(arguments):
        seen_status.set()
        hold.wait(30)
        return None  # then the real status

    analysis.before["analysis_job_status"] = held_status
    before = image_count(mapping_twin_proxy)
    plan = {**PLAN, "per_tile": ["eds_point"], "eds_exposure_s": 0.05, "find_regions": 2, "revisit": {"fov_m": 3e-8}}
    reserved, started = start(mapper_proxy, plan)
    assert seen_status.wait(120)
    call(mapper_proxy, "abort", {"job_id": started["job_id"]})
    hold.set()
    assert wait(mapper_proxy, started["job_id"])["state"] == "aborted"
    del analysis.before["analysis_job_status"]

    # Break the first link of a revisit image: the job aborts after the image is taken and saved.
    adds = []

    def abort_on_first_link(arguments):
        adds.append(arguments["id"])
        if len(adds) == 1:
            call(mapper_proxy, "abort", {"job_id": resumed["job_id"]})
            raise ConnectionError("link lost")
        return None

    analysis.before["add_poi"] = abort_on_first_link
    resumed = call(mapper_proxy, "start", {"reservation_id": reserved["reservation_id"]})
    assert wait(mapper_proxy, resumed["job_id"])["state"] in {"aborted", "done"}
    final = call(mapper_proxy, "start", {"reservation_id": reserved["reservation_id"]})
    if "error" not in final:
        assert wait(mapper_proxy, final["job_id"])["state"] == "done"
        job_id = final["job_id"]
    else:
        job_id = resumed["job_id"]
    results = call(mapper_proxy, "results", {"job_id": job_id})
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    summary = results["summary"]
    assert analysis.calls.count("assemble_map") == 1  # the resumed job waited for the stitch it had started
    assert analysis.calls.count("score_map") == 1
    visited = [v for v in summary["revisits"]["first"] if "image" in v]
    tiles = summary["steps_total"] - 4
    assert image_count(mapping_twin_proxy) - before == tiles + len(visited)  # no region was imaged twice
    store = analysis.server.tools.store
    high_mag = store.list_pois(results["result_keys"]["map"], kind="high_mag")
    assert len(high_mag) == len(visited) and len({p["id"] for p in high_mag}) == len(visited)
    assert len(store.list_pois(results["result_keys"]["map"], kind="candidate")) == 2


def test_drift_is_interpolated_inside_the_fit_and_extrapolated_after_it():
    drift = {"times_s": [0.0, 100.0, 200.0], "xy_m": [[0.0, 0.0], [1e-9, -2e-9], [3e-9, -2e-9]]}
    assert drift_at(drift, 50.0) == pytest.approx((0.5e-9, -1e-9))
    assert drift_at(drift, 300.0) == pytest.approx((5e-9, -2e-9))
    assert drift_at(None, 10.0) == (0.0, 0.0)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_pipeline_over_the_real_mcp_wire_format(mapper_proxy, mapping_twin_proxy, tmp_path_factory, monkeypatch):
    import uvicorn

    from pyTEMlib.mcp_server import create_server

    # The run file is written under the test's temporary folders: let this server read that drive.
    mcp, _tools = create_server(include_basic=False, local_roots=[Path(Path.cwd().anchor)])
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(mcp.http_app(path="/mcp"), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert time.time() < deadline
        time.sleep(0.05)
    try:
        monkeypatch.setattr(Mapper, "analysis_url", f"http://127.0.0.1:{port}/mcp")
        reserved, started = start(mapper_proxy, {**PLAN, "per_tile": ["eds_point"], "eds_exposure_s": 0.05, "find_regions": 1, "revisit": {"fov_m": 3e-8}})
        assert wait(mapper_proxy, started["job_id"])["state"] == "done"
        results = call(mapper_proxy, "results", {"job_id": started["job_id"]})
        call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
        summary = results["summary"]
        assert "stitch_error" not in summary, summary.get("stitch_error")
        assert results["result_keys"]["map"] and summary["stitching"]["eds_pois"] == summary["steps_total"] - 3
        assert summary["regions"]["count"] == 1 and summary["regions"]["first"][0]["score"] is not None
        assert all("poi" in v for v in summary["revisits"]["first"] if "image" in v)
    finally:
        server.should_exit = True
        thread.join(10)
