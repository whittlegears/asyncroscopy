"""Twin -> Mapper -> the analysis server's stitching, scored against the twin's ground truth.

This is the test that says the twin is useful for rehearsing mapping: the
stage really errs, stitching by reported position alone is visibly wrong,
and image refinement recovers the true layout to under a pixel.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
]


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def test_stitched_map_matches_ground_truth(mapper_proxy, mapping_twin_proxy):
    from asyncroscopy.analysis_server.map_store import MapStore
    from asyncroscopy.analysis_server.map_tools import MapTools

    mapping_twin_proxy.unblank_beam()
    mapping_twin_proxy.set_defocus(0.0)
    fov, imsize = 1.0e-7, 128
    plan = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": fov, "overlap": 0.25, "imsize": imsize, "dwell_s": 2e-6}
    reserved = call(mapper_proxy, "reserve", {"plan": plan})
    started = call(mapper_proxy, "start", {"reservation_id": reserved["reservation_id"]})
    deadline = time.time() + 120
    while call(mapper_proxy, "job_status", {"job_id": started["job_id"]})["state"] not in {"done", "failed", "aborted"}:
        assert time.time() < deadline
        time.sleep(0.05)
    results = call(mapper_proxy, "results", {"job_id": started["job_id"]})
    assert results["state"] == "done", results
    run_file = results["result_keys"]["run_file"]
    n_tiles = results["summary"]["steps_total"]
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})

    # The analysis server reads local run files only under its allowed roots; the run is in a pytest temp dir.
    tools = MapTools(MapStore(local_roots=[Path(run_file).parent]))
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": run_file, "stage_sigma_m": 8e-9})["reservation_id"])
    assert tools.jobs.wait(job["job_id"], timeout=120)["state"] == "done"
    meta = tools.store.map_metadata(tools.jobs.results(job["job_id"])["result_keys"]["map"])
    positions = np.array(meta["tile_positions_px"])

    truth = json.loads(mapping_twin_proxy.export_ground_truth())
    images = [a for a in truth["acquisitions"] if a["kind"] == "image"][-n_tiles:]
    pixel = fov / imsize
    true_centres = np.array([[a["center_true_m"][1], a["center_true_m"][0]] for a in images]) / pixel  # (row, col)
    reported = np.array([[a["reported_m"][1], a["reported_m"][0]] for a in images]) / pixel

    def error(estimate_top_left):
        centres = estimate_top_left + imsize / 2.0
        diff = centres - true_centres
        diff -= np.median(diff, axis=0)
        return np.hypot(diff[:, 0], diff[:, 1])

    stage_only = error(reported - imsize / 2.0)
    refined = error(positions)
    assert np.median(stage_only) > 2.0, "twin stage error too small to test stitching"
    assert np.median(refined) < 1.0, (np.median(refined), refined)
    assert np.median(refined) < np.median(stage_only)
