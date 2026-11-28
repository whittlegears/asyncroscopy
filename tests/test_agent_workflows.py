"""The sample-map LangGraph workflow, run offline against the real pyTEMlib analysis server.

The instrument is a fake ``acquire_map_grid`` that cuts tiles from a synthetic
scene and saves them as HDF5, with its stage axes swapped like the digital
twin's (stage x moves the view down the image rows). The pyTEMlib analysis
server runs in process on a temporary folder, reached through a FastMCP
in-memory client, so the stitching, scoring and notes are the real ones.
"""

from __future__ import annotations

import asyncio
import json

import h5py
import numpy as np
import pytest

pytest.importorskip("langgraph.types")  # the stubs provide langgraph but not this
pytest.importorskip("pyTEMlib.analysis_server")

from fastmcp import Client  # noqa: E402
from pyTEMlib.analysis_server.server import create_server  # noqa: E402

from asyncroscopy.agent_workflows import sample_map  # noqa: E402
from asyncroscopy.agent_workflows.runner import WorkflowRunner  # noqa: E402
from asyncroscopy.agent_workflows.tools import ToolBox, ToolCallError  # noqa: E402
from asyncroscopy.mapping.run_file import image_frame_positions  # noqa: E402

PIXEL = 1e-9
TILE = 256


def _scene(size=900, seed=3, n_blobs=40):
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    scene = ndimage.gaussian_filter(rng.random((size, size)), 6) * 0.3
    yy, xx = np.mgrid[0:size, 0:size]
    for r, c in rng.uniform(30, size - 30, size=(n_blobs, 2)):
        scene += np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 4.0**2))
    return scene.astype(np.float32)


class FakeInstrument:
    """acquire_map_grid over a synthetic scene; stage x moves the view down the rows, stage y right."""

    def __init__(self, folder, calibrated=True, fail=None):
        self.folder, self.calibrated, self.fail = folder, calibrated, fail
        self.calls = []

    def tools(self):
        async def DigitalTwin_set_fov(fov):
            self.calls.append(("set_fov", fov))
            return None

        async def DigitalTwin_acquire_map_grid(spec):
            self.calls.append(("acquire_map_grid", spec))
            if self.fail:
                raise RuntimeError(self.fail)
            cfg = json.loads(spec)
            scene = _scene()
            step = int(TILE * (1 - cfg["overlap"]))
            rng = np.random.default_rng(5)
            tiles = []
            for r in range(cfg["rows"]):
                for c in range(cfg["cols"]):
                    y0 = 20 + r * step + int(rng.integers(-3, 4))
                    x0 = 20 + c * step + int(rng.integers(-3, 4))
                    path = self.folder / f"tile_{r}{c}.h5"
                    with h5py.File(path, "w") as h5:
                        h5["image/HAADF"] = scene[y0 : y0 + TILE, x0 : x0 + TILE]
                    stage = [y0 * PIXEL, x0 * PIXEL]  # swapped axes
                    tiles.append({"row": r, "col": c, "key": str(path), "stage_xy_m": stage, "commanded_xy_m": stage})
            manifest = {"grid": [cfg["rows"], cfg["cols"]], "overlap": cfg["overlap"], "fov_m": TILE * PIXEL, "imsize_px": TILE,
                        "pixel_size_nm": PIXEL * 1e9, "stage_origin_m": tiles[0]["stage_xy_m"], "tiles": tiles}
            if self.calibrated:
                manifest["calibration"] = {"px_per_m": [[-1 / PIXEL, 0.0], [0.0, -1 / PIXEL]], "measured": True}
            return json.dumps(manifest)

        return {"DigitalTwin_set_fov": DigitalTwin_set_fov, "DigitalTwin_acquire_map_grid": DigitalTwin_acquire_map_grid}


def _analysis_tools(client, names):
    def make(name):
        async def call(**arguments):
            result = await client.call_tool(name, arguments, raise_on_error=True)
            if result.structured_content is not None:
                data = result.structured_content
                return data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data
            return json.loads(result.content[0].text)

        return call

    return {name: make(name) for name in names}


async def _run(tmp_path, instrument, answers, request=None):
    """Run the graph to the end, answering each pause with the next answer. Returns (report, pauses, client results)."""
    mcp, _tools = create_server(local_roots=[str(tmp_path)], annotations_path=str(tmp_path / "notes.sqlite"))
    async with Client(mcp) as client:
        names = [t.name for t in await client.list_tools()]
        tools = {**_analysis_tools(client, names), **instrument.tools()}
        assert sample_map.missing_tools(tools, local_runs=True) == []
        runner = WorkflowRunner()
        runner.register("sample_map", lambda cp: sample_map.build_graph(tools, checkpointer=cp, run_dir=tmp_path / "runs"))
        step = await runner.start("sample_map", {"request": {"name": "demo", "rows": 3, "cols": 3, "overlap": 0.2,
                                                             "fov_m": TILE * PIXEL, "top_k": 6, "shortlist_size": 3, **(request or {})}})
        pauses = []
        while step["waiting_for"] is not None:
            pauses.append(step["waiting_for"])
            step = await runner.resume(step["thread_id"], answers(step["waiting_for"]))
        notes = None
        if step["state"].get("map"):
            notes = await _analysis_tools(client, ["list_annotations"])["list_annotations"](dataset=step["state"]["map"])
        return step, pauses, notes


def _answer_all(actor="human:operator"):
    def answers(pause):
        if pause["kind"] == "approve_scan":
            return {"approved": True}
        first, *rest = pause["shortlist"]
        return {"actor": actor, "decisions": [{"id": first["id"], "status": "accepted", "note": "real feature"}]
                + [{"id": p["id"], "status": "rejected", "note": "edge"} for p in rest]}

    return answers


def test_full_run_stitches_scores_and_records_the_review(tmp_path):
    instrument = FakeInstrument(tmp_path)
    step, pauses, notes = asyncio.run(_run(tmp_path, instrument, _answer_all()))
    report = step["state"]["report"]
    assert [p["kind"] for p in pauses] == ["approve_scan", "review_candidates"]
    assert report["status"] == "done", report["errors"]
    stitch = report["stitch"]
    assert stitch["tiles"] == 9
    assert stitch["matches_used"] == stitch["matches_found"] >= 12
    assert stitch["median_residual_px"] < 0.5
    assert stitch["fallback_tiles"] == 0 and stitch["unplaced_tiles"] == []
    assert report["candidates"] >= 3
    kinds = sorted(n["kind"] for n in notes["notes"])
    assert kinds == ["accepted", "rejected", "rejected"]
    assert {n["author"] for n in notes["notes"]} == {"human:operator"}
    accepted = next(n for n in notes["notes"] if n["kind"] == "accepted")
    assert accepted["links"] and accepted["links"][0] in {d["id"] for d in report["decisions"]}
    assert ("set_fov", TILE * PIXEL) in instrument.calls


def test_calibration_puts_tiles_where_pytemlib_expects_them(tmp_path):
    """Without the conversion the swapped stage puts tile (0, 1) below tile (0, 0); pyTEMlib then cannot match it."""
    calibrated, _, _ = asyncio.run(_run(tmp_path / "a", FakeInstrument(_mkdir(tmp_path / "a")), _answer_all()))
    raw, _, _ = asyncio.run(_run(tmp_path / "b", FakeInstrument(_mkdir(tmp_path / "b"), calibrated=False), _answer_all()))
    good, bad = calibrated["state"]["report"]["stitch"], raw["state"]["report"]["stitch"]
    print(f"\nwith calibration: {good['matches_used']}/{good['matches_found']} matches, median {good['median_residual_px']} px; "
          f"raw stage axes: {bad.get('matches_used')}/{bad.get('matches_found')} matches, fallback tiles {bad.get('fallback_tiles')}")
    assert good["matches_used"] > (bad.get("matches_used") or 0)


def test_declined_scan_never_moves_the_stage(tmp_path):
    instrument = FakeInstrument(tmp_path)
    step, pauses, _ = asyncio.run(_run(tmp_path, instrument, lambda pause: {"approved": False}))
    assert [p["kind"] for p in pauses] == ["approve_scan"]
    assert step["finished"] and step["state"]["status"] == "declined"
    assert instrument.calls == []


def test_instrument_failure_is_reported_not_raised(tmp_path):
    instrument = FakeInstrument(tmp_path, fail="Stage-to-image calibration failed: stage x moved 1.2 nm but the image moved 21.2 px")
    step, _, _ = asyncio.run(_run(tmp_path, instrument, _answer_all()))
    report = step["state"]["report"]
    assert report["status"] == "failed"
    assert "calibration failed" in report["errors"][0]
    assert report["map"] is None


def test_bad_request_fails_before_any_pause(tmp_path):
    step, pauses, _ = asyncio.run(_run(tmp_path, FakeInstrument(tmp_path), _answer_all(), request={"overlap": 0.9}))
    assert pauses == []
    assert step["state"]["report"]["status"] == "failed"


def test_image_frame_positions_undo_the_swapped_axes():
    stage = [[0.0, 0.0], [0.0, 200e-9], [200e-9, 0.0]]  # tile (0,1) is +y on this stage, tile (1,0) is +x
    centres = image_frame_positions(stage, [[-1 / PIXEL, 0.0], [0.0, -1 / PIXEL]], PIXEL)
    assert centres[1] == pytest.approx([200e-9, 0.0])  # beside: +x in the image frame
    assert centres[2] == pytest.approx([0.0, 200e-9])  # below: +y


def test_toolbox_finds_prefixed_tools_and_wraps_errors():
    async def DigitalTwin_move_stage(position):
        raise RuntimeError("DevFailed: out of range")

    box = ToolBox({"DigitalTwin_move_stage": DigitalTwin_move_stage, "AutoScriptMicroscope_move_stage": DigitalTwin_move_stage},
                  prefer=["DigitalTwin"])
    assert box.find("move_stage") == "DigitalTwin_move_stage"
    with pytest.raises(ToolCallError, match="out of range"):
        asyncio.run(box.call("move_stage", position=[0] * 5))


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_small_calibrated_grids_get_the_identity_scan_linear():
    calibrated = {"calibration": {"px_per_m": [[1, 0], [0, 1]]}, "tiles": [{}] * 9}
    assert sample_map.assemble_arguments({}, calibrated) == {"scan_linear": [[1.0, 0.0], [0.0, 1.0]]}
    assert sample_map.assemble_arguments({"scan_linear": "measure"}, calibrated) == {}
    assert sample_map.assemble_arguments({}, {**calibrated, "tiles": [{}] * 30}) == {}
    assert sample_map.assemble_arguments({}, {"tiles": [{}] * 9}) == {}  # no calibration: let pyTEMlib measure
    assert sample_map.assemble_arguments({"stage_sigma_m": 1e-10, "scan_linear": [[1, 0.01], [0, 1]]}, calibrated) == {
        "stage_sigma_m": 1e-10, "scan_linear": [[1.0, 0.01], [0.0, 1.0]]}
