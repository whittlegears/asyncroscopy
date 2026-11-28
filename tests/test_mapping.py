"""Grid scans: the stage-to-image calibration, and the digital twin's grid stitched by pyTEMlib."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asyncroscopy.mapping.run_file import write_map_run


def _make_scene(h=700, w=700, seed=11):
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    scene = ndimage.gaussian_filter(rng.random((h, w)), 4)
    scene = (scene - scene.min()) / (scene.max() - scene.min())
    return np.clip(scene * 220 + rng.normal(0, 5, scene.shape), 0, 255).astype(np.uint8)


class TestStageCalibration:
    PIXEL = 0.05e-9

    def _fake_scope(self, px_per_m, scene=None):
        """acquire/move callables over a random scene with a known stage-to-image matrix."""
        scene = _make_scene(600, 600, seed=4).astype(float) if scene is None else scene
        state = {"pos": np.zeros(5)}
        matrix = np.asarray(px_per_m, dtype=float)

        def move(position):
            state["pos"] = np.asarray(position, dtype=float)

        def acquire():
            shift_row, shift_col = matrix @ state["pos"][:2]
            # Content moves by +shift, so the view moves by -shift.
            r0, c0 = int(round(200 - shift_row)), int(round(200 - shift_col))
            return scene[r0 : r0 + 128, c0 : c0 + 128]

        return acquire, move, state

    def test_recovers_a_rotated_mirrored_matrix(self):
        from asyncroscopy.mapping.calibration import measure_stage_to_image

        scale = 1 / self.PIXEL
        angle = np.radians(30)
        truth = scale * np.array([[np.cos(angle), np.sin(angle)], [np.sin(angle), -np.cos(angle)]])
        acquire, move, state = self._fake_scope(truth)
        start = [1e-9, -2e-9, 0, 0, 0]
        cal = measure_stage_to_image(acquire, move, start, self.PIXEL, step_m=25 * self.PIXEL)
        assert np.asarray(cal.px_per_m) == pytest.approx(truth, rel=0.05, abs=0.05 * scale)
        assert list(state["pos"]) == start

    def test_grid_neighbours_land_beside_each_other(self):
        from asyncroscopy.mapping.calibration import StageCalibration, plan_calibrated_grid

        cal = StageCalibration.from_matrix([[0.0, -2e10], [-2e10, 0.0]], self.PIXEL)
        order = {(e["row"], e["col"]): e["offset_m"] for e in plan_calibrated_grid(2, 2, 100, cal)}
        assert cal.image_shift(*order[(0, 1)]) == pytest.approx((0.0, -100.0))
        assert cal.image_shift(*order[(1, 0)]) == pytest.approx((-100.0, 0.0))

    def test_nominal_is_the_old_assumption(self):
        """calibrate: false keeps the old convention: stage x moves the view right, stage y down."""
        from asyncroscopy.mapping.calibration import StageCalibration, plan_calibrated_grid

        step_px = 80
        order = {(e["row"], e["col"]): e["offset_m"] for e in plan_calibrated_grid(2, 3, step_px, StageCalibration.nominal(self.PIXEL, flip_x=True))}
        assert [(r, c) for r, c in order] == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]  # serpentine
        assert order[(0, 1)] == pytest.approx([-step_px * self.PIXEL, 0.0], abs=1e-15)  # flip_x: -x
        assert order[(1, 0)] == pytest.approx([0.0, step_px * self.PIXEL], abs=1e-15)

    def test_refuses_a_shift_of_the_wrong_length(self):
        from asyncroscopy.mapping.calibration import CalibrationError, measure_stage_to_image

        # The image moves half as far as the pixel size says it should.
        acquire, move, state = self._fake_scope(0.5 / self.PIXEL * np.array([[0, -1.0], [-1.0, 0]]))
        with pytest.raises(CalibrationError, match="moved"):
            measure_stage_to_image(acquire, move, [0, 0, 0, 0, 0], self.PIXEL, step_m=25 * self.PIXEL)
        assert list(state["pos"]) == [0, 0, 0, 0, 0]


class TestAcquireMapGridDigitalTwin:
    # 10 nm tiles of 256 px. Smaller tiles (64-128 px) leave too few pixels in each
    # overlap for the correlation, and the twin's lattice then fools it.
    FOV = 10e-9
    CAL_FOV = 20e-9  # the test moves see particle edges at 20 nm, not only lattice
    IMSIZE = 256
    START = [-7.5e-9, -7.5e-9, 0.0, 0.0, 0.0]  # keeps a 3 x 3 grid on the 30 nm sample

    def _scan(self, twin, scan_proxy, spec, fov=None, imsize=None):
        scan_proxy.imsize = imsize or self.IMSIZE
        scan_proxy.dwell_time = 1e-6
        scan_proxy.output_format = ".h5"
        scan_proxy.scan_region = [0.0, 0.0, 1.0, 1.0]
        twin.set_fov(fov or self.FOV)
        twin.move_stage(self.START)
        start = list(twin.get_stage())
        try:
            return start, json.loads(twin.acquire_map_grid(json.dumps(spec)))
        finally:
            twin.move_stage([0.0] * 5)

    def test_calibrated_grid_puts_tiles_in_image_order(self, mapping_twin_proxy, scan_proxy):
        _start, spec = self._scan(
            mapping_twin_proxy, scan_proxy, {"rows": 3, "cols": 3, "overlap": 0.25, "calibration_fov_m": self.CAL_FOV}
        )
        pixel = self.FOV / self.IMSIZE
        # Stage x moves the content up the rows, stage y left along the columns.
        expected = [[-1 / pixel, 0.0], [0.0, -1 / pixel]]
        assert np.asarray(spec["calibration"]["px_per_m"]) == pytest.approx(np.asarray(expected), rel=0.15, abs=0.15 / pixel)
        assert spec["calibration"]["calibration_fov_m"] == pytest.approx(self.CAL_FOV)
        assert {(t["row"], t["col"]) for t in spec["tiles"]} == {(r, c) for r in range(3) for c in range(3)}
        for t in spec["tiles"]:
            assert Path(t["key"]).exists()
            assert len(t["stage_xy_m"]) == len(t["commanded_xy_m"]) == 2

    def test_pytemlib_stitches_the_twin_grid_to_the_true_stage(self, mapping_twin_proxy, scan_proxy, tmp_path):
        """The twin's grid, written as a run and stitched by pyTEMlib, against where the stage really put each tile."""
        import asyncio

        pytest.importorskip("pyTEMlib.analysis_server")
        pytest.importorskip("langgraph.types")
        from fastmcp import Client

        from asyncroscopy.agent_workflows.sample_map import assemble_arguments
        from pyTEMlib.analysis_server.server import create_server

        _start, spec = self._scan(
            mapping_twin_proxy, scan_proxy, {"rows": 3, "cols": 3, "overlap": 0.25, "calibration_fov_m": self.CAL_FOV}
        )
        run = write_map_run(tmp_path / "runs" / "twin.h5", spec)
        mcp, tools = create_server(local_roots=[str(tmp_path)])

        async def stitch():
            async with Client(mcp) as client:
                # The workflow's own choice of arguments: the identity scan_linear for a small grid.
                job = (await client.call_tool("assemble_map", {"run": str(run), **assemble_arguments({}, spec)})).structured_content
                await client.call_tool("analysis_job_status", {"job_id": job["job_id"], "wait_s": 25})
                return (await client.call_tool("analysis_job_results", {"job_id": job["job_id"]})).structured_content

        results = asyncio.run(stitch())
        meta = tools.store.map_metadata(results["result_keys"]["map"])
        placed = np.asarray(meta["tile_positions_px"])  # (row, col) per tile, in run order
        pixel = self.FOV / self.IMSIZE
        actual = np.asarray([t["stage_xy_m"] for t in spec["tiles"]])
        # The twin draws stage x down the rows and stage y across the columns.
        truth = (actual - actual[0]) / pixel
        errors = np.hypot(*((placed - placed[0]) - truth).T)
        summary = results["summary"]
        print(f"\ntwin 3x3 via pyTEMlib: max {errors.max():.2f} px, mean {errors.mean():.2f} px ({pixel * 1e12:.0f} pm/px); "
              f"matches {summary['matches_used']}/{summary['matches_found']}, fallback tiles {summary['fallback_tiles']}")
        assert summary["unplaced_tiles"] == []
        assert errors.max() < 1.5  # was up to 9 px before the identity scan_linear

    def test_known_calibration_skips_the_test_moves(self, mapping_twin_proxy, scan_proxy):
        pixel = self.FOV / self.IMSIZE
        matrix = [[-1 / pixel, 0.0], [0.0, -1 / pixel]]
        _start, spec = self._scan(
            mapping_twin_proxy, scan_proxy, {"rows": 1, "cols": 2, "stage_to_image_px_per_m": matrix}
        )
        assert spec["calibration"]["measured"] is False
        assert len(spec["tiles"]) == 2

    def test_lattice_at_the_mapping_fov_stops_before_the_grid(self, mapping_twin_proxy, scan_proxy):
        import tango

        # 6 nm at 128 px: mostly lattice. The test moves lock one lattice period off.
        with pytest.raises(tango.DevFailed, match="calibration failed"):
            self._scan(mapping_twin_proxy, scan_proxy, {"rows": 2, "cols": 2}, fov=6e-9, imsize=128)

    def test_featureless_view_stops_before_the_grid(self, mapping_twin_proxy, scan_proxy):
        import tango

        off_sample = [1e-6, 1e-6, 0.0, 0.0, 0.0]  # vacuum: nothing to correlate
        scan_proxy.imsize = 128
        mapping_twin_proxy.set_fov(self.FOV)
        mapping_twin_proxy.move_stage(off_sample)
        try:
            with pytest.raises(tango.DevFailed):
                mapping_twin_proxy.acquire_map_grid(json.dumps({"rows": 2, "cols": 2}))
            assert list(mapping_twin_proxy.get_stage())[:2] == pytest.approx(off_sample[:2], abs=1e-10)
        finally:
            mapping_twin_proxy.move_stage([0.0] * 5)

    def test_rejects_bad_spec(self, twin_proxy):
        import tango

        with pytest.raises(tango.DevFailed):
            twin_proxy.acquire_map_grid(json.dumps({"rows": 0, "cols": 3}))
        with pytest.raises(tango.DevFailed):
            twin_proxy.acquire_map_grid(json.dumps({"rows": 2, "cols": 2, "overlap": 0.9}))
