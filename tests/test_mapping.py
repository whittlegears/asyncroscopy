"""Tests for asyncroscopy.mapping (grid stitching + sample map bundles)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asyncroscopy.mapping import parse_map_grid_result, plan_grid
from asyncroscopy.mapping.grid_stitcher import GridStitcher
from asyncroscopy.mapping.montage import build_sample_map, load_acquisition_image
from asyncroscopy.mapping.overview import register_overview


def _make_scene(h=700, w=700, seed=11):
    import cv2

    rng = np.random.default_rng(seed)
    scene = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 4)
    scene = (scene - scene.min()) / (scene.max() - scene.min())
    return np.clip(scene * 220 + rng.normal(0, 5, scene.shape), 0, 255).astype(np.uint8)


def _make_shapes_scene(size=1200, seed=5):
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((size, size), 40, dtype=np.uint8)
    for _ in range(220):
        x, y = rng.integers(0, size, 2)
        r = int(rng.integers(8, 40))
        color = int(rng.integers(60, 230))
        if rng.random() < 0.5:
            cv2.circle(img, (int(x), int(y)), r, color, -1)
        else:
            w, h = rng.integers(10, 50, 2)
            cv2.rectangle(img, (int(x), int(y)), (int(x + int(w)), int(y + int(h))), color, -1)
    return img


def _write_png(path, image):
    import cv2

    cv2.imwrite(str(path), image)


def _grid_tiles(scene, rows=2, cols=2, overlap=0.2, jitter=5, seed=3):
    rng = np.random.default_rng(seed)
    h, w = scene.shape
    th = int(h / (rows - (rows - 1) * overlap))
    tw = int(w / (cols - (cols - 1) * overlap))
    step_y, step_x = int(th * (1 - overlap)), int(tw * (1 - overlap))
    tiles = {}
    for r in range(rows):
        for c in range(cols):
            jx = int(rng.integers(-jitter, jitter + 1)) if (r, c) != (0, 0) else 0
            jy = int(rng.integers(-jitter, jitter + 1)) if (r, c) != (0, 0) else 0
            y0 = min(max(0, r * step_y + jy), h - th)
            x0 = min(max(0, c * step_x + jx), w - tw)
            tiles[(r, c)] = scene[y0: y0 + th, x0: x0 + tw].copy()
    return tiles


class TestPlanGrid:
    def test_snake_order(self):
        order = plan_grid(2, 3, 1e-6, 1e-6, snake=True)
        cols = [cell["col"] for cell in order]
        assert cols == [0, 1, 2, 2, 1, 0]

    def test_offsets_and_flips(self):
        order = plan_grid(2, 2, 2e-6, 3e-6, snake=False, flip_x=True)
        by_rc = {(cell["row"], cell["col"]): cell["offset_m"] for cell in order}
        assert by_rc[(0, 1)] == [-2e-6, 0.0]
        assert by_rc[(1, 0)] == [0.0, 3e-6]


class TestLoadAcquisitionImage:
    def test_h5_prefers_haadf(self, tmp_path):
        path = tmp_path / "scan.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("image/BF", data=np.zeros((8, 8)))
            h5.create_dataset("image/HAADF", data=np.ones((8, 8)))
        assert load_acquisition_image(path).mean() == 1.0

    def test_tiff(self, tmp_path):
        import tifffile

        path = tmp_path / "scan.tiff"
        tifffile.imwrite(path, np.full((6, 6), 7, dtype=np.uint16))
        assert load_acquisition_image(path)[0, 0] == 7

    def test_h5_without_image_raises(self, tmp_path):
        path = tmp_path / "spec.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("spectrum", data=np.zeros(100))
        with pytest.raises(ValueError):
            load_acquisition_image(path)


class TestParseMapGridResult:
    def test_roundtrip(self):
        payload = json.dumps({"tiles": [{"row": 0, "col": 0, "key": "k"}]})
        assert parse_map_grid_result(payload)["tiles"][0]["key"] == "k"

    def test_missing_tiles_raises(self):
        with pytest.raises(ValueError):
            parse_map_grid_result("{}")

    def test_malformed_tile_raises(self):
        with pytest.raises(ValueError):
            parse_map_grid_result(json.dumps({"tiles": [{"row": 0}]}))


class TestBuildSampleMap:
    def test_end_to_end_bundle(self, tmp_path):
        scene = _make_scene()
        tiles = _grid_tiles(scene, rows=2, cols=2, overlap=0.2)
        store = tmp_path / "acq"
        store.mkdir()
        entries = []
        for (r, c), img in tiles.items():
            key = f"scan_HAADF_r{r}c{c}.h5"
            with h5py.File(store / key, "w") as h5:
                h5.create_dataset("image/HAADF", data=img)
            entries.append(
                {"row": r, "col": c, "key": key,
                 "stage_xy_m": [c * 1e-6, r * 1e-6],
                 "eds_key": f"eds_r{r}c{c}" if (r, c) == (0, 0) else None}
            )

        summary = build_sample_map(
            entries,
            resolve_key=lambda key: store / key,
            out_dir=tmp_path / "maps" / "demo",
            name="demo",
            overlap=0.2,
            pixel_size_nm=0.05,
        )

        assert summary["tiles_stitched"] == 4
        assert summary["edges_measured"] == 4
        assert summary["mean_edge_residual_px"] < 1.0

        bundle = Path(summary["bundle_path"])
        manifest = json.loads((bundle / "map.json").read_text())
        assert manifest["format"] == "sciagent-map/1.1"
        assert manifest["pixel_size_nm"] == 0.05
        assert len(manifest["source_tiles"]) == 4
        eds = [a for a in manifest["annotations"] if a["type"] == "eds_point"]
        assert len(eds) == 1
        assert (bundle / "index.html").exists()
        assert (bundle / "tiles" / "0" / "0" / "0.png").exists()

    def test_stitch_accuracy(self):
        scene = _make_scene()
        tiles = _grid_tiles(scene, rows=3, cols=3, overlap=0.2)
        result = GridStitcher(overlap=0.2).stitch(tiles)
        assert sum(e.used for e in result.edges) == 12
        assert result.mean_residual < 0.5


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

    def test_nominal_matches_the_old_plan_grid(self):
        from asyncroscopy.mapping.calibration import StageCalibration, plan_calibrated_grid

        step_px = 80
        cal = StageCalibration.nominal(self.PIXEL, flip_x=True)
        new = plan_calibrated_grid(2, 3, step_px, cal)
        old = plan_grid(2, 3, step_px * self.PIXEL, step_px * self.PIXEL, flip_x=True)
        assert [(e["row"], e["col"]) for e in new] == [(e["row"], e["col"]) for e in old]
        for a, b in zip(new, old):
            assert a["offset_m"] == pytest.approx(b["offset_m"], abs=1e-15)

    def test_refuses_a_shift_of_the_wrong_length(self):
        from asyncroscopy.mapping.calibration import CalibrationError, measure_stage_to_image

        # The image moves half as far as the pixel size says it should.
        acquire, move, state = self._fake_scope(0.5 / self.PIXEL * np.array([[0, -1.0], [-1.0, 0]]))
        with pytest.raises(CalibrationError, match="moved"):
            measure_stage_to_image(acquire, move, [0, 0, 0, 0, 0], self.PIXEL, step_m=25 * self.PIXEL)
        assert list(state["pos"]) == [0, 0, 0, 0, 0]


class TestStitchAgainstTruth:
    def test_positions_match_where_the_tiles_were_cut(self):
        scene = _make_shapes_scene(900, seed=8)
        rng = np.random.default_rng(1)
        tile, overlap = 256, 0.2
        step = int(tile * (1 - overlap))
        tiles, truth = {}, {}
        for r in range(3):
            for c in range(3):
                y0 = 20 + r * step + int(rng.integers(-6, 7))
                x0 = 20 + c * step + int(rng.integers(-6, 7))
                tiles[(r, c)] = scene[y0 : y0 + tile, x0 : x0 + tile]
                truth[(r, c)] = (x0, y0)
        result = GridStitcher(overlap=overlap).stitch(tiles)
        assert all(e.used for e in result.edges)
        x00, y00 = result.positions[(0, 0)]
        errors = [
            np.hypot((result.positions[k][0] - x00) - (truth[k][0] - truth[(0, 0)][0]),
                     (result.positions[k][1] - y00) - (truth[k][1] - truth[(0, 0)][1]))
            for k in tiles
        ]
        assert max(errors) < 0.5


class TestStageFallback:
    def test_flagged_seam_uses_the_fitted_stage_matrix(self):
        # One seam lands on a blank patch and cannot be measured; the stage says where it goes.
        scene = _make_shapes_scene(900, seed=8).astype(np.float32)
        tile, overlap = 256, 0.2
        step = int(tile * (1 - overlap))
        pixel = 1e-9
        rng = np.random.default_rng(2)
        tiles, truth, stage = {}, {}, {}
        for r in range(3):
            for c in range(3):
                y0 = 20 + r * step + int(rng.integers(-6, 7))
                x0 = 20 + c * step + int(rng.integers(-6, 7))
                tiles[(r, c)] = scene[y0 : y0 + tile, x0 : x0 + tile].copy()
                truth[(r, c)] = (x0, y0)
                # A stage that reads back with 0.5 px of error, axes swapped like the twin.
                stage[(r, c)] = ((y0 + rng.normal(0, 0.5)) * pixel, (x0 + rng.normal(0, 0.5)) * pixel)
        # Blank the strip where tiles (2, 1) and (2, 2) overlap so that seam is weak.
        tiles[(2, 1)][:, step:] = tiles[(2, 1)].mean()
        tiles[(2, 2)][:, : tile - step] = tiles[(2, 2)].mean()
        without = GridStitcher(overlap=overlap).stitch(tiles)
        with_stage = GridStitcher(overlap=overlap).stitch(tiles, stage_positions=stage)

        def worst(result):
            x00, y00 = result.positions[(0, 0)]
            return max(
                np.hypot((result.positions[k][0] - x00) - (truth[k][0] - truth[(0, 0)][0]),
                         (result.positions[k][1] - y00) - (truth[k][1] - truth[(0, 0)][1]))
                for k in tiles
            )

        assert any(not e.used for e in with_stage.edges)
        assert np.asarray(with_stage.stage_to_canvas_px_per_m) == pytest.approx(np.array([[0, 1e9], [1e9, 0]]), abs=0.02e9)
        assert worst(with_stage) < 1.5
        assert worst(with_stage) <= worst(without) + 1e-6


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
            return start, parse_map_grid_result(twin.acquire_map_grid(json.dumps(spec)))
        finally:
            twin.move_stage([0.0] * 5)

    def _stitch_errors(self, spec, use_stage=False):
        """Stitched tile positions against the twin's true stage positions, in pixels."""
        pixel = self.FOV / self.IMSIZE
        tiles = {(t["row"], t["col"]): load_acquisition_image(t["key"]) for t in spec["tiles"]}
        actual = {(t["row"], t["col"]): t["stage_xy_m"] for t in spec["tiles"]}
        # The twin's stage repeats to 30 pm (under 1 px), so a seam 3 px off the stage is wrong.
        stitcher = GridStitcher(overlap=spec["overlap"], max_stage_disagreement_px=3.0 if use_stage else None)
        result = stitcher.stitch(tiles, stage_positions=actual if use_stage else None)
        x00, y00 = result.positions[(0, 0)]
        errors_px = []
        for key, (x_m, y_m) in actual.items():
            # The twin draws stage x down the image rows and stage y across the columns.
            true_row = (x_m - actual[(0, 0)][0]) / pixel
            true_col = (y_m - actual[(0, 0)][1]) / pixel
            col, row = result.positions[key][0] - x00, result.positions[key][1] - y00
            errors_px.append(float(np.hypot(row - true_row, col - true_col)))
        print(f"\ntwin 3x3 stitch error vs true stage: max {max(errors_px):.2f} px, "
              f"mean {np.mean(errors_px):.2f} px ({pixel * 1e12:.0f} pm/px); "
              f"flagged seams {[(e.a, e.b, e.reason) for e in result.flagged_edges]}")
        return result, errors_px

    def test_calibrated_grid_puts_tiles_in_image_order(self, mapping_twin_proxy, scan_proxy):
        _start, spec = self._scan(
            mapping_twin_proxy, scan_proxy, {"rows": 3, "cols": 3, "overlap": 0.25, "calibration_fov_m": self.CAL_FOV}
        )
        pixel = self.FOV / self.IMSIZE
        # Stage x moves the content up the rows, stage y left along the columns.
        expected = [[-1 / pixel, 0.0], [0.0, -1 / pixel]]
        assert np.asarray(spec["calibration"]["px_per_m"]) == pytest.approx(np.asarray(expected), rel=0.15, abs=0.15 / pixel)
        assert {(t["row"], t["col"]) for t in spec["tiles"]} == {(r, c) for r in range(3) for c in range(3)}
        result, _errors = self._stitch_errors(spec)
        assert len(result.edges) == 12
        # Most seams measure; the rest are flagged rather than silently trusted.
        assert sum(e.used for e in result.edges) >= 6
        assert all(e.reason in {"weak", "inconsistent"} for e in result.flagged_edges)

    @pytest.mark.xfail(
        strict=False,
        reason="Known limit: at 10 nm the twin's lattice still fools about a quarter of the seams. "
        "Flagged seams fall back to the planned step, which carries the calibration's few-percent "
        "scale error, so some runs place a tile several pixels off. Next step: refit the "
        "stage-to-image matrix from the trusted seams and use it for the flagged ones.",
    )
    def test_stitch_lands_within_a_pixel_of_the_true_stage(self, mapping_twin_proxy, scan_proxy):
        _start, spec = self._scan(
            mapping_twin_proxy, scan_proxy, {"rows": 3, "cols": 3, "overlap": 0.25, "calibration_fov_m": self.CAL_FOV}
        )
        _result, errors_px = self._stitch_errors(spec)
        assert max(errors_px) < 1.0

    def test_stage_fallback_lands_within_a_pixel(self, mapping_twin_proxy, scan_proxy):
        # The twin reads back its true stage position, so this is the best case: a
        # real stage read-back adds its own error to every flagged seam, and the
        # 3 px stage check only makes sense where the stage repeats that well.
        _start, spec = self._scan(
            mapping_twin_proxy, scan_proxy, {"rows": 3, "cols": 3, "overlap": 0.25, "calibration_fov_m": self.CAL_FOV}
        )
        result, errors_px = self._stitch_errors(spec, use_stage=True)
        assert result.stage_to_canvas_px_per_m is not None
        assert max(errors_px) < 1.0

    def test_grid_acquisition_and_map_build(self, mapping_twin_proxy, scan_proxy, tmp_path):
        start, spec = self._scan(mapping_twin_proxy, scan_proxy, {"rows": 2, "cols": 2, "overlap": 0.25, "calibration_fov_m": self.CAL_FOV})

        assert spec["grid"] == [2, 2]
        assert len(spec["tiles"]) == 4
        assert spec["pixel_size_nm"] == pytest.approx(self.FOV / self.IMSIZE * 1e9)
        assert spec["calibration"]["measured"] is True
        assert spec["calibration"]["calibration_fov_m"] == pytest.approx(self.CAL_FOV)
        for t in spec["tiles"]:
            assert Path(t["key"]).exists()
            assert len(t["stage_xy_m"]) == len(t["commanded_xy_m"]) == 2

        summary = build_sample_map(
            spec["tiles"],
            resolve_key=lambda key: Path(key),
            out_dir=tmp_path / "map",
            name="twin_map",
            overlap=spec["overlap"],
            pixel_size_nm=spec["pixel_size_nm"],
            stage_origin_m=spec["stage_origin_m"],
        )
        assert summary["tiles_stitched"] == 4
        bundle = Path(summary["bundle_path"])
        assert (bundle / "map.json").exists()
        assert (bundle / "index.html").exists()

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


class TestRegisterOverview:
    def test_recovers_known_scale_and_offset(self):
        import cv2

        scene = _make_shapes_scene()
        fine = scene[350:850, 350:850].copy()
        overview_region = scene[100:900, 100:900].copy()
        overview = cv2.resize(overview_region, (240, 240), interpolation=cv2.INTER_AREA)

        reg = register_overview(overview, fine)

        assert reg is not None
        assert reg.scale == pytest.approx(800 / 240, rel=0.05)
        assert reg.offset_px[0] == pytest.approx(100 - 350, abs=10)
        assert reg.offset_px[1] == pytest.approx(100 - 350, abs=10)
        assert abs(reg.rotation_deg) < 2.0
        assert reg.inliers >= 12

    def test_unrelated_images_do_not_match(self):
        rng = np.random.default_rng(1)
        fine = rng.integers(0, 255, (400, 400), dtype=np.uint8)
        overview = rng.integers(0, 255, (150, 150), dtype=np.uint8)
        assert register_overview(overview, fine) is None
