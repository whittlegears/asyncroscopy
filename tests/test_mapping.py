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
from asyncroscopy.mcp.mapping_mcp_server import MappingServer


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


class TestAcquireMapGridDigitalTwin:
    def test_grid_acquisition_and_map_build(self, twin_proxy, scan_proxy, tmp_path):
        scan_proxy.imsize = 64
        scan_proxy.dwell_time = 1e-6
        scan_proxy.output_format = ".h5"
        twin_proxy.set_fov(5e-6)

        start = list(twin_proxy.get_stage())
        payload = twin_proxy.acquire_map_grid(
            json.dumps({"rows": 2, "cols": 2, "overlap": 0.2, "settle_s": 0.0})
        )
        spec = parse_map_grid_result(payload)

        assert spec["grid"] == [2, 2]
        assert len(spec["tiles"]) == 4
        assert {(t["row"], t["col"]) for t in spec["tiles"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert spec["pixel_size_nm"] is not None and spec["pixel_size_nm"] > 0
        end = list(twin_proxy.get_stage())
        assert end == pytest.approx(start, abs=1e-9)
        for t in spec["tiles"]:
            assert Path(t["key"]).exists()

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


class TestMappingServer:
    def test_incremental_batches_and_finalize(self, tmp_path):
        scene = _make_shapes_scene(size=900)
        tiles = _grid_tiles(scene, rows=2, cols=2, overlap=0.2)
        store = tmp_path / "acq"
        store.mkdir()
        paths = {}
        for rc, img in tiles.items():
            p = store / f"tile_{rc[0]}_{rc[1]}.png"
            _write_png(p, img)
            paths[rc] = p

        server = MappingServer(name="test", output_root=tmp_path / "maps")
        started = server.start_map(name="demo", overlap=0.2, pixel_size_nm=0.1)
        map_id = started["map_id"]

        batch1 = server.add_tiles(map_id, [
            {"row": 0, "col": 0, "image_path": str(paths[(0, 0)])},
            {"row": 0, "col": 1, "image_path": str(paths[(0, 1)])},
        ])
        assert batch1["tiles_total"] == 2

        batch2 = server.add_tiles(map_id, [
            {"row": 1, "col": 0, "image_path": str(paths[(1, 0)])},
            {"row": 1, "col": 1, "image_path": str(paths[(1, 1)])},
        ])
        assert batch2["tiles_total"] == 4
        assert batch2["edges_measured"] == batch2["edges_total"] == 4

        status = server.get_map_status(map_id)
        assert status["grid_extent"] == [2, 2]

        preview = server.preview_map(map_id)
        assert preview["encoding"] == "base64"
        assert len(preview["payload"]) > 0

        summary = server.finalize_map(map_id)
        bundle = Path(summary["bundle_path"])
        assert (bundle / "map.json").exists()
        assert summary["tiles_stitched"] == 4

        with pytest.raises(ValueError):
            server.get_map_status(map_id)

    def test_register_overview_updates_manifest(self, tmp_path):
        scene = _make_shapes_scene(size=900)
        tiles = _grid_tiles(scene, rows=2, cols=2, overlap=0.2, jitter=0)
        store = tmp_path / "acq"
        store.mkdir()

        server = MappingServer(name="test", output_root=tmp_path / "maps")
        map_id = server.start_map(name="withoverview")["map_id"]
        for rc, img in tiles.items():
            p = store / f"tile_{rc[0]}_{rc[1]}.png"
            _write_png(p, img)
            server.add_tiles(map_id, [{"row": rc[0], "col": rc[1], "image_path": str(p)}])

        import cv2

        overview_region = scene[0:900, 0:900]
        overview = cv2.resize(overview_region, (300, 300), interpolation=cv2.INTER_AREA)
        overview_path = store / "overview.png"
        _write_png(overview_path, overview)

        result = server.register_overview(map_id, image_path=str(overview_path))
        assert result["matched"] is True
        assert result["scale"] > 1.0

        summary = server.finalize_map(map_id)
        assert summary["overview_count"] == 1
        manifest = json.loads((Path(summary["bundle_path"]) / "map.json").read_text())
        assert len(manifest["overviews"]) == 1
        assert manifest["overviews"][0]["image_path"] == "overviews/overview_000.png"
        assert (Path(summary["bundle_path"]) / "overviews" / "overview_000.png").exists()

    def test_unknown_map_id_raises(self):
        server = MappingServer(name="test", output_root="/tmp/unused")
        with pytest.raises(ValueError):
            server.add_tiles("nope", [{"row": 0, "col": 0, "image_path": "x.png"}])

    def test_image_b64_round_trip(self, tmp_path):
        import base64

        import cv2

        img = _make_shapes_scene(size=300)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        server = MappingServer(name="test", output_root=tmp_path / "maps")
        map_id = server.start_map()["map_id"]
        result = server.add_tiles(map_id, [{"row": 0, "col": 0, "image_b64": b64}])
        assert result["tiles_total"] == 1
