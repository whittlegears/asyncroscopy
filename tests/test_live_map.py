"""Tests for asyncroscopy.mapping.live_map and detectors: findings stay on their evidence."""

from __future__ import annotations

import numpy as np
import pytest

from asyncroscopy.mapping.detectors import BlobDetector
from asyncroscopy.mapping.live_map import LiveMap


def _scene(size=900, seed=3, n_blobs=40):
    """Smooth texture with bright gaussian blobs at known positions (row, col)."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    scene = ndimage.gaussian_filter(rng.random((size, size)), 6) * 0.3
    yy, xx = np.mgrid[0:size, 0:size]
    blobs = rng.uniform(30, size - 30, size=(n_blobs, 2))
    for r, c in blobs:
        scene += np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 4.0**2))
    return scene.astype(np.float32), blobs


def _cut(scene, tile=256, overlap=0.2, rows=3, cols=3, seed=1):
    rng = np.random.default_rng(seed)
    step = int(tile * (1 - overlap))
    out = {}
    for r in range(rows):
        for c in range(cols):
            y0 = 20 + r * step + int(rng.integers(-5, 6))
            x0 = 20 + c * step + int(rng.integers(-5, 6))
            out[(r, c)] = (scene[y0 : y0 + tile, x0 : x0 + tile].copy(), (x0, y0))
    return out


class TestBlobDetector:
    def test_finds_the_planted_blobs(self):
        scene, blobs = _scene(300, n_blobs=6)
        hits = BlobDetector(sigma_px=4, threshold=5)(scene)
        found = np.array([h["px"] for h in hits])
        for r, c in blobs:
            assert np.min(np.hypot(found[:, 0] - r, found[:, 1] - c)) < 2.0

    def test_flat_image_has_no_hits(self):
        assert BlobDetector()(np.zeros((64, 64))) == []


class TestLiveMap:
    def test_findings_arrive_before_the_map_is_complete(self):
        scene, _ = _scene()
        tiles = _cut(scene)
        live = LiveMap(overlap=0.2, pixel_size_nm=0.1)
        live.add_detector("blob", BlobDetector(threshold=6), version="test")
        first = live.add_tile(0, 0, tiles[(0, 0)][0])
        assert first["new_annotations"], "the first tile alone should already give findings"
        assert all(a["status"] == "provisional" and a["author"] == "model:blob@test" for a in live.list_annotations())

    def test_pins_follow_their_tile_and_land_on_the_true_feature(self):
        scene, blobs = _scene()
        tiles = _cut(scene)
        live = LiveMap(overlap=0.2)
        live.add_detector("blob", BlobDetector(threshold=6))
        for key, (image, _origin) in tiles.items():
            live.add_tile(*key, image)
        x00, y00 = tiles[(0, 0)][1]
        cx00, cy00 = live.positions[(0, 0)]
        live_pins = [a for a in live.list_annotations() if a["status"] != "duplicate"]
        assert live_pins
        placed = [a for a in live_pins if a["placed"]]
        assert len(placed) >= 0.7 * len(live_pins)
        for a in placed:
            x, y = a["canvas_xy"]
            # Back to scene coordinates through tile (0, 0)'s true origin.
            r, c = y - cy00 + y00, x - cx00 + x00
            assert np.min(np.hypot(blobs[:, 0] - r, blobs[:, 1] - c)) < 2.0

    def test_overlap_duplicates_are_merged(self):
        scene, blobs = _scene()
        tiles = _cut(scene)
        live = LiveMap(overlap=0.2)
        live.add_detector("blob", BlobDetector(threshold=6))
        for key, (image, _origin) in tiles.items():
            live.add_tile(*key, image)
        all_pins = live.list_annotations()
        unique = [a for a in all_pins if a["status"] != "duplicate"]
        assert len(unique) < len(all_pins), "blobs in overlaps should be found twice and merged"
        # Each unique pin is a different feature.
        xy = np.array([a["canvas_xy"] for a in unique])
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        assert d.min() > live.merge_radius_px

    def test_review_keeps_history_and_rejected_pins_do_not_block_others(self):
        scene, _ = _scene()
        tiles = _cut(scene)
        live = LiveMap(overlap=0.2)
        live.add_tile(0, 0, tiles[(0, 0)][0])
        pin = live.annotate((0, 0), (100.0, 120.0), "vacancy?", author="agent:explorer", reason="dark spot")
        assert live.describe(pin)["status"] == "provisional"
        live.review(pin, "human:operator", "rejected", note="contamination")
        record = live.describe(pin)
        assert record["status"] == "rejected"
        assert [h["action"] for h in record["history"]] == ["proposed", "reviewed"]
        assert record["history"][1]["from"] == "provisional"

    def test_human_findings_start_accepted(self):
        live = LiveMap()
        live.add_tile(0, 0, np.random.default_rng(0).random((64, 64)))
        pin = live.annotate((0, 0), [(10, 10), (10, 30), (30, 30)], "grain boundary", author="human:operator")
        assert live.describe(pin)["status"] == "accepted"
        assert live.describe(pin)["kind"] == "polygon"

    def test_crop_returns_source_pixels(self):
        scene, _ = _scene()
        tiles = _cut(scene)
        live = LiveMap(overlap=0.2)
        live.add_tile(0, 1, tiles[(0, 1)][0], source="/data/tile01.h5")
        pin = live.annotate((0, 1), (50.0, 60.0), "x", author="human:a")
        crop = live.crop(pin, half_size=10)
        assert crop["source"] == "/data/tile01.h5"
        np.testing.assert_array_equal(crop["pixels"], tiles[(0, 1)][0][40:60, 50:70])

    def test_unplaced_tile_findings_are_marked(self):
        live = LiveMap(overlap=0.2)
        rng = np.random.default_rng(1)
        live.add_tile(0, 0, rng.random((128, 128)))
        live.add_tile(0, 1, np.zeros((128, 128)))  # nothing to register against
        pin = live.annotate((0, 1), (10.0, 10.0), "x", author="agent:a")
        assert live.describe(pin)["placed"] is False

    def test_bad_points_are_refused(self):
        live = LiveMap()
        live.add_tile(0, 0, np.ones((32, 32)))
        with pytest.raises(ValueError):
            live.annotate((0, 0), (40.0, 1.0), "x", author="human:a")
        with pytest.raises(ValueError):
            live.annotate((5, 5), (1.0, 1.0), "x", author="human:a")

    def test_save_and_load_round_trip(self, tmp_path):
        scene, _ = _scene()
        tiles = _cut(scene, rows=2, cols=2)
        live = LiveMap(name="demo", overlap=0.2, pixel_size_nm=0.05)
        live.add_detector("blob", BlobDetector(threshold=6), version="1")
        for key, (image, _o) in tiles.items():
            live.add_tile(*key, image, stage_xy_m=(key[1] * 2e-8, key[0] * 2e-8))
        pin = live.annotate((1, 1), (20.0, 20.0), "note", author="human:a")
        live.review(pin, "human:b", "rejected")
        live.save(tmp_path / "m")
        back = LiveMap.load(tmp_path / "m")
        assert back.status()["tiles_total"] == 4
        assert back.describe(pin)["history"] == live.describe(pin)["history"]
        assert back.positions.keys() == live.positions.keys()
        for k in live.positions:
            assert back.positions[k] == pytest.approx(live.positions[k], abs=1e-6)

    def test_events_are_emitted_in_order(self):
        live = LiveMap()
        seen = []
        live.listeners.append(lambda event, record: seen.append(event))
        live.add_detector("blob", BlobDetector(threshold=6))
        scene, _ = _scene(300, n_blobs=3)
        live.add_tile(0, 0, scene[:128, :128])
        assert seen[0] == "tile_ready"
        assert seen[-1] == "stitch_updated"
