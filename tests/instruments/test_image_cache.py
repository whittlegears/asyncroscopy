"""ImageCache: what get_image_data_cached serves. The latest saved 2-D images, newest first, at most image_cache_size,
filled by the DATA writer from what it saves."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from asyncroscopy.data.data_writer import save_acquisition
from asyncroscopy.instruments.image_cache import ImageCache


def test_entries_come_back_newest_first():
    cache = ImageCache(3)
    cache.add("stem_image", [("k1", "HAADF", np.zeros((2, 2)))])
    cache.add("stem_image", [("k2", "HAADF", np.ones((2, 2)))])
    assert [cache.get(i)["key"] for i in range(2)] == ["k2", "k1"]
    assert np.array_equal(cache.get(0)["image"], np.ones((2, 2)))


def test_the_cache_is_bounded_and_drops_the_oldest():
    cache = ImageCache(2)
    for key in ("k1", "k2", "k3"):
        cache.add("stem_image", [(key, "HAADF", np.zeros((2, 2)))])
    assert len(cache) == cache.size == 2
    assert [cache.get(i)["key"] for i in range(2)] == ["k3", "k2"]
    with pytest.raises(IndexError, match="out of range"):
        cache.get(2)


def test_only_images_are_kept_and_are_copies():
    cache = ImageCache(4)
    image = np.zeros((3, 3))
    cache.add("spectrum", [("s", "eds", np.zeros(16))])
    cache.add("stem_data", [("d", "BM-Ceta", np.zeros((2, 2, 4, 4)))])
    cache.add("camera_image", [("c", "BM-Ceta", SimpleNamespace(data=image))])
    image[0, 0] = 5.0
    assert len(cache) == 1 and cache.get(0)["key"] == "c" and cache.get(0)["image"][0, 0] == 0.0


def test_an_empty_cache_and_a_bad_size_are_refused():
    with pytest.raises(LookupError, match="no image"):
        ImageCache(1).get(0)
    with pytest.raises(ValueError):
        ImageCache(0)


@pytest.mark.parametrize("output_format", [".h5", ".tiff"])
def test_the_data_writer_fills_the_saving_device_cache(tmp_path, output_format):
    device = SimpleNamespace(instrument_id="test/microscope/cache", acquisition_save_directory=str(tmp_path), image_cache=ImageCache(4))
    data_server = SimpleNamespace(save_path=str(tmp_path), register_path=lambda path: f"key:{path}")
    images = [np.full((4, 4), 1, dtype=np.uint16), np.full((4, 4), 2, dtype=np.uint16)]
    saved = save_acquisition(device, data_server, "stem_image", ["HAADF", "BF"], images, output_format=output_format)
    keys = [saved, saved] if output_format == ".h5" else json.loads(saved)
    assert [(device.image_cache.get(i)["key"], device.image_cache.get(i)["detector"]) for i in range(2)] == [(keys[1], "BF"), (keys[0], "HAADF")]
    assert np.array_equal(device.image_cache.get(0)["image"], images[1])
