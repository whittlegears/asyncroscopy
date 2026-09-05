"""Tests for base ElectronMicroscope behavior shared across vendor backends."""

import json

import numpy as np
import pytest
import tango

from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope import electron_microscope


class FakeArray:
    metadata = {"detector": "HAADF"}
    shape = (2, 2)
    dtype = np.dtype("int64")

    def read(self, slices=None):
        array = np.arange(4).reshape(2, 2)
        return array[slices] if slices else array


class FakeContainer(dict):
    metadata = {}


class FakeDataProxy:
    def __init__(self, uri: str = "http://microscope:9091"):
        self._uri = uri

    def get_config(self) -> str:
        return json.dumps({"uri": self._uri})


def make_twin() -> DigitalTwin:
    twin = DigitalTwin.__new__(DigitalTwin)
    twin._detector_proxies = {}
    return twin


def test_get_image_data_cached_reads_metadata_from_tiled(monkeypatch):
    twin = make_twin()
    twin._detector_proxies["data"] = FakeDataProxy()
    twin._remember_acquired_key("frame.h5")

    tiled_node = FakeContainer(image=FakeContainer(HAADF=FakeArray()))
    monkeypatch.setattr(electron_microscope, "open_client", lambda uri, api_key=None: {"frame.h5": tiled_node})

    metadata_json, preview_bytes = twin.get_image_data_cached(0)
    metadata = json.loads(metadata_json)

    assert metadata["key"] == "frame.h5"
    assert metadata["uri"] == "http://microscope:9091"
    assert metadata["datasets"][0]["shape"] == [2, 2]
    assert metadata["datasets"][0]["dtype"] == "int64"
    assert preview_bytes == np.array(metadata["datasets"][0]["preview"]).tobytes()


def test_get_image_data_cached_without_prior_acquisition_fails_honestly():
    twin = make_twin()

    with pytest.raises(tango.DevFailed):
        twin.get_image_data_cached(0)


def test_get_image_data_cached_without_data_device_fails_honestly():
    twin = make_twin()
    twin._remember_acquired_key("frame.h5")

    with pytest.raises(tango.DevFailed):
        twin.get_image_data_cached(0)


def test_get_image_data_cached_index_zero_is_most_recent(monkeypatch):
    """The docstring promises index 0 = most recent acquisition; the cache
    appends oldest-first, so lookups must index from the end."""
    twin = make_twin()
    twin._detector_proxies["data"] = FakeDataProxy()
    for key in ["oldest.h5", "middle.h5", "newest.h5"]:
        twin._remember_acquired_key(key)

    tiled_node = FakeContainer(image=FakeContainer(HAADF=FakeArray()))
    monkeypatch.setattr(
        electron_microscope,
        "open_client",
        lambda uri, api_key=None: {name: tiled_node for name in ["oldest.h5", "middle.h5", "newest.h5"]},
    )

    for index, expected in [(0, "newest.h5"), (1, "middle.h5"), (2, "oldest.h5")]:
        metadata = json.loads(twin.get_image_data_cached(index)[0])
        assert metadata["key"] == expected

    with pytest.raises(tango.DevFailed):
        twin.get_image_data_cached(3)
    with pytest.raises(tango.DevFailed):
        twin.get_image_data_cached(-1)


def test_remember_acquired_key_caps_history():
    twin = make_twin()
    for i in range(electron_microscope.MAX_CACHED_IMAGE_KEYS + 10):
        twin._remember_acquired_key(f"key-{i}")

    assert len(twin._cached_images) == electron_microscope.MAX_CACHED_IMAGE_KEYS
    assert twin._cached_images[0] == "key-10"
    assert twin._cached_images[-1] == f"key-{electron_microscope.MAX_CACHED_IMAGE_KEYS + 9}"
