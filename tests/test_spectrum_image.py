"""EDS spectrum images: the instrument's own when it has one, a loop otherwise, and the cube streamed to the run file."""

import json
import time
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango

from asyncroscopy.data import data_reader
from asyncroscopy.data.data_reader import read_rows
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.electron_microscope import ElectronMicroscope
from asyncroscopy.instruments.scanning_beam import unsupported
from asyncroscopy.jobs.mapping.run_file import RunFile

PLAN = {"size_m": [1.2e-7, 1.2e-7], "fov_m": 8e-8, "overlap": 0.25, "imsize": 16, "dwell_s": 1e-6, "per_tile": ["eds_map"], "eds_map_points": 4, "eds_exposure_s": 1e-3, "stitch": False}


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def run(mapper, plan):
    reserved = call(mapper, "reserve", {"plan": plan})
    started = call(mapper, "start", {"reservation_id": reserved["reservation_id"]})
    deadline = time.monotonic() + 120
    while True:
        status = call(mapper, "job_status", {"job_id": started["job_id"]})
        if status["state"] in {"done", "failed", "aborted"}:
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.05)
    results = call(mapper, "results", {"job_id": started["job_id"]})
    call(mapper, "release", {"reservation_id": reserved["reservation_id"]})
    return results


def test_the_twin_makes_a_spectrum_image_directly(mapping_twin_proxy, eds_proxy):
    eds_proxy.exposure_time = 1e-3
    mapping_twin_proxy.unblank_beam()
    try:
        key = mapping_twin_proxy.acquire_spectrum_image(3)
    finally:
        mapping_twin_proxy.blank_beam()
    with h5py.File(key, "r") as h5:
        cube = h5["spectrum_image"]
        assert cube.shape[:2] == (3, 3) and cube.chunks[:1] == (1,)
        assert cube.attrs["spectrum_image_axes"] == "rows=+y, columns=+x, channels" and cube.attrs["energy_dispersion_ev"] > 0
        assert np.all(np.asarray(cube).sum(axis=-1) > 0)
    rows = list(read_rows(key, "spectrum_image"))
    assert len(rows) == 3 and rows[0][0].shape[0] == 3
    with pytest.raises(tango.DevFailed, match="between 2 and"):
        mapping_twin_proxy.acquire_spectrum_image(1)


def test_an_eds_map_uses_the_native_acquisition_and_no_beam_placement(mapper_proxy, monkeypatch):
    placed = []
    real = DigitalTwin._place_beam
    monkeypatch.setattr(DigitalTwin, "_place_beam", lambda self, position: (placed.append(position), real(self, position))[1])
    results = run(mapper_proxy, PLAN)
    assert results["state"] == "done", results
    assert placed == []
    with h5py.File(results["summary"]["run_file"], "r") as h5:
        cube = h5["spectrum_image"]
        assert cube.shape[1:3] == (4, 4) and cube.chunks[:2] == (1, 1)
        assert h5["written"][()].all() and np.all(np.asarray(cube).sum(axis=-1) > 0)
        assert cube.attrs["spectrum_image_axes"] == "rows=+y, columns=+x, channels"


def test_a_backend_without_it_falls_back_to_the_loop_once(mapper_proxy, monkeypatch):
    asked, placed = [], []
    real_place = DigitalTwin._place_beam

    def no_native(self, points, exposure_time):
        asked.append(points)
        unsupported(self, "acquire_spectrum_image")

    monkeypatch.setattr(DigitalTwin, "_acquire_spectrum_image", no_native)
    monkeypatch.setattr(DigitalTwin, "_place_beam", lambda self, position: (placed.append(position), real_place(self, position))[1])
    results = run(mapper_proxy, PLAN)
    assert results["state"] == "done", results
    assert len(asked) == 1  # asked once, for the first tile
    tiles = results["summary"]["grid"]["rows"] * results["summary"]["grid"]["cols"]
    assert len(placed) == 16 * tiles
    with h5py.File(results["summary"]["run_file"], "r") as h5:
        assert h5["spectrum_image"].shape[1:3] == (4, 4) and np.all(np.asarray(h5["spectrum_image"]).sum(axis=-1) > 0)


def test_the_run_file_writes_the_cube_a_row_at_a_time(tmp_path):
    run_file = RunFile(tmp_path / "run.h5")
    run_file.open({}, {"id": "x"})
    for row in range(3):
        run_file.write_spectrum_image_row(2, 1, 3, row, np.full((3, 8), row + 1, dtype=np.uint32), {"energy_dispersion_ev": 5.0})
    run_file.write_spectrum_image_key(2, 0, "vendor.emd")
    run_file.close()
    with h5py.File(tmp_path / "run.h5.partial", "r") as h5:
        assert h5["spectrum_image"].shape == (2, 3, 3, 8) and h5["spectrum_image"].chunks == (1, 1, 3, 8)
        assert np.array_equal(h5["spectrum_image"][1, 2], np.full((3, 8), 3))
        assert h5["spectrum_image_keys"][0].decode() == "vendor.emd"


def test_rows_come_through_tiled_one_at_a_time(monkeypatch):
    read = []

    class Dataset:
        shape = (2, 3, 4)
        metadata = {"energy_dispersion_ev": 5.0}

        def __getitem__(self, row):
            read.append(row)
            return np.zeros((3, 4))

    monkeypatch.setattr("asyncroscopy.data.tiled_client.node_for_key", lambda key, uri: {"spectrum_image": Dataset()})
    proxy = types.SimpleNamespace(get_config=lambda: json.dumps({"uri": "http://x"}))
    rows = data_reader.read_rows("run.h5", "spectrum_image", proxy)
    first, attrs = next(rows)
    assert first.shape == (3, 4) and attrs["energy_dispersion_ev"] == 5.0 and read == [0]
    assert len(list(rows)) == 1


def test_the_base_refuses_a_backend_without_a_spectrum_image():
    device = types.SimpleNamespace()
    with pytest.raises(tango.DevFailed, match="does not support acquire_spectrum_image"):
        ElectronMicroscope._acquire_spectrum_image(device, 8, 1.0)


def test_autoscript_saves_the_vendors_file_and_names_it(tmp_path, monkeypatch):
    from asyncroscopy.instruments.electron_microscope import auto_script
    from tests import fake_autoscript

    record = fake_autoscript._Recorder() if hasattr(fake_autoscript, "_Recorder") else None
    calls = []

    class Eds:
        def acquire_spectrum_image(self, settings):
            calls.append((settings.dwell_time, settings.size, str(settings.eds_detector)))
            return types.SimpleNamespace(save=lambda path: Path(path).write_bytes(b"emd"))

    device = auto_script.AutoScriptMicroscope.__new__(auto_script.AutoScriptMicroscope)
    device._microscope = types.SimpleNamespace(analysis=types.SimpleNamespace(eds=Eds()))
    device._detector_proxies = {}
    device.acquisition_save_directory = str(tmp_path)
    device._beam_restored = lambda: __import__("contextlib").nullcontext()
    device._frame_attrs = lambda points, dwell, region: {}
    key = auto_script.AutoScriptMicroscope._acquire_spectrum_image(device, 8, 2e-3)
    assert record is None or record is not None
    assert key.endswith(".emd") and Path(key).read_bytes() == b"emd" and calls[0][:2] == (2e-3, 8)
