"""The DigitalTwin's camera image, its in-memory image cache, and the beam commands and attributes that share
place_beam's one check: acquire_camera_image, get_image_data_cached, place_beam_list, beam_pos, and the read-only
fov, defocus, beam_state and manufacturer attributes."""

import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango

from asyncroscopy.instruments.electron_microscope import digital_twin_base, digital_twin_particles, electron_microscope
from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.digital_twin_particles import DigitalTwinParticles
from asyncroscopy.instruments.electron_microscope.jeol import JeolMicroscope
from asyncroscopy.instruments.scanning_beam import unsupported_commands
from asyncroscopy.routines.dose import ELECTRON_CHARGE


@pytest.fixture
def camera(twin_proxy, camera_proxy):
    """The shared CAMERA device and twin, put back as they were."""
    saved = {name: getattr(camera_proxy, name) for name in ("imsize", "exposure_time", "readout_area", "output_format")}
    fov = twin_proxy.get_fov()
    twin_proxy.set_fov(1e-7)
    camera_proxy.imsize, camera_proxy.exposure_time, camera_proxy.readout_area = 64, 0.01, "Full"
    yield camera_proxy
    for name, value in saved.items():
        setattr(camera_proxy, name, value)
    twin_proxy.set_fov(fov)
    twin_proxy.unblank_beam()


@pytest.fixture
def scan(scan_proxy):
    """The shared SCAN device, put back as it was."""
    saved = (scan_proxy.imsize, scan_proxy.dwell_time, list(scan_proxy.scan_region))
    yield scan_proxy
    scan_proxy.imsize, scan_proxy.dwell_time, scan_proxy.scan_region = saved


def _image(key: str) -> tuple[np.ndarray, dict]:
    with h5py.File(Path(key), "r") as h5:
        return h5["image"][()], dict(h5["image"].attrs)


def test_the_twin_camera_image_has_the_camera_shape_and_pixel_size(twin_proxy, camera):
    image, attrs = _image(twin_proxy.acquire_camera_image())
    assert image.shape == (64, 64)
    assert attrs["acquisition_type"] == "camera_image" and attrs["detector"] == camera.camera_detector
    assert attrs["fov_m"] == pytest.approx(1e-7) and attrs["pixel_size_m"] == pytest.approx(1e-7 / 64)
    camera.readout_area = "Half"
    image, attrs = _image(twin_proxy.acquire_camera_image())
    assert image.shape == (64, 64)
    assert attrs["fov_m"] == pytest.approx(0.5e-7) and attrs["pixel_size_m"] == pytest.approx(0.5e-7 / 64)
    assert attrs["illuminated_fov_m"] == pytest.approx(1e-7)


def test_the_twin_camera_counts_scale_with_exposure_and_beam_current(twin_proxy, camera):
    short, attrs = _image(twin_proxy.acquire_camera_image())
    camera.exposure_time = 0.04
    long, _ = _image(twin_proxy.acquire_camera_image())
    assert long.mean() / short.mean() == pytest.approx(4.0, rel=0.05)
    electrons = attrs["beam_current_pa"] * 1e-12 * 0.01 / ELECTRON_CHARGE / 64**2
    assert 0.5 * electrons < short.mean() <= electrons * 1.01  # the objective aperture passes a fraction of the beam
    assert short.std() > 0  # shot noise


def test_a_blanked_beam_puts_nothing_on_the_camera(twin_proxy, camera):
    twin_proxy.blank_beam()
    image, attrs = _image(twin_proxy.acquire_camera_image())
    assert not image.any() and attrs["beam_current_pa"] == 0.0


def test_the_camera_output_format_reaches_the_saved_file(twin_proxy, camera):
    camera.output_format = ".tiff"
    assert twin_proxy.acquire_camera_image().endswith(".tiff")


def test_a_camera_pixel_finer_than_the_mass_thickness_model_is_refused(twin_proxy, camera):
    twin_proxy.set_fov(5e-9)
    with pytest.raises(tango.DevFailed, match="phase contrast"):
        twin_proxy.acquire_camera_image()


def test_only_twins_with_a_camera_model_offer_the_camera_image():
    assert "acquire_camera_image" not in unsupported_commands(DigitalTwin)
    assert "acquire_camera_image" in unsupported_commands(DigitalTwinParticles)


def test_the_image_cache_returns_the_newest_first_and_matches_the_saved_data(twin_proxy, scan):
    scan.imsize, scan.dwell_time, scan.scan_region = 16, 1e-6, [0.0, 0.0, 1.0, 1.0]
    first = twin_proxy.acquire_scanned_image(["haadf"])
    second = twin_proxy.acquire_scanned_image(["haadf", "bf"])
    newest = [json.loads(twin_proxy.get_image_data_cached(i)[0]) for i in range(3)]
    assert [(m["key"], m["detector"]) for m in newest] == [(second, "BF"), (second, "HAADF"), (first, "HAADF")]
    meta, raw = twin_proxy.get_image_data_cached(0)
    meta = json.loads(meta)
    image = np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["shape"])
    with h5py.File(second, "r") as h5:
        assert np.array_equal(image, h5["image/BF"][()])


def test_the_image_cache_is_bounded(twin_proxy, scan):
    scan.imsize, scan.dwell_time, scan.scan_region = 8, 1e-6, [0.0, 0.0, 1.0, 1.0]
    keys = [twin_proxy.acquire_scanned_image(["haadf"]) for _ in range(10)]
    meta = json.loads(twin_proxy.get_image_data_cached(7)[0])
    assert meta["cached"] == meta["cache_size"] == 8 and meta["key"] == keys[2]
    with pytest.raises(tango.DevFailed, match="out of range"):
        twin_proxy.get_image_data_cached(8)
    with pytest.raises(tango.DevFailed, match="out of range"):
        twin_proxy.get_image_data_cached(-1)


def _place_beam_error(twin_proxy, position) -> str:
    with pytest.raises(tango.DevFailed) as refused:
        twin_proxy.place_beam(position)
    return refused.value.args[0].desc


def test_place_beam_list_visits_each_position_and_leaves_the_beam_at_the_last(twin_proxy):
    twin_proxy.place_beam_list([0.2, 0.3, 0.6, 0.4])
    assert list(twin_proxy.beam_pos) == pytest.approx([0.6, 0.4])


def test_place_beam_list_refuses_an_out_of_range_position_with_place_beam_error(twin_proxy):
    twin_proxy.place_beam([0.5, 0.5])
    expected = _place_beam_error(twin_proxy, [1.5, 0.5])
    with pytest.raises(tango.DevFailed) as refused:
        twin_proxy.place_beam_list([0.25, 0.25, 1.5, 0.5])
    assert refused.value.args[0].desc == expected
    assert list(twin_proxy.beam_pos) == pytest.approx([0.5, 0.5])  # the good first position was not visited either
    with pytest.raises(tango.DevFailed, match="pairs"):
        twin_proxy.place_beam_list([0.25, 0.25, 0.5])


def test_place_beam_list_is_unsupported_where_place_beam_is():
    assert {"place_beam", "place_beam_list"} <= set(unsupported_commands(JeolMicroscope))
    assert "place_beam_list" not in unsupported_commands(DigitalTwin)


def test_a_beam_pos_write_goes_through_place_beam_check(twin_proxy):
    twin_proxy.beam_pos = [0.3, 0.7]
    assert list(twin_proxy.beam_pos) == pytest.approx([0.3, 0.7])
    expected = _place_beam_error(twin_proxy, [1.5, 0.5])
    with pytest.raises(tango.DevFailed) as refused:
        twin_proxy.beam_pos = [1.5, 0.5]
    assert expected in refused.value.args[0].desc
    assert list(twin_proxy.beam_pos) == pytest.approx([0.3, 0.7])


def test_beam_pos_and_manufacturer_are_declared_once():
    sources = [inspect.getsource(module) for module in (digital_twin_base, digital_twin_particles, electron_microscope)]
    assert sum(source.count("beam_pos = attribute(") for source in sources) == 1
    assert sum(source.count("manufacturer = attribute(") for source in sources) == 1


def test_the_read_only_attributes_read_what_the_commands_read(twin_proxy):
    twin_proxy.set_fov(1.5e-7)
    twin_proxy.set_defocus(-2e-8)
    assert twin_proxy.fov == pytest.approx(twin_proxy.get_fov()) == pytest.approx(1.5e-7)
    assert twin_proxy.defocus == pytest.approx(twin_proxy.get_defocus()) == pytest.approx(-2e-8)
    twin_proxy.blank_beam()
    assert twin_proxy.beam_state is True and json.loads(twin_proxy.get_parameters())["beam_blanked"] is True
    twin_proxy.unblank_beam()
    assert twin_proxy.beam_state is False
    assert twin_proxy.manufacturer == json.loads(twin_proxy.get_parameters())["manufacturer"] == "UTKTeam"
    twin_proxy.set_defocus(0.0)


@pytest.mark.parametrize("name, value", [("fov", 1e-7), ("defocus", 1e-8), ("beam_state", True), ("manufacturer", "x")])
def test_the_read_only_attributes_refuse_writes(twin_proxy, name, value):
    with pytest.raises(tango.DevFailed):
        setattr(twin_proxy, name, value)
