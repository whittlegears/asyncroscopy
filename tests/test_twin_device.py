"""DigitalTwin through Tango, with the realistic effects switched on."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango


def acquire(twin, scan, imsize=64, fov=1e-7, region=(0.0, 0.0, 1.0, 1.0), detectors=("haadf",)):
    scan.imsize = imsize
    scan.dwell_time = 1e-6
    scan.scan_region = list(region)
    twin.set_fov(fov)
    path = Path(twin.acquire_scanned_image(list(detectors)))
    with h5py.File(path, "r") as h5:
        group = h5["image"] if "image" in h5 else h5
        return {name: (group[name][()], dict(group[name].attrs)) for name in group}


@pytest.fixture(autouse=True)
def beam_on(mapping_twin_proxy):
    """Other tests (Mapper abort) leave the beam blanked, as a real abort would."""
    mapping_twin_proxy.unblank_beam()


class TestRealisticTwin:
    def test_reported_position_is_not_the_true_position(self, mapping_twin_proxy):
        mapping_twin_proxy.move_stage([5e-8, -3e-8, 0.0, 0.0, 0.0])
        reported = np.array(mapping_twin_proxy.get_stage())
        truth = json.loads(mapping_twin_proxy.export_ground_truth())
        true = np.array(truth["moves"][-1]["true"])
        assert np.allclose(reported[:2], [5e-8, -3e-8], atol=6e-9)
        assert not np.allclose(reported[:2], true[:2], atol=1e-10)

    def test_parameters_never_reveal_the_true_position(self, mapping_twin_proxy):
        mapping_twin_proxy.move_stage([5e-8, -3e-8, 0.0, 0.0, 0.0])
        params = json.loads(mapping_twin_proxy.get_parameters())
        reported = np.array(mapping_twin_proxy.get_stage())
        view = params["viewport_world_angstrom"]
        centre = np.array([view["x_min"] + view["x_max"], view["y_min"] + view["y_max"]]) / 2 * 1e-10
        assert np.allclose(centre, reported[:2], atol=1e-12)
        assert "particle_count" not in params and "sample_seed" not in params

    def test_stage_device_shows_reported_position(self, mapping_twin_proxy, mapping_stage_proxy):
        mapping_twin_proxy.move_stage([1e-8, 1e-8, 0.0, 0.0, 0.0])
        assert mapping_stage_proxy.x == pytest.approx(mapping_twin_proxy.get_stage()[0])

    def test_writing_the_stage_device_counts_as_a_move(self, mapping_twin_proxy, mapping_stage_proxy):
        moves_before = len(json.loads(mapping_twin_proxy.export_ground_truth())["moves"])
        mapping_stage_proxy.x = 2e-8
        mapping_twin_proxy.get_stage()
        moves_after = len(json.loads(mapping_twin_proxy.export_ground_truth())["moves"])
        assert moves_after == moves_before + 1

    def test_images_are_raw_counts_with_provenance(self, mapping_twin_proxy, mapping_scan_proxy):
        mapping_twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
        image, attrs = acquire(mapping_twin_proxy, mapping_scan_proxy)["HAADF"]
        assert image.dtype == np.uint16
        assert image.shape == (64, 64)
        assert attrs["units"] == "counts"
        assert json.loads(attrs["stage_reported_m"])[0] == pytest.approx(mapping_twin_proxy.get_stage()[0])
        assert attrs["fov_m"] == pytest.approx(1e-7)

    def test_repeated_acquisition_gets_fresh_noise(self, mapping_twin_proxy, mapping_scan_proxy):
        a, _ = acquire(mapping_twin_proxy, mapping_scan_proxy)["HAADF"]
        b, _ = acquire(mapping_twin_proxy, mapping_scan_proxy)["HAADF"]
        assert not np.array_equal(a, b)

    def test_multiple_detectors_share_one_scan(self, mapping_twin_proxy, mapping_scan_proxy):
        before = len(json.loads(mapping_twin_proxy.export_ground_truth())["acquisitions"])
        images = acquire(mapping_twin_proxy, mapping_scan_proxy, detectors=("haadf", "bf"))
        after = len(json.loads(mapping_twin_proxy.export_ground_truth())["acquisitions"])
        assert set(images) == {"HAADF", "BF"}
        assert after == before + 1
        # Bright-field contrast is inverted relative to HAADF.
        haadf, bf = images["HAADF"][0].astype(float), images["BF"][0].astype(float)
        assert np.corrcoef(haadf.ravel(), bf.ravel())[0, 1] < 0

    def test_unknown_detector_is_a_clear_error(self, mapping_twin_proxy, mapping_scan_proxy):
        with pytest.raises(tango.DevFailed, match="simulates scanning detectors"):
            acquire(mapping_twin_proxy, mapping_scan_proxy, detectors=("secondary",))

    def test_scan_region_is_honoured(self, mapping_twin_proxy, mapping_scan_proxy):
        image, attrs = acquire(mapping_twin_proxy, mapping_scan_proxy, region=(0.0, 0.0, 0.5, 0.25))["HAADF"]
        assert image.shape == (16, 32)
        mapping_scan_proxy.scan_region = [0.0, 0.0, 1.0, 1.0]

    def test_scan_region_is_left_top_width_height_like_the_scan_device(self, mapping_twin_proxy, mapping_scan_proxy):
        # The convention SCAN validates and AutoScript uses; an offset region tells it apart from corners.
        try:
            centre, _ = acquire(mapping_twin_proxy, mapping_scan_proxy, region=(0.25, 0.25, 0.5, 0.5))["HAADF"]
            assert centre.shape == (32, 32)
            corner, _ = acquire(mapping_twin_proxy, mapping_scan_proxy, region=(0.5, 0.5, 0.25, 0.25))["HAADF"]
            assert corner.shape == (16, 16)
        finally:
            mapping_scan_proxy.scan_region = [0.0, 0.0, 1.0, 1.0]

    def test_bad_inputs_are_refused_and_leave_the_twin_working(self, mapping_twin_proxy, mapping_scan_proxy, mapping_stage_proxy):
        before = list(mapping_twin_proxy.get_stage())
        fov = mapping_twin_proxy.get_fov()
        for bad in (float("nan"), float("inf"), -1e-7, 1.0):
            with pytest.raises(tango.DevFailed, match="field of view"):
                mapping_twin_proxy.set_fov(bad)
        with pytest.raises(tango.DevFailed, match="nanometres"):
            mapping_twin_proxy.set_defocus(20.0)  # nm passed as metres
        with pytest.raises(tango.DevFailed, match="finite"):
            mapping_twin_proxy.move_stage([float("nan")] * 5)
        with pytest.raises(tango.DevFailed, match="alpha"):
            mapping_twin_proxy.move_stage([before[0], before[1], before[2], 90.0, 0.0])
        assert list(mapping_twin_proxy.get_stage()) == pytest.approx(before)
        assert mapping_twin_proxy.get_fov() == pytest.approx(fov)
        image, _ = acquire(mapping_twin_proxy, mapping_scan_proxy, fov=fov)["HAADF"]
        assert np.all(np.isfinite(image))
        json.loads(mapping_twin_proxy.export_ground_truth(), parse_constant=lambda c: pytest.fail(f"{c} in ground truth"))  # strict JSON for the GUI

    def test_get_fov_round_trips(self, mapping_twin_proxy):
        mapping_twin_proxy.set_fov(2.5e-7)
        assert mapping_twin_proxy.get_fov() == pytest.approx(2.5e-7)

    def test_simulated_time_advances(self, mapping_twin_proxy, mapping_scan_proxy):
        t0 = json.loads(mapping_twin_proxy.get_parameters())["simulated_time_s"]
        acquire(mapping_twin_proxy, mapping_scan_proxy, imsize=128)
        t1 = json.loads(mapping_twin_proxy.get_parameters())["simulated_time_s"]
        assert t1 - t0 == pytest.approx(128 * 128 * 1e-6, rel=0.01)

    def test_ground_truth_lists_particles_and_acquisitions(self, mapping_twin_proxy):
        truth = json.loads(mapping_twin_proxy.export_ground_truth())
        assert len(truth["particles"]) > 10
        assert truth["acquisitions"][-1]["kind"] in {"image", "spectrum"}
        assert {"true_m", "reported_m", "drift_start_m"} <= set(truth["acquisitions"][0])

    def test_eds_spectrum_has_energy_axis(self, mapping_twin_proxy, eds_proxy):
        truth = json.loads(mapping_twin_proxy.export_ground_truth())
        particle = max(truth["particles"], key=lambda p: p["semi_axes_m"][0])
        cx, cy = particle["center_m"]
        mapping_twin_proxy.set_fov(1e-7)
        mapping_twin_proxy.move_stage([cx, cy, 0.0, 0.0, 0.0])
        mapping_twin_proxy.place_beam([0.5, 0.5])
        eds_proxy.exposure_time = 2.0
        with h5py.File(mapping_twin_proxy.acquire_spectrum("eds"), "r") as h5:
            counts = h5["spectrum"][()]
            attrs = dict(h5["spectrum"].attrs)
        assert counts.shape == (2048,)
        assert counts.sum() > 0
        assert 0 < attrs["live_time_s"] <= attrs["real_time_s"] == pytest.approx(2.0)
        assert attrs["energy_dispersion_ev"] == pytest.approx(20000 / 2048)


class TestIdealTwinUnchanged:
    def test_default_twin_reports_what_it_was_told(self, twin_proxy):
        twin_proxy.move_stage([1e-9, 2e-9, 0.0, 0.0, 0.0])
        assert np.allclose(twin_proxy.get_stage()[:2], [1e-9, 2e-9])
        truth = json.loads(twin_proxy.export_ground_truth())
        assert np.allclose(truth["moves"][-1]["true"][:2], [1e-9, 2e-9])


def test_impossible_twin_configuration_is_refused_at_start(mapping_twin_proxy):
    from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin

    class Bad:
        beam_current_pa, particle_radius_mean, sample_size_xy = -50.0, 1e-9, 2e-6
        particle_elements, fov_scale_error = "", 0.0
        eds_channels, eds_energy_max, contamination_dose = 0, 20000.0, 0.0

    with pytest.raises(ValueError) as refused:
        DigitalTwin._check_properties(Bad())
    for name in ("beam_current_pa", "particle_elements", "eds_channels", "contamination_dose"):
        assert name in str(refused.value)


def test_a_detector_named_twice_is_acquired_once(mapping_twin_proxy, mapping_scan_proxy):
    images = acquire(mapping_twin_proxy, mapping_scan_proxy, detectors=("HAADF", "haadf"))
    assert list(images) == ["HAADF"]
