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


class TestRealisticTwin:
    def test_reported_position_is_not_the_true_position(self, mapping_twin_proxy):
        mapping_twin_proxy.move_stage([5e-8, -3e-8, 0.0, 0.0, 0.0])
        reported = np.array(mapping_twin_proxy.get_stage())
        truth = json.loads(mapping_twin_proxy.export_ground_truth())
        true = np.array(truth["moves"][-1]["true"])
        assert np.allclose(reported[:2], [5e-8, -3e-8], atol=6e-9)
        assert not np.allclose(reported[:2], true[:2], atol=1e-10)

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
