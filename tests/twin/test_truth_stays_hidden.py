"""What a twin really is (the sample, the true stage pose, the true beam tilt) reaches export_ground_truth only:
never an acquisition's saved attributes and never a reply an agent gets."""

import json
from pathlib import Path

import numpy as np
import pytest
import tango

from asyncroscopy.instruments.electron_microscope import polycrystalline_gold_digital_twin

HIDDEN_GOLD = {"grain_size_angstrom", "empty_grain_fraction", "last_rendered_atom_count", "raw_potential_cache_hits",
               "corrector_aberrations", "volume_size_xy_nm", "volume_thickness_nm", "lattice_constant_angstrom"}


def _server_device(name):
    return tango.Util.instance().get_device_by_name(name)


def test_gold_acquisitions_carry_no_sample_truth(tango_ctx, monkeypatch):
    saved = []

    def capture(device, data_server, kind, detectors, arrays, dataset_attrs=None, file_attrs=None, **kwargs):
        saved.append((dataset_attrs, file_attrs))
        return "key"

    monkeypatch.setattr(polycrystalline_gold_digital_twin, "save_acquisition", capture)
    gold = _server_device("asyncroscopy/atomistic/polycrystal")
    gold._set_fov(1e-8)
    gold._acquire_scanned_image(32, 1e-5, ["haadf"], [0.0, 0.0, 1.0, 1.0])
    (dataset_attrs, file_attrs) = saved[-1]
    for attrs in [*dataset_attrs, file_attrs]:
        assert not HIDDEN_GOLD & set(attrs), HIDDEN_GOLD & set(attrs)
        assert not any("grain" in key for key in attrs)
    truth = gold._ground_truth()
    record = truth["acquisitions"][file_attrs["acquisition_index"]]
    assert record["last_rendered_atom_count"] > 0 and record["corrector_aberrations"]
    assert "grain_size_angstrom" in truth["volume"]


def test_the_tilt_twin_reports_the_commanded_pose_not_the_true_one(tilt_twin_proxy):
    twin = _server_device("asyncroscopy/digitaltwintilt/default")
    noise = (twin.stage_translation_noise_std_nm, twin.randomness_scale)
    twin.stage_translation_noise_std_nm, twin.randomness_scale = 5.0, 1.0
    try:
        commanded = [2e-8, -1e-8, 0.0, 0.0, 0.0]
        tilt_twin_proxy.move_stage(commanded)
        assert not np.allclose(twin._stage_position, commanded, atol=1e-12)  # the stage landed somewhere else
        parameters = json.loads(tilt_twin_proxy.get_parameters())
        assert "stage_actual" not in parameters
        assert np.allclose(list(tilt_twin_proxy.get_stage()), commanded, rtol=0.0, atol=1e-15)
        assert json.loads(twin.export_ground_truth())["sample_seed"] == int(twin.sample_seed)
    finally:
        twin.stage_translation_noise_std_nm, twin.randomness_scale = noise
        tilt_twin_proxy.move_stage([0.0] * 5)


def test_the_tilt_twin_reports_the_beam_tilt_as_set(tilt_twin_proxy):
    twin = _server_device("asyncroscopy/digitaltwintilt/default")
    noise = (twin.beam_tilt_noise_std_mrad, twin.randomness_scale)
    twin.beam_tilt_noise_std_mrad, twin.randomness_scale = 0.5, 1.0
    try:
        twin._set_beam_tilt([1e-3, 0.0])
        assert not np.allclose(twin._beam_tilt, [1e-3, 0.0], atol=1e-9)
        assert np.allclose(twin._get_beam_tilt(), [1e-3, 0.0])
        assert json.loads(twin._get_parameters())["beam_tilt_mrad"] == pytest.approx([1.0, 0.0])
    finally:
        twin.beam_tilt_noise_std_mrad, twin.randomness_scale = noise
        twin._set_beam_tilt([0.0, 0.0])


def test_no_ground_truth_file_lands_in_the_tree_tiled_serves(tango_ctx, twin_proxy, scan_proxy, sem_twin_proxy, data_save_dir, ground_truth_dir):
    scan_proxy.imsize, scan_proxy.dwell_time, scan_proxy.scan_region = 16, 1e-6, [0.0, 0.0, 1.0, 1.0]
    sem_scan = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/scan/sem"))
    sem_scan.imsize, sem_scan.dwell_time = 16, 1e-6
    stem_key = twin_proxy.acquire_scanned_image(["haadf"])
    sem_key = sem_twin_proxy.acquire_scanned_image(["se"])

    served = Path(data_save_dir).resolve()
    served_files = [path for path in served.rglob("*") if path.is_file()]
    assert {Path(stem_key).name, Path(sem_key).name} <= {path.name for path in served_files}
    leaked = [path for path in served_files if path.suffix == ".jsonl" or "ground_truth" in path.relative_to(served).parts]
    assert not leaked, leaked
    assert not Path(ground_truth_dir).resolve().is_relative_to(served)

    truth_files = {path.name for path in Path(ground_truth_dir).iterdir()}
    assert {"asyncroscopy_digitaltwin_default_acquisitions.jsonl", "asyncroscopy_sem_default_acquisitions.jsonl"} <= truth_files
    acquisitions = json.loads(twin_proxy.export_ground_truth())["acquisitions"]
    assert acquisitions and acquisitions[-1]["imsize"] == 16
