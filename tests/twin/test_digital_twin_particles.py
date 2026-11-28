import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango


def test_particle_twin_starts_with_persistent_voxel_world(particles_twin_proxy):
    assert particles_twin_proxy.state() == tango.DevState.ON
    assert particles_twin_proxy.manufacturer == "UTKTeam Particle Digital Twin"


def test_particle_twin_saves_haadf_with_world_metadata(
    particles_twin_proxy,
    scan_proxy,
    stage_proxy,
):
    scan_proxy.imsize = 48
    scan_proxy.dwell_time = 1e-6
    stage_proxy.position = [0.0, 0.0, 0.0, 0.0, 0.0]

    path = Path(particles_twin_proxy.acquire_scanned_image(["HAADF"]))

    with h5py.File(path, "r") as h5:
        image = h5["image/HAADF"]
        assert image.shape == (48, 48)
        assert image.attrs["sample_type"] == "oriented_particles"
        assert "particle_count" not in image.attrs
        assert json.loads(image.attrs["world_shape"]) == [48, 48, 24]


def test_particle_twin_stage_tilt_recooks_projection(
    particles_twin_proxy,
    scan_proxy,
    stage_proxy,
):
    scan_proxy.imsize = 48
    stage_proxy.position = [0.0, 0.0, 0.0, 0.0, 0.0]
    untilted_path = Path(particles_twin_proxy.acquire_scanned_image(["HAADF"]))
    with h5py.File(untilted_path, "r") as h5:
        untilted = h5["image/HAADF"][()]

    stage_proxy.position = [0.0, 0.0, 0.0, 12.0, 0.0]
    tilted_path = Path(particles_twin_proxy.acquire_scanned_image(["HAADF"]))
    with h5py.File(tilted_path, "r") as h5:
        tilted = h5["image/HAADF"][()]

    assert np.mean(np.abs(tilted - untilted)) > 1e-4


def test_particle_twin_fov_round_trip(particles_twin_proxy):
    particles_twin_proxy.set_fov(18e-9)
    assert particles_twin_proxy.get_fov() == pytest.approx(18e-9)


def test_particle_twin_keeps_its_truth_for_the_operator_and_checks_a_beam_position_write(particles_twin_proxy):
    import json

    from asyncroscopy.instruments.electron_microscope.digital_twin_particles import DigitalTwinParticles
    from asyncroscopy.utils.commands import declared_hints

    parameters = json.loads(particles_twin_proxy.get_parameters())
    assert "particle_count" not in parameters and "sample_seed" not in parameters
    truth = json.loads(particles_twin_proxy.export_ground_truth())
    assert truth["particle_count"] >= 0 and "sample_seed" in truth
    assert declared_hints(DigitalTwinParticles)["export_ground_truth"]["operator_only"] is True
    with pytest.raises(tango.DevFailed, match="place_beam takes"):
        particles_twin_proxy.beam_pos = [1.2, 0.3]
    particles_twin_proxy.beam_pos = [0.2, 0.3]
    assert list(particles_twin_proxy.beam_pos) == pytest.approx([0.2, 0.3])
    particles_twin_proxy.place_beam([0.25, 0.75])
    assert list(particles_twin_proxy.beam_pos) == pytest.approx([0.25, 0.75])


@pytest.mark.parametrize("twin_class", ["DigitalTwinParticles", "PolycrystallineGoldDigitalTwin"])
def test_a_twin_without_a_camera_model_refuses_a_camera_image(twin_class):
    """It used to return a HAADF scan labelled as a camera image; now it says it has no camera."""
    from asyncroscopy.instruments.electron_microscope import digital_twin_particles, polycrystalline_gold_digital_twin

    module = {"DigitalTwinParticles": digital_twin_particles, "PolycrystallineGoldDigitalTwin": polycrystalline_gold_digital_twin}[twin_class]
    cls = getattr(module, twin_class)
    twin = cls.__new__(cls)
    with pytest.raises(tango.DevFailed, match="no camera model"):
        cls._acquire_camera_image(twin, 64, 0.1, "BM-Ceta", "Full", 1, ".h5")


