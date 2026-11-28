import json

import h5py
import pytest

from asyncroscopy.analysis.grading import grade_detections


def test_perfect_detections():
    truth = [{"row": 10, "col": 10}, {"row": 40, "col": 50}]
    grade = grade_detections([(10.5, 9.5), (40, 50)], truth)

    assert (grade.true_positives, grade.false_positives, grade.false_negatives) == (2, 0, 0)
    assert grade.f1 == 1.0
    assert grade.mean_position_error_px == pytest.approx(0.354, abs=1e-3)


def test_misses_and_false_alarms():
    truth = [{"row": 10, "col": 10}, {"row": 40, "col": 50}]
    grade = grade_detections([(10, 11), (90, 90)], truth)

    assert (grade.true_positives, grade.false_positives, grade.false_negatives) == (1, 1, 1)
    assert grade.precision == 0.5
    assert grade.recall == 0.5


def test_one_detection_cannot_claim_two_particles():
    truth = [{"row": 10, "col": 10}, {"row": 11, "col": 10}]
    grade = grade_detections([(10.4, 10)], truth)

    assert grade.true_positives == 1
    assert grade.false_negatives == 1


def test_a_large_particle_accepts_a_detection_inside_its_radius():
    grade = grade_detections([(10, 18)], [{"row": 10, "col": 10, "radius_px": 9}])
    assert grade.true_positives == 1


def test_nothing_to_find_and_nothing_found_is_perfect():
    assert grade_detections([], []).f1 == 1.0


def test_twin_ground_truth_matches_particles_found_in_its_image(twin_proxy, scan_proxy):
    """The hidden answer lines up with the image the agent sees."""
    import numpy as np
    from scipy import ndimage

    scan_proxy.imsize = 128
    scan_proxy.dwell_time = 1e-6
    twin_proxy.set_fov(12e-9)
    twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
        image = h5["image/HAADF"][()]
    truth = json.loads(twin_proxy.get_ground_truth())

    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=float), 2)
    labels, count = ndimage.label(smooth > smooth.mean() + smooth.std())
    found = ndimage.center_of_mass(labels > 0, labels, range(1, count + 1))
    grade = grade_detections(found, truth["particles"])

    assert truth["imsize_px"] == 128
    assert truth["pixel_size_m"] == pytest.approx(12e-9 / 128)
    assert len(truth["particles"]) >= 2
    assert grade.recall == 1.0
    assert grade.mean_position_error_px < 2.0



def test_a_twin_without_particle_positions_refuses_to_give_ground_truth(tango_ctx):
    import tango

    particles = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwinparticles/default"))
    with pytest.raises(tango.DevFailed, match="no ground truth"):
        particles.get_ground_truth()
