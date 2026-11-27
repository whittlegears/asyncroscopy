"""Specimen atlas: registration evidence, stage mapping, verified return-to-site, remount benchmark."""

import math

import numpy as np
import pytest

from asyncroscopy.atlas import revisit
from asyncroscopy.atlas.registration import (
    MatchCriteria,
    center_crop,
    judge,
    locate,
    ncc_map,
    offset_from_center,
)
from asyncroscopy.atlas.stage_mapping import StageMapping, fit_rigid
from asyncroscopy.atlas.store import Atlas
from asyncroscopy.atlas.synthetic import SyntheticMicroscope

FOV, N = 2e-6, 128


def blobs(seed=0, n=60, size=128):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = np.zeros((size, size))
    for _ in range(n):
        y, x = rng.uniform(0, size, 2)
        img += rng.uniform(0.3, 1) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * rng.uniform(1.5, 4) ** 2))
    return img


# -- registration -------------------------------------------------------------

def test_ncc_map_finds_exact_crop():
    img = blobs(1)
    score = ncc_map(img, img[30:70, 40:90])
    assert np.unravel_index(np.argmax(score), score.shape) == (30, 40)
    assert score.max() == pytest.approx(1.0, abs=1e-6)


def test_locate_recovers_shift_with_evidence():
    img = blobs(2, size=160)
    ref, moved = img[16:144, 16:144], img[10:138, 23:151]   # features move 6 px down, 7 px left
    loc = locate(center_crop(ref, 0.6), moved)
    dx, dy = offset_from_center(loc, moved.shape)
    assert dx == pytest.approx(-7, abs=0.3) and dy == pytest.approx(6, abs=0.3)
    assert loc.ncc > 0.99 and loc.consistency == 1.0
    assert judge(loc, MatchCriteria()).ok


def test_periodic_pattern_is_ambiguous_not_a_match():
    yy, xx = np.mgrid[:128, :128]
    lattice = np.cos(2 * np.pi * xx / 16) * np.cos(2 * np.pi * yy / 16)
    lattice += np.random.default_rng(0).normal(0, 0.05, lattice.shape)
    loc = locate(center_crop(lattice, 0.5), np.roll(lattice, 5, axis=1))
    verdict = judge(loc, MatchCriteria())
    assert verdict.status == "ambiguous"
    assert loc.ambiguity > 0.8


def test_unrelated_image_is_no_match():
    loc = locate(center_crop(blobs(3), 0.6), blobs(4))
    assert judge(loc, MatchCriteria()).status in ("no_match", "ambiguous")


def test_rotation_search_finds_remount_angle():
    from scipy import ndimage
    img = blobs(5, n=120, size=200)
    rotated = ndimage.rotate(img, 6.0, reshape=False, order=1, mode="reflect")
    loc = locate(center_crop(img, 0.5), rotated, rotations_deg=range(-10, 11))
    assert abs(abs(loc.rotation_deg) - 6.0) < 0.6


# -- stage mapping ------------------------------------------------------------

def test_mapping_inverts_jacobian():
    j = [[-0.9, 0.4], [-0.4, -0.85]]
    m = StageMapping(stage_to_image=j, calibrated=True)
    d = m.stage_delta_for_image_offset((1e-7, -2e-7))
    assert np.allclose(np.asarray(j) @ d, [-1e-7, 2e-7])


def test_fit_rigid_recovers_rotation_and_translation():
    old = np.array([[0, 0], [10e-6, 0], [0, 8e-6], [-5e-6, 4e-6]])
    t = math.radians(3.0)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    new = old @ r.T + [2e-6, -1e-6]
    fit = fit_rigid(old, new)
    assert fit.rotation_deg == pytest.approx(3.0, abs=1e-6)
    assert np.allclose(fit.translation_m, [2e-6, -1e-6])
    assert np.allclose(fit.apply(old[1]), new[1])
    assert fit.rms_residual_m == pytest.approx(0.0, abs=1e-12)


# -- verified return ------------------------------------------------------------

@pytest.fixture()
def scope():
    return SyntheticMicroscope(seed=11, fov_m=FOV, imsize=N)


def test_calibration_measures_the_true_mapping(tmp_path, scope):
    atlas = Atlas(tmp_path)
    mapping = revisit.calibrate_mapping(atlas, scope, FOV, N)
    assert mapping.calibrated
    # Exact without stage noise; 20 nm move noise on 300 nm steps leaves ~10 % per element.
    assert np.allclose(mapping.stage_to_image, scope.true_stage_to_image(), atol=0.2)
    assert Atlas(tmp_path).mapping.calibrated     # persisted


def test_return_to_site_matches_and_records_visit(tmp_path, scope):
    atlas = Atlas(tmp_path)
    revisit.calibrate_mapping(atlas, scope, FOV, N)
    scope.move_stage([4e-6, -3e-6, 0, 0, 0])
    site = revisit.save_site(atlas, scope, "grain boundary", FOV, N, provenance={"action_id": "a1"})
    scope.move_stage([0, 0, 0, 0, 0])
    steps = []
    visit = revisit.return_to_site(atlas, site.id, scope,
                                   step=lambda name, args, fn: (steps.append(name), fn())[1])
    assert visit["status"] == "match", visit["reasons"]
    assert scope.true_error_nm(site.stage[:2]) < 3 * N and visit["residual_nm"] < 2.5 * FOV / N * 1e9
    assert {"move_stage", "acquire_survey", "acquire"} <= set(steps)
    reopened = Atlas(tmp_path)                     # a teammate opens the atlas later
    s = reopened.get(site.id)
    assert s.provenance == {"action_id": "a1"} and s.visits[0]["status"] == "match"
    assert reopened.visit_image(site.id, s.visits[0]["id"]).shape == (N, N)
    assert reopened.reference(site.id).shape == (N, N)
    assert [r["kind"] for r in reopened.history()][-1] == "site_visited"


def test_wrong_specimen_is_never_a_match(tmp_path, scope):
    atlas = Atlas(tmp_path)
    revisit.calibrate_mapping(atlas, scope, FOV, N)
    site = revisit.save_site(atlas, scope, "here", FOV, N)
    other = SyntheticMicroscope(seed=999, fov_m=FOV, imsize=N)     # a different specimen in the holder
    visit = revisit.return_to_site(atlas, site.id, other)
    assert visit["status"] != "match"


def test_remount_benchmark_finds_sites_and_is_never_confidently_wrong(tmp_path):
    from asyncroscopy.atlas.benchmark import run, synthetic_backend

    mic, remount, truth, sim = synthetic_backend(seed=11, fov=FOV, imsize=N)
    px, py, _ = sim.periodic_patch
    report = run(mic, remount, truth, tmp_path, n_sites=4, n_landmarks=3, fov=FOV, imsize=N, spread=24e-6,
                 rotation_deg=4.0, shift_m=(3e-6, -2e-6), seed=11,
                 extra_sites=[("periodic", np.array([px, py]))], log=lambda *_: None)
    s = report["summary"]
    assert s["confidently_wrong"] == 0
    assert s["landmarks_verified"] >= 2
    assert s["fitted_rotation_deg"] == pytest.approx(4.0, abs=0.5)
    assert s["found"] >= 3
    periodic = next(r for r in report["rows"] if r["kind"] == "extra")
    assert periodic["status"] != "match"
    assert (tmp_path / "benchmark.md").exists()
