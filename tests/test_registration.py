"""Image registration and stage mapping, and a walkthrough of the return-to-site skill through register_images."""

import math

import numpy as np
import pytest

from asyncroscopy.imaging.registration import MatchCriteria, center_crop, judge, locate, ncc_map, offset_from_center
from asyncroscopy.imaging.stage_mapping import StageMapping, fit_rigid
from asyncroscopy.imaging.synthetic import SyntheticMicroscope

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


# -- the return-to-site skill, step by step, through the register_images MCP method ---------


class SkillHarness:
    """What an agent following skills/return-to-site/SKILL.md does, with a synthetic microscope."""

    def __init__(self, monkeypatch, scope):
        from asyncroscopy.mcp.mcp_server import MCPServer

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        self.server = MCPServer("t", "localhost", 1, blocked_classes=[], blocked_functions={},
                                data_device_address="", verbose=False)
        self.images: dict[str, np.ndarray] = {}
        monkeypatch.setattr(self.server, "_fetch_image_array", lambda key: self.images[key])
        self.scope = scope

    def acquire(self, fov, size=N) -> str:
        key = f"k{len(self.images)}"
        self.images[key] = self.scope.acquire(fov, size)
        return key

    def register(self, *args, **kw) -> dict:
        return self.server.register_images(*args, **kw)

    def calibrate(self, fov) -> list[list[float]]:
        """Skill section 2: step +s and -s on each axis from a freshly imaged start."""
        start = self.scope.get_stage()
        step = 0.15 * fov
        cols = []
        for axis in (0, 1):
            offsets = []
            for sign in (1, -1):
                self.scope.move_stage(start)
                a = self.acquire(fov)
                moved = list(start)
                moved[axis] += sign * step
                self.scope.move_stage(moved)
                r = self.register(a, self.acquire(fov), image_fov_m=fov)
                assert r["status"] == "match", r
                offsets.append(np.array(r["offset_m"]))
            cols.append((offsets[0] - offsets[1]) / (2 * step))
        self.scope.move_stage(start)
        return [[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]]

    def go_back(self, site, j, max_rotation_deg=0.0) -> dict:
        self.scope.move_stage(site["stage"])
        survey = self.acquire(4 * FOV, 2 * N)
        r = self.register(site["key"], survey, reference_fov_m=FOV, image_fov_m=4 * FOV, stage_to_image=j,
                          max_rotation_deg=max_rotation_deg)
        if r["status"] != "match":
            return r
        pos = self.scope.get_stage()
        pos[0] += r["stage_correction_m"][0]
        pos[1] += r["stage_correction_m"][1]
        self.scope.move_stage(pos)
        for _ in range(3):
            r = self.register(site["key"], self.acquire(FOV), image_fov_m=FOV, stage_to_image=j,
                              max_rotation_deg=max_rotation_deg and 1.0)
            if r["status"] != "match" or math.hypot(*r["offset_px"]) <= 2:
                return r
            pos = self.scope.get_stage()
            pos[0] += r["stage_correction_m"][0]
            pos[1] += r["stage_correction_m"][1]
            self.scope.move_stage(pos)
        return r


@pytest.fixture()
def harness(monkeypatch):
    return SkillHarness(monkeypatch, SyntheticMicroscope(seed=11, fov_m=FOV, imsize=N))


def test_calibration_through_register_images_measures_the_true_mapping(harness):
    j = harness.calibrate(FOV)
    # Exact without stage noise; 20 nm move noise on 300 nm steps leaves ~10 % per element.
    assert np.allclose(j, harness.scope.true_stage_to_image(), atol=0.2)


def test_skill_returns_to_a_site_after_a_remount_shift(harness):
    j = harness.calibrate(FOV)
    harness.scope.move_stage([4e-6, -3e-6, 0, 0, 0])
    site = {"key": harness.acquire(FOV), "stage": harness.scope.get_stage()}
    harness.scope.remount(0.0, (1.2e-6, -0.8e-6))                 # unload / reload, shifted
    r = harness.go_back(site, j)
    assert r["status"] == "match", r
    assert harness.scope.true_error_nm(site["stage"][:2]) < 3 * FOV / N * 1e9


def test_wrong_specimen_is_never_reported_as_the_site(harness, monkeypatch):
    j = harness.calibrate(FOV)
    site = {"key": harness.acquire(FOV), "stage": harness.scope.get_stage()}
    harness.scope = SyntheticMicroscope(seed=999, fov_m=FOV, imsize=N)   # a different specimen in the holder
    assert harness.go_back(site, j)["status"] != "match"


def test_register_images_refuses_a_periodic_patch(harness):
    px, py, _ = harness.scope.periodic_patch
    harness.scope.move_stage([px, py, 0, 0, 0])
    a = harness.acquire(FOV)
    harness.scope.move_stage([px + 1e-7, py, 0, 0, 0])
    assert harness.register(a, harness.acquire(FOV), image_fov_m=FOV)["status"] != "match"
