"""A fast synthetic microscope for testing and benchmarking return-to-site.

It is not a physics model. It exists to answer one question quickly and with
ground truth: *after the specimen is unloaded and remounted, does registration
find the saved sites, how close does it get, and does it ever confidently
pick the wrong spot?*

What it models:

* a specimen of random particles (Gaussian blobs of varied size and
  brightness) over a large area, plus an optional **periodic patch** of
  identical particles on a perfect grid, where a translation-only match is
  genuinely ambiguous;
* a stage whose axes are rotated and differently scaled relative to the scan
  (so an uncalibrated identity mapping is wrong), with Gaussian move noise;
* a specimen **mount**: rotation and translation in the stage plane, changed
  by :meth:`SyntheticMicroscope.remount`;
* shot noise and a slow background.

``true_error_nm`` compares the current stage position to where a site
*actually* is now, which the code under test never gets to see.
"""

from __future__ import annotations

import math

import numpy as np


def _rot(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


class SyntheticMicroscope:
    def __init__(self, *, seed: int = 0, extent_m: float = 40e-6, n_particles: int = 20000,
                 particle_radius_m: tuple[float, float] = (30e-9, 120e-9),
                 periodic_patch: tuple[float, float, float] | None = (-14e-6, 14e-6, 5e-6),
                 periodic_pitch_m: float = 400e-9,
                 scan_rotation_deg: float = 23.0, stage_gain: tuple[float, float] = (1.0, 0.93),
                 move_noise_m: float = 20e-9, counts: float = 60.0, fov_m: float = 2e-6, imsize: int = 128):
        self.rng = np.random.default_rng(seed)
        self.extent_m = extent_m
        # --- specimen (specimen coordinates, metres) ---
        n = n_particles
        xy = self.rng.uniform(-extent_m / 2, extent_m / 2, size=(n, 2))
        radius = self.rng.uniform(*particle_radius_m, size=n)
        amp = self.rng.uniform(0.3, 1.0, size=n)
        if periodic_patch is not None:
            px, py, size = periodic_patch
            inside = (np.abs(xy[:, 0] - px) < size / 2) & (np.abs(xy[:, 1] - py) < size / 2)
            xy, radius, amp = xy[~inside], radius[~inside], amp[~inside]
            g = np.arange(-size / 2, size / 2, periodic_pitch_m)
            gx, gy = np.meshgrid(px + g, py + g)
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            xy = np.vstack([xy, grid])
            radius = np.concatenate([radius, np.full(len(grid), periodic_pitch_m * 0.25)])
            amp = np.concatenate([amp, np.full(len(grid), 0.8)])
        self.periodic_patch = periodic_patch
        self.particles = xy
        self.radius = radius
        self.amp = amp
        # --- optics: image offset (m) = A @ (feature_stage - stage) ---
        self.A = _rot(scan_rotation_deg) @ np.diag(stage_gain)
        self.move_noise_m = move_noise_m
        self.counts = counts
        # --- mount: feature_stage = R @ specimen + t ---
        self.mount_rotation_deg = 0.0
        self.mount_translation = np.zeros(2)
        # --- state ---
        self.stage = np.zeros(5)
        self.fov_m = fov_m
        self.imsize = imsize
        self.acquisitions = 0
        self.moves = 0

    # -- the instrument protocol ----------------------------------------------
    def get_stage(self) -> list[float]:
        return [float(v) for v in self.stage]

    def move_stage(self, position) -> None:
        target = np.asarray(position, dtype=float).copy()
        if len(target) < 5:
            target = np.concatenate([target, np.zeros(5 - len(target))])
        if self.move_noise_m:
            target[:2] += self.rng.normal(0.0, self.move_noise_m, size=2)
        self.stage = target
        self.moves += 1

    def acquire(self, fov_m: float, imsize: int) -> np.ndarray:
        self.fov_m, self.imsize = float(fov_m), int(imsize)
        self.acquisitions += 1
        px = fov_m / imsize
        c = (np.arange(imsize) - (imsize - 1) / 2.0) * px
        ox, oy = np.meshgrid(c, c)                       # image offsets, x right / y down
        offsets = np.stack([ox.ravel(), oy.ravel()])
        stage_pts = self.stage[:2, None] + np.linalg.solve(self.A, offsets)      # stage coords per pixel
        spec_pts = _rot(-self.mount_rotation_deg) @ (stage_pts - self.mount_translation[:, None])
        # Particles near the view only.
        centre = spec_pts.mean(axis=1)
        reach = fov_m * 0.75 + 3 * self.radius.max()
        near = np.max(np.abs(self.particles - centre), axis=1) < reach
        image = np.zeros(offsets.shape[1])
        for (x, y), r, a in zip(self.particles[near], self.radius[near], self.amp[near]):
            d2 = (spec_pts[0] - x) ** 2 + (spec_pts[1] - y) ** 2
            image += a * np.exp(-d2 / (2 * r * r))
        image = image.reshape(imsize, imsize)
        image += 0.1 * (1 + np.sin(spec_pts[0].reshape(imsize, imsize) / 7e-6))   # slow background
        noisy = self.rng.poisson(np.clip(image, 0, None) * self.counts) / self.counts
        return noisy.astype(np.float32)

    def current_fov(self) -> float:
        return self.fov_m

    def current_imsize(self) -> int:
        return self.imsize

    def settings(self) -> dict:
        return {"backend": "synthetic", "fov_m": self.fov_m, "imsize": self.imsize,
                "stage": self.get_stage(), "counts": self.counts}

    # -- test helpers (the code under test never calls these) ----------------------
    def true_stage_to_image(self) -> np.ndarray:
        """The J a calibration should measure: stage move d shifts features by -A @ d."""
        return -self.A

    def remount(self, rotation_deg: float, translation_m: tuple[float, float]) -> None:
        """Unload and remount the specimen at a new angle and position."""
        self.mount_rotation_deg = float(rotation_deg)
        self.mount_translation = np.asarray(translation_m, dtype=float)

    def where_is(self, original_stage_xy) -> np.ndarray:
        """Stage x, y that centres a feature saved at *original_stage_xy* on the first mount."""
        return _rot(self.mount_rotation_deg) @ np.asarray(original_stage_xy, dtype=float) + self.mount_translation

    def true_error_nm(self, original_stage_xy) -> float:
        return float(np.linalg.norm(self.stage[:2] - self.where_is(original_stage_xy)) * 1e9)

    def in_periodic_patch(self, original_stage_xy) -> bool:
        if self.periodic_patch is None:
            return False
        px, py, size = self.periodic_patch
        return abs(original_stage_xy[0] - px) < size / 2 and abs(original_stage_xy[1] - py) < size / 2
