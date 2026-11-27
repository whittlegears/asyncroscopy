"""Synthetic images for the analysis server's tests: map tiles, a fine image and its coarse parent, particles.

Generated from a seed, so the tests need no data files and no microscope. The same generators are in
pyTEMlib's tests/conftest.py, which tests the stitching and registration these feed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

PIXEL = 1e-9  # metres per pixel of the synthetic scenes
TILE = 256  # side of a tile cut by make_tiles
STEP = 8  # parent pixels are STEP child pixels wide


def gaussian_spots(shape=(256, 256), spots=((60, 80, 6.0, 40.0), (180, 150, 4.0, 25.0), (128, 230, 5.0, 2.0)), seed=1):
    """Gaussian particles (row, col, sigma px, height) on a background of 10 with noise of standard deviation 1."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    image = 10.0 + rng.normal(0.0, 1.0, shape)
    for row, col, sigma, height in spots:
        image += height * np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma**2))
    return image


def fine_scene(size=2400, seed=0):
    """Smooth texture with brighter blobs, seen at the child's (fine) pixel size."""
    rng = np.random.default_rng(seed)
    image = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), 6)
    blobs = np.zeros((size, size))
    blobs[rng.integers(0, size, 300), rng.integers(0, size, 300)] = 1.0
    return image / image.std() + 8 * ndimage.gaussian_filter(blobs, 10) / ndimage.gaussian_filter(blobs, 10).max()


def parent_of(scene, step=STEP):
    """What a coarse scan sees: the scene smoothed, then sampled at every step-th pixel centre."""
    smooth = ndimage.gaussian_filter(scene, 0.4 * step)
    centres = np.arange(step // 2, scene.shape[0], step) - 0.5
    rr, cc = np.meshgrid(centres, centres, indexing="ij")
    return ndimage.map_coordinates(smooth, [rr, cc], order=1)


def scene(size=1300, blobs=500, seed=0):
    """Blurred blobs of random brightness on a noisy background."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size))
    rows, cols = rng.integers(0, size, blobs), rng.integers(0, size, blobs)
    image[rows, cols] = rng.uniform(50, 200, blobs)
    image = ndimage.gaussian_filter(image, 4) * 30 + 20
    return image + rng.normal(0, 1.0, image.shape)


def make_tiles(stage_error_px, overlap=0.2, grid=4, seed=1, reported_noise_px=0.0, image=None):
    """Tiles cut where the stage really was; positions reported where it was told to go.

    Returns the tiles (pyTEMlib ``Tile``s) and their true top-left (row, col) in the scene.
    """
    from pyTEMlib.image_tools import Tile

    rng = np.random.default_rng(seed)
    image = scene() if image is None else image
    step = int(TILE * (1 - overlap))
    tiles, true_top_left = [], []
    for r in range(grid):
        for c in range(grid):
            commanded = np.array([40 + r * step, 40 + c * step], dtype=float)
            true = commanded + rng.normal(0, stage_error_px, 2)
            ti = np.clip(np.round(true).astype(int), 0, image.shape[0] - TILE)
            tiles.append(image[ti[0] : ti[0] + TILE, ti[1] : ti[1] + TILE].copy())
            true_top_left.append(ti.astype(float))
            reported = commanded + rng.normal(0, reported_noise_px, 2) if reported_noise_px else commanded
            centre_xy = ((reported[1] + TILE / 2) * PIXEL, (reported[0] + TILE / 2) * PIXEL)
            tiles[-1] = Tile(tiles[-1], centre_xy, PIXEL, index=len(tiles) - 1)
    return tiles, np.array(true_top_left)
