"""
Cross-scale registration: place a lower-magnification "overview" acquisition
into the same pixel coordinate system as a fine-resolution stitched mosaic.

This is the "zoom out and correlate" half of the mapping pipeline: after a
fine grid has been acquired and stitched (:mod:`grid_stitcher`), the agent
can step the field of view out, take one or more coarser overview images, and
register each one against the fine canvas with :func:`register_overview`.
The result places the overview as an outer, independently-sourced zoom layer
in the same map bundle rather than a synthetic downsample of the fine pixels.

Unlike the grid stitcher (pure-translation, phase correlation - appropriate
because adjacent same-magnification tiles cannot differ in scale or
rotation), an overview and the fine mosaic differ in scale by construction
and may differ in rotation if the instrument's scan rotation was not held
fixed between magnifications. Registration therefore uses scale-invariant
feature matching (ORB) and a similarity transform (RANSAC), the same
approach the classic chain stitcher used for its zoom-relation detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OverviewRegistration:
    """Placement of an overview image within a fine canvas's coordinate system.

    ``matrix`` is the 2x3 similarity transform mapping a point in the
    overview image's *native* pixel coordinates to the fine canvas's *native*
    pixel coordinates: ``fine_xy = matrix @ [overview_x, overview_y, 1]``.
    """

    matrix: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    scale: float
    rotation_deg: float
    offset_px: Tuple[float, float]
    inliers: int
    match_count: int

    def to_native_size(self, overview_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Return the (width, height) the overview spans in fine-canvas pixels."""
        h, w = overview_shape[:2]
        return w * self.scale, h * self.scale


def register_overview(
    overview: np.ndarray,
    fine_canvas: np.ndarray,
    max_dim: int = 1024,
    min_inliers: int = 12,
    ratio_test: float = 0.75,
) -> Optional[OverviewRegistration]:
    """Register a coarser *overview* image against a finer *fine_canvas*.

    Returns ``None`` (rather than raising) when matching fails - too few
    keypoints, too few good matches, or too few RANSAC inliers - so callers
    can prompt the agent to retry with a different overview position instead
    of crashing a long-running acquisition loop.
    """
    ov_grey = _to_uint8_grey(overview)
    fine_grey = _to_uint8_grey(fine_canvas)

    ov_resized, s_o = _resize_for_match(ov_grey, max_dim)
    fine_resized, s_f = _resize_for_match(fine_grey, max_dim)

    orb = cv2.ORB_create(nfeatures=4000)
    kp_o, desc_o = orb.detectAndCompute(ov_resized, None)
    kp_f, desc_f = orb.detectAndCompute(fine_resized, None)
    if desc_o is None or desc_f is None or len(kp_o) < 8 or len(kp_f) < 8:
        logger.warning("register_overview: not enough keypoints (overview=%s, fine=%s).",
                        0 if desc_o is None else len(kp_o), 0 if desc_f is None else len(kp_f))
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(desc_o, desc_f, k=2)
    good = [m for pair in raw_matches if len(pair) == 2
            for m, n in [pair] if m.distance < ratio_test * n.distance]
    if len(good) < min_inliers:
        logger.warning("register_overview: only %d good matches.", len(good))
        return None

    pts_o = np.float32([kp_o[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_f = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M_resized, inlier_mask = cv2.estimateAffinePartial2D(
        pts_o, pts_f, method=cv2.RANSAC, ransacReprojThreshold=6.0, confidence=0.995
    )
    if M_resized is None or inlier_mask is None:
        logger.warning("register_overview: RANSAC failed to find a transform.")
        return None
    inliers = int(inlier_mask.sum())
    if inliers < min_inliers:
        logger.warning("register_overview: only %d/%d inliers.", inliers, len(good))
        return None

    A = M_resized[:, :2].astype(np.float64)
    t = M_resized[:, 2].astype(np.float64)
    A_native = (s_o / s_f) * A
    t_native = t / s_f

    scale = float(np.linalg.norm(A_native[:, 0]))
    rotation_deg = float(np.degrees(np.arctan2(A_native[1, 0], A_native[0, 0])))

    matrix = (
        (float(A_native[0, 0]), float(A_native[0, 1]), float(t_native[0])),
        (float(A_native[1, 0]), float(A_native[1, 1]), float(t_native[1])),
    )
    return OverviewRegistration(
        matrix=matrix,
        scale=scale,
        rotation_deg=rotation_deg,
        offset_px=(float(t_native[0]), float(t_native[1])),
        inliers=inliers,
        match_count=len(good),
    )


def _to_uint8_grey(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _resize_for_match(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return image, scale
