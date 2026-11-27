"""Mark a controller's answer against the sample's ground truth. 100 is perfect.

Three parts:

* **marker** (30 points): the marker element is right or it is not.
* **rule** (30 points): how close ``size_ratio`` is to the true ratio. No points with the wrong marker, and no points
  for an answer no better than "size does not depend on the element" (a ratio of 1).
* **map** (40 points): the share map and the hole map, cell by cell. A cell's error is 0 for a hole called a hole,
  1 for a hole missed or a film cell called a hole, and otherwise the difference between the answered and the true
  share. The points are the improvement over the best answer that knows nothing about place: no holes and the
  sample's mean share everywhere. An answer no better than that gets none.

The truth comes from the twin's ``export_ground_truth``: each region's seed, element shares and whether it is a hole.
"""

from __future__ import annotations

import numpy as np

from benchmarks.hackathon.task import GRID, Answer, cell_centres

POINTS = {"marker": 30.0, "rule": 30.0, "map": 40.0}


def truth_maps(truth: dict) -> tuple[np.ndarray, np.ndarray]:
    """(share_map, hole_map) of the truth: each cell takes the region its centre lies in (the nearest seed)."""
    seeds = np.array([region["seed_m"] for region in truth["regions"]])
    share = np.array([region["element_shares"][truth["marker_element"]] for region in truth["regions"]])
    hole = np.array([region["hole"] for region in truth["regions"]])
    x, y = np.meshgrid(cell_centres(), cell_centres())
    region = np.argmin((x[..., None] - seeds[:, 0]) ** 2 + (y[..., None] - seeds[:, 1]) ** 2, axis=-1)
    return np.where(hole[region], 0.0, share[region]), hole[region]


def true_size_ratio(truth: dict) -> float:
    """The radius where every particle is the marker over the radius where none is, from the regions' own factors."""
    regions = [region for region in truth["regions"] if not region["hole"]]
    share = np.array([region["element_shares"][truth["marker_element"]] for region in regions])
    slope, intercept = np.polyfit(share, [region["radius_factor"] for region in regions], 1)
    return float((intercept + slope) / intercept)


def map_error(share_map: np.ndarray, hole_map: np.ndarray, true_share: np.ndarray, true_hole: np.ndarray) -> float:
    """The mean error over the cells, as the module docstring defines it."""
    error = np.abs(np.clip(share_map, 0.0, 1.0) - true_share)
    error[hole_map != true_hole] = 1.0
    error[hole_map & true_hole] = 0.0
    return float(error.mean())


def score(answer: Answer, truth: dict) -> dict:
    """The answer's points and what they were made of."""
    true_share, true_hole = truth_maps(truth)
    share_map = np.asarray(answer.share_map, dtype=np.float64).reshape(GRID, GRID)
    hole_map = np.asarray(answer.hole_map, dtype=bool).reshape(GRID, GRID)

    marker_right = answer.marker_element == truth["marker_element"]
    true_ratio = true_size_ratio(truth)
    rule = max(0.0, 1.0 - abs(float(answer.size_ratio) - true_ratio) / (1.0 - true_ratio)) if marker_right else 0.0

    error = map_error(share_map, hole_map, true_share, true_hole)
    knows_nothing = map_error(np.full((GRID, GRID), true_share[~true_hole].mean()), np.zeros((GRID, GRID), bool), true_share, true_hole)
    map_skill = max(0.0, 1.0 - error / knows_nothing)

    parts = {"marker": POINTS["marker"] * marker_right, "rule": POINTS["rule"] * rule, "map": POINTS["map"] * map_skill}
    return {
        "score": float(sum(parts.values())),
        **{f"{name}_points": float(value) for name, value in parts.items()},
        "marker_answered": answer.marker_element, "marker_true": truth["marker_element"],
        "size_ratio_answered": float(answer.size_ratio), "size_ratio_true": true_ratio,
        "map_error": error, "map_error_knowing_nothing": knows_nothing,
        "holes_found": int((hole_map & true_hole).sum()), "holes_true": int(true_hole.sum()), "holes_wrong": int((hole_map & ~true_hole).sum()),
    }
