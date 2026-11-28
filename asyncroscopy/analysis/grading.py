"""Grade what an agent found against a twin's ground truth.

Detections and true objects are (row, column) points in image pixels. Each true
object can match at most one detection, nearest pairs first, within
``max_distance_px`` (or within the object's own radius when that is larger).
Unmatched detections are false positives; unmatched objects are misses.
Nearest-first matching is simple and can differ from the best possible matching
only when objects overlap.

Note: the SEGMENTATION device returns centroids as [x, y] = (column, row);
swap them before grading.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# A detection within 3 px of a small object's centre counts as finding it.
DEFAULT_MAX_DISTANCE_PX = 3.0


@dataclass
class DetectionGrade:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_position_error_px: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def grade_detections(
    detections: list[tuple[float, float]],
    truth: list[dict],
    max_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
) -> DetectionGrade:
    """Match (row, col) detections to ground-truth particles ({"row", "col", "radius_px"?})."""
    pairs = []
    for i, (row, col) in enumerate(detections):
        for j, item in enumerate(truth):
            distance = math.hypot(row - float(item["row"]), col - float(item["col"]))
            limit = max(float(max_distance_px), float(item.get("radius_px", 0.0)))
            if distance <= limit:
                pairs.append((distance, i, j))
    pairs.sort()
    used_detections: set[int] = set()
    used_truth: set[int] = set()
    errors = []
    for distance, i, j in pairs:
        if i in used_detections or j in used_truth:
            continue
        used_detections.add(i)
        used_truth.add(j)
        errors.append(distance)

    tp = len(errors)
    fp = len(detections) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionGrade(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_position_error_px=round(sum(errors) / tp, 3) if tp else None,
    )
