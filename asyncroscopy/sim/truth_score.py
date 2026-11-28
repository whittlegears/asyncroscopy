"""Score a run against a digital twin's own ground truth (``export_ground_truth``); ``docs/MCP/twin_to_instrument.md``
describes the score. The geometry helpers are shared with ``benchmarks.evaluate``."""

from __future__ import annotations

from typing import Any

import numpy as np

score_version = "1.0.0"
score_label = "twin score: measured against the digital twin's own ground truth, not a real specimen"


def acquisition_index(record: dict) -> int | None:
    found = record.get("index", record.get("acquisition_index"))
    return None if found is None else int(found)


def truth_marker(truth: dict) -> dict:
    indices = [i for i in (acquisition_index(a) for a in truth.get("acquisitions") or []) if i is not None]
    dose = truth.get("dose") if isinstance(truth.get("dose"), dict) else {}
    return {"last_acquisition": max(indices) if indices else None, "simulated_time_s": truth.get("simulated_time_s"),
            "total_electrons": dose.get("total_electrons")}


def inside_field(centres: np.ndarray, centre: tuple[float, float], half_x: float, half_y: float) -> np.ndarray:
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    return (np.abs(centres[:, 0] - centre[0]) < half_x) & (np.abs(centres[:, 1] - centre[1]) < half_y)


def points_on_particles(points: np.ndarray, centres: np.ndarray, radii: np.ndarray, tolerance_m: float = 0.0) -> np.ndarray:
    """Which particles a point lands on, within the particle's radius or ``tolerance_m``, whichever is larger."""
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points) or not len(centres):
        return np.zeros(len(centres), dtype=bool)
    distance = np.hypot(points[:, None, 0] - centres[None, :, 0], points[:, None, 1] - centres[None, :, 1]).min(axis=0)
    return distance <= np.maximum(np.asarray(radii, dtype=np.float64), float(tolerance_m))


def median_error(estimates: Any, truths: Any, remove_offset: bool = True) -> float:
    """Median distance from estimates to truths, both (n, 2); ``remove_offset`` drops the common shift of a stitched frame."""
    diff = np.asarray(estimates, dtype=np.float64) - np.asarray(truths, dtype=np.float64)
    if remove_offset:
        diff = diff - np.median(diff, axis=0)
    return float(np.median(np.hypot(diff[:, 0], diff[:, 1])))


def particles(truth: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ids, centres (n, 2) m, radii m) of the twin's particles."""
    items = [p for p in truth.get("particles") or [] if isinstance(p, dict) and p.get("center_m")]
    ids = np.array([int(p.get("id", k)) for k, p in enumerate(items)], dtype=np.int64)
    centres = np.array([p["center_m"][:2] for p in items], dtype=np.float64).reshape(-1, 2)
    radii = np.array([min((p.get("semi_axes_m") or [0.0, 0.0])[:2]) for p in items], dtype=np.float64)
    return ids, centres, radii


def reported_features(record: dict) -> list[tuple[float, float]]:
    """The stage positions the run's ``find_feature`` nodes reported."""
    nodes = (record.get("blueprint") or {}).get("nodes") or {}
    outputs = record.get("outputs") or {}
    points = []
    for name, node in nodes.items():
        found = outputs.get(name)
        if node.get("block") != "find_feature" or not isinstance(found, dict):
            continue
        for feature in found.get("features") or []:
            if isinstance(feature, dict) and isinstance(feature.get("x_m"), (int, float)) and isinstance(feature.get("y_m"), (int, float)):
                points.append((float(feature["x_m"]), float(feature["y_m"])))
    return list(dict.fromkeys(points))


def twin_score(marker: dict, truth: dict, record: dict) -> dict:
    """The twin score of a run, from the ``truth_marker`` taken before it, the truth after it and the runner's record."""
    since = marker.get("last_acquisition")
    acquisitions = [a for a in truth.get("acquisitions") or [] if isinstance(a, dict) and acquisition_index(a) is not None
                    and (since is None or acquisition_index(a) > since)]
    images = [a for a in acquisitions if a.get("center_true_m") and a.get("fov_m")]
    ids, centres, radii = particles(truth)
    imaged = np.zeros(len(ids), dtype=bool)
    for image in images:
        half = float(image["fov_m"]) / 2.0
        imaged |= inside_field(centres, tuple(image["center_true_m"][:2]), half, half)
    pixels = [float(a["fov_m"]) / float(a["imsize"]) for a in images if a.get("imsize")]
    tolerance = 2.0 * float(np.median(pixels)) if pixels else 0.0
    points = reported_features(record)
    on_particles = sum(bool(points_on_particles(np.array([point]), centres, radii, tolerance).any()) for point in points)
    stage = [(a["reported_m"][:2], a["true_m"][:2]) for a in images if a.get("reported_m") and a.get("true_m")]
    errors = [float(np.hypot(r[0] - t[0], r[1] - t[1])) for r, t in stage]
    return {
        "label": score_label, "version": score_version, "known": bool(truth.get("particles")) or bool(images),
        "acquisitions": len(acquisitions), "images": len(images),
        "particles_total": len(ids), "particles_imaged": int(imaged.sum()),
        "features_reported": len(points), "features_on_particles": on_particles,
        "feature_precision": on_particles / len(points) if points else None,
        "stage_error_m": {"median": float(np.median(errors)), "max": float(max(errors)), "images": len(errors)} if errors else None,
    }


def twin_time_and_dose(marker: dict, truth: dict) -> tuple[float | None, dict | None]:
    """(twin clock seconds the run took, its dose or None when the twin keeps no dose map)."""
    before, after = marker.get("simulated_time_s"), truth.get("simulated_time_s")
    clock = float(after) - float(before) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
    dose = truth.get("dose") if isinstance(truth.get("dose"), dict) else None
    if dose is None:
        return clock, None
    total, start = dose.get("total_electrons"), marker.get("total_electrons")
    return clock, {"total_electrons": float(total) - float(start or 0.0) if isinstance(total, (int, float)) else None,
                   "max_dose_e_per_a2": dose.get("max_dose_e_per_a2")}
