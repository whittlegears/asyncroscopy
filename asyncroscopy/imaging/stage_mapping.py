"""Image <-> stage mapping and specimen-remount transforms.

Pixel to stage: stage axes are generally not aligned with the scan axes and can have
different gains. Moving the stage by ``d = (dx, dy)`` metres shifts image
features by ``J @ d`` metres (image x right, y down). To bring a feature at
image offset ``o`` from the centre to the centre, ``J @ d = -o``, so
``d = -J^-1 @ o``.

A mapped coordinate is only a *starting guess*: re-image and register
(``register_images``) before treating a position as a previously visited site.

:func:`calibrate` measures ``J`` on the instrument with two known stage steps.
:func:`fit_rigid` estimates how a remounted specimen moved (rotation and
translation in the stage plane) from landmarks reacquired at old and new stage
positions.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

IDENTITY = [[1.0, 0.0], [0.0, 1.0]]


def validate_matrix(matrix) -> np.ndarray:
    j = np.asarray(matrix, dtype=float)
    if j.shape != (2, 2) or not np.all(np.isfinite(j)):
        raise ValueError(f"stage_to_image must be a finite 2x2 matrix, got {matrix!r}")
    if abs(np.linalg.det(j)) < 1e-6:
        raise ValueError(f"stage_to_image {matrix!r} is singular")
    return j


@dataclass
class StageMapping:
    """``stage_to_image``: image shift (m) per stage move (m)."""

    stage_to_image: list[list[float]] = field(default_factory=lambda: [r[:] for r in IDENTITY])
    calibrated: bool = False
    measured_at: float | None = None
    step_m: float | None = None
    residual: float | None = None
    notes: list[str] = field(default_factory=list)

    def stage_delta_for_image_offset(self, offset_m: tuple[float, float]) -> tuple[float, float]:
        """Stage move (m) that brings the feature at image offset *offset_m* to the centre."""
        j = validate_matrix(self.stage_to_image)
        d = -np.linalg.solve(j, np.asarray(offset_m, dtype=float))
        return float(d[0]), float(d[1])

    def pixel_to_stage(self, stage_xy: tuple[float, float], px: float, py: float,
                       shape: tuple[int, int], pixel_size_m: float) -> tuple[float, float]:
        """Stage x, y (m) that centres pixel (px, py) of an image taken at *stage_xy*."""
        rows, cols = shape
        offset = ((px - (cols - 1) / 2.0) * pixel_size_m, (py - (rows - 1) / 2.0) * pixel_size_m)
        dx, dy = self.stage_delta_for_image_offset(offset)
        return stage_xy[0] + dx, stage_xy[1] + dy

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict | None) -> StageMapping:
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def calibrate(measure_shift, step_m: float) -> StageMapping:
    """Measure ``stage_to_image`` with two known stage steps.

    *measure_shift(dx_m, dy_m)* must move the stage by (dx, dy) from a fixed
    start, image, and return the feature shift (sx, sy) in metres relative to
    the start image (x right, y down). It is called for +x and +y steps;
    callers should approach from the same side each time to keep backlash out.
    """
    sx_x, sy_x = measure_shift(step_m, 0.0)
    sx_y, sy_y = measure_shift(0.0, step_m)
    j = np.array([[sx_x, sx_y], [sy_x, sy_y]], dtype=float) / step_m
    notes = []
    gains = np.linalg.norm(j, axis=0)
    if np.any(gains < 0.5) or np.any(gains > 2.0):
        notes.append(f"stage gains {gains.round(3).tolist()} are far from 1; check the step size "
                     f"or the image pixel size")
    validate_matrix(j)
    return StageMapping(stage_to_image=j.tolist(), calibrated=True, measured_at=time.time(),
                        step_m=step_m, notes=notes)


# ----------------------------------------------------------------------------
# Remount transform
# ----------------------------------------------------------------------------

@dataclass
class MountTransform:
    """Maps a stage position recorded on the original mount to the current mount.

    ``new = R(rotation) @ old + translation`` in the stage x/y plane.
    """

    rotation_deg: float = 0.0
    translation_m: list[float] = field(default_factory=lambda: [0.0, 0.0])
    rms_residual_m: float | None = None
    landmarks: list[str] = field(default_factory=list)
    fitted_at: float | None = None

    def apply(self, xy: tuple[float, float]) -> tuple[float, float]:
        t = math.radians(self.rotation_deg)
        c, s = math.cos(t), math.sin(t)
        x, y = float(xy[0]), float(xy[1])
        return (c * x - s * y + self.translation_m[0], s * x + c * y + self.translation_m[1])

    @property
    def identity(self) -> bool:
        return self.rotation_deg == 0.0 and not any(self.translation_m)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict | None) -> MountTransform:
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def fit_rigid(old_xy, new_xy, names: list[str] | None = None) -> MountTransform:
    """Least-squares rotation + translation (Kabsch, 2-D) from landmark pairs.

    One landmark gives translation only. Two or more also give rotation, and
    three or more give a residual that says whether the fit can be trusted.
    """
    old = np.asarray(old_xy, dtype=float).reshape(-1, 2)
    new = np.asarray(new_xy, dtype=float).reshape(-1, 2)
    if len(old) != len(new) or len(old) == 0:
        raise ValueError("need the same, non-zero number of old and new landmark positions")
    if len(old) == 1:
        t = new[0] - old[0]
        return MountTransform(0.0, t.tolist(), None, list(names or []), time.time())
    co, cn = old.mean(axis=0), new.mean(axis=0)
    h = (old - co).T @ (new - cn)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T
    t = cn - r @ co
    pred = (r @ old.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - new) ** 2, axis=1)))) if len(old) >= 3 else None
    angle = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return MountTransform(angle, t.tolist(), rms, list(names or []), time.time())
