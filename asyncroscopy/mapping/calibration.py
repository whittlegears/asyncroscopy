"""Which way each stage axis moves the image, measured before a grid scan.

A stage move of (dx, dy) metres shifts the image content by some (rows, cols) in
pixels. The 2 x 2 matrix between them is different on every microscope: the twin
maps stage x to image rows, a Thermo scope adds the scan rotation, a mirrored
axis flips a sign. A grid planned with the wrong matrix puts tile (0, 1) above
tile (0, 0) instead of beside it, and the stitcher then compares the wrong
edges. So the grid scan measures the matrix first, with one small move per axis.

Method: image the start position, move each stage axis by a known step, image
again, and cross-correlate (asyncroscopy.analysis.drift.correlation_shift). The
two shifts are the columns of the matrix.

The measurement is checked before it is used:
- each shift must have the length the pixel size predicts (within ``tolerance``),
- the two axes must be close to perpendicular.
On a periodic lattice the correlation can lock one lattice period off and
return a shift of the wrong length; that fails the first check, and the scan
stops with the numbers instead of building a wrong map. Calibrate at a field of
view that shows non-periodic features, or pass a known matrix.

No Tango and no opencv here: the caller supplies ``acquire`` and ``move``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

from asyncroscopy.analysis.drift import correlation_shift

DEFAULT_STEP_FRACTION = 0.2  # of the field of view; the correlation needs most of the frame to overlap
DEFAULT_TOLERANCE = 0.15  # allowed relative error in shift length
MAX_AXIS_COSINE = 0.2  # |cos| of the angle between the axes; 0.2 allows about 78 to 102 degrees


class CalibrationError(ValueError):
    """The measured stage-to-image matrix failed a check."""


@dataclass
class StageCalibration:
    """Image content shift per metre of stage travel.

    ``px_per_m[i][j]`` is the shift along image axis i (0 = rows, 1 = columns)
    for one metre of travel on stage axis j (0 = x, 1 = y). A positive shift
    means the content moves to larger row/column indices.
    """

    px_per_m: list[list[float]]
    pixel_size_m: float
    step_m: float
    confidence: list[float]
    length_error: list[float]
    axis_cosine: float
    measured: bool = True
    calibration_fov_m: float | None = None  # set when the test moves ran at another field of view

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_matrix(cls, px_per_m: Sequence[Sequence[float]], pixel_size_m: float) -> "StageCalibration":
        """A known matrix, for example from an earlier scan's manifest."""
        matrix = np.asarray(px_per_m, dtype=float)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) == 0:
            raise CalibrationError(f"stage_to_image_px_per_m must be an invertible 2 x 2 matrix, got {px_per_m!r}")
        return cls(matrix.tolist(), float(pixel_size_m), 0.0, [1.0, 1.0], [0.0, 0.0], _axis_cosine(matrix), measured=False)

    @classmethod
    def nominal(cls, pixel_size_m: float, flip_x: bool = False, flip_y: bool = False) -> "StageCalibration":
        """The old assumption: stage x moves the view right, stage y moves it down.

        Moving the view right moves the content left, hence the minus signs.
        """
        scale = 1.0 / float(pixel_size_m)
        sx = -1.0 if flip_x else 1.0
        sy = -1.0 if flip_y else 1.0
        return cls.from_matrix([[0.0, -sy * scale], [-sx * scale, 0.0]], pixel_size_m)

    def stage_offset(self, view_row_px: float, view_col_px: float) -> tuple[float, float]:
        """Stage (dx, dy) in metres that moves the view by (rows, cols) pixels.

        Moving the view down by n rows moves the content up by n rows.
        """
        content = -np.array([view_row_px, view_col_px], dtype=float)
        dx, dy = np.linalg.solve(np.asarray(self.px_per_m, dtype=float), content)
        return float(dx), float(dy)

    def image_shift(self, dx_m: float, dy_m: float) -> tuple[float, float]:
        """Content shift (rows, cols) in pixels that a stage move of (dx, dy) produces."""
        rows, cols = np.asarray(self.px_per_m, dtype=float) @ np.array([dx_m, dy_m], dtype=float)
        return float(rows), float(cols)


def _axis_cosine(matrix: np.ndarray) -> float:
    a, b = matrix[:, 0], matrix[:, 1]
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(abs(a @ b) / norm) if norm else 1.0


def measure_stage_to_image(
    acquire: Callable[[], np.ndarray],
    move: Callable[[Sequence[float]], None],
    start: Sequence[float],
    pixel_size_m: float,
    step_m: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> StageCalibration:
    """Measure the matrix with one move per stage axis, then return to ``start``.

    ``acquire()`` returns one 2D image at the current position; ``move(position)``
    moves to a full stage vector [x, y, z, alpha, beta]. Raises CalibrationError
    when a check fails; the stage is back at ``start`` either way.
    """
    if not pixel_size_m > 0 or not step_m > 0:
        raise ValueError("pixel_size_m and step_m must be > 0")
    start = [float(v) for v in start]
    expected_px = step_m / pixel_size_m
    columns, confidence, length_error = [], [], []
    try:
        move(start)
        reference = np.asarray(acquire(), dtype=float)
        for axis in (0, 1):
            target = list(start)
            target[axis] += step_m
            move(target)
            shift_row, shift_col, conf = correlation_shift(reference, np.asarray(acquire(), dtype=float))
            columns.append([shift_row / step_m, shift_col / step_m])
            confidence.append(conf)
            length_error.append(float(np.hypot(shift_row, shift_col) / expected_px - 1.0))
    finally:
        move(start)

    matrix = np.array(columns, dtype=float).T
    calibration = StageCalibration(matrix.tolist(), float(pixel_size_m), float(step_m), confidence, length_error, _axis_cosine(matrix))

    problems = []
    for name, error in zip("xy", length_error):
        if abs(error) > tolerance:
            problems.append(
                f"stage {name} moved {step_m * 1e9:.3g} nm ({expected_px:.1f} px) but the image moved "
                f"{(1 + error) * expected_px:.1f} px"
            )
    if calibration.axis_cosine > MAX_AXIS_COSINE:
        angle = np.degrees(np.arccos(min(1.0, calibration.axis_cosine)))
        problems.append(f"the stage axes appear {angle:.0f} degrees apart in the image, not 90")
    if problems:
        raise CalibrationError(
            "Stage-to-image calibration failed: "
            + "; ".join(problems)
            + ". The image may be periodic (a lattice) or featureless at this field of view. "
            "Calibrate at a lower magnification, or pass stage_to_image_px_per_m from an earlier scan."
        )
    return calibration


def plan_calibrated_grid(
    rows: int,
    cols: int,
    step_px: float,
    calibration: StageCalibration,
    snake: bool = True,
) -> list[dict]:
    """Acquisition order for a rows x cols grid whose neighbours are ``step_px`` apart in the image.

    Tile (r, c) sits r * step_px rows below and c * step_px columns right of tile
    (0, 0), whatever the stage axes are. Each entry is
    ``{"row", "col", "offset_m": [dx, dy]}`` relative to the start position.
    """
    order = []
    for r in range(rows):
        columns = range(cols) if (not snake or r % 2 == 0) else range(cols - 1, -1, -1)
        for c in columns:
            dx, dy = calibration.stage_offset(r * step_px, c * step_px)
            order.append({"row": r, "col": c, "offset_m": [dx, dy]})
    return order
