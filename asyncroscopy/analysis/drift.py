"""Stage and specimen drift from two images of the same area.

Method: cross-correlate the two frames with FFTs after a Hann window removes
their edges. The correlation peak sits at the shift between the frames, and a
parabola through the peak and its neighbours gives sub-pixel precision.
(Fully normalized phase correlation was tried and rejected: on smooth images it
amplifies noise at frequencies with no signal and locks onto zero shift.)

Typical use: after a stage move, take two short frames a few seconds apart and
wait until the drift is small enough before a long acquisition.

Limits:
- On a periodic lattice the correlation has a peak at every lattice period, so a
  shift of more than half a period can be wrong. ``confidence`` then drops.
  Measure drift at a field of view that shows non-periodic features.
- Drift during one frame distorts that frame instead of shifting it. This
  measures only the shift between frames.
- Shifts of more than about a quarter of the frame are unreliable, because the
  window leaves little overlap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Default limit: drift during one frame under half a pixel does not blur the image.
DEFAULT_MAX_DRIFT_PER_FRAME_PX = 0.5


def hann_window(shape: tuple[int, int]) -> np.ndarray:
    """2D Hann window: tapers an image to zero at its edges before an FFT."""
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


@dataclass
class DriftResult:
    shift_row_px: float
    shift_col_px: float
    shift_m: float
    rate_m_per_s: float
    drift_per_frame_px: float
    confidence: float
    max_drift_per_frame_px: float
    passed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _parabola_offset(before: float, peak: float, after: float) -> float:
    """Sub-pixel position of a peak from the samples around it.

    The highest sample is within half a pixel of the true peak, so the result
    is clipped to +-0.5.
    """
    curvature = before - 2.0 * peak + after
    if curvature == 0.0:
        return 0.0
    return float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))


def correlation_shift(reference: np.ndarray, moved: np.ndarray) -> tuple[float, float, float]:
    """Return (shift_row, shift_col, confidence) of ``moved`` relative to ``reference``.

    A positive shift means the content of ``moved`` sits at larger indices.
    ``confidence`` is 1 - (second highest peak / highest peak): near 0 when the
    match is ambiguous (for example a periodic lattice), near 1 when it is clear.
    It is a rule of thumb for ranking measurements, not a probability.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(moved, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Need two 2D images of the same shape, got {a.shape} and {b.shape}")

    window = hann_window(a.shape)
    surface = np.fft.ifft2(np.conj(np.fft.fft2((a - a.mean()) * window)) * np.fft.fft2((b - b.mean()) * window)).real

    rows, cols = surface.shape
    row, col = np.unravel_index(int(np.argmax(surface)), surface.shape)
    peak = surface[row, col]
    if peak <= 0:
        raise ValueError("The frames have no structure to correlate (blank or saturated?)")

    d_row = _parabola_offset(surface[(row - 1) % rows, col], peak, surface[(row + 1) % rows, col])
    d_col = _parabola_offset(surface[row, (col - 1) % cols], peak, surface[row, (col + 1) % cols])

    # The FFT wraps around: indices past the middle are negative shifts.
    shift_row = (row if row <= rows // 2 else row - rows) + d_row
    shift_col = (col if col <= cols // 2 else col - cols) + d_col

    # The second peak is the highest other local maximum. The main peak is broad,
    # so its 3 x 3 neighbourhood is left out; its shoulders are not separate peaks.
    is_local_max = np.ones(surface.shape, dtype=bool)
    for d_r in (-1, 0, 1):
        for d_c in (-1, 0, 1):
            if d_r or d_c:
                is_local_max &= surface > np.roll(surface, (d_r, d_c), axis=(0, 1))
    for d_r in (-1, 0, 1):
        for d_c in (-1, 0, 1):
            is_local_max[(row + d_r) % rows, (col + d_c) % cols] = False
    second = float(surface[is_local_max].max()) if is_local_max.any() else 0.0
    confidence = float(np.clip(1.0 - max(second, 0.0) / peak, 0.0, 1.0))
    return float(shift_row), float(shift_col), confidence


def measure_drift(
    reference: np.ndarray,
    moved: np.ndarray,
    pixel_size_m: float,
    seconds_between: float,
    frame_time_s: float,
    max_drift_per_frame_px: float = DEFAULT_MAX_DRIFT_PER_FRAME_PX,
) -> DriftResult:
    """Measure drift between two frames and judge it against a per-frame limit.

    ``seconds_between`` is the time between the starts of the two frames.
    ``passed`` is True when the drift during one frame of ``frame_time_s`` stays
    below ``max_drift_per_frame_px``.
    """
    if not pixel_size_m > 0:
        raise ValueError("pixel_size_m must be > 0")
    if not seconds_between > 0:
        raise ValueError("seconds_between must be > 0; the two frames need different start times")
    if not frame_time_s >= 0:
        raise ValueError("frame_time_s must be >= 0")
    shift_row, shift_col, confidence = correlation_shift(reference, moved)
    shift_px = float(np.hypot(shift_row, shift_col))
    rate_px_per_s = shift_px / seconds_between
    drift_per_frame_px = rate_px_per_s * frame_time_s
    return DriftResult(
        shift_row_px=shift_row,
        shift_col_px=shift_col,
        shift_m=shift_px * pixel_size_m,
        rate_m_per_s=rate_px_per_s * pixel_size_m,
        drift_per_frame_px=drift_per_frame_px,
        confidence=confidence,
        max_drift_per_frame_px=max_drift_per_frame_px,
        passed=drift_per_frame_px <= max_drift_per_frame_px,
    )
