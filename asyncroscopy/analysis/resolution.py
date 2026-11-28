"""Finest detail an image shows, from its power spectrum (FFT resolution).

For a HAADF image of a crystal, each lattice spacing the probe resolves gives
spots in the power spectrum. The finest spacing whose spots stand clearly above
the noise is the resolution this image shows.

How "clearly above the noise" is decided:
- Shot noise is white, so the outer part of the spectrum holds noise only. Its
  mean power is measured in NOISE_RING.
- For noise, each power value follows an exponential distribution: the chance
  that one value exceeds t times the mean is exp(-t). The threshold t is set so
  that, over all the frequencies searched, the expected number of noise values
  above it is ``false_alarms``.

Limits:
- This measures what the image shows. On a crystal that depends on the zone
  axis; an amorphous area shows no spots at all.
- The noise ring must hold noise only. If lattice spots reach it, use smaller pixels.
- The search stops where the noise ring starts, so the finest result possible is
  2 / NOISE_RING[0] pixels (about 2.35 px), not the 2 px Nyquist limit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from asyncroscopy.analysis.drift import hann_window

NYQUIST = 0.5  # cycles per pixel
# Outer ring used to measure the noise, as fractions of Nyquist. It starts far
# out (spots rarely reach it) and stops short of 1.0 (the corners of the
# spectrum lie beyond Nyquist in one direction only).
NOISE_RING = (0.85, 0.98)
# Frequencies below this fraction of Nyquist are not searched: there the window
# and the image's overall brightness changes leak into the spectrum.
MIN_FREQUENCY = 0.05
# A Hann window spreads one spot over about +-2 frequency rings, plus 1 for
# rounding to rings. The reported spot is the brightest pixel this many rings
# inside the finest ring that shows signal.
SPOT_SPREAD_RINGS = 3
# Half-width, in rings, of the band checked around an expected spacing.
SPACING_BAND_RINGS = 1.5
METERS_TO_ANGSTROM = 1e10


@dataclass
class ResolutionResult:
    resolution_angstrom: float | None
    nyquist_angstrom: float
    resolved_spacings: dict[str, bool]
    target_angstrom: float | None
    passed: bool | None

    def as_dict(self) -> dict:
        return asdict(self)


def _spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Power spectrum (zero frequency centred), each pixel's frequency in cycles/px, and ring count."""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Need a 2D image, got shape {a.shape}")
    power = np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * hann_window(a.shape)))) ** 2
    rows, cols = power.shape
    k = np.hypot(((np.arange(rows) - rows // 2) / rows)[:, None], ((np.arange(cols) - cols // 2) / cols)[None, :])
    return power, k, min(rows, cols)


def _threshold(noise_mean: float, n_values: int, false_alarms: float) -> float:
    """Power above which a value is signal. A real image's spectrum is symmetric,
    so only half of the values searched are independent."""
    return noise_mean * math.log(max(n_values / 2, 1) / false_alarms)


def finest_signal_frequency(image: np.ndarray, false_alarms: float = 0.01) -> float | None:
    """Frequency (cycles per pixel) of the finest spot above the noise, or None."""
    power, k, size = _spectrum(image)
    noise = power[(k >= NOISE_RING[0] * NYQUIST) & (k <= NOISE_RING[1] * NYQUIST)]
    if noise.size == 0 or noise.mean() <= 0:
        return None
    ring = np.round(k * size).astype(int)
    searched = (ring >= MIN_FREQUENCY * NYQUIST * size) & (ring < NOISE_RING[0] * NYQUIST * size)
    above = searched & (power > _threshold(noise.mean(), int(searched.sum()), false_alarms))
    if not above.any():
        return None
    finest_ring = int(ring[above].max())
    near = searched & (ring >= finest_ring - SPOT_SPREAD_RINGS)
    brightest = np.unravel_index(int(np.argmax(np.where(near, power, -np.inf))), power.shape)
    return float(k[brightest])


def spacing_resolved(image: np.ndarray, spacing_px: float, false_alarms: float = 0.01) -> bool:
    """True if spots of a lattice spacing (in pixels) stand above the noise."""
    power, k, size = _spectrum(image)
    k_spacing = 1.0 / spacing_px
    if not MIN_FREQUENCY * NYQUIST <= k_spacing < NOISE_RING[0] * NYQUIST:
        return False  # outside the range this method can judge
    noise = power[(k >= NOISE_RING[0] * NYQUIST) & (k <= NOISE_RING[1] * NYQUIST)]
    band = np.abs(k - k_spacing) <= SPACING_BAND_RINGS / size
    return bool(power[band].max() > _threshold(noise.mean(), int(band.sum()), false_alarms))


def fft_resolution(
    image: np.ndarray,
    pixel_size_m: float,
    expected_spacings_m: dict[str, float] | None = None,
    target_m: float | None = None,
) -> ResolutionResult:
    """The finest spacing with spots above the noise, and which expected spacings show.

    ``expected_spacings_m`` names lattice spacings to look for, e.g. {"Au(111)": 2.355e-10}.
    ``target_m`` is the resolution to reach; ``passed`` is None without it.
    """
    if not pixel_size_m > 0:
        raise ValueError("pixel_size_m must be > 0")
    finest_k = finest_signal_frequency(image)
    resolution_m = None if finest_k is None else pixel_size_m / finest_k
    return ResolutionResult(
        resolution_angstrom=None if resolution_m is None else resolution_m * METERS_TO_ANGSTROM,
        nyquist_angstrom=2 * pixel_size_m * METERS_TO_ANGSTROM,
        resolved_spacings={
            name: spacing_resolved(image, spacing / pixel_size_m) for name, spacing in (expected_spacings_m or {}).items()
        },
        target_angstrom=None if target_m is None else target_m * METERS_TO_ANGSTROM,
        passed=None if target_m is None else resolution_m is not None and resolution_m <= target_m,
    )


def nyquist_fraction(image: np.ndarray) -> float:
    """Finest detail shown, as a fraction of the Nyquist limit (0 = none, at most about 0.85).

    Optimizers use this to compare images taken at the same settings; it needs no pixel size.
    """
    finest_k = finest_signal_frequency(image)
    return 0.0 if finest_k is None else finest_k / NYQUIST
