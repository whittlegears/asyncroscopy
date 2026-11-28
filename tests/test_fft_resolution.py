import copy
from types import SimpleNamespace

import numpy as np
import pytest

from asyncroscopy.analysis.resolution import fft_resolution, nyquist_fraction
from asyncroscopy.instruments.electron_microscope.hardware.corrector_digital_twin import DEFAULT_ABERRATIONS
from asyncroscopy.instruments.electron_microscope.polycrystalline_gold_digital_twin import (
    PolycrystallineGoldDigitalTwin as Gold,
)

PIXEL = 0.5e-10  # 0.5 Angstrom
SIZE = 256


def _lattice(periods_px, counts=200.0, blur_px=0.0, seed=0):
    """Square lattice with the given periods in pixels, imaged with shot noise."""
    row, col = np.mgrid[0:SIZE, 0:SIZE]
    image = sum(np.cos(2 * np.pi * col / p) + np.cos(2 * np.pi * row / p) for p in periods_px)
    image = image - image.min()
    if blur_px:  # Gaussian blur of blur_px standard deviation
        k = np.fft.fftfreq(SIZE)
        k_col, k_row = np.meshgrid(k, k)
        image = np.fft.ifft2(np.fft.fft2(image) * np.exp(-2 * (np.pi * blur_px) ** 2 * (k_col**2 + k_row**2))).real
        image = np.clip(image, 0, None)
    return np.random.default_rng(seed).poisson(image / image.mean() * counts).astype(float)


def _noise_only():
    return np.random.default_rng(3).poisson(100.0, (SIZE, SIZE)).astype(float)


def test_finest_lattice_spacing_is_found():
    result = fft_resolution(_lattice([16, 4]), PIXEL)

    assert result.resolution_angstrom == pytest.approx(4 * 0.5, rel=0.05)
    assert result.nyquist_angstrom == pytest.approx(1.0)


def test_blur_hides_the_fine_spacing():
    result = fft_resolution(
        _lattice([16, 4], blur_px=2.5),
        PIXEL,
        expected_spacings_m={"coarse": 16 * PIXEL, "fine": 4 * PIXEL},
        target_m=3e-10,
    )

    assert result.resolution_angstrom == pytest.approx(16 * 0.5, rel=0.05)
    assert result.resolved_spacings == {"coarse": True, "fine": False}
    assert result.passed is False


def test_pure_noise_shows_nothing():
    result = fft_resolution(_noise_only(), PIXEL, target_m=2e-10)

    assert result.resolution_angstrom is None
    assert result.passed is False
    assert nyquist_fraction(_noise_only()) == 0.0


def test_spacings_outside_the_searched_range_are_not_claimed():
    result = fft_resolution(_lattice([16, 4]), PIXEL, expected_spacings_m={"too fine": 2.1 * PIXEL, "too coarse": 100 * PIXEL})

    assert result.resolved_spacings == {"too fine": False, "too coarse": False}


def test_nyquist_fraction():
    assert nyquist_fraction(_lattice([16, 4])) == pytest.approx(2 / 4, rel=0.05)


def test_bad_input_is_rejected():
    with pytest.raises(ValueError, match="pixel_size_m"):
        fft_resolution(np.zeros((8, 8)), 0.0)
    with pytest.raises(ValueError, match="2D"):
        fft_resolution(np.zeros(8), 1e-10)


def _render_gold(c1_m: float) -> tuple[np.ndarray, float]:
    """A 256 px gold-twin image over 4 nm at defocus c1_m, without starting Tango."""
    fov_m, imsize = 4e-9, 256
    sample = SimpleNamespace(
        volume_size_xy_nm=6.0, volume_thickness_nm=2.0, grain_size_angstrom=40.0, empty_grain_fraction=0.0,
        lattice_constant_angstrom=4.08, zone_axis_max_index=1, zone_axis_max_deviation_deg=0.0,
        acceleration_voltage_ev=200_000.0, convergence_angle_mrad=30.0, _fov=fov_m, _stage_position=np.zeros(5),
        beam_current_pa=100.0, blur_noise_level=0.0, sample_seed=1,
    )
    aberrations = copy.deepcopy(DEFAULT_ABERRATIONS)
    aberrations["C1"] = [c1_m]
    sample._sync_stage_from_proxy = lambda: None
    sample._corrector_coefficients = lambda: aberrations
    sample._noise_seed = lambda seed: seed
    sample._raw_sample_potential = lambda n: Gold._raw_sample_potential(sample, n)
    Gold._generate_sample(sample, seed=41)
    return Gold._render_stem_image(sample, imsize, 4e-6), fov_m / imsize


def test_defocus_makes_the_gold_twin_resolution_worse():
    """Physics check: 20 nm of defocus must coarsen the finest detail shown (by about 50% here)."""
    focused, pixel = _render_gold(0.0)
    defocused, _ = _render_gold(20e-9)

    assert fft_resolution(defocused, pixel).resolution_angstrom > 1.2 * fft_resolution(focused, pixel).resolution_angstrom
