import h5py
import numpy as np
import pytest

from asyncroscopy.analysis.drift import correlation_shift, measure_drift

AREA = 512  # size of the simulated area the frames are cut from
FRAME = 128  # frame size
CORNER = 150  # where the frame is cut from the area
FEATURE_SCALE = 0.01  # Gaussian low-pass width (cycles/px)^2: features about 10 px across


def _fourier_shift(image: np.ndarray, d_row: float, d_col: float) -> np.ndarray:
    k_row = np.fft.fftfreq(image.shape[0])[:, None]
    k_col = np.fft.fftfreq(image.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(image) * np.exp(-2j * np.pi * (k_row * d_row + k_col * d_col))).real


def _frames(d_row: float, d_col: float, noise: float = 0.1, seed: int = 1):
    """Two frames cut from one random area, the second moved by (d_row, d_col).

    Cutting from a larger area (not np.roll) gives real edges, as on the microscope.
    """
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(AREA)
    k_col, k_row = np.meshgrid(k, k)
    area = np.fft.ifft2(np.fft.fft2(rng.normal(size=(AREA, AREA))) * np.exp(-(k_col**2 + k_row**2) / FEATURE_SCALE)).real
    cut = slice(CORNER, CORNER + FRAME)
    a, b = area[cut, cut], _fourier_shift(area, d_row, d_col)[cut, cut]
    return a + rng.normal(0, noise * a.std(), a.shape), b + rng.normal(0, noise * b.std(), b.shape)


@pytest.mark.parametrize(("d_row", "d_col"), [(3, -5), (-7, 2), (2.3, -1.6), (10.5, -8.25), (0, 0)])
def test_shift_is_recovered_to_a_fraction_of_a_pixel(d_row, d_col):
    shift_row, shift_col, confidence = correlation_shift(*_frames(d_row, d_col))

    assert shift_row == pytest.approx(d_row, abs=0.15)
    assert shift_col == pytest.approx(d_col, abs=0.15)
    assert confidence > 0.3  # clear match; the value itself is a rule of thumb


def test_periodic_lattice_gives_low_confidence():
    row, col = np.mgrid[0:FRAME, 0:FRAME]
    lattice = np.cos(2 * np.pi * col / 8) + np.cos(2 * np.pi * row / 8)

    _, _, confidence = correlation_shift(lattice, np.roll(lattice, (1, 2), axis=(0, 1)))

    assert confidence < 0.2


def test_a_blank_frame_is_refused_not_passed():
    with pytest.raises(ValueError, match="no structure"):
        measure_drift(np.ones((32, 32)), np.ones((32, 32)), pixel_size_m=1e-10, seconds_between=1.0, frame_time_s=1.0)


def test_drift_rate_and_pass_fail():
    first, second = _frames(0, 4, noise=0.0)  # 4 px in 10 s

    slow_scan = measure_drift(first, second, pixel_size_m=1e-10, seconds_between=10.0, frame_time_s=2.0)
    fast_scan = measure_drift(first, second, pixel_size_m=1e-10, seconds_between=10.0, frame_time_s=0.5)

    assert slow_scan.rate_m_per_s == pytest.approx(4e-11, rel=0.05)
    assert slow_scan.drift_per_frame_px == pytest.approx(0.8, rel=0.05)
    assert slow_scan.passed is False
    assert fast_scan.passed is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pixel_size_m": 0.0, "seconds_between": 1.0, "frame_time_s": 1.0}, "pixel_size_m"),
        ({"pixel_size_m": 1e-10, "seconds_between": 0.0, "frame_time_s": 1.0}, "seconds_between"),
        ({"pixel_size_m": 1e-10, "seconds_between": 1.0, "frame_time_s": -1.0}, "frame_time_s"),
    ],
)
def test_bad_inputs_are_rejected(kwargs, message):
    first, second = _frames(1, 1)
    with pytest.raises(ValueError, match=message):
        measure_drift(first, second, **kwargs)


def test_images_must_match():
    with pytest.raises(ValueError, match="same shape"):
        correlation_shift(np.zeros((8, 8)), np.zeros((8, 16)))


def test_a_stage_move_on_the_twin_is_measured_in_size_and_direction(twin_proxy, scan_proxy):
    """End to end: acquire, move the stage, acquire again, and recover the move."""
    scan_proxy.imsize = 128
    scan_proxy.dwell_time = 1e-6
    twin_proxy.set_fov(100e-9)
    move_x_m = 5e-9
    twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    try:
        with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            first = h5["image/HAADF"][()]
            pixel_size_m = h5["image/HAADF"].attrs["pixel_size_m"]
        twin_proxy.move_stage([move_x_m, 0.0, 0.0, 0.0, 0.0])
        with h5py.File(twin_proxy.acquire_scanned_image(["haadf"]), "r") as h5:
            second = h5["image/HAADF"][()]
    finally:
        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])

    result = measure_drift(first, second, pixel_size_m, seconds_between=1.0, frame_time_s=1.0)

    # The twin draws stage x along image rows: moving the stage by +x moves the
    # sample in the image by -x, so the content shifts to smaller row indices.
    assert result.shift_row_px == pytest.approx(-move_x_m / pixel_size_m, rel=0.05)
    assert abs(result.shift_col_px) < 0.3
