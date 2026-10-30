"""The twin is hard enough to be worth passing and consistent across scales (docs/designs/multiscale_maps.md, P1).

* The support film has a seeded texture visible at survey and atomic pixel sizes, and turning it on does not
  move a single particle.
* Survey and atomic drawings of the same particle agree where they meet (0.2 nm pixels).
* Atoms land at their sub-pixel positions, not at the pixel they fall in.
* The scanning detectors' gain and offset drift slowly, within their stated bounds, the same way every run.
* A small frame's dose stays on the area it hit, however large the sample.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from asyncroscopy.instruments.electron_microscope.twin import EventRng, SampleParams
from asyncroscopy.instruments.electron_microscope.twin.detector import DetectorDrift, DetectorParams
from asyncroscopy.instruments.electron_microscope.twin.dose import ANGSTROM2_PER_M2, DoseMap, DoseParams
from asyncroscopy.instruments.electron_microscope.twin.sample_model import SampleModel

SEED = 12345


def _sample(**options) -> SampleModel:
    params = SampleParams(size_xy_m=2e-6, particle_count=60, radius_mean_m=8e-9, radius_std_m=2e-9, support_thickness_m=1e-8, **options)
    return SampleModel(params, EventRng(SEED))


# ----------------------------------------------------------------------
# Support texture
# ----------------------------------------------------------------------
def test_the_texture_does_not_move_any_particle():
    plain, textured = _sample(), _sample(support_texture=0.3)
    assert plain.particle_table() == textured.particle_table()
    assert plain.support_texture(0.0, 0.0, 8, 1e-9) is None


@pytest.mark.parametrize("pixel_m", [2e-7, 2e-8, 2e-9, 2e-10])
def test_the_texture_is_visible_at_every_pixel_size(pixel_m):
    texture = _sample(support_texture=0.3).support_texture(-5e-7, -5e-7, 256, pixel_m)
    assert texture.shape == (256, 256)
    assert 0.05 < float(texture.std()) < 0.6  # structure inside one frame, not only between frames
    assert float(texture.min()) >= -1.0  # the film never gets a negative thickness


def test_the_texture_is_the_same_wherever_it_is_asked_for():
    sample = _sample(support_texture=0.3)
    whole = sample.support_texture(0.0, 0.0, 64, 1e-8)
    part = sample.support_texture(16 * 1e-8, 8 * 1e-8, 16, 1e-8)
    assert np.allclose(part, whole[8:24, 16:32], atol=1e-9)
    assert np.array_equal(whole, _sample(support_texture=0.3).support_texture(0.0, 0.0, 64, 1e-8))


# ----------------------------------------------------------------------
# Survey and atomic drawings agree at the switch
# ----------------------------------------------------------------------
def _largest_particle(sample: SampleModel) -> int:
    return int(np.argmax(sample.semi_axes[:, 0] * sample.semi_axes[:, 1]))


def test_survey_and_atomic_drawings_of_a_particle_agree_at_the_switch():
    sample = _sample(subpixel_atoms=True)
    i = _largest_particle(sample)
    reach = float(sample.semi_axes[i, :2].max()) * 1.3
    pixel, n = 0.2e-9, int(2 * reach / 0.2e-9)
    x0, y0 = sample.centers[i, 0] - reach, sample.centers[i, 1] - reach
    survey = sample.projected_weight(x0, y0, n, pixel, "survey", 1.7, {})
    atomic = sample.projected_weight(x0, y0, n, pixel, "atomic", 1.7, {})
    # Seen through a probe a few atoms wide, the two drawings are one particle of one mass.
    blur = 1e-9 / pixel
    survey, atomic = ndimage.gaussian_filter(survey, blur), ndimage.gaussian_filter(atomic, blur)
    assert atomic.sum() == pytest.approx(survey.sum(), rel=0.1)
    assert np.corrcoef(survey.ravel(), atomic.ravel())[0, 1] > 0.9


# ----------------------------------------------------------------------
# Sub-pixel atoms
# ----------------------------------------------------------------------
@pytest.mark.parametrize("subpixel", [True, False])
def test_an_atom_is_drawn_at_its_sub_pixel_position(subpixel):
    sample = _sample(subpixel_atoms=subpixel)
    i = _largest_particle(sample)
    offset = np.array([0.0123e-9, -0.0377e-9])  # x, y from the particle centre
    sample._atoms[i] = (offset[None, :], np.array([79.0]))  # one gold atom
    pixel, n = 0.1e-9, 32
    x0, y0 = sample.centers[i, 0] - n * pixel / 2, sample.centers[i, 1] - n * pixel / 2
    weight = sample.projected_weight(x0, y0, n, pixel, "atomic", 1.7, {})
    rows, cols = np.indices(weight.shape)
    drawn = np.array([(weight * cols).sum(), (weight * rows).sum()]) / weight.sum()
    # Pixel (r, c) is centred at (x0 + (c + 0.5) * pixel, y0 + (r + 0.5) * pixel).
    true = (sample.centers[i] + offset - np.array([x0, y0])) / pixel - 0.5
    error = float(np.abs(drawn - true).max())
    if subpixel:
        assert error < 0.01
    else:  # snapped to the pixel it falls in
        assert error > 0.1


# ----------------------------------------------------------------------
# Detector drift
# ----------------------------------------------------------------------
def test_detector_gain_and_offset_drift_within_bounds_and_repeat():
    base = DetectorParams(gain=2.0, offset=10.0)
    drift = DetectorDrift(0.1, 2.0, EventRng(SEED).for_event("detector_drift"))
    times = np.linspace(0.0, 4 * 3600.0, 400)
    gains = np.array([drift.at(base, t).gain for t in times])
    offsets = np.array([drift.at(base, t).offset for t in times])
    assert np.all(np.abs(gains / 2.0 - 1.0) <= 0.1 + 1e-12) and np.all(np.abs(offsets - 10.0) <= 2.0 + 1e-12)
    assert gains.max() - gains.min() > 0.05 * 2.0  # it really moves over hours
    assert abs(drift.at(base, 100.0).gain - drift.at(base, 101.0).gain) < 1e-3 * 2.0  # but slowly
    again = DetectorDrift(0.1, 2.0, EventRng(SEED).for_event("detector_drift"))
    assert again.at(base, 1234.0) == drift.at(base, 1234.0)
    assert DetectorDrift(0.0, 0.0, EventRng(SEED).for_event("detector_drift")).at(base, 999.0) == base


def test_impossible_drift_is_refused():
    with pytest.raises(ValueError):
        DetectorDrift(1.5, 0.0, EventRng(SEED).for_event("detector_drift"))


# ----------------------------------------------------------------------
# Dose stays where it landed
# ----------------------------------------------------------------------
def _frame(cx: float, cy: float, fov: float, n: int = 128) -> tuple[np.ndarray, np.ndarray, float]:
    pixel = fov / n
    coords = (np.arange(n) + 0.5) * pixel - fov / 2
    xs, ys = np.meshgrid(cx + coords, cy + coords)
    return xs, ys, pixel


def test_a_small_frame_keeps_its_dose_on_the_area_it_hit_on_a_large_sample():
    dose = DoseMap(DoseParams(enabled=True, cell_m=20e-6 / 200), (-10e-6, 10e-6, -10e-6, 10e-6), margin_m=5e-6)
    xs, ys, pixel = _frame(1.234e-6, -2.5e-6, 50e-9)
    electrons = 300.0
    dose.add_frame(xs, ys, electrons, pixel)
    expected = electrons / (pixel * pixel * ANGSTROM2_PER_M2)
    inside = dose.dose_at(xs[16:-16, 16:-16], ys[16:-16, 16:-16])
    assert inside.mean() == pytest.approx(expected, rel=0.1)
    assert float(dose.dose_at(1.234e-6 + 200e-9, -2.5e-6)) == 0.0  # 200 nm away: untouched
    assert dose.total_electrons() == pytest.approx(electrons * xs.size, rel=1e-6)


def test_a_survey_frame_and_a_fine_frame_add_up_where_they_overlap():
    dose = DoseMap(DoseParams(enabled=True, cell_m=1e-7), (-10e-6, 10e-6, -10e-6, 10e-6))
    survey = _frame(0.0, 0.0, 10e-6)
    fine = _frame(0.0, 0.0, 50e-9)
    dose.add_frame(*survey[:2], 100.0, survey[2])
    dose.add_frame(*fine[:2], 100.0, fine[2])
    each = [100.0 / (p * p * ANGSTROM2_PER_M2) for p in (survey[2], fine[2])]
    assert float(dose.dose_at(0.0, 0.0)) == pytest.approx(sum(each), rel=0.1)


def test_dose_map_memory_grows_with_frames_not_with_the_sample():
    small = DoseMap(DoseParams(enabled=True, cell_m=1e-8), (-1e-6, 1e-6, -1e-6, 1e-6))
    huge = DoseMap(DoseParams(enabled=True, cell_m=1e-5), (-1e-3, 1e-3, -1e-3, 1e-3))
    for dose in (small, huge):
        xs, ys, pixel = _frame(0.0, 0.0, 50e-9)
        dose.add_frame(xs, ys, 10.0, pixel)
    # One 50 nm frame fills about (8 + 1)^2 to (16 + 1)^2 cells of its own size, on a 2 um or a 2 mm sample.
    assert huge.summary()["exposed_cells"] <= 300 and small.summary()["exposed_cells"] <= 300


def test_a_parked_beam_counts_over_the_spot_spread():
    dose = DoseMap(DoseParams(enabled=True, cell_m=1e-7, spot_spread_m=1e-8), (-1e-6, 1e-6, -1e-6, 1e-6))
    dose.add_point(0.0, 0.0, 1e6, 1e-20)
    cell = 1e-7 / 2**3  # the finest cell at least 10 nm wide: 12.5 nm
    assert float(dose.dose_at(1e-9, 1e-9)) == pytest.approx(1e6 / (cell * cell * ANGSTROM2_PER_M2), rel=1e-9)


def test_overlapping_survey_frames_see_the_same_film_wherever_their_grids_start():
    sample = _sample(support_texture=0.3)
    pixel, n = 2e-8, 128
    first = sample.support_texture(0.0, 0.0, n, pixel)
    second = sample.support_texture(0.37 * pixel, 0.61 * pixel, n, pixel)  # a grid starting a fraction of a pixel later
    rows, cols = np.mgrid[0 : n - 1, 0 : n - 1].astype(float)
    on_second = ndimage.map_coordinates(first, [rows + 0.61, cols + 0.37], order=1)
    assert np.corrcoef(on_second.ravel(), second[:-1, :-1].ravel())[0, 1] > 0.95


def test_the_line_jitter_in_ground_truth_gives_back_every_line_offset():
    from asyncroscopy.instruments.electron_microscope.twin.scan_distortion import ScanParams, line_jitter_m, pixel_sample_coordinates

    rng = EventRng(SEED)
    args = (64, 1e-7, (0.0, 0.0, 1.0, 1.0), 1e-6)
    origin = lambda _t: np.zeros(2)  # noqa: E731
    truth: dict = {}
    jittered, _ = pixel_sample_coordinates(*args, ScanParams(line_jitter_px=0.5), origin, 0.0, rng.for_event("scan", 7), truth=truth)
    straight, _ = pixel_sample_coordinates(*args, ScanParams(), origin, 0.0, rng.for_event("scan", 7))
    again = line_jitter_m(rng.for_event("scan", 7), truth["line_jitter"]["rows"], truth["line_jitter"]["std_m"])
    assert np.allclose(jittered[:, 0] - straight[:, 0], again, atol=1e-18)


def test_recording_a_frames_dose_stays_fast():
    import time

    dose = DoseMap(DoseParams(enabled=True, cell_m=1e-7), (-5e-6, 5e-6, -5e-6, 5e-6))
    xs, ys, pixel = _frame(0.0, 0.0, 2e-6, n=512)
    started = time.perf_counter()
    dose.add_frame(xs, ys, 300.0, pixel)
    assert time.perf_counter() - started < 0.08
    assert dose.total_electrons() == pytest.approx(300.0 * xs.size, rel=1e-6)
