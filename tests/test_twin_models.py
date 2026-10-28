"""Tests for the digital twin's models (no Tango needed)."""

import numpy as np
import pytest

from asyncroscopy.instruments.electron_microscope.twin import (
    DoseMap,
    EdsParams,
    EventRng,
    SampleModel,
    SampleParams,
    ScanParams,
    SimClock,
    StageModel,
    StageParams,
    pixel_sample_coordinates,
    point_spectrum,
)
from asyncroscopy.instruments.electron_microscope.twin.detector import (
    DetectorParams,
    count,
)
from asyncroscopy.instruments.electron_microscope.twin.dose import DoseParams
from asyncroscopy.instruments.electron_microscope.twin.optics import aberrations_nm
from asyncroscopy.instruments.electron_microscope.twin.render import (
    Renderer,
    RenderParams,
)


def survey_sample(seed=7):
    return SampleModel(
        SampleParams(
            size_xy_m=1e-6,
            particle_count=150,
            radius_mean_m=8e-9,
            radius_std_m=3e-9,
            aspect_ratio=0.6,
            elements=("Au", "Fe"),
            support_thickness_m=10e-9,
        ),
        EventRng(seed),
    )


def render(sample, fov, origin, imsize=128, defocus=0.0, region=(0, 0, 1, 1)):
    xs, ys = pixel_sample_coordinates(
        imsize, fov, region, 1e-6, ScanParams(), lambda t: np.asarray(origin, float), 0.0, EventRng(1).for_event("scan")
    )
    return Renderer(sample, RenderParams()).render(xs, ys, fov / imsize, "HAADF", aberrations_nm(None, 300e3, defocus), 100.0)


def sharpness(image):
    gy, gx = np.gradient(image.astype(float))
    return float(np.mean(gx**2 + gy**2))


class TestDeterminism:
    def test_same_seed_same_sample(self):
        a, b = survey_sample(3), survey_sample(3)
        assert np.array_equal(a.centers, b.centers)
        assert a.elements == b.elements

    def test_different_seed_different_sample(self):
        assert not np.array_equal(survey_sample(3).centers, survey_sample(4).centers)

    def test_event_streams_are_independent_and_repeatable(self):
        rng = EventRng(11)
        assert rng.for_event("stage", 1).normal() == rng.for_event("stage", 1).normal()
        assert rng.for_event("stage", 1).normal() != rng.for_event("stage", 2).normal()
        assert rng.for_event("stage", 1).normal() != rng.for_event("eds", 1).normal()

    def test_particles_do_not_overlap(self):
        s = survey_sample()
        d = np.hypot(*(s.centers[:, None, :] - s.centers[None, :, :]).transpose(2, 0, 1))
        r = s.semi_axes[:, 0]
        np.fill_diagonal(d, np.inf)
        assert np.all(d >= (r[:, None] + r[None, :]) - 1e-12)


class TestStage:
    def make(self, **kw):
        return StageModel(StageParams(**kw), EventRng(5), SimClock(0.0))

    def test_ideal_stage_agrees_everywhere(self):
        stage = self.make()
        stage.move([1e-7, -2e-7, 0, 0, 0])
        assert np.allclose(stage.true, stage.commanded)
        assert np.allclose(stage.reported, stage.commanded)

    def test_reported_differs_from_true_with_noise(self):
        stage = self.make(noise_std_m=150e-9, reported_noise_std_m=10e-9)
        stage.move([1e-6, 0, 0, 0, 0])
        assert abs(stage.true[0] - stage.reported[0]) > 1e-9
        assert abs(stage.reported[0] - 1e-6) < 6 * 10e-9

    def test_backlash_only_on_reversal(self):
        stage = self.make(backlash_m=(200e-9, 0.0))
        stage.move([1e-6, 0, 0, 0, 0])
        stage.move([2e-6, 0, 0, 0, 0])
        assert stage.true[0] == pytest.approx(2e-6)
        stage.move([1e-6, 0, 0, 0, 0])
        assert stage.true[0] == pytest.approx(1e-6 + 200e-9)

    def test_moves_take_time(self):
        stage = self.make(speed_m_s=1e-6, move_overhead_s=0.5)
        stage.move([2e-6, 0, 0, 0, 0])
        assert stage.clock.now() == pytest.approx(2.5)

    def test_drift_grows_with_time_and_is_repeatable(self):
        a, b = self.make(drift_m_per_min=1e-9, drift_random_walk_m=0.2e-9), self.make(drift_m_per_min=1e-9, drift_random_walk_m=0.2e-9)
        assert np.linalg.norm(a.drift_offset_at(600)) > np.linalg.norm(a.drift_offset_at(60))
        b.drift_offset_at(300)  # sampling more often must not change the walk
        assert np.allclose(a.drift_offset_at(900), b.drift_offset_at(900))

    def test_clock_does_not_sleep_at_zero_scale(self):
        import time

        clock = SimClock(0.0)
        start = time.perf_counter()
        clock.advance(3600)
        assert time.perf_counter() - start < 0.1
        assert clock.now() == 3600


class TestImaging:
    def test_particle_right_of_centre_appears_in_right_half(self):
        s = survey_sample()
        i = int(np.argmax(s.semi_axes[:, 0]))
        image = render(s, 200e-9, (s.centers[i, 0] - 50e-9, s.centers[i, 1]))
        row, col = np.unravel_index(np.argmax(image), image.shape)
        assert col > image.shape[1] // 2
        assert abs(row - image.shape[0] // 2) < image.shape[0] // 6

    def test_overlapping_tiles_agree_in_brightness(self):
        s = survey_sample()
        fov = 200e-9
        a = render(s, fov, (0.0, 0.0))
        b = render(s, fov, (fov / 2, 0.0))
        overlap_a, overlap_b = a[:, 64:], b[:, :64]
        assert overlap_a.mean() == pytest.approx(overlap_b.mean(), rel=0.02)

    def test_defocus_blurs_atomic_image(self):
        s = survey_sample()
        i = int(np.argmax(s.semi_axes[:, 0]))
        focused = render(s, 20e-9, s.centers[i])
        blurred = render(s, 20e-9, s.centers[i], defocus=20e-9)
        assert sharpness(blurred) < 0.8 * sharpness(focused)

    def test_large_defocus_blurs_survey_image(self):
        s = survey_sample()
        i = int(np.argmax(s.semi_axes[:, 0]))
        focused = render(s, 100e-9, s.centers[i])
        blurred = render(s, 100e-9, s.centers[i], defocus=500e-9)
        assert sharpness(blurred) < 0.8 * sharpness(focused)

    def test_atomic_level_resolves_lattice(self):
        s = survey_sample()
        i = int(np.argmax(s.semi_axes[:, 0]))
        image = render(s, 5e-9, s.centers[i], imsize=128)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean())))
        centre = spectrum.shape[0] // 2
        spectrum[centre - 3 : centre + 4, centre - 3 : centre + 4] = 0
        assert spectrum.max() > 20 * np.median(spectrum)

    def test_scan_region_shape_and_bad_region(self):
        xs, _ys = pixel_sample_coordinates(64, 1e-7, (0.25, 0.0, 0.75, 0.5), 1e-6, ScanParams(), lambda t: np.zeros(2), 0.0, EventRng(1).for_event("scan"))
        assert xs.shape == (32, 32)
        with pytest.raises(ValueError):
            pixel_sample_coordinates(64, 1e-7, (0.5, 0.0, 0.25, 1.0), 1e-6, ScanParams(), lambda t: np.zeros(2), 0.0, EventRng(1).for_event("scan"))

    def test_counts_are_raw_poisson(self):
        fraction = np.full((200, 200), 0.1)
        counts = count(fraction, 1e-6, DetectorParams(beam_current_pa=50.0), EventRng(2).for_event("d"))
        expected = 0.1 * 50e-12 * 1e-6 / 1.602176634e-19
        assert counts.dtype == np.uint16
        assert counts.mean() == pytest.approx(expected, rel=0.02)
        assert counts.var() == pytest.approx(expected, rel=0.1)

    def test_unknown_detector_rejected(self):
        s = survey_sample()
        xs, ys = pixel_sample_coordinates(16, 1e-7, (0, 0, 1, 1), 1e-6, ScanParams(), lambda t: np.zeros(2), 0.0, EventRng(1).for_event("scan"))
        with pytest.raises(ValueError, match="Unknown scanning detector"):
            Renderer(s, RenderParams()).render(xs, ys, 1e-7 / 16, "SE", aberrations_nm(None, 300e3, 0.0), 0.0)


class TestEds:
    def test_spectrum_has_lines_of_the_element_under_beam(self):
        energies, counts, live = point_spectrum({"Au": 9.0, "C": 10.0}, 50.0, 10.0, EdsParams(), EventRng(3).for_event("eds"))
        assert energies.shape == counts.shape == (2048,)
        au_la = np.argmin(np.abs(energies - 9713))
        background = np.median(counts[(energies > 4000) & (energies < 6000)])
        assert counts[au_la - 2 : au_la + 3].max() > 5 * max(background, 1.0)
        assert 0 < live <= 10.0

    def test_no_material_means_no_lines(self):
        _energies, counts, _live = point_spectrum({}, 50.0, 1.0, EdsParams(), EventRng(3).for_event("eds"))
        assert counts.sum() == 0

    def test_dead_time_reduces_live_time_at_high_rate(self):
        low = point_spectrum({"Au": 9.0}, 10.0, 1.0, EdsParams(), EventRng(3).for_event("eds"))[2]
        high = point_spectrum({"Au": 9.0}, 5000.0, 1.0, EdsParams(), EventRng(3).for_event("eds"))[2]
        assert high < low


class TestDose:
    def test_contamination_builds_and_saturates(self):
        dose = DoseMap(DoseParams(enabled=True, cell_m=10e-9, contamination_angstrom_max=50.0, contamination_dose_e_per_a2=10.0), (-1e-7, 1e-7, -1e-7, 1e-7))
        xs, ys = np.meshgrid(np.linspace(-2e-8, 2e-8, 64), np.linspace(-2e-8, 2e-8, 64))
        before = dose.contamination_angstrom(0.0, 0.0)
        for _ in range(50):
            dose.add_frame(xs, ys, 1e4, 4e-8 / 64)
        after = dose.contamination_angstrom(0.0, 0.0)
        assert before == 0.0
        assert 0 < after <= 50.0
        assert dose.contamination_angstrom(9e-8, 9e-8) == 0.0

    def test_disabled_dose_does_nothing(self):
        dose = DoseMap(DoseParams(enabled=False), (-1e-7, 1e-7, -1e-7, 1e-7))
        dose.add_frame(np.zeros((2, 2)), np.zeros((2, 2)), 1e6, 1e-9)
        assert dose.summary()["max_dose_e_per_a2"] == 0.0
