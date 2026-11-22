"""A sample that differs from place to place, a specimen that is not flat and a focus that drifts (no Tango needed)."""

import numpy as np
import pytest

from asyncroscopy.instruments.electron_microscope.twin import SampleModel, SampleParams
from asyncroscopy.instruments.electron_microscope.twin.optics import FocusDrift
from asyncroscopy.instruments.electron_microscope.twin.regions import HOLE, Regions
from asyncroscopy.sim.rng import EventRng

SAMPLE = dict(size_xy_m=2e-6, particle_count=1500, radius_mean_m=6e-9, radius_std_m=1e-9, elements=("Au", "Pt", "Fe", "Pd"), support_thickness_m=1e-8)


def sample(seed=5, **options):
    return SampleModel(SampleParams(**SAMPLE, **options), EventRng(seed))


class TestRegions:
    def test_a_sample_without_regions_is_unchanged(self):
        plain, relief = sample(), sample(height_relief_m=1e-7)
        assert np.array_equal(plain.centers, relief.centers) and plain.elements == relief.elements
        assert plain.support_texture(0.0, 0.0, 8, 1e-9) is None
        assert "regions" not in plain.ground_truth()

    def test_same_seed_same_regions(self):
        a, b = sample(region_count=12), sample(region_count=12)
        assert np.array_equal(a.centers, b.centers) and np.array_equal(a.region_of, b.region_of)

    def test_holes_have_no_particles_and_no_film(self):
        s = sample(region_count=20, region_hole_fraction=0.4)
        holes = np.nonzero(s.regions.film <= HOLE)[0]
        assert len(holes) > 0 and not np.isin(s.region_of, holes).any()
        x, y = s.regions.seeds[holes[0]]
        assert s.support_change_at(x, y) == HOLE
        assert np.all(s.support_texture(x - 4e-9, y - 4e-9, 8, 1e-9) == HOLE)

    def test_regions_differ_in_density(self):
        s = sample(region_count=12, region_hole_fraction=0.0)
        counts = np.bincount(s.region_of, minlength=12)
        assert counts.max() > 3 * max(1, counts.min())

    def test_particle_size_follows_the_marker_element(self):
        s = sample(region_count=12, region_hole_fraction=0.0)
        truth = s.ground_truth()
        share = np.array([r["element_shares"][truth["marker_element"]] for r in truth["regions"]])
        crowded = [i for i in range(12) if (s.region_of == i).sum() >= 10]
        mean_radius = np.array([s.semi_axes[s.region_of == i, 0].mean() for i in crowded])
        assert np.corrcoef(share[crowded], mean_radius)[0, 1] < -0.8

    def test_the_grid_lookup_agrees_with_the_point_lookup(self):
        regions = Regions.draw(np.random.default_rng(0), 30, 1e-6, ("Au", "Fe"), 0.2)
        xs, ys = np.linspace(-9e-7, 9e-7, 40), np.linspace(-5e-7, 7e-7, 30)
        grid = regions.index_on_grid(xs, ys)
        assert grid.shape == (30, 40)
        assert all(grid[r, c] == regions.index_at(xs[c], ys[r]) for r in range(0, 30, 3) for c in range(0, 40, 3))

    def test_impossible_region_options_are_refused(self):
        with pytest.raises(ValueError, match="region_hole_fraction"):
            sample(region_count=5, region_hole_fraction=1.0)


class TestHeightRelief:
    def test_relief_has_the_asked_size_and_keeps_the_gradient(self):
        s = sample(height_relief_m=2e-7, height_relief_scale_m=5e-7, height_gradient=(0.01, 0.0))
        x, y = np.meshgrid(np.linspace(-1e-6, 1e-6, 80), np.linspace(-1e-6, 1e-6, 80))
        relief = s.height_at(x, y) - 0.01 * x
        assert 1e-7 < relief.std() < 3e-7
        assert np.ndim(s.height_at(0.0, 0.0)) == 0


class TestFocusDrift:
    def test_no_drift_is_steady(self):
        assert FocusDrift(0.0, 0.0, np.random.default_rng(0)).at(5000.0) == (0.0, [0.0, 0.0])

    def test_drift_starts_at_zero_and_stays_within_twice_its_swing(self):
        drift = FocusDrift(4e-8, 1e-8, np.random.default_rng(3))
        assert drift.at(0.0) == (0.0, [0.0, 0.0])
        c1 = np.array([drift.at(t)[0] for t in np.arange(0.0, 3 * 86400.0, 300.0)])
        assert 1e-8 < np.abs(c1).max() <= 8e-8

    def test_same_seed_same_drift(self):
        a, b = FocusDrift(4e-8, 1e-8, np.random.default_rng(3)), FocusDrift(4e-8, 1e-8, np.random.default_rng(3))
        assert a.at(7200.0) == b.at(7200.0)
