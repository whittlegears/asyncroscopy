"""The twin must survive whatever a hackathon script or an agent throws at it."""

import numpy as np
import pytest

from asyncroscopy.instruments.electron_microscope.twin import (
    EventRng,
    SampleModel,
    SampleParams,
    ScanParams,
    pixel_sample_coordinates,
    point_spectrum,
)
from asyncroscopy.instruments.electron_microscope.twin.detector import DetectorParams, count
from asyncroscopy.instruments.electron_microscope.twin.eds_model import EdsParams
from asyncroscopy.instruments.electron_microscope.twin.optics import aberrations_nm
from asyncroscopy.instruments.electron_microscope.twin.render import Renderer, RenderParams


def render(sample, fov, origin=(0.0, 0.0), imsize=32, region=(0, 0, 1, 1), defocus=0.0):
    xs, ys = pixel_sample_coordinates(imsize, fov, region, 1e-6, ScanParams(), lambda t: np.asarray(origin, float), 0.0, EventRng(0).for_event("s"))
    return Renderer(sample, RenderParams()).render(xs, ys, fov / imsize, "HAADF", aberrations_nm(None, 300e3, defocus), 50.0)


@pytest.fixture(scope="module")
def sample():
    return SampleModel(SampleParams(size_xy_m=5e-7, particle_count=30, radius_mean_m=8e-9, radius_std_m=3e-9, aspect_ratio=0.6, support_thickness_m=1e-8), EventRng(9))


def test_zero_or_impossible_particle_counts():
    # More particles than fit: the generator gives up gracefully with what fits.
    crowded = SampleModel(SampleParams(size_xy_m=3e-8, particle_count=10_000, radius_mean_m=8e-9), EventRng(1))
    assert 0 < len(crowded) < 10_000
    empty = SampleModel(SampleParams(size_xy_m=1e-7, particle_count=0), EventRng(1))
    image = render(empty, 1e-7)
    assert np.isfinite(image).all()


@pytest.mark.parametrize("fov", [1e-11, 2e-10, 1e-5, 1e-3])
def test_extreme_fields_of_view_render(sample, fov):
    image = render(sample, fov, imsize=32)
    assert image.shape == (32, 32) and np.isfinite(image).all() and (image >= 0).all()


def test_stage_far_off_sample_shows_support_only(sample):
    image = render(sample, 1e-7, origin=(1e-3, -1e-3))
    assert np.isfinite(image).all()
    assert image.std() < 1e-6  # nothing there but uniform support film


@pytest.mark.parametrize("region", [(0, 0, 1 / 32, 1 / 32), (0.999, 0.999, 1.0, 1.0), (0.0, 0.5, 1.0, 0.5000001)])
def test_tiny_scan_regions(sample, region):
    image = render(sample, 1e-7, region=region)
    assert min(image.shape) >= 1 and np.isfinite(image).all()


@pytest.mark.parametrize("region", [(0.5, 0, 0.5, 1), (-0.1, 0, 1, 1), (0, 0, 1.2, 1)])
def test_invalid_scan_regions_rejected(region):
    with pytest.raises(ValueError):
        pixel_sample_coordinates(32, 1e-7, region, 1e-6, ScanParams(), lambda t: np.zeros(2), 0.0, EventRng(0).for_event("s"))


def test_huge_defocus_still_renders(sample):
    image = render(sample, 1e-7, defocus=1e-5)
    assert np.isfinite(image).all()


@pytest.mark.parametrize("dwell", [0.0, 1e-12, 1.0])
def test_dwell_extremes_give_valid_counts(dwell):
    counts = count(np.full((8, 8), 0.5), dwell, DetectorParams(beam_current_pa=50.0), EventRng(2).for_event("d"))
    assert counts.dtype == np.uint16 and counts.min() >= 0


def test_counts_saturate_instead_of_wrapping():
    counts = count(np.full((4, 4), 1.0), 10.0, DetectorParams(beam_current_pa=1e6), EventRng(2).for_event("d"))
    assert (counts == np.iinfo(np.uint16).max).all()


@pytest.mark.parametrize("real_time", [0.0, 1e-9, 100.0])
def test_eds_time_extremes(real_time):
    energies, counts, live = point_spectrum({"Au": 5.0}, 50.0, real_time, EdsParams(), EventRng(3).for_event("e"))
    assert np.isfinite(counts).all() and counts.min() >= 0 and 0 <= live <= real_time


def test_unknown_element_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown sample elements"):
        SampleModel(SampleParams(elements=("Unobtainium",)), EventRng(1))
