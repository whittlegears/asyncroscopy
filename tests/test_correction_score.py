import math

import numpy as np
import pyTEMlib.probe_tools as probe_tools
import pytest

from asyncroscopy.analysis.correction import (
    ANGLE_STEP_MRAD,
    MAX_ANGLE_MRAD,
    REQUIRED,
    TERMS,
    ceos_to_pytemlib,
    chi,
    correction_score,
    electron_wavelength_m,
)
from asyncroscopy.instruments.electron_microscope.hardware.corrector_digital_twin import DEFAULT_ABERRATIONS

KV300 = 300e3
LAM = electron_wavelength_m(KV300)


def _zero(**overrides):
    """All required aberrations present and zero, then the given values."""
    return {**{name: [0.0, 0.0] for name in REQUIRED}, **overrides}


def test_wavelength_at_300_kv():
    assert LAM == pytest.approx(1.9687e-12, rel=1e-3)


def test_defocus_alone_matches_the_closed_form():
    c1 = 10e-9
    # pi * C1 * theta^2 / lambda = pi/4
    expected_mrad = math.sqrt(LAM / (4 * c1)) * 1e3
    assert correction_score(_zero(C1=[c1]), KV300).aberration_free_angle_mrad == pytest.approx(expected_mrad, abs=ANGLE_STEP_MRAD)


def test_spherical_aberration_alone_matches_the_closed_form():
    c3 = 1e-6
    # (2 pi / lambda) * C3 * theta^4 / 4 = pi/4
    expected_mrad = (LAM / (2 * c3)) ** 0.25 * 1e3
    assert correction_score(_zero(C3=[c3]), KV300).aberration_free_angle_mrad == pytest.approx(expected_mrad, abs=ANGLE_STEP_MRAD)


def test_astigmatism_limit_does_not_depend_on_its_direction():
    angles = [
        correction_score(_zero(A1=[5e-9 * math.cos(a), 5e-9 * math.sin(a)]), KV300).aberration_free_angle_mrad
        for a in (0.0, 0.7, 2.0)
    ]
    assert max(angles) - min(angles) <= 2 * ANGLE_STEP_MRAD


def test_target_and_best_next_correction():
    score = correction_score(_zero(C1=[20e-9], C3=[50e-9]), KV300, target_mrad=25.0)

    assert score.passed is False
    assert score.best_next_correction == "C1"
    assert score.gain_if_corrected_mrad["C1"] > 0


def test_balancing_defocus_shows_a_negative_gain():
    # Negative defocus partly cancels positive C3, so removing it shrinks the angle.
    # The deepest dip that still stays within -pi/4 is C1 = -sqrt(C3 * lambda / 2).
    c3 = 1e-6
    score = correction_score(_zero(C1=[-math.sqrt(c3 * LAM / 2) * 0.95], C3=[c3]), KV300)

    assert score.gain_if_corrected_mrad["C1"] < 0


def test_no_aberrations_reach_the_end_of_the_search():
    score = correction_score(_zero(), KV300)

    assert score.aberration_free_angle_mrad == pytest.approx(MAX_ANGLE_MRAD)
    assert score.best_next_correction is None


def test_fast_tableau_is_refused():
    fast = {"WD": [0.0, 0.0], "C1": [1e-9, 0.0], "A1": [0.0, 0.0], "A2": [0.0, 0.0], "B2": [0.0, 0.0]}
    with pytest.raises(ValueError, match="Standard or Enhanced"):
        correction_score(fast, KV300)


def test_a_two_component_aberration_given_one_value_is_refused():
    with pytest.raises(ValueError, match="B2 needs an"):
        correction_score(_zero(B2=[1e-9]), KV300)


def test_twin_default_aberrations_score_without_error():
    score = correction_score(DEFAULT_ABERRATIONS, KV300, target_mrad=30.0)

    assert 0 < score.aberration_free_angle_mrad < MAX_ANGLE_MRAD
    assert set(score.gain_if_corrected_mrad) >= {"C1", "C3", "S3", "A3", "D4", "B4", "A4"}


def test_conversion_to_pytemlib_gives_the_same_aberration_function():
    """pyTEMlib's make_chi on the converted values equals chi on the CEOS values, for every term."""
    rng = np.random.default_rng(7)
    scales = {"C1": 5e-9, "A1": 5e-9, "B2": 50e-9, "A2": 50e-9, "C3": 1e-6, "S3": 500e-9, "A3": 500e-9,
              "B4": 5e-6, "D4": 5e-6, "A4": 5e-6, "C5": 1e-3, "A5": 1e-3}
    assert set(scales) == set(TERMS)
    ceos = {name: [rng.normal() * scale, 0.0 if name in ("C1", "C3", "C5") else rng.normal() * scale]
            for name, scale in scales.items()}

    theta = np.linspace(0.0, 0.04, 41)[:, None] * np.ones((1, 72))
    phi = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)[None, :] * np.ones((41, 1))
    ours = chi(ceos, LAM, theta, phi)
    pytemlib = probe_tools.make_chi(phi, theta, {**ceos_to_pytemlib(ceos), "wavelength": LAM * 1e9})

    np.testing.assert_allclose(pytemlib, ours, rtol=1e-9, atol=1e-9 * np.abs(ours).max())


def test_d4_is_the_three_fold_term_and_b4_the_one_fold():
    """A readable spot check of the factors and of the D4/B4 order the twin once had swapped."""
    assert ceos_to_pytemlib({"D4": [1e-9, 0.0]}) == {"C43a": pytest.approx(5.0), "C43b": 0.0}
    assert ceos_to_pytemlib({"B4": [1e-9, 0.0]}) == {"C41a": pytest.approx(5.0), "C41b": 0.0}
