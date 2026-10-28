"""The calibration scripts recover the stage errors the twin was configured with."""

import pytest

from asyncroscopy.calibration.stage_repeatability import measure
from tests.conftest import MAPPING_TWIN_PROPERTIES


def test_stage_repeatability_recovers_backlash_and_noise(mapping_twin_proxy, mapping_scan_proxy):
    mapping_scan_proxy.imsize = 256
    mapping_scan_proxy.dwell_time = 2e-6
    mapping_scan_proxy.scan_region = [0.0, 0.0, 1.0, 1.0]
    mapping_twin_proxy.set_defocus(0.0)
    mapping_twin_proxy.set_fov(150e-9)
    mapping_twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    result = measure(mapping_twin_proxy, 150e-9, 256, excursion=60e-9, repeats=10)
    assert result["stage_backlash_x"] == pytest.approx(MAPPING_TWIN_PROPERTIES["stage_backlash_x"], abs=2.5e-9)
    assert result["stage_backlash_y"] == pytest.approx(MAPPING_TWIN_PROPERTIES["stage_backlash_y"], abs=2.5e-9)
    assert result["stage_move_noise_std"] == pytest.approx(MAPPING_TWIN_PROPERTIES["stage_move_noise_std"], rel=0.5)
    assert result["stage_reported_noise_std"] == pytest.approx(MAPPING_TWIN_PROPERTIES["stage_reported_noise_std"], rel=0.6)
