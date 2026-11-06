"""A twin's Mapper must not search a smaller stage uncertainty than the twin's stage really has.

The stitcher searches each overlap only within the stated uncertainty (stage_sigma_m) and drops a match that
disagrees with the stage by more than a few of them, so a value below the twin's real error (its random
stage_move_noise_std and its backlash on a reversal, together) makes it reject true matches: on the twin a tile
the stage put 470 nm from where it reported was left unplaced with a 150 nm stage_sigma_m. A twin works the
error out from its own stage model and reports it through its STAGE device (DigitalTwinSTAGE's
position_sigma_m), which a Mapper without its own stage_sigma_m reads.
"""

import json
import math
from pathlib import Path

import pytest
import tango
import yaml

from tests.conftest import DRIFT_TWIN_PROPERTIES

ROOT = Path(__file__).resolve().parents[1]


def _twins_with_a_mapper() -> list[Path]:
    found = []
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        instrument = raw.get("instrument") or {}
        mappers = [d for d in (raw.get("devices") or {}).values() if isinstance(d, dict) and d.get("class_name") == "Mapper"]
        if str(instrument.get("class_name", "")).startswith("DigitalTwin") and mappers:
            found.append(path)
    return found


def test_there_are_twin_configurations_with_a_mapper():
    assert _twins_with_a_mapper()


@pytest.mark.parametrize("path", _twins_with_a_mapper(), ids=lambda p: p.name)
def test_a_twin_mapper_takes_the_stage_error_from_the_twin_stage_or_states_at_least_it(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    stage = raw["devices"].get("stage") or {}
    assert stage.get("class_name") == "DigitalTwinSTAGE", f"{path.name}: a twin's STAGE device is DigitalTwinSTAGE, not a test fixture"
    properties = raw["instrument"].get("properties") or {}
    backlash = max(float(properties.get("stage_backlash_x", 0.0)), float(properties.get("stage_backlash_y", 0.0)))
    noise = (float(properties.get("stage_move_noise_std", 0.0)) ** 2 + backlash**2) ** 0.5 * 0.99
    for device in raw["devices"].values():
        if isinstance(device, dict) and device.get("class_name") == "Mapper":
            sigma = float((device.get("properties") or {}).get("stage_sigma_m", 0.0))
            assert sigma == 0.0 or sigma >= noise, f"{path.name}: Mapper stage_sigma_m {sigma} is below the twin's stage error with backlash {noise / 0.99:.3g}"


def test_the_twin_reports_its_stage_error_and_a_mapper_without_its_own_reads_it(tango_ctx, monkeypatch):
    twin = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/drift"))
    stage = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/stage/drift"))
    twin.get_stage()  # the twin reports its stage error once its STAGE answers
    expected = math.sqrt(DRIFT_TWIN_PROPERTIES["stage_move_noise_std"] ** 2 + DRIFT_TWIN_PROPERTIES["stage_reported_noise_std"] ** 2)
    assert stage.position_sigma_m == pytest.approx(expected)

    mapper = tango.Util.instance().get_device_by_name("asyncroscopy/mapper/drift")
    assert mapper._stage_sigma_m() == pytest.approx(3e-9)  # its own stage_sigma_m wins
    monkeypatch.setattr(mapper, "stage_sigma_m", 0.0)
    assert mapper._stage_sigma_m() == pytest.approx(expected)
    plan = {"box_m": [-1e-7, 1e-7, -1e-7, 1e-7], "fov_m": 1e-7, "overlap": 0.25, "imsize": 32, "stitch": False}
    resolved = mapper.resolve_plan(json.loads(json.dumps(plan)))
    assert resolved["stage_sigma_m"] == pytest.approx(expected)
