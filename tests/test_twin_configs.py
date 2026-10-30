"""A twin configuration's Mapper must not state a smaller stage uncertainty than its stage really has.

The stitcher searches each overlap only within the stated uncertainty (stage_sigma_m) and drops a match that
disagrees with the stage by more than a few of them, so a value below the twin's real error (its random
stage_move_noise_std and its backlash on a reversal, together) makes it reject true matches: on the twin a tile
the stage put 470 nm from where it reported was left unplaced with a 150 nm stage_sigma_m.
"""

from pathlib import Path

import pytest
import yaml

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
def test_the_mapper_states_at_least_the_twin_stage_error(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    properties = raw["instrument"].get("properties") or {}
    backlash = max(float(properties.get("stage_backlash_x", 0.0)), float(properties.get("stage_backlash_y", 0.0)))
    noise = (float(properties.get("stage_move_noise_std", 0.0)) ** 2 + backlash**2) ** 0.5 * 0.99
    for device in raw["devices"].values():
        if isinstance(device, dict) and device.get("class_name") == "Mapper":
            sigma = float((device.get("properties") or {}).get("stage_sigma_m", 0.0))
            assert sigma >= noise, f"{path.name}: Mapper stage_sigma_m {sigma} is below the twin's stage error with backlash {noise / 0.99:.3g}"
