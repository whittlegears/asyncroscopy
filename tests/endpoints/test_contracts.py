"""The shared schemas are valid and agree with each other."""


import jsonschema
import pytest

from asyncroscopy.contracts import load, validate

SCHEMAS = ["envelope", "map_plan", "trace_event"]


@pytest.mark.parametrize("name", SCHEMAS)
def test_schema_is_valid(name):
    schema = load(name)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["schema_version"].count(".") == 2


def test_minimal_map_plan_validates():
    validate({"box_m": [0, 1e-6, 0, 1e-6], "fov_m": 2e-7}, "map_plan")


def test_unknown_plan_key_is_rejected():
    with pytest.raises(jsonschema.ValidationError):
        validate({"box_m": [0, 1e-6, 0, 1e-6], "fov_m": 2e-7, "speed": "fast"}, "map_plan")


def test_eds_map_must_be_named():
    with pytest.raises(jsonschema.ValidationError):
        validate({"box_m": [0, 1e-6, 0, 1e-6], "fov_m": 2e-7, "per_tile": ["eds_everything"]}, "map_plan")


def test_replies_carry_the_envelope_schema_version():
    from asyncroscopy.contracts import ENVELOPE_VERSION

    assert ENVELOPE_VERSION == load("envelope")["schema_version"]
