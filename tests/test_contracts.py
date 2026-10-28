"""The shared schemas are valid, agree with each other, and any copies elsewhere match."""

import importlib.util
import json

import jsonschema
import pytest

from asyncroscopy.contracts import HERE, load, validate

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


@pytest.mark.skipif(importlib.util.find_spec("pyTEMlib.mcp_server") is None, reason="pyTEMlib MCP server not installed")
def test_pytemlib_copy_of_the_envelope_matches():
    from pyTEMlib import mcp_server

    ours = json.loads((HERE / "envelope.schema.json").read_text(encoding="utf-8"))
    theirs = json.loads(mcp_server.ENVELOPE_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert ours == theirs, "pyTEMlib's envelope.schema.json has drifted from asyncroscopy's"
