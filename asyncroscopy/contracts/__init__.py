"""Shared JSON Schemas: the job envelope, map plans and Instrument Trace events. See README.md."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
ENVELOPE_VERSION = "1.0.0"


@cache
def load(name: str) -> dict:
    """Load ``<name>.schema.json`` from this folder."""
    return json.loads((HERE / f"{name}.schema.json").read_text(encoding="utf-8"))


def validate(instance: dict, name: str, definition: str | None = None) -> None:
    """Raise jsonschema.ValidationError unless ``instance`` matches the schema (or one of its $defs)."""
    schema = load(name)
    if definition is not None:
        schema = {"$ref": f"#/$defs/{definition}", **{k: v for k, v in schema.items() if k in ("$defs", "$schema")}}
    jsonschema.validate(instance, schema)
