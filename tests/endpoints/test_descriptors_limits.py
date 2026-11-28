"""with_declared_limits: a device's declared ranges and units in a tool's JSON input schema."""

from asyncroscopy.mcp.descriptors import with_declared_limits


def test_a_number_gets_its_range_a_string_its_choices_and_both_their_units_in_words():
    params = {"type": "object", "properties": {"fov": {"type": "number", "description": "Field of view."}, "mode": {"type": "string"}}}
    hint = {"args": [{"name": "fov", "min": 1e-9, "max": 1e-4, "units": "m"}, {"name": "mode", "enum": ["a", "b"]}]}
    out = with_declared_limits(params, hint)
    assert out["properties"]["fov"] == {"type": "number", "minimum": 1e-9, "maximum": 1e-4, "description": "Field of view. Units: m."}
    assert out["properties"]["mode"] == {"type": "string", "enum": ["a", "b"]}
    assert params["properties"]["fov"] == {"type": "number", "description": "Field of view."}  # the schema it was given is not changed
    assert with_declared_limits(out, hint) == out  # applying it twice adds nothing


def test_an_array_gets_one_range_on_its_items_and_ranges_by_element_in_words():
    one = {"properties": {"arg": {"type": "array", "items": {"type": "number"}}}}
    assert with_declared_limits(one, {"args": [{"name": "position", "min": 0.0, "max": 1.0}]})["properties"]["arg"]["items"] == {
        "type": "number", "minimum": 0.0, "maximum": 1.0}
    each = with_declared_limits(one, {"args": [{"name": "position", "elements": ["x", "alpha"], "units": ["m", "deg"], "min": [-1, -30], "max": [1, 30]}]})
    assert each["properties"]["arg"]["items"] == {"type": "number"}  # no one range holds for every element
    assert each["properties"]["arg"]["description"] == "Elements x: -1..1 m, alpha: -30..30 deg."


def test_an_optional_argument_gets_them_on_its_value_and_nothing_is_guessed():
    optional = {"properties": {"x": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None}}}
    assert with_declared_limits(optional, {"args": [{"name": "x", "max": 5}]})["properties"]["x"]["anyOf"][0] == {"type": "number", "maximum": 5}
    two = {"properties": {"a": {"type": "number"}, "b": {"type": "number"}}}
    assert with_declared_limits(two, {"args": [{"name": "c", "max": 5}]}) == two  # no property of that name, and more than one
    assert with_declared_limits(two, {}) == two and with_declared_limits({}, {"args": [{"name": "a"}]}) == {}


def test_number_choices_are_said_in_words_since_some_providers_refuse_an_enum_of_numbers():
    params = {"properties": {"binning": {"type": "integer", "description": "Binning."}}}
    out = with_declared_limits(params, {"args": [{"name": "binning", "enum": [1, 2, 4]}]})
    assert out["properties"]["binning"] == {"type": "integer", "description": "Binning. One of 1, 2, 4."}
