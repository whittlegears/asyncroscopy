"""Tests for the APERTURE base-class commands, run against TestAperture.

Mechanism selection was previously attribute-only, which made it unreachable
through command-only bridges like the MCP server: devices boot with the
non-retractable C1 mechanism selected, so retract could never succeed. These
tests exercise the command-based selection flow end to end.
"""

import json

import pytest
import tango


@pytest.fixture
def restored_aperture(aperture_proxy: tango.DeviceProxy):
    """Yield the aperture proxy and restore its selection state afterwards.

    The device is session-scoped and shared with the digital twin tests, so
    mechanism/aperture changes must not leak.
    """
    original_mechanism = aperture_proxy.get_mechanism()
    yield aperture_proxy
    aperture_proxy.set_mechanism(original_mechanism)


def test_get_available_mechanisms_matches_attribute(aperture_proxy: tango.DeviceProxy):
    assert list(aperture_proxy.get_available_mechanisms()) == list(
        aperture_proxy.available_mechanisms
    )


def test_set_mechanism_round_trips(restored_aperture: tango.DeviceProxy):
    restored_aperture.set_mechanism("OBJ")

    assert restored_aperture.get_mechanism() == "OBJ"
    assert restored_aperture.mechanism == "OBJ"
    assert list(restored_aperture.get_available_apertures()) == list(
        restored_aperture.available_apertures
    )


def test_set_mechanism_rejects_unknown_name(restored_aperture: tango.DeviceProxy):
    with pytest.raises(tango.DevFailed, match="Unknown mechanism"):
        restored_aperture.set_mechanism("nope")

    # The attribute write path validates identically.
    with pytest.raises(tango.DevFailed, match="Unknown mechanism"):
        restored_aperture.mechanism = "nope"

    # A rejected write must not change the selection.
    assert restored_aperture.get_mechanism() != "nope"


def test_set_selected_aperture_validates(restored_aperture: tango.DeviceProxy):
    restored_aperture.set_mechanism("OBJ")
    original = restored_aperture.selected_aperture

    restored_aperture.set_selected_aperture("40 um")
    assert restored_aperture.selected_aperture == "40 um"

    with pytest.raises(tango.DevFailed, match="Unknown aperture"):
        restored_aperture.set_selected_aperture("999 um")

    restored_aperture.set_selected_aperture(original)


def test_get_aperture_info_reports_full_state(restored_aperture: tango.DeviceProxy):
    restored_aperture.set_mechanism("SA")

    info = json.loads(restored_aperture.get_aperture_info())

    assert info["mechanism"] == "SA"
    assert info["available_mechanisms"] == list(restored_aperture.available_mechanisms)
    assert info["selected_aperture"] == restored_aperture.selected_aperture
    assert info["aperture_type"] == "Circular"
    assert info["aperture_diameter_m"] == pytest.approx(
        restored_aperture.aperture_diameter
    )
    assert info["insertion_state"] in ("Inserted", "Retracted")
    assert info["retractable"] is True
    assert isinstance(info["enabled"], bool)
    assert len(info["position"]) == 2


def test_retract_reachable_through_commands_alone(restored_aperture: tango.DeviceProxy):
    """The MCP-visible flow: select a retractable mechanism, retract, re-insert."""
    restored_aperture.set_mechanism("C1")
    with pytest.raises(tango.DevFailed, match="not retractable"):
        restored_aperture.retract()

    restored_aperture.set_mechanism("OBJ")
    assert restored_aperture.retractable is True

    restored_aperture.retract()
    assert restored_aperture.insertion_state == "Retracted"

    restored_aperture.insert()
    assert restored_aperture.insertion_state == "Inserted"
