"""
Tests that the DigitalTwin exposes the same Tango surface as the real
AutoScriptMicroscope: read attributes, acquire_scanned_data_advanced,
register_stage, and a simulated aperture device.

TestClassSurfaceParity locks in full command/attribute parity between each
twin device class and its real (AutoScript) counterpart, so a command added
to only one side fails CI instead of silently diverging the MCP tool surface.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango
from tango.server import attribute as TangoAttr

from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.auto_script import AutoScriptMicroscope
from asyncroscopy.instruments.electron_microscope.hardware.TestAperture import TestAperture
from asyncroscopy.instruments.electron_microscope.hardware.aperture_autoscript import (
    AutoScriptAPERTURE,
)
from asyncroscopy.instruments.electron_microscope.hardware.TestStage import TestStage
from asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript import (
    AutoScriptSTAGE,
)
from asyncroscopy.instruments.electron_microscope.hardware.TestCorrector import TestCorrector
from asyncroscopy.instruments.electron_microscope.hardware.corrector import CORRECTOR


def tango_commands(cls) -> dict[str, tuple]:
    """Map command name -> (in_dtype, out_dtype) for every @command on cls."""
    commands = {}
    for name in dir(cls):
        try:
            member = getattr(cls, name)
        except Exception:
            continue
        meta = getattr(member, "__tango_command__", None)
        if meta:
            commands[name] = (meta[1][0][0], meta[1][1][0])
    return commands


def tango_attributes(cls) -> set[str]:
    return {
        name
        for klass in cls.__mro__
        for name, member in vars(klass).items()
        if isinstance(member, TangoAttr)
    }


class TestClassSurfaceParity:
    """The MCP bridge exposes Tango commands as tools, so command parity here
    is tool parity on a deployment."""

    PAIRS = [
        (DigitalTwin, AutoScriptMicroscope),
        (TestAperture, AutoScriptAPERTURE),
        (TestStage, AutoScriptSTAGE),
        (TestCorrector, CORRECTOR),
    ]

    @pytest.mark.parametrize("twin_cls,real_cls", PAIRS, ids=lambda c: c.__name__)
    def test_commands_and_dtypes_match(self, twin_cls, real_cls):
        twin_cmds = tango_commands(twin_cls)
        real_cmds = tango_commands(real_cls)

        assert set(twin_cmds) == set(real_cmds), (
            f"command sets diverge: only-twin={sorted(set(twin_cmds) - set(real_cmds))}, "
            f"only-real={sorted(set(real_cmds) - set(twin_cmds))}"
        )
        for name in twin_cmds:
            assert twin_cmds[name] == real_cmds[name], (
                f"{name}: twin dtypes {twin_cmds[name]} != real dtypes {real_cmds[name]}"
            )

    @pytest.mark.parametrize("twin_cls,real_cls", PAIRS, ids=lambda c: c.__name__)
    def test_attributes_match(self, twin_cls, real_cls):
        twin_attrs = tango_attributes(twin_cls)
        real_attrs = tango_attributes(real_cls)
        # beam_pos is a deliberate twin-only introspection attribute; attributes
        # are not exposed as MCP tools, so it does not affect tool parity.
        allowed_twin_extras = {"beam_pos"}

        assert twin_attrs - real_attrs <= allowed_twin_extras, (
            f"unexpected twin-only attributes: {sorted(twin_attrs - real_attrs - allowed_twin_extras)}"
        )
        assert real_attrs <= twin_attrs, (
            f"real-only attributes missing from twin: {sorted(real_attrs - twin_attrs)}"
        )


class TestDigitalTwinParity:

    def test_read_attributes_match_real_microscope(self, twin_proxy: tango.DeviceProxy):
        twin_proxy.set_fov(5e-8)
        twin_proxy.set_defocus(3e-9)

        assert twin_proxy.fov == pytest.approx(5e-8)
        assert twin_proxy.defocus == pytest.approx(3e-9)
        assert twin_proxy.acceleration_voltage == pytest.approx(200e3)
        assert twin_proxy.camera_length > 0.0

        twin_proxy.blank_beam()
        assert twin_proxy.beam_state is True
        twin_proxy.unblank_beam()
        assert twin_proxy.beam_state is False

    def test_acquire_scanned_data_advanced_saves_stem_data(
        self,
        twin_proxy: tango.DeviceProxy,
        scan_proxy: tango.DeviceProxy,
    ):
        scan_proxy.imsize = 32
        scan_proxy.dwell_time = 1e-6

        saved_path = Path(twin_proxy.acquire_scanned_data_advanced())

        assert saved_path.suffix == ".h5"
        assert saved_path.exists()
        with h5py.File(saved_path, "r") as h5:
            data = h5["stem_data"][()]
            assert data.shape == (32, 32)
            assert h5["stem_data"].attrs["acquisition_type"] == "stem_data"
            assert h5["stem_data"].attrs["detector"] == "BM-Ceta"

    def test_register_stage_publishes_position(
        self,
        twin_proxy: tango.DeviceProxy,
        stage_proxy: tango.DeviceProxy,
    ):
        twin_proxy.move_stage([1e-6, -2e-6, 0.0, 0.0, 0.0])

        twin_proxy.register_stage()

        assert stage_proxy.x == pytest.approx(1e-6)
        assert stage_proxy.y == pytest.approx(-2e-6)
        twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])

    def test_aperture_device_simulates_mechanisms(self, aperture_proxy: tango.DeviceProxy):
        mechanisms = list(aperture_proxy.available_mechanisms)
        assert "OBJ" in mechanisms and "C2" in mechanisms

        aperture_proxy.mechanism = "OBJ"
        names = list(aperture_proxy.available_apertures)
        aperture_proxy.selected_aperture = names[1]

        assert aperture_proxy.selected_aperture == names[1]
        assert aperture_proxy.aperture_type == "Circular"
        assert aperture_proxy.aperture_diameter > 0.0
        assert aperture_proxy.retractable is True

        aperture_proxy.retract()
        assert aperture_proxy.insertion_state == "Retracted"
        aperture_proxy.insert()
        assert aperture_proxy.insertion_state == "Inserted"

    def test_non_retractable_mechanism_rejects_retract(self, aperture_proxy: tango.DeviceProxy):
        aperture_proxy.mechanism = "C2"
        with pytest.raises(tango.DevFailed):
            aperture_proxy.retract()

    def test_unknown_aperture_is_rejected(self, aperture_proxy: tango.DeviceProxy):
        aperture_proxy.mechanism = "OBJ"
        with pytest.raises(tango.DevFailed):
            aperture_proxy.selected_aperture = "999 um"

    def test_get_parameters_reports_aperture_mechanisms(self, twin_proxy: tango.DeviceProxy):
        parameters = json.loads(twin_proxy.get_parameters())

        assert parameters["acceleration_voltage"] == pytest.approx(200e3)
        assert "camera_length_m" in parameters
        apertures = parameters["apertures"]
        assert set(apertures) == {"C1", "C2", "C3", "OBJ", "SA"}
        # Inserted mechanisms report their selected aperture, like the real column
        assert apertures["C2"] in ("150 um", "70 um", "50 um", "30 um")
