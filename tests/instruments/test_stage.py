import math

import pytest

from asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript import AutoScriptSTAGE


class FakeAutoScriptPosition:
    x = 1.0
    y = 2.0
    z = 3.0
    a = math.radians(4.0)
    b = None


class FakeAutoScriptStage:
    def __init__(self, holder_type: str):
        self.holder_type = holder_type
        self.position = FakeAutoScriptPosition()
        self.last_absolute_move = None
        self.last_safe_move = None

    def get_holder_type(self) -> str:
        return self.holder_type

    def absolute_move(self, position) -> None:
        self.last_absolute_move = position

    def absolute_move_safe(self, position) -> None:
        self.last_safe_move = position


class FakeSpecimen:
    def __init__(self, stage: FakeAutoScriptStage):
        self.stage = stage


class FakeMicroscope:
    def __init__(self, stage: FakeAutoScriptStage):
        self.specimen = FakeSpecimen(stage)


def make_autoscript_stage(holder_type: str) -> tuple[AutoScriptSTAGE, FakeAutoScriptStage]:
    stage_device = AutoScriptSTAGE.__new__(AutoScriptSTAGE)
    hardware_stage = FakeAutoScriptStage(holder_type)
    stage_device._microscope = FakeMicroscope(hardware_stage)
    return stage_device, hardware_stage


def test_test_stage_reports_beta_tilt_enabled(stage_proxy):
    assert stage_proxy.beta_tilt_enabled is True


def test_autoscript_single_tilt_read_returns_stable_five_value_position():
    stage_device, _ = make_autoscript_stage("SingleTilt")

    assert stage_device._read_position() == [1.0, 2.0, 3.0, 4.0, 0.0]


def test_autoscript_single_tilt_write_accepts_four_value_position():
    stage_device, hardware_stage = make_autoscript_stage("SingleTilt")

    stage_device._write_position([1.0, 2.0, 3.0, 4.0])

    # z and alpha are the same as now, so this is an x/y move and uses the plain absolute_move.
    assert hardware_stage.last_absolute_move == (1.0, 2.0, 3.0, math.radians(4.0))


def test_autoscript_single_tilt_write_refuses_a_beta_instead_of_dropping_it():
    stage_device, hardware_stage = make_autoscript_stage("SingleTilt")

    with pytest.raises(ValueError, match="no beta tilt"):
        stage_device._write_position([1.0, 2.0, 3.0, 4.0, 5.0])
    stage_device._write_position([1.0, 2.0, 3.0, 4.0, 0.0])  # beta 0 is no beta

    assert hardware_stage.last_safe_move is None
    assert hardware_stage.last_absolute_move == (1.0, 2.0, 3.0, math.radians(4.0))


def test_autoscript_double_tilt_write_passes_beta_move_through_the_safe_move():
    stage_device, hardware_stage = make_autoscript_stage("DoubleTilt")

    stage_device._write_position([1.0, 2.0, 3.0, 4.0, 5.0])

    # beta changes from None to 5 deg, so this uses AutoScript's absolute_move_safe, which zeroes the tilt before moving.
    assert hardware_stage.last_absolute_move is None
    assert hardware_stage.last_safe_move == (
        1.0,
        2.0,
        3.0,
        math.radians(4.0),
        math.radians(5.0),
    )


def test_autoscript_stage_reads_its_axis_limits_and_they_narrow_the_configured_travel(monkeypatch):
    from asyncroscopy.instruments.electron_microscope.hardware.stage import narrow_travel
    from tests.fake_autoscript import HARDWARE_LIMITS, FakeAutoScriptClient

    client = FakeAutoScriptClient()
    device = AutoScriptSTAGE.__new__(AutoScriptSTAGE)
    device.warn_stream = lambda *a: None
    device._microscope = client
    limits = device._read_hardware_limits()
    assert [c["value"] for c in client.calls if c["path"] == "specimen.stage.get_axis_limits"] == ["X", "Y", "Z", "A", "B"]
    assert limits["x"] == pytest.approx(HARDWARE_LIMITS["stage_x"]) and limits["alpha"] == pytest.approx((-70.0, 70.0))
    assert client.writes("specimen.stage.absolute") == []  # reading limits moves nothing

    travel = {"x": (-2e-3, 2e-3), "y": (-5e-4, 5e-4), "z": (-5e-4, 5e-4), "alpha": (-35.0, 35.0), "beta": (-20.0, 20.0)}
    narrowed, apart = narrow_travel(travel, limits)
    assert narrowed["x"] == pytest.approx(HARDWARE_LIMITS["stage_x"]) and narrowed["y"] == (-5e-4, 5e-4) and apart == []
    _, apart = narrow_travel(travel, {"x": (5e-3, 6e-3)})
    assert apart and apart[0].startswith("x")

    object.__setattr__(client.specimen.stage, "holder_type", "SingleTilt")
    assert "beta" not in device._read_hardware_limits()  # a single-tilt holder has no beta to ask about
    device._microscope = None
    assert device._read_hardware_limits() == {}  # not connected: the configured limits apply
