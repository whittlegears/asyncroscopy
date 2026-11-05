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

    # Same z and alpha as now: an x/y move, so the plain absolute_move.
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

    # beta changes (None -> 5 deg): AutoScript's absolute_move_safe, which zeroes the tilt before moving.
    assert hardware_stage.last_absolute_move is None
    assert hardware_stage.last_safe_move == (
        1.0,
        2.0,
        3.0,
        math.radians(4.0),
        math.radians(5.0),
    )
