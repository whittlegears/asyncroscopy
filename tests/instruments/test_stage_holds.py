"""The stage's holds: a limits conflict set_holder never clears, a connect failure the holder alarm never hides,
and a relative move that is an absolute move to a checked target."""

import json
import math
import pathlib
import subprocess
import sys
import tempfile

import pytest
from tango import DevState

from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE, narrow_travel
from asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript import AutoScriptSTAGE
from tests.fake_autoscript import FakeAutoScriptClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAVEL = {"x": (-2e-3, 2e-3), "y": (-1e-3, 1e-3), "z": (-1e-3, 1e-3), "alpha": (-70.0, 70.0), "beta": (-30.0, 30.0)}


class RecordingStage(STAGE):
    moved: list

    def _read_position(self):
        return [0.0] * 5

    def _write_position(self, value):
        self.moved.append(list(value))

    def _read_beta_tilt_enabled(self):
        return True

    def _is_moving(self):
        return False


def _held_stage(hardware: dict) -> RecordingStage:
    """A stage as init_device leaves it when the configured travel and the instrument's limits do not overlap."""
    d = RecordingStage.__new__(RecordingStage)
    d.moved = []
    d._configured_travel = dict(TRAVEL)
    d._hardware_limits = dict(hardware)
    d._travel, apart = narrow_travel(TRAVEL, hardware)
    d._holders, d._holder, d._frozen = {}, "", None
    d._limits = dict(d._travel)
    d._limits_conflict = STAGE._conflict(apart) if apart else None
    d._holder_path = pathlib.Path(tempfile.mkdtemp()) / "h.json"
    d._apply_limits = lambda: None
    d._read_hardware_limits = lambda: dict(hardware)
    state, status = [DevState.ALARM if apart else DevState.ON], [""]
    d.get_state, d.set_state = (lambda: state[0]), (lambda s: state.__setitem__(0, s))
    d.get_status, d.set_status = (lambda: status[0]), (lambda s: status.__setitem__(0, s))
    d._tango_properties = {}
    d.high_tilt_deg, d.high_tilt_max_abs_z_m = 60.0, 1.0
    d.state = state
    return d


def test_set_holder_does_not_clear_a_limits_conflict():
    d = _held_stage({"x": (5e-3, 6e-3)})
    with pytest.raises(ValueError, match="do not overlap"):
        STAGE.move_relative(d, [1e-3, 0, 0, 0, 0])
    STAGE.set_holder(d, "")
    assert d._limits_conflict and d.state[0] == DevState.ALARM
    assert json.loads(STAGE.get_holder(d))["limits_conflict"]
    for move in (lambda: STAGE.move_relative(d, [1e-3, 0, 0, 0, 0]), lambda: STAGE.write_position(d, [1e-3, 0, 0, 0, 0])):
        with pytest.raises(ValueError, match="do not overlap"):
            move()
    assert d.moved == []


def test_set_holder_reads_the_hardware_limits_again():
    d = _held_stage({})
    d._read_hardware_limits = lambda: {"beta": (-10.0, 10.0)}
    STAGE.set_holder(d, "")
    assert d._limits["beta"] == (-10.0, 10.0)
    d._read_hardware_limits = lambda: {"beta": (40.0, 50.0)}
    STAGE.set_holder(d, "")
    assert d._limits_conflict and "beta" in d._limits_conflict and d.state[0] == DevState.ALARM


def _autoscript_stage(client) -> AutoScriptSTAGE:
    device = AutoScriptSTAGE.__new__(AutoScriptSTAGE)
    device._tango_properties = {}
    device._microscope, device._frozen, device._holder, device._limits_conflict = client, None, "", None
    device._limits = {"x": (-1e-5, 1e-5), "y": (-1e-5, 1e-5), "z": (-1e-5, 1e-5), "alpha": (-30.0, 30.0), "beta": (-20.0, 20.0)}
    device.high_tilt_deg, device.high_tilt_max_abs_z_m = 90.0, 1.0
    return device


def test_a_relative_move_goes_to_the_checked_target_as_an_absolute_move():
    client = FakeAutoScriptClient()
    device = _autoscript_stage(client)
    STAGE.move_relative(device, [2e-6, 0.0, 0.0, 5.0, 0.0])
    (call,) = client.writes("specimen.stage.absolute_move_safe")  # the tilt changes, so the safe absolute move
    assert call["value"][:3] == pytest.approx([2e-6, 0.0, 0.0]) and call["value"][3] == pytest.approx(math.radians(5.0))
    STAGE.move_relative(device, [1e-6, 0.0, 0.0, 0.0, 0.0])
    safe, plain = client.writes("specimen.stage.absolute_move")
    assert plain["path"] == "specimen.stage.absolute_move" and plain["value"][0] == pytest.approx(3e-6)
    with pytest.raises(ValueError, match="outside the stage limits"):
        STAGE.move_relative(device, [9e-6, 0.0, 0.0, 0.0, 0.0])
    assert not hasattr(client.specimen.stage, "relative_move_safe")


def test_a_relative_move_is_refused_while_the_stage_moves():
    client = FakeAutoScriptClient()
    device = _autoscript_stage(client)
    client.specimen.stage.moving = True
    with pytest.raises(ValueError, match="the stage is moving"):
        STAGE.move_relative(device, [1e-6, 0.0, 0.0, 0.0, 0.0])
    assert client.writes("specimen.stage.absolute") == []


CONNECT_FAILURE = """
import json, sys
from tango.test_context import DeviceTestContext
from asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript import AutoScriptSTAGE
properties = {"holders": json.dumps({"a": {}, "b": {}}), "holder_file": sys.argv[1], "simulated": False, "hardware_host": ""}
with DeviceTestContext(AutoScriptSTAGE, properties=properties) as stage:
    print(json.dumps({"state": str(stage.State()), "status": stage.Status()}))
"""


def test_a_connect_failure_is_not_hidden_by_the_holder_alarm(tmp_path):
    record = tmp_path / "holder.json"
    record.write_text("not json", encoding="utf-8")
    done = subprocess.run([sys.executable, "-c", CONNECT_FAILURE, str(record)], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    (line,) = [line for line in done.stdout.splitlines() if line.startswith("{")]
    reply = json.loads(line)
    assert reply["state"] == "FAULT"
    assert "Not connected to AutoScript" in reply["status"] and "could not be read" in reply["status"]
