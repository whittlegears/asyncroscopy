"""The twin's stage belongs to its STAGE device: the microscope only orchestrates it and keeps the clock."""

import json

import numpy as np
import pytest
import tango

from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
from asyncroscopy.instruments.electron_microscope.hardware.stage_digital_twin import DigitalTwinSTAGE
from asyncroscopy.sim.rng import EventRng
from asyncroscopy.sim.stage_model import StageModel, StageParams, StageTrajectory
from asyncroscopy.utils.commands import declared_hints


class CountingStage:
    """The microscope's stage proxy, counting how often rendering asks it for the trajectory."""

    def __init__(self, proxy):
        self._proxy = proxy
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._proxy, name)

    def frame_trajectory(self, request):
        self.calls += 1
        return self._proxy.frame_trajectory(request)


@pytest.fixture
def mapping_twin_device(tango_ctx):
    return tango.Util.instance().get_device_by_name("asyncroscopy/digitaltwin/mapping")


def test_the_microscope_has_no_stage_model_properties():
    for name in ("stage_bias_x", "stage_backlash_x", "stage_settle_drift", "stage_speed", "stage_reported_noise_std", "drift_rate_per_min", "stage_travel_m"):
        assert not hasattr(DigitalTwin, name), name
    assert hasattr(DigitalTwinSTAGE, "stage_backlash_x")


def test_rendering_a_frame_asks_the_stage_once_however_many_lines_it_has(mapping_twin_device):
    device = mapping_twin_device
    counting = CountingStage(device._detector_proxies["stage"])
    device._detector_proxies["stage"] = counting
    try:
        device._render_stem_image(32, 1e-7, ["HAADF"], [0.0, 0.0, 1.0, 1.0])
    finally:
        device._detector_proxies["stage"] = counting._proxy
    assert counting.calls == 1


def test_the_stage_commands_that_reveal_the_truth_are_operator_only():
    hints = declared_hints(DigitalTwinSTAGE)
    assert hints["move_on_clock"]["operator_only"] and hints["frame_trajectory"]["operator_only"] and hints["stage_moves"]["operator_only"]


def test_a_direct_write_counts_as_a_move_on_the_microscope_clock(mapping_twin_device, mapping_stage_proxy, mapping_twin_proxy):
    before = mapping_twin_device.instrument_clock_s()
    start = list(mapping_twin_proxy.get_stage())
    mapping_stage_proxy.x = start[0] + 5e-9
    stage = list(mapping_twin_proxy.get_stage())
    assert stage[0] == pytest.approx(start[0] + 5e-9, abs=5e-9)  # reported, with its readback error
    assert mapping_twin_device.instrument_clock_s() > before  # the move took simulated time
    assert json.loads(mapping_stage_proxy.stage_moves())[-1]["commanded"][0] == pytest.approx(start[0] + 5e-9)


def test_the_trajectory_gives_the_same_drift_as_the_model():
    params = StageParams(noise_std_m=1e-9, reported_noise_std_m=1e-10, backlash_m=(2e-9, 3e-9), settle_drift_m_s=2e-9, settle_tau_s=30.0, drift_m_per_min=5e-10, drift_random_walk_m=2e-10, move_overhead_s=1.0)
    model = StageModel(params, EventRng(11))
    assert model.move([1e-8, 2e-8, 0, 0, 0], at_s=100.0) > 0
    trajectory = StageTrajectory.from_dict(json.loads(json.dumps(model.trajectory(110.0, 400.0).to_dict())))
    for t in (110.0, 150.0, 181.0, 300.0, 399.5):
        assert np.array_equal(trajectory.true_xy_at(t), model.true_xy_at(t))
        assert np.array_equal(trajectory.drift_offset_at(t), model.drift_offset_at(t))
    assert np.array_equal(trajectory.reported, model.reported)


def test_a_restarted_microscope_starts_a_new_model_where_the_stage_stands(tango_ctx):
    stage = tango.Util.instance().get_device_by_name("asyncroscopy/stage/dose")
    request = {"session": "one", "seed": 3, "clock_s": 0.0, "position": [1e-8, 0.0, 0.0, 0.0, 0.0], "time_scale": 0.0}
    stage.move_on_clock(json.dumps(request))
    reply = json.loads(stage.frame_trajectory(json.dumps({"session": "two", "seed": 3, "clock_s": 0.0})))
    assert reply["trajectory"]["commanded"][0] == pytest.approx(1e-8)  # the model took up where the stage was left
    assert json.loads(stage.stage_moves())[-1]["commanded"][0] == pytest.approx(1e-8)
    assert len(json.loads(stage.stage_moves())) == 1
