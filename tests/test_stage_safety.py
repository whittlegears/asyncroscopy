"""Stage safety: moves outside the stage's travel are refused with the limits in the message, and no
command holds the microscope for long (a long move returns at once and stage_moving says when it ends).

``slow_stage_twin_proxy`` is a twin whose stage moves take real time (1 um per second) and that has no
STAGE device, so its own travel (stage_travel_m, 10 um) applies; the mapping twin uses the STAGE
device's limits (+/-1 mm by default).
"""

import threading
import time
from types import SimpleNamespace

import pytest
import tango

from asyncroscopy.instruments.electron_microscope.digital_twin import MOVE_BLOCK_S
from asyncroscopy.instruments.electron_microscope.mapper import Mapper


def wait_until_still(twin, timeout=15.0):
    deadline = time.monotonic() + timeout
    while twin.read_attribute("stage_moving").value:
        assert time.monotonic() < deadline, "the stage never stopped"
        time.sleep(0.1)


def timed(fn, *args):
    t0 = time.monotonic()
    fn(*args)
    return time.monotonic() - t0


@pytest.fixture
def slow_twin(slow_stage_twin_proxy):
    wait_until_still(slow_stage_twin_proxy)
    slow_stage_twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    wait_until_still(slow_stage_twin_proxy)
    return slow_stage_twin_proxy


def test_a_move_outside_the_stage_limits_is_refused_with_the_limits(mapping_twin_proxy):
    before = list(mapping_twin_proxy.get_stage())
    with pytest.raises(tango.DevFailed, match=r"stage x = 0.002 is outside the stage limits \[-0.001, 0.001\]"):
        mapping_twin_proxy.move_stage([2e-3, 0.0, 0.0, 0.0, 0.0])
    assert list(mapping_twin_proxy.get_stage()) == pytest.approx(before)


def test_the_stage_device_refuses_writes_outside_its_limits(mapping_stage_proxy):
    config = mapping_stage_proxy.get_attribute_config("x")
    assert (float(config.min_value), float(config.max_value)) == (-1e-3, 1e-3)
    with pytest.raises(tango.DevFailed):
        mapping_stage_proxy.x = 2e-3
    with pytest.raises(tango.DevFailed, match=r"stage alpha = 50 is outside the stage limits \[-35.0, 35.0\]"):
        mapping_stage_proxy.position = [0.0, 0.0, 0.0, 50.0, 0.0]


def test_without_a_stage_device_the_twin_keeps_to_its_own_travel(slow_twin):
    with pytest.raises(tango.DevFailed, match=r"stage y = -2e-05 is outside the stage limits \[-1e-05, 1e-05\]"):
        slow_twin.move_stage([0.0, -2e-5, 0.0, 0.0, 0.0])


def test_a_long_move_returns_at_once_and_blank_beam_still_works(slow_twin):
    # 5 um at 1 um/s: five real seconds, longer than any command may hold the device.
    t0 = time.monotonic()
    assert timed(slow_twin.move_stage, [5e-6, 0.0, 0.0, 0.0, 0.0]) < 1.0
    assert slow_twin.read_attribute("stage_moving").value
    assert timed(slow_twin.blank_beam) < 1.0
    with pytest.raises(tango.DevFailed, match="still moving"):
        slow_twin.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(tango.DevFailed, match="still moving"):
        slow_twin.acquire_scanned_image(["haadf"])
    wait_until_still(slow_twin)
    assert 4.5 < time.monotonic() - t0 < 7.0
    assert list(slow_twin.get_stage())[0] == pytest.approx(5e-6, abs=1e-7)


def test_a_short_move_holds_the_device_no_longer_than_the_limit(slow_twin, tango_ctx):
    # 1.5 um: 1.5 s, short enough to wait for, so other clients queue behind it for at most that long.
    mover = tango.DeviceProxy(tango_ctx.get_device_access("asyncroscopy/digitaltwin/slowstage"))
    thread = threading.Thread(target=mover.move_stage, args=([1.5e-6, 0.0, 0.0, 0.0, 0.0],))
    thread.start()
    time.sleep(0.2)
    held = timed(slow_twin.blank_beam)
    thread.join()
    assert held < MOVE_BLOCK_S + 0.5
    assert not slow_twin.read_attribute("stage_moving").value


def test_the_mapper_waits_for_a_long_move_and_stops_waiting_on_abort(slow_twin):
    slow_twin.move_stage([4e-6, 0.0, 0.0, 0.0, 0.0])
    mapper = SimpleNamespace(aborting=lambda job: False, stage_timeout_s=600.0)
    t0 = time.monotonic()
    assert Mapper._wait_for_stage(mapper, {}, slow_twin) is True
    assert time.monotonic() - t0 > 2.5 and not slow_twin.read_attribute("stage_moving").value
    slow_twin.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    aborted = SimpleNamespace(aborting=lambda job: True, stage_timeout_s=600.0)
    assert Mapper._wait_for_stage(aborted, {}, slow_twin) is False


def test_a_stage_that_never_arrives_fails_the_step_instead_of_waiting_forever():
    stuck = SimpleNamespace(read_attribute=lambda name: SimpleNamespace(value=True))
    mapper = SimpleNamespace(aborting=lambda job: False, stage_timeout_s=0.3)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="stage_timeout_s"):
        Mapper._wait_for_stage(mapper, {}, stuck)
    assert time.monotonic() - t0 < 2.0
