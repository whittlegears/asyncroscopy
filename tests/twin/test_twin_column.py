"""The twin's column: valves, fluorescent screen and the beam current measured on it, and the map that will not
start while the beam cannot reach the detector."""

import json
import time

import pytest
import tango

from asyncroscopy.routines.beam import beam_path_blocked
from tests.twin.test_twin_device import acquire

PLAN = {"box_m": [-1.0e-7, 1.0e-7, -1.0e-7, 1.0e-7], "fov_m": 1.0e-7, "overlap": 0.0, "imsize": 32, "dwell_s": 1e-6, "stitch": False}


@pytest.fixture(autouse=True)
def column_ready(mapping_twin_proxy):
    """Every test starts and ends with the beam on the detector, as the other tests expect."""
    def ready():
        mapping_twin_proxy.set_column_valves("open")
        mapping_twin_proxy.set_screen("retracted")
        mapping_twin_proxy.unblank_beam()
    ready()
    yield
    ready()


def call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def haadf(twin, scan):
    return acquire(twin, scan)["HAADF"][0].astype(float)


def test_the_rule_reads_what_both_backends_report():
    assert beam_path_blocked({"column_valves": "Opened", "screen": "Retracted"}) is None
    assert beam_path_blocked({}) is None  # a backend that reports neither is not judged
    assert "column valves" in beam_path_blocked({"column_valves": "Closed"})
    assert "screen" in beam_path_blocked({"column_valves": "Opened", "screen": "Inserted"})


def test_closed_valves_give_a_black_image(mapping_twin_proxy, mapping_scan_proxy):
    lit = haadf(mapping_twin_proxy, mapping_scan_proxy)
    mapping_twin_proxy.set_column_valves("close")
    assert json.loads(mapping_twin_proxy.get_parameters())["column_valves"] == "Closed"
    dark = haadf(mapping_twin_proxy, mapping_scan_proxy)
    assert dark.var() < 0.5 * lit.var() and dark.max() < lit.max()  # only the detector's offset and read noise


def test_an_inserted_screen_takes_the_beam_from_the_detector_and_measures_it(mapping_twin_proxy, mapping_scan_proxy):
    lit = haadf(mapping_twin_proxy, mapping_scan_proxy)
    assert mapping_twin_proxy.get_screen_current() == 0.0  # the beam does not reach a retracted screen
    mapping_twin_proxy.set_screen("inserted")
    assert json.loads(mapping_twin_proxy.get_parameters())["screen"] == "Inserted"
    assert mapping_twin_proxy.get_screen_current() == pytest.approx(json.loads(mapping_twin_proxy.get_parameters())["beam_current_pa"])
    assert haadf(mapping_twin_proxy, mapping_scan_proxy).var() < 0.5 * lit.var()
    mapping_twin_proxy.blank_beam()
    assert mapping_twin_proxy.get_screen_current() == 0.0


def test_autofocus_is_refused_without_a_beam(mapping_twin_proxy):
    mapping_twin_proxy.set_column_valves("close")
    with pytest.raises(tango.DevFailed, match="no beam reaches the detector"):
        mapping_twin_proxy.auto_focus()


def test_bad_words_are_refused(mapping_twin_proxy):
    for command, word in (("set_column_valves", "ajar"), ("set_screen", "halfway")):
        with pytest.raises(tango.DevFailed):
            getattr(mapping_twin_proxy, command)(word)


def test_a_map_does_not_start_while_the_beam_is_blocked(mapper_proxy, mapping_twin_proxy):
    def run() -> dict:
        reserve = call(mapper_proxy, "reserve", {"plan": PLAN})
        started = call(mapper_proxy, "start", {"reservation_id": reserve["reservation_id"]})
        status = started
        deadline = time.time() + 60
        while "job_id" in started and time.time() < deadline:
            status = call(mapper_proxy, "job_status", {"job_id": started["job_id"]})
            if status["state"] in ("done", "failed", "aborted"):
                break
            time.sleep(0.05)
        call(mapper_proxy, "release", {"reservation_id": reserve["reservation_id"]})
        return status

    acquisitions = lambda: len(json.loads(mapping_twin_proxy.export_ground_truth())["acquisitions"])  # noqa: E731
    before = acquisitions()
    for block, unblock, named in ((("set_column_valves", "close"), ("set_column_valves", "open"), "column valves"), (("set_screen", "inserted"), ("set_screen", "retracted"), "screen")):
        getattr(mapping_twin_proxy, block[0])(block[1])
        refused = run()
        assert named in json.dumps(refused) and refused.get("state") != "done", refused
        assert acquisitions() == before  # nothing was imaged
        getattr(mapping_twin_proxy, unblock[0])(unblock[1])
    assert run()["state"] == "done"
