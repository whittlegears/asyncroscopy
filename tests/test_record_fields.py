"""Every dataset records its permanent id, instrument, kind, sample and sample load, and code version (brief 7.2)."""

import json
import time
from pathlib import Path

import h5py
import pytest
import tango

from asyncroscopy.data.record import code_version, current_sample, record_fields
from asyncroscopy.safety.policy import operator_only


def attrs_of(key: str) -> dict:
    with h5py.File(Path(key), "r") as h5:
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h5.attrs.items()}


def test_a_new_load_gets_a_new_id_and_survives_a_restart(mapping_stage_proxy):
    first = json.loads(mapping_stage_proxy.load_sample(json.dumps({"sample_id": "Au-on-C 7", "who": "alice"})))
    again = json.loads(mapping_stage_proxy.load_sample(json.dumps({"sample_id": "Au-on-C 7", "who": "alice"})))
    assert first["sample_id"] == again["sample_id"] == "Au-on-C 7"
    assert first["sample_load_id"] != again["sample_load_id"]  # the same sample, loaded twice, is two loads
    mapping_stage_proxy.Init()
    assert json.loads(mapping_stage_proxy.get_sample()) == again
    with pytest.raises(tango.DevFailed, match="sample_id"):
        mapping_stage_proxy.load_sample(json.dumps({"sample_id": " "}))
    assert operator_only("TestStage", "load_sample") and not operator_only("TestStage", "get_sample")


def test_an_image_records_the_loaded_sample_and_where_it_came_from(mapping_stage_proxy, mapping_twin_proxy):
    loaded = json.loads(mapping_stage_proxy.load_sample(json.dumps({"sample_id": "grid 12", "who": "bob"})))
    mapping_twin_proxy.set_timeout_millis(60_000)
    first, second = (attrs_of(mapping_twin_proxy.acquire_scanned_image(["haadf"])) for _ in range(2))
    for attrs in (first, second):
        assert attrs["instrument_id"] == "asyncroscopy/digitaltwin/mapping" and attrs["kind"] == "stem_image"
        assert attrs["sample_id"] == "grid 12" and attrs["sample_load_id"] == loaded["sample_load_id"]
        assert attrs["code_version"] == code_version() and attrs["code_version"].startswith("asyncroscopy ")
        assert abs(attrs["created"] - time.time()) < 120
    assert first["id"] != second["id"] and len(first["id"]) == 32


def test_a_map_run_keeps_its_id_and_refuses_to_resume_on_another_sample_load(mapper_proxy, mapping_stage_proxy):
    loaded = json.loads(mapping_stage_proxy.load_sample(json.dumps({"sample_id": "particles A"})))
    plan = {"box_m": [-1.2e-7, 1.2e-7, -8e-8, 8e-8], "fov_m": 8e-8, "overlap": 0.2, "imsize": 256, "dwell_s": 2e-6}
    call = lambda verb, body: json.loads(getattr(mapper_proxy, verb)(json.dumps(body)))

    def finish(job_id):
        deadline = time.time() + 120
        while (status := call("job_status", {"job_id": job_id}))["state"] not in ("done", "failed", "aborted") and time.time() < deadline:
            time.sleep(0.05)
        return status

    def stop_after_a_tile():
        started = call("start", {"reservation_id": reserve["reservation_id"]})
        while call("job_status", {"job_id": started["job_id"]})["progress"]["done"] < 1:
            time.sleep(0.01)
        call("abort", {"job_id": started["job_id"]})
        finish(started["job_id"])
        return call("results", {"job_id": started["job_id"]})["result_keys"]["run_file"]

    reserve = call("reserve", {"plan": plan})
    before = attrs_of(stop_after_a_tile())
    assert before["sample_load_id"] == loaded["sample_load_id"] and before["kind"] == "map_run"
    again = attrs_of(stop_after_a_tile())  # a resume on the same load keeps the run's id
    assert again["id"] == before["id"] and again["instrument_id"] == "asyncroscopy/mapper/default" and again["trace_offset"] >= 0
    mapping_stage_proxy.load_sample(json.dumps({"sample_id": "particles B"}))
    refused = finish(call("start", {"reservation_id": reserve["reservation_id"]})["job_id"])
    assert refused["state"] == "failed" and "reloaded" in refused["last_error"]["message"]
    call("release", {"reservation_id": reserve["reservation_id"]})


def test_record_fields_say_why_the_sample_is_unknown():
    class Silent:
        def get_sample(self):
            raise RuntimeError("Command get_sample not found\nmore")

    assert current_sample(None)["sample_error"].startswith("no STAGE device")
    unknown = record_fields("x/y/z", "stem_image", current_sample(Silent()))
    assert "sample_id" not in unknown and "sample_load_id" not in unknown
    assert unknown["sample_error"].endswith("Command get_sample not found")


def test_the_dose_ledger_adds_up_every_frame_that_covered_a_point(mapping_stage_proxy, mapping_twin_proxy, tmp_path, monkeypatch):
    from asyncroscopy.data.dose_ledger import dose_at, exposure_of
    from asyncroscopy.routines.dose import frame_dose

    loaded = json.loads(mapping_stage_proxy.load_sample(json.dumps({"sample_id": "dose test"})))
    mapping_twin_proxy.set_timeout_millis(60_000)
    before = json.loads(mapping_twin_proxy.get_dose(json.dumps({"x_m": 0.0, "y_m": 0.0})))
    assert before["sample_load_id"] == loaded["sample_load_id"] and before["frames"] == 0 and before["dose_e_per_a2"] == 0.0
    mapping_twin_proxy.unblank_beam()
    try:
        keys = [mapping_twin_proxy.acquire_scanned_image(["haadf"]) for _ in range(2)]
    finally:
        mapping_twin_proxy.blank_beam()
    attrs = []
    for key in keys:
        with h5py.File(Path(key), "r") as h5:
            attrs.append(dict(h5["image"][next(iter(h5["image"]))].attrs))
    expected = sum(exposure_of(a)["dose_e_per_a2"] for a in attrs)
    centre = json.loads(attrs[0]["stage_reported_m"]) if isinstance(attrs[0]["stage_reported_m"], str) else list(attrs[0]["stage_reported_m"])
    here = json.loads(mapping_twin_proxy.get_dose(json.dumps({"x_m": centre[0], "y_m": centre[1]})))
    assert here["frames"] == 2 and here["frames_here"] == 2 and here["dose_e_per_a2"] == pytest.approx(expected) and expected > 0
    away = json.loads(mapping_twin_proxy.get_dose(json.dumps({"x_m": centre[0] + 1e-3, "y_m": centre[1]})))
    assert away["frames_here"] == 0 and "spectra" in away["not_counted"]
    mapping_stage_proxy.load_sample(json.dumps({"sample_id": "dose test"}))  # a new load starts from nothing
    assert json.loads(mapping_twin_proxy.get_dose("{}"))["frames"] == 0
    with pytest.raises(tango.DevFailed, match="both x_m and y_m"):
        mapping_twin_proxy.get_dose(json.dumps({"x_m": 0.0}))
    a = attrs[0]
    assert exposure_of(a)["dose_e_per_a2"] == pytest.approx(frame_dose(float(a["beam_current_pa"]), float(a["dwell_time_s"]), float(a["fov_m"]), round(float(a["fov_m"]) / float(a["pixel_size_m"]))))
    assert exposure_of({k: v for k, v in a.items() if k != "beam_current_pa"}) is None
    assert dose_at("no such load")["frames"] == 0


def test_a_partial_scan_region_is_read_as_left_top_width_height():
    from asyncroscopy.data.dose_ledger import exposure_of

    attrs = {"stage_reported_m": [0.0, 0.0, 0, 0, 0], "fov_m": 1e-6, "pixel_size_m": 1e-8, "dwell_time_s": 1e-6, "beam_current_pa": 10.0}
    assert exposure_of(attrs)["box_m"] == pytest.approx([-5e-7, 5e-7, -5e-7, 5e-7])
    assert exposure_of({**attrs, "scan_region": [0.6, 0.0, 0.4, 1.0]})["box_m"] == pytest.approx([1e-7, 5e-7, -5e-7, 5e-7])
    assert exposure_of({**attrs, "scan_region": [0.25, 0.25, 0.5, 0.5]})["box_m"] == pytest.approx([-2.5e-7, 2.5e-7, -2.5e-7, 2.5e-7])
    assert exposure_of({**attrs, "scan_region": [0.0, 0.0, 0.0, 1.0]}) is None
