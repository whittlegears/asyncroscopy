"""The SAMPLE device owns which sample is loaded and its dose; the STAGE keeps travel and holder limits.

The dose ledger is one append-only file for every sample ever loaded, so ``dose_at`` answers from an index per
``sample_load_id`` and reads only the lines added since it last looked.
"""

import json

import pytest

from asyncroscopy.data.dose_ledger import INDEX_SUFFIX, LEDGER_ENV, NOT_COUNTED, dose_at, ledger_path
from asyncroscopy.instruments.common.sample import SAMPLE
from asyncroscopy.instruments.electron_microscope.hardware.stage import STAGE
from asyncroscopy.utils.commands import declared_hints


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    return path


def exposure(load, box=(0.0, 1.0, 0.0, 1.0), dose=2.0, **extra):
    return json.dumps({"sample_load_id": load, "box_m": list(box), "dose_e_per_a2": dose, **extra}) + "\n"


def append(path, *lines):
    with open(path, "a", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)


def test_the_sample_is_the_sample_devices_and_the_stage_has_only_its_limits():
    hints = declared_hints(SAMPLE)
    assert hints["load_sample"]["operator_only"] and hints["load_sample"]["risk"] == "high"
    assert hints["get_sample"]["risk"] == "low" and hints["get_dose"]["risk"] == "low"
    stage = declared_hints(STAGE)
    assert "set_holder" in stage and not {"get_sample", "load_sample", "get_dose"} & set(stage)


def test_the_dose_of_one_load_is_added_up_at_a_point(ledger):
    append(ledger, exposure("a"), exposure("b", dose=5.0), exposure("a", box=(2.0, 3.0, 0.0, 1.0)), exposure("a", dose=1.0, current_source="default"))
    reply = dose_at("a", 0.5, 0.5)
    assert ledger_path() == ledger and reply["exposures"] == 3 and reply["exposures_here"] == 2
    assert reply["dose_e_per_a2"] == pytest.approx(3.0) and "1 of these exposures" in reply["from_unmeasured_current"]
    assert dose_at("b") == {"sample_load_id": "b", "exposures": 1, "not_counted": NOT_COUNTED}
    assert dose_at("nobody", 0.5, 0.5)["exposures_here"] == 0


def test_a_query_reads_only_the_lines_added_since_the_last_one(ledger):
    append(ledger, *(exposure("old") for _ in range(50)), exposure("this"))
    assert dose_at("this")["exposures"] == 1
    # Overwritten in place, the indexed lines would be unreadable to a scan of the file; the index does not read them again.
    size = ledger.stat().st_size
    ledger.write_bytes(b"#" * (size - 1) + b"\n")
    append(ledger, exposure("this"), '{"cut short')
    assert dose_at("this")["exposures"] == 2
    assert dose_at("old")["exposures"] == 50
    append(ledger, ' by a crash"}\n', exposure("this"))
    assert dose_at("this")["exposures"] == 3  # the broken line is passed over


def test_the_index_is_rebuilt_from_a_shorter_file_or_when_damaged(ledger):
    append(ledger, exposure("a"), exposure("a"))
    assert dose_at("a")["exposures"] == 2
    ledger.write_text(exposure("a"), encoding="utf-8")
    assert dose_at("a")["exposures"] == 1
    ledger.with_name(ledger.name + INDEX_SUFFIX).write_bytes(b"not a database, but long enough to be read as one" * 50)
    assert dose_at("a")["exposures"] == 1


def test_get_dose_without_a_recorded_load_gives_no_dose_and_says_why(tmp_path):
    device = SAMPLE.__new__(SAMPLE)
    device._sample_path = tmp_path / "none.json"
    reply = json.loads(SAMPLE.get_dose(device, "{}"))
    assert reply["exposures"] is None and "load_sample" in reply["error"]
