"""The integrity manifest notices a changed file and an altered, removed, inserted or moved manifest line."""

import json

import pytest

from asyncroscopy.data import integrity


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setenv(integrity.LOG_ENV, str(tmp_path / "integrity" / "manifest.jsonl"))
    monkeypatch.setenv(integrity.KEY_ENV, "11" * 32)
    monkeypatch.setenv("ASYNCROSCOPY_DOSE_LEDGER", str(tmp_path / "dose" / "ledger.jsonl"))
    integrity._index_cache.clear()
    return tmp_path / "integrity" / "manifest.jsonl"


def _data(tmp_path, name, content=b"frame bytes"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_a_sealed_unchanged_file_checks_and_verifies(tmp_path, manifest):
    path = _data(tmp_path, "a.h5")
    entry = integrity.seal(path, key="a", writer="test")
    assert integrity.check_file(path)["sha256"] == entry["sha256"]
    report = integrity.verify()
    assert report["ok"] and report["checked"] == 1 and report["head"] == entry["mac"]


def test_a_file_never_sealed_is_reported_as_unsealed_not_changed(tmp_path, manifest):
    integrity.seal(_data(tmp_path, "a.h5"))
    other = _data(tmp_path, "b.h5")
    assert integrity.check_file(other) is None
    report = integrity.verify(paths=[str(other)])
    assert report["ok"] and report["checked"] == 0 and report["unsealed"] == [str(other.resolve())]


def test_a_changed_file_is_refused(tmp_path, manifest):
    path = _data(tmp_path, "a.h5")
    integrity.seal(path, key="a")
    path.write_bytes(b"frame bytez")
    with pytest.raises(integrity.IntegrityError, match="changed since it was written"):
        integrity.check_file(path)
    report = integrity.verify()
    assert not report["ok"] and report["changed"] == ["a"]


def test_a_missing_file_is_reported(tmp_path, manifest):
    path = _data(tmp_path, "a.h5")
    integrity.seal(path, key="a")
    path.unlink()
    report = integrity.verify()
    assert not report["ok"] and report["missing"] == ["a"]


def _lines(manifest):
    return manifest.read_text(encoding="utf-8").splitlines()


def _write(manifest, lines):
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    integrity._index_cache.clear()


def _three(tmp_path):
    for name in ("a.h5", "b.h5", "c.h5"):
        integrity.seal(_data(tmp_path, name, name.encode()))


def test_an_edited_line_breaks_its_seal(tmp_path, manifest):
    _three(tmp_path)
    lines = _lines(manifest)
    entry = json.loads(lines[1])
    entry["size"] += 1
    lines[1] = json.dumps(entry)
    _write(manifest, lines)
    problems = integrity.verify_chain(manifest)["problems"]
    assert any("line 2 was changed" in p for p in problems)


@pytest.mark.parametrize("change", ["remove", "move", "insert"])
def test_a_removed_moved_or_inserted_line_breaks_the_chain(tmp_path, manifest, change):
    _three(tmp_path)
    lines = _lines(manifest)
    if change == "remove":
        del lines[1]
    elif change == "move":
        lines[0], lines[1] = lines[1], lines[0]
    else:
        lines.insert(1, lines[2])
    _write(manifest, lines)
    problems = integrity.verify_chain(manifest)["problems"]
    assert any("does not follow the line before it" in p for p in problems)


def test_a_line_cut_short_is_reported_and_the_next_seal_still_appends(tmp_path, manifest):
    _three(tmp_path)
    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write('{"path": "cut')
    integrity._index_cache.clear()
    assert any("not a record" in p for p in integrity.verify_chain(manifest)["problems"])
    later = _data(tmp_path, "d.h5", b"d")
    entry = integrity.seal(later, key="d")
    assert integrity.check_file(later)["sha256"] == entry["sha256"]  # not glued onto the cut line
    chain = integrity.verify_chain(manifest)
    assert chain["entries"] == 4 and chain["head"] == entry["mac"] and entry["seq"] == 5
    assert len(chain["problems"]) == 1  # only the cut line itself


def test_a_different_key_cannot_verify_the_chain(tmp_path, manifest, monkeypatch):
    _three(tmp_path)
    monkeypatch.setenv(integrity.KEY_ENV, "22" * 32)
    problems = integrity.verify_chain(manifest)["problems"]
    assert len(problems) == 3 and all("changed after it was written" in p for p in problems)


def test_the_key_file_is_made_once(tmp_path, monkeypatch):
    monkeypatch.delenv(integrity.KEY_ENV, raising=False)
    key_file = tmp_path / "key"
    monkeypatch.setenv(integrity.KEY_FILE_ENV, str(key_file))
    first = integrity._key()
    assert len(first) == 32 and integrity._key() == first


def test_a_sealed_file_is_not_sealed_again_as_something_else(tmp_path, manifest):
    path = _data(tmp_path, "a.h5", b"first")
    first = integrity.seal(path, key="a")
    assert integrity.seal(path, key="a") == first  # unchanged: the same entry, nothing appended
    path.write_bytes(b"second")
    with pytest.raises(integrity.IntegrityError, match="not sealed again"):
        integrity.seal(path, key="a")
    assert len(_lines(manifest)) == 1


def test_save_acquisition_seals_what_it_writes(tmp_path, manifest):
    import numpy as np

    from asyncroscopy.data.data_writer import save_acquisition

    class Device:
        instrument_id = "test/instrument"
        acquisition_save_directory = str(tmp_path)

    key = save_acquisition(Device(), None, "stem_image", ["haadf"], [np.arange(16, dtype=np.uint16).reshape(4, 4)])
    entry = integrity.check_file(key)
    assert entry is not None and entry["writer"] == "test/instrument"
    with open(key, "r+b") as handle:  # an edit after the fact
        handle.seek(-4, 2)
        handle.write(b"\x00\x01\x02\x03")
    integrity._index_cache.clear()
    with pytest.raises(integrity.IntegrityError):
        integrity.check_file(key)


def test_the_verify_command_says_what_is_wrong(tmp_path, manifest, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("verify_integrity", Path(__file__).resolve().parents[1] / "startup_scripts" / "verify_integrity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = _data(tmp_path, "a.h5")
    integrity.seal(path, key="a")
    assert module.main([]) == 0
    path.write_bytes(b"edited")
    integrity._index_cache.clear()
    assert module.main([]) == 1 and '"a"' in capsys.readouterr().out


def _ledger_frame(x_m):
    return {"stage_reported_m": [x_m, 0.0], "fov_m": 1e-6, "pixel_size_m": 1e-8, "dwell_time_s": 1e-6, "beam_current_pa": 10.0, "axes": "rows=+y, columns=+x"}


def test_the_dose_ledger_is_a_chain_and_an_edit_shows(tmp_path, manifest, monkeypatch):
    from asyncroscopy.data import dose_ledger

    ledger = tmp_path / "dose" / "ledger.jsonl"
    monkeypatch.setenv(dose_ledger.LEDGER_ENV, str(ledger))
    for index in range(3):
        assert dose_ledger.record_frame(_ledger_frame(index * 1e-5), "L", f"k{index}", "i")
    assert integrity.verify(logs=(ledger,))["ok"]
    assert dose_ledger.dose_at("L", 0.0, 0.0)["exposures_here"] == 1
    lines = _lines(ledger)
    entry = json.loads(lines[0])
    entry["dose_e_per_a2"] /= 10  # an exposure made to look smaller
    lines[0] = json.dumps(entry)
    _write(ledger, lines)
    report = integrity.verify(logs=(ledger,))
    assert not report["ok"] and any("ledger.jsonl line 1 was changed" in p for p in report["log_problems"])


def test_ledger_lines_from_before_the_chain_are_counted_not_failed(tmp_path, manifest, monkeypatch):
    from asyncroscopy.data import dose_ledger

    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(dose_ledger.LEDGER_ENV, str(ledger))
    old = {"at": 1.0, "sample_load_id": "L", "instrument_id": "i", "key": "old", "box_m": [-1e-6, 1e-6, -1e-6, 1e-6], "dose_e_per_a2": 5.0, "kind": "frame"}
    ledger.write_text(json.dumps(old) + "\n", encoding="utf-8")
    dose_ledger.record_frame(_ledger_frame(0.0), "L", "new", "i")
    report = integrity.verify(logs=(ledger,))
    assert report["ok"] and report["unsealed_before_chain"] == {str(ledger): 1}
    assert dose_ledger.dose_at("L", 0.0, 0.0)["exposures_here"] == 2
    with open(ledger, "a", encoding="utf-8") as handle:  # an unsealed line slipped in after the chain began
        handle.write(json.dumps({**old, "key": "late"}) + "\n")
    assert any("line 3 has no seal" in p for p in integrity.verify(logs=(ledger,))["log_problems"])


def test_stripping_every_seal_or_editing_a_legacy_line_shows(tmp_path, manifest, monkeypatch):
    from asyncroscopy.data import dose_ledger

    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(dose_ledger.LEDGER_ENV, str(ledger))
    old = {"at": 1.0, "sample_load_id": "L", "instrument_id": "i", "key": "old", "box_m": [-1e-6, 1e-6, -1e-6, 1e-6], "dose_e_per_a2": 5.0, "kind": "frame"}
    ledger.write_text(json.dumps(old) + "\n" + '{"cut', encoding="utf-8")  # a legacy line, then one cut short by a crash
    for index in range(2):
        dose_ledger.record_frame(_ledger_frame(index * 1e-5), "L", f"k{index}", "i")
    report = integrity.verify(logs=(ledger,))
    assert report["ok"] and report["unsealed_before_chain"] == {str(ledger): 2}
    assert report["log_heads"][str(ledger)] == json.loads(_lines(ledger)[-1])["mac"]

    lines = _lines(ledger)
    edited = list(lines)
    edited[0] = json.dumps({**old, "dose_e_per_a2": 0.5})
    _write(ledger, edited)
    assert any("not the unsealed lines its first seal recorded" in p for p in integrity.verify(logs=(ledger,))["log_problems"])

    stripped = [json.dumps({k: v for k, v in json.loads(line).items() if k not in ("mac", "seq", "prev", "legacy_lines", "legacy_sha256")}) if index >= 2 else line for index, line in enumerate(lines)]
    _write(ledger, stripped)
    report = integrity.verify(logs=(ledger,))
    assert report["log_heads"][str(ledger)] == ""  # every seal gone: the head no longer matches one copied elsewhere


def test_an_unreadable_key_keeps_the_dose_unsealed_and_verify_says_so(tmp_path, manifest, monkeypatch):
    from asyncroscopy.data import dose_ledger

    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(dose_ledger.LEDGER_ENV, str(ledger))
    dose_ledger.record_frame(_ledger_frame(0.0), "L", "sealed", "i")
    monkeypatch.setenv(integrity.KEY_ENV, "not hex")
    assert dose_ledger.record_frame(_ledger_frame(0.0), "L", "unsealed", "i")
    assert dose_ledger.dose_at("L", 0.0, 0.0)["exposures_here"] == 2
    monkeypatch.setenv(integrity.KEY_ENV, "11" * 32)
    assert any("line 2 has no seal" in p for p in integrity.verify(logs=(ledger,))["log_problems"])
