import json

import pytest

from asyncroscopy.training import dataset
from asyncroscopy.training.recorder import TrainingRecorder


def _record(directory, prompt, answer):
    """Record one run with one model call, as the agent does, and return its id."""
    recorder = TrainingRecorder(directory)
    recorder.start_run(prompt)
    recorder.record_call([{"role": "user", "content": prompt}], {"role": "assistant", "content": answer},
                         tools=[{"type": "function", "function": {"name": "acquire_scanned_image"}}])
    return recorder.end_run(answer)


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_export_keeps_passed_runs_and_leaves_out_held_out_prompts(tmp_path):
    good = _record(tmp_path, "Take a HAADF image", "Saved key-1")
    held = _record(tmp_path, "Measure the drift", "0.1 nm/s")
    bad = _record(tmp_path, "Correct the aberrations", "I could not")
    _record(tmp_path, "Ungraded request", "...")
    dataset.add_grade(tmp_path, good, passed=True)
    dataset.add_grade(tmp_path, held, passed=True)
    dataset.add_grade(tmp_path, bad, passed=False)

    summary = dataset.export_training_set(tmp_path, tmp_path / "train.jsonl", holdout_prompts=["measure the  DRIFT"])

    assert (summary["runs_recorded"], summary["runs_passed"], summary["runs_held_out"]) == (4, 2, 1)
    [example] = _read(tmp_path / "train.jsonl")
    assert example["messages"] == [{"role": "user", "content": "Take a HAADF image"},
                                   {"role": "assistant", "content": "Saved key-1"}]
    assert example["tools"][0]["function"]["name"] == "acquire_scanned_image"


def test_newest_grade_wins(tmp_path):
    run_id = _record(tmp_path, "Take a HAADF image", "Saved")
    dataset.add_grade(tmp_path, run_id, passed=False)
    dataset.add_grade(tmp_path, run_id, passed=True)

    assert dataset.load_grades(tmp_path)[run_id]["passed"] is True


def test_grading_an_unknown_run_is_refused(tmp_path):
    with pytest.raises(ValueError, match="No recorded run"):
        dataset.add_grade(tmp_path, "typo", passed=True)


def test_a_damaged_line_is_skipped_not_fatal(tmp_path):
    run_id = _record(tmp_path, "Take a HAADF image", "Saved")
    with (tmp_path / "runs.jsonl").open("a") as handle:
        handle.write('{"id": "half-writ\n')

    with pytest.warns(RuntimeWarning, match="not valid JSON"):
        assert [run["id"] for run in dataset.load_runs(tmp_path)] == [run_id]


def test_command_line_grade_and_export(tmp_path, capsys):
    run_id = _record(tmp_path, "Take a HAADF image", "Saved")

    assert dataset.main(["grade", str(tmp_path), run_id, "--passed"]) == 0
    capsys.readouterr()
    assert dataset.main(["export", str(tmp_path), str(tmp_path / "out.jsonl")]) == 0

    assert json.loads(capsys.readouterr().out)["examples_written"] == 1
