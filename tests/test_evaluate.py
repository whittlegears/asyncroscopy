"""benchmarks.evaluate: evaluators separate from the twin and the agents (no Tango, no stack).

Runs are built by hand: trace events like the ones the MCP server and the jobs write, and
images served by a loader, with a ground truth shaped like the twin's export_ground_truth().
"""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stdout

import numpy as np
import pytest

from benchmarks.evaluate import (
    EVALUATORS,
    Evaluator,
    RunData,
    Score,
    best_budget,
    composite,
    data_keys,
    feature_units,
    features,
    features_table,
    get_evaluator,
    keep_tuning,
    register_evaluator,
    score_event,
    score_run,
    stale_after_s,
    write_table,
)
from benchmarks.evaluate.__main__ import main as cli_main
from asyncroscopy.trace.writer import TraceWriter

PIXEL = 4e-10
SIZE = 128
FOV = PIXEL * SIZE


def particles(seed=0, n=12):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-0.4 * FOV, 0.4 * FOV, size=(n, 2))
    # keep them apart, so "found" is about blur, not crowding
    keep = [0]
    for i in range(1, n):
        if all(np.hypot(*(centres[i] - centres[j])) > 6e-9 for j in keep):
            keep.append(i)
    return centres[keep]


def render(centres, blur_px, seed=1, centre=(0.0, 0.0), signal=40.0):
    """Gaussian blobs at the particle centres (rows = +y, columns = +x), blurred, with Poisson noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    x = centre[0] - FOV / 2 + (xx + 0.5) * PIXEL
    y = centre[1] - FOV / 2 + (yy + 0.5) * PIXEL
    width = math.hypot(1.5 * PIXEL, blur_px * PIXEL)
    image = np.full((SIZE, SIZE), 20.0)
    for cx, cy in centres:
        image += signal * (1.5 * PIXEL / width) ** 2 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2))
    return rng.poisson(image).astype(np.float64)


def attrs(index, centre=(0.0, 0.0), dwell=2e-6):
    return {"acquisition_index": index, "pixel_size_m": PIXEL, "dwell_time_s": dwell, "beam_current_pa": 50.0, "stage_reported_m": [*centre, 0.0, 0.0, 0.0]}


def event(key, ts, clock, duration=0.5, action="DigitalTwin_acquire_scanned_image", **extra):
    return {"source": "mcp", "action": action, "ok": True, "ts": ts, "duration_s": duration, "result": key, "result_key": key,
            "clock_s": clock, "clock": "instrument", "run": "r1", **extra}


def truth_doc(centres, records):
    return {
        "particles": [{"id": i, "center_m": [float(x), float(y)], "semi_axes_m": [1.6e-9, 1.6e-9, 0.6e-9]} for i, (x, y) in enumerate(centres)],
        "acquisitions": records,
    }


@pytest.fixture
def run_with_truth():
    centres = particles()
    store = {
        "sharp.h5": {"image/HAADF": (render(centres, 0.0, seed=1), attrs(1))},
        "sharp_again.h5": {"image/HAADF": (render(centres, 0.0, seed=2), attrs(2))},
        "blurred.h5": {"image/HAADF": (render(centres, 12.0, seed=3), attrs(3))},
    }
    records = [{"index": i, "center_true_m": [0.0, 0.0]} for i in (1, 2, 3)]
    events = [event("sharp.h5", 100.0, 10.0), event("sharp_again.h5", 101.0, 11.0), event("blurred.h5", 102.0, 12.0)]
    truth = truth_doc(centres, records)
    truth["focus_by_acquisition"] = {"1": {"c1_m": 2e-9, "a1_abs_m": 0.0}, "2": {"c1_m": -3e-9, "a1_abs_m": 1e-9}, "3": {"c1_m": 150e-9, "a1_abs_m": 0.0}}
    return RunData("r1", events, truth=truth, loader=lambda key: store[key]), centres


# ----------------------------------------------------------------------
def test_builtins_are_registered_with_units_and_directions():
    for name in ("information", "resolution", "dose", "time"):
        cls = EVALUATORS[name]
        assert cls.objectives and all(better in ("max", "min") for _units, better in cls.objectives.values())
    with pytest.raises(ValueError, match="registered"):
        get_evaluator("no_such_evaluator")
    with pytest.raises(ValueError, match="no option"):
        get_evaluator("dose", not_an_option=1)


def test_data_keys_come_from_result_key_and_json_results():
    assert data_keys({"result_key": "a.h5", "result": json.dumps({"key": "a.h5", "spectrum_key": "s.h5", "keys": ["b"]})}) == ["a.h5", "s.h5", "b"]
    assert data_keys({"result": "not json"}) == []


def test_information_counts_unique_ground_truth_features_and_blur_loses_them(run_with_truth):
    data, centres = run_with_truth
    per_event = [score_event(e, data, ["information"])["information"] for e in data.events]
    sharp, blurred = per_event[0]["features_found"]["value"], per_event[2]["features_found"]["value"]
    assert sharp >= 0.8 * len(centres)
    assert blurred < sharp
    assert per_event[0]["recall"]["value"] == pytest.approx(sharp / len(centres))
    run = score_run(data, ["information"])["information"]
    # The same particles imaged twice count once.
    assert run["features_found"]["value"] <= len(centres)
    assert run["features_found"]["value"] >= sharp
    assert "ground-truth" in run["features_found"]["reason"]
    assert run["features_per_s"]["value"] == pytest.approx(run["features_found"]["value"] / 2.0)  # clock 10 -> 12 s
    assert run["images"]["value"] == 3
    # Only the two sharp frames were taken in focus (2 and 3 nm; the blurred one at 150 nm was not).
    in_focus_only = RunData("r1", data.events[:2], truth=data.truth, loader=data.loader)
    assert run["features_in_focus"]["value"] == score_run(in_focus_only, ["information"])["information"]["features_found"]["value"]


def test_information_without_truth_falls_back_to_the_pytemlib_proxy(run_with_truth):
    data, _ = run_with_truth
    proxy = RunData("r1", data.events, loader=data.loader)
    run = score_run(proxy, ["information"])["information"]
    assert "proxy" in run["features_found"]["reason"]
    assert run["features_found"]["value"] >= 0
    assert "recall" not in run


def test_resolution_uses_the_recorded_focus_error_else_the_baseline_score(run_with_truth):
    data, _ = run_with_truth
    run = score_run(data, ["resolution"])["resolution"]
    assert run["in_focus_fraction"]["value"] == pytest.approx(2 / 3)
    assert run["focus_error_m"]["value"] == pytest.approx(math.hypot(3e-9, 1e-9))
    proxy = score_run(RunData("r1", data.events, loader=data.loader), ["resolution"])["resolution"]
    assert 0.0 <= proxy["sharpness"]["value"] <= 1.0
    assert "focus_error_m" not in proxy


def test_dose_accumulates_where_frames_overlap_and_counts_job_dose():
    image = np.ones((SIZE, SIZE))
    store = {"a.h5": {"i": (image, attrs(1, (0.0, 0.0)))}, "b.h5": {"i": (image, attrs(2, (FOV / 2, 0.0)))}}
    tuner = {"source": "job", "action": "Tuner.step", "ok": True, "ts": 99.0, "result": json.dumps({"dose_e_per_a2": 120.0})}
    data = RunData("r", [tuner, event("a.h5", 100.0, 1.0), event("b.h5", 101.0, 2.0)], loader=lambda key: store[key])
    frame = 50e-12 * 2e-6 / 1.602176634e-19 / (PIXEL / 1e-10) ** 2
    dose = get_evaluator("dose", threshold_e_per_a2=1.5 * frame).score_run(data)
    assert dose["max_dose_e_per_a2"].value == pytest.approx(2 * frame)
    assert dose["over_threshold_fraction"].value == pytest.approx(1 / 3)  # half of each frame overlaps: 2 of 6 quarter-columns
    assert dose["job_dose_e_per_a2"].value == pytest.approx(120.0)
    assert dose["exposed_area_m2"].value == pytest.approx(1.5 * FOV * FOV)
    assert score_event(tuner, data, ["dose"])["dose"]["job_dose_e_per_a2"]["value"] == 120.0


def test_time_splits_wall_time_into_busy_and_idle():
    events = [
        {"ts": 0.0, "duration_s": 2.0, "clock_s": 5.0, "clock": "instrument"},
        {"ts": 1.0, "duration_s": 2.0},  # overlaps the first: busy 0-3
        {"ts": 10.0, "duration_s": 1.0, "clock_s": 20.0, "clock": "instrument"},
    ]
    scores = get_evaluator("time").score_run(RunData("r", events))
    assert scores["wall_s"].value == pytest.approx(11.0)
    assert scores["busy_s"].value == pytest.approx(4.0)
    assert scores["idle_s"].value == pytest.approx(7.0)
    assert scores["clock_s"].value == pytest.approx(15.0)


def test_a_new_evaluator_in_a_few_lines_and_a_failing_one_does_not_break_the_others():
    @register_evaluator
    class ToolCalls(Evaluator):
        name = "tool_calls_test"
        objectives = {"calls": ("calls", "min")}

        def score_run(self, data):
            return {"calls": Score(sum(e.get("source") == "mcp" for e in data.events), "calls", "MCP tool calls")}

    @register_evaluator
    class Broken(Evaluator):
        name = "broken_test"

        def score_run(self, data):
            raise RuntimeError("boom")

    try:
        out = score_run(RunData("r", [{"source": "mcp"}, {"source": "job"}]), ["tool_calls_test", "broken_test", "time"])
        assert out["tool_calls_test"]["calls"]["value"] == 1
        assert out["broken_test"]["error"]["value"] is None and "boom" in out["broken_test"]["error"]["reason"]
        assert "wall_s" in out["time"]
    finally:
        EVALUATORS.pop("tool_calls_test", None)
        EVALUATORS.pop("broken_test", None)


def test_composite_is_a_weighted_sum_and_names_what_it_left_out(run_with_truth):
    data, _ = run_with_truth
    reward_cls = composite("reward_test", {"information.features_found": 1.0, "dose.max_dose_e_per_a2": -0.01, "information.not_an_objective": 5.0})
    reward = reward_cls().score_run(data)["reward"]
    info = score_run(data, ["information"])["information"]["features_found"]["value"]
    dose = score_run(data, ["dose"])["dose"]["max_dose_e_per_a2"]["value"]
    assert reward.value == pytest.approx(info - 0.01 * dose)
    assert "not_an_objective" in reward.reason
    assert composite("strict_test", {"information.not_an_objective": 1.0}, strict=True)().score_run(data)["reward"].value is None


def test_features_are_one_flat_row_per_run(run_with_truth, tmp_path):
    data, _ = run_with_truth
    row = features(data)
    assert row["run.id"] == "r1" and row["run.events"] == 3 and row["run.acquire_calls"] == 3
    assert "information.features_found" in row and "dose.max_dose_e_per_a2" in row and "time.idle_s" in row
    assert all(not isinstance(v, (dict, list)) for v in row.values())
    assert feature_units()["dose.max_dose_e_per_a2"] == "e/A^2"
    path = write_table(features_table([data, RunData("empty", [])]), tmp_path / "runs.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and lines[0].startswith("run.id")


def test_stopping_rule():
    # Tuning for 10 s triples the rate for the 90 s left: worth it.
    assert keep_tuning(1.0, 3.0, 10.0, 100.0).keep_tuning
    # The same gain, but drift makes it stale after 3 s: not worth 10 s of tuning.
    stale = stale_after_s(depth_of_field_m=20e-9, focus_drift_m_per_s=5e-9, residual_m=5e-9)
    assert stale == pytest.approx(3.0)
    decision = keep_tuning(1.0, 3.0, 10.0, 100.0, stale)
    assert not decision.keep_tuning and "stale" in decision.reason
    # Too little time left to use the better focus.
    assert not keep_tuning(1.0, 3.0, 10.0, 12.0).keep_tuning
    assert stale_after_s(20e-9, 0.0) == math.inf
    assert best_budget([{"budget": "a", "tune_s": 0, "information": 5}, {"budget": "b", "tune_s": 4, "information": 9}, {"budget": "c", "tune_s": 9, "information": 9}])["budget"] == "b"


def test_cli_reads_a_run_from_trace_files(tmp_path):
    trace = TraceWriter(tmp_path / "trace" / "mcp_agent.jsonl", "mcp", defaults={"run": "cli-run"})
    trace.record("DigitalTwin_acquire_scanned_image", True, 1000.0, duration_s=1.0, result="missing.h5", result_key="missing.h5", clock_s=3.0)
    trace.record("DigitalTwin_get_defocus", True, 1002.0, duration_s=0.1, clock_s=4.0)
    TraceWriter(tmp_path / "trace" / "mcp_agent.jsonl", "mcp", defaults={"run": "other"}).record("x", True, 1003.0)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert cli_main(["--run", "cli-run", "--trace-dir", str(tmp_path / "trace"), "--evaluators", "time,dose", "--json", "-"]) == 0
    out = json.loads(buffer.getvalue())
    assert out["events"] == 2
    assert out["scores"]["time"]["clock_s"]["value"] == pytest.approx(1.0)
    assert "missing.h5" in out["unreadable_keys"]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert cli_main(["--run", "cli-run", "--trace-dir", str(tmp_path / "trace"), "--weights", "time.busy_s=2"]) == 0
    assert "reward" in buffer.getvalue()
    EVALUATORS.pop("reward", None)
    assert cli_main(["--run", "nobody", "--trace-dir", str(tmp_path / "trace")]) == 1


def test_workflow_benchmark_shows_evaluator_columns_without_changing_verdicts():
    from benchmarks import run_workflows

    row = {"ok": False, "seconds": 3.0, "checks": {"moved": {"ok": False}}, "scores": {"time": {"idle_s": {"value": 4.0}, "clock_s": {"value": 9.0}}}}
    results = {"model": "m", "stack": "twin", "workflows": {"w/general": {"passed": 0, "of": 1, "rows": [row]}}, "passed": 0, "of": 1,
               "sleepwalking_calls": 0, "score_columns": ["time.clock_s", "time.idle_s"]}
    table = run_workflows.markdown(results)
    assert "| time.clock_s | time.idle_s |" in table
    assert "| w | general | 0/1 | 3 | moved | 9 | 4 |" in table
    # Results written before the columns existed still render as before.
    del results["score_columns"]
    assert "| w | general | 0/1 | 3 | moved |\n" in run_workflows.markdown(results)
