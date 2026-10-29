"""Job bookkeeping that must not grow with a job's size: done ranges, the step file's tail, the trace,
and a run file that survives a crash mid-map."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from asyncroscopy.instruments.electron_microscope.mapper import Mapper, compact_grid, tile_at
from asyncroscopy.jobs.job_device import done_count, done_ranges_from, is_done, mark_done, read_last_steps


def test_done_ranges_merge_in_any_order_and_ignore_repeats():
    job = {"done_ranges": []}
    for index in [0, 1, 2, 10, 11, 5, 4, 3, 2, 12, 9, 6, 7, 8]:
        mark_done(job, index)
    assert job["done_ranges"] == [[0, 13]]
    assert done_count(job) == 13 and is_done(job, 12) and not is_done(job, 13)
    job = {"done_ranges": []}
    for index in range(0, 1_000_000, 2):
        mark_done(job, index)
    assert len(job["done_ranges"]) == 500_000 and done_count(job) == 500_000
    assert is_done(job, 999_998) and not is_done(job, 999_999)


def test_a_job_saved_with_a_list_of_done_steps_is_converted():
    assert done_ranges_from({"done": [3, 0, 1, 2, 7]}) == [[0, 4], [7, 8]]
    assert done_ranges_from({"done_ranges": [[0, 4]]}) == [[0, 4]]


def test_only_the_tail_of_a_long_step_file_is_read_and_a_torn_last_line_is_skipped(tmp_path):
    path = tmp_path / "job.steps.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for k in range(200_000):
            handle.write(json.dumps({"index": k, "record": {"k": k}}) + "\n")
        handle.write('{"index": 200000, "rec')  # cut short by a crash
    t0 = time.perf_counter()
    last = read_last_steps(path, 1000)
    elapsed = time.perf_counter() - t0
    assert list(last) == [str(k) for k in range(199_000, 200_000)]
    assert elapsed < 0.5  # 1 000 lines read, not 200 000


def test_the_run_file_keeps_every_tile_written_before_a_crash(tmp_path):
    path = tmp_path / "run.h5.partial"
    writer = f"""
import sys, time, numpy as np
sys.path.insert(0, {str(Path.cwd())!r})
from asyncroscopy.instruments.electron_microscope.mapper import write_tile
from pathlib import Path
plan = {{"fov_m": 1e-7, "imsize": 32}}
for k in range(100000):
    tile = {{"index": k, "row": 0, "col": k, "x": 0.0, "y": 0.0}}
    write_tile(Path({str(path)!r}), 100000, tile, plan, [0, 0, 0, 0, 0], time.time(), f"img{{k}}", {{"image/HAADF": (np.full((32, 32), k % 60000, np.uint16), {{"detector": "HAADF"}})}}, None, None)
    print(k, flush=True)
"""
    process = subprocess.Popen([sys.executable, "-c", writer], stdout=subprocess.PIPE, text=True)
    written = -1
    deadline = time.time() + 60
    while written < 20 and time.time() < deadline:
        written = int(process.stdout.readline())
    # Mid-write of the next tile, at worst. On Windows the venv's python.exe is a launcher with a
    # child interpreter: kill the whole tree, as a crash of the device server would.
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        process.kill()
    process.wait()
    with h5py.File(path, "r") as h5:
        done = np.flatnonzero(h5["written"][()])
        assert len(done) >= written + 1
        assert int(h5["tiles"][written][0, 0]) == written


def test_the_trace_copied_into_the_run_file_reads_only_this_jobs_part_of_the_shared_trace(tmp_path):
    trace = tmp_path / "jobs.jsonl"
    with trace.open("w", encoding="utf-8") as handle:
        for k in range(300_000):  # other jobs' history
            handle.write(json.dumps({"action": "Mapper.step", "ok": True, "run_id": "old", "step": k}) + "\n")
    offset = trace.stat().st_size
    with trace.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"action": "Mapper.reserve", "ok": True, "run_id": "job1"}) + "\n")
        for k in range(50):
            handle.write(json.dumps({"action": "Mapper.step", "ok": True, "run_id": "job1", "step": k}) + "\n")
        handle.write(json.dumps({"action": "Mapper.step", "ok": False, "run_id": "job1", "step": 50, "error": "stage"}) + "\n")
        handle.write(json.dumps({"action": "Mapper.step", "ok": True, "run_id": "other", "step": 0}) + "\n")
    run = tmp_path / "run.h5"
    with h5py.File(run, "w"):
        pass
    device = SimpleNamespace(trace_path=str(trace), _file=lambda job: run)
    t0 = time.perf_counter()
    Mapper._write_trace(device, {"reservation_id": "job1", "trace_offset": offset})
    elapsed = time.perf_counter() - t0
    with h5py.File(run, "r") as h5:
        events = [json.loads(line) for line in h5["trace"][()]]
        assert [e["action"] for e in events] == ["Mapper.reserve", "Mapper.step"] and events[1]["ok"] is False
        assert h5["trace"].attrs["trace_offset"] == offset
    assert elapsed < 0.5


def test_a_job_saved_with_a_step_list_still_runs_its_tiles():
    legacy = {"steps": [{"index": 0, "row": 0, "col": 0, "x": 1e-7, "y": 2e-7}], "resolved_plan": {}}
    assert Mapper.step_at(None, legacy, 0) == {"phase": "tiles", "index": 0, "row": 0, "col": 0, "x": 1e-7, "y": 2e-7}


def test_a_grid_of_a_million_tiles_is_a_few_numbers():
    xs = np.linspace(0, 1e-3, 1000)
    grid = compact_grid(xs, xs)
    assert len(json.dumps(grid)) < 200
    tile = tile_at(grid, "serpentine", 1999)  # second row, run backwards
    assert (tile["row"], tile["col"]) == (1, 0) and abs(tile["y"] - xs[1]) < 1e-15
