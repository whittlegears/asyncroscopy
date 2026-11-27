"""A run's progress (step, tiles done, newest image) can be read by another process: the runner keeps it in a file
beside the run record, and a server that is not driving the run reports it as still going while another process
holds the run's lock."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import time

from asyncroscopy.agents.workflows import runner
from asyncroscopy.agents.workflows.runner import run_workflow
from asyncroscopy.agents.workflows.server import WorkflowServer

from tests.agents.test_workflows import INPUTS, twin


def test_a_running_map_keeps_its_progress_in_a_file_and_removes_it_when_done(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROGRESS_EVERY_S", 0.01)
    path = tmp_path / "runs" / "watched.progress.json"
    seen = []

    def at_tile(n):
        if n == 2:  # two tiles taken: the file, as another process would read it now (this one reads its memory)
            seen.append(json.loads(path.read_text(encoding="utf-8")))
        return None

    instrument, analysis, _ = twin(on_acquire=at_tile)
    plain_call = instrument.call

    async def slow_call(name, arguments=None):  # a real endpoint takes time, so the progress task gets its turns
        await asyncio.sleep(0.02)
        return await plain_call(name, arguments)

    instrument.call = slow_call
    record = asyncio.run(run_workflow("map_grid", INPUTS, instrument=instrument, analysis=analysis, state_dir=tmp_path, run_id="watched"))
    assert record["state"] == "done"
    assert not path.exists()  # the record says how it ended
    assert seen
    assert seen[0]["step"] == "tile" and seen[0]["latest_image"] == "stem_image_HAADF_0002.h5"
    assert seen[0]["loop"]["node"] == "tile" and seen[0]["loop"]["done"] == 2 and seen[0]["loop"]["total"] == 4
    # Each tile taken so far, where it was taken and how wide, so an app can draw the map as it grows.
    tiles = seen[0]["tiles"]
    assert [t["key"] for t in tiles] == ["stem_image_HAADF_0001.h5", "stem_image_HAADF_0002.h5"] and seen[0]["tiles_total"] == 2
    assert all(t["fov_m"] == INPUTS["fov_m"] for t in tiles)
    assert (tiles[0]["x_m"], tiles[0]["y_m"]) != (tiles[1]["x_m"], tiles[1]["y_m"])


def test_the_file_is_read_only_for_a_run_this_process_is_not_driving(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "r1.progress.json").write_text(json.dumps({"step": "tile", "tool_calls": 7}), encoding="utf-8")
    assert runner.progress("r1") is None  # no state_dir: memory only, as before
    assert runner.progress("r1", tmp_path) == {"step": "tile", "tool_calls": 7}
    (tmp_path / "runs" / "r1.progress.json").write_text("{half", encoding="utf-8")
    assert runner.progress("r1", tmp_path) is None


def test_a_run_another_process_drives_is_still_going_with_its_progress(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "r2.json").write_text(json.dumps({"run_id": "r2", "workflow": "map_grid", "state": "running", "started": time.time()}),
                                  encoding="utf-8")
    tile = {"key": "stem_image_HAADF_0003.h5", "x_m": 1e-7, "y_m": 2e-7, "fov_m": 3e-7}
    (runs / "r2.progress.json").write_text(json.dumps({"step": "tile", "loop": {"node": "tile", "done": 3, "total": 9, "level": None},
                                                       "tiles": [tile], "tiles_total": 1}), encoding="utf-8")
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        from asyncroscopy.agents.workflows.runner import _run_lock
        with _run_lock(Path({str(tmp_path)!r}), "r2"):
            print("held", flush=True)
            sys.stdin.readline()
    """)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        server = WorkflowServer(state_dir=tmp_path)
        reply = asyncio.run(server._wait("r2", 0))
        assert reply["state"] == "running" and reply["progress"]["loop"]["done"] == 3
        # A model's reply leaves the tiles out; the app asks for them.
        assert "tiles" not in reply["progress"] and reply["progress"]["tiles_total"] == 1
        assert asyncio.run(server._wait("r2", 0, with_tiles=True))["progress"]["tiles"] == [tile]
    finally:
        holder.communicate("\n", timeout=30)
    reply = asyncio.run(WorkflowServer(state_dir=tmp_path)._wait("r2", 0))
    assert reply["state"] == "stopped"  # nobody drives it now: it was cut off
