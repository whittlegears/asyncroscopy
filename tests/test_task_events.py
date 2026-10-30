"""Every agent task writes task_start and task_end, so a benchmark can time it against the instrument trace."""

import asyncio
import json

import pytest

from asyncroscopy.agents import run as agent_run
from asyncroscopy.contracts import validate


def events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_task_is_bracketed_by_start_and_end(tmp_path, monkeypatch):
    path = tmp_path / "tasks.jsonl"
    monkeypatch.setenv(agent_run.TASK_TRACE_ENV, str(path))

    async def fake(name, prompt, model, progress=False):
        await asyncio.sleep(0.05)
        return "done", False

    monkeypatch.setattr(agent_run, "_run", fake)
    assert asyncio.run(agent_run.run_named_graph("general", "image the sample")) == "done"
    start, end = events(path)
    for event in (start, end):
        validate(event, "trace_event")
    assert start["action"] == "task_start" and start["arguments"] == {"graph": "general", "request": "image the sample"}
    assert end["action"] == "task_end" and end["ok"] and end["task_id"] == start["task_id"] and end["duration_s"] >= 0.05


def test_a_failed_or_crashed_task_says_so(tmp_path, monkeypatch):
    path = tmp_path / "tasks.jsonl"
    monkeypatch.setenv(agent_run.TASK_TRACE_ENV, str(path))

    async def failing(name, prompt, model, progress=False):
        return "map did not complete", True

    async def crashing(name, prompt, model, progress=False):
        raise RuntimeError("model server down")

    monkeypatch.setattr(agent_run, "_run", failing)
    asyncio.run(agent_run.run_named_graph("mapping", "map it"))
    monkeypatch.setattr(agent_run, "_run", crashing)
    with pytest.raises(RuntimeError):
        asyncio.run(agent_run.run_named_graph("team", "map it"))
    ends = [e for e in events(path) if e["action"] == "task_end"]
    assert [e["ok"] for e in ends] == [False, False]
    assert ends[0]["error"] == "the graph reported a failure" and "model server down" in ends[1]["error"]


def test_no_task_events_when_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv(agent_run.TASK_TRACE_ENV, "")

    async def fake(name, prompt, model, progress=False):
        return "done", False

    monkeypatch.setattr(agent_run, "_run", fake)
    monkeypatch.chdir(tmp_path)
    asyncio.run(agent_run.run_named_graph("general", "x"))
    assert not (tmp_path / "outputs").exists()
