"""The online Instrument Trace: every event of a run, live, in order, from every trace file.

The brief's test (docs/designs/concepts_brief.md, item 7): a subscriber attached during a
4-tile twin map receives exactly what the retrospective reader returns for that run, in
order, with no duplicates, each event within 2 s of being written. Also: several processes
appending to one file, file growth and partial lines, rotation, and the operator endpoint's
cursor tool.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import jsonschema
import pytest
import tango

from asyncroscopy.contracts import load
from asyncroscopy.mcp.mcp_server import MCPServer
from asyncroscopy.mcp.tool_log import TraceWriter, read_tool_log
from asyncroscopy.trace.stream import TraceStream, follow_trace, read_trace, trace_since
from asyncroscopy.trace import tail

ROOT = Path(__file__).resolve().parents[1]
LATENCY_S = 2.0
MAP_PLAN = {"box_m": [-1e-7, 1e-7, -1e-7, 1e-7], "fov_m": 1.2e-7, "overlap": 0.15, "imsize": 32, "dwell_s": 1e-6}
JSON_REQUEST = type("CommandInfo", (), {
    "in_type": tango.CmdArgType.DevString, "out_type": tango.CmdArgType.DevString,
    "in_type_desc": ':param request: JSON {"plan": {...}, "reservation_id": "..."}.', "out_type_desc": "",
})()


def rotate(src, dst, timeout_s: float = 5.0):
    """Rename like a log rotator on Windows must: a reader or writer may hold the file open for a moment."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return os.replace(src, dst)
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def wait_until(condition, timeout_s: float, what: str):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


class Subscriber:
    """Follows the trace on a thread, noting when each event arrived."""

    def __init__(self, sources, **filters):
        self.got: list[tuple[float, dict]] = []
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.stream = TraceStream(sources, **filters)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for event in self.stream.follow(timeout=300, poll_s=0.05, stop=self.stop):
                self.got.append((time.time(), event))
        except BaseException as exc:  # reported by close()
            self.error = exc

    @property
    def events(self) -> list[dict]:
        return [e for _, e in self.got]

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "subscriber did not stop"
        if self.error is not None:
            raise self.error


def assert_live_matches_retrospective(sub: Subscriber, sources, **filters):
    expected = read_trace(sources, **filters)
    wait_until(lambda: len(sub.events) >= len(expected), 10, f"{len(expected)} events live (have {len(sub.events)})")
    sub.close()
    assert sub.events == expected  # the same events, in the same order
    keys = [(e["file"], e["seq"]) for e in sub.events]
    assert len(set(keys)) == len(keys), "an event was delivered twice"
    late = [(round(at - e["written"], 3), e["action"]) for at, e in sub.got if at - e["written"] > LATENCY_S]
    assert not late, f"events later than {LATENCY_S} s: {late}"
    worst = max((at - e["written"] for at, e in sub.got), default=0.0)
    print(f"trace stream: {len(expected)} events live == retrospective, worst latency {worst:.3f} s")
    return expected


# ----------------------------------------------------------------------
# The brief's test: a 4-tile twin map, followed live
# ----------------------------------------------------------------------
def test_a_live_subscriber_gets_exactly_the_runs_events_in_order_within_2_s(tango_ctx, mapper_proxy, job_dir, tmp_path, monkeypatch):
    run = f"run-{os.getpid()}-{time.time_ns()}"
    headers = {"x-asyncroscopy-caller": "trace-test", "x-asyncroscopy-run": run}
    monkeypatch.setattr("fastmcp.server.dependencies.get_http_headers", lambda include_all=False: dict(headers))
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    mcp_trace = tmp_path / "mcp_agent.jsonl"
    server = MCPServer(
        "test", "localhost", 1234, blocked_classes=[], blocked_functions={}, data_device_address="asyncroscopy/data/default",
        verbose=False, endpoint_role="agent", trace_path=str(mcp_trace), clock_device=tango_ctx.get_device_access("asyncroscopy/digitaltwin/mapping"),
    )
    server._envelope_prefixes = {"Mapper"}
    server._devices = {"Mapper": mapper_proxy}
    server.tools = {"Mapper": {
        verb: server._create_wrapper(getattr(mapper_proxy, verb), JSON_REQUEST, verb, "Mapper", tool_prefix="Mapper") for verb in ("reserve", "start")
    }}
    sources = [job_dir / "trace", mcp_trace]

    sub = Subscriber(sources, run=run)  # attached before the map starts; it follows from the first line
    try:
        reply = server.run_job("Mapper", MAP_PLAN, wait=False)
        assert reply.get("estimate", {}).get("tiles") == 4, reply
        deadline = time.monotonic() + 120
        while (state := json.loads(mapper_proxy.job_status(json.dumps({"job_id": reply["job_id"]})))["state"]) not in ("done", "failed", "aborted"):
            assert time.monotonic() < deadline, "map did not finish"
            time.sleep(0.05)
        assert state == "done"
        mapper_proxy.release(json.dumps({"reservation_id": reply["reservation_id"]}))
        wait_until(lambda: any(e["action"] == "Mapper.done" for e in sub.events), 10, "the job's last event live")
    finally:
        sub.stop.set()
    events = assert_live_matches_retrospective(sub, sources, run=run)

    actions = [e["action"] for e in events]
    assert actions.count("Mapper.step") == 4
    for expected in ("Mapper_reserve", "Mapper.reserve", "Mapper_start", "Mapper.done"):
        assert expected in actions, actions
    assert actions.index("Mapper.reserve") < actions.index("Mapper.step") < actions.index("Mapper.done")
    schema = load("trace_event")
    for event in events:
        jsonschema.validate({k: v for k, v in event.items() if k != "file"}, schema)
        assert event["run"] == run and event["seq"] >= 1 and "ts" in event and "clock_s" in event
    jobs = [e for e in events if e["source"] == "job"]
    assert all(e["simulated"] is True and e["dry_run"] is False and e["clock"] == "instrument" for e in jobs)
    assert all(e["writer"] == "asyncroscopy/mapper/default" for e in jobs)
    mcp = [e for e in events if e["source"] == "mcp"]
    assert all(e["caller"] == "trace-test" and e["clock"] == "instrument" for e in mcp)
    # Each file's events arrive in their seq order; the job's clock_s is the twin's.
    for name in {e["file"] for e in events}:
        seqs = [e["seq"] for e in events if e["file"] == name]
        assert seqs == sorted(seqs)
    # The command line shows the same run.
    assert tail.main([str(s) for s in sources] + ["--run", run, "--no-follow", "--json"]) == 0


# ----------------------------------------------------------------------
# Writers: seq across processes, old lines still read
# ----------------------------------------------------------------------
WRITER = """
import sys, time
from asyncroscopy.mcp.tool_log import TraceWriter
w = TraceWriter(sys.argv[1], "script", writer=sys.argv[2])
for i in range(int(sys.argv[3])):
    w.record("tick", True, time.time(), run="multi", step=i)
"""


def test_processes_appending_one_file_get_one_rising_seq(tmp_path):
    path = tmp_path / "shared.jsonl"
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(path), f"w{i}", "60"], env=env) for i in range(3)]
    for proc in procs:
        assert proc.wait(timeout=120) == 0
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [e["seq"] for e in lines] == list(range(1, 181))  # file order is seq order, no gaps, no repeats
    assert all(b["written"] >= a["written"] for a, b in zip(lines, lines[1:]))
    for writer in ("w0", "w1", "w2"):  # each writer's own events stay in its order
        assert [e["step"] for e in lines if e["writer"] == writer] == list(range(60))
    assert len(read_trace(path, run="multi")) == 180


def test_old_lines_without_seq_are_still_read_and_numbering_continues(tmp_path):
    path = tmp_path / "jobs.jsonl"
    old = [{"schema_version": "1.1.0", "ts": 1000.0 + i, "source": "job", "action": "Old.step", "ok": True} for i in range(3)]
    path.write_text("".join(json.dumps(e) + "\n" for e in old), encoding="utf-8")
    TraceWriter(path, "job").record("New.step", True, time.time())
    events = read_trace(path)
    assert [e["action"] for e in events] == ["Old.step"] * 3 + ["New.step"]
    assert events[-1]["seq"] == 1 and "seq" not in events[0]
    assert len(read_tool_log(path)) == 4  # the one-file reader is unchanged
    for event in events:
        jsonschema.validate({k: v for k, v in event.items() if k != "file"}, load("trace_event"))
    # pyTEMlib's analysis trace (tool, run, seconds) reads alongside, its tool as the action.
    (tmp_path / "analysis.jsonl").write_text(json.dumps({"ts": 1001.5, "source": "analysis", "tool": "score_map", "run": "r", "ok": True}) + "\n", encoding="utf-8")
    merged = read_trace(tmp_path)
    assert [e["action"] for e in merged][:3] == ["Old.step", "Old.step", "score_map"]


# ----------------------------------------------------------------------
# Growth, partial lines and rotation, followed live
# ----------------------------------------------------------------------
def test_growth_and_a_line_cut_short_are_read_once(tmp_path):
    path = tmp_path / "jobs.jsonl"
    writer = TraceWriter(path, "job")
    stream = TraceStream(tmp_path, grace_s=0.0)
    writer.record("a", True, time.time())
    assert [e["action"] for e in stream.poll()] == ["a"]
    line = json.dumps({"ts": time.time(), "source": "script", "action": "b", "ok": True})
    with path.open("a", encoding="utf-8") as handle:  # half a line: a writer caught mid-append
        handle.write(line[:10])
    assert stream.poll() == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line[10:] + "\n")
    writer.record("c", True, time.time())
    assert [e["action"] for e in stream.poll()] == ["b", "c"]
    assert stream.poll() == []


def test_rotation_while_following_loses_and_repeats_nothing(tmp_path):
    jobs, mcp = tmp_path / "jobs.jsonl", tmp_path / "mcp_agent.jsonl"
    job_writer, mcp_writer = TraceWriter(jobs, "job"), TraceWriter(mcp, "mcp")
    sub = Subscriber(tmp_path, run="r")
    written = []

    def burst(n, tag):
        for i in range(n):
            written.append(job_writer.record(f"{tag}.job", True, time.time(), run="r", step=i))
            written.append(mcp_writer.record(f"{tag}.mcp", True, time.time(), run="r", step=i))
            mcp_writer.record("other", True, time.time(), run="someone-else")
            time.sleep(0.005)

    burst(5, "first")
    wait_until(lambda: len(sub.events) >= 4, 5, "the first events")
    burst(3, "unread")  # written, then the file is rotated before the follower reads them
    rotate(jobs, tmp_path / "jobs.jsonl.1")
    burst(4, "second")
    rotate(tmp_path / "jobs.jsonl.1", tmp_path / "jobs.jsonl.2")  # and rotated again: .1 -> .2, jobs -> .1
    rotate(jobs, tmp_path / "jobs.jsonl.1")
    burst(4, "third")
    events = assert_live_matches_retrospective(sub, tmp_path, run="r")
    assert len(events) == len(written) == 32
    assert [(e["action"], e["step"]) for e in events] == [(e["action"], e["step"]) for e in written]  # write order
    assert [e["seq"] for e in events if e["file"] == "jobs.jsonl"] == list(range(1, 17))  # seq survives rotation


def test_cursor_carries_on_across_rotation_with_none_twice(tmp_path):
    path = tmp_path / "jobs.jsonl"
    writer = TraceWriter(path, "job")
    for i in range(5):
        writer.record("x", True, time.time(), step=i, job_id="j1")
    time.sleep(0.6)  # older than the live grace period
    first = trace_since(tmp_path, limit=3, job="j1")
    assert [e["step"] for e in first["events"]] == [0, 1, 2] and first["more"]
    writer.record("x", True, time.time(), step=5, job_id="j1")
    rotate(path, tmp_path / "jobs.jsonl.1")
    writer.record("x", True, time.time(), step=6, job_id="j1")
    time.sleep(0.6)
    rest = trace_since(tmp_path, cursor=first["cursor"], job="j1")
    assert [e["step"] for e in rest["events"]] == [3, 4, 5, 6] and not rest["more"]
    assert trace_since(tmp_path, cursor=rest["cursor"], job="j1")["events"] == []
    with pytest.raises(ValueError):
        trace_since(tmp_path, cursor="not-a-cursor")


def test_follow_trace_filters_by_caller_device_and_since(tmp_path):
    writer = TraceWriter(tmp_path / "jobs.jsonl", "job", writer="asyncroscopy/mapper/default")
    writer.record("Mapper.step", True, time.time() - 100, caller="alice")
    time.sleep(0.05)  # Windows' clock ticks in ~15 ms steps: keep alice's write time clearly before the cut
    cut = time.time()
    time.sleep(0.05)
    writer.record("Mapper.step", True, time.time(), caller="bob")
    TraceWriter(tmp_path / "mcp.jsonl", "mcp").record("Tuner_start", True, time.time(), caller="bob")
    assert [e["action"] for e in read_trace(tmp_path, caller="bob")] == ["Mapper.step", "Tuner_start"]
    assert [e["caller"] for e in read_trace(tmp_path, device="asyncroscopy/mapper/default")] == ["alice", "bob"]
    assert [e["action"] for e in read_trace(tmp_path, device="Tuner")] == ["Tuner_start"]
    assert len(read_trace(tmp_path, since=cut)) == 2
    live = list(follow_trace(tmp_path, timeout=1.0, poll_s=0.05, caller="alice"))
    assert [e["caller"] for e in live] == ["alice"]


# ----------------------------------------------------------------------
# The operator endpoint's cursor tool
# ----------------------------------------------------------------------
class _NoDevices:
    value_string: list = []

    def get_device_exported(self, pattern):
        return self


def endpoint(role: str, trace_dir: Path) -> MCPServer:
    server = MCPServer(
        "test", "localhost", 0, blocked_classes=[], blocked_functions={}, data_device_address="asyncroscopy/data/default",
        verbose=False, endpoint_role=role, trace_path=str(trace_dir / f"mcp_{role}.jsonl"), database=_NoDevices(),
    )
    server.setup(print_summary=False)
    return server


def test_trace_since_is_on_the_operator_endpoint_only_and_pages_by_cursor(tmp_path):
    from fastmcp import Client

    trace_dir = tmp_path / "trace"
    operator, agent = endpoint("operator", trace_dir), endpoint("agent", trace_dir)

    async def names(server):
        return {t.name for t in await server.mcp.list_tools()}

    assert "trace_since" in asyncio.run(names(operator))
    assert "trace_since" not in asyncio.run(names(agent))

    writer = TraceWriter(trace_dir / "jobs.jsonl", "job")
    for i in range(5):
        writer.record("Mapper.step", True, time.time(), run="gui-run", step=i)
    writer.record("Mapper.step", True, time.time(), run="another", step=99)
    time.sleep(0.6)

    async def poll(cursor, limit):
        async with Client(operator.mcp) as client:
            reply = await client.call_tool("trace_since", {"cursor": cursor, "run": "gui-run", "limit": limit})
            return reply.structured_content

    page = asyncio.run(poll("", 2))
    assert [e["step"] for e in page["events"]] == [0, 1] and page["more"] is True
    page = asyncio.run(poll(page["cursor"], 10))
    assert [e["step"] for e in page["events"]] == [2, 3, 4] and page["more"] is False
    writer.record("Mapper.done", True, time.time(), run="gui-run")
    time.sleep(0.6)
    page = asyncio.run(poll(page["cursor"], 10))
    assert [e["action"] for e in page["events"]] == ["Mapper.done"]
    assert asyncio.run(poll(page["cursor"], 10))["events"] == []
