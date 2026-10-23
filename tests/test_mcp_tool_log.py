"""MCP bridge call log, one-line errors and the repeat-failure warning."""

import json
import time

import pytest
import tango

from asyncroscopy.mcp.mcp_server import MCPServer, describe_error
from asyncroscopy.mcp.tool_log import ToolCallLog, read_tool_log

CMD_INFO = type(
    "CommandInfo",
    (),
    {
        "in_type": tango.CmdArgType.DevString,
        "out_type": tango.CmdArgType.DevString,
        "in_type_desc": ":param position: (not documented)",
        "out_type_desc": "",
    },
)


def _wrapper(monkeypatch, tmp_path, func, log=True):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer(
        "test", "localhost", 1234,
        blocked_classes=[], blocked_functions={}, data_device_address="",
        tool_log_path=str(tmp_path / "calls.jsonl") if log else None,
    )
    return server, server._create_wrapper(func, CMD_INFO, "move", "Stage")


def test_rows_round_trip_and_filter_by_time(tmp_path):
    log = ToolCallLog(tmp_path / "log.jsonl")
    t0 = time.time()
    log.record("Stage_move", {"x": 1e-9}, t0, 0.01, result="ok")
    log.record("Stage_move", {}, t0 + 10, 0.5, error=RuntimeError("boom"))
    rows = read_tool_log(tmp_path / "log.jsonl")
    assert [r["ok"] for r in rows] == [True, False]
    assert rows[1]["error"] == "RuntimeError: boom"
    assert len(read_tool_log(tmp_path / "log.jsonl", t0 + 5, t0 + 20)) == 1
    assert read_tool_log(tmp_path / "missing.jsonl") == []
    with (tmp_path / "log.jsonl").open("a") as handle:
        handle.write('{"ts": 1, "tool": "cut sho\n')
    assert len(read_tool_log(tmp_path / "log.jsonl")) == 2


def test_log_failure_never_fails_the_call(tmp_path):
    log = ToolCallLog(tmp_path / "log.jsonl")
    (tmp_path / "log.jsonl").mkdir()  # a directory cannot be opened for append
    with pytest.warns(RuntimeWarning):
        log.record("Stage_move", {"x": 1}, time.time(), 0.1, result="ok")


def test_wrapper_logs_successful_and_failed_calls(monkeypatch, tmp_path):
    def move(position):
        if position == "bad":
            raise ValueError("out of range")
        return f"moved {position}"

    _, wrapper = _wrapper(monkeypatch, tmp_path, move)
    assert wrapper("1e-9") == "moved 1e-9"
    with pytest.raises(RuntimeError):
        wrapper("bad")
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [(r["tool"], r["arguments"], r["ok"]) for r in rows] == [
        ("Stage_move", {"position": "1e-9"}, True),
        ("Stage_move", {"position": "bad"}, False),
    ]


def test_logging_is_off_by_default(monkeypatch, tmp_path):
    server, wrapper = _wrapper(monkeypatch, tmp_path, lambda position: "ok", log=False)
    assert wrapper("x") == "ok"
    assert server.tool_log is None and not (tmp_path / "calls.jsonl").exists()


def test_repeated_identical_failure_tells_the_model_to_stop(monkeypatch, tmp_path):
    def move(position):
        raise ValueError("stage fault")

    _, wrapper = _wrapper(monkeypatch, tmp_path, move)
    with pytest.raises(RuntimeError) as first:
        wrapper("1")
    assert "Do not call it again" not in str(first.value)
    with pytest.raises(RuntimeError) as second:
        wrapper("1")
    assert "failed 2 times in a row" in str(second.value)
    with pytest.raises(RuntimeError) as other:
        wrapper("2")  # different arguments start a new count
    assert "Do not call it again" not in str(other.value)


def test_devfailed_is_summarised_to_one_line():
    try:
        tango.Except.throw_exception("API_Reason", "first\n  line", "origin()")
    except tango.DevFailed as exc:
        assert describe_error(exc) == "API_Reason: first line"
    assert describe_error(ValueError("x")) == "ValueError: x"


def test_identical_failures_far_apart_are_not_a_loop(monkeypatch, tmp_path):
    import asyncroscopy.mcp.mcp_server as server_module

    def move(position):
        raise ValueError("stage fault")

    _, wrapper = _wrapper(monkeypatch, tmp_path, move)
    clock = [1000.0]
    monkeypatch.setattr(server_module.time, "monotonic", lambda: clock[0])
    with pytest.raises(RuntimeError):
        wrapper("1")
    clock[0] += server_module.REPEAT_FAILURE_WINDOW_S + 1
    with pytest.raises(RuntimeError) as later:
        wrapper("1")
    assert "Do not call it again" not in str(later.value)
