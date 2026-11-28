"""Settings that are Tango attributes, read and written through the bridge's allowlist."""

from __future__ import annotations

import pytest

from asyncroscopy.mcp.approvals import parse_held
from asyncroscopy.mcp.mcp_server import MCPServer


def _server(monkeypatch, tango_ctx, tmp_path=None, role="agent", rules=None):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer(
        "t", "localhost", 1234, blocked_functions={}, blocked_classes=[], data_device_address="d", verbose=False,
        attributes={"SCAN": ["dwell_time", "imsize"]},
        needs_approval=rules or {}, approvals_db=str(tmp_path / "held.sqlite") if tmp_path else None, role=role,
    )
    server._devices_by_class = {"SCAN": [tango_ctx.get_device_access("asyncroscopy/scan/default")]}
    return server


def test_read_and_write_an_allowed_setting(monkeypatch, tango_ctx, scan_proxy):
    server = _server(monkeypatch, tango_ctx)
    before = scan_proxy.imsize
    try:
        assert server.write_attribute("SCAN", "imsize", 96) == 96
        assert server.read_attribute("SCAN", "imsize") == 96 == scan_proxy.imsize
        assert server.read_attribute("SCAN", "dwell_time") == pytest.approx(scan_proxy.dwell_time)
    finally:
        scan_proxy.imsize = before


def test_unlisted_attributes_and_classes_are_refused(monkeypatch, tango_ctx):
    server = _server(monkeypatch, tango_ctx)
    with pytest.raises(ValueError, match="not offered"):
        server.write_attribute("SCAN", "scan_region", [0, 0, 1, 1])
    with pytest.raises(ValueError, match="not offered"):
        server.read_attribute("CAMERA", "exposure_time")


def test_a_listed_write_waits_for_the_operator(monkeypatch, tango_ctx, scan_proxy, tmp_path):
    rules = {"SCAN": ["write_imsize"]}
    agent = _server(monkeypatch, tango_ctx, tmp_path, "agent", rules)
    operator = _server(monkeypatch, tango_ctx, tmp_path, "operator", rules)
    before = scan_proxy.imsize
    try:
        with pytest.raises(RuntimeError) as held:
            agent.write_attribute("SCAN", "imsize", 80)
        call_id = parse_held(str(held.value))
        assert scan_proxy.imsize == before  # not written yet
        assert agent.write_attribute("SCAN", "dwell_time", 2e-6) == pytest.approx(2e-6)  # not listed: runs
        done = operator.approve_call(call_id)
        assert done["status"] == "done" and done["result"] == 80 == scan_proxy.imsize
    finally:
        scan_proxy.imsize = before
        scan_proxy.dwell_time = 1e-6


def test_tools_are_registered_only_with_an_allowlist(monkeypatch, tango_ctx):
    names = []
    server = _server(monkeypatch, tango_ctx)
    monkeypatch.setattr(server.mcp, "add_tool", lambda tool: names.append(getattr(tool, "__name__", getattr(tool, "name", ""))))
    monkeypatch.setattr(server, "_find_tools", lambda: {})
    server.setup(print_summary=False)
    assert {"read_attribute", "write_attribute"} <= set(names)
