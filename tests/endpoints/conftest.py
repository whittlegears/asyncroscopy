"""Fixtures for tests of real MCP endpoints over the session's Tango devices."""

from __future__ import annotations

import socket
import threading
import time

import pytest
import tango
import uvicorn

from asyncroscopy.mcp.mcp_server import MCPServer


@pytest.fixture
def stack(monkeypatch, tango_ctx, tmp_path):
    """make(role, *devices, needs_approval=..., port=1) -> an MCPServer on the session devices, sharing one permissions
    file. Endpoints made with the same ``port`` are one stack (``stack_id`` tango://localhost:<port>)."""
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", lambda name: tango.DeviceProxy(tango_ctx.get_device_access(name)))
    permissions_file = tmp_path / "permissions.json"

    def make(role, *names, needs_approval=None, urls=None, port=1, **extra):
        database = type("Db", (), {"get_device_exported": lambda self, p: type("E", (), {"value_string": list(names)})()})()
        server = MCPServer("perm-test", "localhost", port, blocked_classes=[], needs_approval=needs_approval or {},
                           data_device_address="asyncroscopy/data/default", verbose=False, endpoint_role=role, database=database,
                           permissions_file=str(permissions_file), urls=urls, **extra)
        server.setup(print_summary=False)
        return server

    make.permissions_file = permissions_file
    return make


@pytest.fixture
def served():
    """serve(server) -> its HTTP address, from uvicorn in a thread on a free 127.0.0.1 port."""
    running = []

    def serve(server):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        web = uvicorn.Server(uvicorn.Config(server.mcp.http_app(), host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=web.run, daemon=True)
        thread.start()
        running.append((web, thread))
        deadline = time.monotonic() + 20
        while not web.started:
            assert time.monotonic() < deadline, "the test MCP server did not start"
            time.sleep(0.05)
        return f"http://127.0.0.1:{port}/mcp"

    yield serve
    for web, thread in running:
        web.should_exit = True
        thread.join(10)
