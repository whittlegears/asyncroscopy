"""Test helpers for instruments built on asyncroscopy, inside this repository or outside it.

A team with its own instrument package tests it from its own repository, without
editing asyncroscopy's ``tests/conftest.py``::

    from asyncroscopy.testing import instrument_context, mcp_server_for, prop_path

    def test_my_scope(tmp_path):
        devices = [
            {"class": OptiScope, "devices": [{"name": "asyncroscopy/instrument/default"}]},
            {"class": TileScan, "devices": [{"name": "asyncroscopy/tilescan/default",
                                             "properties": {"state_directory": prop_path(tmp_path / "jobs")}}]},
        ]
        with instrument_context(devices):
            server = mcp_server_for(devices)
            tools = server.list_tool_descriptions()
            assert tools["OptiScope_move_xy"].startswith("Move the stage")

PyTango allows one in-process test context per interpreter, so give each test
module its own pytest run, or share one context through a session fixture.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from tango.test_context import MultiDeviceTestContext

from asyncroscopy.mcp.mcp_server import MCPServer


def prop_path(path: str | os.PathLike) -> str:
    """A path a test-context device property can carry.

    The test context's file database cuts property values at ":", so a Windows
    path like C:/Users/... arrives as "C:" and files land in the working
    directory. The path is given without its drive ("/Users/..."), which Windows
    reads on the working directory's drive; a relative "../../.." path would
    instead risk the 260-character limit. Keep pytest's temp folder on the same
    drive as the working directory (``--basetemp``).
    """
    drive, rest = os.path.splitdrive(os.path.abspath(path))
    if drive and drive.lower() != os.path.splitdrive(os.getcwd())[0].lower():
        raise RuntimeError(f"{path} is on another drive than the working directory; run pytest with --basetemp on this drive")
    return rest.replace("\\", "/")


_in_process_started = False  # Tango allows one in-process device server per interpreter...
_process_open = 0  # ...and none while a process=True context is open


@contextmanager
def instrument_context(devices_info: list[dict], process: bool = False, startup_timeout_s: float | None = None) -> Iterator[MultiDeviceTestContext]:
    """Start the given devices in one Tango test context (no database, no ports to configure).

    ``devices_info`` is MultiDeviceTestContext's list of {"class": ..., "devices": [{"name": ..., "properties": {...}}]}.
    ``process=True`` runs them in a child process, so command timeouts behave as on a real
    server (an in-process context does not enforce the 3 s client timeout). ``startup_timeout_s``:
    how long the devices may take to start (PyTango's default is 5 s; a large twin sample takes longer).
    """
    global _in_process_started, _process_open
    if _in_process_started or (_process_open and not process):
        raise RuntimeError(
            "Tango allows one in-process instrument_context per Python interpreter, and no other context "
            "(process=True included) after it; start it once in a session-scoped pytest fixture and share it, "
            "and open any process=True contexts before it"
        )
    with MultiDeviceTestContext(devices_info, process=process, timeout=startup_timeout_s) as ctx:
        if process:
            _process_open += 1
        else:
            _in_process_started = True
        try:
            yield ctx
        finally:
            if process:
                _process_open -= 1


class _DeviceList:
    """Stands in for the Tango database: the MCP server only asks it which devices are exported."""

    def __init__(self, names: list[str]) -> None:
        self.value_string = names

    def get_device_exported(self, pattern: str) -> "_DeviceList":
        return self


class TestMCPServer(MCPServer):
    """An MCPServer over devices in a test context, with the tool descriptions an agent would read."""

    __test__ = False  # not a pytest test class

    def list_tool_risks(self) -> dict[str, str]:
        """Tool name -> "low" or "high": what approve mode would hold, without calling anything."""
        return dict(self.tool_risks)

    def list_tool_descriptions(self) -> dict[str, str]:
        """Tool name -> the description an agent reads. In an async test, await alist_tool_descriptions."""
        return asyncio.run(self.alist_tool_descriptions())

    async def alist_tool_descriptions(self) -> dict[str, str]:
        return {t.name: t.description or "" for t in await self.mcp.list_tools()}


def mcp_server_for(
    devices_info: list[dict],
    approvals_device_address: str | None = None,
    endpoint_role: str = "agent",
    risk_overrides: dict[str, str] | None = None,
    trace_path: str | Path = "",
    auth: dict | None = None,
) -> TestMCPServer:
    """The MCP server the agent would talk to, over the devices of a running ``instrument_context``.

    Give ``approvals_device_address`` (an Approvals device in the same context) to exercise
    the approval mode; without it every call runs, as on a server with no Approvals device.
    ``auth`` is the MCP config's auth section (bearer tokens; see asyncroscopy.mcp.endpoint_auth):
    serve ``server.mcp.http_app()`` and call it with tokens from startup_scripts/dev_tokens.py.
    """
    names = [device["name"] for group in devices_info for device in group["devices"]]
    server = TestMCPServer(
        "test",
        "localhost",
        0,
        blocked_classes=["DataBase", "DServer"],
        blocked_functions={"*": ["Init", "Kill", "RestartServer"]},
        data_device_address="asyncroscopy/data/default",
        verbose=False,
        endpoint_role=endpoint_role,
        approvals_device_address=approvals_device_address,
        risk_overrides=risk_overrides,
        trace_path=str(trace_path),
        database=_DeviceList(names),
        auth=auth,
    )
    server.setup(print_summary=False)
    return server
