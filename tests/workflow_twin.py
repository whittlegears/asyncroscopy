"""Helpers for the hill-climb workflow tests: the real instrument MCP server over the shared Tango test devices.

The digital twin's renderer does not respond to stage z or to the corrector, so
the image and its score are fakes: ``acquire_scanned_image`` notes the real device
state (read through Tango) under a new key, and ``resolution`` turns that state
into a synthetic sharpness peak. Everything else (``get_stage``, ``move_stage``,
``correct_aberration``, ``get_correction_limits``) is the real command reached
through the real MCP tool wrappers, with their real argument names.
"""

from __future__ import annotations

import json
import math

from fastmcp import Client

from asyncroscopy.mcp.mcp_server import MCPServer


def mcp_server(tango_ctx, monkeypatch, devices):
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer("test", "localhost", 1, blocked_functions={"*": ["Init"]}, blocked_classes=[],
                       data_device_address=tango_ctx.get_device_access("asyncroscopy/data/default"), verbose=False)
    monkeypatch.setattr(server, "_list_all_devices", lambda: [tango_ctx.get_device_access(n) for n in devices])
    server.setup(print_summary=False)
    return server


def client_tools(client: Client, names, log: list):
    """Each MCP tool as an async callable, as the agent's tools reach it. Calls are logged as (name, arguments)."""

    def make(name):
        async def call(**arguments):
            log.append((name, arguments))
            result = await client.call_tool(name, arguments, raise_on_error=True)
            data = result.structured_content
            if data is not None:
                return data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data
            return json.loads(result.content[0].text)

        return call

    return {name: make(name) for name in names}


class SyntheticImages:
    """acquire_scanned_image and ANALYSIS.resolution over a Gaussian sharpness peak of the device state."""

    def __init__(self, distance, fail_on=None, nyquist=2.0):
        self.distance, self.fail_on, self.nyquist = distance, fail_on, nyquist
        self.keys: dict[str, float] = {}

    def tools(self):
        async def DigitalTwin_acquire_scanned_image(detector_list):
            if self.fail_on is not None and len(self.keys) + 1 >= self.fail_on:
                raise RuntimeError("DevFailed: scan timed out")
            key = f"synthetic/{len(self.keys)}"
            self.keys[key] = math.exp(-0.5 * self.distance() ** 2)
            return key

        async def ANALYSIS_resolution(key):
            sharpness = self.keys[key]
            return json.dumps({"key": key, "nyquist_angstrom": self.nyquist,
                               "resolution_angstrom": self.nyquist / sharpness if sharpness > 1e-12 else None})

        return {"DigitalTwin_acquire_scanned_image": DigitalTwin_acquire_scanned_image, "ANALYSIS_resolution": ANALYSIS_resolution}


async def run_workflow(name, build, inputs, answer):
    """Start a workflow with WorkflowRunner, answer every pause with ``answer(pause)``; return (final step, pauses)."""
    from asyncroscopy.agent_workflows.runner import WorkflowRunner

    runner = WorkflowRunner()
    runner.register(name, build)
    step = await runner.start(name, inputs)
    pauses = []
    while step["waiting_for"] is not None:
        pauses.append(step["waiting_for"])
        step = await runner.resume(step["thread_id"], answer(step["waiting_for"]))
    return step, pauses
