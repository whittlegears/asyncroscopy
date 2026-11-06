"""Run by test_testing_helpers.py in a fresh interpreter: what an out-of-tree team's test does."""

import asyncio
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

from asyncroscopy.instruments.nano_indenter.indent_grid import IndentGrid
from asyncroscopy.instruments.nano_indenter.sim_indenter import SimIndenter
from asyncroscopy.safety.approvals import Approvals
from asyncroscopy.testing import instrument_context, mcp_server_for, prop_path

work = Path(tempfile.mkdtemp(dir=sys.argv[1]))
devices = [
    {"class": SimIndenter, "devices": [{"name": "asyncroscopy/instrument/default", "properties": {"seed": 3}}]},
    {"class": IndentGrid, "devices": [{"name": "asyncroscopy/indentgrid/default", "properties": {
        "state_directory": prop_path(work / "state"), "trace_path": prop_path(work / "trace.jsonl")}}]},
    {"class": Approvals, "devices": [{"name": "asyncroscopy/approvals/default", "properties": {
        "default_mode": "approve", "mode_file": prop_path(work / "mode.json"), "trace_path": prop_path(work / "approvals.jsonl")}}]},
]
with instrument_context(devices):
    server = mcp_server_for(devices, approvals_device_address="asyncroscopy/approvals/default")
    tools = server.list_tool_descriptions()
    outcome = {}
    calls = {"SimIndenter_indent": [0.001], "SimIndenter_retract": [], "IndentGrid_reserve": [json.dumps({"plan": {"origin_m": [0, 0], "nx": 2, "ny": 1, "pitch_m": 1e-5, "load_n": 0.01}})]}
    for name, args in calls.items():
        prefix, command = name.split("_", 1)
        try:
            reply = server.tools[prefix][command](*args)
            outcome[name] = "error" if isinstance(reply, str) and '"error"' in reply else "ran"
        except Exception as exc:
            outcome[name] = "held" if "APPROVAL REQUIRED" in str(exc) else f"error: {exc}"
    # Overrides may name the tool; one that matches nothing is reported at start-up.
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        overridden = mcp_server_for(devices, approvals_device_address="asyncroscopy/approvals/default",
                                    risk_overrides={"SimIndenter_move_xy": "low", "SimIndentr.indent": "high", "SimIndenter.retract": "high", "SimIndenter_ret*": "low"})
    try:
        with instrument_context(devices):
            second_context = "started"
    except RuntimeError as exc:
        second_context = str(exc)
    print(json.dumps({
        "override_by_tool_name": overridden.list_tool_risks()["SimIndenter_move_xy"],
        "first_override_wins": overridden.list_tool_risks()["SimIndenter_retract"],
        "risk_warnings": overridden.risk_warnings,
        "startup_warnings": [line for line in printed.getvalue().splitlines() if line.startswith("[safety]")],
        "second_context": second_context,
        "indent_doc": tools.get("SimIndenter_indent", "")[:40],
        "envelope": server.devices["IndentGrid"].envelope,
        "retract_doc": tools.get("SimIndenter_retract", ""),
        "move_doc": tools.get("SimIndenter_move_xy", ""),
        "risks": {n: server.list_tool_risks()[n] for n in ("SimIndenter_indent", "SimIndenter_retract", "IndentGrid_reserve", "IndentGrid_start")},
        "outcome": outcome,
        "async_descriptions_match": asyncio.run(server.alist_tool_descriptions()) == tools,
    }))
