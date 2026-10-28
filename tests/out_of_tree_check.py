"""Run by test_testing_helpers.py in a fresh interpreter: what an out-of-tree team's test does."""

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
    held = {}
    for name in ("SimIndenter_indent", "SimIndenter_retract", "IndentGrid_reserve"):
        try:
            server.tools[name.split("_", 1)[0]][name.split("_", 1)[1]](*(["{}"] if name.endswith("reserve") else [0.001] if name.endswith("indent") else []))
            held[name] = False
        except Exception as exc:
            held[name] = "APPROVAL REQUIRED" in str(exc)
    print(json.dumps({
        "indent_doc": tools.get("SimIndenter_indent", "")[:40],
        "envelope": "IndentGrid" in server._envelope_prefixes,
        "retract_doc": tools.get("SimIndenter_retract", ""),
        "held": held,
    }))
