"""Run by test_yaml_sim.py in a fresh interpreter: the example YAML instrument as a Tango device behind the MCP server."""

import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import tango

from asyncroscopy.instruments.yaml_sim.device import device_class
from asyncroscopy.testing import instrument_context, mcp_server_for, prop_path

work = Path(tempfile.mkdtemp(dir=sys.argv[1]))
example = Path(__file__).resolve().parents[1] / "asyncroscopy" / "instruments" / "yaml_sim" / "examples" / "xrd.yaml"
devices = [{"class": device_class("XRDTwin"), "devices": [{"name": "asyncroscopy/instrument/default", "properties": {
    "yaml_path": prop_path(example), "output_directory": prop_path(work)}}]}]
outcome = {}
with instrument_context(devices):
    proxy = tango.DeviceProxy("asyncroscopy/instrument/default")
    server = mcp_server_for(devices)
    tools = server.list_tool_descriptions()
    risks = server.list_tool_risks()
    xrd = server.tools["XRDTwin"]

    def attempt(func: Callable[..., Any], *args: Any) -> Any:
        try:
            return func(*args)
        except Exception as exc:
            return f"error: {exc}"

    outcome["refused_without_tube"] = attempt(xrd["acquire_diffractogram"])
    outcome["xray_on"] = attempt(xrd["xray_on"], 1200.0)
    outcome["out_of_range"] = attempt(xrd["set_start_deg"], 1.0)
    try:
        proxy.write_attribute("count_time_s", 5.0)
        outcome["attribute_out_of_range"] = "accepted"
    except tango.DevFailed as exc:
        outcome["attribute_out_of_range"] = exc.args[0].desc
    xrd["set_step_count"](301)
    key = xrd["acquire_diffractogram"]()
    with h5py.File(key, "r") as h5:
        attrs = dict(h5["spectrum"].attrs)
        outcome["spectrum"] = {"shape": list(h5["spectrum"].shape), **{k: (v.item() if hasattr(v, "item") else v) for k, v in attrs.items() if k.startswith("axis_")}}
    outcome["status"] = json.loads(xrd["get_status"]())
    outcome["tube_attribute"] = proxy.tube_power_w
    proxy.Init()
    outcome["after_init"] = sorted(c.cmd_name for c in proxy.command_list_query() if c.cmd_name not in ("Init", "State", "Status"))
print(json.dumps({
    "tools": sorted(tools),
    "set_doc": tools.get("XRDTwin_set_start_deg", ""),
    "acquire_doc": tools.get("XRDTwin_acquire_diffractogram", ""),
    "risks": {name: risks[name] for name in ("XRDTwin_get_start_deg", "XRDTwin_set_start_deg", "XRDTwin_xray_on", "XRDTwin_xray_off", "XRDTwin_acquire_diffractogram", "XRDTwin_get_status")},
    "outcome": outcome,
}, default=str))
