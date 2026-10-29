"""Run by test_yaml_sim.py in a fresh interpreter: the AFM example as a Tango job device behind the MCP server.

It checks the probe interlocks a newcomer's instrument relies on: a stage setting with
``requires`` refuses set_<name> while the tip is engaged, and the force map lifts the tip
before each move and approaches again before each curve (before_move / after_move).
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import h5py
import tango

# The asyncroscopy beside this script (as pytest's pythonpath gives the tests), not another checkout's install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asyncroscopy.instruments.yaml_sim.device import device_class  # noqa: E402
from asyncroscopy.testing import instrument_context, mcp_server_for, prop_path  # noqa: E402

work = Path(tempfile.mkdtemp(dir=sys.argv[1]))
example = Path(__file__).resolve().parents[1] / "asyncroscopy" / "instruments" / "yaml_sim" / "examples" / "afm.yaml"
devices = [{"class": device_class("AFMTwin", jobs=True), "devices": [{"name": "asyncroscopy/instrument/default", "properties": {
    "yaml_path": prop_path(example), "output_directory": prop_path(work),
    "state_directory": prop_path(work / "jobs"), "trace_path": prop_path(work / "trace.jsonl")}}]}]
outcome = {}


def attempt(func, *args):
    try:
        return func(*args)
    except Exception as exc:
        return f"error: {exc}"


with instrument_context(devices):
    proxy = tango.DeviceProxy("asyncroscopy/instrument/default")
    server = mcp_server_for(devices)
    afm = server.tools["AFMTwin"]
    afm["approach_tip"]()
    outcome["set_stage_engaged"] = attempt(afm["set_stage_x_m"], 1e-4)
    outcome["stage_after_refusal"] = proxy.stage_x_m
    afm["retract_tip"]()
    outcome["set_stage_retracted"] = attempt(afm["set_stage_x_m"], 1e-4)
    outcome["stage_after_set"] = proxy.stage_x_m
    # The job's steps drive the simulator directly, so a dry run could record nothing: it is refused, nothing moves.
    outcome["dry_run"] = server.run_job("AFMTwin", {"axes": {"stage_x_m": {"start": 0, "stop": 2e-4, "points": 3}}, "dry_run": True})
    outcome["stage_after_dry_run"] = proxy.stage_x_m
    ran = server.run_job("AFMTwin", {"axes": {"stage_x_m": {"start": 0, "stop": 2e-4, "points": 3}}})
    deadline = time.monotonic() + 90
    status = ran
    while status.get("state") not in {"done", "failed", "aborted"} and time.monotonic() < deadline:
        status = server.wait_for_job(ran["job_id"], max_wait_s=5)
    outcome["job_state"] = status.get("state")
    outcome["job_error"] = status.get("last_error")
    keys = status.get("summary", {}).get("keys", {}).get("acquire_force_curve", [])
    outcome["curves"] = len(keys)
    with h5py.File(keys[-1], "r") as h5:
        curve = h5["spectrum"][()]
        outcome["curve_units"] = h5["spectrum"].attrs["units"]
        outcome["curve_min"] = float(curve.min())
    outcome["tip_after_job"] = bool(proxy.tip_engaged)
    outcome["stage_after_job"] = proxy.stage_x_m
print(json.dumps(outcome, default=str))
