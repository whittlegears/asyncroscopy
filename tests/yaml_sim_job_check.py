"""Run by test_yaml_sim.py in a fresh interpreter: the Raman example (a YAML instrument with jobs) as a Tango job device behind the MCP server."""

import json
import sys
import tempfile
import time
from pathlib import Path

import h5py
import tango

from asyncroscopy.instruments.yaml_sim.device import device_class
from asyncroscopy.testing import instrument_context, mcp_server_for, prop_path

work = Path(tempfile.mkdtemp(dir=sys.argv[1]))
example = Path(__file__).resolve().parents[1] / "asyncroscopy" / "instruments" / "yaml_sim" / "examples" / "raman.yaml"
devices = [{"class": device_class("RamanTwin", jobs=True), "devices": [{"name": "asyncroscopy/instrument/default", "properties": {
    "yaml_path": prop_path(example), "output_directory": prop_path(work),
    "state_directory": prop_path(work / "jobs"), "trace_path": prop_path(work / "trace.jsonl")}}]}]
outcome = {}


def attempt(func, *args):
    try:
        return func(*args)
    except Exception as exc:
        return f"error: {exc}"


def wait(server, job_id, limit_s=60.0):
    deadline = time.monotonic() + limit_s
    status = {}
    while time.monotonic() < deadline:
        status = server.wait_for_job(job_id, max_wait_s=5)
        if status.get("state") not in {"queued", "running", "aborting"}:
            break
    return status


with instrument_context(devices):
    proxy = tango.DeviceProxy("asyncroscopy/instrument/default")
    server = mcp_server_for(devices)
    raman = server.tools["RamanTwin"]
    risks = server.list_tool_risks()
    outcome["risks"] = {name: risks[name] for name in ("RamanTwin_reserve", "RamanTwin_start", "RamanTwin_abort", "RamanTwin_laser_off", "RamanTwin_move_xy", "RamanTwin_set_grating")}
    outcome["move_doc"] = server.list_tool_descriptions().get("RamanTwin_move_xy", "")
    outcome["plan_example"] = json.loads(proxy.plan_example)
    outcome["refused_without_laser"] = server.run_job("RamanTwin", {"axes": {"stage_x_m": {"start": 0, "stop": 1e-4, "points": 3}}})
    outcome["move_two_inputs"] = attempt(raman["move_xy"], {"x_m": 1e-4, "y_m": -1e-4})
    outcome["move_missing"] = attempt(raman["move_xy"], {"x_m": 1e-4})
    outcome["grating_choice"] = attempt(raman["set_grating"], "1200")
    raman["laser_on"](5.0)
    outcome["out_of_limits"] = server.run_job("RamanTwin", {"axes": {"stage_x_m": {"start": 0, "stop": 0.01, "points": 3}}})
    ran = server.run_job("RamanTwin", {"axes": {"stage_x_m": {"start": -1e-4, "stop": 1e-4, "points": 3}}, "settings": {"exposure_s": 0.05}})
    outcome["estimate"] = ran.get("estimate")
    done = wait(server, ran["job_id"]) if ran.get("state") != "done" else ran
    outcome["done_state"] = done.get("state")
    keys = done.get("summary", {}).get("keys", {}).get("acquire_raman_spectrum", [])
    outcome["keys"] = len(keys)
    with h5py.File(keys[-1], "r") as h5:
        outcome["last_point"] = {"point": json.loads(h5.attrs["point"]), "units": h5["spectrum"].attrs["units"], "axis_units": h5["spectrum"].attrs["axis_units"]}
    attempt(raman["release"], {"reservation_id": ran["reservation_id"]})
    long = server.run_job("RamanTwin", {"axes": {"stage_x_m": {"start": 0, "stop": 1e-4, "points": 20}}, "settings": {"exposure_s": 1.0}})
    outcome["busy_set"] = attempt(raman["set_exposure_s"], 2.0)
    outcome["busy_acquire"] = attempt(raman["acquire_raman_spectrum"])
    outcome["status_while_running"] = attempt(raman["get_status"])
    raman["abort"]({"job_id": long["job_id"]})
    stopped = wait(server, long["job_id"], limit_s=20)
    outcome["aborted_state"] = stopped.get("state")
    outcome["laser_after_abort"] = proxy.laser_power_mw
print(json.dumps(outcome, default=str))
