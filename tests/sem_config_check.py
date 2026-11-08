"""Run by test_map_profiles.py in a fresh interpreter: maps the stack configs/digital_twin_sem.yaml describes.

The devices, their properties and their addresses come from the launcher's own reading of the file. What a test must
change is in OVERRIDES: DATA and the analysis server are not started, and files go under the given directory.
"""

import importlib
import json
import sys
import tempfile
import time
from pathlib import Path

import h5py
import tango

from asyncroscopy.testing import instrument_context, prop_path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launcher_run_servers", ROOT / "startup_scripts" / "run_servers.py")
run_servers = importlib.util.module_from_spec(spec)
sys.modules["launcher_run_servers"] = run_servers
spec.loader.exec_module(run_servers)

work = Path(tempfile.mkdtemp(dir=sys.argv[1]))
config = run_servers.load_config(ROOT / "configs" / "digital_twin_sem.yaml")
LEFT_OUT = {"data"}
OVERRIDES = {
    "instrument": {"acquisition_save_directory": [prop_path(work)], "time_scale": ["0"], "data_device_address": None, "specimen": None},
    "mapper": {"output_directory": [prop_path(work / "maps")], "state_directory": [prop_path(work / "state")], "trace_path": [prop_path(work / "trace.jsonl")], "analysis_url": None, "data_device_address": None},
    "sample": {"sample_file": [prop_path(work / "sample.json")]},
    "stage": {"holder_file": [prop_path(work / "holder.json")]},
    "approvals": {"trace_path": [prop_path(work / "approvals.jsonl")], "mode_file": [prop_path(work / "mode.json")]},
}


def entry(klass, name, properties):
    """One device; an override of None leaves the property out (the test database cannot hold a blank value)."""
    return {"class": klass, "devices": [{"name": name, "properties": {k: v for k, v in properties.items() if v is not None}}]}


devices = []
for device in config.support_devices:
    if device.key in LEFT_OUT:
        continue
    klass = getattr(importlib.import_module(device.module_name), device.class_name)
    devices.append(entry(klass, device.device_name, {**device.properties, **OVERRIDES.get(device.key, {})}))
instrument_module = importlib.import_module("asyncroscopy.instruments.scanning_electron_microscope.digital_twin")
devices.append(entry(getattr(instrument_module, config.instrument.class_name), "asyncroscopy/instrument/default", {**run_servers.instrument_properties(config), **OVERRIDES["instrument"]}))

outcome = {"devices": sorted(d.key for d in config.support_devices)}
with instrument_context(devices):
    proxy = tango.DeviceProxy("asyncroscopy/instrument/default")
    outcome["instrument_type"] = proxy.instrument_type
    tango.DeviceProxy("asyncroscopy/sample/default").load_sample(json.dumps({"sample_id": "sem grid", "who": "test"}))
    mapper = tango.DeviceProxy("asyncroscopy/mapper/default")
    plan = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": 1.5e-7, "overlap": 0.0, "imsize": 32, "dwell_s": 1e-7}
    reserved = json.loads(mapper.reserve(json.dumps({"plan": plan})))
    assert "reservation_id" in reserved, reserved
    started = json.loads(mapper.start(json.dumps({"reservation_id": reserved["reservation_id"]})))
    deadline = time.monotonic() + 120
    while True:
        status = json.loads(mapper.job_status(json.dumps({"job_id": started["job_id"]})))
        if status["state"] in {"done", "failed", "aborted"}:
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.05)
    results = json.loads(mapper.results(json.dumps({"job_id": started["job_id"]})))
    outcome["state"] = results["state"]
    outcome["last_error"] = status.get("last_error")
    with h5py.File(results["result_keys"]["run_file"], "r") as h5:
        outcome["tiles"] = list(h5["tiles"].shape)
        outcome["detector"] = str(h5["tiles"].attrs["detector"])
        outcome["profile"] = str(h5.attrs["instrument_profile"])
        outcome["sample_load_id"] = str(h5.attrs.get("sample_load_id", ""))
    outcome["get_sample_load_id"] = json.loads(tango.DeviceProxy("asyncroscopy/sample/default").get_sample())["sample_load_id"]
    outcome["blanked"] = json.loads(proxy.get_parameters())["beam_blanked"]
print("SEM_OUTCOME " + json.dumps(outcome), flush=True)
