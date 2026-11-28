"""endpoint_info's stack_fingerprint, read from a real Tango database and a real device server started as run_servers does."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import tango

from startup_scripts import run_servers
from tests.startup.test_server_startup import start_device_server, tango_database, wait_for_device

DEVICE = '''from tango.server import Device, command


class Fingerprinted(Device):
    @command(dtype_out=str)
    def get_value(self):
        return "{value}"


if __name__ == "__main__":
    Fingerprinted.run_server()
'''


def endpoint(host, port, tmp_path, env):
    check = Path(__file__).resolve().parents[1] / "fingerprint_check.py"
    done = subprocess.run([sys.executable, str(check), host, str(port), str(tmp_path / "permissions.json")], capture_output=True, text=True,
                          timeout=180, env=env, cwd=Path(__file__).resolve().parents[2], check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_fingerprint_is_stable_for_one_stack_and_follows_its_device_code_and_properties(tango_database, tmp_path):
    host, port, env = tango_database
    module = tmp_path / "fingerprinted.py"
    module.write_text(DEVICE.format(value="one"), encoding="utf-8")
    env = {**env, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")])}
    device = run_servers.DeviceConfig("instrument", "Fingerprinted", "fingerprinted", properties={"exposure_s": ["1.0"]})
    run_servers.register_devices([device], {})
    served = start_device_server("fingerprinted", device.instance_name, env)
    try:
        wait_for_device(host, port, device.device_name, timeout=60)
        first = endpoint(host, port, tmp_path, env)
        assert first["devices"] == {"Fingerprinted": {"class": "Fingerprinted", "module": "fingerprinted",
                                                      "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest()}}
        assert first["unreadable_properties"] == [] and first["simulated"]["value"] is False and "declared" in first["simulated"]["established_by"]
        assert first["software"]["package"] == "asyncroscopy" and first["software"]["commit"]
        assert endpoint(host, port, tmp_path, env) == first

        module.write_text(DEVICE.format(value="two"), encoding="utf-8")
        edited = endpoint(host, port, tmp_path, env)
        assert edited["devices"]["Fingerprinted"]["module_sha256"] != first["devices"]["Fingerprinted"]["module_sha256"]
        assert edited["config_digest"] == first["config_digest"] and edited["digest"] != first["digest"]

        tango.Database(host, port).put_device_property(device.device_name, {"exposure_s": ["2.0"]})
        reconfigured = endpoint(host, port, tmp_path, env)
        assert reconfigured["config_digest"] != edited["config_digest"] and reconfigured["digest"] != edited["digest"]
    finally:
        served.stop()
