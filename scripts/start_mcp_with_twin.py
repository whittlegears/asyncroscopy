#!/usr/bin/env python
"""
start_mcp_with_twin.py

Starts the Tango DB, registers and runs the ThermoDigitalTwin,
and then starts the MCP server. This is useful for running the
MCP server along with a mock twin in a single command.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import tango

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asyncroscopy.mcp.mcp_server import MCPServer

class ManagedProcess:
    def __init__(self, name: str, process: subprocess.Popen[str]):
        self.name = name
        self.process = process

def log_stderr(msg: str) -> None:
    """Log to stderr to avoid corrupting MCP stdout JSON-RPC."""
    print(msg, file=sys.stderr, flush=True)

def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])

def make_env(tango_host: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TANGO_HOST"] = tango_host
    env["PYTHONUNBUFFERED"] = "1"
    return env

def wait_for_process_output(
    proc: subprocess.Popen[str],
    expected_text: str,
    timeout: float,
    process_name: str,
) -> None:
    start = time.monotonic()
    seen_lines = []

    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            output = "\n".join(seen_lines)
            raise RuntimeError(
                f"{process_name} exited early with code {proc.returncode}.\n"
                f"Observed output:\n{output}"
            )

        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            line = line.rstrip("\n")
            seen_lines.append(line)
            log_stderr(f"[{process_name}] {line}")
            if expected_text in line:
                return
        else:
            time.sleep(0.05)

    output = "\n".join(seen_lines)
    raise TimeoutError(
        f"Timed out waiting for '{expected_text}' from {process_name}.\n"
        f"Observed output:\n{output}"
    )

def wait_for_device_ready(device_name: str, timeout: float = 10.0) -> None:
    start = time.monotonic()
    last_error = None

    while time.monotonic() - start < timeout:
        try:
            dev = tango.DeviceProxy(device_name)
            dev.ping()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)

    raise TimeoutError(
        f"Timed out waiting for device '{device_name}' readiness. "
        f"Last error: {last_error}"
    )

def start_tango_db(python_bin: str, tango_host: str, work_dir: Path, timeout: float) -> ManagedProcess:
    log_stderr("[startup] Starting Tango DB...")
    env = make_env(tango_host)
    proc = subprocess.Popen(
        [python_bin, "-m", "tango.databaseds.database", "2"],
        cwd=work_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    managed = ManagedProcess(name="tango-db", process=proc)
    wait_for_process_output(proc, "Ready to accept request", timeout, managed.name)
    return managed

def register_device(
    db_host: str,
    db_port: int,
    server_name: str,
    class_name: str,
    device_name: str,
) -> None:
    db = tango.Database(db_host, db_port)
    info = tango.DbDevInfo()
    info.server = server_name
    info._class = class_name
    info.name = device_name

    try:
        db.add_device(info)
        log_stderr(f"[register] registered: {device_name}")
    except tango.DevFailed:
        log_stderr(f"[register] device already present: {device_name}")

def set_twin_properties(
    db_host: str,
    db_port: int,
    twin_device_name: str,
    scan_device_name: str,
    eds_device_name: str,
) -> None:
    db = tango.Database(db_host, db_port)
    db.put_device_property(
        twin_device_name,
        {
            "scan_device_address": [scan_device_name],
            "eds_device_address": [eds_device_name],
        },
    )
    log_stderr(f"[register] property set: {twin_device_name}.scan_device_address={scan_device_name}")
    log_stderr(f"[register] property set: {twin_device_name}.eds_device_address={eds_device_name}")

def start_scan_device(python_bin: str, tango_host: str, instance: str, timeout: float) -> ManagedProcess:
    log_stderr("[startup] Starting SCAN device server...")
    env = make_env(tango_host)
    proc = subprocess.Popen(
        [python_bin, "-m", "asyncroscopy.hardware.SCAN", instance],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    managed = ManagedProcess(name="scan-device", process=proc)
    wait_for_process_output(proc, "Ready to accept request", timeout, managed.name)
    return managed

def start_eds_device(python_bin: str, tango_host: str, instance: str, timeout: float) -> ManagedProcess:
    log_stderr("[startup] Starting EDS device server...")
    env = make_env(tango_host)
    proc = subprocess.Popen(
        [python_bin, "-m", "asyncroscopy.detectors.EDS", instance],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    managed = ManagedProcess(name="eds-device", process=proc)
    wait_for_process_output(proc, "Ready to accept request", timeout, managed.name)
    return managed

def start_digital_twin(python_bin: str, tango_host: str, instance: str, timeout: float) -> ManagedProcess:
    log_stderr("[startup] Starting ThermoDigitalTwin...")
    env = make_env(tango_host)
    proc = subprocess.Popen(
        [python_bin, "-m", "asyncroscopy.ThermoDigitalTwin", instance],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    managed = ManagedProcess(name="digital-twin", process=proc)
    wait_for_process_output(proc, "Ready to accept request", timeout, managed.name)
    return managed

def stop_process(managed: ManagedProcess, timeout: float = 5.0) -> None:
    proc = managed.process
    if proc.poll() is not None:
        return

    log_stderr(f"[shutdown] terminating {managed.name} (pid={proc.pid})")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log_stderr(f"[shutdown] killing {managed.name} (pid={proc.pid})")
        proc.kill()
        proc.wait(timeout=timeout)

def main():
    host = "127.0.0.1"
    port = find_free_port(host)
    tango_host = f"{host}:{port}"
    python_bin = sys.executable

    log_stderr(f"[config] TANGO_HOST={tango_host}")
    os.environ["TANGO_HOST"] = tango_host

    managed_procs = []
    db_dir_obj = tempfile.TemporaryDirectory(prefix="tango-db-run-")
    db_path = Path(db_dir_obj.name)

    try:
        # Start Tango DB
        db_proc = start_tango_db(python_bin, tango_host, db_path, timeout=30.0)
        managed_procs.append(db_proc)

        # Register SCAN + EDS + Twin devices and set required twin properties
        twin_instance = "mcp_instance"
        twin_device_name = "test/digitaltwin/1"
        scan_instance = "scan_instance"
        scan_device_name = "test/scan/1"
        eds_instance = "eds_instance"
        eds_device_name = "test/eds/1"
        register_device(host, port, f"SCAN/{scan_instance}", "SCAN", scan_device_name)
        register_device(host, port, f"EDS/{eds_instance}", "EDS", eds_device_name)
        register_device(host, port, f"ThermoDigitalTwin/{twin_instance}", "ThermoDigitalTwin", twin_device_name)
        set_twin_properties(host, port, twin_device_name, scan_device_name, eds_device_name)

        # Start detector devices first so twin can resolve its proxies at init
        scan_proc = start_scan_device(python_bin, tango_host, scan_instance, timeout=30.0)
        managed_procs.append(scan_proc)
        wait_for_device_ready(scan_device_name, timeout=10.0)
        log_stderr("[startup] SCAN device is fully accessible")

        eds_proc = start_eds_device(python_bin, tango_host, eds_instance, timeout=30.0)
        managed_procs.append(eds_proc)
        wait_for_device_ready(eds_device_name, timeout=10.0)
        log_stderr("[startup] EDS device is fully accessible")

        # Start Twin
        twin_proc = start_digital_twin(python_bin, tango_host, twin_instance, timeout=30.0)
        managed_procs.append(twin_proc)

        # Wait for Twin to be fully responsive
        wait_for_device_ready(twin_device_name, timeout=10.0)
        log_stderr("[startup] ThermoDigitalTwin is fully accessible")

        # Start MCPServer
        log_stderr("[startup] Initializing MCP Server...")
        server = MCPServer(
            name="MCPServer_Twin",
            tango_host=host,
            tango_port=port,
            blocked_classes=["DataBase", "DServer"],
            verbose=False,
        )

        log_stderr(f"[startup] Starting MCP Server. Exported devices: {server.list_devices()}")
        server.start()

    except KeyboardInterrupt:
        log_stderr("\n[shutdown] KeyboardInterrupt received. Shutting down...")
    except Exception as exc:
        log_stderr(f"\n[error] Fatal error: {exc}")
        sys.exit(1)
    finally:
        for proc in reversed(managed_procs):
            stop_process(proc)
        db_dir_obj.cleanup()

if __name__ == "__main__":
    main()
