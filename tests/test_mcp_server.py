"""End-to-end DB-mode tests for MCP + ThermoDigitalTwin integration."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pytest
import tango

import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asyncroscopy.mcp.mcp_server import MCPServer


@dataclass
class ManagedProcess:
    """A subprocess wrapper with a name for logging."""

    name: str
    process: subprocess.Popen[str]


class TestMCPServerDBMode:
    """Test suite for MCP server in DB mode with ThermoDigitalTwin."""

    @staticmethod
    def wait_for_device_ready(device_name: str, timeout: float = 10.0) -> None:
        """Wait until a Tango device can be opened and pinged successfully."""
        start = time.monotonic()
        last_error: Exception | None = None

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

    @staticmethod
    def find_free_port(host: str = "127.0.0.1") -> int:
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def make_env(tango_host: str) -> dict[str, str]:
        """Create environment with TANGO_HOST and unbuffered output."""
        env = os.environ.copy()
        env["TANGO_HOST"] = tango_host
        env["PYTHONUNBUFFERED"] = "1"
        return env

    @staticmethod
    def wait_for_process_output(
        proc: subprocess.Popen[str],
        expected_text: str,
        timeout: float,
        process_name: str,
    ) -> None:
        """Wait for a process to output a specific string."""
        start = time.monotonic()
        seen_lines: list[str] = []

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
                print(f"[{process_name}] {line}")
                if expected_text in line:
                    return
            else:
                time.sleep(0.05)

        output = "\n".join(seen_lines)
        raise TimeoutError(
            f"Timed out waiting for '{expected_text}' from {process_name}.\n"
            f"Observed output:\n{output}"
        )

    @staticmethod
    def stop_process(managed: ManagedProcess, timeout: float = 5.0) -> None:
        """Terminate a managed process."""
        proc = managed.process
        if proc.poll() is not None:
            return

        print(f"[shutdown] terminating {managed.name} (pid={proc.pid})")
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[shutdown] killing {managed.name} (pid={proc.pid})")
            proc.kill()
            proc.wait(timeout=timeout)

    def start_tango_db(
        self,
        python_bin: str,
        tango_host: str,
        work_dir: Path,
        timeout: float,
    ) -> ManagedProcess:
        """Start the Tango database server."""
        env = self.make_env(tango_host)
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
        self.wait_for_process_output(
            proc, "Ready to accept request", timeout, managed.name
        )
        return managed

    @staticmethod
    def register_digital_twin(
        db_host: str, db_port: int, instance: str, device_name: str
    ) -> None:
        """Register ThermoDigitalTwin device in Tango DB."""
        db = tango.Database(db_host, db_port)
        info = tango.DbDevInfo()
        info.server = f"ThermoDigitalTwin/{instance}"
        info._class = "ThermoDigitalTwin"
        info.name = device_name

        try:
            db.add_device(info)
            print(f"[register] registered: {device_name}")
        except tango.DevFailed:
            print(f"[register] device already present: {device_name}")

    def start_digital_twin(
        self,
        python_bin: str,
        tango_host: str,
        instance: str,
        timeout: float,
    ) -> ManagedProcess:
        """Start the ThermoDigitalTwin device server."""
        env = self.make_env(tango_host)
        proc = subprocess.Popen(
            [python_bin, "-m", "asyncroscopy.ThermoDigitalTwin", instance],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        managed = ManagedProcess(name="digital-twin", process=proc)
        self.wait_for_process_output(
            proc, "Ready to accept request", timeout, managed.name
        )
        return managed

    @pytest.fixture(scope="class")
    def test_infrastructure(self) -> Generator[tuple[str, int], None, None]:
        """Pytest fixture that sets up the complete test infrastructure (DB + Digital Twin)."""
        host = "127.0.0.1"
        port = self.find_free_port(host)
        tango_host = f"{host}:{port}"
        python_bin = __import__("sys").executable

        print(f"\n[config] TANGO_HOST={tango_host}")
        os.environ["TANGO_HOST"] = tango_host

        managed_procs: list[ManagedProcess] = []

        try:
            # Start Tango DB
            with tempfile.TemporaryDirectory(prefix="tango-db-") as db_dir:
                db_path = Path(db_dir)
                print(f"[config] tango-db working dir={db_path}")

                try:
                    db_proc = self.start_tango_db(
                        python_bin=python_bin,
                        tango_host=tango_host,
                        work_dir=db_path,
                        timeout=30.0,
                    )
                except Exception as exc:
                    pytest.skip(f"Tango DB could not be started in this environment: {exc}")
                managed_procs.append(db_proc)

                # Register and start Digital Twin
                instance = "test_instance"
                device_name = "test/digitaltwin/1"
                self.register_digital_twin(host, port, instance, device_name)

                try:
                    twin_proc = self.start_digital_twin(
                        python_bin=python_bin,
                        tango_host=tango_host,
                        instance=instance,
                        timeout=30.0,
                    )
                except Exception as exc:
                    pytest.skip(
                        f"ThermoDigitalTwin server could not be started in this environment: {exc}"
                    )
                managed_procs.append(twin_proc)

                try:
                    self.wait_for_device_ready(device_name, timeout=10.0)
                except Exception as exc:
                    pytest.skip(f"ThermoDigitalTwin not ready for queries: {exc}")

                yield host, port

        finally:
            # Cleanup in reverse order
            for proc in reversed(managed_procs):
                self.stop_process(proc)

    def test_mcp_tool_discovery(
        self,
        test_infrastructure: tuple[str, int],
    ) -> None:
        """Test that MCP discovers all Tango device tools."""
        host, port = test_infrastructure

        # Create MCPServer and discover tools
        server = MCPServer(
            name="MCPServerTest",
            tango_host=host,
            tango_port=port,
            blocked_classes=["DataBase", "DServer"],
            blocked_functions=[],
        )

        server.setup(print_summary=True)
        tools = server.tools

        mcp_tools = asyncio.run(server.mcp.list_tools())
        mcp_tool_names = {tool.name for tool in mcp_tools}
        assert "list_devices" in mcp_tool_names, (
            "list_devices should be auto-registered as an MCP tool"
        )

        # Verify ThermoDigitalTwin was discovered
        assert "ThermoDigitalTwin" in tools, (
            "ThermoDigitalTwin class not found in MCP tool discovery"
        )

        # Verify expected tools exist
        thermo_tools = tools["ThermoDigitalTwin"]
        expected_tools = {
            "Connect",
            "Disconnect",
            "get_scanned_image",
        }

        for expected_tool in expected_tools:
            assert (
                expected_tool in thermo_tools
            ), f"Expected tool {expected_tool} not found"

        # Verify blocked classes are not exposed
        assert "DataBase" not in tools, "DataBase should be blocked"
        assert "DServer" not in tools, "DServer should be blocked"

    def test_list_devices_is_available(
        self,
        test_infrastructure: tuple[str, int],
    ) -> None:
        """Test that list_devices MCPServer method is available."""
        host, port = test_infrastructure

        server = MCPServer(
            name="MCPServerTest",
            tango_host=host,
            tango_port=port,
            blocked_classes=["DataBase", "DServer"],
        )

        server.setup(print_summary=False)

        # Verify list_devices method exists
        assert hasattr(
            server, "list_devices"
        ), "list_devices method not found on MCPServer"

        # Call it to verify it works
        deadline = time.monotonic() + 3.0
        has_digital_twin = False
        devices: list[str] = []

        # Tango DB can transiently reject very early connection attempts.
        while time.monotonic() < deadline and not has_digital_twin:
            devices = server.list_devices()
            assert isinstance(devices, list), "list_devices should return a list"

            for device_name in devices:
                try:
                    dev = tango.DeviceProxy(device_name)
                    if dev.info().dev_class == "ThermoDigitalTwin":
                        has_digital_twin = True
                        break
                except Exception:
                    continue

            if not has_digital_twin:
                time.sleep(0.1)

        assert has_digital_twin, (
            f"No ThermoDigitalTwin-class device found in list_devices output: {devices}"
        )

    def test_blocked_classes_respected(
        self,
        test_infrastructure: tuple[str, int],
    ) -> None:
        """Test that blocked_classes parameter is respected."""
        host, port = test_infrastructure

        server = MCPServer(
            name="MCPServerTest",
            tango_host=host,
            tango_port=port,
            blocked_classes=["DataBase", "DServer", "ThermoDigitalTwin"],
        )

        server.setup(print_summary=False)
        tools = server.tools

        # Verify blocked classes are not in tools
        for blocked_class in ["DataBase", "DServer", "ThermoDigitalTwin"]:
            assert (
                blocked_class not in tools
            ), f"Blocked class {blocked_class} was exposed"

class TestMCPSerialization:
    def test_devencoded_type_maps_to_object_schema(self) -> None:
        mapped = MCPServer._tango_type_to_python(tango.CmdArgType.DevEncoded)
        assert mapped is dict

    def test_devencoded_payload_is_json_safe(self) -> None:
        normalized = MCPServer._normalize_command_result(
            tango.CmdArgType.DevEncoded,
            ('{"shape":[2,2],"dtype":"uint8"}', b"\x00\x01\xff\x10"),
        )

        assert isinstance(normalized, dict)
        assert normalized["encoding"] == "base64"
        assert normalized["metadata"] == '{"shape":[2,2],"dtype":"uint8"}'
        assert isinstance(normalized["payload"], str)
        assert base64.b64decode(normalized["payload"]) == b"\x00\x01\xff\x10"
