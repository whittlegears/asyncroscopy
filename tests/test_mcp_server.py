"""End-to-end DB-mode tests for MCP + DigitalTwin integration."""

from __future__ import annotations

import os
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pytest
import tango

import asyncio

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult

from asyncroscopy.data.data_reader import numpy_to_python
from asyncroscopy.mcp import mcp_server
from asyncroscopy.mcp.components import data_tools
from asyncroscopy.mcp.mcp_server import COMMAND_TIMEOUT_MILLIS, MCPServer


def mcp_kwargs(**overrides):
    config = {
        "blocked_classes": ["DataBase", "DServer"],
        "blocked_functions": {"*": ["Init"]},
        "data_device_address": "asyncroscopy/data/default",
    }
    config.update(overrides)
    return config


@dataclass
class ManagedProcess:
    """A subprocess wrapper with a name for logging."""

    name: str
    process: subprocess.Popen[str]


@pytest.mark.skip(
    reason=(
        "DB-mode Tango discovery cannot safely run in the same interpreter as "
        "the in-process MultiDeviceTestContext. Server startup is covered by "
        "tests/test_server_startup.py."
    )
)
class TestMCPServerDBMode:
    """Test suite for MCP server in DB mode with DigitalTwin."""

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

        raise TimeoutError(f"Timed out waiting for device '{device_name}' readiness. Last error: {last_error}")

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
        self.wait_for_process_output(proc, "Ready to accept request", timeout, managed.name)
        return managed

    @staticmethod
    def register_digital_twin(
        db_host: str, db_port: int, instance: str, device_name: str
    ) -> None:
        """Register DigitalTwin device in Tango DB."""
        db = tango.Database(db_host, db_port)
        info = tango.DbDevInfo()
        info.server = f"DigitalTwin/{instance}"
        info._class = "DigitalTwin"
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
        """Start the DigitalTwin device server."""
        env = self.make_env(tango_host)
        proc = subprocess.Popen(
            [python_bin, "-m", "asyncroscopy.instruments.electron_microscope.digital_twin", instance],
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
                device_name = "asyncroscopy/digitaltwin/default"
                self.register_digital_twin(host, port, instance, device_name)

                try:
                    twin_proc = self.start_digital_twin(
                        python_bin=python_bin,
                        tango_host=tango_host,
                        instance=instance,
                        timeout=30.0,
                    )
                except Exception as exc:
                    pytest.skip(f"DigitalTwin server could not be started in this environment: {exc}")
                managed_procs.append(twin_proc)

                try:
                    self.wait_for_device_ready(device_name, timeout=10.0)
                except Exception as exc:
                    pytest.skip(f"DigitalTwin not ready for queries: {exc}")

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
        server = MCPServer(name="MCPServerTest", tango_host=host, tango_port=port, **mcp_kwargs())

        server.setup(print_summary=True)
        tools = server.tools

        mcp_tools = asyncio.run(server.mcp.list_tools())
        mcp_tool_names = {tool.name for tool in mcp_tools}
        assert "list_devices" in mcp_tool_names, (
            "list_devices should be auto-registered as an MCP tool"
        )

        # Verify DigitalTwin was discovered
        assert "DigitalTwin" in tools, "DigitalTwin class not found in MCP tool discovery"

        # Verify expected tools exist
        digital_twin_tools = tools["DigitalTwin"]
        expected_tools = {
            "Connect",
            "Disconnect",
            "acquire_scanned_image",
        }

        for expected_tool in expected_tools:
            assert expected_tool in digital_twin_tools, (
                f"Expected tool {expected_tool} not found"
            )

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
            **mcp_kwargs(),
        )

        server.setup(print_summary=False)

        # Verify list_devices method exists
        assert hasattr(server, "list_devices"), "list_devices method not found on MCPServer"

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
                    if dev.info().dev_class == "DigitalTwin":
                        has_digital_twin = True
                        break
                except Exception:
                    continue

            if not has_digital_twin:
                time.sleep(0.1)

        assert has_digital_twin, f"No DigitalTwin-class device found in list_devices output: {devices}"

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
            **mcp_kwargs(blocked_classes=["DataBase", "DServer", "DigitalTwin"]),
        )

        server.setup(print_summary=False)
        tools = server.tools

        # Verify blocked classes are not in tools
        for blocked_class in ["DataBase", "DServer", "DigitalTwin"]:
            assert blocked_class not in tools, f"Blocked class {blocked_class} was exposed"


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
        assert normalized["metadata"] == {"shape": [2, 2], "dtype": "uint8"}
        assert isinstance(normalized["payload"], str)
        assert base64.b64decode(normalized["payload"]) == b"\x00\x01\xff\x10"

    def test_numpy_to_python_converts_nested_structures(self) -> None:
        payload = {
            "array": np.array([1, 2], dtype=np.int64),
            "scalar": np.float32(1.5),
            "tuple": (np.int32(3), np.array([[4, 5]], dtype=np.int64)),
            "list": [np.uint8(6)],
        }

        converted = numpy_to_python(payload)

        assert converted == {
            "array": [1, 2],
            "scalar": 1.5,
            "tuple": (3, [[4, 5]]),
            "list": [6],
        }
        assert isinstance(converted["tuple"], tuple)

    def test_normalize_command_result_converts_numpy_non_encoded(self) -> None:
        result = np.array([1, 2, 3], dtype=np.uint16)
        normalized = MCPServer._normalize_command_result(tango.CmdArgType.DevString, result)
        assert normalized == [1, 2, 3]

    def test_normalize_command_result_converts_numpy_nested(self) -> None:
        result = {
            "array": np.array([1, 2], dtype=np.int64),
            "scalar": np.float32(1.25),
            "nested": {"values": [np.uint8(3), np.array([[4, 5]])]},
        }

        normalized = MCPServer._normalize_command_result(tango.CmdArgType.DevString, result)

        assert normalized == {
            "array": [1, 2],
            "scalar": 1.25,
            "nested": {"values": [3, [[4, 5]]]},
        }

    def test_view_data_describes_an_image_as_text_and_png(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FakeArray:
            metadata = {"detector": "HAADF", "pixel_size_m": 1e-9}
            shape = (3, 3)
            dtype = np.dtype("int64")

            def read(self, slices=None):
                array = np.arange(9).reshape(3, 3)
                return array[slices] if slices else array

        class FakeContainer(dict):
            metadata = {"source": "microscope"}

        node = FakeContainer(image=FakeContainer(HAADF=FakeArray()))
        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        monkeypatch.setattr(server.data, "tiled_node", lambda key, hints=True: {"frame.h5": node}[key])

        text = server.data.view_data("frame.h5", format="text")
        assert text.structured_content["kind"] == "image"
        assert text.structured_content["dataset"] == "image/HAADF"
        assert "3 x 3 pixels" in text.content[0].text
        assert not text.content[0].text.startswith("[")  # plain text, not a JSON-encoded list
        assert not [b for b in text.content if getattr(b, "type", "") == "image"]

        png = server.data.view_data("frame.h5")
        images = [b for b in png.content if getattr(b, "type", "") == "image"]
        assert len(images) == 1 and base64.b64decode(images[0].data)[:8] == b"\x89PNG\r\n\x1a\n"

    def test_view_data_errors_reach_the_agent_as_tool_errors(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)

        def missing(key, hints=True):
            raise data_tools.tiled_client.DataKeyNotFound(f"Tiled has no key {key!r}.")

        monkeypatch.setattr(server.data, "tiled_node", missing)
        with pytest.raises(ToolError, match="has no key 'nope.h5'"):
            server.data.view_data("nope.h5")

    def test_a_stuck_tiled_read_ends_at_the_deadline(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        monkeypatch.setattr("asyncroscopy.mcp.components.data_tools.DATA_TOOL_DEADLINE_S", 0.2)
        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        monkeypatch.setattr(server.data, "tiled_node", lambda key, hints=True: time.sleep(2))

        started = time.monotonic()
        with pytest.raises(ToolError, match="did not answer"):
            server.data.view_data("slow.h5")
        assert time.monotonic() - started < 1.5

    def test_list_data_lists_newest_first_and_filters(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FakeArray:
            def __init__(self, shape, metadata):
                self.shape, self.metadata, self.dtype = shape, metadata, np.dtype("float64")

            def read(self, slices=None):
                return np.zeros(self.shape)[slices] if slices else np.zeros(self.shape)

        class FakeContainer(dict):
            metadata: dict = {}

        spectrum = {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0, "detector": "eds"}
        client = FakeContainer(
            [
                ("old_image.h5", FakeContainer(image=FakeArray((8, 8), {"detector": "HAADF"}))),
                ("spectrum_eds.h5", FakeContainer(spectrum=FakeArray((64,), spectrum))),
                ("new_image.h5", FakeContainer(image=FakeArray((8, 8), {"detector": "HAADF"}))),
            ]
        )
        monkeypatch.setattr(data_tools.tiled_client, "uri_from_data_device", lambda address: "http://tiled")
        monkeypatch.setattr(data_tools.tiled_client, "connect", lambda uri: client)
        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)

        listed = server.data.list_data()
        assert [d["key"] for d in listed["data"]] == ["new_image.h5", "spectrum_eds.h5", "old_image.h5"]
        assert listed["data"][1]["kind"] == "spectrum" and listed["data"][1]["detector"] == "eds"
        assert [d["key"] for d in server.data.list_data(kind="image")["data"]] == ["new_image.h5", "old_image.h5"]
        assert [d["key"] for d in server.data.list_data(search="SPECTRUM")["data"]] == ["spectrum_eds.h5"]
        assert [d["key"] for d in server.data.list_data(limit=1, offset=1)["data"]] == ["spectrum_eds.h5"]


class TestMCPServerTypeMapping:
    def test_tango_types_map_to_python(self) -> None:
        assert MCPServer._tango_type_to_python(tango.CmdArgType.DevString) is str
        assert MCPServer._tango_type_to_python(tango.CmdArgType.DevVarDoubleArray) == list[float]
        assert MCPServer._tango_type_to_python(tango.CmdArgType.DevUChar) == np.uint8
        assert MCPServer._tango_type_to_python(tango.CmdArgType.DevEncoded) is dict


class TestMCPToolInvocation:
    def test_wrapper_supports_positional_and_keyword(self, monkeypatch) -> None:
        # Mock Database and DeviceProxy to avoid connection errors
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        # Mock objects for wrapping
        def mock_func(val):
            return val

        cmd_info = type(
            "CommandInfo",
            (),
            {
                "in_type": tango.CmdArgType.DevString,
                "out_type": tango.CmdArgType.DevString,
                "in_type_desc": ":param config_json: (not documented)\n:type config_json: DevString",
                "out_type_desc": "result",
            },
        )

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs())
        wrapper = server._create_wrapper(mock_func, cmd_info, "MyCmd", "MyClass")

        # 1. Positional call
        assert wrapper("hello") == "hello"

        # 2. Keyword call with the generic Tango argument name
        import inspect

        sig = inspect.signature(wrapper)
        param_name = list(sig.parameters.keys())[0]        
        assert param_name == "config_json" 
        assert wrapper(**{param_name: "world"}) == "world"

    def test_void_wrapper_supports_no_args(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        def mock_func():
            return "done"

        cmd_info = type(
            "CommandInfo",
            (),
            {
                "in_type": tango.CmdArgType.DevVoid,
                "out_type": tango.CmdArgType.DevString,
                "in_type_desc": "",
                "out_type_desc": "",
            },
        )

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs())
        wrapper = server._create_wrapper(mock_func, cmd_info, "VoidCmd", "MyClass")

        assert wrapper() == "done"


class TestMCPRegistration:
    def test_setup_registers_native_tools(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        calls = []

        def record_tool(method):
            calls.append(getattr(method, "name", None) or method.__name__)  # load_skill is added as a Tool (its description names the skills)

        monkeypatch.setattr(server.mcp, "add_tool", record_tool)
        monkeypatch.setattr(server, "_find_tools", lambda: {})

        server.setup(print_summary=False)

        assert set(calls) == {"list_data", "view_data", "list_devices", "list_skills", "load_skill", "endpoint_info", "action_space", "find_tools", "call_tool", "stop_all"}


class TestMCPCommandTimeout:
    def test_find_tools_raises_proxy_timeout_above_tango_default(self, monkeypatch) -> None:
        class FakeDb:
            def get_device_exported(self, pattern):
                return type("Result", (), {"value_string": ["asyncroscopy/twin/default"]})()

        timeouts = []

        class FakeProxy:
            def __init__(self, name):
                self.name = name

            def set_timeout_millis(self, millis):
                timeouts.append(millis)

            def info(self):
                return type("Info", (), {"dev_class": "Twin"})()

            def command_list_query(self):
                return []

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: FakeDb())
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", FakeProxy)

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        server._find_tools()

        assert timeouts == [COMMAND_TIMEOUT_MILLIS]
        assert COMMAND_TIMEOUT_MILLIS > 3000


class TestMCPSpectrumPreview:
    png_magic = b"\x89PNG\r\n\x1a\n"

    def test_acquire_spectrum_result_gains_png_preview(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FakeSpectrum:
            metadata = {"elements": json.dumps(["Au", "Pt", "Fe"])}

            def read(self):
                return np.array([0.5, 0.3, 0.2])

        class FakeContainer(dict):
            metadata: dict = {}

        node = FakeContainer(spectrum=FakeSpectrum())

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        monkeypatch.setattr(server.data, "tiled_node", lambda key, hints=True: {"spectrum_eds.h5": node}[key])
        result = server.data.augment_with_preview("acquire_spectrum", "spectrum_eds.h5")

        assert isinstance(result, ToolResult)
        assert result.structured_content == {"result": "spectrum_eds.h5"}
        image_blocks = [
            block for block in result.content if getattr(block, "type", "") == "image"
        ]
        assert len(image_blocks) == 1
        assert base64.b64decode(image_blocks[0].data)[: len(self.png_magic)] == self.png_magic

    def test_spectrum_preview_failure_names_the_reason(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FakeContainer(dict):
            metadata: dict = {}

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        monkeypatch.setattr(server.data, "tiled_node", lambda key, hints=True: {"spectrum_eds.h5": FakeContainer()}[key])
        result = server.data.augment_with_preview("acquire_spectrum", "spectrum_eds.h5")

        assert isinstance(result, ToolResult)
        assert result.structured_content == {"result": "spectrum_eds.h5"}
        texts = [getattr(block, "text", "") for block in result.content]
        assert any("image preview unavailable" in text for text in texts)
        assert any("holds no arrays" in text for text in texts)

    def test_unrelated_commands_keep_plain_results(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)

        assert server.data.augment_with_preview("get_stage", "[0.0,0.0]") == "[0.0,0.0]"
        assert server.data.augment_with_preview("acquire_spectrum", "") == ""
        assert server.data.augment_with_preview("acquire_spectrum", 42) == 42

    def test_other_acquire_commands_get_a_preview_but_no_failure_note(self, monkeypatch) -> None:
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FakeArray:
            metadata = {"axis_offset": 10.0, "axis_scale": 0.02, "axis_label": "two_theta", "axis_units": "deg"}

            def read(self):
                return np.sin(np.linspace(0, 6, 200)) ** 2

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        monkeypatch.setattr(server.data, "tiled_node", lambda key, hints=True: {"xrd_scan.h5": FakeArray()}[key])
        result = server.data.augment_with_preview("acquire_diffractogram", "xrd_scan.h5")
        assert isinstance(result, ToolResult)
        assert [getattr(b, "type", "") for b in result.content] == ["text", "text", "image"]
        assert result.content[0].text == "xrd_scan.h5" and "peak" in result.content[1].text.lower()  # the summary a text-only model reads

        def missing(key, hints=True):
            raise data_tools.tiled_client.DataKeyNotFound("no such key")

        monkeypatch.setattr(server.data, "tiled_node", missing)
        assert server.data.augment_with_preview("acquire_diffractogram", "ok") == "ok"


class TestDataToolRobustness:
    @pytest.fixture
    def server(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        return MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)

    def test_a_preview_that_breaks_never_fails_the_acquisition(self, server, monkeypatch) -> None:
        def broken(key, hints=True):
            raise RuntimeError("401 Unauthorized: wrong Tiled API key")

        monkeypatch.setattr(server.data, "tiled_node", broken)
        result = server.data.augment_with_preview("acquire_scanned_image", "scan_20261028.h5")
        assert isinstance(result, ToolResult) and result.structured_content == {"result": "scan_20261028.h5"}
        assert any("401 Unauthorized" in getattr(block, "text", "") for block in result.content)
        assert any("instead of acquiring again" in getattr(block, "text", "") for block in result.content)
        assert not any("try again" in getattr(block, "text", "").lower() for block in result.content)

    def test_an_acquisition_taken_with_the_beam_blanked_fails_naming_the_calls_that_retake_it(self, server, monkeypatch) -> None:
        monkeypatch.setattr(server.data, "tiled_node", lambda key, hints=True: None)
        blanked = {"text": "taken with the beam blanked", "beam_blanked": True}
        monkeypatch.setattr("asyncroscopy.mcp.components.data_tools.views.view", lambda node, key, fmt: (blanked, b"png"))
        with pytest.raises(ToolError) as raised:
            server.data.augment_with_preview("acquire_scanned_image", "scan_20261101.h5", "DigitalTwin", {"fov_m": 1e-6})
        error = json.loads(str(raised.value))
        assert error["error"]["code"] == "beam_blanked" and error["warnings"] == [{"code": "noise_only_file", "key": "scan_20261101.h5"}]
        assert error["next_actions"] == [{"tool": "DigitalTwin_unblank_beam", "args": {}}, {"tool": "DigitalTwin_acquire_scanned_image", "args": {"fov_m": 1e-6}}]
        # What the agent graphs show the model (asyncroscopy.agents.guidance): the retake first, the key last.
        from asyncroscopy.agents.guidance import render_error

        prose = render_error(f"Error calling tool 'DigitalTwin_acquire_scanned_image': ToolError: {raised.value}")
        assert "noise-only file scan_20261101.h5 was saved; do not view or report it" in prose
        assert prose.startswith("Error calling tool 'DigitalTwin_acquire_scanned_image': ToolError: Not acquired: the beam is blanked. Next call "
                                "DigitalTwin_unblank_beam, then DigitalTwin_acquire_scanned_image again")
        monkeypatch.setattr("asyncroscopy.mcp.components.data_tools.views.view", lambda node, key, fmt: ({"text": "512 x 512 pixels"}, b"png"))
        assert server.data.augment_with_preview("acquire_scanned_image", "scan_20261101.h5", "DigitalTwin").structured_content == {"result": "scan_20261101.h5"}

    def test_view_data_turns_unexpected_errors_into_tool_errors(self, server, monkeypatch) -> None:
        def broken(key, hints=True):
            raise KeyError("a plug-in kind looked up a missing attribute")

        monkeypatch.setattr(server.data, "tiled_node", broken)
        with pytest.raises(ToolError, match="could not read the data"):
            server.data.view_data("frame.h5")

    def test_only_results_that_look_like_data_keys_are_previewed(self, server, monkeypatch) -> None:
        looked_up = []
        monkeypatch.setattr(server.data, "tiled_node", lambda key, hints=True: looked_up.append(key))
        tableau = json.dumps({"C1": 1e-9, "A1": [2e-9, 0.1]})
        assert server.data.augment_with_preview("acquire_tableau", tableau) == tableau
        assert server.data.augment_with_preview("acquire_spectrum", "no such thing") == "no such thing"
        assert looked_up == []

    def test_a_full_lane_answers_busy_and_other_lanes_still_work(self, server) -> None:
        release = threading.Event()
        server.data.lanes["view"] = data_tools.Lane("view_data", 1)
        with pytest.raises(ToolError, match="did not answer"):
            server.data.with_deadline(release.wait, lane="view", deadline_s=0.1)
        with pytest.raises(ToolError, match="still waiting for Tiled"):
            server.data.with_deadline(lambda: "x", lane="view", deadline_s=0.1)
        assert server.data.with_deadline(lambda: "listed", lane="list", deadline_s=1.0) == "listed"
        release.set()
        time.sleep(0.2)
        assert server.data.with_deadline(lambda: "free again", lane="view", deadline_s=1.0) == "free again"

    def test_list_data_keeps_going_past_unreadable_entries_and_reports_more_exactly(self, server, monkeypatch) -> None:
        class FakeArray:
            metadata: dict = {}
            dtype = np.dtype("float64")

            def __init__(self, shape):
                self.shape = shape

            def read(self, slices=None):
                return np.zeros(self.shape)

        class Broken:
            metadata: dict = {}
            structure_family = "array"

            @property
            def shape(self):
                raise RuntimeError("500 Internal Server Error")

        client = {"a.h5": FakeArray((4, 4)), "bad.h5": Broken(), "c.h5": FakeArray((4, 4))}
        monkeypatch.setattr(data_tools.tiled_client, "uri_from_data_device", lambda address: "http://tiled")
        monkeypatch.setattr(data_tools.tiled_client, "connect", lambda uri: client)

        listed = server.data.list_data()
        assert [(d["key"], d["kind"]) for d in listed["data"]] == [("c.h5", "image"), ("bad.h5", "unreadable"), ("a.h5", "image")]
        assert listed["more"] is False and listed["next_offset"] is None
        assert listed["looked"] == "looked at keys 0 to 3 of the newest 3 (of 3 in all)"
        assert server.data.list_data(limit=3)["more"] is False
        page = server.data.list_data(limit=2)
        assert [d["key"] for d in page["data"]] == ["c.h5", "bad.h5"] and page["more"] is True
        assert [d["key"] for d in server.data.list_data(limit=2, offset=2)["data"]] == ["a.h5"]
        assert server.data.list_data(limit=2, offset=2)["more"] is False
        assert page["next_offset"] == 2
        with pytest.raises(ToolError, match="unknown kind 'imgae'"):
            server.data.list_data(kind="imgae")

    def test_list_data_cut_short_by_its_budget_resumes_where_it_stopped(self, server, monkeypatch) -> None:
        class FakeArray:
            metadata: dict = {}
            dtype = np.dtype("float64")
            shape = (4, 4)

            def read(self, slices=None):
                return np.zeros(self.shape)

        client = {f"k{i}.h5": FakeArray() for i in range(5)}
        monkeypatch.setattr(data_tools.tiled_client, "uri_from_data_device", lambda address: "http://tiled")
        monkeypatch.setattr(data_tools.tiled_client, "connect", lambda uri: client)
        monkeypatch.setattr(data_tools, "LIST_DATA_BUDGET_S", 0.0)
        pages, offset = 0, 0
        while True:
            page = server.data.list_data(kind="spectrum", offset=offset)
            pages += 1
            if not page["more"]:
                break
            assert page["next_offset"] > offset and "continue with offset" in page["looked"]
            offset = page["next_offset"]
        assert pages == 5


class TestDataViewsHook:
    def test_data_views_accepts_one_module_or_a_list(self, tmp_path) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("launcher_run_mcp_views", Path(__file__).resolve().parents[1] / "startup_scripts" / "run_mcp.py")
        run_mcp = importlib.util.module_from_spec(spec)
        sys.modules["launcher_run_mcp_views"] = run_mcp
        spec.loader.exec_module(run_mcp)
        base = (Path(__file__).resolve().parents[1] / "configs" / "mcp_dt.yaml").read_text(encoding="utf-8")
        for value, expected in (("mypkg.views", ["mypkg.views"]), ("[a.views, b.views]", ["a.views", "b.views"])):
            path = tmp_path / "mcp.yaml"
            path.write_text(base.replace("\nmcp:\n", f"\nmcp:\n  data_views: {value}\n", 1), encoding="utf-8")
            config = run_mcp.load_config(path)
            assert config.mcp.data_views == expected
            assert "--data-views-json" in run_mcp.build_command(config)
        # The agent endpoint's config names the analysis server, so wait_for_job covers its jobs too.
        command = run_mcp.build_command(run_mcp.load_config(Path(__file__).resolve().parents[1] / "configs" / "mcp_dt.yaml"))
        assert command[command.index("--analysis-url") + 1] == "http://127.0.0.1:8003/mcp"

    def test_a_data_views_module_that_does_not_import_stops_the_server_with_its_name(self, capsys) -> None:
        code = mcp_server.main([
            "--name", "t", "--tango-host", "h", "--tango-port", "1", "--transport", "stdio", "--http-host", "h", "--http-port", "1",
            "--blocked-classes-json", "[]", "--blocked-functions-json", "{}", "--data-device-address", "d",
            "--data-views-json", '["no_such_views_module_xyz"]',
        ])
        assert code == 1
        assert "MCP ERROR: mcp.data_views lists 'no_such_views_module_xyz'" in capsys.readouterr().out


def test_list_data_never_asks_tiled_for_a_page_larger_than_it_serves():
    from tiled.server.dependencies import MAX_PAGE_SIZE

    from asyncroscopy.mcp.components.data_tools import LIST_DATA_WINDOW

    assert LIST_DATA_WINDOW <= MAX_PAGE_SIZE
