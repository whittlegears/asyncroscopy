"""The bridge exposes Tango attributes as get_/set_ tools and can refresh discovery."""

import asyncio
import json

import numpy as np
import pytest
import tango
from fastmcp.tools import ToolResult
from tango import AttrDataFormat, AttrWriteType, CmdArgType

from asyncroscopy.mcp import mcp_server
from asyncroscopy.mcp.mcp_server import (
    COMMAND_TIMEOUT_MILLIS,
    COMMAND_TIMEOUT_OVERRIDES_MILLIS,
    MCPServer,
    attribute_tool_infos,
    attribute_value_type,
    plain_attribute_value,
)


def mcp_kwargs(**overrides):
    config = {
        "blocked_classes": ["DataBase", "DServer"],
        "blocked_functions": {"*": ["Init"]},
        "data_device_address": "asyncroscopy/data/default",
    }
    config.update(overrides)
    return config


class FakeAttrInfo:
    def __init__(self, name, data_type, data_format=AttrDataFormat.SCALAR, writable=AttrWriteType.READ_WRITE,
                 description="", unit="", min_value="Not specified", max_value="Not specified"):
        self.name = name
        self.data_type = data_type
        self.data_format = data_format
        self.writable = writable
        self.description = description
        self.unit = unit
        self.min_value = min_value
        self.max_value = max_value


class FakeAttrValue:
    def __init__(self, value):
        self.value = value


class FakeScanProxy:
    """SCAN-like device: attributes only, no commands beyond State/Status."""

    attributes = [
        FakeAttrInfo("dwell_time", CmdArgType.DevDouble, description="Dwell time", unit="s", min_value="1e-07"),
        FakeAttrInfo("imsize", CmdArgType.DevLong),
        FakeAttrInfo("scan_region", CmdArgType.DevDouble, data_format=AttrDataFormat.SPECTRUM),
        FakeAttrInfo("output_format", CmdArgType.DevString),
        FakeAttrInfo("State", CmdArgType.DevState, writable=AttrWriteType.READ),
        FakeAttrInfo("Status", CmdArgType.DevString, writable=AttrWriteType.READ),
        FakeAttrInfo("readonly_flag", CmdArgType.DevBoolean, writable=AttrWriteType.READ),
    ]
    store = {"dwell_time": 1e-6, "imsize": 256, "scan_region": np.array([0.0, 0.0, 1.0, 1.0]),
             "output_format": ".h5", "readonly_flag": True}
    timeouts: list = []
    writes: list = []

    def __init__(self, name):
        self.name = name

    def set_timeout_millis(self, millis):
        FakeScanProxy.timeouts.append((self.name, millis))

    def info(self):
        return type("Info", (), {"dev_class": "SCAN"})()

    def command_list_query(self):
        return [type("Cmd", (), {"cmd_name": name, "in_type": CmdArgType.DevVoid, "out_type": CmdArgType.DevString,
                                 "in_type_desc": "", "out_type_desc": ""})() for name in ("State", "Status")]

    def attribute_list_query_ex(self):
        return list(self.attributes)

    def read_attribute(self, name):
        return FakeAttrValue(FakeScanProxy.store[name])

    def write_attribute(self, name, value):
        FakeScanProxy.writes.append((name, value))
        FakeScanProxy.store[name] = value

    def command_inout(self, name, *args):
        return f"{name}-called"


@pytest.fixture
def scan_server(monkeypatch):
    FakeScanProxy.timeouts = []
    FakeScanProxy.writes = []
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", FakeScanProxy)
    server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
    monkeypatch.setattr(server, "_list_all_devices", lambda: ["asyncroscopy/scan/default"])
    return server


class TestAttributeTypeMapping:
    def test_scalar_spectrum_and_image(self):
        assert attribute_value_type(FakeAttrInfo("a", CmdArgType.DevDouble)) == CmdArgType.DevDouble
        assert attribute_value_type(FakeAttrInfo("a", CmdArgType.DevDouble, AttrDataFormat.SPECTRUM)) == CmdArgType.DevVarDoubleArray
        assert attribute_value_type(FakeAttrInfo("a", CmdArgType.DevString, AttrDataFormat.SPECTRUM)) == CmdArgType.DevVarStringArray
        assert attribute_value_type(FakeAttrInfo("a", CmdArgType.DevUChar, AttrDataFormat.IMAGE)) == CmdArgType.DevEncoded

    def test_tool_infos_describe_and_respect_writability(self):
        infos = attribute_tool_infos(FakeAttrInfo("dwell_time", CmdArgType.DevDouble, description="Dwell", unit="s", max_value="1"))
        assert [(info.mode, info.in_type, info.out_type) for info in infos] == [
            ("read", CmdArgType.DevVoid, CmdArgType.DevDouble),
            ("write", CmdArgType.DevDouble, CmdArgType.DevVoid),
        ]
        assert infos[0].out_type_desc == "Dwell; unit: s; max: 1"
        assert infos[1].in_type_desc.startswith(":param dwell_time:")

        readonly = attribute_tool_infos(FakeAttrInfo("flag", CmdArgType.DevBoolean, writable=AttrWriteType.READ))
        assert [info.mode for info in readonly] == ["read"]

    def test_plain_attribute_value(self):
        assert plain_attribute_value(tango.DevState.ON) == "ON"
        assert plain_attribute_value(np.array([1, 2])) == [1, 2]
        assert plain_attribute_value(np.float64(1.5)) == 1.5
        assert plain_attribute_value(("a", np.int32(3))) == ["a", 3]


class TestAttributeDiscovery:
    def test_attributes_become_get_and_set_tools(self, scan_server):
        tools = scan_server._find_tools()["SCAN"]

        assert "get_dwell_time" in tools and "set_dwell_time" in tools
        assert "get_scan_region" in tools and "set_scan_region" in tools
        assert "get_readonly_flag" in tools and "set_readonly_flag" not in tools
        # State/Status are already commands; their attribute twins are skipped.
        assert "get_State" not in tools and "get_Status" not in tools
        assert "State" in tools

        get_dwell, info, device_name = tools["get_dwell_time"]
        assert device_name == "asyncroscopy/scan/default"
        assert info.attribute_name == "dwell_time"
        assert get_dwell() == 1e-6

        set_dwell, _, _ = tools["set_dwell_time"]
        set_dwell(2e-6)
        assert FakeScanProxy.writes == [("dwell_time", 2e-6)]
        assert get_dwell() == 2e-6

        get_region, _, _ = tools["get_scan_region"]
        assert get_region() == [0.0, 0.0, 1.0, 1.0]

    def test_blocklist_applies_to_attribute_tools(self, scan_server):
        scan_server.blocked_functions["SCAN"] = ["set_output_format"]
        tools = scan_server._find_tools()["SCAN"]
        assert "get_output_format" in tools
        assert "set_output_format" not in tools

    def test_wrapped_setter_has_attribute_named_parameter(self, scan_server):
        tools = scan_server._find_tools()["SCAN"]
        func, info, device_name = tools["set_dwell_time"]
        wrapper = scan_server._create_wrapper(func, info, "set_dwell_time", "SCAN", device_name)

        import inspect

        assert list(inspect.signature(wrapper).parameters) == ["dwell_time"]
        assert wrapper.__name__ == "SCAN_set_dwell_time"
        assert "Tango Attribute: dwell_time" in wrapper.__doc__
        wrapper(dwell_time=5e-6)
        assert FakeScanProxy.store["dwell_time"] == 5e-6

    def test_registered_tools_are_callable_through_fastmcp(self, scan_server):
        scan_server.setup(print_summary=False)
        names = {tool.name for tool in asyncio.run(scan_server.mcp.list_tools())}
        assert {"SCAN_get_dwell_time", "SCAN_set_imsize", "list_acquisitions", "refresh_devices"} <= names

        result = asyncio.run(scan_server.mcp.call_tool("SCAN_get_imsize", {}))
        payload = result.structured_content if isinstance(result, ToolResult) else result
        assert 256 in json.loads(json.dumps(payload)).values() if isinstance(payload, dict) else payload == 256


class TestAttributeRebuild:
    def test_stale_attribute_proxy_is_rebuilt_via_attribute_read(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FreshProxy:
            def set_timeout_millis(self, ms):
                pass

            def read_attribute(self, name):
                assert name == "imsize"
                return FakeAttrValue(np.int64(512))

            def command_inout(self, *args):
                raise AssertionError("attribute tools must not fall back to command_inout")

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", lambda name: FreshProxy())
        info = attribute_tool_infos(FakeAttrInfo("imsize", CmdArgType.DevLong))[0]

        def stale():
            raise RuntimeError("Caught an unknown exception!")

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        wrapper = server._create_wrapper(stale, info, "get_imsize", "SCAN", "asyncroscopy/scan/default")
        assert wrapper() == 512


class TestTimeoutOverrides:
    def test_long_data_commands_get_their_own_proxy_timeout(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
        proxies = []

        class FakeDataProxy:
            def __init__(self, name):
                self.name = name
                self.timeout = None
                proxies.append(self)

            def set_timeout_millis(self, millis):
                self.timeout = millis

            def info(self):
                return type("Info", (), {"dev_class": "DATA"})()

            def command_list_query(self):
                return [type("Cmd", (), {"cmd_name": name})() for name in ("get_config", "register_save_path")]

            def command_inout(self, name, *args):
                return (self.timeout, name)

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", FakeDataProxy)
        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        monkeypatch.setattr(server, "_list_all_devices", lambda: ["asyncroscopy/data/default"])

        tools = server._find_tools()["DATA"]
        assert tools["get_config"][0]() == (COMMAND_TIMEOUT_MILLIS, "get_config")
        assert tools["register_save_path"][0]() == (COMMAND_TIMEOUT_OVERRIDES_MILLIS["register_save_path"], "register_save_path")
        assert COMMAND_TIMEOUT_OVERRIDES_MILLIS["register_save_path"] > COMMAND_TIMEOUT_MILLIS


class TestRefreshDevices:
    def test_refresh_registers_devices_started_after_setup(self, scan_server, monkeypatch):
        monkeypatch.setattr(scan_server, "_list_all_devices", lambda: [])
        scan_server.setup(print_summary=False)
        assert "SCAN" not in scan_server.tools

        monkeypatch.setattr(scan_server, "_list_all_devices", lambda: ["asyncroscopy/scan/default"])
        summary = scan_server.refresh_devices()

        assert "get_dwell_time" in summary["tools"]["SCAN"]
        assert summary["skipped_devices"] == {}
        names = {tool.name for tool in asyncio.run(scan_server.mcp.list_tools())}
        assert "SCAN_get_dwell_time" in names

        # A second refresh replaces rather than duplicates.
        scan_server.refresh_devices()
        names_again = [tool.name for tool in asyncio.run(scan_server.mcp.list_tools())]
        assert names_again.count("SCAN_get_dwell_time") == 1


class TestKeyListPreview:
    def test_tiff_key_list_previews_first_key_and_keeps_list(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FakeArray:
            def read(self):
                return np.arange(4, dtype=np.uint16).reshape(2, 2)

        class FakeDataProxy:
            def get_config(self):
                return json.dumps({"uri": "http://microscope:9091"})

        resolved = []

        class FakeClient(dict):
            def __getitem__(self, key):
                resolved.append(key)
                return super().__getitem__(key)

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", lambda address: FakeDataProxy())
        monkeypatch.setattr(
            "asyncroscopy.mcp.mcp_server.open_client",
            lambda uri, api_key=None: FakeClient({"stem_20260904T1_HAADF.tiff": FakeArray()}),
        )
        keys = json.dumps(["stem_20260904T1_HAADF.tiff", "stem_20260904T1_BF-S.tiff"])

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        result = server._augment_with_preview("acquire_scanned_image", keys)

        assert isinstance(result, ToolResult)
        assert result.structured_content == {"result": keys}
        assert resolved == ["stem_20260904T1_HAADF.tiff"]
        assert any(getattr(block, "type", "") == "image" for block in result.content)


class TestListAcquisitionsTool:
    def test_lists_via_data_device_uri(self, monkeypatch):
        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)

        class FakeDataProxy:
            def get_config(self):
                return json.dumps({"uri": "http://microscope:9091"})

        class Node:
            metadata = {"instrument_class": "DigitalTwin"}

        monkeypatch.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", lambda address: FakeDataProxy())
        monkeypatch.setattr(
            "asyncroscopy.mcp.mcp_server.open_client",
            lambda uri, api_key=None: {"stem_image_HAADF_20260904T100000000000.h5": Node()},
        )

        server = MCPServer("test", "localhost", 1234, **mcp_kwargs(), verbose=False)
        listed = server.list_acquisitions(acquisition_type="stem_image")
        assert listed == [
            {
                "key": "stem_image_HAADF_20260904T100000000000.h5",
                "timestamp": "2026-09-04T10:00:00",
                "metadata": {"instrument_class": "DigitalTwin"},
            }
        ]
