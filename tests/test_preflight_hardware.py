"""The hardware pre-flight against fakes: what it reports, and that it sends no command that acts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from asyncroscopy.safety.policy import is_read_only
from startup_scripts import preflight_hardware as preflight

ROOT = Path(__file__).resolve().parents[1]
SPECTRA = ROOT / "configs" / "Spectra300.yaml"
MCP_YAMLS = [ROOT / "configs" / "mcp.yaml", ROOT / "configs" / "mcp_operator.yaml"]
KEY = "lab-test-key"


class FakeValue:
    def __init__(self, value):
        self.value = value


class FakeProxy:
    """Answers the read commands the pre-flight may send; records every call, and any other attribute touched."""

    def __init__(self, device_class: str, log: list, state: str = "ON", status: str = "ok", replies: dict | None = None):
        self._class, self._log, self._state, self._status = device_class, log, state, status
        self._replies = replies or {}

    def set_timeout_millis(self, timeout: int) -> None:
        self._log.append((self._class, "set_timeout_millis"))

    def command_inout(self, name: str):
        self._log.append((self._class, name))
        if name == "State":
            return self._state
        if name == "Status":
            return self._status
        if name not in self._replies:
            raise RuntimeError(f"fake has no reply for {name}")
        return self._replies[name]

    def read_attribute(self, name: str):
        self._log.append((self._class, f"read:{name}"))
        return FakeValue(self._replies[f"attr:{name}"])

    def __getattr__(self, name: str):  # write_attribute, any command called as a method, ...
        self.__dict__.setdefault("_log", []).append((self.__dict__.get("_class"), f"FORBIDDEN:{name}"))
        raise AssertionError(f"the pre-flight touched {name}")


PARAMETERS = {
    "optical_mode": "Stem", "projector_mode": "Imaging", "acceleration_voltage": 300000.0, "column_valves": "Closed",
    "beam_current_pa": 50.0, "beam_current_source": "default", "beam_current_reason": "no screen-current measurement since connecting",
}


class Stack:
    """A fake hardware stack: devices by name, open TCP ports, Tiled and MCP answers."""

    def __init__(self):
        self.log: list = []
        self.states: dict[str, str] = {}
        self.approvals_mode = "approve"
        self.server_version = "1.18.0.5236"
        self.client_version = "1.18.0"
        self.tiled_key = KEY
        self.open_ports = {("10.46.217.241", 9095), ("10.46.217.241", 9092), ("127.0.0.1", 9091), ("127.0.0.1", 8003)}
        self.mcp_calls: list[str] = []
        self.databases: set[str] = {"10.46.217.241", "localhost"}

    def probes(self) -> preflight.Probes:
        return preflight.Probes(
            database=self.database, device=self.device, tcp=self.tcp, http_get=self.http_get, mcp_tools=self.mcp_tools,
            autoscript_client_version=lambda: self.client_version, autoscript_server_version=lambda host, port: self.server_version,
        )

    def database(self, host: str, port: int) -> str:
        if host not in self.databases:
            raise ConnectionRefusedError(f"no database at {host}")
        return f"{host}:{port}"

    def device(self, trl: str) -> FakeProxy:
        name = trl.split("/", 3)[-1]
        key = name.split("/")[1]
        replies = {}
        device_class = key.upper()
        if key == "instrument":
            device_class = "AutoScriptMicroscope"
            replies = {"get_stage": [1e-6, -2e-6, 0.0, 0.0, 0.0], "get_parameters": json.dumps(PARAMETERS), "attr:beam_state": True}
        elif key == "approvals":
            device_class = "Approvals"
            replies = {"get_mode": self.approvals_mode, "get_simulated": False}
        state = self.states.get(key, "ON")
        return FakeProxy(device_class, self.log, state=state, status=f"{key} status text", replies=replies)

    def tcp(self, host: str, port: int) -> None:
        if (host, port) not in self.open_ports:
            raise ConnectionRefusedError(f"{host}:{port} refused")

    def http_get(self, url: str, headers: dict[str, str]) -> tuple[int, str]:
        if url.endswith("/api/v1/"):
            return 200, json.dumps({"api_version": 0, "library_version": "0.2.9"})
        if headers.get("Authorization") != f"Apikey {self.tiled_key}":
            return 401, '{"detail": "Invalid API key"}'
        return 200, "{}"

    def mcp_tools(self, url: str, headers: dict[str, str]) -> list[str]:
        self.mcp_calls.append(url)
        return ["list_devices", "endpoint_info"]


@pytest.fixture
def stack(monkeypatch) -> Stack:
    monkeypatch.setenv("ASYNCROSCOPY_TILED_API_KEY", KEY)
    monkeypatch.delenv("ASYNCROSCOPY_MCP_TOKEN", raising=False)
    return Stack()


def results(report: preflight.Report) -> dict[str, tuple[str, str]]:
    return {name: (result, detail) for result, name, detail in report.rows}


def test_a_healthy_stack_passes_and_only_reads(stack):
    report = preflight.run(SPECTRA, MCP_YAMLS, stack.probes())

    assert report.failures == []
    rows = results(report)
    assert rows["approvals mode"][0] == "PASS"
    assert rows["stage position"][0] == "PASS" and "x=1.000 um" in rows["stage position"][1]
    assert rows["beam current"][0] == "WARN"  # not measured yet: said, not failed
    assert rows["autoscript server version"][0] == "PASS"
    assert "7521" in rows["autoscript server port 10.46.217.241:9095"][1]
    assert rows["tiled accepts the API key"][0] == "PASS"
    # The MCP endpoints were not running: skipped, not failed.
    assert [r for r, name, _ in report.rows if name.startswith("mcp ")] == ["SKIP", "SKIP"]
    # Every configured device was asked for its State.
    assert sum(1 for _cls, name in stack.log if name == "State") == 11

    sent = [(cls, name) for cls, name in stack.log if name != "set_timeout_millis" and not name.startswith("read:")]
    assert sent and all(is_read_only(cls, name) for cls, name in sent), sent
    assert not [entry for entry in stack.log if "FORBIDDEN" in entry[1]]
    assert {name for _cls, name in stack.log if name.startswith("read:")} == {"read:beam_state"}


@pytest.mark.parametrize("command", ["move_stage", "unblank_beam", "blank_beam", "set_mode", "acquire_scanned_image", "approve", "Init"])
def test_read_only_device_refuses_any_command_that_acts_before_sending_it(command):
    log: list = []
    device = preflight.ReadOnlyDevice(FakeProxy("AutoScriptMicroscope", log), "AutoScriptMicroscope")
    with pytest.raises(preflight.NotReadOnly):
        device.command(command)
    assert (("AutoScriptMicroscope", command)) not in log
    with pytest.raises(preflight.NotReadOnly):
        device.attribute("position")


def test_a_faulted_device_fails_and_its_status_is_printed(stack):
    stack.states["stage"] = "FAULT"

    rows = results(preflight.run(SPECTRA, MCP_YAMLS, stack.probes()))

    result, detail = rows["device asyncroscopy/stage/default"]
    assert result == "FAIL" and "FAULT" in detail and "stage status text" in detail


def test_auto_mode_on_hardware_fails(stack):
    stack.approvals_mode = "auto"
    report = preflight.run(SPECTRA, MCP_YAMLS, stack.probes())
    assert "approvals mode" in report.failures


def test_tango_only_on_the_lan_address_fails_localhost(stack):
    stack.databases = {"10.46.217.241"}
    report = preflight.run(SPECTRA, MCP_YAMLS, stack.probes())
    assert report.failures == ["tango database at localhost:9094"]


def test_the_default_or_a_wrong_tiled_key_fails(stack, monkeypatch):
    stack.tiled_key = "the-key-run_servers-used"
    report = preflight.run(SPECTRA, MCP_YAMLS, stack.probes())
    assert "tiled accepts the API key" in report.failures

    monkeypatch.setenv("ASYNCROSCOPY_TILED_API_KEY", "secret")
    report = preflight.run(SPECTRA, MCP_YAMLS, stack.probes())
    assert "config: ASYNCROSCOPY_TILED_API_KEY" in report.failures


def test_a_simulated_config_fails(stack, tmp_path):
    raw = yaml.safe_load(SPECTRA.read_text(encoding="utf-8"))
    raw["simulated"] = True
    path = tmp_path / "sim.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    report = preflight.run(path, [], stack.probes())
    assert "config: simulated: false" in report.failures


def test_autoscript_server_and_client_versions_must_match(stack):
    stack.server_version = "1.21.0.100"
    report = preflight.run(SPECTRA, MCP_YAMLS, stack.probes())
    assert report.failures == ["autoscript server version"]


def test_autoscript_server_is_not_contacted_when_asked_not_to(stack):
    def refuse(host, port):
        raise AssertionError("contacted the AutoScript server")

    probes = stack.probes()
    probes.autoscript_server_version = refuse
    rows = results(preflight.run(SPECTRA, MCP_YAMLS, probes, ask_autoscript_server=False))
    assert rows["autoscript server version"][0] == "SKIP"


def test_unreachable_ceos_and_analysis_server_fail_and_running_endpoints_are_listed(stack):
    stack.open_ports -= {("10.46.217.241", 9092), ("127.0.0.1", 8003)}
    stack.open_ports |= {("127.0.0.1", 8000), ("127.0.0.1", 8001)}

    report = preflight.run(SPECTRA, MCP_YAMLS, stack.probes())

    assert set(report.failures) == {"ceos at 10.46.217.241:9092", "analysis server (mapper) at http://127.0.0.1:8003/mcp"}
    assert stack.mcp_calls == ["http://127.0.0.1:8000/mcp", "http://127.0.0.1:8001/mcp"]
    rows = results(report)
    assert rows["mcp agent endpoint (mcp.yaml) at http://127.0.0.1:8000/mcp"] == ("PASS", "2 tools")


def test_main_exits_nonzero_on_any_failure(monkeypatch, capsys):
    failed = preflight.Report()
    failed.add("FAIL", "something")
    monkeypatch.setattr(preflight, "run", lambda *args, **kwargs: failed)
    assert preflight.main(["--yaml", str(SPECTRA)]) == 1
    assert "PRE-FLIGHT FAILED" in capsys.readouterr().out

    passed = preflight.Report()
    passed.add("WARN", "something")
    monkeypatch.setattr(preflight, "run", lambda *args, **kwargs: passed)
    assert preflight.main(["--yaml", str(SPECTRA)]) == 0
