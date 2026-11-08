#!/usr/bin/env python
"""Read-only pre-flight for a hardware stack: is everything up, and is it what we think it is?

Run it after run_servers.py (and after the analysis server), on the microscope PC:

    uv run --no-sync startup_scripts/preflight_hardware.py --yaml configs/Spectra300.yaml

It never moves, blanks or unblanks, acquires, writes an attribute or switches the approval mode.
Every Tango command it sends goes through ``ReadOnlyDevice``, which refuses any command the risk
policy does not count as a pure read (``asyncroscopy.safety.policy.is_read_only``). It reads:

* the YAML: ``simulated: false``, and an ASYNCROSCOPY_TILED_API_KEY that is set and not "secret";
* the Tango database at tango.host and at localhost;
* every configured device: it answers, and its State is not FAULT or UNKNOWN (Status is printed when not ON);
* the AutoScript client version in this Python, and the server's version (its own short AutoScript connection,
  which only reads ``service.autoscript.server.version``; skip it with --no-autoscript-server);
* the microscope: stage position, optical mode, beam blanked, beam current if the screen current was measured;
* the Approvals device: mode is approve, and it says the stack is not simulated;
* CEOS: a TCP connect to the corrector's ceos_host:ceos_port, nothing sent;
* Tiled at the DATA device's address, with the API key (an authenticated read);
* the analysis server at the Mapper's analysis_url (MCP initialize and list_tools);
* the MCP endpoints in --mcp-yaml (by default configs/mcp.yaml and configs/mcp_operator.yaml): list_tools,
  or SKIP when nothing listens yet.

It prints one line per check and exits 1 when any check FAILs, 0 otherwise (WARN and SKIP do not fail).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy import settings
from asyncroscopy.core.net import WILDCARD_HOSTS, is_loopback_host
from asyncroscopy.safety.policy import is_read_only

DEFAULT_YAML = PROJECT_DIR / "configs" / "Spectra300.yaml"
DEFAULT_MCP_YAMLS = (PROJECT_DIR / "configs" / "mcp.yaml", PROJECT_DIR / "configs" / "mcp_operator.yaml")
EXPECTED_AUTOSCRIPT = "1.18"  # the client wheels pyproject.toml installs (stubs/AutoScript_v_1.18)
AUTOSCRIPT_DEFAULT_PORT = 7521
APPROVALS_MODULE = "asyncroscopy.safety.approvals"
DEVICE_TIMEOUT_MS = 5000
NET_TIMEOUT_S = 5.0
MCP_TIMEOUT_S = 15.0

# Attributes this script may read. Attribute reads never act, but the list keeps the script's reach explicit.
READ_ATTRIBUTES = frozenset({"beam_state"})
BAD_STATES = frozenset({"FAULT", "UNKNOWN"})


class ProxyLike(Protocol):
    def set_timeout_millis(self, timeout: int) -> None: ...
    def command_inout(self, name: str) -> Any: ...
    def read_attribute(self, name: str) -> Any: ...


class NotReadOnly(RuntimeError):
    """The pre-flight was asked to send a command that is not a pure read; it refuses before sending."""


class ReadOnlyDevice:
    """A Tango device proxy that only sends commands the risk policy counts as reads (get_*, list_*, State, Status)."""

    def __init__(self, proxy: ProxyLike, device_class: str) -> None:
        self._proxy = proxy
        self._class = device_class
        proxy.set_timeout_millis(DEVICE_TIMEOUT_MS)

    def command(self, name: str) -> Any:
        if not is_read_only(self._class, name):
            raise NotReadOnly(f"{self._class}.{name} is not a read-only command; the pre-flight never sends it")
        return self._proxy.command_inout(name)

    def attribute(self, name: str) -> Any:
        if name not in READ_ATTRIBUTES:
            raise NotReadOnly(f"{self._class}.{name} is not in the pre-flight's list of attributes to read")
        return self._proxy.read_attribute(name).value

    def state(self) -> str:
        return str(self.command("State")).rsplit(".", 1)[-1].upper()

    def status(self) -> str:
        return str(self.command("Status")).strip()


@dataclass
class Probes:
    """Everything that touches the network, so tests can replace it. The defaults are the real calls."""

    database: Callable[[str, int], str]
    device: Callable[[str], ProxyLike]
    tcp: Callable[[str, int], None]
    http_get: Callable[[str, dict[str, str]], tuple[int, str]]
    mcp_tools: Callable[[str, dict[str, str]], list[str]]
    autoscript_client_version: Callable[[], str]
    autoscript_server_version: Callable[[str, int], str]


def _database(host: str, port: int) -> str:
    import tango

    database = tango.Database(host, port)
    return f"{database.get_db_host()}:{database.get_db_port()}"


def _device(trl: str) -> ProxyLike:
    import tango

    return tango.DeviceProxy(trl)


def _tcp(host: str, port: int) -> None:
    socket.create_connection((host, port), timeout=NET_TIMEOUT_S).close()


def _http_get(url: str, headers: dict[str, str]) -> tuple[int, str]:
    try:
        with urlopen(Request(url, headers=headers), timeout=NET_TIMEOUT_S) as response:
            return response.status, response.read(2000).decode("utf-8", "replace")
    except HTTPError as exc:
        return exc.code, exc.read(2000).decode("utf-8", "replace")


def _mcp_tools(url: str, headers: dict[str, str]) -> list[str]:
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    async def tools() -> list[str]:
        async with Client(StreamableHttpTransport(url, headers=headers), timeout=MCP_TIMEOUT_S) as client:
            return [tool.name for tool in await client.list_tools()]

    return asyncio.run(asyncio.wait_for(tools(), MCP_TIMEOUT_S))


def _autoscript_client_version() -> str:
    import autoscript_tem_microscope_client  # noqa: F401  (importable, not only installed)

    return importlib.metadata.version("autoscript-tem-microscope-client")


def _autoscript_server_version(host: str, port: int) -> str:
    from autoscript_tem_microscope_client import TemMicroscopeClient

    client = TemMicroscopeClient()
    client.connect(host, port)
    try:
        return str(client.service.autoscript.server.version)
    finally:
        client.disconnect()  # this script's own connection only; the instrument device keeps its own


REAL_PROBES = Probes(
    database=_database,
    device=_device,
    tcp=_tcp,
    http_get=_http_get,
    mcp_tools=_mcp_tools,
    autoscript_client_version=_autoscript_client_version,
    autoscript_server_version=_autoscript_server_version,
)


@dataclass
class Report:
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, result: str, name: str, detail: str = "") -> None:
        self.rows.append((result, name, detail))
        print(f"{result:<5} {name:<44} {detail}", flush=True)

    @property
    def failures(self) -> list[str]:
        return [name for result, name, _ in self.rows if result == "FAIL"]


def _error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    for line in text:  # a Tango DevFailed prints many lines; the "desc" one says what happened
        if "desc" in line.lower():
            return line.strip()[:240]
    return f"{type(exc).__name__}: {text[0][:240] if text else ''}"


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.strip().split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _local(host: str) -> str:
    return "127.0.0.1" if host.strip() in WILDCARD_HOSTS else host


def _mcp_headers() -> dict[str, str]:
    """The caller name for the trace, and a bearer token when the endpoint verifies tokens (ASYNCROSCOPY_MCP_TOKEN)."""
    headers = {"X-Asyncroscopy-Caller": "preflight_hardware"}
    token = (settings.get().mcp_token or "").strip()
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    return headers


def check_config(config: Any, report: Report) -> None:
    if config.simulated:
        report.add("FAIL", "config: simulated: false", f"{config.path.name} says simulated: true; this pre-flight is for hardware")
    else:
        report.add("PASS", "config: simulated: false", config.path.name)
    key = settings.get().tiled_api_key or ""
    if not key or key == "secret":
        report.add("FAIL", "config: ASYNCROSCOPY_TILED_API_KEY", "unset or the default 'secret': set the lab key in this shell and in every other process")
    else:
        report.add("PASS", "config: ASYNCROSCOPY_TILED_API_KEY", "set (not printed)")


def check_database(config: Any, probes: Probes, report: Report) -> bool:
    reachable = False
    for host in dict.fromkeys([config.tango_host, "localhost"]):
        name = f"tango database at {host}:{config.tango_port}"
        try:
            report.add("PASS", name, probes.database(host, config.tango_port))
            reachable = True
        except Exception as exc:
            report.add("FAIL", name, _error(exc))
    return reachable


def _devices(config: Any) -> list[tuple[str, str, dict[str, list[str]]]]:
    """(device name, class, properties) for every device run_servers starts, the instrument last."""
    support = [(d.device_name, d.class_name, d.properties) for d in config.support_devices]
    return [*support, ("asyncroscopy/instrument/default", config.instrument.class_name, config.instrument.properties)]


def check_devices(config: Any, probes: Probes, report: Report) -> dict[str, ReadOnlyDevice]:
    """Every configured device answers and is not FAULT; returns the ones that answered."""
    answered: dict[str, ReadOnlyDevice] = {}
    for name, device_class, _properties in _devices(config):
        label = f"device {name}"
        try:
            device = ReadOnlyDevice(probes.device(f"tango://{config.tango_host}:{config.tango_port}/{name}"), device_class)
            state = device.state()
        except Exception as exc:
            report.add("FAIL", label, f"no answer: {_error(exc)}")
            continue
        answered[name] = device
        status = ""
        if state != "ON":
            try:
                status = device.status()
            except Exception as exc:
                status = f"(Status unreadable: {_error(exc)})"
        detail = f"{state}{(' - ' + status) if status else ''}"
        report.add("FAIL" if state in BAD_STATES else "PASS", label, detail[:300])
    return answered


def check_autoscript(config: Any, probes: Probes, report: Report, ask_server: bool) -> None:
    try:
        client = probes.autoscript_client_version()
    except Exception as exc:
        report.add("FAIL", "autoscript client in this Python", f"cannot import autoscript_tem_microscope_client: {_error(exc)}")
        client = ""
    else:
        same = _version_tuple(client) == _version_tuple(EXPECTED_AUTOSCRIPT)
        report.add("PASS" if same else "WARN", "autoscript client in this Python", f"{client} (pyproject installs {EXPECTED_AUTOSCRIPT})")

    host, port = config.instrument.hardware_host, config.instrument.hardware_port
    if not host or not port:
        report.add("FAIL", "autoscript server address", "instrument.hardware_host/hardware_port are not set")
        return
    hint = f" (AutoScript's own default port is {AUTOSCRIPT_DEFAULT_PORT}: confirm the port with the facility)" if port != AUTOSCRIPT_DEFAULT_PORT else ""
    try:
        probes.tcp(host, port)
        report.add("PASS", f"autoscript server port {host}:{port}", f"accepts TCP{hint}")
    except Exception as exc:
        report.add("FAIL", f"autoscript server port {host}:{port}", f"{_error(exc)}{hint}")
        return
    if not ask_server:
        report.add("SKIP", "autoscript server version", "--no-autoscript-server")
        return
    try:
        server = probes.autoscript_server_version(host, port)
    except Exception as exc:
        report.add("WARN", "autoscript server version", f"the server did not say: {_error(exc)}")
        return
    if not client:
        report.add("WARN", "autoscript server version", f"{server}; no client to compare with")
    elif _version_tuple(server) != _version_tuple(client):
        report.add("FAIL", "autoscript server version", f"server {server}, client {client}: install the client wheels that match the server")
    else:
        report.add("PASS", "autoscript server version", f"{server} (client {client})")


def check_microscope(config: Any, devices: dict[str, ReadOnlyDevice], report: Report) -> None:
    microscope = devices.get("asyncroscopy/instrument/default")
    if microscope is None:
        report.add("FAIL", "microscope reads", "the instrument device did not answer")
        return
    try:
        x, y, z, alpha, beta = [float(v) for v in microscope.command("get_stage")]
        report.add("PASS", "stage position", f"x={x * 1e6:.3f} um y={y * 1e6:.3f} um z={z * 1e6:.3f} um alpha={alpha:.2f} deg beta={beta:.2f} deg")
    except Exception as exc:
        report.add("FAIL", "stage position", _error(exc))
    try:
        parameters = json.loads(microscope.command("get_parameters"))
    except Exception as exc:
        report.add("FAIL", "microscope parameters", _error(exc))
        parameters = {}
    if parameters:
        report.add("PASS", "optical mode", f"{parameters.get('optical_mode')}, projector {parameters.get('projector_mode')}, "
                   f"{float(parameters.get('acceleration_voltage') or math.nan) / 1e3:.0f} kV, column valves {parameters.get('column_valves')}")
        source = parameters.get("beam_current_source")
        if source == "measured":
            report.add("PASS", "beam current", f"{parameters.get('beam_current_pa'):.2f} pA, measured {parameters.get('beam_current_age_s')} s ago")
        else:
            report.add("WARN", "beam current", f"not measured: {parameters.get('beam_current_reason', 'no reason given')}. Dose budgets are refused until it is.")
    try:
        blanked = bool(microscope.attribute("beam_state"))
        report.add("PASS" if blanked else "WARN", "beam blanked", "yes" if blanked else "NO: the beam is on the sample")
    except Exception as exc:
        report.add("FAIL", "beam blanked", _error(exc))


def check_approvals(config: Any, devices: dict[str, ReadOnlyDevice], report: Report) -> None:
    names = [d.device_name for d in config.support_devices if d.module_name == APPROVALS_MODULE]
    if not names:
        report.add("FAIL", "approvals", "no Approvals device in the config: nothing holds high-risk agent calls")
        return
    approvals = devices.get(names[0])
    if approvals is None:
        report.add("FAIL", "approvals", f"{names[0]} did not answer")
        return
    try:
        mode = str(approvals.command("get_mode"))
        report.add("PASS" if mode == "approve" else "FAIL", "approvals mode", mode if mode == "approve" else f"{mode}: must be approve on hardware")
    except Exception as exc:
        report.add("FAIL", "approvals mode", _error(exc))
    try:
        simulated = bool(approvals.command("get_simulated"))
        report.add("FAIL" if simulated else "PASS", "approvals: stack is hardware", "simulated: true" if simulated else "simulated: false")
    except Exception as exc:
        report.add("FAIL", "approvals: stack is hardware", _error(exc))


def check_ceos(config: Any, probes: Probes, report: Report) -> None:
    for device in config.support_devices:
        if "ceos_host" not in device.properties:
            continue
        host, port = device.properties["ceos_host"][0], int(device.properties.get("ceos_port", ["9092"])[0])
        try:
            probes.tcp(host, port)
            report.add("PASS", f"ceos at {host}:{port}", "accepts TCP (nothing sent)")
        except Exception as exc:
            report.add("FAIL", f"ceos at {host}:{port}", f"{_error(exc)}; CEOS usually runs on the corrector PC: confirm its address")


def check_tiled(config: Any, probes: Probes, report: Report) -> None:
    uri = f"http://{_local(config.tiled.host)}:{config.tiled.port}"
    key = settings.get().tiled_api_key
    key = "secret" if key is None else key
    try:
        status, body = probes.http_get(f"{uri}/api/v1/", {"Accept": "application/json"})
        about = json.loads(body) if status == 200 else {}
    except Exception as exc:
        report.add("FAIL", f"tiled at {uri}", _error(exc))
        return
    if "api_version" not in about:
        report.add("FAIL", f"tiled at {uri}", f"HTTP {status}: not a Tiled server")
        return
    report.add("PASS", f"tiled at {uri}", f"tiled {about.get('library_version', '?')}")
    try:
        status, body = probes.http_get(f"{uri}/api/v1/metadata/", {"Accept": "application/json", "Authorization": f"Apikey {key}"})
    except Exception as exc:
        report.add("FAIL", "tiled accepts the API key", _error(exc))
        return
    if status == 200:
        report.add("PASS", "tiled accepts the API key", "authenticated read")
    elif status in (401, 403):
        report.add("FAIL", "tiled accepts the API key", f"HTTP {status}: run_servers was started with a different ASYNCROSCOPY_TILED_API_KEY")
    else:
        report.add("FAIL", "tiled accepts the API key", f"HTTP {status}: {body[:160]}")


def _listening(probes: Probes, url: str) -> bool:
    parts = urlsplit(url)
    try:
        probes.tcp(parts.hostname or "127.0.0.1", parts.port or 80)
        return True
    except Exception:
        return False


def check_mcp(name: str, url: str, probes: Probes, report: Report, required: bool) -> None:
    if not _listening(probes, url):
        if required:
            report.add("FAIL", f"{name} at {url}", "nothing is listening there")
        else:
            report.add("SKIP", f"{name} at {url}", "not running (start it, then run this again)")
        return
    try:
        tools = probes.mcp_tools(url, _mcp_headers())
    except Exception as exc:
        report.add("FAIL", f"{name} at {url}", f"MCP initialize/list_tools failed: {_error(exc)}")
        return
    report.add("PASS" if tools else "FAIL", f"{name} at {url}", f"{len(tools)} tools")


def check_analysis(config: Any, probes: Probes, report: Report) -> None:
    for device in config.support_devices:
        url = (device.properties.get("analysis_url") or [""])[0]
        if url:
            host = urlsplit(url).hostname or ""
            if not is_loopback_host(host):
                report.add("WARN", "analysis_url is loopback", f"{url}: the analysis server has no login; keep it on 127.0.0.1")
            check_mcp(f"analysis server ({device.key})", url, probes, report, required=True)


def check_endpoints(paths: list[Path], probes: Probes, report: Report) -> None:
    from startup_scripts import run_mcp

    for path in paths:
        try:
            mcp = run_mcp.load_config(path).mcp
        except Exception as exc:
            report.add("FAIL", f"mcp config {path.name}", _error(exc))
            continue
        if mcp.transport != "streamable-http":
            report.add("SKIP", f"mcp endpoint {path.name}", f"transport {mcp.transport}")
            continue
        url = f"http://{_local(mcp.http_host)}:{mcp.http_port}/mcp"
        check_mcp(f"mcp {mcp.endpoint_role} endpoint ({path.name})", url, probes, report, required=False)


def run(config_path: Path, mcp_paths: list[Path], probes: Probes = REAL_PROBES, ask_autoscript_server: bool = True) -> Report:
    from startup_scripts import run_servers

    report = Report()
    print(f"Read-only pre-flight for {config_path}", flush=True)
    try:
        config = run_servers.load_config(config_path)
    except Exception as exc:
        report.add("FAIL", f"config {config_path.name}", _error(exc))
        return report
    check_config(config, report)
    devices: dict[str, ReadOnlyDevice] = {}
    if check_database(config, probes, report):
        devices = check_devices(config, probes, report)
    check_autoscript(config, probes, report, ask_autoscript_server)
    check_microscope(config, devices, report)
    check_approvals(config, devices, report)
    check_ceos(config, probes, report)
    check_tiled(config, probes, report)
    check_analysis(config, probes, report)
    check_endpoints(mcp_paths, probes, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML, help="The run_servers config of the stack (default: configs/Spectra300.yaml).")
    parser.add_argument("--mcp-yaml", type=Path, action="append", default=None,
                        help="An MCP endpoint config to check if it is running; repeat for several (default: configs/mcp.yaml and configs/mcp_operator.yaml).")
    parser.add_argument("--no-autoscript-server", action="store_true", help="Do not open a connection to the AutoScript server to read its version.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args.yaml, args.mcp_yaml or list(DEFAULT_MCP_YAMLS), ask_autoscript_server=not args.no_autoscript_server)
    print()
    if report.failures:
        print(f"PRE-FLIGHT FAILED: {len(report.failures)} check(s): {', '.join(report.failures)}", flush=True)
        return 1
    warnings = sum(1 for result, _, _ in report.rows if result == "WARN")
    print(f"PRE-FLIGHT PASSED ({warnings} warning(s)). Nothing was moved, acquired or written.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
