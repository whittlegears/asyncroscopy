"""The MCP HTTP endpoints refuse a foreign Host or Origin (DNS rebinding) before the MCP handler."""

import socket
import threading
import time

import httpx
import pytest

from asyncroscopy.mcp.agent_mcp_server import AgentMCPServer
from asyncroscopy.mcp.host_guard import allowed_hosts
from asyncroscopy.mcp.mcp_server import MCPServer
from startup_scripts import run_llm, run_mcp

INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


class _Exported:
    value_string: list[str] = []

    def get_device_exported(self, pattern: str) -> "_Exported":
        return self


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _serve(start) -> str:
    port = _free_port()
    threading.Thread(target=start, args=(port,), daemon=True).start()
    deadline = time.monotonic() + 20
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return f"http://127.0.0.1:{port}/mcp"
        except OSError:
            assert time.monotonic() < deadline, "the endpoint did not start"
            time.sleep(0.05)


def _instrument_endpoint(role: str, extra: list[str]) -> str:
    server = MCPServer("t", "localhost", 0, blocked_functions={}, blocked_classes=[], data_device_address="",
                       endpoint_role=role, database=_Exported(), verbose=False)
    return _serve(lambda port: server.start(transport="streamable-http", host="127.0.0.1", port=port, allowed_hosts=extra, show_banner=False))


def _workflow_operator_endpoint() -> str:
    server = AgentMCPServer("t", mcp_urls=[], endpoint_role="operator", verbose=False)
    return _serve(lambda port: server.start(transport="streamable-http", host="127.0.0.1", port=port, show_banner=False))


@pytest.fixture(scope="module")
def endpoints():
    return {"agent": _instrument_endpoint("agent", ["proxy.lab.example"]), "operator": _instrument_endpoint("operator", []),
            "workflow_operator": _workflow_operator_endpoint()}


@pytest.mark.parametrize("which", ["agent", "operator", "workflow_operator"])
def test_a_foreign_host_or_origin_is_refused(endpoints, which):
    url = endpoints[which]
    port = url.split(":")[2].split("/")[0]
    evil = httpx.post(url, json=INITIALIZE, headers={**HEADERS, "Host": f"evil.example:{port}"})
    assert evil.status_code == 403 and "evil.example" in evil.text
    origin = httpx.post(url, json=INITIALIZE, headers={**HEADERS, "Origin": "http://evil.example"})
    assert origin.status_code == 403
    for host in (f"127.0.0.1:{port}", f"localhost:{port}"):
        ok = httpx.post(url, json=INITIALIZE, headers={**HEADERS, "Host": host, "Origin": f"http://{host}"})
        assert ok.status_code == 200, ok.text


def test_a_configured_host_is_answered(endpoints):
    url = endpoints["agent"]
    assert httpx.post(url, json=INITIALIZE, headers={**HEADERS, "Host": "proxy.lab.example"}).status_code == 200
    assert httpx.post(endpoints["operator"], json=INITIALIZE, headers={**HEADERS, "Host": "proxy.lab.example"}).status_code == 403


def test_the_allowlist():
    hosts = allowed_hosts("192.168.1.20", ["Proxy.Example"])
    assert {"localhost", "127.0.0.1", "::1", "192.168.1.20", "proxy.example"} <= hosts and "evil.example" not in hosts
    assert socket.gethostname().lower() in allowed_hosts("0.0.0.0")


def test_the_configs_carry_allowed_hosts(tmp_path):
    mcp_yaml = tmp_path / "mcp.yaml"
    mcp_yaml.write_text("tango: {host: localhost, port: 9094}\nmcp: {name: X, transport: streamable-http, http_host: 127.0.0.1, http_port: 8000,"
                        " data_device_address: a/b/c, allowed_hosts: [proxy.lab.example], operator_port: 8001}\n", encoding="utf-8")
    config = run_mcp.load_config(mcp_yaml)
    assert "--allowed-hosts-json" in run_mcp.build_command(config) and '["proxy.lab.example"]' in run_mcp.build_command(config)
    assert run_mcp.operator_config(config).mcp.allowed_hosts == []
    llm_yaml = tmp_path / "llm.yaml"
    llm_yaml.write_text("agent: {name: A, transport: streamable-http, http_host: 127.0.0.1, http_port: 8002, allowed_hosts: [proxy.lab.example]}\n", encoding="utf-8")
    assert '["proxy.lab.example"]' in run_llm.build_command(run_llm.load_config(llm_yaml))
