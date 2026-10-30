"""Bearer-token verification at the MCP endpoints (mcp.auth; asyncroscopy.mcp.endpoint_auth).

The servers run in this process (uvicorn on an ephemeral 127.0.0.1 port) over a small fake twin
and the real Approvals device of the test context; tokens come from startup_scripts/dev_tokens.py,
the stand-in for WG1's token service.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import tango

from asyncroscopy.mcp.endpoint_auth import AuthSettings, required_role, remote_without_auth
from asyncroscopy.mcp.mcp_server import MCPServer

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}_auth", ROOT / "startup_scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dev_tokens = _script("dev_tokens")


# ---------------------------------------------------------------------- a fake twin, the real Approvals device
def _cmd(name, in_type, out_type, in_desc=""):
    return type("Cmd", (), {"cmd_name": name, "in_type": in_type, "out_type": out_type, "in_type_desc": in_desc, "out_type_desc": ""})()


class FakeTwin:
    COMMANDS = [
        _cmd("get_stage", tango.CmdArgType.DevVoid, tango.CmdArgType.DevVarDoubleArray),
        _cmd("move_stage", tango.CmdArgType.DevVarDoubleArray, tango.CmdArgType.DevString, ":param position: [x, y, z, alpha, beta]."),
    ]

    def __init__(self):
        self.position = [0.0] * 5

    def set_timeout_millis(self, ms):
        pass

    def info(self):
        return type("Info", (), {"dev_class": "DigitalTwin"})()

    def dev_name(self):
        return "asyncroscopy/digitaltwin/default"

    def command_list_query(self):
        return self.COMMANDS

    def get_stage(self):
        return list(self.position)

    def move_stage(self, position):
        self.position = list(position)
        return f"moved to {list(position)}"


class LiveServer:
    """An MCPServer's HTTP app served by uvicorn in a thread on a free 127.0.0.1 port."""

    def __init__(self, server: MCPServer):
        import uvicorn

        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self.uvicorn = uvicorn.Server(uvicorn.Config(server.mcp.http_app(), host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.uvicorn.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 20
        while not self.uvicorn.started:
            assert time.monotonic() < deadline, "the test MCP server did not start"
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{self.port}/mcp"

    def stop(self):
        self.uvicorn.should_exit = True
        self.thread.join(10)


def _client(url, token=None, caller="self-reported"):
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    headers = {"X-Asyncroscopy-Caller": caller}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return Client(StreamableHttpTransport(url, headers=headers), timeout=30)


def http_status(url, token=None):
    """Status of a bare MCP initialize request: 401 when the token check refuses it."""
    import httpx

    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
    return httpx.post(url, json=body, headers=headers, timeout=10).status_code


def tools(url, token=None):
    async def go():
        async with _client(url, token) as client:
            return {t.name for t in await client.list_tools()}
    return asyncio.run(go())


def call(url, token, name, args=None, caller="self-reported"):
    """(is_error, text) of one tool call."""
    async def go():
        async with _client(url, token, caller) as client:
            result = await client.call_tool(name, args or {}, raise_on_error=False)
            return result.is_error, "".join(getattr(b, "text", "") for b in result.content or [])
    return asyncio.run(go())


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    key_dir = tmp_path_factory.mktemp("dev_tokens")
    section = dev_tokens.init(key_dir)
    other = tmp_path_factory.mktemp("other_keys")
    dev_tokens.init(other)
    return {"dir": key_dir, "other": other, "auth": {**section, "public_key_file": str(key_dir / "public.pem"), "audience": "scope-1"}}


def token(keys, sub, *scopes, **kw):
    kw.setdefault("audience", "scope-1")
    return dev_tokens.mint(keys["dir"], sub, list(scopes), **kw)


@pytest.fixture(scope="module")
def stack(keys, tango_ctx, tmp_path_factory):
    """Agent and operator endpoints with mcp.auth, over the fake twin and the real Approvals device."""
    trace_dir = tmp_path_factory.mktemp("auth_trace")
    twin = FakeTwin()
    approvals_name = "asyncroscopy/approvals/default"

    def proxy(name):
        return tango.DeviceProxy(tango_ctx.get_device_access(approvals_name)) if "approvals" in name else twin

    database = type("Db", (), {"get_device_exported": lambda self, p: type("E", (), {"value_string": ["asyncroscopy/digitaltwin/default", approvals_name]})()})()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("asyncroscopy.mcp.mcp_server.DeviceProxy", proxy)
        servers = {}
        for role in ("agent", "operator"):
            server = MCPServer("auth-test", "localhost", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False,
                               endpoint_role=role, approvals_device_address=approvals_name, trace_path=str(trace_dir / f"{role}.jsonl"),
                               database=database, auth=keys["auth"])
            server.setup(print_summary=False)
            servers[role] = server
        live = {role: LiveServer(server) for role, server in servers.items()}
        try:
            yield {"agent": live["agent"].url, "operator": live["operator"].url, "servers": servers, "trace": trace_dir, "twin": twin}
        finally:
            for server in live.values():
                server.stop()


@pytest.fixture(autouse=True)
def auto_mode_after(approvals_proxy):
    yield
    approvals_proxy.set_mode(json.dumps({"mode": "auto", "who": "test-teardown"}))


def trace(stack, role):
    path = stack["trace"] / f"{role}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


# ---------------------------------------------------------------------- refusals
def test_no_token_is_refused(stack, keys):
    assert http_status(stack["agent"]) == 401 and http_status(stack["operator"]) == 401
    assert http_status(stack["agent"], token(keys, "alice", "instrument:read")) == 200
    with pytest.raises(Exception):
        tools(stack["agent"])
    with pytest.raises(Exception):
        call(stack["agent"], None, "DigitalTwin_get_stage")


def test_an_in_memory_client_has_no_token_and_is_refused(stack):
    from fastmcp import Client

    async def go():
        async with Client(stack["servers"]["agent"].mcp) as client:
            return await client.call_tool("DigitalTwin_get_stage", {}, raise_on_error=False)

    result = asyncio.run(go())
    assert result.is_error and "bearer token" in result.content[0].text


@pytest.mark.parametrize("bad", ["expired", "wrong_issuer", "wrong_audience", "wrong_key", "garbage"])
def test_invalid_tokens_are_refused(stack, keys, bad):
    if bad == "expired":
        tok = token(keys, "alice", "instrument:read", ttl_s=-60)
    elif bad == "wrong_issuer":
        tok = token(keys, "alice", "instrument:read", issuer="https://someone-else.example")
    elif bad == "wrong_audience":
        tok = token(keys, "alice", "instrument:read", audience="another-instrument")
    elif bad == "wrong_key":
        tok = dev_tokens.mint(keys["other"], "alice", ["instrument:read"], audience="scope-1")
    else:
        tok = "not.a.jwt"
    assert http_status(stack["agent"], tok) == 401
    with pytest.raises(Exception):
        tools(stack["agent"], tok)
    with pytest.raises(Exception):
        call(stack["agent"], tok, "DigitalTwin_get_stage")


def test_a_token_without_an_expiry_is_refused(stack, keys):
    from joserfc import jwk, jwt

    key = jwk.import_key((keys["dir"] / "private.pem").read_bytes(), "RSA")
    forever = jwt.encode({"alg": "RS256"}, {"iss": keys["auth"]["issuer"], "aud": "scope-1", "sub": "alice", "scope": "instrument:read"}, key)
    assert http_status(stack["agent"], forever) == 401


# ---------------------------------------------------------------------- roles from the risk policy
def test_a_read_token_can_get_stage_but_not_move_stage(stack, keys):
    tok = token(keys, "reader", "instrument:read")
    offered = tools(stack["agent"], tok)
    assert "DigitalTwin_get_stage" in offered and "DigitalTwin_move_stage" not in offered
    assert "run_job" not in offered and "endpoint_info" in offered
    error, text = call(stack["agent"], tok, "DigitalTwin_get_stage")
    assert not error, text
    error, text = call(stack["agent"], tok, "DigitalTwin_move_stage", {"position": [1, 0, 0, 0, 0]})
    assert error and "operate role" in text
    assert stack["twin"].position == [0.0] * 5


def test_an_operate_token_moves_the_stage_through_the_approval_gate(stack, keys, approvals_proxy):
    approvals_proxy.set_mode(json.dumps({"mode": "approve", "who": "test"}))
    alice = token(keys, "alice@ornl.gov", "instrument:operate")
    move = {"position": [2.0, 1.0, 0.0, 0.0, 0.0]}

    # Held: the request names the token's subject, not the self-reported header.
    error, text = call(stack["agent"], alice, "DigitalTwin_move_stage", move, caller="mallory")
    assert error and "APPROVAL REQUIRED" in text, text
    request_id = text.split("request_id=")[1].split(")")[0]
    pending = [p for p in json.loads(approvals_proxy.list_pending()) if p["request_id"] == request_id]
    assert pending and pending[0]["caller"] == "alice@ornl.gov"
    held = [e for e in trace(stack, "agent") if e.get("request_id") == request_id and e.get("error") == "approval_required"]
    assert held and held[-1]["caller"] == "alice@ornl.gov"

    # Only an approve token may decide on the operator endpoint, and "who" is the token's subject.
    for scopes in (["instrument:read"], ["instrument:operate"]):
        error, text = call(stack["operator"], token(keys, "eve", *scopes), "Approvals_approve", {"request_id": request_id, "who": "eve"})
        assert error and "approve role" in text
    assert json.loads(approvals_proxy.request_status(request_id))["state"] == "pending"
    assert "Approvals_approve" not in tools(stack["operator"], token(keys, "eve", "instrument:operate"))
    approver = token(keys, "operator-bob", "instrument:approve")
    assert {"Approvals_approve", "Approvals_deny", "Approvals_set_mode"} <= tools(stack["operator"], approver)
    error, text = call(stack["operator"], approver, "Approvals_approve", {"request_id": request_id, "who": "someone-else"})
    assert not error, text
    status = json.loads(approvals_proxy.request_status(request_id))
    assert status["state"] == "approved" and status["decided_by"] == "operator-bob"
    decided = [e for e in trace(stack, "operator") if e.get("action") == "Approvals_approve"]
    assert decided and decided[-1]["caller"] == "operator-bob"

    # The same call again runs, and the trace records who asked and who approved.
    error, text = call(stack["agent"], alice, "DigitalTwin_move_stage", move)
    assert not error and "moved" in text, text
    ran = [e for e in trace(stack, "agent") if e.get("action") == "DigitalTwin_move_stage" and e.get("ok")]
    assert ran[-1]["caller"] == "alice@ornl.gov" and ran[-1]["approved_by"] == "operator-bob"


def test_call_tool_needs_the_role_of_the_tool_it_runs(stack, keys, approvals_proxy):
    reader = token(keys, "reader", "instrument:read")
    error, text = call(stack["agent"], reader, "call_tool", {"name": "DigitalTwin_move_stage", "arguments": {"position": [7, 0, 0, 0, 0]}})
    assert error and "operate role" in text, text
    assert stack["twin"].position[0] != 7
    before = approvals_proxy.get_mode()
    for scopes in (["instrument:read"], ["instrument:operate"]):
        for target, args in (("Approvals_set_mode", {"request": json.dumps({"mode": "approve", "who": "eve"})}),
                             ("Approvals_set_auto_allowed", {"request": json.dumps({"patterns": ["*"], "who": "eve"})})):
            error, text = call(stack["operator"], token(keys, "eve", *scopes), "call_tool", {"name": target, "arguments": args})
            assert error and "approve role" in text, text
    assert approvals_proxy.get_mode() == before
    assert json.loads(approvals_proxy.get_auto_allowed())["switched_on"] == []
    error, text = call(stack["operator"], reader, "find_tools", {"query": "approvals"})
    assert not error and "Approvals_set_mode" not in text and "Approvals_get_mode" in text, text


def test_set_auto_allowed_records_the_token_subject(stack, keys, approvals_proxy, job_dir):
    bob = token(keys, "operator-bob", "instrument:approve")
    try:
        error, text = call(stack["operator"], bob, "Approvals_set_auto_allowed", {"request": json.dumps({"patterns": ["DigitalTwin_set_screen"], "who": "someone-else"})})
        assert not error, text
        assert json.loads((job_dir / "approvals_mode.json").read_text())["who"] == "operator-bob"
    finally:
        approvals_proxy.set_auto_allowed(json.dumps({"patterns": [], "who": "test"}))


def test_set_mode_needs_the_approve_role(stack, keys, approvals_proxy):
    error, text = call(stack["operator"], token(keys, "alice", "instrument:operate"), "Approvals_set_mode", {"mode": "approve", "who": "alice"})
    assert error and "approve role" in text
    error, text = call(stack["operator"], token(keys, "operator-bob", "instrument:approve"), "Approvals_set_mode", {"mode": "approve", "who": "x"})
    assert not error, text
    assert approvals_proxy.get_mode() == "approve"


def test_roles_follow_the_risk_policy():
    risks = {"DigitalTwin_get_stage": "low", "DigitalTwin_move_stage": "high", "Approvals_approve": "high", "Mapper_job_status": "low"}
    keys_ = {"DigitalTwin_get_stage": "DigitalTwin.get_stage", "DigitalTwin_move_stage": "DigitalTwin.move_stage",
             "Approvals_approve": "Approvals.approve", "Mapper_job_status": "Mapper.job_status"}
    assert required_role("DigitalTwin_get_stage", risks, keys_) == "read"
    assert required_role("Mapper_job_status", risks, keys_) == "read"
    assert required_role("DigitalTwin_move_stage", risks, keys_) == "operate"
    assert required_role("Approvals_approve", risks, keys_) == "approve"
    assert required_role("run_job", risks, keys_) == "operate"
    assert required_role("view_data", risks, keys_) == "read"


# ---------------------------------------------------------------------- the config
def test_auth_config_is_checked(keys, tmp_path):
    good = keys["auth"]
    assert AuthSettings.from_config(None) is None and AuthSettings.from_config({}) is None
    settings = AuthSettings.from_config(good)
    assert settings.roles_of({"scope": "instrument:operate"}) == {"read", "operate"}
    assert settings.roles_of({"scp": ["instrument:approve"]}) == {"read", "approve"}
    assert settings.roles_of({"scope": "openid"}) == set()
    custom = AuthSettings.from_config({**good, "roles": {"operate": ["stem:drive"]}, "role_claims": "groups"})
    assert custom.roles_of({"groups": ["stem:drive"]}) == {"operate"}
    for bad, match in [
        ({k: v for k, v in good.items() if k != "issuer"}, "issuer"),
        ({k: v for k, v in good.items() if k != "audience"}, "audience"),
        ({**good, "jwks_url": "https://tokens.example/jwks"}, "exactly one"),
        ({k: v for k, v in good.items() if k != "public_key_file"}, "exactly one"),
        ({**good, "public_key_file": str(tmp_path / "missing.pem")}, "does not exist"),
        ({**good, "algorithm": "HS256"}, "asymmetric"),
        ({**good, "roles": {"admin": ["x"]}}, "unknown role"),
        ({**good, "issuer_url": "x"}, "unknown keys"),
    ]:
        with pytest.raises(ValueError, match=match):
            AuthSettings.from_config(bad)


def test_without_auth_nothing_changes(monkeypatch):
    """No mcp.auth: no verifier, no gate, and the caller is the self-reported header (the existing suite covers the rest)."""
    from fastmcp import Client

    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    server = MCPServer("t", "h", 1, blocked_classes=[], blocked_functions={}, data_device_address="d", verbose=False)
    assert server.auth is None and server.mcp.auth is None
    server.mcp.add_tool(server.endpoint_info)

    async def go():
        async with Client(server.mcp) as client:
            return await client.call_tool("endpoint_info", {}, raise_on_error=False)

    assert not asyncio.run(go()).is_error
    monkeypatch.setattr("fastmcp.server.dependencies.get_http_headers", lambda include_all=False: {"x-asyncroscopy-caller": "general-agent"})
    assert server._caller() == "general-agent"


# ---------------------------------------------------------------------- run_mcp refuses open hardware without auth
def test_remote_without_auth_rule():
    assert remote_without_auth("0.0.0.0", simulated=False, auth=None)
    assert remote_without_auth("10.46.217.5", simulated=False, auth=None)
    assert remote_without_auth("0.0.0.0", simulated=True, auth=None) is None  # a twin, as at the hackathon
    assert remote_without_auth("127.0.0.1", simulated=False, auth=None) is None
    assert remote_without_auth("localhost", simulated=False, auth=None) is None
    assert remote_without_auth("0.0.0.0", simulated=False, auth={"issuer": "x"}) is None


def _mcp_config(tmp_path, replace: str, extra: str = "") -> Path:
    text = (ROOT / "configs" / "mcp.yaml").read_text(encoding="utf-8").replace("http_host: 127.0.0.1", replace)
    path = tmp_path / "mcp_open.yaml"
    path.write_text(text + extra, encoding="utf-8")
    return path


def test_run_mcp_refuses_a_non_local_hardware_endpoint_without_auth(tmp_path, monkeypatch, keys, capsys):
    run_mcp = _script("run_mcp")
    hardware = lambda *a: False  # noqa: E731
    twin = lambda *a: True  # noqa: E731
    opened = run_mcp.load_config(_mcp_config(tmp_path, "http_host: 0.0.0.0\n  allow_remote_bind: true"))
    assert "not simulated" in run_mcp.remote_auth_problem(opened, simulated=hardware)
    assert run_mcp.remote_auth_problem(opened, simulated=twin) is None
    one_ip = run_mcp.load_config(_mcp_config(tmp_path, "http_host: 10.46.217.5\n  allow_remote_bind: true"))
    assert run_mcp.remote_auth_problem(one_ip, simulated=hardware)
    # A LAN address without allow_remote_bind binds 127.0.0.1 on hardware (asyncroscopy.mcp.binding): local, so it may start.
    one_ip_local = run_mcp.load_config(_mcp_config(tmp_path, "http_host: 10.46.217.5"))
    assert run_mcp.remote_auth_problem(one_ip_local, simulated=hardware) is None
    # 0.0.0.0 without allow_remote_bind binds 127.0.0.1 on hardware: local, so it may start.
    wildcard = run_mcp.load_config(_mcp_config(tmp_path, "http_host: 0.0.0.0"))
    assert run_mcp.remote_auth_problem(wildcard, simulated=hardware) is None
    assert run_mcp.remote_auth_problem(run_mcp.load_config(ROOT / "configs" / "mcp.yaml"), simulated=hardware) is None

    auth_yaml = (f"  auth:\n    public_key_file: {json.dumps(keys['auth']['public_key_file'])}\n"
                 f"    issuer: {json.dumps(keys['auth']['issuer'])}\n    audience: scope-1\n")
    with_auth = run_mcp.load_config(_mcp_config(tmp_path, "http_host: 0.0.0.0\n  allow_remote_bind: true", auth_yaml))
    assert run_mcp.remote_auth_problem(with_auth, simulated=hardware) is None
    command = run_mcp.build_command(with_auth)
    assert json.loads(command[command.index("--auth-json") + 1])["audience"] == "scope-1"
    assert "--auth-json" not in run_mcp.build_command(opened)

    # main() stops before starting anything.
    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.stack_simulated", hardware)
    monkeypatch.setattr(run_mcp, "ProcessManager", lambda **kw: pytest.fail("the server must not be started"))
    assert run_mcp.main(["--yaml", str(_mcp_config(tmp_path, "http_host: 0.0.0.0\n  allow_remote_bind: true"))]) == 1
    assert "mcp.auth" in capsys.readouterr().err


def test_mcp_server_main_refuses_too(monkeypatch, capsys):
    from asyncroscopy.mcp import mcp_server

    monkeypatch.setattr(mcp_server, "stack_simulated", lambda *a: False)
    monkeypatch.setattr(mcp_server, "MCPServer", lambda **kw: pytest.fail("the server must not be built"))
    argv = ["--name", "x", "--tango-host", "h", "--tango-port", "1", "--transport", "streamable-http", "--http-host", "0.0.0.0",
            "--http-port", "1", "--blocked-classes-json", "[]", "--blocked-functions-json", "{}", "--data-device-address", "d", "--allow-remote-bind"]
    assert mcp_server.main(argv) == 1
    assert "MCP ERROR" in capsys.readouterr().out


# ---------------------------------------------------------------------- agents send the token
def test_agents_send_the_token(monkeypatch):
    from asyncroscopy.agents.config import AgentConfig
    from asyncroscopy.agents.mcp_tools import McpServer

    monkeypatch.delenv("ASYNCROSCOPY_MCP_TOKEN", raising=False)
    assert "Authorization" not in AgentConfig.from_env()._headers()
    assert "Authorization" not in McpServer("http://x/mcp", {}).headers
    monkeypatch.setenv("ASYNCROSCOPY_MCP_TOKEN", "abc.def.ghi")
    config = AgentConfig.from_env()
    assert config._headers()["Authorization"] == "Bearer abc.def.ghi"
    assert "abc" not in repr(config)
    assert McpServer("http://x/mcp", {"X-Asyncroscopy-Caller": "agent-swarm"}).headers["Authorization"] == "Bearer abc.def.ghi"
    assert McpServer("http://x/mcp", {"Authorization": "Bearer other"}).headers["Authorization"] == "Bearer other"


def test_an_agent_with_a_token_reaches_an_auth_endpoint(stack, keys, monkeypatch):
    from asyncroscopy.agents.config import AgentConfig

    monkeypatch.setenv("ASYNCROSCOPY_MCP_TOKEN", token(keys, "agent-7", "instrument:read"))
    config = AgentConfig.from_env(mcp_url=stack["agent"], analysis_url="")
    result = asyncio.run(config.instrument().call("DigitalTwin_get_stage", {}))
    assert not result.is_error
    info = asyncio.run(config.info())
    assert info.get("role") == "agent"


def test_outputs_is_gitignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "outputs/" in ignored
    assert dev_tokens.DEFAULT_DIR.relative_to(ROOT).parts[0] == "outputs"
