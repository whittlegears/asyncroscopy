"""Tests for the hermes backend against a real local HTTP gateway stand-in."""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from asyncroscopy.mcp.backends import create_backend
from asyncroscopy.mcp.backends.base import BackendConfig, BackendUnsupported
from asyncroscopy.mcp.backends.hermes_backend import HermesBackend

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class GatewayHandler(BaseHTTPRequestHandler):
    served_models = ["hermes-agent"]
    answer = "The stage is at 0,0."
    requests_seen: list[dict] = []

    def log_message(self, format, *args):
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json({"data": [{"id": m} for m in self.served_models]})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        GatewayHandler.requests_seen.append({
            "path": self.path,
            "payload": payload,
            "authorization": self.headers.get("Authorization"),
        })
        if self.path == "/v1/chat/completions":
            self._send_json({
                "choices": [{"message": {"role": "assistant", "content": self.answer}}],
                "model": payload.get("model"),
            })
        else:
            self._send_json({"error": "not found"}, status=404)


@pytest.fixture
def gateway():
    GatewayHandler.requests_seen = []
    GatewayHandler.served_models = ["hermes-agent"]
    server = HTTPServer(("127.0.0.1", 0), GatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def _config(url: str, **overrides) -> BackendConfig:
    return BackendConfig(hermes_url=url, **overrides)


class TestHermesBackend:
    def test_initialize_succeeds_against_live_gateway(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        asyncio.run(backend.initialize())

    def test_initialize_rejects_empty_url(self):
        backend = HermesBackend(BackendConfig(), [])
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(backend.initialize())
        assert "hermes_url" in str(excinfo.value)

    def test_initialize_rejects_unreachable_gateway(self):
        backend = HermesBackend(_config("http://127.0.0.1:1"), [])
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(backend.initialize())
        assert "not reachable" in str(excinfo.value)

    def test_initialize_rejects_missing_model(self, gateway):
        GatewayHandler.served_models = ["some-other-model"]
        backend = HermesBackend(_config(gateway), [])
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(backend.initialize())
        assert "does not serve model" in str(excinfo.value)

    def test_query_round_trips_over_the_socket(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        result = asyncio.run(backend.query("where is the stage?", 5))
        assert result == "The stage is at 0,0."
        [seen] = GatewayHandler.requests_seen
        assert seen["path"] == "/v1/chat/completions"
        assert seen["payload"]["model"] == "hermes-agent"
        assert seen["payload"]["messages"] == [{"role": "user", "content": "where is the stage?"}]
        assert seen["payload"]["stream"] is False

    def test_query_sends_bearer_when_key_configured(self, gateway):
        backend = HermesBackend(_config(gateway, hermes_api_key="sekrit"), [])
        asyncio.run(backend.query("hello", 5))
        [seen] = GatewayHandler.requests_seen
        assert seen["authorization"] == "Bearer sekrit"

    def test_query_omits_bearer_without_key(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        asyncio.run(backend.query("hello", 5))
        [seen] = GatewayHandler.requests_seen
        assert seen["authorization"] is None

    def test_complete_is_unsupported(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        with pytest.raises(BackendUnsupported):
            asyncio.run(backend.complete({"messages": []}))

    def test_connect_mcp_is_unsupported(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        with pytest.raises(BackendUnsupported):
            asyncio.run(backend.connect_mcp("http://127.0.0.1:8000/mcp", "streamable_http"))

    def test_capabilities(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        assert backend.capabilities() == {"complete": False, "connect_mcp": False}


class TestFactory:
    def test_hermes_constructs_without_langchain(self):
        backend = create_backend("hermes", BackendConfig(hermes_url="http://127.0.0.1:8642"), [])
        assert isinstance(backend, HermesBackend)
        assert backend.name == "hermes"

    def test_name_is_case_insensitive(self):
        backend = create_backend("Hermes", BackendConfig(hermes_url="http://127.0.0.1:8642"), [])
        assert backend.name == "hermes"
