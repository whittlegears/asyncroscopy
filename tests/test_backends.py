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
    # When set, POST /v1/chat/completions answers (status, payload) instead.
    completion_error: tuple[int, dict] | None = None

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
            if GatewayHandler.completion_error is not None:
                status, body = GatewayHandler.completion_error
                self._send_json(body, status=status)
                return
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
    GatewayHandler.completion_error = None
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

    def test_connect_mcp_requires_a_hermes_install(self, gateway, tmp_path):
        backend = HermesBackend(_config(gateway, hermes_home=str(tmp_path / "missing")), [])
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(backend.connect_mcp("http://127.0.0.1:8000/mcp", "streamable_http"))
        assert "Hermes install" in str(excinfo.value)

    def test_capabilities(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        assert backend.capabilities() == {
            "complete": False,
            "connect_mcp": True,
            "skills": False,
        }

    def test_last_trace_is_empty_for_the_opaque_gateway(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        asyncio.run(backend.query("hello", 5))
        assert backend.last_trace() == []

    def test_capabilities_report_skills_when_service_is_set(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        backend.set_skills_service(object())
        assert backend.capabilities()["skills"] is True

    def test_query_surfaces_gateway_error_body(self, gateway):
        GatewayHandler.completion_error = (
            401,
            {"error": {"message": "Invalid API key"}},
        )
        backend = HermesBackend(_config(gateway, hermes_api_key="wrong"), [])
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(backend.query("hello", 5))
        message = str(excinfo.value)
        assert "HTTP 401" in message
        assert "Invalid API key" in message
        assert gateway in message

    def test_query_surfaces_non_json_error_body(self, gateway):
        GatewayHandler.completion_error = (500, {"boom": "not the openai shape"})
        backend = HermesBackend(_config(gateway), [])
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(backend.query("hello", 5))
        assert "HTTP 500" in str(excinfo.value)

    def test_query_names_the_gateway_when_it_dies_mid_run(self, gateway):
        backend = HermesBackend(_config("http://127.0.0.1:1"), [])
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(backend.query("hello", 5))
        message = str(excinfo.value)
        assert "not reachable" in message
        assert "http://127.0.0.1:1" in message


class _StubSkillsService:
    """Minimal stand-in for SkillsService: two skills, an in-memory usage log."""

    def __init__(self, results=None, texts=None, find_error=None):
        self.results = results if results is not None else [
            {"id": "probe-alignment", "name": "Probe alignment", "score": 0.9},
            {"id": "haadf-imaging", "name": "HAADF imaging", "score": 0.4},
        ]
        self.texts = texts or {
            "probe-alignment": "---\nname: Probe alignment\n---\nAlign the probe first.",
            "haadf-imaging": "---\nname: HAADF imaging\n---\nUse the HAADF detector.",
        }
        self.find_error = find_error
        self.usage_calls: list[tuple[list[str], str, bool]] = []

    def find_skills(self, query, k=5):
        if self.find_error is not None:
            raise self.find_error
        return self.results[:k]

    def load_skill(self, skill_id):
        try:
            return self.texts[skill_id]
        except KeyError:
            raise KeyError(f"No enabled skill has the id '{skill_id}'.")

    def record_usage(self, skill_ids, run_task_hash, success):
        self.usage_calls.append((list(skill_ids), run_task_hash, success))
        return len(skill_ids)


class TestHermesSkills:
    def test_query_injects_matching_skills_as_a_system_message(self, gateway):
        backend = HermesBackend(_config(gateway), [])
        backend.set_skills_service(_StubSkillsService())

        result = asyncio.run(backend.query("align the probe", 5))

        assert result == "The stage is at 0,0."
        [seen] = GatewayHandler.requests_seen
        messages = seen["payload"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Align the probe first." in messages[0]["content"]
        assert "Use the HAADF detector." in messages[0]["content"]
        assert messages[1] == {"role": "user", "content": "align the probe"}

    def test_query_records_usage_of_injected_skills(self, gateway):
        service = _StubSkillsService()
        backend = HermesBackend(_config(gateway), [])
        backend.set_skills_service(service)

        asyncio.run(backend.query("align the probe", 5))

        [(skill_ids, run_hash, success)] = service.usage_calls
        assert skill_ids == ["probe-alignment", "haadf-imaging"]
        assert success is True
        assert run_hash  # a stable non-empty hash of the prompt

    def test_query_without_skill_matches_sends_only_the_prompt(self, gateway):
        service = _StubSkillsService(results=[])
        backend = HermesBackend(_config(gateway), [])
        backend.set_skills_service(service)

        asyncio.run(backend.query("hello", 5))

        [seen] = GatewayHandler.requests_seen
        assert seen["payload"]["messages"] == [{"role": "user", "content": "hello"}]
        assert service.usage_calls == []

    def test_skill_search_failure_never_costs_the_answer(self, gateway):
        service = _StubSkillsService(find_error=RuntimeError("index empty"))
        backend = HermesBackend(_config(gateway), [])
        backend.set_skills_service(service)

        result = asyncio.run(backend.query("hello", 5))

        assert result == "The stage is at 0,0."
        [seen] = GatewayHandler.requests_seen
        assert seen["payload"]["messages"] == [{"role": "user", "content": "hello"}]

    def test_disabled_skill_between_search_and_load_is_skipped(self, gateway):
        service = _StubSkillsService(
            texts={"haadf-imaging": "---\nname: HAADF imaging\n---\nUse the HAADF detector."}
        )
        backend = HermesBackend(_config(gateway), [])
        backend.set_skills_service(service)

        asyncio.run(backend.query("image the sample", 5))

        [seen] = GatewayHandler.requests_seen
        system = seen["payload"]["messages"][0]["content"]
        assert "Use the HAADF detector." in system
        assert "probe-alignment" not in system
        [(skill_ids, _, _)] = service.usage_calls
        assert skill_ids == ["haadf-imaging"]


class TestHermesConnectMCP:
    @pytest.fixture
    def hermes_home(self, tmp_path):
        (tmp_path / "config.yaml").write_text("agent:\n  max_turns: 150\n", encoding="utf-8")
        (tmp_path / "skills").mkdir()
        return tmp_path

    def _servers(self, hermes_home):
        import yaml

        return yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))["mcp_servers"]

    def test_connect_mcp_registers_server_in_gateway_config(self, gateway, hermes_home):
        backend = HermesBackend(_config(gateway, hermes_home=str(hermes_home)), [])
        asyncio.run(backend.connect_mcp("http://127.0.0.1:8001/mcp", "streamable_http"))
        servers = self._servers(hermes_home)
        assert servers["asyncroscopy-8001"] == {"url": "http://127.0.0.1:8001/mcp", "timeout": 120}

    def test_connect_mcp_preserves_other_config(self, gateway, hermes_home):
        import yaml

        backend = HermesBackend(_config(gateway, hermes_home=str(hermes_home)), [])
        asyncio.run(backend.connect_mcp("http://127.0.0.1:8000/mcp", "streamable_http"))
        data = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
        assert data["agent"] == {"max_turns": 150}

    def test_connect_mcp_reuses_existing_entry_for_same_url(self, gateway, hermes_home):
        backend = HermesBackend(_config(gateway, hermes_home=str(hermes_home)), [])
        asyncio.run(backend.connect_mcp("http://127.0.0.1:8001/mcp", "streamable_http"))
        asyncio.run(backend.connect_mcp("http://127.0.0.1:8001/mcp", "streamable_http"))
        assert list(self._servers(hermes_home)) == ["asyncroscopy-8001"]

    def test_connect_mcp_keeps_both_instruments(self, gateway, hermes_home):
        backend = HermesBackend(_config(gateway, hermes_home=str(hermes_home)), [])
        asyncio.run(backend.connect_mcp("http://127.0.0.1:8000/mcp", "streamable_http"))
        asyncio.run(backend.connect_mcp("http://127.0.0.1:8001/mcp", "streamable_http"))
        servers = self._servers(hermes_home)
        assert set(servers) == {"asyncroscopy-8000", "asyncroscopy-8001"}


class TestHermesSkillImport:
    class _ProposalService:
        def __init__(self, store_root):
            self.store = type("Store", (), {"root": store_root})()
            self.proposals: list[tuple[str, str]] = []

        def propose_skill(self, name, content):
            self.proposals.append((name, content))
            return "id"

        def find_skills(self, query, k=5):
            return []

    @pytest.fixture
    def hermes_home(self, tmp_path):
        home = tmp_path / "hermes"
        (home / "skills").mkdir(parents=True)
        (home / "config.yaml").write_text("{}", encoding="utf-8")
        return home

    def _write_skill(self, home, category, name, body="Do the thing."):
        skill_dir = home / "skills" / category / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\n---\n\n{body}", encoding="utf-8"
        )

    def test_new_hermes_skill_is_proposed_after_baseline(self, gateway, hermes_home, tmp_path):
        store_root = tmp_path / "store"
        store_root.mkdir()
        service = self._ProposalService(store_root)
        self._write_skill(hermes_home, "microscopy", "pre-existing")

        backend = HermesBackend(_config(gateway, hermes_home=str(hermes_home)), [])
        backend.set_skills_service(service)
        assert service.proposals == []

        self._write_skill(hermes_home, "microscopy", "focus-sweep")
        asyncio.run(backend.query("hello", 5))
        assert [name for name, _ in service.proposals] == ["focus-sweep"]

    def test_exported_store_skills_are_never_imported_back(self, gateway, hermes_home, tmp_path):
        store_root = tmp_path / "store"
        store_root.mkdir()
        service = self._ProposalService(store_root)

        backend = HermesBackend(_config(gateway, hermes_home=str(hermes_home)), [])
        backend.set_skills_service(service)
        self._write_skill(hermes_home, "asyncroscopy", "from-the-store")
        asyncio.run(backend.query("hello", 5))
        assert service.proposals == []


class TestFactory:
    def test_hermes_constructs_without_langchain(self):
        backend = create_backend("hermes", BackendConfig(hermes_url="http://127.0.0.1:8642"), [])
        assert isinstance(backend, HermesBackend)
        assert backend.name == "hermes"

    def test_name_is_case_insensitive(self):
        backend = create_backend("Hermes", BackendConfig(hermes_url="http://127.0.0.1:8642"), [])
        assert backend.name == "hermes"
