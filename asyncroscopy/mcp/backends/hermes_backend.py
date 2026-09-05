"""Hermes Agent backend: delegates queries to a Hermes OpenAI-compatible gateway over HTTP."""

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from asyncroscopy.skills.hermes_bridge import (
    IMPORT_STATE_FILE,
    import_new_skills,
    resolve_hermes_home,
)

from .base import Agent, AgentBackend, BackendConfig


class HermesBackend(AgentBackend):
    """Talks to a separately running Hermes Agent gateway (default port 8642).

    Query maps to one gateway chat completion, which runs Hermes's full
    autonomous agent loop server-side and returns only the final answer.
    Complete is unsupported: the gateway never exposes a raw model turn.
    ConnectMCP registers the server in the gateway's own config.yaml (which
    Hermes hot-reloads), so the gateway inherits the tools rather than the
    device. The operator's skill library flows both ways: relevant skills are
    searched per query and injected as a system message (usage logged like
    langgraph), and skills the Hermes agent writes to its own library are
    proposed back into the store for GUI review.
    """

    name = "hermes"
    supports_complete = False
    supports_connect_mcp = True

    # How many skills to search for (and at most inject) per query.
    skill_context_k = 3

    # Gateway agent loops run multi-step acquisitions; 300s is not enough.
    completion_timeout_s = 900.0

    def __init__(self, config: BackendConfig, agents: list[Agent]):
        super().__init__(config, agents)
        self._base_url = (config.hermes_url or "").rstrip("/")
        self._model = config.hermes_model or "hermes-agent"
        self._api_key = config.hermes_api_key or ""

    async def initialize(self) -> None:
        if not self._base_url:
            raise RuntimeError(
                "agent_backend is 'hermes' but hermes_url is empty. "
                "Set hermes_url to the Hermes gateway, e.g. http://127.0.0.1:8642"
            )
        await asyncio.to_thread(self._verify_gateway)

    async def query(self, prompt: str, max_steps: int) -> str:
        messages, skill_ids = await asyncio.to_thread(self._build_messages, prompt)
        answer = await asyncio.to_thread(self._run_completion, messages)
        if skill_ids:
            await asyncio.to_thread(self._record_skill_usage, skill_ids, prompt, answer)
        await asyncio.to_thread(self._import_agent_skills)
        return answer

    async def connect_mcp(self, url: str, transport: str) -> int:
        return await asyncio.to_thread(self._register_gateway_mcp, url)

    def _register_gateway_mcp(self, url: str) -> int:
        """Write the MCP server into the gateway's config.yaml; Hermes hot-reloads it.

        The gateway owns its tools, so this is the hermes equivalent of tool
        inheritance: one entry per URL, an existing entry for the same URL is
        reused. Returns 0 — the device never sees the gateway's tool list.
        """
        import yaml

        home = resolve_hermes_home(self.config.hermes_home)
        if home is None or not (home / "config.yaml").is_file():
            raise RuntimeError(
                "No Hermes install found (set the hermes_home device property "
                "or the HERMES_HOME environment variable)."
            )
        config_path = home / "config.yaml"
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        servers = data.setdefault("mcp_servers", {})
        for existing_name, entry in servers.items():
            if isinstance(entry, dict) and entry.get("url") == url:
                name = existing_name
                entry.setdefault("timeout", 120)
                break
        else:
            name = self._server_name(url, servers)
            servers[name] = {"url": url, "timeout": 120}
        config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"[SYSTEM]: Registered MCP server '{name}' ({url}) in {config_path}")
        return 0

    @staticmethod
    def _server_name(url: str, existing: dict) -> str:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "local"
        port = parsed.port or 80
        if host in ("127.0.0.1", "localhost"):
            name = f"asyncroscopy-{port}"
        else:
            name = f"asyncroscopy-{host.replace('.', '-')}-{port}"
        while name in existing:
            name += "x"
        return name

    # ------------------------------------------------------------------
    # Skills: inject the operator's skill library as request context
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str) -> tuple[list[dict], list[str]]:
        """Return the chat messages for a prompt, plus the ids of injected skills.

        The Hermes gateway runs its agent loop server-side and cannot call the
        device's find_skills/load_skill tools, so relevant skills are searched
        here and prepended as a system message instead. Any failure in the
        skill machinery is printed and swallowed — skills must never cost an
        answer.
        """
        user_message = {"role": "user", "content": prompt}
        if self._skills_service is None:
            return [user_message], []
        try:
            results = self._skills_service.find_skills(prompt, self.skill_context_k)
            loaded: list[tuple[str, str]] = []
            for result in results:
                skill_id = result.get("id")
                if not skill_id:
                    continue
                try:
                    loaded.append((skill_id, self._skills_service.load_skill(skill_id)))
                except KeyError:
                    continue  # disabled or vanished between search and load
            if not loaded:
                return [user_message], []
            sections = "\n\n".join(
                f"<skill id=\"{skill_id}\">\n{text}\n</skill>" for skill_id, text in loaded
            )
            system = {
                "role": "system",
                "content": (
                    "The operator maintains a library of approved skills for this "
                    "instrument. The following skills matched the current task — "
                    "follow the ones that apply and ignore the rest.\n\n" + sections
                ),
            }
            return [system, user_message], [skill_id for skill_id, _ in loaded]
        except Exception as exc:
            print(f"[SKILL CONTEXT ERROR]: {exc}")
            return [user_message], []

    def set_skills_service(self, service) -> None:
        super().set_skills_service(service)
        # Baseline scan: record what already exists so only future Hermes-authored
        # skills become proposals.
        self._import_agent_skills()

    def _import_agent_skills(self) -> None:
        store = getattr(self._skills_service, "store", None)
        if store is None:
            return
        try:
            home = resolve_hermes_home(self.config.hermes_home)
            if home is None or not (home / "skills").is_dir():
                return
            state_path = Path(store.root) / IMPORT_STATE_FILE
            proposed = import_new_skills(home / "skills", self._skills_service, state_path)
            if proposed:
                print(f"[SYSTEM]: Proposed {len(proposed)} Hermes-authored skill(s): {proposed}")
        except Exception as exc:
            print(f"[SKILL IMPORT ERROR]: {exc}")

    def _record_skill_usage(self, skill_ids: list[str], prompt: str, answer: str) -> None:
        """Log which skills were injected and whether the run produced an answer."""
        try:
            from asyncroscopy.skills.usage import task_hash

            self._skills_service.record_usage(skill_ids, task_hash(prompt), bool(answer))
        except Exception as exc:
            print(f"[USAGE LOG ERROR]: {exc}")

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _request(self, path: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST" if data is not None else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Hermes gateway at {self._base_url} rejected {path} "
                f"(HTTP {exc.code}): {self._error_detail(exc)}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Hermes gateway not reachable at {self._base_url}: {exc}. "
                "Start it with 'hermes gateway' (see docs/Operation/hermes_setup.md)."
            ) from exc

    @staticmethod
    def _error_detail(exc: "urllib.error.HTTPError") -> str:
        """Extract the gateway's error message from an HTTP error response body."""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            return exc.reason or "no detail"
        try:
            parsed = json.loads(body)
            error = parsed.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if isinstance(error, str) and error:
                return error
        except (json.JSONDecodeError, AttributeError):
            pass
        detail = body.strip()
        return detail[:300] if detail else (exc.reason or "no detail")

    def _verify_gateway(self) -> None:
        listing = self._request("/v1/models")
        model_ids = [entry.get("id") for entry in listing.get("data") or []]
        if self._model not in model_ids:
            raise RuntimeError(
                f"Hermes gateway at {self._base_url} does not serve model '{self._model}'. "
                f"Available: {model_ids}"
            )

    def _run_completion(self, messages: list[dict]) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        response = self._request("/v1/chat/completions", payload, timeout=self.completion_timeout_s)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Hermes gateway returned no choices: {json.dumps(response)[:500]}")
        return choices[0].get("message", {}).get("content") or ""
