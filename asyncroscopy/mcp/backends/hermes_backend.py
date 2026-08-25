"""Hermes Agent backend: delegates queries to a Hermes OpenAI-compatible gateway over HTTP."""

import asyncio
import json
import urllib.request
import urllib.error

from .base import Agent, AgentBackend, BackendConfig


class HermesBackend(AgentBackend):
    """Talks to a separately running Hermes Agent gateway (default port 8642).

    Query maps to one gateway chat completion, which runs Hermes's full
    autonomous agent loop server-side and returns only the final answer.
    Complete is unsupported: the gateway never exposes a raw model turn, so a
    caller wanting per-tool control cannot get it from this backend.
    """

    name = "hermes"
    supports_complete = False
    supports_connect_mcp = False

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
        return await asyncio.to_thread(self._run_completion, prompt)

    def _request(self, path: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST" if data is not None else "GET"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _verify_gateway(self) -> None:
        try:
            listing = self._request("/v1/models")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Hermes gateway not reachable at {self._base_url}: {exc}. "
                "Start it from the hermes-agent checkout before starting this device."
            ) from exc
        model_ids = [entry.get("id") for entry in listing.get("data") or []]
        if self._model not in model_ids:
            raise RuntimeError(
                f"Hermes gateway at {self._base_url} does not serve model '{self._model}'. "
                f"Available: {model_ids}"
            )

    def _run_completion(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        response = self._request("/v1/chat/completions", payload, timeout=300.0)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Hermes gateway returned no choices: {json.dumps(response)[:500]}")
        return choices[0].get("message", {}).get("content") or ""
