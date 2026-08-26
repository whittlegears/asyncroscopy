"""Tango device exposing a pluggable AI agent backend (langgraph or hermes) over Tango commands."""

import asyncio
import json
from pathlib import Path

import tango
from tango.server import Device, attribute, command, device_property

from asyncroscopy.mcp.backends import create_backend
from asyncroscopy.mcp.backends.base import Agent, BackendConfig, BackendUnsupported
from asyncroscopy.skills import SkillsService

__all__ = ["Agent", "LLM"]


class LLM(Device):
    mcp_url = device_property(dtype=str, default_value="http://127.0.0.1:8000/mcp")
    startup_agents = device_property(dtype=(str,), default_value=())
    ollama_model = device_property(dtype=str, default_value="gemma4:31b")
    model_name = device_property(dtype=str, default_value="")
    model_provider = device_property(dtype=str, default_value="ollama")
    local_model_path = device_property(dtype=str, default_value="")
    ollama_host = device_property(dtype=str, default_value="http://localhost:11434")
    api_key = device_property(dtype=str, default_value="")
    use_init_chat_model = device_property(dtype=bool, default_value=False)
    agent_backend = device_property(dtype=str, default_value="langgraph")
    # Defaults to the Hermes gateway's standard local address so SetBackend('hermes')
    # works from a config that never mentions hermes (e.g. gemma-llm.yaml); the hermes
    # backend still verifies the gateway is actually reachable before going live.
    hermes_url = device_property(dtype=str, default_value="http://127.0.0.1:8642")
    hermes_model = device_property(dtype=str, default_value="hermes-agent")
    hermes_api_key = device_property(dtype=str, default_value="")
    skills_dir = device_property(dtype=str, default_value="outputs/agent_skills")
    embedding_model = device_property(dtype=str, default_value="nomic-embed-text")
    reflection_min_tool_steps = device_property(dtype=int, default_value=4)

    max_steps = attribute(label="Max Steps", dtype=int, access=tango.AttrWriteType.READ_WRITE)
    agents = attribute(dtype=(str,), max_dim_x=100)
    tools = attribute(dtype=str)
    backend = attribute(dtype=str)
    backend_capabilities = attribute(dtype=str)

    green_mode = tango.GreenMode.Asyncio

    async def init_device(self) -> None:
        await Device.init_device(self)
        self.set_state(tango.DevState.INIT)
        self._max_steps = 10
        self._backend = None

        self._agents: list[Agent] = []
        if self.startup_agents:
            self._agents = [Agent(**json.loads(agent_json)) for agent_json in self.startup_agents]
            print(f"[SYSTEM]: Loaded startup agents: {self._agents}")

        self._skills_service = None
        self._skill_sync_buffers: dict[str, dict[int, list]] = {}
        try:
            self._skills_service = SkillsService.at(
                Path(self.skills_dir), self.ollama_host, self.embedding_model
            )
        except Exception as e:
            print(f"[SYSTEM]: Skill store unavailable: {e}")

        try:
            self._backend = await self._start_backend(self.agent_backend)
            print(f"[SYSTEM]: Using agent backend '{self._backend.name}'")
            self.set_state(tango.DevState.ON)
        except Exception as e:
            self.set_state(tango.DevState.FAULT)
            self.set_status(f"Initialization failed: {e}")
            self.error_stream(f"Failed to start: {e}")

    def _backend_config(self) -> BackendConfig:
        """Snapshot the device properties a backend needs at creation time."""
        return BackendConfig(
            model_name=self.model_name or "",
            model_provider=self.model_provider or "ollama",
            ollama_model=self.ollama_model or "",
            ollama_host=self.ollama_host or "http://localhost:11434",
            local_model_path=self.local_model_path or "",
            api_key=self.api_key or "",
            use_init_chat_model=bool(self.use_init_chat_model),
            hermes_url=self.hermes_url or "",
            hermes_model=self.hermes_model or "hermes-agent",
            hermes_api_key=self.hermes_api_key or "",
            reflection_min_tool_steps=int(self.reflection_min_tool_steps),
        )

    async def _start_backend(self, name: str):
        """Build and initialize the named backend, connecting MCP before it goes live.

        Raises on any failure, leaving ``self._backend`` untouched so the caller
        decides whether the device keeps its current backend or goes FAULT.
        """
        backend = create_backend(name, self._backend_config(), self._agents)
        await backend.initialize()

        if self.mcp_url and backend.supports_connect_mcp:
            try:
                await backend.connect_mcp(self.mcp_url, "streamable_http")
            except Exception as e:
                print(f"[SYSTEM]: Failed to connect to MCP Server at {self.mcp_url}: {e}")

        if self._skills_service is not None:
            backend.set_skills_service(self._skills_service)

        return backend

    def read_max_steps(self) -> int:
        return self._max_steps

    def write_max_steps(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_steps must be at least 1.")
        self._max_steps = value

    def read_agents(self) -> list[str]:
        """Return a list of the names of all currently spawned agents."""
        return [agent.name for agent in self._agents]

    def read_tools(self) -> str:
        """JSON list of MCP tools inherited via ConnectMCP, e.g. [{"name": "..."}, ...]."""
        if self._backend is None:
            return json.dumps([])
        return json.dumps([{"name": name} for name in self._backend.tool_names()])

    def read_backend(self) -> str:
        """Name of the active agent backend, read by llm_bridge.py's /health endpoint."""
        if self._backend is None:
            raise RuntimeError("No agent backend initialized.")
        return self._backend.name

    def read_backend_capabilities(self) -> str:
        """JSON capability flags of the active backend, e.g. {"complete": true, "connect_mcp": true}."""
        if self._backend is None:
            raise RuntimeError("No agent backend initialized.")
        return json.dumps(self._backend.capabilities())


    @command(dtype_in=str, dtype_out=str)
    async def Query(self, prompt: str) -> str:
        """Query the agent backend with a prompt, returning the final response."""
        self.set_state(tango.DevState.RUNNING)
        try:
            return await self._backend.query(prompt, self._max_steps)
        except Exception as e:
            print(f"\n[CRITICAL ERROR]: {e}")
            return str(e)
        finally:
            self.set_state(tango.DevState.ON)

    @command(
        dtype_in=str,
        doc_in="OpenAI-style {'messages': [...], 'tools': [...]}",
        dtype_out=str,
        doc_out="JSON {'message': {...}} on success, or {'error': {'message': ...}} on failure",
    )
    async def Complete(self, request_json: str) -> str:
        """OpenAI-compatible single-step chat completion for the llm_bridge.py HTTP bridge.

        Unlike Query, this does not run the agent loop or execute any tools
        itself — it asks the backend for one raw model decision (tool_calls or
        final text) so the caller (e.g. SciAgentGUI's own agent loop) can
        execute tools and drive the conversation. Backends without a raw model
        turn (hermes) return an error payload here; callers should consult the
        backend_capabilities attribute and fall back to Query.
        """
        try:
            request = json.loads(request_json)
            message = await self._backend.complete(request)
            return json.dumps({"message": message})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    @command(
        dtype_in=str,
        doc_in="Backend name: langgraph | hermes",
        dtype_out=bool,
        doc_out="True when the named backend is now live",
    )
    async def SetBackend(self, name: str) -> bool:
        """Switch the agent backend at runtime.

        The new backend is fully built, initialized and MCP-connected before it
        replaces the old one, so a failed switch (an unreachable hermes gateway,
        an unknown name) leaves the current backend running and returns False
        with the reason in the device status.
        """
        normalized = (name or "").strip().lower()
        if self._backend is not None and self._backend.name == normalized:
            return True

        self.set_state(tango.DevState.INIT)
        try:
            self._backend = await self._start_backend(normalized)
            print(f"[SYSTEM]: Switched agent backend to '{self._backend.name}'")
            self.set_status(f"Agent backend: {self._backend.name}")
            return True
        except Exception as e:
            message = f"Backend switch to '{name}' failed: {e}"
            self.error_stream(message)
            self.set_status(message)
            return False
        finally:
            self.set_state(tango.DevState.ON if self._backend is not None else tango.DevState.FAULT)

    @command(
        dtype_in=str,
        doc_in="Chunked skill sync envelope: {'version', 'sync_id', 'seq', 'total', 'skills': [...]}",
        dtype_out=str,
        doc_out="JSON {'status': 'applied'|'buffered', ...} or {'error': {'message': ...}}",
    )
    async def SyncSkills(self, payload_json: str) -> str:
        """Replace the device's skill store with the GUI's skills and reindex.

        Large payloads arrive as multiple calls sharing a sync_id; the full
        replacement runs only once every chunk of the newest sync_id is here.
        """
        if self._skills_service is None:
            return json.dumps({"error": {"message": "The skill store is unavailable on this device."}})
        try:
            envelope = json.loads(payload_json)
            sync_id = str(envelope.get("sync_id", "single"))
            seq = int(envelope.get("seq", 0))
            total = int(envelope.get("total", 1))
            skills = envelope.get("skills") or []

            if sync_id not in self._skill_sync_buffers:
                self._skill_sync_buffers = {sync_id: {}}
            self._skill_sync_buffers[sync_id][seq] = skills

            received = self._skill_sync_buffers[sync_id]
            if len(received) < total:
                return json.dumps({"status": "buffered", "received": len(received), "of": total})

            merged = [skill for _, chunk in sorted(received.items()) for skill in chunk]
            self._skill_sync_buffers = {}
            report = await asyncio.to_thread(self._skills_service.sync_from_payload, merged)
            print(f"[SYSTEM]: Skill sync applied: {report}")
            return json.dumps({"status": "applied", **report})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    @command(
        dtype_in=str,
        doc_in="JSON {'query': '...', 'k': 5}",
        dtype_out=str,
        doc_out="JSON {'results': [...]} or {'error': {'message': ...}}",
    )
    async def SearchSkills(self, request_json: str) -> str:
        """Hybrid search over the synced skill store, refusing with the reason when it cannot run."""
        if self._skills_service is None:
            return json.dumps({"error": {"message": "The skill store is unavailable on this device."}})
        try:
            request = json.loads(request_json)
            query = str(request.get("query", "")).strip()
            if not query:
                return json.dumps({"error": {"message": "No query given."}})
            k = int(request.get("k", 5))
            results = await asyncio.to_thread(self._skills_service.find_skills, query, k)
            return json.dumps({"results": results})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    @command(
        dtype_in=str,
        doc_in="JSON {'skill_ids': [...], 'task_hash': '...', 'success': true|false}",
        dtype_out=str,
        doc_out="JSON {'status': 'recorded', 'rows': n} or {'error': {'message': ...}}",
    )
    async def RecordSkillUsage(self, request_json: str) -> str:
        """Log which skills a client-side run loaded and whether it succeeded."""
        if self._skills_service is None:
            return json.dumps({"error": {"message": "The skill store is unavailable on this device."}})
        try:
            request = json.loads(request_json)
            skill_ids = request.get("skill_ids") or []
            if not isinstance(skill_ids, list) or not skill_ids:
                return json.dumps({"error": {"message": "Send {'skill_ids': [...]} with at least one id."}})
            rows = await asyncio.to_thread(
                self._skills_service.record_usage,
                skill_ids,
                str(request.get("task_hash", "")),
                bool(request.get("success")),
            )
            return json.dumps({"status": "recorded", "rows": rows})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    @command(
        dtype_out=str,
        doc_out="JSON {'skills': [{'id', 'name', 'enabled', 'loads', 'successes', 'failures', 'last_used_at'}, ...]}",
    )
    async def SkillUsageReport(self) -> str:
        """Per-skill usage statistics for the GUI's skills-by-usage view."""
        if self._skills_service is None:
            return json.dumps({"error": {"message": "The skill store is unavailable on this device."}})
        try:
            report = await asyncio.to_thread(self._skills_service.usage_report)
            return json.dumps({"skills": report})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    @command(
        dtype_out=str,
        doc_out="JSON {'proposals': [{'id', 'name', 'content', 'created_at'}, ...]} or {'error': ...}",
    )
    async def ListSkillProposals(self) -> str:
        """List agent-written skill drafts waiting for the GUI to pull into review."""
        if self._skills_service is None:
            return json.dumps({"error": {"message": "The skill store is unavailable on this device."}})
        try:
            proposals = await asyncio.to_thread(self._skills_service.list_proposals)
            return json.dumps({"proposals": proposals})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    @command(
        dtype_in=str,
        doc_in="Proposal id from ListSkillProposals",
        dtype_out=str,
        doc_out="JSON {'status': 'removed'} or {'error': {'message': ...}}",
    )
    async def RemoveSkillProposal(self, proposal_id: str) -> str:
        """Delete one skill proposal, called by the GUI after it has imported it."""
        if self._skills_service is None:
            return json.dumps({"error": {"message": "The skill store is unavailable on this device."}})
        try:
            removed = await asyncio.to_thread(
                self._skills_service.remove_proposal, proposal_id.strip()
            )
            if not removed:
                return json.dumps({"error": {"message": f"No proposal has the id '{proposal_id}'."}})
            return json.dumps({"status": "removed"})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    @command(
        dtype_in=str,
        doc_in="JSON config of the MCP server: {'url': '...', 'transport': '...'}",
        dtype_out=bool,
        doc_out="Success status"
    )
    async def ConnectMCP(self, config: str) -> bool:
        """Connect to an MCP server and inherit its tools. Returns true for success."""
        try:
            args = json.loads(config)
            url = args.get("url")
            transport = args.get("transport", "streamable_http")
            await self._backend.connect_mcp(url, transport)
        except BackendUnsupported as e:
            self.error_stream(str(e))
            return False
        except Exception as e:
            self.error_stream(f"Failed to connect to MCP server: {e}")
            return False

        return True

    @command(
        dtype_in=str,
        doc_in="JSON config of the Agent: {'name': '...', 'system_prompt': '...', 'model': '...', 'tools': ['*']}",
        dtype_out=bool,
        doc_out="Success status"
    )
    def SpawnAgent(self, config: str) -> bool:
        """Creates a new agent in the swarm."""
        try:
            args = json.loads(config)
            agent = Agent(
                name=args["name"],
                system_prompt=args["system_prompt"],
                model=args.get("model", self.ollama_model),
                tools=args.get("tools", ["*"]),
                description=args.get("description", "")
            )
            self._agents.append(agent)
            print(f"\n[SYSTEM]: Successfully spawned agent '{agent.name}'")
            return True
        except Exception as e:
            self.error_stream(f"Failed to spawn agent: {e}")
            return False



if __name__ == "__main__":
    LLM.run_server()
