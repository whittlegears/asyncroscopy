"""Tango device exposing a pluggable AI agent backend (langgraph or hermes) over Tango commands."""

import json

import tango
from tango.server import Device, attribute, command, device_property

from asyncroscopy.mcp.backends import create_backend
from asyncroscopy.mcp.backends.base import Agent, BackendConfig, BackendUnsupported

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
    hermes_url = device_property(dtype=str, default_value="")
    hermes_model = device_property(dtype=str, default_value="hermes-agent")
    hermes_api_key = device_property(dtype=str, default_value="")

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

        config = BackendConfig(
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
        )

        try:
            self._backend = create_backend(self.agent_backend, config, self._agents)
            print(f"[SYSTEM]: Using agent backend '{self._backend.name}'")
            await self._backend.initialize()

            if self.mcp_url and self._backend.supports_connect_mcp:
                mcp_config = json.dumps({"url": self.mcp_url, "transport": "streamable_http"})
                if not await self.ConnectMCP(mcp_config):
                    print(f"[SYSTEM]: Failed to connect to MCP Server at {self.mcp_url}.")

            self.set_state(tango.DevState.ON)
        except Exception as e:
            self.set_state(tango.DevState.FAULT)
            self.set_status(f"Initialization failed: {e}")
            self.error_stream(f"Failed to start: {e}")

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
