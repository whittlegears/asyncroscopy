"""Tango device exposing the asyncroscopy LangGraph agent layer.

The device is a thin wrapper over :mod:`asyncroscopy.agent`: it builds a chat
model (Ollama by default, or any API-key provider), loads tools from an MCP
server, keeps a roster of worker agents, and runs either a single skills-aware
ReAct agent or the supervisor swarm (``asyncroscopy.agent.graphs``).

Commands
--------
Query(prompt)          run the agent(s) and return the final answer
Complete(request_json) OpenAI-style single-step completion (no tools executed);
                       used by SciAgentGUI's llm_bridge and by TangoChatModel
ConnectMCP(config)     connect to another MCP server and inherit its tools
SpawnAgent(config)     add a worker agent to the swarm
RunWorkflow(config)    run a deterministic LangGraph workflow by name
ReloadSkills()         re-scan the skills directories
"""

import asyncio
import json
import sys
import time
from typing import Any

import tango
from tango.server import Device, attribute, command, device_property

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: F401
    from langchain_core.tools import BaseTool

    from asyncroscopy.agent.config import DEFAULT_MCP_URL, DEFAULT_OLLAMA_MODEL, DEFAULT_SKILLS_DIR, Agent, AgentSettings
    from asyncroscopy.agent.graphs.supervisor import (
        build_supervisor_graph,
        extract_json,
        parse_routing_decision,
        run_supervisor,
    )
    from asyncroscopy.agent.messages import langchain_message_to_openai, openai_messages_to_langchain
    from asyncroscopy.agent.state import SupervisorState as AgentState  # noqa: F401  (backwards compatible name)
    from asyncroscopy.agent.tools import filter_tools
except ImportError as exc:
    print(f"Missing dependencies ({exc})! Please run:")
    print("uv sync --extra agent")
    sys.exit(1)


class LLM(Device):
    mcp_url = device_property(dtype=str, default_value=DEFAULT_MCP_URL)
    startup_agents = device_property(dtype=(str,), default_value=())
    ollama_model = device_property(dtype=str, default_value=DEFAULT_OLLAMA_MODEL)
    use_init_chat_model = device_property(dtype=bool, default_value=False)
    model_provider = device_property(dtype=str, default_value="ollama")
    # Newer settings (see asyncroscopy.agent.config.AgentSettings). ``model_name``
    # wins over ``ollama_model`` when set; ``api_key_env`` names the environment
    # variable that holds the provider's key (never the key itself).
    model_name = device_property(dtype=str, default_value="")
    api_key_env = device_property(dtype=str, default_value="")
    base_url = device_property(dtype=str, default_value="")
    skills_dirs = device_property(dtype=(str,), default_value=())
    temperature = device_property(dtype=float, default_value=0.0)

    max_steps = attribute(label="Max Steps", dtype=int, access=tango.AttrWriteType.READ_WRITE)

    green_mode = tango.GreenMode.Asyncio

    async def init_device(self) -> None:
        await Device.init_device(self)
        self.set_state(tango.DevState.INIT)
        self._max_steps = 10

        # Registries
        self._agents: list[Agent] = []
        self._tools: list[BaseTool] = []
        self._mcp_targets: list[str] = []
        self._registry: Any = None
        self._model: Any = None

        if self.startup_agents:
            self._agents = [Agent(**json.loads(agent_json)) for agent_json in self.startup_agents]
            print(f"[SYSTEM]: Loaded startup agents: {self._agents}")

        try:
            settings = self._settings()

            if settings.provider == "ollama":
                from asyncroscopy.agent.ollama import ensure_ollama_running

                await ensure_ollama_running()

            from asyncroscopy.agent.models import build_chat_model

            self.info_stream(f"Initializing chat model: provider={settings.provider} model={settings.model}")
            self._model = build_chat_model(settings)

            print("\n[SYSTEM]: Pre-warming model (cold start)...")
            sys.stdout.flush()
            start_warmup = time.time()
            await self._model.ainvoke([HumanMessage(content=" ")])
            print(f"[SYSTEM]: Model pre-warmed in {time.time() - start_warmup:.2f}s!")

            self._load_skills(settings)

            # Connect to an MCP server initially if specified
            if self.mcp_url:
                config = json.dumps({"url": self.mcp_url, "transport": "streamable_http"})
                if not await self.ConnectMCP(config):
                    print(f"[SYSTEM]: Failed to connect to MCP Server at {self.mcp_url}.")

            self.set_state(tango.DevState.ON)
        except Exception as e:
            self.set_state(tango.DevState.FAULT)
            self.set_status(f"Initialization failed: {e}")
            self.error_stream(f"Failed to start: {e}")

    # ------------------------------------------------------------------ settings
    def _settings(self) -> AgentSettings:
        """Translate the Tango device properties into AgentSettings."""
        provider = (self.model_provider or "ollama").strip().lower()
        if provider == "tango":
            raise ValueError("The LLM device cannot use provider='tango' (it would call itself).")
        skills_dirs = list(self.skills_dirs) or [str(DEFAULT_SKILLS_DIR)]
        return AgentSettings(
            provider=provider,
            model=self.model_name or self.ollama_model,
            api_key_env=self.api_key_env or None,
            base_url=self.base_url or None,
            mcp_url=self.mcp_url or None,
            skills_dirs=skills_dirs,
            temperature=float(self.temperature),
            max_steps=self._max_steps,
            use_init_chat_model=bool(self.use_init_chat_model),
        )

    def _load_skills(self, settings: AgentSettings | None = None) -> int:
        from asyncroscopy.agent.skills.registry import SkillRegistry

        settings = settings or self._settings()
        self._registry = SkillRegistry.from_dirs(settings.resolved_skills_dirs)
        print(f"[SYSTEM]: Loaded {len(self._registry)} skill(s) from {settings.skills_dirs}")
        return len(self._registry)

    # ---------------------------------------------------------------- attributes
    def read_max_steps(self) -> int:
        return self._max_steps

    def write_max_steps(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_steps must be at least 1.")
        self._max_steps = value

    @attribute(dtype=(str,), max_dim_x=100)
    def agents(self) -> list[str]:
        """Return a list of the names of all currently spawned agents."""
        return [agent.name for agent in self._agents]

    @attribute(dtype=str)
    def tools(self) -> str:
        """JSON list of MCP tools inherited via ConnectMCP, e.g. [{"name": "..."}, ...].

        Consumed by scripts/llm_bridge.py (in the sciagentgui repo) for its /health
        endpoint and startup tool count.
        """
        return json.dumps([{"name": t.name} for t in self._tools])

    @attribute(dtype=str)
    def skills(self) -> str:
        """JSON list of installed skills: [{"name", "description", "version"}, ...]."""
        if self._registry is None:
            return "[]"
        return json.dumps(
            [{"name": s.name, "description": s.description, "version": s.version} for s in self._registry.list()]
        )

    # ------------------------------------------------------------------ commands
    @command(dtype_in=str, dtype_out=str)
    async def Query(self, prompt: str) -> str:
        """Query the agent swarm with a prompt, returning the final response."""
        self.set_state(tango.DevState.RUNNING)
        try:
            return await self._run_swarm(prompt)
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
        """OpenAI-compatible single-step chat completion.

        Unlike Query, this does not run any graph or execute tools itself: it
        converts the request into one chat-model call and returns the model's
        raw decision (tool_calls or final text) so the caller (SciAgentGUI's
        llm_bridge, or asyncroscopy.agent.models.TangoChatModel) can execute
        tools and drive the conversation.
        """
        try:
            request = json.loads(request_json)
            messages = self._openai_messages_to_langchain(request.get("messages") or [])
            tools = request.get("tools") or []
            model = self._model.bind_tools(tools) if tools else self._model
            response = await model.ainvoke(messages)
            return json.dumps({"message": self._langchain_message_to_openai(response)})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    @staticmethod
    def _openai_messages_to_langchain(messages: list[dict]) -> list[BaseMessage]:
        """Convert OpenAI-style chat messages into LangChain message objects."""
        return openai_messages_to_langchain(messages)

    @staticmethod
    def _langchain_message_to_openai(message: BaseMessage) -> dict:
        """Convert a LangChain AIMessage into an OpenAI-style assistant message dict."""
        return langchain_message_to_openai(message)

    @command(
        dtype_in=str,
        doc_in="JSON config of the MCP server: {'url': '...', 'transport': '...'}",
        dtype_out=bool,
        doc_out="Success status",
    )
    async def ConnectMCP(self, config: str) -> bool:
        """Connect to an MCP server and inherit its tools. Returns true for success."""
        try:
            from asyncroscopy.agent.tools import load_mcp_tools

            args = json.loads(config)
            url = args.get("url")
            transport = args.get("transport", "streamable_http")

            print(f"\n[SYSTEM]: Connecting to MCP Server at {url}...")
            tools = await load_mcp_tools(url, transport)

            self._mcp_targets.append(url)
            self._tools.extend(tools)
            print(f"[SYSTEM]: Connected. Inherited {len(tools)} tools.")
        except Exception as e:
            self.error_stream(f"Failed to connect to MCP server: {e}")
            return False

        return True

    @command(
        dtype_in=str,
        doc_in="JSON config of the Agent: {'name': '...', 'system_prompt': '...', 'model': '...', 'tools': ['*']}",
        dtype_out=bool,
        doc_out="Success status",
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
                description=args.get("description", ""),
            )
            self._agents.append(agent)
            print(f"\n[SYSTEM]: Successfully spawned agent '{agent.name}'")
            return True
        except Exception as e:
            self.error_stream(f"Failed to spawn agent: {e}")
            return False

    @command(dtype_out=int, doc_out="Number of skills loaded")
    def ReloadSkills(self) -> int:
        """Re-scan the skills directories and rebuild the search index."""
        if self._registry is None:
            return self._load_skills()
        return self._registry.reload()

    @command(
        dtype_in=str,
        doc_in="JSON {'name': 'image_eds_survey', 'input': {...}, 'use_model': true}",
        dtype_out=str,
        doc_out="JSON of the final workflow state (or {'error': ...})",
    )
    async def RunWorkflow(self, config: str) -> str:
        """Run a deterministic LangGraph workflow (see asyncroscopy.agent.graphs.workflows)."""
        self.set_state(tango.DevState.RUNNING)
        try:
            from asyncroscopy.agent.graphs.workflows import get_workflow_builder

            args = json.loads(config) if config else {}
            name = args.get("name", "image_eds_survey")
            builder = get_workflow_builder(name)
            model = self._model if args.get("use_model", True) else None
            graph = builder(self._tools, model=model)
            print(f"\n{'=' * 50}\n[WORKFLOW]: {name} {args.get('input') or {}}\n{'=' * 50}")
            result = await graph.ainvoke(
                dict(args.get("input") or {}), config={"recursion_limit": max(self._max_steps, 10)}
            )
            return json.dumps(result, default=str)
        except Exception as e:
            print(f"\n[WORKFLOW ERROR]: {e}")
            return json.dumps({"error": str(e)})
        finally:
            self.set_state(tango.DevState.ON)

    # ------------------------------------------------------------------ helpers
    def _get_agent_tools(self, allowed_patterns: list[str]) -> list:
        """Return a list of filtered tools based on the allowed glob patterns."""
        return filter_tools(self._tools, allowed_patterns)

    def _build_agent_executor(self, agent: Agent):
        """Filter this agent's tools and construct its skills-aware ReAct executor."""
        from asyncroscopy.agent.graphs.react import build_react_graph

        agent_tools = self._get_agent_tools(agent.tools)
        print(f"[SYSTEM]: Binding {len(agent_tools)} tools to {agent.name}")
        return build_react_graph(
            self._model, agent_tools, self._registry, system_prompt=agent.system_prompt, name=agent.name
        )

    def _extract_json(self, text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) if present."""
        return extract_json(text)

    def _parse_routing_decision(self, content: str, valid_options: list[str], fallback: str) -> tuple[str, str]:
        """Parse a supervisor response's {'next': ...} decision, falling back on any error or invalid value."""
        return parse_routing_decision(content, valid_options, fallback)

    async def _stream_agent(self, agent_executor, messages, agent_label: str = "") -> str:
        """Run an executor while streaming tokens and tool calls to stdout."""
        from asyncroscopy.agent.streaming import stream_agent

        return await stream_agent(agent_executor, messages, label=agent_label)

    async def _run_swarm(self, prompt: str) -> str:
        """Run the agent swarm with a given prompt, returning the final response."""
        if not self._agents:
            return "Swarm Error: No agents available. Please use the SpawnAgent command to create at least one worker before querying."

        # If there is a single agent, run it like a single agent (no need for supervisor/routing)
        if len(self._agents) == 1:
            agent = self._agents[0]
            agent_executor = self._build_agent_executor(agent)
            print(f"\n[{agent.name}] is working...")
            return await self._stream_agent(
                agent_executor, [HumanMessage(content=prompt)], agent_label=agent.name
            )

        graph = build_supervisor_graph(self._model, self._agents, self._build_agent_executor, self._stream_agent)
        return await run_supervisor(graph, prompt, self._max_steps)


# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    LLM.run_server()
