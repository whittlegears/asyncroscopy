"""``swarm``: a supervisor and worker agents (spawned at run time) that inherit their tools from MCP servers.

The agent MCP server (``asyncroscopy.mcp.agent_mcp_server``) serves it as ``query_agent`` with no graph
named, beside the ``mapping``, ``general`` and ``team`` graphs of this package.
"""

import asyncio
import fnmatch
import json
import operator
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Annotated, Sequence, TypedDict

from asyncroscopy.core.names import CALLER_HEADER, DEFAULT_MCP_URL

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.mcp_tools import McpServer
from asyncroscopy import settings
from asyncroscopy.agents.profile import load_profile

DEFAULT_MAX_STEPS = 10
ollama_port = 11434
DEFAULT_OLLAMA_HOST = f"http://localhost:{ollama_port}"


def ollama_host() -> str:
    """The Ollama server: OLLAMA_HOST, the variable the Ollama client behind ChatOllama reads, else DEFAULT_OLLAMA_HOST;
    a missing scheme or port is filled in as the Ollama client does."""
    host = settings.get().ollama_host or DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = f"http://{host}"
    scheme, _, rest = host.partition("://")
    address = rest.split("/", 1)[0]
    if not address.rsplit("]", 1)[-1].count(":"):
        rest = rest.replace(address, f"{address}:{ollama_port}", 1)
    return f"{scheme}://{rest}".rstrip("/")


@dataclass
class Agent:
    """Represents a single AI agent in the swarm."""

    name: str
    system_prompt: str
    tools: list[str]  # tool names, which may be glob patterns like math_*
    model: str | None = None
    description: str = ""


class AgentState(TypedDict):
    """State dictionary for each Agent node in the swarm graph."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    current_task: str


@dataclass
class _StreamState:
    """What one streamed agent run has printed and collected so far."""

    prefix: str
    transcript: list[str] | None
    start_time: float
    first_token_received: bool = False
    final_content: str = ""
    generated: str = ""


def _record(transcript: list[str] | None, line: str) -> None:
    """Add a line to the transcript when there is one."""
    if transcript is not None:
        transcript.append(line)


def _on_model_stream(state: _StreamState, event: dict) -> None:
    """Print one streamed chunk and time the first token."""
    chunk = event["data"]["chunk"]
    if not state.first_token_received:
        ttft = time.time() - state.start_time
        print(f"\n{state.prefix}[DIAGNOSTIC]: Time to first token: {ttft:.2f}s")
        state.first_token_received = True
    if chunk.content:
        state.generated += chunk.content
        print(chunk.content, end="")
        sys.stdout.flush()


def _on_model_end(state: _StreamState, event: dict) -> None:
    """Keep the model output as the final answer when it calls no tools."""
    output = event["data"]["output"]
    tool_calls = getattr(output, "tool_calls", None) or []
    if not tool_calls:
        state.final_content = (output.content or "").strip()
    print()


def _on_tool_start(state: _StreamState, event: dict) -> None:
    """Print and record a tool call that is starting."""
    tool_name = event["name"]
    tool_input = event["data"].get("input")
    line = f"{state.prefix}EXECUTING TOOL: {tool_name}({tool_input})"
    print(f"{state.prefix}[EXECUTING TOOL]: {tool_name}({tool_input})")
    _record(state.transcript, line)


def _on_tool_end(state: _StreamState, event: dict) -> None:
    """Print and record the result of a tool call."""
    output = event["data"].get("output")
    print(f"{state.prefix}[TOOL RESULT]: {output}")
    _record(state.transcript, f"{state.prefix}TOOL RESULT: {output}")


_STREAM_HANDLERS = {
    "on_chat_model_stream": _on_model_stream,
    "on_chat_model_end": _on_model_end,
    "on_tool_start": _on_tool_start,
    "on_tool_end": _on_tool_end,
}


def _handle_stream_event(state: _StreamState, event: dict) -> None:
    """Update the stream state for one event from the agent."""
    kind = event["event"]
    if kind == "on_chat_model_start":
        state.start_time = time.time()
        state.first_token_received = False
        return
    handler = _STREAM_HANDLERS.get(kind)
    if handler:
        handler(state, event)


def _supervisor_prompt(agent_roster: str, options: list[str]) -> SystemMessage:
    """Build the system message that tells the supervisor how to route."""
    instructions = (
        f"Below are the available agents and what each is for:\n{agent_roster}\n\n"
        "Based on the conversation, decide which agent should act next to progress the user's request. "
        "Only output FINISH if the user's request has been fully and concretely answered — "
        "not if an agent asked a question, refused, said it lacks the ability, or otherwise failed to "
        "complete the task; in that case, route to a different, more suitable agent instead."
    )
    return SystemMessage(
        content=(
            f"You are the Swarm Supervisor. {instructions}\n"
            "Respond with JSON containing two keys:\n"
            f"1. 'next': One of {options}\n"
            "2. 'task': The exact, isolated sub-task that ONLY this specific agent should perform right now. "
            "Do NOT include steps intended for other agents.\n\n"
            "Example output:\n"
            '{"next": "image", "task": "Acquire a scanned HAADF image."}'
        )
    )


class AgentSwarm:
    """A supervisor plus N ReAct worker agents, bound to tools inherited over MCP."""

    def __init__(
        self,
        model: str | None = None,
        model_provider: str = "ollama",
        max_steps: int = DEFAULT_MAX_STEPS,
        startup_agents: list[Agent] | None = None,
        use_init_chat_model: bool = False,
        num_ctx: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            model (str): Model name, interpreted by ``model_provider``. None: the default agent profile's model.
            model_provider (str): Provider for ``init_chat_model`` when
                ``use_init_chat_model`` is set; "ollama" otherwise. Defaults to "ollama".
            max_steps (int): LangGraph recursion limit for a single query. Defaults to 10.
            startup_agents (list[Agent], optional): Agents to preload at startup.
            use_init_chat_model (bool): Initialize the model through LangChain's
                ``init_chat_model`` (OpenAI, Anthropic, ...) instead of ChatOllama.
            verbose (bool): If True, print model and MCP connection progress. Defaults to True.
        """
        # Ollama's default context cuts long tool lists without warning, so always set it: the profile's.
        profile = load_profile(settings.get().agent_profile)
        self.model = model or profile.model
        self.model_provider = model_provider
        self.max_steps = int(max_steps)
        self.use_init_chat_model = use_init_chat_model
        self.num_ctx = int(num_ctx or profile.num_ctx)
        self.verbose = verbose

        self._agents: list[Agent] = list(startup_agents or [])
        self._tools: list[BaseTool] = []
        self._mcp_clients: list[McpServer] = []
        self._model = None
        self.startup_error: str | None = None


    @property
    def ready(self) -> bool:
        """True once a model is available"""
        return self._model is not None

    async def start(self, mcp_urls: list[str] | None = None) -> bool:
        """
        Create the model, pre-warm it, and inherit tools from every MCP URL.
        Returns True when a model is created.
        """
        if self._agents and self.verbose:
            print(f"[SYSTEM]: Loaded startup agents: {self._agents}")

        started = True
        try:
            if not self.use_init_chat_model or self.model_provider == "ollama":
                await self.ensure_ollama_running()
            self._model = self._make_model(self.model)
            print("\n[SYSTEM]: Pre-warming model (Cold Start)...")
            sys.stdout.flush()
            start_warmup = time.time()
            await self._model.ainvoke([HumanMessage(content=" ")])
            print(f"[SYSTEM]: Model pre-warmed in {time.time() - start_warmup:.2f}s!")
        except Exception as e:
            self._model = None
            self.startup_error = f"{type(e).__name__}: {e}"
            self._error(f"[SYSTEM]: Failed to start the model: {self.startup_error}")
            self._error("[SYSTEM]: Serving MCP anyway; query_agent reports this error unless a caller names a model.")
            started = False

        # The tools are inherited even without a model: a caller may name its own model (use_model) per query.
        for url in mcp_urls or [DEFAULT_MCP_URL]:
            if not await self.connect_mcp(url):
                print(f"[SYSTEM]: Failed to connect to MCP Server at {url}.")

        return started

    def _make_model(self, model: str):
        """The chat model ``model`` under this swarm's provider and context."""
        if self.use_init_chat_model:
            self._log(f"Initializing {model} via init_chat_model")
            return init_chat_model(model=model, model_provider=self.model_provider, temperature=0)
        from langchain_ollama import ChatOllama

        self._log(f"Initializing {model} via ChatOllama")
        return ChatOllama(model=model, temperature=0, reasoning=False, num_ctx=self.num_ctx)

    def use_model(self, model: str) -> None:
        """Run the next queries on ``model`` (a name under this swarm's provider), such as the one a client has
        chosen. Not pre-warmed: the client is already using it, so it is loaded."""
        if model and (model != self.model or self._model is None):
            self._model = self._make_model(model)
            self.model = model
            self.startup_error = None

    def _log(self, message: str) -> None:
        """Print a progress line when verbose, otherwise stay silent."""
        if self.verbose:
            print(message)

    def _error(self, message: str) -> None:
        """Report a failure regardless of verbosity, since callers cannot see our state."""
        print(message, file=sys.stderr)


    def add_tools(self, tools: list[BaseTool]) -> None:
        """Give the agents extra tools that run in this process."""
        have = {t.name for t in self._tools}
        self._tools.extend(t for t in tools if t.name not in have)

    async def connect_mcp(self, url: str, transport: str = "streamable_http") -> bool:
        """Connect to an MCP server and inherit its tools. Returns true for success."""
        try:
            client = McpServer(url, {CALLER_HEADER: "agent-swarm"})

            print(f"\n[SYSTEM]: Connecting to MCP Server at {url}...")

            tools = await client.langchain_tools()

            self._mcp_clients.append(client)
            self._tools.extend(tools)
            print(f"[SYSTEM]: Connected. Inherited {len(tools)} tools.")
        except Exception as e:
            self._error(f"Failed to connect to MCP server {url!r}: {e}")
            return False

        return True

    async def ensure_ollama_running(self, host: str | None = None, timeout: int = 10) -> None:
        """Check if Ollama server is running, offloaded to prevent blocking the event loop. ``host``: the server
        (default: ``ollama_host()``, the same one ChatOllama talks to)."""
        host = host or ollama_host()

        def _sync_check():
            tags_url = f"{host.rstrip('/')}/api/tags"
            try:
                with urllib.request.urlopen(tags_url, timeout=1):
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
                pass

            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except FileNotFoundError:
                raise RuntimeError("Ollama binary not found on PATH.")

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    with urllib.request.urlopen(tags_url, timeout=1):
                        return
                except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
                    time.sleep(0.5)

            raise RuntimeError(f"Ollama endpoint '{tags_url}' did not respond.")

        await asyncio.to_thread(_sync_check)


    @property
    def agents(self) -> list[str]:
        """Return a list of the names of all currently spawned agents."""
        return [agent.name for agent in self._agents]

    @property
    def tools(self) -> str:
        """JSON list of MCP tools inherited over MCP, e.g. [{"name": "..."}, ...]."""
        return json.dumps([{"name": t.name} for t in self._tools])

    def set_max_steps(self, max_steps: int) -> int:
        """Set the LangGraph recursion limit used per query and return the new value."""
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        self.max_steps = int(max_steps)
        return self.max_steps


    async def query(self, prompt: str, include_transcript: bool = True) -> str:
        """
        Query the agent swarm with a prompt, returning the final response.

        If `include_transcript` is true (default), the routing decisions, 
        tool calls, and generated text are also included.
        """
        if not self.ready:
            return f"Agent Error: no model is available ({self.startup_error})."
        try:
            transcript: list[str] | None = [] if include_transcript else None
            final = await self._run_swarm(prompt, transcript=transcript)
            if not transcript:
                return final
            return "\n".join([*transcript, "", "FINAL ANSWER:", final])
        except Exception as e:
            print(f"\n[CRITICAL ERROR]: {e}")
            return str(e)

    async def complete(self, request_json: str) -> str:
        """OpenAI-compatible single-step chat completion."""
        if not self.ready:
            return json.dumps({"error": {"message": f"no model is available ({self.startup_error})"}})
        try:
            request = json.loads(request_json)
            messages = self._openai_messages_to_langchain(request.get("messages") or [])
            tools = request.get("tools") or []
            model = self._model.bind_tools(tools) if tools else self._model
            response = await model.ainvoke(messages)
            return json.dumps({"message": self._langchain_message_to_openai(response)})
        except Exception as e:
            return json.dumps({"error": {"message": str(e)}})

    def spawn_agent(self, config: str) -> str:
        """Create a new agent in the swarm from a JSON config, returning its name."""
        try:
            args = json.loads(config)
            agent = Agent(
                name=args["name"],
                system_prompt=args["system_prompt"],
                model=args.get("model", self.model),
                tools=args.get("tools", ["*"]),
                description=args.get("description", ""),
            )
            self._agents.append(agent)
            print(f"\n[SYSTEM]: Successfully spawned agent '{agent.name}'")
            return agent.name
        except Exception as e:
            self._error(f"Failed to spawn agent: {e}")
            return ""


    @staticmethod
    def _openai_messages_to_langchain(messages: list[dict]) -> list[BaseMessage]:
        """Convert OpenAI-style chat messages into LangChain message objects."""
        converted: list[BaseMessage] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")
            if role == "system":
                converted.append(SystemMessage(content=content or ""))
            elif role == "assistant":
                tool_calls = [
                    {
                        "name": call["function"]["name"],
                        "args": json.loads(call["function"].get("arguments") or "{}"),
                        "id": call.get("id", ""),
                    }
                    for call in (message.get("tool_calls") or [])
                ]
                converted.append(AIMessage(content=content or "", tool_calls=tool_calls))
            elif role == "tool":
                converted.append(
                    ToolMessage(content=content or "", tool_call_id=message.get("tool_call_id", ""))
                )
            else:
                converted.append(HumanMessage(content=content or ""))
        return converted

    @staticmethod
    def _langchain_message_to_openai(message: BaseMessage) -> dict:
        """Convert a LangChain AIMessage into an OpenAI-style assistant message dict."""
        result: dict = {"role": "assistant", "content": message.content or ""}
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.get("id") or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("args") or {}),
                    },
                }
                for index, call in enumerate(tool_calls)
            ]
        return result


    def _get_agent_tools(self, allowed_patterns: list[str]) -> list:
        """Returnes a list of filtered tools based on the allowed glob patterns."""
        if "*" in allowed_patterns:
            return self._tools
        return [t for t in self._tools if any(fnmatch.fnmatch(t.name, pat) for pat in allowed_patterns)]

    def _build_agent_executor(self, agent: Agent):
        """Filter this agent's tools and construct its ReAct executor."""
        agent_tools = self._get_agent_tools(agent.tools)
        print(f"[SYSTEM]: Binding {len(agent_tools)} tools to {agent.name}")
        return create_agent(model=self._model, tools=agent_tools, system_prompt=agent.system_prompt)

    def _extract_json(self, text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) if present."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    def _parse_routing_decision(self, content: str, valid_options: list[str], fallback: str) -> tuple[str, str]:
        """Parse a supervisor response's {'next': ...} decision, falling back on any error or invalid value."""
        try:
            decision = json.loads(self._extract_json(content))
            next_agent = decision.get("next", fallback)
            subtask = decision.get("task", "")

            return next_agent if next_agent in valid_options else fallback, subtask
        except Exception as e:
            print(f"[SUPERVISOR ERROR]: {e}")
            return fallback, ""

    async def _stream_agent(
        self,
        agent_executor,
        messages,
        agent_label: str = "",
        transcript: list[str] | None = None,
        recursion_limit: int | None = None,
    ) -> str:
        """Run a create_agent executor while streaming tokens and tool calls."""
        prefix = f"[{agent_label}] " if agent_label else ""
        state = _StreamState(prefix=prefix, transcript=transcript, start_time=time.time())

        config = {"recursion_limit": recursion_limit} if recursion_limit else None
        async for event in agent_executor.astream_events({"messages": messages}, config=config, version="v2"):
            _handle_stream_event(state, event)

        if state.generated.strip():
            _record(transcript, f"{prefix}{state.generated.strip()}")

        if state.final_content:
            print(f"{prefix}[FINAL ANSWER RETURNED]:\n{state.final_content}\n{'=' * 50}")
        return state.final_content

    async def _run_single_agent(self, prompt: str, transcript: list[str] | None) -> str:
        """Run the only agent on the prompt with no supervisor."""
        agent = self._agents[0]
        agent_executor = self._build_agent_executor(agent)
        print(f"\n[{agent.name}] is working...")
        # max_steps limits a lone agent too.
        return await self._stream_agent(
            agent_executor, [HumanMessage(content=prompt)], agent_label=agent.name, transcript=transcript,
            recursion_limit=self.max_steps,
        )

    def _make_agent_node(self, agent: Agent, transcript: list[str] | None):
        """Build the graph node that runs one worker agent."""
        agent_executor = self._build_agent_executor(agent)

        async def node(state: AgentState):
            task = state.get("current_task", "Execute assigned tool.")
            print(f"\n[{agent.name}] assigned task: '{task}'")
            _record(transcript, f"{agent.name} assigned task: {task}")

            content = await self._stream_agent(
                agent_executor, [HumanMessage(content=task)], agent_label=agent.name, transcript=transcript
            )
            print(f"[{agent.name}] finished.\n")
            return {
                "messages": [
                    HumanMessage(content=f"[{agent.name}]: {content}", name=agent.name)
                ]
            }

        return node

    def _make_supervisor_node(self, agent_names: list[str], transcript: list[str] | None):
        """Build the graph node that picks the next agent."""
        options = agent_names + ["FINISH"]
        agent_roster = "\n".join(f"- {a.name}: {a.description or a.system_prompt}" for a in self._agents)

        async def supervisor_node(state: AgentState):
            print("\n[Supervisor] Evaluating routing...")

            # More than one message means an agent has already replied.
            has_delegated = len(state["messages"]) > 1

            if not has_delegated:
                # The first turn must go to an agent, so FINISH is not allowed yet.
                valid_options, fallback = agent_names, agent_names[0]
            else:
                valid_options, fallback = options, "FINISH"

            sys_prompt = _supervisor_prompt(agent_roster, options)
            response = await self._model.ainvoke([sys_prompt] + state["messages"])
            next_agent, subtask = self._parse_routing_decision(response.content, valid_options, fallback)
            print(f"[Supervisor] Decision: {next_agent}")
            _record(transcript, f"Supervisor -> {next_agent}: {subtask}" if next_agent != "FINISH" else "Supervisor -> FINISH")
            if next_agent == "FINISH":
                print("[Supervisor] Decision: FINISH\n")

            return {
                "next_agent": next_agent,
                "current_task": subtask,
            }

        return supervisor_node

    def _build_swarm_graph(self, transcript: list[str] | None):
        """Compile the graph of the supervisor and every worker agent."""
        builder = StateGraph(AgentState)
        agent_names = [a.name for a in self._agents]

        for agent in self._agents:
            builder.add_node(agent.name, self._make_agent_node(agent, transcript))

        builder.add_node("Supervisor", self._make_supervisor_node(agent_names, transcript))
        builder.add_edge(START, "Supervisor")

        for name in agent_names:
            builder.add_edge(name, "Supervisor")

        def route(state: AgentState):
            return "FINISH" if state["next_agent"] == "FINISH" else state["next_agent"]

        mapping = {name: name for name in agent_names}
        mapping["FINISH"] = END
        builder.add_conditional_edges("Supervisor", route, mapping)

        return builder.compile()

    async def _run_swarm(self, prompt: str, transcript: list[str] | None = None) -> str:
        """Run the swarm on ``prompt`` and return its final response; ``transcript`` collects what happened."""
        if not self._agents:
            return "Swarm Error: No agents available. Please use the spawn_agent tool to create at least one worker before querying."

        # A single agent needs no supervisor or routing.
        if len(self._agents) == 1:
            return await self._run_single_agent(prompt, transcript)

        graph = self._build_swarm_graph(transcript)

        print(f"\n{'=' * 50}\n[NEW REQUEST]: {prompt}\n{'=' * 50}")

        last_response = None
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": self.max_steps},
        ):
            for node_name, state_update in chunk.items():
                if node_name != "Supervisor" and "messages" in state_update:
                    msg = state_update["messages"][-1]
                    last_response = msg.content

        return (
            last_response
            if last_response is not None
            else "Swarm Error: No agent produced a response before routing finished."
        )