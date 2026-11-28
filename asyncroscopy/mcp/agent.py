"""Ollama/LangChain agent swarm that inherits its tools from MCP servers."""

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

try:
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from langgraph.graph import END, START, StateGraph
except ImportError:
    print("Missing dependencies! Please run:")
    print("uv sync --extra agent --extra ollama")
    sys.exit(1)

try:
    from langchain.agents.middleware import wrap_tool_call
except ImportError:  # the test stubs of the LangChain stack have no middleware
    wrap_tool_call = None

from asyncroscopy.training.recorder import TrainingRecorder, to_openai_message


def _tool_error_middleware():
    """Return a failed tool call to the model as an error message instead of ending the run.

    Without it, create_agent re-raises a tool's exception (a Tango DevFailed arrives
    as a ToolException) and the whole query stops, so the model never sees the
    error or the bridge's "this failed N times, stop" hint. There is no retry: an
    instrument command such as move_stage must not run twice because it timed out.
    """
    if wrap_tool_call is None:
        return []

    @wrap_tool_call
    async def report_tool_errors(request, handler):
        try:
            return await handler(request)
        except Exception as exc:
            name = request.tool_call.get("name", "tool")
            return ToolMessage(content=f"{name} failed: {exc}", tool_call_id=request.tool_call["id"], status="error")

    return [report_tool_errors]

DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_MODEL = "gemma4:31b"
DEFAULT_MAX_STEPS = 10


@dataclass
class Agent:
    """Represents a single AI agent in the swarm."""

    name: str
    system_prompt: str
    tools: list[str]  # List of tool names, supporting glob patterns (e.g., ["math_*", "read_file"])
    model: str | None = None
    description: str = ""


class AgentState(TypedDict):
    """State dictionary for each Agent node in the swarm graph."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    current_task: str


class AgentSwarm:
    """A supervisor plus N ReAct worker agents, bound to tools inherited over MCP."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        model_provider: str = "ollama",
        max_steps: int = DEFAULT_MAX_STEPS,
        startup_agents: list[Agent] | None = None,
        use_init_chat_model: bool = False,
        verbose: bool = True,
        training_log_dir: str | None = None,
    ):
        """
        Args:
            model (str): Model name, interpreted by ``model_provider``. Defaults to "gemma4:31b".
            model_provider (str): Provider for ``init_chat_model`` when
                ``use_init_chat_model`` is set; "ollama" otherwise. Defaults to "ollama".
            max_steps (int): LangGraph recursion limit for a single query. Defaults to 10.
            startup_agents (list[Agent], optional): Agents to preload at startup.
            use_init_chat_model (bool): Initialize the model through LangChain's
                ``init_chat_model`` (OpenAI, Anthropic, ...) instead of ChatOllama.
            verbose (bool): If True, print model and MCP connection progress. Defaults to True.
            training_log_dir (str, optional): Directory where every prompt and output is
                recorded as JSON-lines training data. None disables it.
        """
        self.model = model
        self.model_provider = model_provider
        self.max_steps = int(max_steps)
        self.use_init_chat_model = use_init_chat_model
        self.verbose = verbose

        self._agents: list[Agent] = list(startup_agents or [])
        self._tools: list[BaseTool] = []
        self._mcp_clients: list[MultiServerMCPClient] = []
        self._model = None
        self.startup_error: str | None = None
        self._recorder: TrainingRecorder | None = (
            TrainingRecorder(training_log_dir, model=model) if training_log_dir else None
        )
        self._tool_schemas: dict[str, list[dict]] = {}  # agent name -> tools it is offered, for the recorder

    # ----------------------------------------------------------------------
    # Startup
    # ----------------------------------------------------------------------

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

        try:
            if not self.use_init_chat_model or self.model_provider == "ollama":
                await self.ensure_ollama_running()

            if self.use_init_chat_model:
                # Initialize from most model providers (e.g., OpenAI)
                self._log("Initializing via init_chat_model")
                self._model = init_chat_model(
                    model=self.model,
                    model_provider=self.model_provider,
                    temperature=0,
                )
            else:  # Initialize locally via Ollama
                from langchain_ollama import ChatOllama

                self._log("Initializing via ChatOllama")
                self._model = ChatOllama(
                    model=self.model,
                    temperature=0,
                    reasoning=False,
                )

            print("\n[SYSTEM]: Pre-warming model (Cold Start)...")
            sys.stdout.flush()
            start_warmup = time.time()
            await self._model.ainvoke([HumanMessage(content=" ")])
            print(f"[SYSTEM]: Model pre-warmed in {time.time() - start_warmup:.2f}s!")
        except Exception as e:
            self._model = None
            self.startup_error = f"{type(e).__name__}: {e}"
            self._error(f"[SYSTEM]: Failed to start the model: {self.startup_error}")
            self._error("[SYSTEM]: Serving MCP anyway; query_agent will report this error.")
            return False

        for url in mcp_urls or [DEFAULT_MCP_URL]:
            if not await self.connect_mcp(url):
                print(f"[SYSTEM]: Failed to connect to MCP Server at {url}.")

        return True

    def _log(self, message: str) -> None:
        """Print a progress line when verbose, otherwise stay silent."""
        if self.verbose:
            print(message)

    def _error(self, message: str) -> None:
        """Report a failure regardless of verbosity, since callers cannot see our state."""
        print(message, file=sys.stderr)

    # ----------------------------------------------------------------------
    # MCP connectivity
    # ----------------------------------------------------------------------

    async def connect_mcp(self, url: str, transport: str = "streamable_http") -> bool:
        """Connect to an MCP server and inherit its tools. Returns true for success."""
        try:
            server_id = f"server_{len(self._mcp_clients)}"
            client = MultiServerMCPClient({server_id: {"url": url, "transport": transport}})

            print(f"\n[SYSTEM]: Connecting to MCP Server at {url}...")

            tools = await client.get_tools()

            self._mcp_clients.append(client)
            self._tools.extend(tools)
            print(f"[SYSTEM]: Connected. Inherited {len(tools)} tools.")
        except Exception as e:
            self._error(f"Failed to connect to MCP server {url!r}: {e}")
            return False

        return True

    async def ensure_ollama_running(self, host: str = "http://localhost:11434", timeout: int = 10) -> None:
        """Check if Ollama server is running, offloaded to prevent blocking the event loop."""

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

        # Run the blocking network and sleep calls in a separate thread
        await asyncio.to_thread(_sync_check)

    # ----------------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Public commands
    # ----------------------------------------------------------------------

    async def query(self, prompt: str, include_transcript: bool = True) -> str:
        """
        Query the agent swarm with a prompt, returning the final response.

        If `include_transcript` is true (default), the routing decisions, 
        tool calls, and generated text are also included.
        """
        if not self.ready:
            return f"Agent Error: no model is available ({self.startup_error})."
        if self._recorder is not None:
            self._recorder.start_run(prompt)
        final, error = "", None
        try:
            transcript: list[str] | None = [] if include_transcript else None
            final = await self._run_swarm(prompt, transcript=transcript)
            if not transcript:
                return final
            return "\n".join([*transcript, "", "FINAL ANSWER:", final])
        except Exception as e:
            print(f"\n[CRITICAL ERROR]: {e}")
            error = str(e)
            # Keep what was done before the failure (for example a step limit hit
            # after several tool calls): the caller needs it to decide what next.
            if transcript:
                return "\n".join([*transcript, "", f"ERROR: {error}"])
            return error
        finally:
            # Also runs when the query is cancelled, so the run record is always written.
            if self._recorder is not None:
                self._recorder.end_run(final, error=error)

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
            if self._recorder is not None:
                self._recorder.record_call(request.get("messages") or [], response, agent="complete", tools=tools)
            return json.dumps({"message": to_openai_message(response)})
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

    # ----------------------------------------------------------------------
    # Message conversion
    # ----------------------------------------------------------------------

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

    # Kept as a name for existing callers; the recorder owns the conversion.
    _langchain_message_to_openai = staticmethod(to_openai_message)

    @staticmethod
    def _tool_schema(tool) -> dict:
        """A tool's OpenAI-style definition, as fine-tuning data expects."""
        try:
            from langchain_core.utils.function_calling import convert_to_openai_tool

            return convert_to_openai_tool(tool)
        except Exception:  # unusual tool objects: keep at least the name and description
            return {"type": "function", "function": {"name": tool.name, "description": getattr(tool, "description", "")}}

    # ----------------------------------------------------------------------
    # Swarm internals
    # ----------------------------------------------------------------------

    def _get_agent_tools(self, allowed_patterns: list[str]) -> list:
        """Returnes a list of filtered tools based on the allowed glob patterns."""
        if "*" in allowed_patterns:
            return self._tools
        return [t for t in self._tools if any(fnmatch.fnmatch(t.name, pat) for pat in allowed_patterns)]

    def _build_agent_executor(self, agent: Agent):
        """Filter this agent's tools and construct its ReAct executor."""
        agent_tools = self._get_agent_tools(agent.tools)
        print(f"[SYSTEM]: Binding {len(agent_tools)} tools to {agent.name}")
        if self._recorder is not None:
            self._tool_schemas[agent.name] = [self._tool_schema(tool) for tool in agent_tools]
        return create_agent(
            model=self._model, tools=agent_tools, system_prompt=agent.system_prompt, middleware=_tool_error_middleware()
        )

    def _extract_json(self, text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) if present."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    async def _structured_route(self, messages, valid_options: list[str]) -> tuple[str, str] | None:
        """Ask for the routing decision as structured output, constrained to the valid choices.

        Returns None when the model cannot do structured output or its answer is not
        a valid choice; the supervisor then parses free text as before.
        """
        if not getattr(self, "structured_routing", True) or not hasattr(self._model, "with_structured_output"):
            return None
        schema = {
            "title": "RouteDecision",
            "description": "Which agent acts next, and the sub-task for it alone.",
            "type": "object",
            "properties": {
                "next": {"type": "string", "enum": list(valid_options), "description": "The agent to act next, or FINISH."},
                "task": {"type": "string", "description": "The isolated sub-task for that agent; empty for FINISH."},
            },
            "required": ["next", "task"],
        }
        try:
            answer = await self._model.with_structured_output(schema).ainvoke(messages)
        except Exception as exc:  # providers without structured output, or a reply that does not parse
            print(f"[Supervisor] structured routing unavailable ({type(exc).__name__}); parsing text")
            return None
        if isinstance(answer, dict) and answer.get("next") in valid_options:
            return str(answer["next"]), str(answer.get("task", ""))
        return None

    def _parse_routing_decision(self, content: str, valid_options: list[str], fallback: str) -> tuple[str, str]:
        """Parse a supervisor response's {'next': ...} decision, falling back on any error or invalid value."""
        # Why the fallback was used, if it was; the supervisor puts it in the transcript.
        self._routing_note = ""
        try:
            decision = json.loads(self._extract_json(content))
            next_agent = decision.get("next", fallback)
            subtask = decision.get("task", "")
            if next_agent not in valid_options:
                self._routing_note = f"Supervisor chose {next_agent!r}, which is not one of {valid_options}; using {fallback!r}"
                return fallback, subtask
            return next_agent, subtask
        except Exception as e:
            print(f"[SUPERVISOR ERROR]: {e}")
            self._routing_note = f"Supervisor reply was not JSON ({str(content)[:80]!r}); using {fallback!r}"
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
        start_time = time.time()
        first_token_received = False
        final_content = ""
        generated = ""

        def record(line: str) -> None:
            if transcript is not None:
                transcript.append(line)

        config = {"recursion_limit": recursion_limit} if recursion_limit else None
        model_inputs: dict = {}  # event run_id -> messages the model was given
        tool_inputs: dict = {}  # event run_id -> tool arguments
        async for event in agent_executor.astream_events({"messages": messages}, config=config, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_start":
                # A new generation round is starting (could be a tool-call round or the final answer)
                start_time = time.time()
                first_token_received = False
                given = (event.get("data", {}).get("input") or {}).get("messages") or []
                # Chat-model events carry a batch of conversations; the agent sends one.
                model_inputs[event.get("run_id")] = given[0] if given and isinstance(given[0], list) else given

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if not first_token_received:
                    ttft = time.time() - start_time
                    print(f"\n{prefix}[DIAGNOSTIC]: Time to first token: {ttft:.2f}s")
                    first_token_received = True
                if chunk.content:
                    generated += chunk.content
                    print(chunk.content, end="")
                    sys.stdout.flush()

            elif kind == "on_chat_model_end":
                output = event["data"]["output"]
                if self._recorder is not None:
                    self._recorder.record_call(
                        model_inputs.pop(event.get("run_id"), None) or [],
                        output,
                        agent=agent_label,
                        tools=self._tool_schemas.get(agent_label),
                        latency_s=round(time.time() - start_time, 3),
                    )
                tool_calls = getattr(output, "tool_calls", None) or []
                if not tool_calls:
                    # This round produced no tool calls, so it's the final answer
                    final_content = (output.content or "").strip()
                print()

            elif kind == "on_tool_start":
                tool_name = event["name"]
                tool_input = event["data"].get("input")
                tool_inputs[event.get("run_id")] = tool_input
                line = f"{prefix}EXECUTING TOOL: {tool_name}({tool_input})"
                print(f"{prefix}[EXECUTING TOOL]: {tool_name}({tool_input})")
                record(line)

            elif kind == "on_tool_end":
                output = event["data"].get("output")
                if self._recorder is not None:
                    self._recorder.record_tool(
                        event.get("name", ""),
                        tool_inputs.pop(event.get("run_id"), event["data"].get("input")),
                        getattr(output, "content", output),
                        agent=agent_label,
                    )
                print(f"{prefix}[TOOL RESULT]: {output}")
                record(f"{prefix}TOOL RESULT: {output}")

        if generated.strip():
            record(f"{prefix}{generated.strip()}")

        if final_content:
            print(f"{prefix}[FINAL ANSWER RETURNED]:\n{final_content}\n{'=' * 50}")
        return final_content

    async def _run_swarm(self, prompt: str, transcript: list[str] | None = None) -> str:
        """
        Run the agent swarm with a given prompt, returning the final response.

        When `transcript` is supplied, it accumulates the run's routing decisions,
        tool calls, and generated text for the caller to display.
        """
        if not self._agents:
            return "Swarm Error: No agents available. Please use the spawn_agent tool to create at least one worker before querying."

        def record(line: str) -> None:
            if transcript is not None:
                transcript.append(line)

        # If there is a single agent, run it like a single agent (no need for supervisor/routing)
        if len(self._agents) == 1:
            agent = self._agents[0]
            agent_executor = self._build_agent_executor(agent)
            print(f"\n[{agent.name}] is working...")
            # max_steps bounds a lone agent too, not only the routed graph below.
            return await self._stream_agent(
                agent_executor, [HumanMessage(content=prompt)], agent_label=agent.name, transcript=transcript,
                recursion_limit=self.max_steps,
            )

        builder = StateGraph(AgentState)
        agent_names = [a.name for a in self._agents]
        options = agent_names + ["FINISH"]

        # Creates a ReAct sub-graph for each Agent
        def create_agent_node(agent: Agent):
            agent_executor = self._build_agent_executor(agent)

            async def node(state: AgentState):
                task = state.get("current_task", "Execute assigned tool.")
                print(f"\n[{agent.name}] assigned task: '{task}'")
                record(f"{agent.name} assigned task: {task}")

                # max_steps bounds each worker too; without it create_agent allows
                # thousands of steps inside one supervisor hop.
                content = await self._stream_agent(
                    agent_executor, [HumanMessage(content=task)], agent_label=agent.name, transcript=transcript,
                    recursion_limit=self.max_steps,
                )
                print(f"[{agent.name}] finished.\n")
                return {
                    "messages": [
                        # The worker's reply, not a user turn: the supervisor must not read
                        # it as a new instruction.
                        AIMessage(content=f"[{agent.name}]: {content}", name=agent.name)
                    ]
                }

            return node

        # Register workers
        for agent in self._agents:
            builder.add_node(agent.name, create_agent_node(agent))

        agent_roster = "\n".join(f"- {a.name}: {a.description or a.system_prompt}" for a in self._agents)

        async def supervisor_node(state: AgentState):
            print("\n[Supervisor] Evaluating routing...")

            # Check if agent has contributed if there's another AI/Human message beyond the original user prompt
            has_delegated = len(state["messages"]) > 1

            instructions = (
                f"Below are the available agents and what each is for:\n{agent_roster}\n\n"
                "Based on the conversation, decide which agent should act next to progress the user's request. "
                "Only output FINISH if the user's request has been fully and concretely answered — "
                "not if an agent asked a question, refused, said it lacks the ability, or otherwise failed to "
                "complete the task; in that case, route to a different, more suitable agent instead."
            )

            if not has_delegated:
                # First turn forces subagent routing; FINISH isn't a valid choice yet.
                valid_options, fallback = agent_names, agent_names[0]
            else:
                valid_options, fallback = options, "FINISH"

            sys_prompt = SystemMessage(
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
            decision = await self._structured_route([sys_prompt] + state["messages"], valid_options)
            if decision is not None:
                next_agent, subtask = decision
                self._routing_note = ""
                response = AIMessage(content=json.dumps({"next": next_agent, "task": subtask}))
            else:
                response = await self._model.ainvoke([sys_prompt] + state["messages"])
            if self._recorder is not None:
                self._recorder.record_call([sys_prompt] + list(state["messages"]), response, agent="Supervisor")
            if decision is None:
                next_agent, subtask = self._parse_routing_decision(response.content, valid_options, fallback)
            if getattr(self, "_routing_note", ""):
                record(self._routing_note)
            print(f"[Supervisor] Decision: {next_agent}")
            record(f"Supervisor -> {next_agent}: {subtask}" if next_agent != "FINISH" else "Supervisor -> FINISH")
            if next_agent == "FINISH":
                print("[Supervisor] Decision: FINISH\n")

            return {
                "next_agent": next_agent,
                "current_task": subtask,
            }

        builder.add_node("Supervisor", supervisor_node)
        builder.add_edge(START, "Supervisor")

        for name in agent_names:
            builder.add_edge(name, "Supervisor")

        def route(state: AgentState):
            return "FINISH" if state["next_agent"] == "FINISH" else state["next_agent"]

        mapping = {name: name for name in agent_names}
        mapping["FINISH"] = END
        builder.add_conditional_edges("Supervisor", route, mapping)

        graph = builder.compile()

        # Graph execution loop
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