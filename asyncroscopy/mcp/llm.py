"""Tango device wrapping an Ollama/LangChain AI agent that connects to an MCP server."""

import operator
from typing import Annotated, Sequence, TypedDict
from dataclasses import dataclass

import sys
import asyncio
import time
import subprocess

import urllib.request
import urllib.error

import fnmatch
import json
import re

import tango
from tango.server import Device, attribute, command, device_property

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import BaseTool
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from langgraph.graph import END, START, StateGraph
except ImportError:
    print("Missing dependencies! Please run:")
    print("uv sync --extra agent")
    sys.exit(1)


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


class LLM(Device):
    mcp_url = device_property(dtype=str, default_value="http://127.0.0.1:8000/mcp")
    startup_agents = device_property(dtype=(str,), default_value=())
    ollama_model = device_property(dtype=str, default_value="gemma4:31b")
    use_init_chat_model = device_property(dtype=bool, default_value=False)
    model_provider = device_property(dtype=str, default_value="ollama")

    max_steps = attribute(label="Max Steps", dtype=int, access=tango.AttrWriteType.READ_WRITE)
    last_metrics = attribute(dtype=str)

    green_mode = tango.GreenMode.Asyncio

    async def init_device(self) -> None:
        await Device.init_device(self)
        self.set_state(tango.DevState.INIT)
        self._max_steps = 10
        self._run_metrics: dict = {}

        # Registries
        self._agents: list[Agent] = []
        self._tools: list[BaseTool] = []
        self._mcp_clients: list[MultiServerMCPClient] = []

        if self.startup_agents:
            self._agents = [Agent(**json.loads(agent_json)) for agent_json in self.startup_agents]
            print(f"[SYSTEM]: Loaded startup agents: {self._agents}")

        try:
            if not self.use_init_chat_model or self.model_provider == "ollama":
                await self.ensure_ollama_running()

            if self.use_init_chat_model: # Initialize from most model providers (e.g., OpenAI)
                self.info_stream("Initializing via init_chat_model")
                self._model = init_chat_model(
                    model=self.ollama_model,
                    model_provider=self.model_provider,
                    temperature=0
                )
            else: # Initialize locally via Ollama
                from langchain_ollama import ChatOllama
                self.info_stream("Initializing via ChatOllama")
                self._model = ChatOllama(
                    model=self.ollama_model,
                    temperature=0,
                    reasoning=False,
                )

            print("\n[SYSTEM]: Pre-warming model into VRAM (Cold Start)...")
            sys.stdout.flush()
            start_warmup = time.time()
            await self._model.ainvoke([HumanMessage(content=" ")])
            print(f"[SYSTEM]: Model pre-warmed in {time.time() - start_warmup:.2f}s!")

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

    def read_max_steps(self) -> int:
        return self._max_steps

    def write_max_steps(self, value: int) -> None:
        if value < 1:
            raise ValueError("max_steps must be at least 1.")
        self._max_steps = value

    def read_last_metrics(self) -> str:
        """JSON metrics of the last QueryJSON: timing and tokens per model call, outcome per tool call."""
        return json.dumps(self._run_metrics, default=str)

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

    async def ensure_ollama_running(self, host: str = "http://localhost:11434", timeout: int = 10) -> None:
        """Check if Ollama server is running, offloaded to prevent blocking the Tango loop."""
        
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
                    start_new_session=True
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
        doc_in="JSON {'prompt', 'query_id', 'temperature', 'seed', 'sequential', 'max_steps'}; all but prompt optional",
        dtype_out=str,
        doc_out="JSON {'query_id', 'answer', 'error'}: error is null on success, answer is null on failure",
    )
    async def QueryJSON(self, request_json: str) -> str:
        """Query with per-call options; failures come back in 'error', never as an answer.

        temperature and seed apply to this call only. sequential=True needs exactly one agent,
        runs its tool calls one at a time in order, records last_metrics, and caps model calls
        at max_steps. Otherwise the swarm runs and max_steps is LangGraph's recursion limit.
        """
        query_id = None
        base_model, base_steps = self._model, self._max_steps
        self._run_metrics = {"query_id": None, "started": time.time(), "model_steps": [], "tool_steps": []}
        self.set_state(tango.DevState.RUNNING)
        t0 = time.perf_counter()
        try:
            try:
                request = json.loads(request_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON: {e}") from e
            if not isinstance(request, dict):
                raise ValueError("the request must be a JSON object")
            query_id = request.get("query_id")
            max_steps = int(request["max_steps"]) if request.get("max_steps") is not None else self._max_steps
            if max_steps < 1:
                raise ValueError("max_steps must be at least 1")
            sequential = bool(request.get("sequential"))
            self._run_metrics.update({
                "query_id": query_id,
                "options": {k: request.get(k) for k in ("temperature", "seed", "sequential", "max_steps")},
                "path": "sequential" if sequential else "swarm",
            })
            if not self._agents:
                raise RuntimeError("no agents available; use SpawnAgent to create one first")
            if sequential and len(self._agents) != 1:
                raise RuntimeError(f"sequential needs exactly one agent, but {len(self._agents)} are spawned")
            sampling = {k: request[k] for k in ("temperature", "seed") if request.get(k) is not None}
            if sampling:
                self._model = base_model.model_copy(update=sampling)
            prompt = str(request.get("prompt", ""))
            if sequential:
                answer = await self._run_sequential_agent(self._agents[0], prompt, max_steps)
            else:
                self._max_steps = max_steps
                answer = await self._run_swarm(prompt)
            self._run_metrics["answer"] = answer
            return json.dumps({"query_id": query_id, "answer": answer, "error": None})
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self._run_metrics["error"] = error
            return json.dumps({"query_id": query_id, "answer": None, "error": error})
        finally:
            self._model, self._max_steps = base_model, base_steps
            self._run_metrics["total_s"] = time.perf_counter() - t0
            self.set_state(tango.DevState.ON)

    @command(
        dtype_in=str,
        doc_in="OpenAI-style {'messages': [...], 'tools': [...]}",
        dtype_out=str,
        doc_out="JSON {'message': {...}} on success, or {'error': {'message': ...}} on failure",
    )
    async def Complete(self, request_json: str) -> str:
        """OpenAI-compatible single-step chat completion for the llm_bridge.py HTTP bridge.

        Unlike Query, this does not run the LangGraph swarm or execute any tools
        itself — it converts the request into one LangChain model call and returns
        the model's raw decision (tool_calls or final text) so the caller (e.g.
        SciAgentGUI's own agent loop) can execute tools and drive the conversation.
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

            server_id = f"server_{len(self._mcp_clients)}"
            client = MultiServerMCPClient({server_id: {"url": url, "transport": transport}})

            print(f"\n[SYSTEM]: Connecting to MCP Server at {url}...")

            tools = await client.get_tools()

            self._mcp_clients.append(client)
            self._tools.extend(tools)
            print(f"[SYSTEM]: Connected. Inherited {len(tools)} tools.")
        except Exception as e:
            self.error_stream(f"Failed to connect to MCP server: {e}")
            return False

        return True

    @command(dtype_out=bool, doc_out="Success status")
    async def DisconnectMCP(self) -> bool:
        """Drop all MCP connections and their tools."""
        self._mcp_clients = []
        self._tools = []
        print("[SYSTEM]: Disconnected from all MCP servers.")
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

    async def _stream_agent(self, agent_executor, messages, agent_label: str = "") -> str:
        """Run a create_agent executor while streaming tokens and tool calls to stdout."""
        prefix = f"[{agent_label}] " if agent_label else ""
        start_time = time.time()
        first_token_received = False
        final_content = ""

        async for event in agent_executor.astream_events({"messages": messages}, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_start":
                # A new generation round is starting (could be a tool-call round or the final answer)
                start_time = time.time()
                first_token_received = False

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if not first_token_received:
                    ttft = time.time() - start_time
                    print(f"\n{prefix}[DIAGNOSTIC]: Time to first token: {ttft:.2f}s")
                    print(f"{prefix}[GENERATION]: ", end="")
                    first_token_received = True
                if chunk.content:
                    print(chunk.content, end="")
                    sys.stdout.flush()

            elif kind == "on_chat_model_end":
                output = event["data"]["output"]
                tool_calls = getattr(output, "tool_calls", None) or []
                if not tool_calls:
                    # This round produced no tool calls, so it's the final answer
                    final_content = (output.content or "").strip()
                print()

            elif kind == "on_tool_start":
                tool_name = event["name"]
                tool_input = event["data"].get("input")
                print(f"{prefix}[EXECUTING TOOL]: {tool_name}({tool_input})")

            elif kind == "on_tool_end":
                output = event["data"].get("output")
                print(f"{prefix}[TOOL RESULT]: {output}")

        if final_content:
            print(f"{prefix}[FINAL ANSWER RETURNED]:\n{final_content}\n{'=' * 50}")
        return final_content

    async def _run_sequential_agent(self, agent: Agent, prompt: str, max_steps: int) -> str:
        """Single-agent loop that runs tool calls one at a time, in issued order.

        LangGraph's ToolNode runs a step's calls concurrently, so a move and an acquire race.
        """
        tools = {t.name: t for t in self._get_agent_tools(agent.tools)}
        model = self._model.bind_tools(list(tools.values())) if tools else self._model
        messages: list[BaseMessage] = [SystemMessage(content=agent.system_prompt), HumanMessage(content=prompt)]
        for step in range(max_steps):
            started, t0 = time.time(), time.perf_counter()
            ttft, response = None, None
            async for chunk in model.astream(messages):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                response = chunk if response is None else response + chunk
            if response is None:
                raise RuntimeError("the model returned an empty response")
            calls = list(getattr(response, "tool_calls", None) or [])
            usage = getattr(response, "usage_metadata", None) or {}
            self._run_metrics.setdefault("model_steps", []).append({
                "agent": agent.name, "started": started, "duration_s": time.perf_counter() - t0, "ttft_s": ttft,
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "tool_calls": [call.get("name") for call in calls],
            })
            messages.append(AIMessage(content=response.content or "", tool_calls=calls))
            if not calls:
                return self._tool_output_text(response).strip()
            for index, call in enumerate(calls):
                messages.append(await self._run_one_tool(tools, call, step, index, agent.name))
        raise RuntimeError(f"step limit: the model was still calling tools after {max_steps} model calls")

    async def _run_one_tool(self, tools: dict, call: dict, step: int, index: int, agent_label: str):
        """Run one tool call; failures become an error ToolMessage, not an exception."""
        name, args, call_id = call.get("name", ""), call.get("args") or {}, call.get("id") or f"call_{step}_{index}"
        started, t0 = time.time(), time.perf_counter()
        print(f"[{agent_label}] [EXECUTING TOOL]: {name}({args})")
        tool, is_error = tools.get(name), False
        if tool is None:
            message, is_error = ToolMessage(content=f"Error: no tool named {name!r} is available.", tool_call_id=call_id), True
        else:
            try:
                message = await tool.ainvoke({"name": name, "args": args, "id": call_id, "type": "tool_call"})
                if not isinstance(message, ToolMessage):
                    message = ToolMessage(content=str(message), tool_call_id=call_id)
                is_error = getattr(message, "status", None) == "error"
            except Exception as exc:
                message, is_error = ToolMessage(content=f"Error: {type(exc).__name__}: {exc}", tool_call_id=call_id), True
        if is_error:
            message.status = "error"
        text = self._tool_output_text(message)
        print(f"[{agent_label}] [TOOL RESULT]: {text[:500]}")
        self._run_metrics.setdefault("tool_steps", []).append({
            "agent": agent_label, "tool": name, "arguments": args, "model_step": step, "call_index": index,
            "started": started, "duration_s": time.perf_counter() - t0, "ok": not is_error, "output": text[:2000],
        })
        return message

    @staticmethod
    def _tool_output_text(message) -> str:
        """Text of a message whose content is a string or a list of blocks."""
        content = getattr(message, "content", message)
        if isinstance(content, list):
            return "\n".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
                if not isinstance(block, dict) or block.get("type", "text") == "text"
            )
        return "" if content is None else str(content)

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

        builder = StateGraph(AgentState)
        agent_names = [a.name for a in self._agents]
        options = agent_names + ["FINISH"]

        # Creates a ReAct sub-graph for each Agent
        def create_agent_node(agent: Agent):
            agent_executor = self._build_agent_executor(agent)

            async def node(state: AgentState):
                task = state.get("current_task", "Execute assigned tool.")
                print(f"\n[{agent.name}] assigned task: '{task}'")

                content = await self._stream_agent(agent_executor, [HumanMessage(content=task)], agent_label=agent.name)
                print(f"[{agent.name}] finished.\n")
                return {
                    "messages": [
                        HumanMessage(content=f"[{agent.name}]: {content}", name=agent.name)
                    ]
                }
            return node

        # Register workers
        for agent in self._agents:
            builder.add_node(agent.name, create_agent_node(agent))

        agent_roster = "\n".join(
            f"- {a.name}: {a.description or a.system_prompt}" for a in self._agents
        )

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
            response = await self._model.ainvoke([sys_prompt] + state["messages"])
            next_agent, subtask = self._parse_routing_decision(response.content, valid_options, fallback)
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
        print(f"\n{'='*50}\n[NEW REQUEST]: {prompt}\n{'=' * 50}")

        last_response = None
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": self._max_steps}
        ):
            for node_name, state_update in chunk.items():
                if node_name != "Supervisor" and "messages" in state_update:
                    msg = state_update["messages"][-1]
                    last_response = msg.content

        return last_response if last_response is not None else "Swarm Error: No agent produced a response before routing finished."

# ----------------------------------------------------------------------
# Server entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    LLM.run_server()