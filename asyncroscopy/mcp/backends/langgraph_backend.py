"""LangGraph supervisor-swarm backend, extracted from the LLM Tango device."""

import operator
from typing import Annotated, Sequence, TypedDict

import sys
import asyncio
import time
import subprocess
import os
import shutil
from pathlib import Path

import urllib.request
import urllib.error

import fnmatch
import json
import re

from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import END, START, StateGraph

from .base import Agent, AgentBackend, BackendConfig


class AgentState(TypedDict):
    """State dictionary for each Agent node in the swarm graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    current_task: str


def _clean_prop(val: str | None) -> str:
    if not val or str(val).strip().lower() in ("", "none", "null"):
        return ""
    return str(val).strip()


async def ensure_ollama_running(host: str, local_model_path: str = "", timeout: int = 10) -> None:
    """Check if Ollama server is running, offloaded to prevent blocking the Tango loop."""

    def _sync_check():
        tags_url = f"{host.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(tags_url, timeout=1):
                return
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
            pass

        ollama_bin = shutil.which("ollama")
        if not ollama_bin and sys.platform == "win32":
            candidates = [
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
                os.path.join(os.environ.get("ProgramFiles", ""), "Ollama", "ollama.exe"),
                r"C:\Users\Public\Ollama\ollama.exe",
            ]
            for cand in candidates:
                if cand and os.path.isfile(cand):
                    ollama_bin = cand
                    break

        if not ollama_bin:
            raise RuntimeError(
                "Ollama binary not found on PATH or standard installation paths. "
                "Please install Ollama (e.g. via 'winget install Ollama.Ollama' or from https://ollama.com) "
                "or set model_provider to a cloud provider like 'openai' or 'anthropic' in your config."
            )

        env = os.environ.copy()
        if local_model_path:
            model_path = Path(local_model_path)
            if (model_path / "manifests").exists():
                env["OLLAMA_MODELS"] = str(model_path)
            elif (model_path.parent / "manifests").exists():
                env["OLLAMA_MODELS"] = str(model_path.parent)

        try:
            subprocess.Popen(
                [ollama_bin, "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to start Ollama server using '{ollama_bin}': {exc}") from exc

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with urllib.request.urlopen(tags_url, timeout=1):
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
                time.sleep(0.5)

        raise RuntimeError(f"Ollama endpoint '{tags_url}' did not respond.")

    await asyncio.to_thread(_sync_check)


class LangGraphBackend(AgentBackend):
    """The original supervisor swarm: LangChain model + LangGraph routing + MCP tools."""

    name = "langgraph"
    supports_complete = True
    supports_connect_mcp = True

    def __init__(self, config: BackendConfig, agents: list[Agent]):
        super().__init__(config, agents)
        self._model = None
        self._tools: list[BaseTool] = []
        self._mcp_clients: list[MultiServerMCPClient] = []

    async def initialize(self) -> None:
        model_name_clean = _clean_prop(self.config.model_name)
        ollama_model_clean = _clean_prop(self.config.ollama_model)
        active_model = model_name_clean or ollama_model_clean or "gemma4:31b"
        provider = (_clean_prop(self.config.model_provider) or "ollama").lower()

        if provider == "ollama":
            host = self.config.ollama_host or "http://localhost:11434"
            await ensure_ollama_running(host=host, local_model_path=self.config.local_model_path)

        if provider == "ollama" and not self.config.use_init_chat_model:
            host = self.config.ollama_host or "http://localhost:11434"
            from langchain_ollama import ChatOllama
            print(f"[SYSTEM]: Initializing via ChatOllama with model '{active_model}'")
            self._model = ChatOllama(
                model=active_model,
                base_url=host,
                temperature=0,
                reasoning=False,
            )
        else:
            print(f"[SYSTEM]: Initializing via init_chat_model for provider '{provider}' with model '{active_model}'")
            init_kwargs = {
                "model": active_model,
                "model_provider": provider,
                "temperature": 0,
            }
            if self.config.api_key:
                init_kwargs["api_key"] = self.config.api_key
            self._model = init_chat_model(**init_kwargs)

        print("\n[SYSTEM]: Pre-warming model into VRAM (Cold Start)...")
        sys.stdout.flush()
        start_warmup = time.time()
        await self._model.ainvoke([HumanMessage(content=" ")])
        print(f"[SYSTEM]: Model pre-warmed in {time.time() - start_warmup:.2f}s!")

    async def query(self, prompt: str, max_steps: int) -> str:
        return await self._run_swarm(prompt, max_steps)

    async def complete(self, request: dict) -> dict:
        messages = self._openai_messages_to_langchain(request.get("messages") or [])
        tools = request.get("tools") or []
        model = self._model.bind_tools(tools) if tools else self._model
        response = await model.ainvoke(messages)
        return self._langchain_message_to_openai(response)

    async def connect_mcp(self, url: str, transport: str) -> int:
        server_id = f"server_{len(self._mcp_clients)}"
        client = MultiServerMCPClient({server_id: {"url": url, "transport": transport}})

        print(f"\n[SYSTEM]: Connecting to MCP Server at {url}...")

        tools = await client.get_tools()

        self._mcp_clients.append(client)
        self._tools.extend(tools)
        print(f"[SYSTEM]: Connected. Inherited {len(tools)} tools.")
        return len(tools)

    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    def set_skills_service(self, service) -> None:
        """Wrap the skill store as find_skills / load_skill tools every agent can use."""
        super().set_skills_service(service)
        self._tools = [t for t in self._tools if t.name not in ("find_skills", "load_skill")]

        def find_skills(query: str, k: int = 5) -> str:
            """Search the operator's skill library by meaning and keyword; returns ranked skill ids."""
            try:
                return json.dumps(service.find_skills(query, k))
            except Exception as exc:
                return f"Skill search is unavailable: {exc}"

        def load_skill(skill_id: str) -> str:
            """Return the full text of one skill by its id."""
            try:
                return service.load_skill(skill_id)
            except KeyError as exc:
                return str(exc)

        self._tools.append(StructuredTool.from_function(find_skills))
        self._tools.append(StructuredTool.from_function(load_skill))

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
        """Returns a list of filtered tools based on the allowed glob patterns."""
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

    async def _run_swarm(self, prompt: str, max_steps: int) -> str:
        """Run the agent swarm with a given prompt, returning the final response."""
        if not self.agents:
            return "Swarm Error: No agents available. Please use the SpawnAgent command to create at least one worker before querying."

        if len(self.agents) == 1:
            agent = self.agents[0]
            agent_executor = self._build_agent_executor(agent)
            print(f"\n[{agent.name}] is working...")
            return await self._stream_agent(
                agent_executor, [HumanMessage(content=prompt)], agent_label=agent.name
            )

        builder = StateGraph(AgentState)
        agent_names = [a.name for a in self.agents]
        options = agent_names + ["FINISH"]

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

        for agent in self.agents:
            builder.add_node(agent.name, create_agent_node(agent))

        agent_roster = "\n".join(
            f"- {a.name}: {a.description or a.system_prompt}" for a in self.agents
        )

        async def supervisor_node(state: AgentState):
            print("\n[Supervisor] Evaluating routing...")

            has_delegated = len(state["messages"]) > 1

            instructions = (
                f"Below are the available agents and what each is for:\n{agent_roster}\n\n"
                "Based on the conversation, decide which agent should act next to progress the user's request. "
                "Only output FINISH if the user's request has been fully and concretely answered — "
                "not if an agent asked a question, refused, said it lacks the ability, or otherwise failed to "
                "complete the task; in that case, route to a different, more suitable agent instead."
            )

            if not has_delegated:
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

        print(f"\n{'='*50}\n[NEW REQUEST]: {prompt}\n{'=' * 50}")

        last_response = None
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": max_steps}
        ):
            for node_name, state_update in chunk.items():
                if node_name != "Supervisor" and "messages" in state_update:
                    msg = state_update["messages"][-1]
                    last_response = msg.content

        return last_response if last_response is not None else "Swarm Error: No agent produced a response before routing finished."
