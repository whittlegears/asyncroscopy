"""FastMCP server exposing the Tango-free agent swarm in ``asyncroscopy.mcp.agent``.

The swarm reaches the instruments as an MCP *client* of one or more MCP servers,
so this process needs no pyTango. Clients (notebooks, chat UIs) reach the swarm
here over MCP, in the same way they would reach any other tool server.
"""

import argparse
import asyncio
import json
import socket
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from asyncroscopy.mcp.agent import DEFAULT_MCP_URL, DEFAULT_MAX_STEPS, DEFAULT_MODEL, Agent, AgentSwarm


def check_port_free(host: str, port: int) -> str | None:
    """Return an error string if (host, port) cannot be bound, else None."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        return str(exc)
    finally:
        probe.close()
    return None


class AgentMCPServer:
    """Serves an `AgentSwarm` as MCP tools over a FastMCP transport."""

    def __init__(
        self,
        name: str,
        mcp_urls: list[str],
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
            name (str): Display name for this MCP server instance.
            mcp_urls (list[str]): Streamable-HTTP MCP endpoints the swarm inherits tools from.
            model (str): Model name, interpreted by ``model_provider``.
            model_provider (str): Provider passed to LangChain's ``init_chat_model``.
            max_steps (int): LangGraph recursion limit per query.
            startup_agents (list[Agent], optional): Agents to preload at startup.
            use_init_chat_model (bool): Use ``init_chat_model`` instead of ChatOllama.
            verbose (bool): If True, print model and MCP connection progress. Defaults to True.
            training_log_dir (str, optional): Record every prompt and output here as
                JSON-lines training data. None disables it.
        """
        self.verbose = verbose
        self.mcp_urls = list(mcp_urls)
        self.swarm = AgentSwarm(
            model=model,
            model_provider=model_provider,
            max_steps=max_steps,
            startup_agents=startup_agents,
            use_init_chat_model=use_init_chat_model,
            verbose=verbose,
            training_log_dir=training_log_dir,
        )

        # Swarm must be initialized inside the server's own event loop
        @asynccontextmanager
        async def lifespan(server: FastMCP):
            await self.swarm.start(self.mcp_urls)
            yield {}

        self.mcp = FastMCP(name, lifespan=lifespan)
        self._query_lock = asyncio.Lock()
        self._workflows = self._make_workflow_runner()

    def _make_workflow_runner(self):
        """Graphs from asyncroscopy.agent_workflows, built from the swarm's MCP tools when first used."""
        try:
            import importlib

            from asyncroscopy.agent_workflows import WORKFLOW_MODULES
            from asyncroscopy.agent_workflows.runner import WorkflowRunner

            modules = {name: importlib.import_module(f"asyncroscopy.agent_workflows.{name}") for name in WORKFLOW_MODULES}
        except ImportError:  # LangGraph is missing (or stubbed in tests)
            return None
        self._workflow_modules = modules
        runner = WorkflowRunner()
        for name, module in modules.items():
            runner.register(name, lambda checkpointer, module=module: module.build_graph(self.swarm._tools, checkpointer=checkpointer))
        return runner

    def setup(self) -> None:
        """Register the swarm tools on the FastMCP instance."""
        for tool_func in (
            self.query_agent,
            self.spawn_agent,
            self.list_agents,
            self.list_agent_tools,
            self.complete,
            self.set_max_steps,
            *(
                (self.list_workflows, self.start_workflow, self.resume_workflow, self.workflow_status)
                if self._workflows is not None
                else ()
            ),
        ):
            self.mcp.add_tool(tool_func)
            if self.verbose:
                print(f"Registered native tool: {tool_func.__name__}")

        # Printed unconditionally (unlike the verbose lines above) so startup GUIs
        # can parse the count from stdout even when quiet mode is on.
        count = 6 + (4 if self._workflows is not None else 0)
        print(f"MCP ready: {count} tool(s) registered", flush=True)

        if self.verbose:
            print("\nAvailable agent tools:")
            print("  - query_agent(prompt: str) -> str")
            print("  - spawn_agent(config: str) -> str")
            print("  - list_agents() -> list[str]")
            print("  - list_agent_tools() -> str")
            print("  - complete(request_json: str) -> str")
            print("  - set_max_steps(max_steps: int) -> int")
            print(f"\nInheriting tools from MCP servers: {', '.join(self.mcp_urls)}")

    async def query_agent(self, prompt: str, include_transcript: bool = True) -> str:
        """
        Ask the agent swarm to carry out a request and return its response.

        If `include_transcript` is true (default), the routing decisions, 
        tool calls, and generated text will also be included above the final answer.
        """
        async with self._query_lock:
            return await self.swarm.query(prompt, include_transcript=include_transcript)

    def spawn_agent(self, config: str) -> str:
        """
        Add a worker agent to the swarm, returning its name or an empty string on failure.

        Args:
            config (str): JSON with 'name', 'system_prompt', optional 'tools' (glob
                patterns such as ["*spectrum"]), optional 'model' and 'description'.
        """
        return self.swarm.spawn_agent(config)

    def list_agents(self) -> list[str]:
        """List the names of all agents currently in the swarm."""
        return self.swarm.agents

    def list_agent_tools(self) -> str:
        """Return a JSON list of the MCP tool names the swarm inherited, e.g. [{"name": "..."}]."""
        return self.swarm.tools

    async def complete(self, request_json: str) -> str:
        """OpenAI-compatible single-step chat completion: {"messages": [...], "tools": [...]}.

        Unlike query_agent this does not run the swarm or execute tools; it returns
        the model's raw decision so the caller can drive the conversation itself.
        """
        return await self.swarm.complete(request_json)

    def list_workflows(self) -> str:
        """Workflows (LangGraph graphs) this server can run, with the tools each still lacks, as JSON.

        A workflow runs its own loop; the model only starts it and answers its questions.
        """
        return json.dumps([
            {**module.WORKFLOW, "missing_tools": module.missing_tools(self.swarm._tools)}
            for module in self._workflow_modules.values()
        ])

    async def start_workflow(self, name: str, inputs_json: str) -> str:
        """Start a workflow. Returns JSON with thread_id, waiting_for (the question it is paused on, or null),
        finished and state. Answer a pause with resume_workflow."""
        return json.dumps(await self._workflows.start(name, json.loads(inputs_json)), default=str)

    async def resume_workflow(self, thread_id: str, answer_json: str) -> str:
        """Answer the question a paused workflow is waiting on, e.g. {"approved": true}; returns the same JSON as start_workflow."""
        return json.dumps(await self._workflows.resume(thread_id, json.loads(answer_json)), default=str)

    async def workflow_status(self, thread_id: str) -> str:
        """Where a workflow run is: what it waits for, whether it finished, and its state, as JSON."""
        return json.dumps(await self._workflows.status(thread_id), default=str)

    def set_max_steps(self, max_steps: int) -> int:
        """Set the LangGraph recursion limit used per query, returning the new value."""
        return self.swarm.set_max_steps(max_steps)

    def start(self, transport: str | None = None, **kwargs) -> None:
        """
        Register the swarm tools and begin serving the MCP protocol.

        The swarm itself is initialized by this instance's FastMCP lifespan, so
        that it runs on the same event loop that serves tool calls.

        Args:
            transport: Transport protocol ("stdio", "http", "sse", or "streamable-http").
                Defaults to None, which uses stdio for local piping to agents.
            **kwargs: Additional keyword arguments to pass to the MCP server.
        """
        self.setup()
        self.mcp.run(transport=transport, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--transport", required=True)
    parser.add_argument("--http-host", required=True)
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--mcp-urls-json", default=json.dumps([DEFAULT_MCP_URL]))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-provider", default="ollama")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--startup-agents-json", default="[]")
    parser.add_argument("--use-init-chat-model", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--training-log-dir",
        default=None,
        help="Record every prompt and output here as JSONL training data (off when not given).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.transport == "streamable-http":
        bind_error = check_port_free(args.http_host, args.http_port)
        if bind_error:
            print(
                f"MCP ERROR: http://{args.http_host}:{args.http_port} is already in use - "
                f"another agent MCP server is likely still running with a different config. "
                f"Stop it or choose a different port. ({bind_error})",
                flush=True,
            )
            return 1

    server = AgentMCPServer(
        name=args.name,
        mcp_urls=json.loads(args.mcp_urls_json),
        model=args.model,
        model_provider=args.model_provider,
        max_steps=args.max_steps,
        startup_agents=[Agent(**agent) for agent in json.loads(args.startup_agents_json)],
        use_init_chat_model=args.use_init_chat_model,
        verbose=not args.quiet,
        training_log_dir=args.training_log_dir,
    )
    if args.transport == "streamable-http":
        print(
            f"Starting {args.name} at http://{args.http_host}:{args.http_port}/mcp "
            f"with tools inherited from {', '.join(server.mcp_urls)}",
            flush=True,
        )
        server.start(transport="streamable-http", host=args.http_host, port=args.http_port)
    else:
        server.start(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())