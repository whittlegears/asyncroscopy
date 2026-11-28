"""FastMCP server exposing the Tango-free agent graphs: the swarm (``asyncroscopy.agents.graphs.swarm``)
and the named graphs (``asyncroscopy.agents.run``).

The agents reach the instruments as an MCP *client* of one or more MCP servers,
so this process needs no pyTango. Clients (notebooks, chat UIs) reach them
here over MCP, in the same way they would reach any other tool server.

It is the entry point that serves the agent layer, so it imports ``asyncroscopy.agents`` only
when it builds the server, never at import: the MCP package stays below the agents.

With ``--endpoint-role operator`` the same module serves the workflow operator endpoint instead: no swarm and no
model, only the tools a person uses to review, approve and reject workflows (``WorkflowServer.operator_tools``),
with ``endpoint_info`` and ``action_space`` saying so. It mirrors the instrument's operator endpoint
(``mcp_server --endpoint-role operator``): a separate process, bound to loopback because it has no login, that an
app calls for a person and never hands to a model. ``docs/MCP/workflow_approval.md`` is its contract.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from fastmcp import FastMCP

from asyncroscopy.core.names import AGENT_GRAPHS, DEFAULT_MCP_URL, endpoint_urls
from asyncroscopy.core.net import is_loopback_host
from asyncroscopy.mcp import host_guard

if TYPE_CHECKING:
    from asyncroscopy.agents.graphs.swarm import Agent

def with_images(answer: str, images: list[dict]):
    """The answer as text, or with images as the answer followed by its image blocks (the latest MAX_COLLECTED)."""
    if not images:
        return answer
    import base64

    from fastmcp.tools import ToolResult
    from fastmcp.utilities.types import Image

    from asyncroscopy.agents.mcp_tools import MAX_COLLECTED

    shown = images[-MAX_COLLECTED:]
    blocks = [Image(data=base64.b64decode(image["base64"]), format=image["mime_type"].split("/")[-1]) for image in shown]
    note = f"\n\n[{len(shown)} image(s) from {', '.join(dict.fromkeys(image['tool'] for image in shown))} follow]"
    return ToolResult(content=[answer + note, *blocks])


GraphName = Literal["", *AGENT_GRAPHS]
ENDPOINT_ROLES = ("agent", "operator")
OPERATOR_ACTIONS = (("review_workflow", "setting.read", "low", "observe"), ("approve_workflow", "other", "high", "act"),
                    ("reject_workflow", "other", "high", "act"))
workflow_services = {"list": "list_workflows", "get": "get_workflow", "run": "run_workflow", "status": "workflow_status",
                     "stop": "stop_workflow", "resume": "resume_workflow", "answer": "answer_workflow", "list_runs": "list_runs",
                     "list_blocks": "list_blocks", "propose": "propose_workflow"}


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
        model: str | None = None,
        model_provider: str = "ollama",
        max_steps: int | None = None,
        startup_agents: list[Agent] | None = None,
        use_init_chat_model: bool = False,
        verbose: bool = True,
        workflows: bool = True,
        workflows_hardware: bool = False,
        endpoint_role: str = "agent",
        urls: dict[str, str | None] | None = None,
        workflow_instruments: list[str] | None = None,
    ) -> None:
        """
        Args:
            name (str): Display name for this MCP server instance.
            mcp_urls (list[str]): Streamable-HTTP MCP endpoints the swarm inherits tools from.
            model (str): Model name, interpreted by ``model_provider``.
            model_provider (str): Provider passed to LangChain's ``init_chat_model``.
            max_steps (int): LangGraph recursion limit per query; None: the swarm's default.
            startup_agents (list[Agent], optional): Agents to preload at startup.
            use_init_chat_model (bool): Use ``init_chat_model`` instead of ChatOllama.
            verbose (bool): If True, print model and MCP connection progress. Defaults to True.
            workflows (bool): Run workflows in this process and offer their tools here and to the agents (default).
                This is the one process workflows run in. They drive the first of ``mcp_urls``.
                False: no workflow tools.
            workflows_hardware (bool): Also allow workflows on a stack that is not simulated.
            endpoint_role (str): "agent" (default) serves the swarm and the workflow tools. "operator" serves only
                the workflow review, approve and reject tools, for a person; no swarm and no model.
            urls (dict): The other endpoints endpoint_info reports (``asyncroscopy.core.names.endpoint_url_names``), from the config.
            workflow_instruments (list[str]): Other instrument endpoints, by URL, that workflows may run on and be
                reviewed for (``instrument_url``), such as the real instrument a twin in ``mcp_urls`` is paired with.
                No other URL is accepted.
        """
        if endpoint_role not in ENDPOINT_ROLES:
            raise ValueError(f"endpoint_role must be one of {ENDPOINT_ROLES}, got {endpoint_role!r}")
        self.verbose = verbose
        self.urls = endpoint_urls(urls)
        self.mcp_urls = list(mcp_urls)
        self.endpoint_role = endpoint_role
        if endpoint_role == "operator":
            from asyncroscopy.workflows.server import Endpoints, WorkflowServer

            self.swarm = None
            self.workflows = WorkflowServer(Endpoints(self.mcp_urls[0] if self.mcp_urls else None, caller="workflow-operator",
                                                      instruments=workflow_instruments), name=name)
            self.mcp = FastMCP(name)
            self._query_lock = asyncio.Lock()
            return

        from asyncroscopy.agents.graphs.swarm import AgentSwarm

        self.swarm = AgentSwarm(
            model=model,
            model_provider=model_provider,
            startup_agents=startup_agents,
            use_init_chat_model=use_init_chat_model,
            verbose=verbose,
            **({} if max_steps is None else {"max_steps": max_steps}),
        )

        self.workflows = None
        if workflows or workflows_hardware:
            from asyncroscopy.workflows.server import Endpoints, WorkflowServer

            self.workflows = WorkflowServer(Endpoints(self.mcp_urls[0] if self.mcp_urls else None, caller="agent-workflow",
                                                      instruments=workflow_instruments), hardware=workflows_hardware)

        # The swarm must be built inside the server's own event loop.
        @asynccontextmanager
        async def lifespan(server: FastMCP):
            await self.swarm.start(self.mcp_urls)
            if self.workflows is not None:
                self.swarm.add_tools(self.workflow_tools())
            yield {}

        self.mcp = FastMCP(name, lifespan=lifespan)
        self._query_lock = asyncio.Lock()

    def workflow_tools(self) -> list:
        """The workflow tools as LangChain tools for the agents. An agent selects them with ``*workflow*``."""
        from langchain_core.tools import StructuredTool

        return [StructuredTool.from_function(coroutine=fn, name=fn.__name__, description=fn.__doc__) for fn in self.workflows.tools()]

    def endpoint_info(self) -> dict:
        """For apps, not models. On the agent server: role 'agent_server', the services it offers (services.workflows: key -> tool name),
        the other endpoints (urls) and the instrument endpoints workflows run on (workflow_instruments, the default first).
        On the workflow operator endpoint: role 'operator', serves 'workflow_approvals', the library and run folder it
        reads, the tools it offers and the instrument endpoints it reviews for (workflow_instruments)."""
        if self.endpoint_role == "agent":
            offered = {fn.__name__ for fn in self.workflows.tools()} if self.workflows is not None else set()
            workflows = {verb: name for verb, name in workflow_services.items() if name in offered}
            return {"role": "agent_server", "urls": dict(self.urls), "services": {"workflows": workflows} if workflows else {},
                    "workflow_instruments": list(self.workflows.endpoints.instruments) if self.workflows is not None else []}
        return {"role": self.endpoint_role, "serves": "workflow_approvals", "urls": dict(self.urls),
                "library": str(self.workflows.library.root.resolve()), "state_dir": str(self.workflows.state_dir.resolve()),
                "tools": [fn.__name__ for fn in self.workflows.operator_tools()],
                "workflow_instruments": list(self.workflows.endpoints.instruments)}

    def action_space(self) -> dict:
        """For apps, not models: the workflow operator controls, in the instrument endpoints' action-space shape."""
        from asyncroscopy.core.names import ROLES
        from asyncroscopy.mcp.action_space import SCHEMA_VERSION, _purpose

        docs = {fn.__name__: fn.__doc__ or "" for fn in self.workflows.operator_tools()}
        actions = [{"tool": tool, "device": None, "command": None, "role": role, "role_source": "declared",
                    "purpose": _purpose(docs[tool], None), "risk": risk, "effect": effect}
                   for tool, role, risk, effect in OPERATOR_ACTIONS]
        return {"schema_version": SCHEMA_VERSION, "endpoint": {"name": self.mcp.name, "role": self.endpoint_role}, "roles": dict(ROLES),
                "instruments": [], "actions": actions, "state": [], "jobs": [], "warnings": []}

    def setup(self) -> None:
        """Register the swarm tools on the FastMCP instance."""
        if self.endpoint_role == "operator":
            native = [*self.workflows.operator_tools(), self.endpoint_info, self.action_space]
            for tool_func in native:
                self.mcp.add_tool(tool_func)
            print(f"MCP workflow operator endpoint ready: {len(native)} tool(s) registered", flush=True)
            return
        native = [self.endpoint_info, self.query_agent, self.spawn_agent, self.list_agents, self.list_agent_tools, self.complete, self.set_max_steps]
        if self.workflows is not None:  # lets a client run a workflow without a model
            native += self.workflows.tools()
        for tool_func in native:
            self.mcp.add_tool(tool_func)
            if self.verbose:
                print(f"Registered native tool: {tool_func.__name__}")

        # Always printed: the startup GUI reads this line.
        print(f"MCP ready: {len(native)} tool(s) registered", flush=True)
        if self.workflows is not None:
            where = "simulated stacks and the real instrument" if self.workflows.hardware else "simulated stacks only"
            print(f"Workflows run in this process ({where}): list_workflows, run_workflow, workflow_status, stop_workflow, ...", flush=True)

        if self.verbose:
            print("\nAvailable agent tools:")
            print("  - query_agent(prompt: str) -> str")
            print("  - spawn_agent(config: str) -> str")
            print("  - list_agents() -> list[str]")
            print("  - list_agent_tools() -> str")
            print("  - complete(request_json: str) -> str")
            print("  - set_max_steps(max_steps: int) -> int")
            print(f"\nInheriting tools from MCP servers: {', '.join(self.mcp_urls)}")

    async def query_agent(self, prompt: str, include_transcript: bool = True, graph: GraphName = "", goal: bool = False,
                          model: str = ""):
        """
        Ask the agents to carry out a request and return the response.

        graph: "" uses the agent swarm; "mapping" runs the fixed map workflow
        (plan, reserve, run, stitch, score, report); "general" runs
        the single tool-using agent with skills; "team" runs a planner with an
        operator and an analyst agent.
        goal: treat the prompt as a goal (a sentence, or JSON with statement,
        criteria and budget): the team (or general, if graph says so) works
        until the goal is checked as met from the servers' replies, retrying
        with feedback within the budget; the reply starts with whether it was met.
        model: the model to run on, a name under this server's provider (Ollama by default), such as the one the
        client has chosen; "" keeps the server's (ASYNCROSCOPY_MODEL, else the agent profile's).
        If `include_transcript` is true (default), the routing decisions,
        tool calls, and generated text will also be included above the final answer.
        Images the tools returned on the way (a segmentation, the atoms found) come after the answer as image blocks,
        the latest few, so the chat shows them. A call the instrument held for a person's approval stops there: the
        reply starts with what is waiting (tool, arguments, request id).
        """
        from asyncroscopy.agents.mcp_tools import collect_held, collect_images, held_summary
        from asyncroscopy.agents.run import run_goal_text, run_named_graph

        async with self._query_lock:
            if model.strip():
                self.swarm.use_model(model.strip())
            with collect_images() as images, collect_held() as held:
                if goal:
                    stripped = prompt.strip()
                    spec = json.loads(stripped) if stripped.startswith("{") else stripped
                    answer = (await run_goal_text(spec, graph if graph in ("team", "general") else "team", model=self.swarm.model))[0]
                elif graph:
                    answer = await run_named_graph(graph, prompt, model=self.swarm.model)
                else:
                    answer = await self.swarm.query(prompt, include_transcript=include_transcript)
        if held:
            answer = f"{held_summary(held)}\n\n{answer}"
        return with_images(answer, images)

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

    def set_max_steps(self, max_steps: int) -> int:
        """Set the LangGraph recursion limit used per query, returning the new value."""
        return self.swarm.set_max_steps(max_steps)

    def start(self, transport: str | None = None, allowed_hosts: list[str] | None = None, **kwargs) -> None:
        """
        Register the swarm tools and begin serving the MCP protocol.

        The swarm itself is initialized by this instance's FastMCP lifespan, so
        that it runs on the same event loop that serves tool calls.

        Args:
            transport: Transport protocol ("stdio", "http", "sse", or "streamable-http").
                Defaults to None, which uses stdio for local piping to agents.
            allowed_hosts: Over HTTP, hosts besides the bind host and this computer that a request's Host and Origin
                may name (``asyncroscopy.mcp.host_guard``).
            **kwargs: Additional keyword arguments to pass to the MCP server.
        """
        self.setup()
        if transport not in (None, "stdio"):
            kwargs["middleware"] = [*kwargs.get("middleware", []), *host_guard.middleware(kwargs.get("host", ""), allowed_hosts or [])]
        self.mcp.run(transport=transport, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--transport", required=True)
    parser.add_argument("--http-host", required=True)
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--mcp-urls-json", default=json.dumps([DEFAULT_MCP_URL]))
    parser.add_argument("--model", default=None, help="Model name; default: the default agent profile's model.")
    parser.add_argument("--model-provider", default="ollama")
    parser.add_argument("--max-steps", type=int, default=None, help="Recursion limit per query; default: the swarm's.")
    parser.add_argument("--startup-agents-json", default="[]")
    parser.add_argument("--use-init-chat-model", action="store_true")
    parser.add_argument("--no-workflows", dest="workflows", action="store_false",
                        help="Do not run workflows here. By default this process runs them on simulated stacks and offers their tools "
                             "(list_workflows, run_workflow, ...).")
    parser.add_argument("--workflows-hardware", action="store_true",
                        help="Also run workflows on a stack that does not say it is simulated (a real instrument). Needs a loopback --http-host.")
    parser.add_argument("--endpoint-role", choices=ENDPOINT_ROLES, default="agent",
                        help="agent (default): the swarm and the workflow tools. operator: only the workflow review, approve and reject "
                             "tools, for a person; needs a loopback --http-host.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--urls-json", default="{}", help='{"operator", "agent_server", "workflow_operator", "tiled", "analysis": URL}: the endpoints endpoint_info reports.')
    parser.add_argument("--allowed-hosts-json", default="[]", help="Hosts besides the bind host and this computer that requests may name in Host or Origin.")
    parser.add_argument("--workflow-instruments-json", default="[]",
                        help="Other instrument endpoint URLs workflows may run on (run_workflow's instrument_url), besides the first of --mcp-urls-json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from asyncroscopy.agents.graphs.swarm import Agent

    if args.endpoint_role == "operator" and args.transport == "streamable-http" and not is_loopback_host(args.http_host):
        print(f"MCP ERROR: --endpoint-role operator needs --http-host 127.0.0.1 (got {args.http_host}): it approves workflows for the "
              "real instrument and has no login.", flush=True)
        return 1
    if args.workflows_hardware and args.transport == "streamable-http" and not is_loopback_host(args.http_host):
        print(f"MCP ERROR: --workflows-hardware needs --http-host 127.0.0.1 (got {args.http_host}): this server has no login, "
              "and with it anyone who can reach the port would run workflows on the real instrument.", flush=True)
        return 1
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
        workflows=args.workflows,
        workflows_hardware=args.workflows_hardware,
        endpoint_role=args.endpoint_role,
        urls=json.loads(args.urls_json),
        workflow_instruments=json.loads(args.workflow_instruments_json),
    )
    if args.transport == "streamable-http":
        print(
            f"Starting {args.name} at http://{args.http_host}:{args.http_port}/mcp "
            f"with tools inherited from {', '.join(server.mcp_urls)}",
            flush=True,
        )
        server.start(transport="streamable-http", host=args.http_host, port=args.http_port, allowed_hosts=json.loads(args.allowed_hosts_json))
    else:
        server.start(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())