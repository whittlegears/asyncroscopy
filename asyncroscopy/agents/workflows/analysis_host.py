"""The pyTEMlib analysis MCP server with the workflow tools beside its own, in one process.

    python -m asyncroscopy.agents.workflows.analysis_host --tiled http://localhost:9091 --port 8003

It takes every option of ``python -m asyncroscopy.analysis_server`` and starts that same server with the workflow
tools in ``OFFERED`` added. A chat client that has the analysis server's tools can then run a workflow.

This lives here because ``asyncroscopy.analysis_server`` must never load the agents or the instrument endpoint.
tests/core/test_layering.py checks that. Workflows still make every instrument call through the instrument's agent
endpoint given by ``--workflows-mcp-url``, so its risk policy, limits and trace apply.

A model may also write a workflow here with list_blocks and propose_workflow. It is checked as text and kept as a
draft, and a draft runs on a digital twin only. On a real instrument nothing runs without ``--workflows-hardware``.
That flag needs a loopback ``--host`` because this server has no login. Even then only a shipped workflow or one a
person approved will run.
"""

from __future__ import annotations

import sys

from starlette.responses import HTMLResponse

from asyncroscopy import settings
from asyncroscopy.analysis_server import __main__ as analysis_cli
from asyncroscopy.analysis_server.server import MarkReadOnly
from asyncroscopy.core.net import is_loopback_host

from .server import Endpoints, WorkflowServer
from .view import render_html

VIEW_PATH = "/workflows"

OFFERED = ("list_workflows", "get_workflow", "run_workflow", "workflow_status", "answer_workflow", "stop_workflow", "resume_workflow", "list_runs",
           "list_blocks", "propose_workflow")
READ_ONLY = frozenset({"list_workflows", "get_workflow", "workflow_status", "list_runs", "list_blocks"})


def parser(env: settings.Settings):
    out = analysis_cli.parser(env)
    out.prog = "python -m asyncroscopy.agents.workflows.analysis_host"
    out.add_argument("--workflows-mcp-url", default="", metavar="URL",
                     help="The instrument's agent endpoint the workflows drive (empty: $ASYNCROSCOPY_MCP_URL, else the default endpoint).")
    out.add_argument("--workflows-hardware", action="store_true",
                     help="Also run workflows on a stack that does not say it is simulated (a real instrument). Needs a loopback --host.")
    return out


def add_workflow_tools(mcp, workflows: WorkflowServer) -> list[str]:
    """Add the workflow tools to ``mcp`` and return their names."""
    added = []
    for fn in workflows.tools():
        if fn.__name__ in OFFERED:
            mcp.tool()(fn)
            added.append(fn.__name__)
    mcp.add_transform(MarkReadOnly(READ_ONLY))

    @mcp.custom_route(VIEW_PATH, methods=["GET"], include_in_schema=False)
    async def workflows_page(request):
        return HTMLResponse(render_html(workflows.library))

    return added


def main(argv: list[str] | None = None) -> int:
    env = settings.get()
    args = parser(env).parse_args(argv)
    if args.workflows_hardware and args.transport != "stdio" and not is_loopback_host(args.host):
        print(f"--workflows-hardware needs --host 127.0.0.1 (got {args.host}): this server has no login, and with it anyone who "
              "can reach the port would run workflows on the real instrument.", file=sys.stderr, flush=True)
        return 1
    mcp, tools = analysis_cli.build(args, env)
    workflows = WorkflowServer(Endpoints(args.workflows_mcp_url or None, caller="workflow"), hardware=args.workflows_hardware)
    added = add_workflow_tools(mcp, workflows)
    where = "simulated stacks and the real instrument" if args.workflows_hardware else "simulated stacks only"
    page = f"; every workflow, drawn, for a person: http://{args.host}:{args.port}{VIEW_PATH}" if args.transport != "stdio" else ""
    analysis_cli.serve(mcp, tools, args, note=f"workflow tools ({where}): {', '.join(added)}{page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
