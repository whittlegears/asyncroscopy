"""Command line for workflows.

    python -m asyncroscopy.workflows blocks                       # list every block
    python -m asyncroscopy.workflows list                         # list workflows and their status
    python -m asyncroscopy.workflows view [--open]                # draw every workflow on one HTML page
    python -m asyncroscopy.workflows show map_grid                # print a workflow's Python
    python -m asyncroscopy.workflows lint map_grid [--endpoint]   # check a graph, with --endpoint also the live tools
    python -m asyncroscopy.workflows run map_grid --input size_m=6e-7
    python -m asyncroscopy.workflows run map_grid --hardware      # run on a real instrument
    python -m asyncroscopy.workflows status <run_id> | stop <run_id> | resume <run_id> | runs

For a workflow written outside the repository:

    python -m asyncroscopy.workflows propose my_workflow.py       # check it and keep it as a draft
    python -m asyncroscopy.workflows show my_workflow             # read it
    python -m asyncroscopy.workflows approve my_workflow          # let it run on an instrument
    python -m asyncroscopy.workflows approve my_workflow --instrument any --days 90
    python -m asyncroscopy.workflows revoke my_workflow           # make it a draft again

A draft runs on a digital twin only. An approval covers the code of the workflow and its blocks as they are now.
Use --instrument STACK_ID to limit it to one instrument and --days N to make it expire.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path

from asyncroscopy.agents.config import AgentConfig
from . import describe_blocks
from .library import DEFAULT_LIBRARY, Library
from .runner import list_runs, read_run, resume_workflow, run_workflow, stop_workflow, twin_only_problem


def _value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


async def _servers(args):
    config = AgentConfig.from_env(caller=args.caller, mcp_url=args.mcp_url)
    if args.run_id:
        config.run_id = args.run_id
    await config.info()  # the analysis server address comes from here
    return config, config.instrument(), config.analysis()


def _show(record: dict) -> int:
    print(json.dumps({k: v for k, v in record.items() if k != "blueprint"}, indent=1, default=str))
    return 0 if record.get("state") == "done" else 1


def _refused(problem: str) -> int:
    print(json.dumps({"state": "refused", "problems": [problem]}, indent=1))
    return 1


def _find(library: Library, name: str):
    """Find a workflow and its status, or print why it cannot be found."""
    try:
        return library.find(name)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m asyncroscopy.workflows", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("target", nargs="?", help="a workflow name, a run id, or (propose) a workflow file")
    parser.add_argument("--library", default=str(DEFAULT_LIBRARY), help="where proposed and approved workflows are kept (default: %(default)s)")
    parser.add_argument("--by", default=None, help="approve: the person approving (default: the logged-in user)")
    parser.add_argument("--yes", action="store_true", help="approve: do not ask to confirm (the workflow is still printed)")
    parser.add_argument("--instrument", default="any", metavar="STACK_ID",
                        help="approve: only for this instrument (an endpoint's stack_id); default: any")
    parser.add_argument("--request", default="", help="propose: what was asked for, in the asker's words (kept with the draft)")
    parser.add_argument("--out", default=None, help="view: the HTML file to write (default: workflows.html beside the library)")
    parser.add_argument("--open", action="store_true", help="view: also open the page in the browser")
    parser.add_argument("--days", type=float, default=None, help="approve: how many days the approval holds (default: until revoked)")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=VALUE", help="a workflow input (repeat)")
    parser.add_argument("--endpoint", action="store_true", help="lint: also check the tools the live endpoint offers")
    parser.add_argument("--mcp-url", default=None, help="the instrument's agent endpoint (default: ASYNCROSCOPY_MCP_URL or the twin's)")
    parser.add_argument("--caller", default="workflow", help="the name this run is traced under")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hardware", action="store_true",
                        help="run, resume: also drive an endpoint that does not say it is simulated (a real instrument)")
    return parser


def _cmd_blocks(args, library: Library) -> int:
    print(json.dumps(describe_blocks(), indent=1))
    return 0


def _cmd_list(args, library: Library) -> int:
    print(json.dumps(library.listed(), indent=1))
    return 0


def _cmd_view(args, library: Library) -> int:
    from .view import render_html

    out = Path(args.out) if args.out else library.root.parent / "workflows.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(library), encoding="utf-8")
    print(out.resolve())
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    return 0


def _cmd_runs(args, library: Library) -> int:
    print(json.dumps(list_runs(), indent=1))
    return 0


def _cmd_status(args, library: Library) -> int:
    return _show(read_run(args.target))


def _cmd_stop(args, library: Library) -> int:
    print(json.dumps(stop_workflow(args.target)))
    return 0


def _cmd_propose(args, library: Library) -> int:
    try:
        source = Path(args.target).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {args.target}: {exc}", file=sys.stderr)
        return 1
    result = library.propose(source, request=args.request)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


def _cmd_revoke(args, library: Library) -> int:
    try:
        print(json.dumps(library.revoke(args.target)))
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    return 0


def _cmd_show(args, library: Library) -> int:
    found = _find(library, args.target)
    if found is None:
        return 1
    print(found[0].source())
    return 0


def _print_for_approval(library: Library, workflow, status: str) -> None:
    """Show the person exactly what they are approving."""
    checked = workflow.lint()
    rehearsals = library.rehearsals(workflow.name, checked["hash"])
    print(workflow.source())
    print(json.dumps({"name": workflow.name, "status": status, "acts": checked["acts"], "blocks": checked["blocks"],
                      "tools": checked["tools"], "problems": checked["problems"], "hash": checked["hash"],
                      "requested": (library.request(workflow.name) or {}).get("request"),
                      "rehearsals_of_this_code_on_a_twin": rehearsals}, indent=1, default=str))
    if not [r for r in rehearsals if r.get("state") == "done"]:
        print("Note: this code has never run to the end on a digital twin.")


def _confirmed(name: str, instrument: str, by: str) -> bool:
    """Ask the person at the terminal to type the workflow's name."""
    if not sys.stdin.isatty():
        print("not approved: confirm at a terminal, or pass --yes", file=sys.stderr)
        return False
    if input(f"Approve {name} to run on {instrument}, as {by}? Type its name to approve: ").strip() != name:
        print("not approved")
        return False
    return True


def _cmd_approve(args, library: Library) -> int:
    found = _find(library, args.target)
    if found is None:
        return 1
    workflow, status = found
    _print_for_approval(library, workflow, status)
    if status == "shipped":
        print(f"{workflow.name} is shipped; there is nothing to approve.")
        return 0
    login = getpass.getuser()
    by = args.by or login
    instrument = args.instrument
    ends = f" for {args.days:g} days" if args.days else ", until revoked"
    print(f"This approval: instrument {instrument}{ends}; given by {by} (login {login}).")
    if not args.yes and not _confirmed(workflow.name, instrument, by):
        return 1
    try:
        print(json.dumps(library.approve(workflow.name, by, login=login, instrument=instrument, days=args.days), indent=1, default=str))
    except (KeyError, ValueError, PermissionError) as exc:
        print(f"not approved: {exc.args[0]}", file=sys.stderr)
        return 1
    return 0


async def _offered_tools(args) -> tuple[list[str], list[str] | None]:
    """List the tools the live instrument and analysis servers offer."""
    _, instrument, analysis = await _servers(args)
    offered = sorted(t.name for t in await instrument.list_tools())
    try:
        offered_analysis = sorted(t.name for t in await analysis.list_tools()) if analysis else []
    except Exception as exc:
        print(f"the analysis server did not answer ({type(exc).__name__}: {exc}); its tools are not checked", file=sys.stderr)
        offered_analysis = None
    return offered, offered_analysis


async def _cmd_lint(args, library: Library) -> int:
    found = _find(library, args.target)
    if found is None:
        return 1
    offered, offered_analysis = await _offered_tools(args) if args.endpoint else (None, None)
    result = found[0].lint(offered, offered_analysis)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


async def _endpoint(args):
    """Connect to the instrument, or print why this command may not drive it."""
    config, instrument, analysis = await _servers(args)
    problem = twin_only_problem(await config.info(), hardware=args.hardware)
    if problem:
        _refused(problem)
        return None
    return config, instrument, analysis, await config.info()


async def _cmd_run(args, library: Library) -> int:
    found = _find(library, args.target)
    if found is None:
        return 1
    workflow, status = found
    endpoint = await _endpoint(args)
    if endpoint is None:
        return 1
    config, instrument, analysis, info = endpoint
    simulated = info.get("simulated") is True
    refusal = None if simulated else library.hardware_problem(workflow.name, status, info.get("stack_id"))
    if refusal:
        return _refused(refusal)
    inputs = {}
    for pair in args.input:
        name, text = pair.split("=", 1)
        inputs[name] = _value(text)
    record = await run_workflow(workflow, inputs, instrument=instrument, analysis=analysis, run_id=config.run_id)
    if simulated:
        library.record_rehearsal(workflow, record, info)
    return _show(record)


async def _cmd_resume(args, library: Library) -> int:
    endpoint = await _endpoint(args)
    if endpoint is None:
        return 1
    _, instrument, analysis, info = endpoint
    simulated = info.get("simulated") is True
    try:
        workflow, status = library.find(str(read_run(args.target)["workflow"]))
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 1
    refusal = None if simulated else library.hardware_problem(workflow.name, status, info.get("stack_id"))
    if refusal:
        return _refused(refusal)
    record = await resume_workflow(args.target, instrument=instrument, analysis=analysis, find=library.workflow)
    if simulated:
        library.record_rehearsal(workflow, record, info)
    return _show(record)


COMMANDS = {"blocks": _cmd_blocks, "list": _cmd_list, "view": _cmd_view, "show": _cmd_show, "lint": _cmd_lint, "run": _cmd_run,
            "status": _cmd_status, "stop": _cmd_stop, "resume": _cmd_resume, "runs": _cmd_runs,
            "propose": _cmd_propose, "approve": _cmd_approve, "revoke": _cmd_revoke}
NEEDS_NO_TARGET = ("blocks", "list", "view", "runs")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    library = Library(args.library)
    if args.command not in NEEDS_NO_TARGET and not args.target:
        parser.error(f"{args.command} needs a workflow or a run id")
    outcome = COMMANDS[args.command](args, library)
    return asyncio.run(outcome) if asyncio.iscoroutine(outcome) else outcome


if __name__ == "__main__":
    sys.exit(main())
