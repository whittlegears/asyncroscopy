"""The workflow tools, which an agent or any MCP client uses to run workflows.

They are served by the agent MCP server (``asyncroscopy.mcp.agent_mcp_server``; ``agent.workflows`` in its config, on
by default), the one process that runs workflows. There is no workflow server of its own.

An agent chooses a workflow, fills in its inputs, runs it and reads the result. For a task of many steps that no
workflow fits, the agent can write one. The workflow then carries out the steps exactly and much faster than a
model calling tools one at a time. The runner makes every instrument call through the instrument's agent endpoint,
so the endpoint's risk policy, limits and trace apply. A workflow is LangGraph code built from
blocks. get_workflow shows that code and list_blocks shows the blocks.

The agent sends what it wrote with propose_workflow, along with the person's own words for the task. The text is
checked and kept as a draft by ``authoring.py`` and ``library.py``. ``view.py`` shows it with every other workflow.
A draft runs on a digital twin only. On an instrument only a shipped workflow or one a person has approved will
run. A person approves or rejects on the command line, or through ``operator_tools`` (review_workflow,
approve_workflow, reject_workflow), which only the workflow operator endpoint serves
(``agent_mcp_server --endpoint-role operator``, ``docs/MCP/workflow_approval.md``). They are never in ``tools``, so
no agent is given them.

Workflows run on a real instrument only when ``hardware`` is set. The server that hosts these tools then has to
bind to loopback because it has no login.
"""

from __future__ import annotations

import asyncio
import getpass
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from asyncroscopy.core.names import CALLER_HEADER

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.mcp_tools import McpServer
from . import runner
from .authoring import RULES, TEMPLATE, TEMPLATE_PARALLEL
from .blocks import describe_blocks
from .graph import Workflow
from .library import APPROVED, SHIPPED, Library, needs_approval

MIN_REQUEST_CHARS = 12  # shorter than this is a placeholder and not a real request
MAX_WAIT_S = 100.0  # below the 120 s our MCP client allows for one call
RECORD_KEYS = ("run_id", "workflow", "state", "queue", "problems", "error", "questions", "inputs", "outputs", "at_node", "cleanup", "unrestored",
               "calls", "tools_called", "started", "ended")
OUTPUT_CHARS = 1500
GOING = ("running", "queued")  # a run in one of these has a process driving it, or waiting to


def requester() -> str | None:
    """The caller named in the MCP request being served (an agent's X-Asyncroscopy-Caller), or None.

    A run's instrument calls carry it, so the instrument's trace, the benchmarks and the evaluators credit the agent
    that asked for the run; the run id still tells the workflow's calls apart. Over stdio there is no request header.
    With mcp.auth the endpoint names the caller from this server's token instead, whatever is sent.
    """
    try:
        from fastmcp.server.dependencies import get_http_headers

        value = get_http_headers(include_all=True).get(CALLER_HEADER.lower(), "")
    except Exception:  # no HTTP request: stdio, or a direct call in a test
        return None
    return value.strip() or None


class Endpoints:
    """Opens the instrument and analysis endpoints for a run and reads the endpoint's info."""

    def __init__(self, mcp_url: str | None = None, caller: str = "workflow") -> None:
        self.mcp_url, self.caller = mcp_url, caller

    async def open(self, run_id: str | None = None, caller: str | None = None) -> tuple[McpServer, McpServer | None, dict]:
        """``caller``: the name the run's calls carry in the instrument's trace (this server's own name when None)."""
        config = AgentConfig.from_env(caller=caller or self.caller, mcp_url=self.mcp_url)
        if run_id:
            config.run_id = run_id
        info = await config.info()
        return config.instrument(), config.analysis(), info


def _brief(value: Any) -> Any:
    """Shorten a run's outputs for a model. A long list becomes its count, first and last item."""
    if isinstance(value, dict):
        return {k: _brief(v) for k, v in value.items()}
    if isinstance(value, list) and len(value) > 6:
        return {"count": len(value), "first": _brief(value[0]), "last": _brief(value[-1])}
    if isinstance(value, str) and len(value) > OUTPUT_CHARS:
        return value[:OUTPUT_CHARS] + "..."
    return value


def _reply(record: dict, state_dir: Path | None = None) -> dict:
    """A run's record for a model. With ``state_dir`` a running run's progress is also read from another process."""
    out = {k: record[k] for k in RECORD_KEYS if record.get(k) not in (None, [], {})}
    if "outputs" in out:
        out["outputs"] = _brief(out["outputs"])
    elsewhere = state_dir if record.get("state") in GOING else None
    live = runner.progress(record["run_id"], elsewhere) if record.get("run_id") else None
    if live:
        out["progress"] = live
    state = out.get("state")
    if state == "running":
        out["next"] = f"still going: call workflow_status with run_id {record['run_id']} and wait_s {MAX_WAIT_S:.0f} again"
    elif state == "queued":
        out["next"] = (f"another run holds the instrument ('queue' says which); this one starts by itself when its turn comes. Call "
                       f"workflow_status with run_id {record['run_id']} and wait_s {MAX_WAIT_S:.0f} again, or stop_workflow to leave the queue")
    elif state == "waiting":
        out["next"] = ("the workflow asks the user a question and does nothing until it is answered. Put each question to the user in "
                       "its own words (with its options; if it has 'show', show them that first, a data key with view_data) and end your "
                       f"turn. When the user has answered, call answer_workflow with run_id {record['run_id']} and their answer. "
                       "Never answer it yourself.")
    elif state in ("stopped", "failed"):
        out["next"] = f"tell the user why. resume_workflow with run_id {record['run_id']} continues from where it stopped, without repeating finished steps"
    elif state == "refused":
        out["next"] = "nothing was run. Fix what problems names and call run_workflow again"
    return out


class WorkflowServer:
    def __init__(self, endpoints: Any = None, state_dir: str | Path = runner.DEFAULT_STATE_DIR,
                 name: str = "asyncroscopy-workflows", hardware: bool = False, library: Library | None = None) -> None:
        self.endpoints = endpoints or Endpoints()
        self.hardware = bool(hardware)  # True allows runs on a real instrument
        self.state_dir = Path(state_dir)
        self.library = library if library is not None else Library(self.state_dir / "library")
        self.tasks: dict[str, asyncio.Task] = {}
        self.mcp = FastMCP(name)
        for fn in self.tools():
            self.mcp.tool()(fn)

    def _workflow(self, name: str) -> tuple[Workflow, str]:
        """Return the workflow and its status in the library."""
        try:
            return self.library.find(str(name))
        except KeyError as exc:
            raise LookupError(f"there is no workflow named {name!r}; call list_workflows") from exc

    def _refusal(self, info: dict, name: str, status: str) -> str | None:
        """Why this workflow may not run on this endpoint's stack, or None.

        A twin runs anything. A real instrument needs ``hardware`` and a shipped or approved workflow.
        """
        if info.get("simulated") is True:
            return None
        return runner.twin_only_problem(info, hardware=self.hardware) or self.library.hardware_problem(name, status, info.get("stack_id"))

    async def _rehearsed(self, info: dict, workflow: Workflow, running) -> dict:
        """Wait for a run to end. A run on a digital twin is saved as a rehearsal for whoever approves later."""
        record = await running
        if info.get("simulated") is True:
            try:
                self.library.record_rehearsal(workflow, record)
            except OSError:  # the runner already saved the run so losing the rehearsal is harmless
                pass
        return record

    def _running(self) -> str | None:
        return next((run_id for run_id, task in self.tasks.items() if not task.done()), None)

    async def _wait(self, run_id: str, wait_s: float, with_tiles: bool = False) -> dict:
        """The run's reply once it ends or ``wait_s`` passes. Its placed tiles are left out unless ``with_tiles``:
        an app drawing the map wants them, a model reading the reply does not."""
        reply = await self._wait_reply(run_id, wait_s)
        progress = reply.get("progress")
        if not with_tiles and isinstance(progress, dict) and "tiles" in progress:
            reply["progress"] = {k: v for k, v in progress.items() if k != "tiles"}
        return reply

    async def _wait_reply(self, run_id: str, wait_s: float) -> dict:
        task = self.tasks.get(run_id)
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=max(0.0, min(float(wait_s), MAX_WAIT_S)))
            except TimeoutError:
                pass
        if task is not None and task.done() and not task.cancelled() and task.exception() is None:
            return _reply(task.result())
        try:
            record = runner.read_run(run_id, self.state_dir)
        except KeyError:
            if task is not None and not task.done():
                return _reply({"run_id": run_id, "state": "running"})
            if task is not None and task.exception() is not None:
                return {"run_id": run_id, "state": "failed", "error": f"{type(task.exception()).__name__}: {task.exception()}"}
            return {"error": f"there is no workflow run {run_id!r}; call list_runs"}
        # A run another process drives (it holds the run's lock) is still going; one nobody drives was cut off.
        if record["state"] in GOING and (task is None or task.done()) and not runner.driven_elsewhere(run_id, self.state_dir):
            record["state"], record["error"] = "stopped", "the workflow server was restarted while this run was going"
        return _reply(record, self.state_dir)

    def _start(self, run_id: str, make: Callable[[], Any]) -> None:
        self.tasks[run_id] = asyncio.get_running_loop().create_task(make())

    async def _continue(self, run_id: str, wait_s: float, answers: dict | None = None) -> dict:
        """Resume a run that was stopped or failed, or one waiting for a person when ``answers`` is given."""
        going = self._running()
        if going:
            return {"error": f"workflow run {going} is still going; call workflow_status with it"}
        try:
            record = runner.read_run(run_id, self.state_dir)
        except KeyError as exc:
            return {"error": str(exc.args[0])}
        if record["state"] in ("done", "refused"):
            return _reply(record)
        try:
            runner.check_answers(record, answers)
        except ValueError as exc:
            return {**_reply(record), "error": str(exc)}
        try:
            instrument, analysis, info = await self._open(run_id, requester())
        except Exception as exc:
            return {"error": f"the instrument endpoint could not be reached: {type(exc).__name__}: {exc}"}
        try:
            again, status = self._workflow(record["workflow"])
        except LookupError as exc:
            return {"state": "refused", "problems": [str(exc)]}
        problem = self._refusal(info, str(record["workflow"]), status)
        if problem:
            return {"state": "refused", "problems": [problem]}
        self._start(run_id, lambda: self._rehearsed(info, again, runner.resume_workflow(
            run_id, instrument=instrument, analysis=analysis, state_dir=self.state_dir, find=self.library.workflow, answers=answers)))
        return await self._wait(run_id, wait_s)

    def review(self, name: str) -> dict:
        """Everything a person needs to decide on the workflow ``name``. Read only. Raises LookupError for no such workflow."""
        workflow, status = self._workflow(name)
        checked = workflow.lint()
        current = checked["hash"]
        approval = self.library.approval(workflow.name)
        rejection = self.library.rejection(workflow.name)
        asked = self.library.request(workflow.name)
        rehearsals = []
        for entry in self.library.rehearsals(workflow.name):
            shown = {k: entry.get(k) for k in ("run_id", "state", "at", "calls", "inputs", "error")}
            shown["problems"] = list(entry.get("problems") or [])
            shown["this_code"] = entry.get("hash") == current
            try:
                run = runner.read_run(str(entry.get("run_id")), self.state_dir)
            except (KeyError, OSError, ValueError):
                run = None
            shown["outputs"] = _brief(run.get("outputs") or {}) if run is not None else None
            if run is not None and run.get("problems"):
                shown["problems"] = list(dict.fromkeys([*shown["problems"], *run["problems"]]))
            rehearsals.append(shown)
        rehearsed = any(r["this_code"] and r["state"] == "done" for r in rehearsals)
        if status == SHIPPED:
            refusal = f"{workflow.name} is shipped with the software; there is nothing to approve"
        elif status == APPROVED:
            refusal = f"{workflow.name} is already approved, as it is now"
        elif not checked["ok"]:
            refusal = f"{workflow.name} does not pass its checks: {checked['problems']}"
        elif not rehearsed:
            refusal = f"{workflow.name} has never run to the end on a digital twin as it is now; run it there first"
        else:
            refusal = None
        return {
            "name": workflow.name, "status": status, "summary": workflow.summary, "inputs": dict(workflow.inputs), "budget": dict(workflow.budget),
            "graph": {k: v for k, v in workflow.blueprint().items() if k in ("nodes", "edges")}, "source": workflow.source(),
            "hash": current, "acts": checked["acts"], "tools": checked["tools"], "blocks": checked["blocks"], "problems": checked["problems"],
            "request": {"text": asked.get("request"), "at": asked.get("at"), "this_code": asked.get("hash") == current} if asked else None,
            "approval": approval,
            "changed_since_approval": approval is not None and approval.get("hash") != current,
            "rejection": {**rejection, "this_code": rejection.get("hash") == current} if rejection else None,
            "rehearsals": rehearsals, "rehearsed": rehearsed,
            "can_approve": refusal is None, "why_not": refusal,
        }

    def operator_tools(self) -> list:
        """The tools a person uses to review, approve and reject workflows. Only the workflow operator endpoint
        (``agent_mcp_server --endpoint-role operator``) serves them; no agent is given them."""
        server = self
        via = self.mcp.name

        async def review_workflow(name: str) -> dict:
            """Everything a person needs to decide whether a workflow may run on the real instrument: its status, the
            user's request and when it was made, the summary, inputs, graph and source, every rehearsal on a digital
            twin (when, state, outputs, problems), any approval or rejection, and whether it changed since it was
            approved. Reads only."""
            try:
                return {"ok": True, **server.review(name)}
            except LookupError as exc:
                return {"ok": False, "error": str(exc)}

        async def approve_workflow(name: str, approver: str, instrument: str | None = None, days: float | None = None) -> dict:
            """A person approves a draft workflow, as it is now, to run on the real instrument. approver: the person's
            name. instrument: the stack_id it is approved for (left out or "any": every instrument). days: how long the
            approval holds (left out: until revoked). Refused unless this exact code ran to the end on a digital twin."""
            try:
                workflow, status = server._workflow(name)
            except LookupError as exc:
                return {"ok": False, "error": str(exc)}
            if status == SHIPPED:
                return {"ok": False, "status": status, "error": f"{workflow.name} is shipped with the software; there is nothing to approve"}
            try:
                record = server.library.approve(workflow.name, approver, login=getpass.getuser(), instrument=instrument, days=days,
                                                require_rehearsal=True, via=via)
            except (KeyError, ValueError, PermissionError) as exc:
                return {"ok": False, "status": status, "error": str(exc.args[0])}
            return {"ok": True, "status": APPROVED, "approval": record}

        async def reject_workflow(name: str, reason: str) -> dict:
            """A person rejects a workflow, with the reason, so it is not approved as it is. An approved one loses its
            approval. The text is kept as a draft and still runs on a digital twin."""
            try:
                record = server.library.reject(name, reason, login=getpass.getuser(), via=via)
            except KeyError as exc:
                return {"ok": False, "error": f"there is no workflow named {name!r}" if not exc.args else str(exc.args[0])}
            except (ValueError, PermissionError) as exc:
                return {"ok": False, "error": str(exc.args[0])}
            return {"ok": True, "rejection": record}

        return [review_workflow, approve_workflow, reject_workflow]

    async def _open(self, run_id: str, caller: str | None):
        return await (self.endpoints.open(run_id, caller=caller) if caller else self.endpoints.open(run_id))

    def tools(self) -> list:
        server = self

        async def list_workflows() -> dict:
            """The workflows that can be run: fixed, checked procedures (a map, ...), each a LangGraph graph built
            from blocks. This one call lists all of them, with each one's summary and inputs (defaults and ranges).
            When to look here: the task has many instrument steps, a loop, or steps that must happen at the same
            time (a map, a grid, a series, a survey). If a workflow fits, run it with run_workflow instead of
            making those calls yourself. If none fits such a task, write one for it (list_blocks, then
            propose_workflow) and tell the user it waits for their approval: a workflow runs the steps exactly and
            far faster than you can call them one by one. Do not call this again for the same task. Make the
            calls one by one yourself only if the user says to.
            A single action (one image, one spectrum, reading or setting one value) needs no workflow: just do it.
            status: 'shipped' and 'approved' ones have been reviewed by a person; a 'draft' was proposed and runs
            on a digital twin only."""
            return {"workflows": server.library.listed()}

        async def get_workflow(name: str) -> dict:
            """One workflow in full: the Python that builds its LangGraph graph ('source'), and the same graph as data
            ('graph': each node's block and inputs, and the edges)."""
            try:
                workflow, status = server._workflow(name)
            except LookupError as exc:
                return {"error": str(exc)}
            return {"name": workflow.name, "status": status, "summary": workflow.summary, "inputs": dict(workflow.inputs), "budget": dict(workflow.budget),
                    "graph": {k: v for k, v in workflow.blueprint().items() if k in ("nodes", "edges")}, "source": workflow.source()}

        async def run_workflow(name: str, inputs: dict | None = None, wait_s: float = 60.0) -> dict:
            """Run a workflow by name. inputs: only the ones you want to change from their defaults, as {name: value};
            a value outside an input's range is refused, never adjusted. The workflow runs by itself, in the
            background; this waits up to wait_s (at most 100) and returns the result if it finished, otherwise
            state 'running' with a run_id for workflow_status. state 'waiting' means it asks the user a question
            ('questions'); 'next' always says what to do. One workflow acts on the instrument at a time: started
            while another is acting, this one is 'queued' and starts by itself when its turn comes. The result's
            outputs hold data keys (a stitched map's key is outputs.stitch.map). Inputs are checked against the
            workflow's declared inputs (an unknown name, a wrong type or a value out of range is refused) before
            anything is opened or run."""
            try:
                workflow, status = server._workflow(name)
            except LookupError as exc:
                return {"state": "refused", "problems": [str(exc)]}
            if inputs is not None and not isinstance(inputs, dict):
                return {"state": "refused", "workflow": workflow.name, "problems": [f"inputs must be an object of name: value, not {inputs!r}"],
                        "next": "nothing was run. Fix what problems names and call run_workflow again"}
            try:
                workflow.check_inputs(inputs)
            except ValueError as exc:
                return {"state": "refused", "workflow": workflow.name, "problems": [str(exc)],
                        "next": "nothing was run. Fix the inputs that problems names (get_workflow's inputs say what each takes) and call run_workflow again"}
            try:
                instrument, analysis, info = await server.endpoints.open()
            except Exception as exc:
                return {"state": "refused", "problems": [f"the instrument endpoint could not be reached: {type(exc).__name__}: {exc}"]}
            problem = server._refusal(info, workflow.name, status)
            if problem:
                return {"state": "refused", "problems": [problem]}
            run_id = uuid.uuid4().hex[:12]
            # Opened again so every call carries the run id, and the name of the caller that asked for the run.
            instrument, analysis, _ = await server._open(run_id, requester())
            server._start(run_id, lambda: server._rehearsed(info, workflow, runner.run_workflow(
                workflow, dict(inputs or {}), instrument=instrument, analysis=analysis, run_id=run_id, state_dir=server.state_dir)))
            return await server._wait(run_id, wait_s)

        async def workflow_status(run_id: str, wait_s: float = 100.0, with_tiles: bool = False) -> dict:
            """How a workflow run is going, waiting up to wait_s (at most 100) for it to finish first; use the full
            wait, so a long run takes few calls. state is running, done, waiting (it asks the user a question),
            stopped, failed or refused; 'next' says what to do. with_tiles adds where each map tile was taken, for an
            app that draws the map as it grows; leave it out."""
            return await server._wait(str(run_id), wait_s, bool(with_tiles))

        async def stop_workflow(run_id: str) -> dict:
            """Stop a running workflow before its next instrument call. Its cleanup (blanking the beam) still runs.
            A stopped run can be continued with resume_workflow."""
            try:
                runner.stop_workflow(str(run_id), server.state_dir)
            except KeyError as exc:
                return {"error": str(exc.args[0])}
            return await server._wait(str(run_id), 15.0)

        async def resume_workflow(run_id: str, wait_s: float = 60.0) -> dict:
            """Continue a stopped or failed workflow run from the step it was at; finished steps are not repeated.
            Use it after the cause of a failure is fixed. A run that
            is 'waiting' for the user's answer is continued with answer_workflow instead."""
            return await server._continue(str(run_id), wait_s)

        async def answer_workflow(run_id: str, answer: str, question_id: str | None = None, wait_s: float = 60.0) -> dict:
            """Give a waiting workflow the user's answer to its question, and let it go on. ONLY with an answer the
            user gave after you put the question to them: never answer for them, and never guess. answer: their
            answer (one of the question's options when it has options). question_id: needed only when the run has
            more than one question waiting (each is in 'questions' with its id)."""
            run_id = str(run_id)
            try:
                record = runner.read_run(run_id, server.state_dir)
            except KeyError as exc:
                return {"error": str(exc.args[0])}
            questions = record.get("questions") or []
            if record["state"] != "waiting" or not questions:
                return {**_reply(record), "error": f"run {run_id} is not waiting for an answer"}
            if question_id is None and len(questions) > 1:
                return {**_reply(record), "error": "this run has several questions waiting; give question_id, one answer per call"}
            return await server._continue(run_id, wait_s, {str(question_id or questions[0]["id"]): answer})

        async def list_runs(limit: int = 10) -> dict:
            """The latest workflow runs, newest first, with their state."""
            return {"runs": runner.list_runs(server.state_dir)[: max(1, int(limit))]}

        async def list_blocks() -> dict:
            """The blocks a workflow's graph is built from: each one's summary, its inputs (with types and defaults)
            and its outputs; with the rules a workflow must follow and two whole examples (a loop; and steps at the
            same time with a question for the user). Read it before writing a workflow with propose_workflow."""
            return {"blocks": describe_blocks(), "rules": RULES, "example": TEMPLATE, "example_parallel": TEMPLATE_PARALLEL}

        async def propose_workflow(user_request: str, source: str) -> dict:
            """Write a new workflow for a task the user gave you, for a scientist to approve. Use it when their
            task has many instrument steps, a loop, or steps at the same time and no workflow from list_workflows
            fits, or when they ask for a workflow. Not for a single action, and not for a task they did not ask
            for. You decide the steps from what you know of the task and the instrument; the scientist checks
            that it is safe for the hardware and worth running.
            user_request: the user's own words for the task, quoted as they wrote them. It is shown to the
            scientist who approves. source: the whole Python module as text, in the shape of list_blocks'
            examples (LangGraph code that wires blocks together; nothing else is accepted). Give it inputs for
            what the user may want to change next time (a size, a count, a field of view), with ranges. The
            reply's 'problems' say exactly what to fix.
            What is kept is a request, not a usable workflow: a scientist must approve it before it runs on an
            instrument, and you cannot approve it."""
            asked = str(user_request or "").strip()
            if len(asked) < MIN_REQUEST_CHARS:
                return {"ok": False, "draft": None, "status": None,
                        "problems": ["user_request is missing: quote the user's own words for the task this workflow is for. A workflow is "
                                     "written for a task the user gave; it is shown to the scientist who approves."],
                        "next": "nothing was kept"}
            offered = offered_analysis = None
            try:
                instrument, analysis, _ = await server.endpoints.open()
                offered = sorted(t.name for t in await instrument.list_tools())
                offered_analysis = sorted(t.name for t in await analysis.list_tools()) if analysis is not None else None
            except Exception:
                pass  # the live tools are checked again when it runs
            result = server.library.propose(str(source), offered, offered_analysis, request=asked)
            if result["ok"]:
                result["next"] = (f"kept as a request named {result['name']}, waiting for a person. Tell the user it is there for them to "
                                  f"look at and approve: {needs_approval(result['name'])} Do not run it for a real task before that.")
            else:
                result["next"] = "nothing was kept. Fix what problems names and call propose_workflow again with the whole module"
            return result

        return [list_workflows, get_workflow, run_workflow, workflow_status, answer_workflow, stop_workflow, resume_workflow, list_runs,
                list_blocks, propose_workflow]
