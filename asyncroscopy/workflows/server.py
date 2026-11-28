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

A workflow server acts on the instrument endpoints it was configured with (``Endpoints``: the first of the agent
server's ``mcp_urls`` and ``agent.workflow_instruments``), and on no other. ``run_workflow``, ``review_workflow`` and
``approve_workflow`` take ``instrument_url``, one of them, so an app that pairs a twin with an instrument rehearses on
the one and runs on the other (``docs/MCP/twin_to_instrument.md``). A rehearsal keeps its results and its twin score,
and the hardware run of an approved workflow gets its rehearsal's calls pre-approved (``signoff``).
"""

from __future__ import annotations

import asyncio
import difflib
import getpass
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from asyncroscopy import settings
from asyncroscopy.core.names import CALLER_HEADER, DEFAULT_MCP_URL

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.mcp_tools import McpServer, bearer_header
from . import runner, signoff
from .authoring import RULES, TEMPLATE, TEMPLATE_PARALLEL
from .blocks import describe_blocks
from .graph import Workflow
from .library import APPROVED, SHIPPED, Library, fingerprint_of, matched_calls, needs_approval, rehearsal_passed

MIN_REQUEST_CHARS = 12  # shorter than this is a placeholder and not a real request
MAX_WAIT_S = 100.0  # below the 120 s our MCP client allows for one call
RECORD_KEYS = ("run_id", "workflow", "state", "queue", "problems", "error", "questions", "approvals", "inputs", "outputs", "at_node", "cleanup", "unrestored",
               "calls", "tools_called", "started", "ended", "instrument_url", "preapproval", "results")
OUTPUT_CHARS = 1500
GOING = ("running", "queued")  # a run in one of these has a process driving it, or waiting to


def requester() -> str | None:
    """The caller named in the MCP request being served (X-Asyncroscopy-Caller), or None; a run's calls carry it."""
    try:
        from fastmcp.server.dependencies import get_http_headers

        value = get_http_headers(include_all=True).get(CALLER_HEADER.lower(), "")
    except Exception:  # no HTTP request: stdio, or a direct call in a test
        return None
    return value.strip() or None


class Endpoints:
    """Opens the instrument and analysis endpoints for a run and reads the endpoint's info.

    ``instruments``: the instrument endpoints, by URL, a run may also act on besides ``mcp_url`` (the default one).
    A URL outside these is refused, whoever sends it.
    """

    def __init__(self, mcp_url: str | None = None, caller: str = "workflow", instruments: list[str] | None = None) -> None:
        self.mcp_url, self.caller = mcp_url, caller
        first = mcp_url or settings.get().mcp_url or DEFAULT_MCP_URL
        self.instruments = list(dict.fromkeys([first, *(str(url) for url in instruments or [])]))

    def url(self, instrument_url: str | None = None) -> str:
        """The instrument endpoint to act on: ``instrument_url`` when it is one of ``instruments``, the first when None."""
        if instrument_url in (None, ""):
            return self.instruments[0]
        if str(instrument_url) not in self.instruments:
            raise LookupError(f"{instrument_url} is not an instrument endpoint this workflow server acts on; it acts on {self.instruments} "
                              "(agent.mcp_urls and agent.workflow_instruments in its config)")
        return str(instrument_url)

    async def open(self, run_id: str | None = None, caller: str | None = None, instrument_url: str | None = None,
                   run_key: str | None = None) -> tuple[McpServer, McpServer | None, dict]:
        """``caller``: the name the run's calls carry in the instrument's trace (this server's own name when None).
        ``run_key``: the secret a hardware run's calls carry with its run id, so its pre-approvals are its own."""
        config = AgentConfig.from_env(caller=caller or self.caller, mcp_url=self.url(instrument_url))
        if run_id:
            config.run_id = run_id
        config.run_key = run_key
        info = await config.info()
        return config.instrument(), config.analysis(), info

    def operator(self, info: dict) -> McpServer | str:
        """The operator endpoint of the stack ``info`` (its endpoint_info) describes, or why it cannot be used. It is
        called with ``ASYNCROSCOPY_OPERATOR_TOKEN`` (the ``preapprove`` role), never the agents' token."""
        url = (info.get("urls") or {}).get("operator")
        if not url:
            return "the stack reports no operator endpoint (endpoint_info.urls.operator)"
        env = settings.get()
        if not env.operator_token and env.mcp_token:
            return ("ASYNCROSCOPY_OPERATOR_TOKEN is not set, and the operator endpoint is never called with the agents' token "
                    "(ASYNCROSCOPY_MCP_TOKEN); give the workflow server a token with only the preapprove role")
        return McpServer(str(url), {CALLER_HEADER: self.caller, **bearer_header(env.operator_token)})


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
    elif state == "waiting" and record.get("approvals") and not record.get("questions"):
        out["next"] = ("a call is held until a person approves it (approvals lists it, with its request_id). Tell the user and end "
                       f"your turn. Once it is approved, call resume_workflow with run_id {record['run_id']}: the call is sent again and runs.")
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
        """Why this workflow may not run on this stack: a real instrument needs ``hardware`` and a reviewed workflow."""
        if info.get("simulated") is True:
            return None
        return runner.twin_only_problem(info, hardware=self.hardware) or self.library.hardware_problem(name, status, info.get("stack_id"))

    async def _rehearsed(self, run_id: str, info: dict, workflow: Workflow, running, operator: McpServer | str, granted: bool) -> dict:
        """Wait for ``run_id`` to end; save a twin run as a rehearsal and remove a hardware run's pre-approval."""
        try:
            record = await running
        except BaseException:
            if granted:
                await signoff.end(operator, run_id)
            if info.get("simulated") is True:
                signoff.forget_rehearsal(self.state_dir, run_id)
            raise
        if info.get("simulated") is True:
            results = await signoff.rehearsal_results(self.state_dir, record, operator)
            try:
                self.library.record_rehearsal(workflow, record, info, results)
            except OSError:  # the runner already saved the run so losing the rehearsal is harmless
                pass
        elif granted and record.get("state") != "waiting":
            await signoff.end(operator, run_id)
        return record

    async def _prepared(self, run_id: str, info: dict, status: str, workflow: Workflow, instrument_url: str | None):
        """Open the run's endpoints and take the twin's truth marker or grant the hardware run's pre-approvals."""
        operator = self.endpoints.operator(info)
        fields: dict[str, Any] = {"instrument_url": self.endpoints.url(instrument_url)}
        run_key = None
        if info.get("simulated") is True:
            await signoff.mark_rehearsal(self.state_dir, run_id, operator)
        elif status == APPROVED:
            run_key = signoff.new_run_key()
            fields["preapproval"] = await signoff.grant(operator, self.library, workflow.name, run_id, run_key, info)
        instrument, analysis, _ = await self.endpoints.open(run_id=run_id, caller=requester(), instrument_url=instrument_url, run_key=run_key)
        return instrument, analysis, operator, fields

    def _running(self) -> str | None:
        return next((run_id for run_id, task in self.tasks.items() if not task.done()), None)

    async def _wait(self, run_id: str, wait_s: float, with_tiles: bool = False) -> dict:
        """The run's reply once it ends or ``wait_s`` passes; placed tiles only ``with_tiles``."""
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
            try:
                return _reply(runner.read_run(run_id, self.state_dir))
            except KeyError:
                return _reply(task.result())
        try:
            record = runner.read_run(run_id, self.state_dir)
        except KeyError:
            if task is not None and not task.done():
                return _reply({"run_id": run_id, "state": "running"})
            if task is not None and task.exception() is not None:
                return {"run_id": run_id, "state": "failed", "error": f"{type(task.exception()).__name__}: {task.exception()}"}
            return {"error": f"there is no workflow run {run_id!r}; call list_runs"}
        if record["state"] in GOING and (task is None or task.done()) and not runner.driven_elsewhere(run_id, self.state_dir):
            record["state"], record["error"] = "stopped", "the workflow server was restarted while this run was going"
        return _reply(record, self.state_dir)

    async def _end_preapproval(self, record: dict) -> None:
        """Remove the pre-approval of a hardware run that was stopped while it waited (no driver is left to remove it)."""
        try:
            _, _, info = await self.endpoints.open(instrument_url=record.get("instrument_url"))
        except Exception:
            return
        await signoff.end(self.endpoints.operator(info), str(record["run_id"]))

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
        url = record.get("instrument_url")
        try:
            _, _, info = await self.endpoints.open(instrument_url=url)
        except LookupError as exc:
            return {"state": "refused", "problems": [str(exc)]}
        except Exception as exc:
            return {"error": f"the instrument endpoint could not be reached: {type(exc).__name__}: {exc}"}
        try:
            again, status = self._workflow(record["workflow"])
        except LookupError as exc:
            return {"state": "refused", "problems": [str(exc)]}
        problem = self._refusal(info, str(record["workflow"]), status)
        if problem:
            return {"state": "refused", "problems": [problem]}
        instrument, analysis, operator, fields = await self._prepared(run_id, info, status, again, url)
        runner.update_run(run_id, self.state_dir, **fields)
        granted = bool((fields.get("preapproval") or {}).get("granted"))
        self._start(run_id, lambda: self._rehearsed(run_id, info, again, runner.resume_workflow(
            run_id, instrument=instrument, analysis=analysis, state_dir=self.state_dir, find=self.library.workflow, answers=answers),
            operator, granted))
        return await self._wait(run_id, wait_s)

    async def instrument_info(self, instrument_url: str | None = None) -> dict | None:
        """The ``endpoint_info`` of the instrument endpoint ``instrument_url`` (the first one when None), or None when it
        cannot be reached. Raises LookupError for an endpoint this server does not act on."""
        self.endpoints.url(instrument_url)
        try:
            _, _, info = await self.endpoints.open(instrument_url=instrument_url)
        except Exception:
            return None
        return info or None

    async def stack_fingerprint(self, instrument_url: str | None = None) -> dict | None:
        """The ``stack_fingerprint`` of the instrument endpoint ``instrument_url``, or None (``instrument_info``)."""
        return (await self.instrument_info(instrument_url) or {}).get("stack_fingerprint")

    def review(self, name: str, info: dict | None = None) -> dict:
        """The sign-off card of ``name`` for the instrument ``info`` describes. Raises LookupError for no such workflow."""
        stack_fingerprint = (info or {}).get("stack_fingerprint")
        workflow, status = self._workflow(name)
        checked = workflow.lint()
        current = checked["hash"]
        approval = self.library.approval(workflow.name)
        rejection = self.library.rejection(workflow.name)
        asked = self.library.request(workflow.name)
        rehearsals = []
        for entry in self.library.rehearsals(workflow.name):
            shown = {k: entry.get(k) for k in ("run_id", "state", "at", "calls", "inputs", "error", "stack_id", "results")}
            shown["problems"] = list(entry.get("problems") or [])
            shown["this_code"] = entry.get("hash") == current
            shown["passed"] = rehearsal_passed(entry)
            shown["stack_fingerprint"] = fingerprint_of(entry)
            shown["this_stack"] = shown["stack_fingerprint"] is not None and shown["stack_fingerprint"] == (stack_fingerprint or {}).get("digest")
            try:
                run = runner.read_run(str(entry.get("run_id")), self.state_dir)
            except (KeyError, OSError, ValueError):
                run = None
            shown["outputs"] = _brief(run.get("outputs") or {}) if run is not None else None
            if run is not None and run.get("problems"):
                shown["problems"] = list(dict.fromkeys([*shown["problems"], *run["problems"]]))
            rehearsals.append(shown)
        passing, unrehearsed = self.library.rehearsal_problem(workflow.name, current, stack_fingerprint)
        rehearsed = bool(passing)
        latest = passing[-1] if passing else None
        signed = self.library.last_approved(workflow.name)
        source = workflow.source()
        expires = (approval or {}).get("expires")
        expired = expires is not None and time.time() > float(expires)
        if status == SHIPPED:
            refusal = f"{workflow.name} is shipped with the software; there is nothing to approve"
        elif status == APPROVED and not expired:
            refusal = f"{workflow.name} is already approved, as it is now"
        elif not checked["ok"]:
            refusal = f"{workflow.name} does not pass its checks: {checked['problems']}"
        elif unrehearsed:
            refusal = f"{workflow.name} as it is now: {unrehearsed}; run it there first"
        else:
            refusal = None
        return {
            "name": workflow.name, "status": status, "summary": workflow.summary, "inputs": dict(workflow.inputs), "budget": dict(workflow.budget),
            "graph": {k: v for k, v in workflow.blueprint().items() if k in ("nodes", "edges")}, "source": source,
            "hash": current, "acts": checked["acts"], "tools": checked["tools"], "blocks": checked["blocks"], "problems": checked["problems"],
            "request_as_reported_by_agent": {"text": asked.get("request"), "at": asked.get("at"), "this_code": asked.get("hash") == current} if asked else None,
            "context_as_reported_by_agent": {"text": asked["context"], "this_code": asked.get("hash") == current} if asked and asked.get("context") else None,
            "last_approved": {k: signed.get(k) for k in ("hash", "by", "at")} if signed else None,
            "diff_from_last_approved": "".join(difflib.unified_diff(signed["source"].splitlines(keepends=True), source.splitlines(keepends=True),
                                                                    "last approved", "now")) if signed else None,
            "stack_fingerprint": stack_fingerprint,
            "rehearsal": {**{k: latest.get(k) for k in ("run_id", "at", "inputs", "stack_id", "stack_fingerprint", "results")},
                          "calls_made": matched_calls(latest, info)} if latest else None,
            "approval": approval,
            "approval_expired": expired,
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

        async def review_workflow(name: str, instrument_url: str | None = None) -> dict:
            """Everything a person needs to decide whether a workflow may run on the real instrument, in one reply: its
            status, the user's request and the conversation's context as the agent reported them, the summary, inputs,
            graph and source, the difference from the code last approved, the latest passing rehearsal on a digital twin
            with its results (data keys, map, time, dose, twin score) and every call it made, every rehearsal, any
            approval or rejection. instrument_url: the instrument endpoint it would be approved for (one this server acts
            on; the first when left out), whose stack_fingerprint a rehearsal must match. Reads only."""
            try:
                return {"ok": True, "instrument_url": server.endpoints.url(instrument_url),
                        **server.review(name, await server.instrument_info(instrument_url))}
            except LookupError as exc:
                return {"ok": False, "error": str(exc)}

        async def approve_workflow(name: str, approver: str, expected_hash: str, rehearsal_run_id: str, instrument: str | None = None,
                                   days: float | None = None, instrument_url: str | None = None) -> dict:
            """A person approves a draft workflow to run on the real instrument. expected_hash: the hash review_workflow
            returned for the code the person read; refused if the draft is not that code now. rehearsal_run_id: the
            rehearsal.run_id review_workflow showed; refused unless it is still the rehearsal the review shows. The
            approval keeps that rehearsal, its calls matched to the instrument's tools as the review showed them, and a
            hardware run must use its inputs. approver: the person's name. instrument: the stack_id it is approved for
            (left out or "any": every instrument). days: how long the approval holds (left out: until revoked).
            instrument_url: as for review_workflow. Refused unless this exact code ran to the end, with every cleanup
            call succeeding, on a digital twin that matches that endpoint's stack_fingerprint now (the same fingerprint
            for a twin, the same software for a real instrument)."""
            try:
                info = await server.instrument_info(instrument_url)
                review = server.review(name, info)
            except LookupError as exc:
                return {"ok": False, "error": str(exc)}
            status = review["status"]
            if str(expected_hash) == review["hash"] and not review["can_approve"]:  # the one decision review_workflow reports
                return {"ok": False, "status": status, "error": review["why_not"]}
            try:
                record = server.library.approve(review["name"], approver, login=getpass.getuser(), instrument=instrument, days=days,
                                                require_rehearsal=True, via=via, expected_hash=str(expected_hash),
                                                instrument_info=info, rehearsal_run_id=str(rehearsal_run_id))
            except (KeyError, ValueError, PermissionError) as exc:
                return {"ok": False, "status": status, "error": str(exc.args[0])}
            return {"ok": True, "status": APPROVED, "approval": record}

        async def reject_workflow(name: str, reason: str, expected_hash: str) -> dict:
            """A person rejects a workflow, with the reason, so it is not approved as it is. expected_hash: the hash
            review_workflow returned for the code the person read; refused if the workflow is not that code now. An
            approved one loses its approval. The text is kept as a draft and still runs on a digital twin."""
            try:
                record = server.library.reject(name, reason, login=getpass.getuser(), via=via, expected_hash=str(expected_hash))
            except KeyError as exc:
                return {"ok": False, "error": f"there is no workflow named {name!r}" if not exc.args else str(exc.args[0])}
            except (ValueError, PermissionError) as exc:
                return {"ok": False, "error": str(exc.args[0])}
            return {"ok": True, "rejection": record}

        return [review_workflow, approve_workflow, reject_workflow]

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

        async def run_workflow(name: str, inputs: dict | None = None, wait_s: float = 60.0, instrument_url: str | None = None) -> dict:
            """Run a workflow by name. inputs: only the ones you want to change from their defaults, as {name: value};
            a value outside an input's range is refused, never adjusted. The workflow runs by itself, in the
            background; this waits up to wait_s (at most 100) and returns the result if it finished, otherwise
            state 'running' with a run_id for workflow_status. state 'waiting' means it asks the user a question
            ('questions'); 'next' always says what to do. One workflow acts on the instrument at a time: started
            while another is acting, this one is 'queued' and starts by itself when its turn comes. The result's
            outputs hold data keys (a stitched map's key is outputs.stitch.map). Inputs are checked against the
            workflow's declared inputs (an unknown name, a wrong type or a value out of range is refused) before
            anything is opened or run. instrument_url: leave it out unless the user or the app names the instrument
            endpoint to run on; only the endpoints this server was configured with are accepted."""
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
                _, _, info = await server.endpoints.open(instrument_url=instrument_url)
            except LookupError as exc:
                return {"state": "refused", "workflow": workflow.name, "problems": [str(exc)]}
            except Exception as exc:
                return {"state": "refused", "problems": [f"the instrument endpoint could not be reached: {type(exc).__name__}: {exc}"]}
            problem = server._refusal(info, workflow.name, status)
            if problem is None and status == APPROVED and info.get("simulated") is not True:
                problem = signoff.input_problem(server.library, workflow.name, workflow.check_inputs(inputs))
            if problem:
                return {"state": "refused", "workflow": workflow.name, "problems": [problem]}
            run_id = uuid.uuid4().hex[:12]
            instrument, analysis, operator, fields = await server._prepared(run_id, info, status, workflow, instrument_url)
            granted = bool((fields.get("preapproval") or {}).get("granted"))
            server._start(run_id, lambda: server._rehearsed(run_id, info, workflow, runner.run_workflow(
                workflow, dict(inputs or {}), instrument=instrument, analysis=analysis, run_id=run_id, state_dir=server.state_dir,
                fields=fields), operator, granted))
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
                stopped = runner.stop_workflow(str(run_id), server.state_dir)
            except KeyError as exc:
                return {"error": str(exc.args[0])}
            if stopped.get("state") == "stopped" and (stopped.get("preapproval") or {}).get("granted"):
                await server._end_preapproval(stopped)
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

        async def propose_workflow(user_request: str, source: str, context: str | None = None) -> dict:
            """Write a new workflow for a task the user gave you, for a scientist to approve. Use it when their
            task has many instrument steps, a loop, or steps at the same time and no workflow from list_workflows
            fits, or when they ask for a workflow. Not for a single action, and not for a task they did not ask
            for. You decide the steps from what you know of the task and the instrument; the scientist checks
            that it is safe for the hardware and worth running.
            user_request: the user's own words for the task, quoted as they wrote them. It is shown to the
            scientist who approves. context: what the conversation established (the sample, what was tried, what
            the user wants kept), shown with it; leave it out when there is nothing to add. source: the whole Python module as text, in the shape of list_blocks'
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
            result = server.library.propose(str(source), offered, offered_analysis, request=asked, context=context)
            if result["ok"]:
                result["next"] = (f"kept as a request named {result['name']}, waiting for a person. Tell the user it is there for them to "
                                  f"look at and approve: {needs_approval(result['name'])} Do not run it for a real task before that.")
            else:
                result["next"] = "nothing was kept. Fix what problems names and call propose_workflow again with the whole module"
            return result

        return [list_workflows, get_workflow, run_workflow, workflow_status, answer_workflow, stop_workflow, resume_workflow, list_runs,
                list_blocks, propose_workflow]
