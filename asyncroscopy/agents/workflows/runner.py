"""Run a workflow's LangGraph graph.

The runner adds these to a bare graph.

- **Checked first.** Before any call is made the graph is linted against the tools the endpoints offer now and
  the inputs are checked against their ranges.
- **Durable.** State is saved after every node in SQLite under ``state_dir``. A run that stopped, failed or whose
  process died continues from the node it was at with ``resume_workflow(run_id)``.
- **Stoppable.** ``stop_workflow(run_id)`` stops the run before its next tool call. It works from any process.
- **Cleans up.** The ``always`` tools of every block the workflow uses are called when the run ends for any reason.
  Blanking the beam is one. They cannot run if this process is killed.
- **Puts settings back.** A block that changes a setting for the run (``zoom``) notes it on the context. A run that
  fails, stops or waits puts it back, and a resumed run sets it again before it continues.
- **One driver at a time.** A run is driven under a lock file, so a run still going in another process is not
  resumed a second time.
- **Questions wait.** A block that asks a person stops the run as ``waiting``. Continue it with
  ``resume_workflow(run_id, answers={question id: answer})``.

Questions are LangGraph interrupts, so nothing stays open while the run waits and the wait survives
a restart.

Not covered here are a hook into the endpoint's e-stop, a watchdog that blanks the beam if the runner dies, and a
reservation that keeps other callers off the instrument.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, interrupt

from asyncroscopy.core.fileio import file_lock

from ..mcp_tools import McpServer
from .blocks import BlockContext, BlockFailed, Stopped, all_blocks
from .checks import switched_off_nodes
from .graph import Workflow, get_workflow, recursion_limit

DEFAULT_STATE_DIR = Path("outputs/workflows")
# Runs going on in this process by run id. ``progress`` reads it.
LIVE: dict[str, BlockContext] = {}
TWIN_ONLY = ("this endpoint does not say its stack is simulated. Workflows run on a digital twin unless hardware is allowed "
             "on purpose: --hardware on the command line (run, resume, serve).")


def twin_only_problem(endpoint_info: dict, hardware: bool = False) -> str | None:
    """Why a workflow may not run here, or None.

    A stack that does not say it is simulated is only driven when the person starting the run sets ``hardware``.
    """
    return None if hardware or endpoint_info.get("simulated") is True else TWIN_ONLY


def progress(run_id: str) -> dict | None:
    """What a run in this process has done so far, or None when it is not running here."""
    ctx = LIVE.get(run_id)
    if ctx is None:
        return None
    acquisitions = sum(1 for c in ctx.calls if "acquire" in c["tool"] and not c["error"])
    out = {"tool_calls": len(ctx.calls), "acquisitions": acquisitions, "last_call": ctx.calls[-1]["tool"] if ctx.calls else None}
    if ctx.step:
        out["step"] = ctx.step
    if ctx.loop:
        out["loop"] = dict(ctx.loop)
    image = latest_image(ctx.calls)
    if image:
        out["latest_image"] = image
    return out


def latest_image(calls: list[dict]) -> str | None:
    """The data key of the newest image a run acquired (its reply starts with the key), or None."""
    for call in reversed(calls):
        if call.get("error") or "acquire" not in call["tool"] or "image" not in call["tool"]:
            continue
        first = str(call.get("result") or "").strip().split("\n", 1)[0].strip()
        if first and " " not in first and "." in first:
            return first
    return None


def _record_path(state_dir: Path, run_id: str) -> Path:
    return state_dir / "runs" / f"{run_id}.json"


def _stop_path(state_dir: Path, run_id: str) -> Path:
    return state_dir / "runs" / f"{run_id}.stop"


def read_run(run_id: str, state_dir: str | Path = DEFAULT_STATE_DIR) -> dict:
    """The record of a run with its workflow, inputs, state and latest outputs. Raises KeyError for an unknown run."""
    path = _record_path(Path(state_dir), run_id)
    if not path.exists():
        raise KeyError(f"no workflow run {run_id!r} under {state_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(state_dir: str | Path = DEFAULT_STATE_DIR) -> list[dict]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in (Path(state_dir) / "runs").glob("*.json")]
    shown = ("run_id", "workflow", "state", "started", "ended", "at_node")
    brief = [{k: r.get(k) for k in shown} for r in runs]
    return sorted(brief, key=lambda r: r["started"] or 0, reverse=True)


def stop_workflow(run_id: str, state_dir: str | Path = DEFAULT_STATE_DIR) -> dict:
    """Ask a run to stop before its next tool call. This works from another process and the cleanup still runs.

    A run waiting for a person's answer is not running, so nothing would read the stop request: it is marked stopped
    here. resume_workflow continues it later, and its question is asked again.
    """
    record = read_run(run_id, state_dir)
    _stop_path(Path(state_dir), run_id).write_text(str(time.time()), encoding="utf-8")
    if record.get("state") == "waiting":
        record.pop("questions", None)
        record.update(state="stopped", error="stopped by the operator while it waited for an answer", ended=time.time())
        _record_path(Path(state_dir), run_id).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        return record
    return {"run_id": run_id, "stopping": True}


@contextlib.asynccontextmanager
async def _checkpointer(state_dir: Path, durable: bool):
    if not durable:
        yield InMemorySaver(), False
        return
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:  # without langgraph-checkpoint-sqlite the run works but cannot be resumed
        yield InMemorySaver(), False
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(state_dir / "checkpoints.sqlite")) as saver:
        yield saver, True


async def _tool_names(server: McpServer | None) -> list[str] | None:
    if server is None:
        return None
    return sorted(t.name for t in await server.list_tools())


async def run_workflow(
    source: str | Workflow, inputs: dict | None = None, *, instrument: McpServer, analysis: McpServer | None = None,
    run_id: str | None = None, state_dir: str | Path = DEFAULT_STATE_DIR, durable: bool = True,
) -> dict:
    """Run a workflow from its start. ``source`` is its name or the ``Workflow``. Returns the run's record.

    The record's ``state`` is done, waiting, stopped or failed. It is refused when a check before the first
    call found a problem.
    """
    workflow = get_workflow(source)
    state_dir = Path(state_dir)
    run_id = run_id or uuid.uuid4().hex[:12]
    unchecked = workflow.lint()
    record = {"run_id": run_id, "workflow": workflow.name, "hash": unchecked["hash"], "inputs": dict(inputs or {}),
              "state": "refused", "started": time.time(), "ended": None, "outputs": {}, "calls": 0, "problems": []}
    if unchecked["hash"] is None:  # the graph could not be built
        record["problems"] = list(unchecked["problems"])
        return record
    record["blueprint"] = workflow.blueprint()
    try:
        offered = await _tool_names(instrument)
        needs_analysis = bool(unchecked.get("needs_analysis"))
        offered_analysis = await _tool_names(analysis) if needs_analysis else None
    except Exception as exc:
        record["problems"] = [f"the endpoint could not be reached to check the workflow against its tools: {type(exc).__name__}: {exc}"]
        return record
    input_problem = None
    try:
        record["inputs"] = workflow.check_inputs(inputs)
    except ValueError as exc:
        input_problem = str(exc)
    # A node this run switches off (eels: false) needs no tool: a map without EELS runs where there is none.
    off = switched_off_nodes(record["blueprint"], record["inputs"]) if input_problem is None else frozenset()
    checked = workflow.lint(offered, offered_analysis, switched_off=off)
    record["problems"] = list(checked["problems"])
    if needs_analysis and analysis is None:
        record["problems"].append("this workflow needs an analysis server, and none is configured (mcp.analysis_url)")
    if input_problem:
        record["problems"].append(input_problem)
    if record["problems"]:
        return record
    return await _drive(record, workflow, state_dir, instrument, analysis, durable, start=True)


def check_answers(record: dict, answers: dict | None) -> dict:
    """Check the answers for a waiting run and return them as {question id: answer}.

    Raises ValueError when the run is waiting and no answer is given, when an id is not one of its questions, or
    when an answer is not one of a question's options.
    """
    answers = dict(answers or {})
    questions = {str(q.get("id")): q for q in record.get("questions") or []}
    if record.get("state") == "waiting" and not answers:
        asked = "; ".join(f"{q.get('question')} (id {i})" for i, q in questions.items())
        raise ValueError(f"run {record.get('run_id')} is waiting for a person's answer: {asked}")
    checked = {}
    for qid, answer in answers.items():
        question = questions.get(str(qid))
        if question is None:
            raise ValueError(f"run {record.get('run_id')} has no question {qid!r}; it is waiting for: {sorted(questions) or 'nothing'}")
        options = [str(o) for o in question.get("options") or []]
        if options:
            chosen = next((o for o in options if o.strip().lower() == str(answer).strip().lower()), None)
            if chosen is None:
                raise ValueError(f"the answer to {question.get('question')!r} must be one of {options}, not {str(answer)[:200]!r}")
            answer = chosen
        checked[str(qid)] = answer
    return checked


async def resume_workflow(
    run_id: str, *, instrument: McpServer, analysis: McpServer | None = None, state_dir: str | Path = DEFAULT_STATE_DIR,
    find: Callable[[str], Workflow] = get_workflow, answers: dict | None = None,
) -> dict:
    """Continue a stopped, failed or waiting run, or one whose process died, from the node it was at.

    ``find`` looks the run's workflow up by name. Pass the library's for a workflow that is not shipped.
    ``answers`` is {question id: answer} for a run waiting for a person. The ids are in the record's ``questions``.
    A question left out keeps waiting.
    """
    state_dir = Path(state_dir)
    record = read_run(run_id, state_dir)
    if record["state"] == "done":
        return record
    if record["state"] == "refused":
        raise ValueError(f"run {run_id} never started: {record['problems']}")
    answers = check_answers(record, answers)
    try:
        workflow = find(str(record["workflow"]))
    except KeyError as exc:
        raise ValueError(f"run {run_id} is of a workflow that is no longer here: {exc.args[0]}") from exc
    if workflow.hash() != record["hash"]:
        raise ValueError(f"workflow {workflow.name} (its code, or a block it uses) has changed since run {run_id} started; "
                         "the run is not resumed. Start a new run.")
    _stop_path(state_dir, run_id).unlink(missing_ok=True)
    return await _drive(record, workflow, state_dir, instrument, analysis, durable=True, start=False, answers=answers)


async def _drive(record: dict, workflow: Workflow, state_dir: Path, instrument: McpServer, analysis: McpServer | None, durable: bool,
                 start: bool, answers: dict | None = None) -> dict:
    blueprint, run_id = record["blueprint"], record["run_id"]
    budget = blueprint.get("budget") or {}
    # The budget covers the whole run, so a resumed run gets only what is left. Time spent waiting does not count.
    record["calls_before"] = int(record.get("calls", 0)) if not start else 0
    record["seconds_before"] = float(record.get("seconds", 0.0)) if not start else 0.0
    record["tools_called_before"] = dict(record.get("tools_called") or {}) if not start else {}
    began = time.monotonic()
    max_calls = budget.get("max_tool_calls")
    ctx = BlockContext(
        instrument, analysis, stop_file=_stop_path(state_dir, run_id),
        max_tool_calls=int(max_calls) - record["calls_before"] if max_calls else None,
        deadline=began + float(budget["max_seconds"]) - record["seconds_before"] if budget.get("max_seconds") else None,
        on_ask=lambda question: interrupt({"ask": question}),
    )
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": recursion_limit(blueprint)}
    used = [all_blocks()[b["block"]] for b in blueprint["nodes"].values()]
    ctx.acts = any(block.acts for block in used)

    def save(**changes: Any) -> None:
        record.update(changes, calls=record["calls_before"] + len(ctx.calls),
                      seconds=round(record["seconds_before"] + time.monotonic() - began, 3),
                      tools_called=tools_called(record["tools_called_before"], ctx.calls))
        path = _record_path(state_dir, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        kept = {k: v for k, v in record.items() if k not in ("calls_before", "seconds_before", "tools_called_before")}
        path.write_text(json.dumps(kept, indent=1, default=str), encoding="utf-8")

    with _run_lock(state_dir, run_id):
        if not start:  # another process may have moved the run on between reading its record and taking the lock
            latest = read_run(run_id, state_dir)
            if latest.get("state") == "done":
                return latest
        # One run at a time drives the instrument: on the twin a second map found the beam blanked by the first one's
        # tiles and failed; on an instrument the two would move the stage and the scan under each other.
        driving = [other for other, live in LIVE.items() if other != run_id and live.acts]
        if ctx.acts and driving:
            record["problems"] = [f"workflow run {driving[0]} is driving the instrument; wait for it (workflow_status) or stop it "
                                  "(stop_workflow), then start this one"]
            record["state"] = "refused"
            return record
        LIVE[run_id] = ctx
        try:
            return await _drive_checked(record, workflow, state_dir, ctx, config, used, save, durable, start, answers or {})
        finally:
            LIVE.pop(run_id, None)


@contextlib.contextmanager
def _run_lock(state_dir: Path, run_id: str):
    """Hold a run's lock while it is driven, so no two processes (or two calls in one) drive it at once.

    The operating system lets go of the lock when the process dies, so a run whose process died can be resumed.
    """
    path = state_dir / "runs" / f"{run_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        try:
            lock = file_lock(handle, deadline=time.monotonic())
            lock.__enter__()
        except TimeoutError:
            raise ValueError(f"run {run_id} is being run now by another process or call; stop it (stop_workflow) "
                             "and wait for it to end before resuming it") from None
        try:
            yield
        finally:
            lock.__exit__(None, None, None)


def tools_called(before: dict, calls: list[dict]) -> dict:
    """What a run called, one entry per tool: {"ok", "failed", "last_args", "last_result"}.

    It stays small however long the run is (a map makes thousands of calls), and it is enough for the agent that
    started the run to count the run's acquisitions and moves as its own (``asyncroscopy.agents.goals``).
    """
    out = {name: dict(entry) for name, entry in (before or {}).items()}
    for call in calls:
        entry = out.setdefault(call["tool"], {"ok": 0, "failed": 0})
        entry["failed" if call.get("error") else "ok"] += 1
        if not call.get("error"):
            entry["last_args"], entry["last_result"] = call.get("args") or {}, str(call.get("result", ""))[:200]
    return out


def _pending(snapshot: Any) -> list[dict]:
    """The questions a stopped graph is waiting on a person for, each with its interrupt id."""
    questions = []
    for task in snapshot.tasks:
        for item in task.interrupts:
            value = item.value if isinstance(item.value, dict) else {}
            if isinstance(value.get("ask"), dict):
                questions.append({"id": item.id, "node": task.name, **value["ask"]})
    return questions


async def _drive_checked(record: dict, workflow: Workflow, state_dir: Path, ctx: BlockContext, config: dict, used: list, save, durable: bool,
                         start: bool, answers: dict) -> dict:
    run_id = record["run_id"]
    async with _checkpointer(state_dir, durable) as (saver, is_durable):
        graph = workflow.compile(ctx, saver)
        if start:
            first: Any = {"inputs": record["inputs"], "outputs": {}, "visits": {}}
        else:
            snapshot = await graph.aget_state(config)
            if not snapshot.next:
                raise ValueError(f"run {run_id} has no saved place to continue from (it was not run with a durable checkpoint)")
            questions = _pending(snapshot)
            # Each waiting node is resumed by its own id. A question with no answer yet keeps waiting.
            answered = [q for q in questions if q["id"] in answers]
            resume = {q["id"]: answers[q["id"]] for q in answered}
            first = Command(resume=resume) if resume else None
            if answers:
                record["answered"] = [*(record.get("answered") or []),
                                      *({"node": q["node"], "question": q.get("question"), "answer": answers[q["id"]], "at": time.time()}
                                        for q in answered)]
        save(state="running", durable=is_durable, ended=None, error=None, questions=None, unrestored=None)
        state, error = "done", None
        try:
            # Settings the run had changed were put back when it ended early. Set them again before it continues.
            ctx.restores = {name: dict(entry) for name, entry in (record.get("restores") or {}).items()} if not start else {}
            for entry in ctx.restores.values():
                await ctx.call(entry["set"]["tool"], entry["set"]["arguments"])
            await graph.ainvoke(first, config)
        except Stopped as exc:
            state, error = "stopped", str(exc)
        except (BlockFailed, GraphRecursionError) as exc:
            state, error = "failed", str(exc)
        except Exception as exc:  # a bug in a block must not skip the cleanup or the record
            state, error = "failed", f"{type(exc).__name__}: {exc}"
        snapshot = await graph.aget_state(config)
        questions = _pending(snapshot)
        if state == "done" and questions:
            state = "waiting"
        cleanup = []
        for pattern in sorted({p for block in used for p in block.always}):
            try:
                await ctx.call_always(pattern)
                cleanup.append({"tool": pattern, "ok": True})
            except Exception as exc:  # the operator must see a cleanup that failed
                cleanup.append({"tool": pattern, "ok": False, "error": str(exc)[:300]})
        if state != "done":  # the run ended part way: put back what it changed and has not put back itself
            for name, entry in ctx.restores.items():
                restore = entry["restore"]
                try:
                    await ctx.call_always(restore["tool"], restore["arguments"])
                    cleanup.append({"tool": restore["tool"], "ok": True, "restored": name})
                except Exception as exc:
                    ctx.unrestored.append({"tool": restore["tool"], "arguments": dict(restore["arguments"]), "error": str(exc)[:300]})
        values = snapshot.values or {}
        save(state=state, error=error, questions=questions or None, ended=time.time(), outputs=values.get("outputs", {}),
             visits=values.get("visits", {}), at_node=list(snapshot.next), cleanup=cleanup, unrestored=list(ctx.unrestored) or None,
             restores=(dict(ctx.restores) or None) if state != "done" else None)
    record.pop("calls_before", None)
    record.pop("seconds_before", None)
    return record
