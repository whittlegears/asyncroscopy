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
- **One run on the instrument at a time.** A run whose workflow acts on the instrument queues for it at the
  endpoint (``turns``), as state ``queued``, behind runs of any process, and gives it back when it has no acting
  block left to run. A run that only reads or analyses does not queue.
- **Questions wait.** A block that asks a person stops the run as ``waiting``. Continue it with
  ``resume_workflow(run_id, answers={question id: answer})``.
- **Held calls wait.** A call the endpoint holds for a person's approval (``approval_required``) stops the run as
  ``waiting``, with the held call in ``approvals``. Once a person approves it, ``resume_workflow(run_id)`` sends it
  again, and it runs.
- **Calls are not sent twice.** The waiting node runs again from its first line on resume. The calls it completed
  before it stopped get their recorded replies from the run's call journal (``blocks.CallJournal``,
  ``runs/<run_id>.journal.json``) instead of being sent again, unless something ran in between that may have undone
  them (its cleanup, the ``always`` tools). The journal is removed when the run ends or is stopped.

Questions and held calls are LangGraph interrupts, so nothing stays open while the run waits and the wait survives
a restart.

Not covered here are a hook into the endpoint's e-stop, a watchdog that blanks the beam if the runner dies, and a
reservation that keeps other callers off the instrument.
"""

from __future__ import annotations

import asyncio
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
from asyncroscopy.safety.permissions import canonical_json

from asyncroscopy.agents.mcp_tools import McpServer
from .blocks import BlockContext, BlockFailed, CallJournal, Stopped, all_blocks
from .checks import switched_off_nodes
from .turns import Turn, acting_ahead
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


PROGRESS_EVERY_S = 1.0  # how often a run's progress file is rewritten
MAX_TILES_SHOWN = 600  # the most placed tiles a progress reply lists


def progress(run_id: str, state_dir: str | Path | None = None) -> dict | None:
    """What a run has done so far, or None when nothing is known.

    A run in this process is read from memory. With ``state_dir``, one that another process drives (the agent server
    started it, the analysis host is asked) is read from the progress file that process keeps beside the run record.
    """
    ctx = LIVE.get(run_id)
    if ctx is not None:
        return _live_progress(ctx)
    if state_dir is None:
        return None
    try:
        found = json.loads(_progress_path(Path(state_dir), run_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def driven_elsewhere(run_id: str, state_dir: str | Path) -> bool:
    """Whether another process holds the run's lock now, so the run is still going there (``_run_lock``)."""
    if not (Path(state_dir) / "runs" / f"{run_id}.lock").exists():
        return False
    try:
        with _run_lock(Path(state_dir), run_id):
            return False
    except ValueError:
        return True


def _progress_path(state_dir: Path, run_id: str) -> Path:
    return state_dir / "runs" / f"{run_id}.progress.json"


def _write_progress(ctx: BlockContext, path: Path) -> None:
    """Replace the progress file whole, so a reader never sees half of it."""
    staged = path.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
    staged.write_text(json.dumps(_live_progress(ctx), default=str), encoding="utf-8")
    try:
        staged.replace(path)
    except OSError:  # a reader holds the file open (Windows): this update is skipped, the next one replaces it
        staged.unlink(missing_ok=True)


async def _keep_progress(ctx: BlockContext, path: Path) -> None:
    """Rewrite a run's progress file every ``PROGRESS_EVERY_S`` until cancelled. A failed write never stops the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with contextlib.suppress(OSError):
            _write_progress(ctx, path)
        await asyncio.sleep(PROGRESS_EVERY_S)


def _live_progress(ctx: BlockContext) -> dict:
    acquisitions = sum(1 for c in ctx.calls if "acquire" in c["tool"] and not c["error"])
    out = {"tool_calls": len(ctx.calls), "acquisitions": acquisitions, "last_call": ctx.calls[-1]["tool"] if ctx.calls else None}
    if ctx.step:
        out["step"] = ctx.step
    if ctx.loop:
        out["loop"] = dict(ctx.loop)
    image = latest_image(ctx.calls)
    if image:
        out["latest_image"] = image
    if ctx.tiles:  # the newest MAX_TILES_SHOWN, so a status reply stays small on a large map
        out["tiles"] = list(ctx.tiles.values())[-MAX_TILES_SHOWN:]
        out["tiles_total"] = len(ctx.tiles)
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


def _journal_path(state_dir: Path, run_id: str) -> Path:
    return state_dir / "runs" / f"{run_id}.journal.json"


def read_run(run_id: str, state_dir: str | Path = DEFAULT_STATE_DIR) -> dict:
    """The record of a run with its workflow, inputs, state and latest outputs. Raises KeyError for an unknown run."""
    path = _record_path(Path(state_dir), run_id)
    if not path.exists():
        raise KeyError(f"no workflow run {run_id!r} under {state_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_run(run_id: str, state_dir: str | Path = DEFAULT_STATE_DIR, **fields: Any) -> dict:
    """Set ``fields`` in a run's record before it is resumed (the workflow server's new ``preapproval``)."""
    record = read_run(run_id, state_dir)
    record.update(fields)
    _record_path(Path(state_dir), run_id).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record


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
        record.pop("approvals", None)
        record.update(state="stopped", error="stopped by the operator while it waited for an answer", ended=time.time())
        _record_path(Path(state_dir), run_id).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        _journal_path(Path(state_dir), run_id).unlink(missing_ok=True)
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
    run_id: str | None = None, state_dir: str | Path = DEFAULT_STATE_DIR, durable: bool = True, fields: dict | None = None,
) -> dict:
    """Run a workflow from its start. ``source`` is its name or the ``Workflow``. Returns the run's record.

    The record's ``state`` is done, waiting, stopped or failed. It is refused when a check before the first
    call found a problem. ``fields`` are kept in the record as they are: the workflow server's ``instrument_url`` and
    ``preapproval``.
    """
    workflow = get_workflow(source)
    state_dir = Path(state_dir)
    run_id = run_id or uuid.uuid4().hex[:12]
    unchecked = workflow.lint()
    record = {"run_id": run_id, "workflow": workflow.name, "hash": unchecked["hash"], "inputs": dict(inputs or {}),
              "state": "refused", "started": time.time(), "ended": None, "outputs": {}, "calls": 0, "problems": [], **dict(fields or {})}
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
        record["problems"].append("this workflow needs an analysis server, and none is configured (mcp.urls.analysis)")
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
    if record.get("state") == "waiting" and not answers and not record.get("approvals"):
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
    record["calls_made_before"] = list(record.get("calls_made") or []) if not start else []
    record["data_keys_before"] = list(record.get("data_keys") or []) if not start else []
    began = time.monotonic()
    max_calls = budget.get("max_tool_calls")
    ctx = BlockContext(
        instrument, analysis, stop_file=_stop_path(state_dir, run_id),
        max_tool_calls=int(max_calls) - record["calls_before"] if max_calls else None,
        deadline=began + float(budget["max_seconds"]) - record["seconds_before"] if budget.get("max_seconds") else None,
        on_ask=lambda question: interrupt({"ask": question}),
        on_hold=lambda held: interrupt({"approval": held}),
        journal=CallJournal(_journal_path(state_dir, run_id)),
    )
    if start:
        ctx.journal.clear()
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": recursion_limit(blueprint)}
    used = [all_blocks()[b["block"]] for b in blueprint["nodes"].values()]
    ctx.acts = any(block.acts for block in used)

    def save(**changes: Any) -> None:
        record.update(changes, calls=record["calls_before"] + len(ctx.calls),
                      seconds=round(record["seconds_before"] + time.monotonic() - began, 3),
                      tools_called=tools_called(record["tools_called_before"], ctx.calls),
                      calls_made=calls_made(record["calls_made_before"], ctx.calls),
                      data_keys=list(dict.fromkeys([*record["data_keys_before"], *ctx.data_keys])))
        record["results"] = results_of(record)
        path = _record_path(state_dir, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        kept = {k: v for k, v in record.items() if not k.endswith("_before")}
        path.write_text(json.dumps(kept, indent=1, default=str), encoding="utf-8")

    with _run_lock(state_dir, run_id):
        if not start:  # another process may have moved the run on between reading its record and taking the lock
            latest = read_run(run_id, state_dir)
            if latest.get("state") == "done":
                return latest
        # One run at a time drives the instrument: on the twin a second map found the beam blanked by the first one's
        # tiles and failed; on an instrument the two would move the stage and the scan under each other.
        # Where the endpoint gives turns (take_turn), a run that acts queues for the instrument there instead, behind
        # runs of any process; without them, a second acting run in this process is refused.
        turn = Turn(instrument, what=f"{workflow.name}, run {run_id}") if ctx.acts and await Turn.offered(instrument) else None
        driving = [other for other, live in LIVE.items() if other != run_id and live.acts]
        if turn is None and ctx.acts and driving:
            record["problems"] = [f"workflow run {driving[0]} is driving the instrument; wait for it (workflow_status) or stop it "
                                  "(stop_workflow), then start this one"]
            record["state"] = "refused"
            return record
        LIVE[run_id] = ctx
        shown = _progress_path(state_dir, run_id)
        keeper = asyncio.get_running_loop().create_task(_keep_progress(ctx, shown))
        try:
            if turn is not None:
                try:
                    await turn.wait(on_queued=lambda queue: save(state="queued", queue=queue), stopped=ctx.stop_file.exists)
                except Stopped as exc:
                    save(state="stopped", error=str(exc), ended=time.time(), queue=None)
                    return record
                record["queue"] = None

                async def clean_up() -> None:
                    ctx.cleaned = await _clean_up(ctx, used)

                # The turn is given back when no block that acts can still run: a map is stitched while the next one starts.
                all_known = all_blocks()
                turn.keep(done_acting=lambda: ctx.step is not None and not acting_ahead(blueprint, all_known, ctx.step), before_giving=clean_up)
            return await _drive_checked(record, workflow, state_dir, ctx, config, used, save, durable, start, answers or {})
        finally:
            if turn is not None:
                await turn.give()
            LIVE.pop(run_id, None)
            keeper.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await keeper
            with contextlib.suppress(OSError):  # the record now says how the run ended
                shown.unlink(missing_ok=True)


async def _clean_up(ctx: BlockContext, used: list) -> list[dict]:
    """Call the ``always`` tools of the blocks a run uses (blanking the beam is one); returns what each did."""
    cleanup = []
    for pattern in sorted({p for block in used for p in block.always}):
        try:
            await ctx.call_always(pattern)
            cleanup.append({"tool": pattern, "ok": True})
        except Exception as exc:  # the operator must see a cleanup that failed
            cleanup.append({"tool": pattern, "ok": False, "error": str(exc)[:300]})
    return cleanup


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


def calls_made(before: list[dict], calls: list[dict]) -> list[dict]:
    """Every distinct call of a run that succeeded, first made first: {"server" (instrument or analysis), "tool",
    "arguments" (as canonical JSON: sorted keys), "count"}. A rehearsal's are what a sign-off covers on the instrument."""
    out = {(c["server"], c["tool"], canonical_json(c["arguments"])): dict(c) for c in before}
    for call in calls:
        if call.get("error"):
            continue
        key = (call.get("server", "instrument"), call["tool"], canonical_json(call.get("args")))
        entry = out.setdefault(key, {"server": key[0], "tool": key[1], "arguments": json.loads(key[2]), "count": 0})
        entry["count"] += 1
    return list(out.values())


def results_of(record: dict) -> dict:
    """What a run produced, in the same shape for a rehearsal on a twin and a run on an instrument, so the two can be
    set side by side: every data key it acquired or derived, the stitched map's key, its instrument calls and its time.
    ``wall_s`` is the time the run was driven (time spent waiting for a person is not counted)."""
    nodes = (record.get("blueprint") or {}).get("nodes") or {}
    outputs = record.get("outputs") or {}
    maps = [str(outputs[name]["map"]) for name, node in nodes.items()
            if node.get("block") == "stitch_map" and isinstance(outputs.get(name), dict) and outputs[name].get("map")]
    made = record.get("calls_made") or []
    return {"data_keys": list(dict.fromkeys([*(record.get("data_keys") or []), *maps])), "map": maps[-1] if maps else None,
            "instrument_calls": sum(c["count"] for c in made if c["server"] == "instrument"),
            "distinct_instrument_calls": sum(1 for c in made if c["server"] == "instrument"), "wall_s": record.get("seconds")}


def _held(snapshot: Any) -> list[dict]:
    """The calls a stopped graph is waiting on a person's approval of, each with its interrupt id."""
    held = []
    for task in snapshot.tasks:
        for item in task.interrupts:
            value = item.value if isinstance(item.value, dict) else {}
            if isinstance(value.get("approval"), dict):
                held.append({"id": item.id, "node": task.name, **value["approval"]})
    return held


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
        save(state="running", durable=is_durable, ended=None, error=None, questions=None, approvals=None, unrestored=None)
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
        held = _held(snapshot)
        if state == "done" and (questions or held):
            state = "waiting"
        if state != "waiting":
            ctx.journal.clear()
        # A run that gave the instrument back early cleaned up then, while it was still its own: a blank now would
        # land in the middle of the next run's tile.
        cleanup = list(ctx.cleaned) if ctx.cleaned is not None else await _clean_up(ctx, used)
        if state != "done":  # the run ended part way: put back what it changed and has not put back itself
            for name, entry in ctx.restores.items():
                restore = entry["restore"]
                try:
                    await ctx.call_always(restore["tool"], restore["arguments"])
                    cleanup.append({"tool": restore["tool"], "ok": True, "restored": name})
                except Exception as exc:
                    ctx.unrestored.append({"tool": restore["tool"], "arguments": dict(restore["arguments"]), "error": str(exc)[:300]})
        values = snapshot.values or {}
        save(state=state, error=error, questions=questions or None, approvals=held or None, ended=time.time(), outputs=values.get("outputs", {}),
             visits=values.get("visits", {}), at_node=list(snapshot.next), cleanup=cleanup, unrestored=list(ctx.unrestored) or None,
             restores=(dict(ctx.restores) or None) if state != "done" else None)
    for key in [k for k in record if k.endswith("_before")]:
        record.pop(key)
    return record
