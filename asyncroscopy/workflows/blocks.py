"""Blocks are the steps a workflow is built from.

A block declares its inputs, its outputs and the tools it calls. Tool names are glob patterns, as in skills. With
these a workflow can be checked before anything runs. A block's ``run`` reaches the instrument only through
``BlockContext.call``.
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asyncroscopy.core.fileio import write_atomic
from asyncroscopy.core.names import DATA_KEY_IN_TEXT, reply_keys
from asyncroscopy.safety.permissions import json_sha256

from asyncroscopy.agents.mcp_tools import McpServer, held_for_approval, result_text

TYPES = {"number": (int, float), "integer": (int,), "string": (str,), "boolean": (bool,), "list": (list,), "object": (dict,), "any": (object,)}


class BlockFailed(RuntimeError):
    """A tool call failed or a block could not do its work. The message says which call and why."""


class Stopped(RuntimeError):
    """The run was stopped or ran out of budget before this call was sent."""


class Unreachable(BlockFailed):
    """A tool call never reached its endpoint: the connection was refused or timed out before the request was sent.

    The call was not delivered, so it cannot have acted, and the runner tries the node again (``graph.TRANSPORT_RETRY``).
    """


def _undelivered_errors() -> tuple[type[BaseException], ...]:
    """Transport errors meaning a request was never sent, from httpx and fastmcp 4's httpx2."""
    found: list[type[BaseException]] = []
    for name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        found += [module.ConnectError, module.ConnectTimeout, module.PoolTimeout]
    return tuple(found)


UNDELIVERED = _undelivered_errors()


def undelivered(exc: BaseException) -> bool:
    """Whether ``exc``, or an exception it was raised from, says the request never reached the endpoint."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, UNDELIVERED):
            return True
        if isinstance(exc, BaseExceptionGroup):
            return all(undelivered(inner) for inner in exc.exceptions)
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


@dataclass(frozen=True)
class Block:
    """One step a workflow can use.

    ``inputs`` maps a name to {"type", "default", "doc"}. An input with no default is required.
    ``outputs`` maps a name to its type.
    ``tools`` are glob patterns of the instrument tools it may call.
    ``analysis_tools`` are the analysis server tools it may call.
    ``always`` are instrument tools the runner calls when the run ends for any reason, such as blanking the beam.
    ``acts`` is True when the block changes the instrument.
    ``run(ctx, args, previous)`` does the work. ``previous`` is this node's output from its last visit, or None on
    the first. A block visited in a loop uses it to keep its place.
    ``tool_input`` names the input that holds a tool name, for a block like ``run_tool``. Lint reads it so the tools
    a workflow can call are known before it runs.
    """

    name: str
    summary: str
    run: Callable[[BlockContext, dict, dict | None], Awaitable[dict]] = field(repr=False, compare=False)
    inputs: dict[str, dict] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    analysis_tools: tuple[str, ...] = ()
    always: tuple[str, ...] = ()
    acts: bool = False
    tool_input: str | None = None

    def describe(self) -> dict:
        return {"name": self.name, "summary": self.summary, "acts": self.acts, "inputs": self.inputs, "outputs": self.outputs,
                "tools": list(self.tools) + list(self.analysis_tools)}


_BLOCKS: dict[str, Block] = {}


def register_block(block: Block) -> Block:
    if block.name in _BLOCKS and _BLOCKS[block.name] is not block:
        raise ValueError(f"a block named {block.name!r} is already registered")
    for port, declared in block.inputs.items():
        if declared.get("type", "any") not in TYPES:
            raise ValueError(f"block {block.name}: input {port} has unknown type {declared.get('type')!r}")
    _BLOCKS[block.name] = block
    return block


def all_blocks() -> dict[str, Block]:
    from . import measure_blocks, shipped_blocks, survey_blocks  # noqa: F401  importing registers them

    return dict(_BLOCKS)


def get_block(name: str) -> Block:
    blocks = all_blocks()
    if name not in blocks:
        raise KeyError(f"no block named {name!r}; blocks: {sorted(blocks)}")
    return blocks[name]


def describe_blocks() -> list[dict]:
    """Every block with its inputs and outputs, sorted by name."""
    return [b.describe() for _, b in sorted(all_blocks().items())]


def parse_reply(text: str) -> Any:
    """Turn a tool's reply into a value.

    JSON is parsed. An acquisition reply starts with a data key on its first line, and that key is returned.
    Anything else is returned as text.
    """
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        first = text.partition("\n")[0].strip()
        return first if DATA_KEY_IN_TEXT.fullmatch(first) else text
    if isinstance(value, dict) and set(value) == {"result"}:
        return value["result"]
    return value


# What AutoScript says when the microscope's own software is scanning (its live view) as an acquisition starts.
LIVE_SCAN = re.compile(r"scan ?task is already active|another scan (?:task )?is (?:already )?(?:active|running)", re.IGNORECASE)
# One option: the step starts over when answered and retries the call before it reads the answer, so a "stop"
# answer could not stop it. Stopping the run (stop_workflow) ends it instead.
LIVE_SCAN_QUESTION = {
    "question": "The microscope's own scan (live view) is running, so the workflow cannot take its image. "
                "Stop the live scan in the microscope software, then choose continue (or stop the run).",
    "options": ["continue"],
}


_journal_scope: ContextVar[dict | None] = ContextVar("journal_scope", default=None)


class CallJournal:
    """The calls a run's waiting node already completed, so that its next execution does not send them again.

    LangGraph runs a node that stopped to wait (a held call, a question) again from its first line. Each call that
    execution makes is matched, in order, against the calls the node's earlier execution completed: same server, tool
    and canonical arguments (``safety.permissions.json_sha256``). A match returns the recorded reply without
    sending anything. The first call that does not match is sent, and so is every call after it: the held call is
    sent again and uses its approval. A node that finishes drops its entries.

    A call that runs after the node stopped (the node's own cleanup, the runner's ``always`` tools and restores) may
    undo what the recorded calls did, such as blanking a beam a recorded call unblanked. It drops every entry, so the
    node runs again from its first line as it would without a journal.

    The journal is a JSON file beside the run record (``path``; None keeps it in memory), so it survives a restart.
    The runner removes it when the run ends or is stopped.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        found = json.loads(path.read_text(encoding="utf-8") or "{}") if path is not None and path.exists() else {}
        self.nodes: dict[str, list[dict]] = found if isinstance(found, dict) else {}

    def _save(self) -> None:
        if self.path is None:
            return
        if self.nodes:
            write_atomic(self.path, json.dumps(self.nodes, default=str))
        else:
            self.path.unlink(missing_ok=True)

    @contextlib.contextmanager
    def node(self, name: str) -> Iterator[None]:
        """The execution of one node: its calls are matched and recorded under ``name``; they are dropped when it finishes."""
        token = _journal_scope.set({"node": name, "cursor": 0, "replaying": True, "stopped": False})
        try:
            yield
            if self.nodes.pop(name, None) is not None:
                self._save()
        finally:
            _journal_scope.reset(token)

    def replay(self, server: str, tool: str, arguments: dict) -> dict | None:
        """The recorded reply of this call, {"text"}, when the node's earlier execution completed it at this point; else None."""
        scope = _journal_scope.get()
        if scope is None or not scope["replaying"]:
            return None
        entries = self.nodes.get(scope["node"], [])
        at = scope["cursor"]
        if at < len(entries) and entries[at]["key"] == [server, tool, json_sha256(arguments)]:
            scope["cursor"] += 1
            return {"text": entries[at]["text"]}
        scope["replaying"] = False
        if len(entries) > at:
            self.nodes[scope["node"]] = entries[:at]
            self._save()
        return None

    def completed(self, server: str, tool: str, arguments: dict, text: str) -> None:
        """A call that ran: recorded while its node is running, else (after the node stopped, or outside every node) it drops entries."""
        scope = _journal_scope.get()
        if scope is None:
            if self.nodes:
                self.clear()
            return
        if scope["stopped"]:
            if self.nodes.pop(scope["node"], None) is not None:
                self._save()
            return
        self.nodes.setdefault(scope["node"], []).append({"key": [server, tool, json_sha256(arguments)], "text": text})
        scope["cursor"] += 1
        self._save()

    def stopped(self) -> None:
        """The running node stops here (it waits, or a call raised): the calls after this are its cleanup."""
        scope = _journal_scope.get()
        if scope is not None:
            scope["stopped"] = True
            scope["replaying"] = False

    def clear(self) -> None:
        self.nodes = {}
        self._save()


class BlockContext:
    """What a block is given to work with. It holds the two endpoints, the stop switch and the budget.

    ``on_ask`` is called with a question for a person and returns their answer. Without it a block that asks fails.
    ``on_hold`` is called with a call the endpoint holds for a person's approval. Without it that call fails the block.
    The runner raises a LangGraph interrupt in both, so the run waits there.
    """

    def __init__(
        self, instrument: McpServer, analysis: McpServer | None = None, *, stop_file: Path | None = None,
        max_tool_calls: int | None = None, deadline: float | None = None,
        on_ask: Callable[[dict], Any] | None = None, on_hold: Callable[[dict], Any] | None = None,
        journal: CallJournal | None = None,
    ) -> None:
        self.instrument, self.analysis = instrument, analysis
        self.stop_file, self.max_tool_calls, self.deadline, self.on_ask = stop_file, max_tool_calls, deadline, on_ask
        self.on_hold = on_hold
        self.journal = journal if journal is not None else CallJournal(None)
        self.calls: list[dict] = []
        # Every data key the run's replies named, in order: acquisitions and what analysis jobs derived from them.
        self.data_keys: list[str] = []
        # Settings a block could not put back. The run record lists them for the operator.
        self.unrestored: list[dict] = []
        # Settings a run changed and must put back if it ends early: {name: {"set": call, "restore": call}}, each call
        # {"tool", "arguments"}. The runner restores them when a run ends without finishing and sets them again on resume.
        self.restores: dict[str, dict] = {}
        self._names: dict[int, list[str]] = {}
        # For a client following the run: the node running now, and the latest loop node's count
        # ({"node", "done", "total"}, plus "level" in a multi-level map). Set by Steps.block.
        self.step: str | None = None
        self.loop: dict | None = None
        # The images a map has taken so far, where and how wide: data key -> {"key", "x_m", "y_m", "fov_m"}.
        self.tiles: dict[str, dict] = {}
        # Whether the run's workflow has a block that acts on the instrument: two such runs never drive it at once.
        self.acts = False
        # What the run's cleanup did when it gave the instrument back before it ended; None while it has not.
        self.cleaned: list[dict] | None = None

    def changed(self, name: str, set_call: dict, restore_call: dict) -> None:
        """Note a setting this run changed, so the runner puts it back if the run fails, stops or waits.

        The first ``restore_call`` noted under ``name`` is kept, because that is the value from before the run. The
        latest ``set_call`` is kept, because a resumed run must continue with it.
        """
        kept = self.restores.get(name)
        self.restores[name] = {"set": dict(set_call), "restore": dict(kept["restore"] if kept else restore_call)}

    def put_back_done(self, name: str) -> None:
        """The workflow put this setting back itself, so the runner need not."""
        self.restores.pop(name, None)

    def check(self) -> None:
        if self.stop_file is not None and self.stop_file.exists():
            raise Stopped("the run was stopped")
        if self.max_tool_calls is not None and len(self.calls) >= self.max_tool_calls:
            raise Stopped(f"the run's budget of {self.max_tool_calls} tool calls is used up")
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise Stopped("the run's time budget is used up")

    def ask(self, question: dict) -> Any:
        """Ask a person a question and return their answer.

        The run stops here and waits. Once answered, the node starts again from its first line; the calls it completed
        before the question get their recorded replies (``CallJournal``) unless something ran in between.
        """
        self.check()
        if self.on_ask is None:
            raise BlockFailed(f"this step asks a person ({str(question.get('question'))[:200]}), and this run has nobody to ask")
        try:
            return self.on_ask(question)
        except BaseException:
            self.journal.stopped()
            raise

    async def tool_names(self, server: McpServer | None = None) -> list[str]:
        server = server or self.instrument
        if id(server) not in self._names:
            try:
                listed = await server.list_tools()
            except Exception as exc:
                if undelivered(exc):
                    raise Unreachable(f"the endpoint's tools could not be listed: {type(exc).__name__}: {exc}") from exc
                raise
            self._names[id(server)] = sorted(t.name for t in listed)
        return self._names[id(server)]

    async def resolve(self, pattern: str, server: McpServer | None = None) -> str:
        matches = [n for n in await self.tool_names(server) if fnmatch.fnmatchcase(n, pattern)]
        if not matches:
            raise BlockFailed(f"the endpoint offers no tool matching {pattern}")
        return matches[0]

    async def _send(self, server: McpServer, pattern: str, arguments: dict | None, checked: bool) -> Any:
        try:
            return await self._send_once(server, pattern, arguments, checked)
        except BaseException:
            self.journal.stopped()
            raise

    async def _send_once(self, server: McpServer, pattern: str, arguments: dict | None, checked: bool) -> Any:
        if checked:
            self.check()
        name = await self.resolve(pattern, server)
        label = "analysis" if server is self.analysis and server is not self.instrument else "instrument"
        while True:
            recorded = self.journal.replay(label, name, arguments or {})
            if recorded is not None:
                return parse_reply(recorded["text"])
            held = None
            try:
                result = await server.call(name, arguments or {})
                text, error, held = result_text(result), bool(result.is_error), held_for_approval(result)
            except Exception as exc:
                text, error = f"{type(exc).__name__}: {exc}", True
                if undelivered(exc):
                    self.calls.append({"tool": name, "server": label, "args": dict(arguments or {}), "result": text[:200], "error": True})
                    raise Unreachable(f"{name} did not reach the endpoint: {text[:600]}") from exc
            self.calls.append({"tool": name, "server": label, "args": dict(arguments or {}), "result": text[:200], "error": error})
            if not (error or text.startswith("Error")):
                self.journal.completed(label, name, arguments or {}, text)
                self.data_keys += [k for k in reply_keys(text, derived=True) if k not in self.data_keys]
                return parse_reply(text)
            if held is not None:
                if not checked or self.on_hold is None:
                    raise BlockFailed(f"{name} is waiting for a person to approve it (request {held.get('request_id')}), and this run cannot wait")
                self.on_hold(held)
                continue
            # The microscope's own scan (its live view) blocks every acquisition. A person can stop it: the run
            # asks them, waits, and tries the call again; it fails only when they choose to stop.
            if checked and self.on_ask is not None and LIVE_SCAN.search(text):
                self.ask(dict(LIVE_SCAN_QUESTION))
                continue
            raise BlockFailed(f"{name} failed: {text[:600]}")

    async def call(self, pattern: str, arguments: dict | None = None) -> Any:
        """Call the instrument tool matching ``pattern`` and return its parsed reply. Every block's tool call goes
        through here (or ``call_always`` / ``analyse``).

        Raises Stopped (stopped, or out of budget), Unreachable (never delivered; the node is tried again) or
        BlockFailed (the tool refused or failed, with its reply).
        """
        return await self._send(self.instrument, pattern, arguments, checked=True)

    async def call_always(self, pattern: str, arguments: dict | None = None) -> Any:
        """Like ``call`` but sent even when the run is stopped or over budget. Use it for cleanup."""
        return await self._send(self.instrument, pattern, arguments, checked=False)

    async def analyse(self, name: str, arguments: dict | None = None) -> Any:
        """Call a tool of the analysis server."""
        if self.analysis is None:
            raise BlockFailed(f"{name} needs an analysis server, and this endpoint names none (mcp.urls.analysis)")
        return await self._send(self.analysis, name, arguments, checked=True)
