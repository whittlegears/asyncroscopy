"""Blocks are the steps a workflow is built from.

A block declares its inputs, its outputs and the tools it calls. Tool names are glob patterns, as in skills. With
these a workflow can be checked before anything runs. A block's ``run`` reaches the instrument only through
``BlockContext.call``.
"""

from __future__ import annotations

import fnmatch
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asyncroscopy.core.names import DATA_KEY_IN_TEXT

from ..mcp_tools import McpServer, result_text

TYPES = {"number": (int, float), "integer": (int,), "string": (str,), "boolean": (bool,), "list": (list,), "object": (dict,), "any": (object,)}


class BlockFailed(RuntimeError):
    """A tool call failed or a block could not do its work. The message says which call and why."""


class Stopped(RuntimeError):
    """The run was stopped or ran out of budget before this call was sent."""


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


class BlockContext:
    """What a block is given to work with. It holds the two endpoints, the stop switch and the budget.

    ``on_ask`` is called with a question for a person and returns their answer. Without it a block that asks fails.
    The runner raises a LangGraph interrupt in it, so the run waits there.
    """

    def __init__(
        self, instrument: McpServer, analysis: McpServer | None = None, *, stop_file: Path | None = None,
        max_tool_calls: int | None = None, deadline: float | None = None,
        on_ask: Callable[[dict], Any] | None = None,
    ) -> None:
        self.instrument, self.analysis = instrument, analysis
        self.stop_file, self.max_tool_calls, self.deadline, self.on_ask = stop_file, max_tool_calls, deadline, on_ask
        self.calls: list[dict] = []
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

        The run stops here and waits. Once answered, the node starts again from its first line. So a block must ask
        before it does anything that should not happen twice.
        """
        self.check()
        if self.on_ask is None:
            raise BlockFailed(f"this step asks a person ({str(question.get('question'))[:200]}), and this run has nobody to ask")
        return self.on_ask(question)

    async def tool_names(self, server: McpServer | None = None) -> list[str]:
        server = server or self.instrument
        if id(server) not in self._names:
            self._names[id(server)] = sorted(t.name for t in await server.list_tools())
        return self._names[id(server)]

    async def resolve(self, pattern: str, server: McpServer | None = None) -> str:
        matches = [n for n in await self.tool_names(server) if fnmatch.fnmatchcase(n, pattern)]
        if not matches:
            raise BlockFailed(f"the endpoint offers no tool matching {pattern}")
        return matches[0]

    async def _send(self, server: McpServer, pattern: str, arguments: dict | None, checked: bool) -> Any:
        if checked:
            self.check()
        name = await self.resolve(pattern, server)
        try:
            result = await server.call(name, arguments or {})
            text, error = result_text(result), bool(result.is_error)
        except Exception as exc:
            text, error = f"{type(exc).__name__}: {exc}", True
        self.calls.append({"tool": name, "args": dict(arguments or {}), "result": text[:200], "error": error})
        if error or text.startswith("Error"):
            raise BlockFailed(f"{name} failed: {text[:600]}")
        return parse_reply(text)

    async def call(self, pattern: str, arguments: dict | None = None) -> Any:
        """Call the instrument tool matching ``pattern`` and return its parsed reply.

        Raises Stopped or BlockFailed.
        """
        return await self._send(self.instrument, pattern, arguments, checked=True)

    async def call_always(self, pattern: str, arguments: dict | None = None) -> Any:
        """Like ``call`` but sent even when the run is stopped or over budget. Use it for cleanup."""
        return await self._send(self.instrument, pattern, arguments, checked=False)

    async def analyse(self, name: str, arguments: dict | None = None) -> Any:
        """Call a tool of the analysis server."""
        if self.analysis is None:
            raise BlockFailed(f"{name} needs an analysis server, and this endpoint names none (mcp.analysis_url)")
        return await self._send(self.analysis, name, arguments, checked=True)
