"""A workflow run's turn on the instrument: queue for it, keep it while the run acts, give it back early.

The instrument endpoint lets one run at a time act on the instrument (``asyncroscopy.mcp.components.turns``: the
turn is kept there because runs may be in different processes). A run whose workflow has a block that acts:

1. queues with ``take_turn`` and waits, as state ``queued``, until the runs ahead of it are done;
2. renews the turn while it acts;
3. gives it back as soon as no block that acts can still run. A map only needs the instrument while it takes its
   tiles: it is stitched on the analysis server, and the next map starts while that goes on.

A run whose workflow never acts (it reads data, stitches, scores) takes no turn and starts at once. An endpoint
without ``take_turn`` (an older one) gives no turns, and the run goes on as it did before.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from ..mcp_tools import McpServer, result_text
from .blocks import Block, Stopped
from .checks import edges_of

LEASE_S = 60.0  # the turn is lost this long after the last renewal: a run whose process died frees the instrument
RENEW_S = 15.0
POLL_S = 2.0  # how often a queued run asks again
TICK_S = 0.5  # how often a run holding the turn looks whether it has finished acting
TAKE, GIVE = "take_turn", "give_turn"


def acting_ahead(blueprint: dict, blocks: dict[str, Block], node: str | None) -> bool:
    """Whether ``node`` or any node the graph can still reach from it uses a block that acts on the instrument.

    Before the first node (``node`` is None) the whole graph is ahead."""
    nodes = blueprint.get("nodes") or {}

    def acts(name: str) -> bool:
        block = blocks.get(str((nodes.get(name) or {}).get("block")))
        return block is None or block.acts  # an unknown block is taken to act

    if node is None or node not in nodes:
        return any(acts(name) for name in nodes)
    after: dict[str, set[str]] = {}
    for edge in edges_of(blueprint):
        after.setdefault(edge["from"], set()).update(t for t in (edge.get("to"), edge.get("otherwise")) if t)
    seen, todo = {node}, [node]
    while todo:
        for following in after.get(todo.pop(), ()):
            if following in nodes and following not in seen:
                seen.add(following)
                todo.append(following)
    return any(acts(name) for name in seen)


async def _reply(server: McpServer, tool: str, arguments: dict | None = None) -> dict:
    result = await server.call(tool, arguments or {})
    text = result_text(result)
    if result.is_error:
        raise RuntimeError(f"{tool} failed: {text[:300]}")
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return reply if isinstance(reply, dict) else {}


class Turn:
    """One run's turn. ``wait`` queues for it, ``keep`` renews it in the background, ``give`` hands it back."""

    def __init__(self, instrument: McpServer, what: str = "", lease_s: float | None = None, poll_s: float | None = None,
                 renew_s: float | None = None) -> None:
        self.instrument, self.what = instrument, what
        self.lease_s = LEASE_S if lease_s is None else lease_s
        self.poll_s = POLL_S if poll_s is None else poll_s
        self.renew_s = RENEW_S if renew_s is None else renew_s
        self.held = False
        self.lost: str | None = None  # why the turn could not be renewed, when that happened
        self._giving = False  # the turn is being handed back now: that must run to its end
        self._keeper: asyncio.Task | None = None

    @staticmethod
    async def offered(instrument: McpServer) -> bool:
        """Whether the endpoint gives turns."""
        return TAKE in {t.name for t in await instrument.list_tools()}

    async def wait(self, on_queued: Callable[[dict], Any] | None = None, stopped: Callable[[], bool] | None = None) -> None:
        """Queue until the turn is this run's. ``on_queued`` gets each queued reply ({"position", "ahead"}) when it
        changes. Raises Stopped, and leaves the queue, when ``stopped()`` turns true while it waits."""
        told = None
        try:
            while True:
                reply = await _reply(self.instrument, TAKE, {"what": self.what, "lease_s": self.lease_s})
                if reply.get("state") == "held":
                    self.held = True
                    return
                shown = {"position": reply.get("position"), "ahead": [{k: e.get(k) for k in ("run", "what", "caller")} for e in reply.get("ahead") or []]}
                if on_queued is not None and json.dumps(shown, sort_keys=True) != told:
                    told = json.dumps(shown, sort_keys=True)
                    on_queued(shown)
                if stopped is not None and stopped():
                    raise Stopped("the run was stopped while it waited for the instrument")
                await asyncio.sleep(self.poll_s)
        except BaseException:
            if not self.held:  # stopped or cancelled while queued: do not keep a place nobody will use
                with contextlib.suppress(Exception):
                    await _reply(self.instrument, GIVE)
            raise

    def keep(self, done_acting: Callable[[], bool] | None = None, before_giving: Callable[[], Awaitable[Any]] | None = None) -> None:
        """Renew the turn in the background until ``give``. When ``done_acting()`` turns true the turn is given back
        at once, after ``before_giving`` (the run's cleanup, which must happen while the instrument is still its own)."""

        async def renew() -> None:
            waited = 0.0
            while self.held:
                await asyncio.sleep(min(TICK_S, self.renew_s))
                waited += min(TICK_S, self.renew_s)
                if done_acting is not None and done_acting():
                    await self._give(before_giving)
                    return
                if waited >= self.renew_s:
                    waited = 0.0
                    try:
                        reply = await _reply(self.instrument, TAKE, {"what": self.what, "lease_s": self.lease_s})
                    except Exception as exc:  # the endpoint did not answer: the lease may still hold, so try again
                        self.lost = f"{type(exc).__name__}: {exc}"[:200]
                        continue
                    self.lost = None if reply.get("state") == "held" else "the turn was lost: it went to another run after the lease ran out"

        self._keeper = asyncio.get_running_loop().create_task(renew())

    async def _give(self, before_giving: Callable[[], Awaitable[Any]] | None = None) -> None:
        if not self.held or self._giving:
            return
        self._giving = True
        try:
            if before_giving is not None:
                await before_giving()
        finally:
            try:
                with contextlib.suppress(Exception):  # a turn that could not be given back ends with its lease
                    await _reply(self.instrument, GIVE)
            finally:
                self.held = self._giving = False

    async def give(self) -> None:
        """Stop renewing and give the turn back (nothing to do when it was given back already)."""
        keeper, self._keeper = self._keeper, None
        if keeper is not None and keeper is not asyncio.current_task():
            if not self._giving:  # cancelled half way through handing back, the turn would stay held until its lease ended
                keeper.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await keeper
        await self._give()
