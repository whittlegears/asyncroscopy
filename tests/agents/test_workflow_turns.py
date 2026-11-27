"""One run at a time acts on the instrument: the endpoint's turn queue, its guard, and workflow runs that queue."""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError

from asyncroscopy.workflows import get_workflow, turns
from asyncroscopy.workflows.blocks import all_blocks
from asyncroscopy.workflows.runner import read_run, run_workflow, stop_workflow
from asyncroscopy.workflows.turns import Turn, acting_ahead
from asyncroscopy.mcp.components.turns import TurnGuard, TurnQueue
from asyncroscopy.mcp.interceptors import Call

INPUTS = {"size_m": 5.0e-7, "fov_m": 3.0e-7}  # a 2 x 2 grid


# the endpoint's queue


def queue_at(now: list[float], **kwargs) -> TurnQueue:
    return TurnQueue(clock=lambda: now[0], **kwargs)


def test_the_first_run_holds_the_turn_and_the_others_queue_in_the_order_they_asked():
    queue = queue_at([0.0])
    assert queue.take("a", "agent", "map A")["state"] == "held"
    second, third = queue.take("b", what="map B"), queue.take("c")
    assert (second["state"], second["position"]) == ("queued", 1) and second["ahead"][0]["run"] == "a" and second["ahead"][0]["what"] == "map A"
    assert (third["position"], [e["run"] for e in third["ahead"]]) == (2, ["a", "b"])
    assert queue.take("b")["position"] == 1  # asking again keeps the place
    assert queue.give("a") == {"gave": "turn", "waiting": 2}
    assert queue.take("c")["state"] == "queued"  # c may not jump the line
    assert queue.take("b")["state"] == "held"
    assert queue.take("c")["position"] == 1
    assert queue.status()["holder"]["run"] == "b" and [w["run"] for w in queue.status()["queue"]] == ["c"]


def test_a_turn_that_is_not_renewed_ends_and_a_run_that_stops_asking_loses_its_place():
    now = [0.0]
    queue = queue_at(now, patience_s=30.0)
    queue.take("dead", lease_s=60)
    queue.take("gone")
    now[0] = 20.0
    assert queue.take("dead", lease_s=60)["state"] == "held"  # renewed: 60 s from now
    assert queue.take("alive")["position"] == 2
    now[0] = 45.0  # "gone" last asked 45 s ago
    assert queue.take("alive")["position"] == 1
    now[0] = 81.0  # the holder's lease ended at 80
    assert queue.take("alive")["state"] == "held"
    assert queue.give("dead") == {"gave": "nothing", "waiting": 0}


def test_leaving_the_queue_and_the_lease_bounds():
    queue = queue_at([0.0])
    queue.take("a")
    queue.take("b")
    assert queue.give("b") == {"gave": "place", "waiting": 0}
    assert queue.take("a", lease_s=1)["expires_in_s"] == 5.0 and queue.take("a", lease_s=1e6)["expires_in_s"] == 600.0


def descriptor(name: str, acts: bool = True, stop: bool = False, role: str = "other"):
    return SimpleNamespace(name=name, acts=acts, stop=stop, action_role=role)


def test_only_the_holder_may_act_while_reads_and_stops_always_pass():
    queue = queue_at([0.0])
    guard = TurnGuard(queue)
    move, read = descriptor("Twin_move_stage", role="stage.move"), descriptor("Twin_get_stage", acts=False)
    blank, abort = descriptor("Twin_blank_beam", role="beam.blank"), descriptor("Mapper_abort_all", stop=True)
    assert guard.applies(move) and not guard.applies(read) and not guard.applies(blank) and not guard.applies(abort)

    def call(run):
        return Call(tool=move, arguments={}, call_args=(), run=run)

    assert guard.before_call(call("anyone")) is None  # nobody holds the turn: nothing changes
    queue.take("mapper", what="map_grid")
    assert guard.before_call(call("mapper")) is None
    for other in ("someone-else", None):
        with pytest.raises(ToolError) as refused:
            guard.before_call(call(other))
        body = json.loads(str(refused.value))
        assert body["error"]["code"] == "instrument_busy" and body["holder"]["run"] == "mapper" and "map_grid" in body["error"]["message"]
    queue.give("mapper")
    assert guard.before_call(call("someone-else")) is None


# which nodes still act


def test_a_map_has_no_acting_block_left_once_it_is_back_home():
    blocks = all_blocks()
    grid = get_workflow("map_grid").blueprint()  # plan -> tile (loop) -> home -> run -> stitch
    assert acting_ahead(grid, blocks, None) and acting_ahead(grid, blocks, "tile") and acting_ahead(grid, blocks, "home")
    assert not acting_ahead(grid, blocks, "run") and not acting_ahead(grid, blocks, "stitch")
    small = get_workflow("map_4x4").blueprint()  # ... -> stitch -> home: it goes home after the stitch
    assert acting_ahead(small, blocks, "stitch")
    sample = get_workflow("map_sample").blueprint()  # its stitch is inside the loop that takes more tiles
    assert acting_ahead(sample, blocks, "stitch")
    assert acting_ahead({"nodes": {"x": {"block": "no_such_block"}}, "edges": []}, blocks, "x")  # unknown: taken to act


# workflow runs


class Endpoint:
    """One run's connection to a fake instrument that gives turns. ``shared`` is the instrument: its state, its turn
    queue and one log of every call with the run that made it."""

    def __init__(self, shared: dict, run: str, turns_offered: bool = True, delay_s: float = 0.002) -> None:
        self.shared, self.run, self.delay_s = shared, run, delay_s
        self.tools = {
            "Twin_get_stage": lambda a: json.dumps(shared["stage"]),
            "Twin_move_stage": lambda a: "ok",
            "Twin_get_image_shift": lambda a: json.dumps([0.0, 0.0]),
            "Twin_get_image_shift_limit": lambda a: json.dumps(1e-6),
            "Twin_set_image_shift": lambda a: "ok",
            "Twin_set_fov": lambda a: "ok",
            "Twin_get_fov": lambda a: json.dumps(5.8e-9),
            "Twin_unblank_beam": lambda a: shared.update(blanked=False) or "ok",
            "Twin_blank_beam": lambda a: shared.update(blanked=True) or "ok",
            "Twin_acquire_scanned_image": self._acquire,
        }
        if turns_offered:
            self.tools.update(take_turn=lambda a: shared["queue"].take(run, "test", a.get("what", ""), a.get("lease_s", 60)),
                              give_turn=lambda a: shared["queue"].give(run), turn_status=lambda a: shared["queue"].status())

    def _acquire(self, args):
        if self.shared["blanked"]:
            return "Error: the beam is blanked"
        self.shared["taken"] += 1
        return f"stem_image_HAADF_{self.shared['taken']:04d}.h5\nimage 256 x 256"

    async def list_tools(self):
        return [SimpleNamespace(name=name) for name in self.tools]

    async def call(self, name, arguments=None):
        await asyncio.sleep(self.delay_s)  # every call lets the other run in, as a real endpoint does
        reply = self.tools[name](dict(arguments or {}))
        self.shared["log"].append((self.run, name))
        text = reply if isinstance(reply, str) else json.dumps({k: v for k, v in reply.items() if k != "new"})
        return SimpleNamespace(is_error=text.startswith("Error"), structured_content=None, content=[SimpleNamespace(text=text)])


class Analysis:
    """A fake analysis server whose stitch takes ``stitch_s``."""

    def __init__(self, shared: dict, run: str, stitch_s: float = 0.0) -> None:
        self.shared, self.run, self.stitch_s = shared, run, stitch_s

    async def list_tools(self):
        return [SimpleNamespace(name=name) for name in ("collect_map_run", "assemble_map", "analysis_job_status", "analysis_job_results")]

    async def call(self, name, arguments=None):
        self.shared["log"].append((self.run, name))
        if name == "analysis_job_status":
            await asyncio.sleep(self.stitch_s)
        reply = {"collect_map_run": {"run": "map_abc", "tiles": 4}, "assemble_map": {"job_id": "j1", "state": "running"},
                 "analysis_job_status": {"job_id": "j1", "state": "done"},
                 "analysis_job_results": {"result_keys": {"map": "map_abc_map_v1"}, "summary": {"tiles": 4}}}[name]
        return SimpleNamespace(is_error=False, structured_content=None, content=[SimpleNamespace(text=json.dumps(reply))])


def instrument_state() -> dict:
    return {"stage": [1e-6, 2e-6, 0.0, 0.0, 0.0], "blanked": True, "taken": 0, "log": [], "queue": TurnQueue()}


@pytest.fixture(autouse=True)
def quick_turns(monkeypatch):
    monkeypatch.setattr(turns, "POLL_S", 0.01)
    monkeypatch.setattr(turns, "TICK_S", 0.01)


ACTING = ("Twin_set_image_shift", "Twin_set_fov", "Twin_move_stage", "Twin_unblank_beam", "Twin_acquire_scanned_image")


def acting(shared: dict) -> list[str]:
    """The runs that acted on the instrument, in order, one entry per unbroken stretch."""
    out: list[str] = []
    for run, name in shared["log"]:
        if name in ACTING and (not out or out[-1] != run):
            out.append(run)
    return out


def test_two_maps_started_together_take_the_instrument_one_after_the_other(tmp_path):
    shared = instrument_state()

    async def both():
        slow = Endpoint(shared, "a", delay_s=0.02)  # its tiles take about a second
        first = asyncio.create_task(run_workflow("map_grid", INPUTS, instrument=slow, analysis=Analysis(shared, "a"), state_dir=tmp_path, run_id="a"))
        while ("a", "Twin_acquire_scanned_image") not in shared["log"]:
            await asyncio.sleep(0.01)
        second = asyncio.create_task(run_workflow("map_grid", INPUTS, instrument=Endpoint(shared, "b"), analysis=Analysis(shared, "b"),
                                                  state_dir=tmp_path, run_id="b"))
        await asyncio.sleep(0.15)
        waiting = read_run("b", tmp_path)
        return await first, await second, waiting

    first, second, waiting = asyncio.run(both())
    assert waiting["state"] == "queued" and waiting["queue"]["position"] == 1 and waiting["queue"]["ahead"][0]["run"] == "a"
    assert "map_grid" in waiting["queue"]["ahead"][0]["what"]
    assert first["state"] == "done" and second["state"] == "done" and second.get("queue") is None
    assert acting(shared) == ["a", "b"]  # never interleaved: without the turn each would blank the other's beam
    assert shared["taken"] == 8 and shared["queue"].status() == {"holder": None, "queue": []}


def test_a_map_gives_the_instrument_back_before_its_stitch_so_the_next_map_starts_meanwhile(tmp_path):
    shared = instrument_state()

    async def both():
        first = asyncio.create_task(run_workflow("map_grid", INPUTS, instrument=Endpoint(shared, "a"), analysis=Analysis(shared, "a", stitch_s=3.0),
                                                 state_dir=tmp_path, run_id="a"))
        while ("a", "Twin_acquire_scanned_image") not in shared["log"]:
            await asyncio.sleep(0.01)
        second = await run_workflow("map_grid", INPUTS, instrument=Endpoint(shared, "b"), analysis=Analysis(shared, "b"), state_dir=tmp_path, run_id="b")
        done_first = first.done()
        return await first, second, done_first

    first, second, first_done_when_second_ended = asyncio.run(both())
    assert first["state"] == "done" and second["state"] == "done"
    assert not first_done_when_second_ended  # the second map was taken and stitched while the first was still stitching
    assert acting(shared) == ["a", "b"]
    log = shared["log"]
    gave = log.index(("a", "give_turn"))
    assert gave < log.index(("a", "analysis_job_results"))  # given back before its own stitch ended
    on_instrument = [name for run, name in log if run == "a" and (name.startswith("Twin_") or name.endswith("_turn"))]
    # It blanked the beam first, while the instrument was still its own, and never touched it again.
    assert on_instrument[-2:] == ["Twin_blank_beam", "give_turn"]
    assert first["cleanup"] == [{"tool": "*_blank_beam", "ok": True}]


def test_a_queued_run_that_is_stopped_leaves_the_queue_and_calls_nothing(tmp_path):
    shared = instrument_state()
    shared["queue"].take("someone-else", what="a long map")

    async def stop_it():
        task = asyncio.create_task(run_workflow("map_grid", INPUTS, instrument=Endpoint(shared, "b"), analysis=Analysis(shared, "b"),
                                                state_dir=tmp_path, run_id="b"))
        for _ in range(500):
            await asyncio.sleep(0.01)
            with contextlib.suppress(KeyError):
                if read_run("b", tmp_path)["state"] == "queued":
                    break
        assert read_run("b", tmp_path)["state"] == "queued"
        stop_workflow("b", tmp_path)
        return await task

    record = asyncio.run(stop_it())
    assert record["state"] == "stopped" and "waited for the instrument" in record["error"]
    assert not [name for run, name in shared["log"] if name.startswith("Twin_")]
    assert shared["queue"].status()["queue"] == [] and shared["queue"].status()["holder"]["run"] == "someone-else"


def test_an_endpoint_without_turns_runs_the_map_as_before(tmp_path):
    shared = instrument_state()
    record = asyncio.run(run_workflow("map_grid", INPUTS, instrument=Endpoint(shared, "a", turns_offered=False), analysis=Analysis(shared, "a"),
                                      state_dir=tmp_path, run_id="a"))
    assert record["state"] == "done" and shared["taken"] == 4
    assert not [name for _run, name in shared["log"] if name.endswith("_turn")]


def test_the_holder_renews_its_turn_and_notes_when_it_was_lost():
    shared = instrument_state()

    async def hold():
        turn = Turn(Endpoint(shared, "a"), what="map", lease_s=60, renew_s=0.02)
        await turn.wait()
        turn.keep()
        deadline = asyncio.get_running_loop().time() + 5.0
        renewals = 0
        while renewals < 3 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)
            renewals = sum(1 for entry in shared["log"] if entry == ("a", "take_turn"))
        shared["queue"].give("a")
        shared["queue"].take("thief")  # as if the lease had run out and another run had taken the turn
        deadline = asyncio.get_running_loop().time() + 5.0
        while "lost" not in (turn.lost or "") and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)
        lost = turn.lost
        await turn.give()
        return renewals, lost

    renewals, lost = asyncio.run(hold())
    assert renewals >= 3 and "lost" in lost
