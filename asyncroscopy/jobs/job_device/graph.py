"""A job's run as a LangGraph graph.

Every job device, the Mapper and the Tuner among them, runs its steps through this one graph.

    START -> start -> gate -+-> step -+
                       ^    |         |
                       +----|---------+   (while steps are left)
                            +-> finish -> END   (all steps done)
                            +-> stop -> END     (aborted)

``start`` calls the device's ``on_start``. ``gate`` picks the next step that is not done and waits there while the job
is suspended or paused. ``step`` calls ``run_step`` and saves its record. ``finish`` calls ``on_finish`` and ``stop``
calls ``on_stop``. A node that raises ends the run, and the worker then makes the instrument safe.

LangGraph runs a graph's single node on the thread that invoked it. Tango needs that, because only the worker
thread is registered for Tango calls.
"""

from __future__ import annotations

import threading
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .files import KEPT_STEP_RESULTS
from .progress import is_done, mark_done, step_count


class JobState(TypedDict, total=False):
    """Where a job's run is. ``index`` is the next step to look at and ``outcome`` is set by the last node."""

    index: int
    more: bool
    outcome: str


def recursion_limit(job: dict) -> int:
    """Enough graph steps for every step of the job to pass the gate and run once."""
    return 2 * step_count(job) + 20


def job_graph(device: Any, job: dict, job_id: str, abort: threading.Event, generation: int, worker: Any) -> StateGraph:
    """Build the graph that runs ``job`` on ``device``. ``worker`` is the thread's record of its job and step."""
    name = type(device).__name__

    def start(state: JobState) -> dict:
        device.on_start(job)
        return {"index": 0}

    def gate(state: JobState) -> dict:
        index = state.get("index", 0)
        while index < step_count(job) and is_done(job, index):
            index += 1
        if index >= step_count(job) or abort.is_set():
            return {"index": index, "more": False}
        if device._suspenders and not device._hold_while_suspended(job, job_id, abort):
            return {"index": index, "more": False}
        held = device._hold_while_paused(job, job_id, abort)
        if held == "aborted":
            return {"index": index, "more": False}
        if held == "continued" and device._suspenders and not device._hold_while_suspended(job, job_id, abort):
            return {"index": index, "more": False}
        return {"index": index, "more": True}

    def step(state: JobState) -> dict:
        index = state["index"]
        started = time.time()
        worker.step = index
        record = device.run_step(job, index)
        if record is None:
            if abort.is_set():
                return {"more": False}
            raise RuntimeError(f"{name}.run_step returned None for step {index} without an abort; return a record of the step")
        with device._lock:
            mark_done(job, index)
            results = job["step_results"]
            results[str(index)] = record
            if len(results) > KEPT_STEP_RESULTS:
                del results[next(iter(results))]
            if generation == device._generation:
                device._append_step(job, index, record)
                device._save(job)
        told = record if isinstance(record, dict) else {}
        device._trace.record(
            f"{name}.step", True, started, duration_s=time.time() - started, job_id=job_id, plan_id=job["plan_id"],
            run_id=job["reservation_id"], step=index, caller=job.get("caller"),
            result=record, result_key=told.get("key"), stage_reported_m=told.get("stage_reported_m"),
            **({"clock_s": told["clock_s"]} if "clock_s" in told else {}),
            **device._trace_fields(job),
        )
        return {"index": index + 1, "more": True}

    def finish(state: JobState) -> dict:
        # Not under the lock, because on_finish may be slow and abort, job_status and release must still answer
        finished = device.on_finish(job) or {}
        with device._lock:
            job["result_keys"].update(finished)
        return {"outcome": "done"}

    def stop(state: JobState) -> dict:
        device.on_stop(job, "aborted")
        return {"outcome": "aborted"}

    def after_gate(state: JobState) -> str:
        if state["more"]:
            return "step"
        return "stop" if abort.is_set() else "finish"

    def after_step(state: JobState) -> str:
        return "gate" if state["more"] else "stop"

    graph = StateGraph(JobState)
    graph.add_node("start", start)
    graph.add_node("gate", gate)
    graph.add_node("step", step)
    graph.add_node("finish", finish)
    graph.add_node("stop", stop)
    graph.add_edge(START, "start")
    graph.add_edge("start", "gate")
    graph.add_conditional_edges("gate", after_gate, ["step", "finish", "stop"])
    graph.add_conditional_edges("step", after_step, ["gate", "stop"])
    graph.add_edge("finish", END)
    graph.add_edge("stop", END)
    return graph
