"""Every job device runs its steps as one LangGraph graph."""

import threading
from types import SimpleNamespace

from asyncroscopy.jobs.job_device.graph import job_graph, recursion_limit
from asyncroscopy.jobs.job_device.progress import step_count


class FakeDevice:
    """A stand-in job device that records what the graph asks of it."""

    def __init__(self, abort_at=None, abort=None):
        self._lock, self._generation, self._suspenders = threading.Lock(), 0, []
        self._trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        self.seen, self.threads, self.abort_at, self.abort = [], set(), abort_at, abort

    def on_start(self, job):
        self.seen.append("start")

    def run_step(self, job, index):
        self.threads.add(threading.get_ident())
        if index == self.abort_at:
            self.abort.set()
            return None
        self.seen.append(index)
        return {"index": index}

    def on_finish(self, job):
        self.seen.append("finish")
        return {"run": "map_1"}

    def on_stop(self, job, reason):
        self.seen.append(reason)

    def _hold_while_paused(self, job, job_id, abort):
        return "running"

    def _append_step(self, job, index, record):
        pass

    def _save(self, job):
        pass

    @staticmethod
    def _trace_fields(job):
        return {}


def job_of(steps):
    job = {"plan": {"steps": [{}] * steps}, "steps": [{}] * steps, "step_count": steps, "done_ranges": [], "step_results": {},
           "result_keys": {}, "plan_id": "p", "reservation_id": "r"}
    assert step_count(job) == steps
    return job


def run(device, job, abort):
    graph = job_graph(device, job, "j", abort, 0, SimpleNamespace(step=None)).compile()
    return graph, graph.invoke({"index": 0}, {"recursion_limit": recursion_limit(job)})


def test_a_job_is_a_langgraph_graph_that_runs_every_step_on_the_callers_thread():
    device, job, abort = FakeDevice(), job_of(3), threading.Event()
    graph, final = run(device, job, abort)
    assert {"start", "gate", "step", "finish", "stop"} <= set(graph.get_graph().nodes)
    assert final["outcome"] == "done" and device.seen == ["start", 0, 1, 2, "finish"]
    assert job["result_keys"] == {"run": "map_1"} and sorted(job["step_results"]) == ["0", "1", "2"]
    assert device.threads == {threading.get_ident()}  # Tango calls only on the worker thread


def test_an_aborted_job_stops_without_finishing():
    abort = threading.Event()
    device, job = FakeDevice(abort_at=1, abort=abort), job_of(3)
    _, final = run(device, job, abort)
    assert final["outcome"] == "aborted" and device.seen == ["start", 0, "aborted"]
