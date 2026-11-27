"""pyTEMlib's analysis server speaks asyncroscopy's wire contract: the same headers, job states and envelope, the same
log lines, and replies the agents' guidance reads."""

import asyncio

import jsonschema
import numpy as np
from fastmcp import FastMCP

from asyncroscopy import settings
from asyncroscopy.agents.guidance import render_reply
from asyncroscopy.contracts import ENVELOPE_VERSION, load as load_schema
from asyncroscopy.core import names
from asyncroscopy.core.fileio import SequencedAppender
from asyncroscopy.trace.writer import read_tool_log
from pyTEMlib.analysis_server import environment, fileio
from pyTEMlib.analysis_server import names as served_names
from pyTEMlib.analysis_server.frames import register as register_frames
from pyTEMlib.analysis_server.jobs import JobRunner
from pyTEMlib.analysis_server.tool_log import ToolCallLog


def test_the_server_uses_asyncroscopys_headers_job_states_and_envelope_version():
    assert (served_names.CALLER_HEADER, served_names.RUN_HEADER) == (names.CALLER_HEADER, names.RUN_HEADER)
    assert served_names.RUNNING_JOB_STATES == names.RUNNING_JOB_STATES
    assert served_names.ACTIVE_JOB_STATES == names.ACTIVE_JOB_STATES
    assert served_names.FINISHED_JOB_STATES == names.FINISHED_JOB_STATES
    assert served_names.ENVELOPE_VERSION == ENVELOPE_VERSION


def test_the_server_reads_the_environment_as_asyncroscopys_settings_do(monkeypatch, tmp_path):
    monkeypatch.setenv("ASYNCROSCOPY_OUTPUTS", str(tmp_path / "out"))
    monkeypatch.setenv("ASYNCROSCOPY_LOG_ROTATE_BYTES", "1234")
    monkeypatch.setenv("ASYNCROSCOPY_TILED_API_KEY", "k3y")
    monkeypatch.setenv("TILED_API_KEY", "plain")
    monkeypatch.setenv("ASYNCROSCOPY_PLUGINS", "my_lab.scorers")
    monkeypatch.delenv("ASYNCROSCOPY_LOG_ROTATE_KEEP", raising=False)
    ours, theirs = settings.get(), environment.get()
    assert theirs.output("trace", "analysis.jsonl") == ours.output("trace", "analysis.jsonl")
    assert (theirs.log_rotate_bytes, theirs.log_rotate_keep) == (ours.log_rotate_bytes, ours.log_rotate_keep)
    assert (theirs.tiled_api_key, theirs.plain_tiled_api_key, theirs.plugins) == (ours.tiled_api_key, ours.plain_tiled_api_key, ours.plugins)


def test_the_servers_log_lines_continue_asyncroscopys_numbering_and_read_back(tmp_path):
    path = tmp_path / "analysis.jsonl"
    SequencedAppender(path).append({"ts": 1.0, "tool": "from_asyncroscopy"})
    fileio.SequencedAppender(path).append({"ts": 2.0, "tool": "from_pytemlib"})
    ToolCallLog(path).record("list_maps", {}, 3.0, 0.01, result="[]", source="analysis", caller="agent", run="r1")
    rows = read_tool_log(path)
    assert [(r["seq"], r["tool"]) for r in rows] == [(1, "from_asyncroscopy"), (2, "from_pytemlib"), (3, "list_maps")]
    assert rows[-1]["source"] == "analysis" and rows[-1]["caller"] == "agent" and rows[-1]["run"] == "r1"


def runner():
    r = JobRunner()
    r.register("stitch", lambda plan: {"steps": 1}, lambda plan, progress: {"result_keys": {"map": "m"}}, resource=lambda plan: plan["run"])
    return r


def envelope(name):
    return {"$ref": f"#/$defs/{name}", "$defs": load_schema("envelope")["$defs"]}


def test_analysis_jobs_answer_in_the_envelope_schema():
    r = runner()
    reserved = r.reserve({"task": "stitch", "run": "run_a.h5"})
    jsonschema.validate(reserved, envelope("reserve_response"))
    started = r.start(reserved["reservation_id"])
    jsonschema.validate(r.wait(started["job_id"]), envelope("status_response"))
    jsonschema.validate(r.results(started["job_id"]), envelope("results_response"))


def test_an_unknown_job_and_a_busy_resource_are_turned_into_the_next_call():
    r = runner()
    unknown = r.status("a1b2c3")
    assert "Mapper_job_status" in render_reply("analysis_job_status", {}, unknown)["note"]
    held = r.reserve({"task": "stitch", "run": "run_d.h5"})
    busy = r.reserve({"task": "stitch", "run": "run_d.h5"})
    assert busy["reservation_id"] == held["reservation_id"]
    assert "instead of starting another" in render_reply("assemble_map", {}, busy)["note"]


class Node:
    def __init__(self, metadata=None, shape=None, children=None):
        self.metadata = metadata or {}
        self._shape = shape
        self._children = children or {}
        self.structure_family = "container" if children is not None else "array"

    def structure(self):
        return type("S", (), {"shape": self._shape})()

    def __contains__(self, name):
        return name in self._children

    def __getitem__(self, name):
        return self._children[name]

    def __iter__(self):
        return iter(self._children)


class Store:
    def __init__(self, nodes, sample=None):
        self.client = nodes
        self.sample = sample or {}

    def map_metadata(self, key):
        raise KeyError(key)

    def sample_of(self, key):
        return dict(self.sample)


IMAGE = {"pixel_size_m": 1e-9, "stage_reported_m": "[1e-6, -2e-6, 0.0, 0.0, 0.0]", "axes": "rows=+y, columns=+x", "scan_region": "[0.0, 0.0, 1.0, 1.0]"}


def test_a_placed_pixel_tells_the_agent_to_check_the_sample_load_first():
    mcp = FastMCP("t")
    store = Store({"img.h5": Node(children={"image": Node(children={"HAADF": Node(IMAGE, shape=(101, 201))})})}, {"sample_load_id": "abc"})
    register_frames(mcp, store)
    placed = asyncio.run(mcp.call_tool("to_stage", {"dataset": "img.h5", "row": 50, "col": 100})).structured_content
    assert "get_sample" in render_reply("to_stage", {}, placed)["note"]


def test_features_of_an_image_taken_with_the_beam_blanked_say_so():
    noise = np.random.default_rng(0).poisson(5, (128, 128)).astype(np.uint16)
    node = Node({**IMAGE, "beam_current_pa": 0.0}, shape=noise.shape)
    node.read = lambda: noise
    store = Store({"dark.h5": Node(children={"image": Node(children={"HAADF": node})})}, sample={"sample_load_id": "L1"})
    tools = {}
    register_frames(type("M", (), {"tool": lambda self: (lambda f: tools.setdefault(f.__name__, f))})(), store)
    reply = tools["find_features"]("dark.h5", n=3)
    assert "beam blanked" in render_reply("find_features", {}, reply)["hint"]
