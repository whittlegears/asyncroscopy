"""Workflow benchmarks: realistic requests to an agent on the digital twin, checked from facts.

Each workflow in benchmarks/workflows/*.yaml is a request (or several turns) a STEM user
or a hackathon team would make, the runners to try it with, and checks computed from the
Instrument Trace, readings taken before and after, and the stored data (see
benchmarks/workflow_checks.py). Nothing is judged from what the model says alone.

Needs a running stack, both MCP endpoints and the analysis server (docs/Operation/quickstart.md):

    uv run python benchmarks/run_workflows.py                      # every twin workflow once
    uv run python benchmarks/run_workflows.py --only map_300nm --n 5
    uv run python benchmarks/run_workflows.py --stack large        # with configs/digital_twin_large_map.yaml
    uv run python benchmarks/run_workflows.py --stack xrd          # with configs/yaml_sim_xrd.yaml
    uv run python benchmarks/run_workflows.py --workflows mine.yaml --stack raman   # a stack declared in mine.yaml or benchmarks/stacks.yaml
    uv run python benchmarks/run_workflows.py --list               # what would run, no stack needed

Runners: ``general`` (the model makes every call), ``mapping`` (the fixed graph; the model
writes the plan), ``team`` (planner, specialists and reporter) and ``goal`` (the team
working toward a goal inferred from the request, checked and retried), each used when the
checkout has it, and any chat graph given with --graph module:function. Results: a JSON file and a Markdown
table next to it (benchmarks/results/workflows_<stack>.json by default).

IN_PLACE_SETUPS are taken where the run starts, so the stage is homed first: a move after the survey adds a landing
error no real zoom has.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import yaml
from _common import (
    endpoint_risks,
    find_tool,
    git_head,
    is_infra_error,
    read_trace,
    unique_caller,
    warm_up,
)
from board import HISTORY, KINDS, parse_bar, previous_run, save_history
from board import report as board_report
import workflow_checks
from workflow_checks import CHECKS, Call, Run, evaluate, unrequested, workflow_allow

from asyncroscopy import settings
from asyncroscopy.agents.goals import MAP_KEY, trace_from_messages
from asyncroscopy.agents.profile import load_profile
from asyncroscopy.agents.skill_middleware import recursion_limit
from asyncroscopy.core.names import DEFAULT_MCP_URL, DEFAULT_TILED_URI, default_operator_url, devices_of_kind
from asyncroscopy.safety.permissions import json_sha256
from asyncroscopy.sim.truth_score import median_error
from asyncroscopy.skills.registry import list_skills
from asyncroscopy.workflows.signoff import twin_truth
from benchmarks.evaluate import RunData
from benchmarks.evaluate import evaluators as known_evaluators
from benchmarks.evaluate import score_run as evaluator_scores

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from asyncroscopy.agents.config import AgentConfig
    from asyncroscopy.agents.mcp_tools import McpServer

HERE = Path(__file__).resolve().parent
WORKFLOW_DIR = HERE / "workflows"
WORKFLOW_KEYS = {"name", "description", "request", "turns", "runners", "timeout_s", "stack", "setup", "checks",
                 "kind", "held_out", "bar", "reference"}
REFERENCE_KEYS = {"calls", "events", "before", "after", "answer", "facts", "skills", "risks"}
TURN_KEYS = {"say", "wait_for"}
SETUPS = {"image", "survey_image", "map", "map_plain", "large_map"}
SURVEY_FOV_M = 6.0e-7  # room for two zoom steps down to atomic columns
IN_PLACE_SETUPS = {"survey_image"}
STACKS_FILE = HERE / "stacks.yaml"
STAGE_SETTLE_S = 600.0  # longest move home; the mapping twin moves 5 um/s in real time
TRANSCRIPTS = HERE.parent / "outputs" / "live" / "transcripts"  # one file per failed run with its messages and calls
TRANSCRIPT_TEXT_CHARS = 4000
STACK_KEYS = {"config", "mcp", "operator", "tiled"}
STACK_DEFAULTS = {"mcp": DEFAULT_MCP_URL, "operator": default_operator_url, "tiled": DEFAULT_TILED_URI}
STACK: dict[str, str] = {**STACK_DEFAULTS, "name": "twin"}
SETUP_MAPS = {
    "map": (4.8e-7, 1.6e-7, ["eds_point"], True),
    "map_plain": (6.0e-7, 1.6e-7, [], False),
    "large_map": (6.0e-6, 4.0e-7, [], True),
}
TEAM_MODULE = "asyncroscopy.agents.graphs.team"  # planner, specialists and reporter
GOAL_MODULE = "asyncroscopy.agents.graphs.goal"  # check the goal from facts, retry with feedback
SINGLE_TURN = {"mapping", "team", "goal"}  # runners that take one request, not a conversation
REFERENCE = "reference"  # no model: the reference calls (only with --runners reference)
CHAT_TOOL_ROUNDS = 40  # tool calls one chat turn may make
EVALUATORS = ["time"]
SCORE_COLUMNS = ["time.clock_s", "time.idle_s"]


class WorkflowError(ValueError):
    pass


def _workflow_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        files += sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    return [f for f in files if f.resolve() != STACKS_FILE.resolve()]


def load_stacks(paths: list[Path]) -> dict[str, dict]:
    """Stacks from benchmarks/stacks.yaml and any ``stacks:`` in the workflow files, with default addresses."""
    stacks: dict[str, dict] = {}
    for file in [STACKS_FILE, *_workflow_files(paths)]:
        if not file.is_file():
            continue
        for name, spec in ((yaml.safe_load(file.read_text(encoding="utf-8")) or {}).get("stacks") or {}).items():
            if not isinstance(spec, dict) or "config" not in spec or set(spec) - STACK_KEYS:
                raise WorkflowError(f"{file.name}: stack {name!r} needs config: (a run_servers config) and may set {sorted(STACK_KEYS - {'config'})}")
            stacks[str(name)] = {**STACK_DEFAULTS, **{k: str(v) for k, v in spec.items()}, "name": str(name)}
    return stacks


def _check_params(where: str, spec: dict) -> None:
    """A check's parameters must fit its function, so a typo fails at load, not as a failed run."""
    fn = CHECKS[spec["check"]]
    params = {k: v for k, v in spec.items() if k not in ("check", "label")}
    try:
        inspect.signature(fn).bind(None, **params)
    except TypeError as exc:
        wanted = [p for p in inspect.signature(fn).parameters if p != "run"]
        raise WorkflowError(f"{where}: check {spec['check']} {exc}; its parameters: {wanted or 'none'}") from exc


def _names_used(code: Any) -> set[str]:
    found = set(code.co_names)
    for const in code.co_consts:
        if inspect.iscode(const):
            found |= _names_used(const)
    return found


def check_source(fn: Any) -> dict[str, str]:
    """{name: source} of a check and everything of workflow_checks it reaches by name."""
    sources: dict[str, str] = {}
    todo = [fn.__name__]
    while todo:
        name = todo.pop()
        if name in sources or name not in vars(workflow_checks):
            continue
        item = vars(workflow_checks)[name]
        if inspect.ismodule(item) or ((inspect.isfunction(item) or inspect.isclass(item)) and inspect.getmodule(item) is not workflow_checks):
            continue
        if inspect.isfunction(item):
            sources[name], codes = inspect.getsource(item), [item.__code__]
        elif inspect.isclass(item):
            sources[name] = inspect.getsource(item)
            codes = [m.__code__ for m in vars(item).values() if inspect.isfunction(m)]
        else:
            sources[name], codes = repr(item), []
        for code in codes:
            todo += sorted(_names_used(code))
    return dict(sorted(sources.items()))


def task_hash(entry: dict) -> str:
    """A benchmark's identity: its YAML entry as written and the source of its checks."""
    used = sorted({spec.get("check") for spec in entry.get("checks") or [] if isinstance(spec, dict) and spec.get("check") in CHECKS})
    return json_sha256({"entry": entry, "checks": {name: check_source(CHECKS[name]) for name in used}})


def load_workflows(paths: list[Path], stacks: dict[str, dict] | None = None) -> list[dict]:
    """Every workflow in the given YAML files (or directories of them), checked for typos."""
    stacks = load_stacks(paths) if stacks is None else stacks
    workflows, seen = [], set()
    for file in _workflow_files(paths):
        document = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        for i, wf in enumerate(document.get("workflows") or []):
            where = f"{file.name}: workflow {i + 1}"
            if not isinstance(wf, dict) or "name" not in wf:
                raise WorkflowError(f"{where}: each workflow needs a name")
            where = f"{file.name}: {wf['name']}"
            written = copy.deepcopy(wf)
            unknown = set(wf) - WORKFLOW_KEYS
            if unknown:
                raise WorkflowError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(WORKFLOW_KEYS)}")
            if wf["name"] in seen:
                raise WorkflowError(f"{where}: the name is used twice")
            seen.add(wf["name"])
            if ("request" in wf) == ("turns" in wf):
                raise WorkflowError(f"{where}: give either request or turns")
            for turn in wf.get("turns") or []:
                if not isinstance(turn, dict) or "say" not in turn or set(turn) - TURN_KEYS:
                    raise WorkflowError(f"{where}: each turn is {{say: ..., wait_for: ...}}")
            if set(wf.get("setup") or []) - SETUPS:
                raise WorkflowError(f"{where}: setup must be among {sorted(SETUPS)}")
            if wf.get("stack", "twin") not in stacks:
                raise WorkflowError(f"{where}: stack {wf.get('stack')!r} is not declared; known: {sorted(stacks)} "
                                    "(declare one in benchmarks/stacks.yaml or under stacks: in this file)")
            for spec in wf.get("checks") or []:
                if not isinstance(spec, dict) or spec.get("check") not in CHECKS:
                    raise WorkflowError(f"{where}: unknown check {spec.get('check') if isinstance(spec, dict) else spec!r}; known: {sorted(CHECKS)}")
                _check_params(where, spec)
            if not wf.get("checks"):
                raise WorkflowError(f"{where}: no checks")
            if wf.get("kind", KINDS[0]) not in KINDS:
                raise WorkflowError(f"{where}: kind must be one of {list(KINDS)}")
            if not isinstance(wf.get("held_out", False), bool):
                raise WorkflowError(f"{where}: held_out is true or false")
            if "bar" in wf:
                try:
                    wf["bar"] = parse_bar(wf["bar"])
                except ValueError as exc:
                    raise WorkflowError(f"{where}: {exc}") from exc
            reference = wf.get("reference")
            if reference is not None and (not isinstance(reference, dict) or set(reference) - REFERENCE_KEYS):
                raise WorkflowError(f"{where}: reference may set {sorted(REFERENCE_KEYS)}")
            reference_calls(wf)
            wf.setdefault("runners", ["general"])
            wf.setdefault("timeout_s", 600)
            wf.setdefault("stack", "twin")
            wf["file"] = file.name
            wf["task_hash"] = task_hash(written)
            wf["capability"] = file.stem  # the board groups by it
            workflows.append(wf)
    return workflows


def _fact(facts: dict, name: str) -> Any:
    value = facts.get(name)
    if value is None:  # a missing value is a broken setup, not an agent test
        raise InfraError(f"placeholder {{{name}}} has no value: the setup did not produce it")
    return value


def render(value: Any, facts: dict) -> Any:
    """Fill {name} placeholders from setup facts; a string that is only a placeholder keeps the fact's type.
    A placeholder whose fact is missing or None raises InfraError."""
    if isinstance(value, str):
        whole = re.fullmatch(r"\{(\w+)\}", value)
        if whole:
            return _fact(facts, whole.group(1))
        return re.sub(r"\{(\w+)\}", lambda m: str(_fact(facts, m.group(1))), value)
    if isinstance(value, list):
        return [render(v, facts) for v in value]
    if isinstance(value, dict):
        return {k: render(v, facts) for k, v in value.items()}
    return value


class Live:
    """The stack's endpoints, with tool names found from what they offer (never hard-coded)."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.instrument = config.instrument()
        self.analysis = config.analysis()
        from asyncroscopy.agents.mcp_tools import McpServer

        self.operator = McpServer(STACK["operator"], {"X-Asyncroscopy-Caller": "workflow-operator"})
        self.names: set[str] = set()
        self.operator_names: set[str] = set()
        self.risks: dict[str, str] | None = None  # from endpoint_info
        self.setup_cache: dict[str, dict] = {}

    async def discover(self) -> None:
        self.names = {t.name for t in await self.instrument.list_tools()}
        self.risks = await endpoint_risks(self.config)
        try:
            self.operator_names = {t.name for t in await self.operator.list_tools()}
        except Exception as exc:
            print(f"[workflows] operator endpoint {STACK['operator']} unavailable ({exc})", flush=True)

    def tool(self, *suffixes: str, prefix: str = "") -> str | None:
        return find_tool(self.names, *suffixes, prefix=prefix)

    async def text(self, name: str | None, args: dict | None = None, server: McpServer | None = None) -> str | None:
        if not name:
            return None
        try:
            return await (server or self.instrument).call_text(name, args or {})
        except Exception:
            return None

    async def required(self, name: str | None, args: dict | None = None, server: McpServer | None = None) -> str:
        """A call the benchmark itself depends on: a missing tool or a failed call is the stack's failure."""
        if not name:
            raise InfraError("a tool the benchmark needs is not offered")
        try:
            return await (server or self.instrument).call_text(name, args or {})
        except Exception as exc:
            raise InfraError(f"{name} failed: {type(exc).__name__}: {exc}") from exc

    async def snapshot(self) -> dict:
        stage = await self.text(self.tool("_get_stage"))
        fov = await self.text(self.tool("_get_fov"))
        defocus = await self.text(self.tool("_get_defocus"))
        return {"stage": _floats(stage), "fov": _float(fov), "defocus": _float(defocus)}

    async def restore(self, state: dict) -> None:
        """Put the stage, field of view and defocus back as they were when the suite started; InfraError if the
        instrument does not answer (a run after it would test a wedged device, not the agent).

        wait_for_stage comes first: a run may leave a long move under way (an agent drove the stage to its edge), and
        the move home is refused until it ends.
        """
        steps = [("_move_stage", {"position": state.get("stage")}, state.get("stage")),
                 ("_set_fov", {"fov": state.get("fov")}, state.get("fov")), ("_set_defocus", {"defocus": state.get("defocus")}, state.get("defocus"))]
        await self.wait_for_stage()
        for suffix, args, value in steps:
            if value is None or not self.tool(suffix):
                continue
            try:
                await self.instrument.call_text(self.tool(suffix), args)
            except Exception as exc:
                raise InfraError(f"re-homing failed: {self.tool(suffix)}: {str(exc)[:200]}") from exc
        await self.wait_for_stage()

    async def wait_for_stage(self, timeout_s: float = STAGE_SETTLE_S) -> None:
        """Wait until a long stage move (stage_moving in get_parameters) has ended: the next run's setup map, or the
        agent's first move, would otherwise be refused while the re-homing move is still under way."""
        if not self.tool("_get_parameters"):
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                parameters = await self.instrument.call_json(self.tool("_get_parameters"), {})
                if isinstance(parameters, str):
                    parameters = json.loads(parameters)
            except Exception:
                return  # nothing to wait on
            if not (isinstance(parameters, dict) and parameters.get("stage_moving")):
                return
            await asyncio.sleep(1.0)
        raise InfraError(f"the stage was still moving {timeout_s:.0f} s after re-homing")

    async def run_map(self, plan: dict, timeout_s: float = 1800) -> dict:
        """Run a Mapper job to the end with whichever job tools the endpoint offers; returns its results."""
        mapper = next(iter(devices_of_kind(await self.instrument.call_json("endpoint_info"), "map")), None)
        if "run_job" in self.names:
            started = await self.instrument.call_json("run_job", {"device": mapper or "Mapper", "plan": plan})
            reservation = started.get("reservation_id")
        else:
            reserved = await self.instrument.call_json(f"{mapper}_reserve", {"request": json.dumps({"plan": plan})})
            if "error" in reserved:
                raise RuntimeError(f"setup map refused: {reserved['error']}")
            reservation = reserved["reservation_id"]
            started = await self.instrument.call_json(f"{mapper}_start", {"request": json.dumps({"reservation_id": reservation})})
        if "error" in started:
            raise RuntimeError(f"setup map did not start: {started['error']}")
        job_id = started["job_id"]
        deadline = time.time() + timeout_s
        status: dict = {}
        while time.time() < deadline:
            if "wait_for_job" in self.names:
                status = await self.instrument.call_json("wait_for_job", {"device": mapper or "Mapper", "job_id": job_id})
            else:
                status = await self.instrument.call_json(f"{mapper}_job_status", {"request": json.dumps({"job_id": job_id})})
                await asyncio.sleep(2)
            if status.get("state") in {"done", "failed", "aborted"}:
                break
        if status.get("state") != "done":
            raise RuntimeError(f"setup map ended {status.get('state')}: {status.get('last_error')}")
        results = status if "result_keys" in status else {}
        if f"{mapper}_results" in self.names:
            results = await self.instrument.call_json(f"{mapper}_results", {"request": json.dumps({"job_id": job_id})})
        if reservation and f"{mapper}_release" in self.names:
            await self.text(f"{mapper}_release", {"request": json.dumps({"reservation_id": reservation})})
        return results

    async def tidy(self, job_events: list[dict]) -> None:
        """Abort and release whatever jobs a run left behind (a reservation is the run id), so the
        next workflow does not find the Mapper busy."""
        last: dict[str, dict] = {}
        for e in job_events:
            if e.get("run_id"):
                last[e["run_id"]] = e
        for run_id, event in last.items():
            device = str(event.get("action", "Mapper.")).split(".", 1)[0]
            if not str(event.get("action", "")).endswith((".done", ".aborted", ".failed")) and event.get("job_id"):
                await self.text(f"{device}_abort", {"request": json.dumps({"job_id": event["job_id"]})})
                await asyncio.sleep(2)
            await self.text(f"{device}_release", {"request": json.dumps({"reservation_id": run_id})})

    async def analysis_job(self, tool: str, args: dict, timeout_s: float = 900) -> dict:
        started = await self.analysis.call_json(tool, args)
        if "job_id" not in started:
            raise RuntimeError(f"{tool} refused: {started}")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = await self.analysis.call_json("analysis_job_status", {"job_id": started["job_id"], "wait_s": 25})
            if status.get("state") in {"done", "failed", "aborted"}:
                break
        return await self.analysis.call_json("analysis_job_results", {"job_id": started["job_id"]})

    async def map_info(self, map_key: str) -> dict:
        try:
            info = await self.analysis.call_json("map_info", {"map": map_key})
        except Exception:
            return {}
        return info if "stage_box_m" in info else {}

    async def poi_records(self, map_key: str) -> list[dict] | None:
        """The map's points of interest as list_pois returns them, or None if there is no such map."""
        try:
            listed = await self.analysis.call_json("list_pois", {"map": map_key})
        except Exception:
            return None
        items = _poi_items(listed)
        return [p for p in items if isinstance(p, dict) and "x_m" in p] if isinstance(items, list) else None

    async def ground_truth(self) -> dict | None:
        """The twin's hidden truth (true stage pose of every acquisition), from the operator endpoint."""
        truth, _ = await twin_truth(self.operator)
        return truth

    async def pois(self, map_key: str) -> list[list[float]]:
        return [[p["x_m"], p["y_m"]] for p in await self.poi_records(map_key) or []]

    async def health(self) -> str:
        """"" when the instrument answers a plain read; otherwise why not (a held monitor, a timeout)."""
        tool = self.tool("_get_stage")
        if not tool:
            return ""
        try:
            await self.instrument.call_text(tool, {})
            return ""
        except Exception as exc:
            return f"{tool} did not answer: {str(exc)[:200]}"

    async def setup(self, name: str) -> dict:
        if name in self.setup_cache and name not in IN_PLACE_SETUPS:
            return self.setup_cache[name]
        if name in ("image", "survey_image"):  # survey_image uses SURVEY_FOV_M
            if name == "survey_image":
                await self.text(self.tool("_set_fov"), {"fov": SURVEY_FOV_M})
            await self.required(self.tool("_unblank_beam"), {})  # a blanked beam images noise
            try:
                key = await self.required(self.tool("_acquire_scanned_image"), {"detector_list": ["haadf"]})
            finally:
                await self.text(self.tool("_blank_beam"), {})
            facts = {"image_key": key, "keys": [key] if key else []}
        else:
            size, fov, per_tile, scored = SETUP_MAPS[name]
            plan = {"size_m": [size, size], "fov_m": fov, "per_tile": per_tile, "eds_exposure_s": 0.2}
            results = await self.run_map(plan, timeout_s=7200)
            run_key = results.get("result_keys", {}).get("run")
            map_key = results.get("result_keys", {}).get("map")  # the Mapper job stitches its own run
            if not map_key:
                stitched = await self.analysis_job("assemble_map", {"run": run_key}, timeout_s=3600)
                map_key = stitched["result_keys"]["map"]
            if scored:
                await self.analysis_job("score_map", {"map": map_key, "top_k": 3})
            pois = await self.pois(map_key)
            info = await self.map_info(map_key)
            spectra = (results.get("summary", {}).get("spectra") or {}).get("first_keys", [])
            tiles = len(results.get("steps", [])) or (results.get("summary", {}).get("grid") or {}).get("rows", 0) * (results.get("summary", {}).get("grid") or {}).get("cols", 0)
            poi = pois[0] if pois else [None, None]
            facts = {
                "run_key": run_key, "map_key": map_key, "pois": pois, "spectrum_key": spectra[0] if spectra else None,
                "poi_x": poi[0], "poi_y": poi[1], "keys": [k for k in (run_key, map_key, *spectra[:1]) if k],
                "map_box": info.get("stage_box_m"), "setup_map": map_key,
                "poi_x_nm": round(poi[0] * 1e9, 1) if poi[0] is not None else None,
                "poi_y_nm": round(poi[1] * 1e9, 1) if poi[1] is not None else None,
                "truth": {"tiles": tiles, "poi_x_nm": round(poi[0] * 1e9, 1) if poi[0] is not None else None},
            }
        self.setup_cache[name] = facts
        return facts


class InfraError(RuntimeError):
    """The stack, not the agent, failed: a wedged device, a setup that produced nothing. Such rows are not counted."""


def _poi_items(listed: Any) -> list | None:
    """The points of a list_pois reply: {"total", "pois"} from the analysis server, {"result": [...]} or a plain list."""
    if isinstance(listed, dict):
        listed = listed.get("pois", listed.get("result"))
    return listed if isinstance(listed, list) else None


def _float(text: str | None) -> float | None:
    try:
        return float(json.loads(text)) if text is not None else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _floats(text: str | None) -> list[float] | None:
    try:
        value = json.loads(text) if text else None
    except json.JSONDecodeError:
        return None
    return [float(v) for v in value] if isinstance(value, list) else None


def data_truth(keys: list[str]) -> dict:
    """Peaks and candidate elements of each 1-D dataset among ``keys``, read with the data layer."""
    from asyncroscopy.data import tiled_client, views
    from asyncroscopy.data.kinds import eds

    uri = STACK["tiled"]
    spectra, strongest = {}, None
    for key in keys[-6:]:
        try:
            node = tiled_client.node_for_key(key, uri)
            ds = views.pick(node, key)
            if ds.ndim != 1:
                continue
            values = np.asarray(ds.node.read(), dtype=np.float64)
            axis = views.energy_axis({**ds.parent_attrs, **ds.attrs})
        except Exception as exc:  # no truth: checks that need it fail with the reason
            spectra[key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        line_axis = axis or views.Axis(0.0, 1.0, "channel", "")
        positions = line_axis.offset + line_axis.scale * np.arange(len(values), dtype=np.float64)
        xray = line_axis.units == "eV" and eds.is_xray(ds)
        rows = eds.line_peaks(positions, values, line_axis) if xray else views.line_peaks(positions, values, line_axis)
        strongest = rows[0][0] if rows else float(positions[int(np.argmax(values))])
        entry = {"strongest": strongest, "units": line_axis.units, "peaks": [round(row[0], 4) for row in rows]}
        if xray:
            entry["peaks_ev"] = [round(row[0], 1) for row in rows]
            likely = eds.likely_elements(rows)  # the elements view_data names, not every fitting line
            entry["elements"] = sorted(part.split(" ")[0] for part in likely.split(", ")) if likely else []
            entry["could_be"] = sorted({el for row in rows for el in eds.note_elements(row[4])})
        spectra[key] = entry
    errors = {k: v["error"] for k, v in spectra.items() if "error" in v}
    return {"spectra": {k: v for k, v in spectra.items() if "error" not in v}, "spectrum_errors": errors, "strongest_peak": strongest}


def produced_keys(run: Run, pattern: str = "*_acquire_*") -> list[str]:
    """Keys the run's acquisitions and map steps produced, newest last."""
    keys = []
    for e in run.events:
        if e.get("source") == "mcp" and e.get("ok") and re.search(r"_acquire_", str(e.get("action"))):
            key = e.get("result_key") or e.get("result")
            if isinstance(key, str) and key:
                keys.append(key)
        if e.get("source") == "job" and str(e.get("action", "")).endswith(".step") and e.get("ok"):
            try:
                step = json.loads(e.get("result") or "{}")
            except json.JSONDecodeError:
                continue
            if step.get("spectrum_key") and step["spectrum_key"] != "eds_map":
                keys.append(step["spectrum_key"])
    return keys


def run_keys(job_events: list[dict]) -> list[str]:
    """Run-file keys of the finished jobs among ``job_events``."""
    keys = []
    for e in job_events:
        if not str(e.get("action", "")).endswith(".done"):
            continue
        result = e.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {}
        if isinstance(result, dict) and result.get("run"):
            keys.append(result["run"])
    return keys


def stitch_truth(map_meta: dict, tile_shape: tuple[int, int], steps: list[dict], truth: dict) -> dict:
    """How far the stitched map put each tile from where the twin really imaged it (pixels), with the
    common offset removed, as tests/analysis/test_map_end_to_end.py measures it; and the same for stage readings alone."""
    positions = np.asarray(map_meta.get("tile_positions_px") or [], dtype=np.float64)
    pixel = float(map_meta["pixel_size_m"])
    images = [a for a in truth.get("acquisitions", []) if a.get("kind") == "image" and a.get("center_true_m") and a.get("reported_m")]
    true_centres, reported, estimated = [], [], []
    for step in steps:
        index, read = step.get("index"), step.get("stage_reported_m")
        if index is None or read is None or index >= len(positions):
            continue
        match = next((a for a in reversed(images) if np.allclose(a["reported_m"][:2], read[:2], rtol=0, atol=1e-15)), None)
        if match is None:
            continue
        true_centres.append([match["center_true_m"][1] / pixel, match["center_true_m"][0] / pixel])  # as row, col
        reported.append([read[1] / pixel, read[0] / pixel])
        estimated.append(positions[index] + np.asarray(tile_shape, dtype=np.float64) / 2.0)
    if len(true_centres) < 2:
        return {"error": f"matched only {len(true_centres)} of {len(steps)} tiles to the twin's ground truth"}
    stage = median_error(reported, true_centres)
    return {"median_px": median_error(estimated, true_centres), "stage_median_px": stage, "stage_error_fov": stage / float(tile_shape[1]), "tiles": len(true_centres)}


def map_metadata_and_tile_shape(map_key: str) -> tuple[dict, tuple[int, int]]:
    """A stitched map's metadata and the shape of its run's tiles, read from Tiled."""
    from asyncroscopy.data import tiled_client

    uri = STACK["tiled"]
    meta = dict(tiled_client.node_for_key(map_key, uri).metadata)
    tiles = tiled_client.node_for_key(meta["run"], uri)["tiles"]
    first = next(iter(tiles))
    return meta, tuple(tiles[first].structure().shape[:2])


def runners_available() -> dict[str, Any]:
    """Runners this checkout offers; team and goal are found by import, so older checkouts still run."""
    found = {"general": "asyncroscopy.agents.graphs.general:make_graph", "mapping": "mapping-graph"}
    for runner, module, attribute in (("team", TEAM_MODULE, "TeamGraph"), ("goal", GOAL_MODULE, "GoalGraph")):
        try:
            if hasattr(importlib.import_module(module), attribute):
                found[runner] = f"{module}:{attribute}"
        except ImportError:
            continue
    return found


def runners_for(wf: dict, available: dict) -> list[str]:
    """A workflow's own runners, plus the multi-agent and --graph runners wherever ``general`` runs."""
    runners = list(wf["runners"])
    for extra in ("team", "goal", "custom"):
        if extra in available and "general" in runners and extra not in runners and not (extra in SINGLE_TURN and wf.get("turns")):
            runners.append(extra)
    return runners


def calls_from_trace(trace: list) -> list[Call]:
    """Calls recorded by the team and goal graphs ({name, args, result, error})."""
    return [Call.from_dict(c) for c in trace or []]


def skills_of(applied: list | None) -> list[str]:
    """Skills whose instructions reached the agent's prompt in full (one cut to its description does not count)."""
    return [a.get("skill") for a in applied or [] if isinstance(a, dict) and a.get("skill") and a.get("in_prompt", True)]


async def run_chat_graph(factory: str, config: AgentConfig, wf: dict, facts: dict, caller: str, started: float) -> tuple[list, str, list[str]]:
    """(messages, final answer, skills applied) of a chat graph over the workflow's turns."""
    module, _, function = factory.partition(":")
    make_graph = getattr(importlib.import_module(module), function or "make_graph")
    graph = await make_graph({"configurable": {"model": config.model, "caller": caller, "agent_config": config}})
    from langchain_core.messages import HumanMessage

    turns = wf.get("turns") or [{"say": wf["request"]}]
    messages: list = []
    skills: list[str] = []
    for turn in turns:
        await wait_for(turn.get("wait_for"), started, caller)
        messages = messages + [HumanMessage(render(turn["say"], facts))]
        state = await graph.ainvoke({"messages": messages}, {"recursion_limit": recursion_limit(CHAT_TOOL_ROUNDS)})
        messages = state["messages"]
        skills += [s for s in skills_of(state.get("skills_applied")) if s not in skills]
    final = messages[-1].content if messages else ""
    return messages, final if isinstance(final, str) else json.dumps(final, default=str), skills


async def wait_for(condition: dict | None, started: float, caller: str) -> None:
    """Hold the next turn until the run's job has done ``job_steps_at_least`` tiles (or the timeout)."""
    if not condition:
        return
    deadline = time.time() + float(condition.get("timeout_s", 300))
    need = int(condition.get("job_steps_at_least", 0))
    while time.time() < deadline:
        events = read_trace(started, [caller])
        steps = sum(1 for e in events if str(e.get("action", "")).endswith(".step"))
        if steps >= need:
            if any(str(e.get("action", "")).endswith(".done") for e in events):
                raise WorkflowError(f"the job finished ({steps} steps) before the next turn could act on it: make the plan bigger")
            return
        await asyncio.sleep(2)


async def ensure_healthy(live: Live, baseline: dict) -> None:
    """Before each run: the instrument answers a plain read, or is re-homed until it does; else InfraError
    (a wedged device from an earlier run must not fail the next workflow)."""
    why = await live.health()
    if not why:
        return
    for _ in range(3):
        await asyncio.sleep(10)
        if not await live.health():
            await live.restore(baseline)
            return
    raise InfraError(f"the instrument is not answering: {why}")


async def _prepare(live: Live, wf: dict, baseline: dict) -> dict:
    """Run the setups of a workflow and restore the baseline."""
    facts: dict = {}
    setups = wf.get("setup") or []
    in_place = bool(IN_PLACE_SETUPS & set(setups))
    if in_place:
        await live.restore(baseline)
    try:
        for name in setups:
            facts.update(await live.setup(name))
    except Exception as exc:
        raise InfraError(f"setup failed: {type(exc).__name__}: {exc}") from exc
    if not in_place:
        await live.restore(baseline)
    return facts


async def _run_team(config: AgentConfig, wf: dict, request: str, run: Run) -> None:
    """Run the request on the team graph and keep its answer."""
    from asyncroscopy.agents.graphs.team import TeamGraph

    state = await asyncio.wait_for(TeamGraph(config).build().ainvoke({"request": request}, {"recursion_limit": 100}), wf["timeout_s"])
    run.final, run.calls = str(state.get("answer", "")), calls_from_trace(state.get("trace"))
    run.skills = skills_of(state.get("skills_applied"))


async def _run_goal(config: AgentConfig, wf: dict, request: str, run: Run) -> None:
    """Run the request on the goal graph and keep its answer."""
    from asyncroscopy.agents.goals import Goal
    from asyncroscopy.agents.graphs.goal import GoalGraph

    goal = Goal.from_text(request, max_seconds=float(wf["timeout_s"]))
    state = await asyncio.wait_for(GoalGraph(goal, "team", config).build().ainvoke({"goal": goal.to_dict()}, {"recursion_limit": 4 * goal.budget.max_attempts + 4}), wf["timeout_s"] * 1.2)
    run.final, run.calls = str(state.get("report") or state.get("answer", "")), calls_from_trace(state.get("trace"))
    run.skills = skills_of(state.get("skills_applied"))


async def _run_chat(factory: str, config: AgentConfig, wf: dict, facts: dict, caller: str, started: float, run: Run, transcript: list) -> None:
    """Run the request on a chat graph and keep its answer and messages."""
    messages, run.final, run.skills = await asyncio.wait_for(run_chat_graph(factory, config, wf, facts, caller, started), wf["timeout_s"])
    run.calls = trace_from_messages(messages)
    transcript[:] = [transcript_entry(m) for m in messages]
    if not (run.final or "").strip():  # keep the model's last message to see why
        facts["last_messages"] = [message_summary(m) for m in messages[-3:]]


def reference_calls(wf: dict) -> list[dict]:
    """The calls of a workflow's reference, each {tool, args, result, ok}; WorkflowError for any other shape."""
    calls = (wf.get("reference") or {}).get("calls") or []
    for call in calls:
        if not isinstance(call, dict) or "tool" not in call or set(call) - {"tool", "args", "result", "ok"}:
            raise WorkflowError(f"{wf['name']}: each reference call is {{tool: ..., args: ..., result: ..., ok: ...}}, not {call!r}")
    return calls


async def replay_reference(config: AgentConfig, wf: dict, run: Run) -> None:
    """Make the reference's calls on the stack in order, with no model, and answer with the reference's answer.

    It shows the task can be done on this stack and that the checks pass on what the stack really produced. A call
    goes to the instrument endpoint when it offers the tool, else to the analysis server. ``{result_N}`` in a later
    call's args or in the answer is the first line of call N's reply (an acquisition's key). A call the stack refuses
    is recorded as failed and the replay goes on, so a reference that expects a refusal replays too."""
    from asyncroscopy.agents.mcp_tools import result_text

    instrument, analysis = config.instrument(), config.analysis()
    offered = {t.name for t in await instrument.list_tools()}
    results: dict[str, str] = {}
    for i, call in enumerate(reference_calls(wf), 1):
        name, args = str(call["tool"]), render(dict(call.get("args") or {}), results)
        if name not in offered and not name[:1].islower():
            same = sorted(n for n in offered if not n[:1].islower() and n.partition("_")[2] == name.partition("_")[2])
            name = same[0] if len(same) == 1 else name
        on_instrument = name in offered or analysis is None or not name[:1].islower()  # an unknown device tool fails there
        reply = await (instrument if on_instrument else analysis).call(name, args)
        text = result_text(reply)
        run.calls.append(Call(name, args, text, bool(reply.is_error)))
        results[f"result_{i}"] = text.partition("\n")[0].strip()
    run.final = render(str(wf["reference"].get("answer", "")), results)


async def _run_graph(mapping_graph: Any, runner: str, factory: str, config: AgentConfig, wf: dict, facts: dict, caller: str, started: float, run: Run, transcript: list) -> None:
    """Run the request of a workflow on the graph its runner names."""
    request = render(wf.get("request") or "", facts)
    if runner == REFERENCE:
        await asyncio.wait_for(replay_reference(config, wf, run), wf["timeout_s"])
    elif runner == "mapping":
        state = await asyncio.wait_for(mapping_graph(config).build().ainvoke({"request": request}), wf["timeout_s"])
        run.final = str(state.get("report") or state.get("error") or "")
    elif runner == "team":
        await _run_team(config, wf, request, run)
    elif runner == "goal":
        await _run_goal(config, wf, request, run)
    else:
        await _run_chat(factory, config, wf, facts, caller, started, run, transcript)


async def _collect_trace(live: Live, run: Run, facts: dict, started: float, caller: str) -> list[str]:
    """Read the trace of a run, tidy its jobs and give its run keys."""
    facts["after"] = await live.snapshot()
    run.events = read_trace(started, [caller])
    facts["all_job_events"] = [e for e in read_trace(started) if e.get("source") == "job"]
    mine = {e.get("run_id") for e in run.events if e.get("source") == "job"}
    await live.tidy([e for e in facts["all_job_events"] if e.get("run_id") in mine])
    return run_keys([e for e in facts["all_job_events"] if e.get("run_id") in mine])


async def _collect_pois(live: Live, wf: dict, run: Run, facts: dict, mine_runs: list[str]) -> list[str]:
    """Record the points of every map a run touched and give the maps it stitched."""
    map_keys = {m for c in run.calls for m in MAP_KEY.findall(c.result)} | set(MAP_KEY.findall(run.final))
    map_keys |= {f"{Path(r).stem}_map" for r in mine_runs}
    if facts.get("setup_map"):
        map_keys.add(facts["setup_map"])
    facts["poi_records"], facts["map_info"] = {}, {}
    for key in sorted(map_keys):
        records = await live.poi_records(key)
        if records is not None:
            facts["poi_records"][key] = records
            facts["map_info"][key] = await live.map_info(key)
    if not wf.get("setup"):
        facts["pois"] = [[p["x_m"], p["y_m"]] for records in facts["poi_records"].values() for p in records if str(p.get("source", "")).startswith("scorer")] or facts.get("pois", [])
    return [k for k in facts["poi_records"] if any(k == f"{Path(r).stem}_map" for r in mine_runs)]


async def _stitch_results(live: Live, wf: dict, run: Run, stitched_here: list[str]) -> dict:
    """Compare each map a run stitched with the ground truth."""
    stitch = {}
    if stitched_here and any(c.get("check") == "stitch_accuracy" for c in wf["checks"]):
        truth_doc = await live.ground_truth()
        steps = [(e.get("run_id"), json.loads(e["result"])) for e in run.events if str(e.get("action", "")).endswith(".step") and e.get("ok") and e.get("result")]
        for key in stitched_here if truth_doc is not None else []:
            try:
                meta, shape = map_metadata_and_tile_shape(key)
                stitch[key] = stitch_truth(meta, shape, [step for run_id, step in steps if f"map_{run_id}" == Path(meta["run"]).stem], truth_doc)
            except Exception as exc:  # so the check fails with the reason
                stitch[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return stitch


def _run_checks(wf: dict, runner: str, run: Run, facts: dict, stitch: dict) -> dict:
    """Add the truth of a run to its facts and evaluate its checks."""
    truth_keys = produced_keys(run) + [k for k in (facts.get("spectrum_key"),) if k]
    truth = data_truth(truth_keys) if truth_keys else {}
    facts["truth"] = {**facts.get("truth", {}), **truth, **({"stitch": stitch} if stitch else {})}
    checks = evaluate(run, render(wf["checks"], {**facts, **facts.get("truth", {})}))
    if runner != "mapping" and not (run.final or "").strip():
        checks["answered"] = {"ok": False, "detail": "the agent gave no final answer"}
    return checks


def _run_scores(caller: str, run: Run) -> dict:
    """Score a run with the extra evaluators."""
    try:  # extra columns only; a failing evaluator leaves the verdict
        return evaluator_scores(RunData(caller, run.events, tiled_uri=STACK.get("tiled")), EVALUATORS)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _save_transcript(caller: str, wf: dict, runner: str, error: str | None, checks: dict, run: Run, transcript: list) -> None:
    """Write the transcript of a failed run to a file."""
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPTS / f"{caller}.json").write_text(json.dumps({
        "workflow": wf["name"], "runner": runner, "error": error, "checks": checks, "final": run.final,
        "messages": transcript, "calls": [c.to_dict() for c in run.calls],
    }, indent=1, default=str), encoding="utf-8")


async def run_once(live: Live, wf: dict, runner: str, factory: str, baseline: dict) -> dict:
    from asyncroscopy.agents.graphs.mapping import MappingGraph

    facts = await _prepare(live, wf, baseline)
    caller = unique_caller(f"wf-{wf['name']}-{runner}")
    config = live.config.for_run(caller=caller)
    facts["before"] = await live.snapshot()
    started = time.time()
    run = Run(facts=facts, has_calls=runner != "mapping", risks=live.risks)
    error, transcript = None, []
    try:
        await _run_graph(MappingGraph, runner, factory, config, wf, facts, caller, started, run, transcript)
    except TimeoutError:
        error = f"timed out after {wf['timeout_s']} s"
    except InfraError:
        raise
    except Exception as exc:  # crash or recursion limit: a failed run
        error = f"{type(exc).__name__}: {exc}"
    seconds = round(time.time() - started, 1)
    if error and len(config.ledger) > len(run.calls):  # stopped part way: keep every call whose reply arrived
        run.calls = [Call.from_dict(c) for c in config.ledger]
    await asyncio.sleep(1.0)  # let the last trace lines land
    mine_runs = await _collect_trace(live, run, facts, started, caller)
    stitched_here = await _collect_pois(live, wf, run, facts, mine_runs)
    stitch = await _stitch_results(live, wf, run, stitched_here)
    checks = _run_checks(wf, runner, run, facts, stitch)
    names = [c.name for c in run.calls] or [str(e["action"]) for e in run.events if e.get("source") == "mcp"]
    scores = _run_scores(caller, run)
    ok = error is None and all(c["ok"] for c in checks.values())
    if not ok:
        _save_transcript(caller, wf, runner, error, checks, run, transcript)
    return {
        "ok": ok,
        "error": error,
        "infra_error": is_infra_error(error),
        "seconds": seconds,
        "caller": caller,
        "checks": checks,
        "tool_calls": len(names),
        "tool_names": names,
        "sleepwalking": unrequested(run, workflow_allow(wf)),  # this workflow's own allow list
        "skills": run.skills,
        "final": (run.final or "")[-600:],
        "scores": scores,
        **({"last_messages": facts["last_messages"]} if facts.get("last_messages") else {}),
    }


def transcript_entry(message: BaseMessage) -> dict:
    """A message as a failed run's transcript keeps it: type, tool name, text, the calls it made with their arguments,
    and why the model stopped (an empty reply right after the model server reloaded says "load", not "stop")."""
    content = message.content if isinstance(message.content, str) else json.dumps(message.content, default=str)
    metadata = getattr(message, "response_metadata", None) or {}
    return {
        "type": getattr(message, "type", ""), "name": getattr(message, "name", None), "text": content[:TRANSCRIPT_TEXT_CHARS],
        "tool_calls": [{"name": c.get("name"), "args": c.get("args")} for c in getattr(message, "tool_calls", None) or []],
        "stop": metadata.get("done_reason") or metadata.get("finish_reason"),
    }


def message_summary(message: BaseMessage) -> dict:
    """A message's type, the end of its text and the tools it called: enough to tell an empty answer's cause."""
    content = message.content if isinstance(message.content, str) else json.dumps(message.content, default=str)
    return {
        "type": getattr(message, "type", ""), "text": content[-300:],
        "tool_calls": [c.get("name") for c in getattr(message, "tool_calls", None) or []],
        "stop": (getattr(message, "response_metadata", None) or {}).get("done_reason") or (getattr(message, "response_metadata", None) or {}).get("finish_reason"),
    }


def score_median(rows: list[dict], column: str) -> str:
    """Median of one evaluator objective ("time.idle_s") over a workflow's rows, for the Markdown table."""
    evaluator, _, objective = column.partition(".")
    values = sorted(
        v for v in ((r.get("scores") or {}).get(evaluator, {}).get(objective, {}).get("value") for r in rows if isinstance(r.get("scores"), dict))
        if isinstance(v, (int, float))
    )
    return f"{values[len(values) // 2]:.3g}" if values else "-"


def markdown(results: dict) -> str:
    columns = results.get("score_columns") or []
    lines = [
        f"# Workflow benchmark: {results['model']}, stack {results['stack']}, {results.get('git', '?')}",
        "",
        "| workflow | runner | passed | median s | failed checks |" + "".join(f" {c} |" for c in columns),
        "|---|---|---|---|---|" + "---|" * len(columns),
    ]
    for key, value in results["workflows"].items():
        name, runner = key.rsplit("/", 1)
        failed = sorted({label for r in value["rows"] for label, c in r["checks"].items() if not c["ok"]} | {"error" for r in value["rows"] if r.get("error")})
        seconds = sorted(r["seconds"] for r in value["rows"]) or [0]
        extra = "".join(f" {score_median(value['rows'], c)} |" for c in columns)
        lines.append(f"| {name} | {runner} | {value['passed']}/{value['of']} | {seconds[len(seconds) // 2]:.0f} | {', '.join(failed) or '-'} |{extra}")
    lines += ["", f"Passed {results['passed']} of {results['of']} runs; sleepwalking calls: {results['sleepwalking_calls']}."]
    return "\n".join(lines) + "\n"


def evaluator_versions(names: list[str]) -> dict[str, str]:
    """{name: version} of the evaluators a run scores with."""
    known = known_evaluators()
    return {name: known[name].version for name in names}


def apply_skills(choice: str | None, off: str | None, environ: dict | None = None) -> dict:
    """Set ASYNCROSCOPY_SKILLS and ASYNCROSCOPY_SKILLS_OFF from --skills and --skills-off; returns the results' ``skills``."""
    environ = os.environ if environ is None else environ
    hidden = [name.strip() for name in (off or "").split(",") if name.strip()]
    if choice == "off":
        environ["ASYNCROSCOPY_SKILLS"] = "0"
    elif choice is not None:
        environ["ASYNCROSCOPY_SKILLS"] = "1"
        if choice != "on":
            wanted = {name.strip() for name in choice.split(",") if name.strip()}
            environ["ASYNCROSCOPY_SKILLS_OFF"] = ""
            known = {skill.name for skill in list_skills()}
            if wanted - known:
                raise WorkflowError(f"--skills names no skill {sorted(wanted - known)}; skills here: {sorted(known)}")
            hidden += sorted(known - wanted)
    if off is not None or choice not in (None, "on", "off"):
        environ["ASYNCROSCOPY_SKILLS_OFF"] = ",".join(hidden)
    now = settings.get()
    return {"on": now.skills, "off": list(now.skills_off), "enabled": [skill.name for skill in list_skills() if not skill.off]}


async def main(args: argparse.Namespace) -> int:
    """It refuses a stack that is not simulated unless --allow-hardware is given, because a benchmark sends an agent's
    requests to the instrument.
    """
    from asyncroscopy.agents.config import AgentConfig

    stacks = load_stacks(args.workflows)
    if args.stack not in stacks:
        print(f"no stack {args.stack!r}; declared: {sorted(stacks)} (benchmarks/stacks.yaml, or stacks: in a workflow file)")
        return 2
    STACK.update(stacks[args.stack])
    env = settings.get()
    for field, value in (("operator", env.operator_url), ("tiled", env.tiled_uri)):
        if value:
            STACK[field] = value
    on_stack = [wf for wf in load_workflows(args.workflows, stacks) if wf["stack"] == args.stack]
    unknown = sorted(set(args.only or ()) - {wf["name"] for wf in on_stack})
    if unknown:
        print(f"no workflow {unknown} on stack {args.stack!r} (names are separated by spaces, not commas); --list shows them")
        return 2
    workflows = [wf for wf in on_stack if (not args.only or wf["name"] in args.only) and not (args.no_held_out and wf.get("held_out"))]
    if not (HERE.parent / STACK["config"]).exists() and not Path(STACK["config"]).exists():
        print(f"stack {args.stack} needs {STACK['config']}, which this checkout does not have: fix config: in its stacks: entry")
        return 2
    try:
        skills = apply_skills(args.skills, args.skills_off)
    except WorkflowError as exc:
        print(exc)
        return 2
    available = runners_available()
    if args.graph:
        available["custom"] = args.graph
    if args.list:
        for wf in workflows:
            runners = runners_for(wf, available)
            bar = parse_bar(wf.get("bar"))
            tags = f"{wf.get('kind', KINDS[0])}{', held out' if wf.get('held_out') else ''}"
            print(f"{wf['name']:32s} {','.join(runners):22s} timeout {wf['timeout_s']:5d} s  bar {bar['pass']}/{bar['of']}  {tags:24s} {wf['file']}")
        print(f"{len(workflows)} workflows on stack {args.stack}; runners here: {sorted(available)}")
        return 0
    config = AgentConfig.from_env(model=args.model, caller="workflows")
    if not env.mcp_url:
        config.mcp_url = STACK["mcp"]
    if not (await config.info()).get("simulated") and not args.allow_hardware:
        print(f"refusing to run: the endpoint {config.mcp_url} does not say its stack is simulated (endpoint_info). "
              "Start a twin stack, or pass --allow-hardware to benchmark on a real instrument.")
        return 2
    live = Live(config)
    await live.discover()
    await warm_up(config)
    baseline = await live.snapshot()
    EVALUATORS[:] = [n.strip() for n in args.evaluators.split(",") if n.strip()]
    unknown_evaluators = sorted(set(EVALUATORS) - set(known_evaluators()))
    if unknown_evaluators:
        print(f"no evaluator {unknown_evaluators}; registered: {sorted(known_evaluators())}")
        return 2
    SCORE_COLUMNS[:] = [c.strip() for c in args.score_columns.split(",") if c.strip()]
    if args.n != "bar" and not (args.n.isdigit() and int(args.n) >= 1):
        print(f"--n is a number of trials or 'bar', not {args.n!r}")
        return 2
    replaying = REFERENCE in (args.runners or [])
    results = {"model": args.model, "stack": args.stack, "git": git_head(), "n": args.n, "started": time.time(), "runners": available,
               "evaluators": evaluator_versions(EVALUATORS), "score_columns": SCORE_COLUMNS, "skills": skills,
               "stack_fingerprint": (await config.info()).get("stack_fingerprint"), "workflows": {}}
    out = Path(args.out or HERE / "results" / f"workflows_{args.stack}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    for wf in workflows:
        for runner in runners_for(wf, available) + ([REFERENCE] if replaying and wf.get("reference") else []):
            if args.runners and runner not in args.runners:
                continue
            if runner != REFERENCE and (runner not in available or (runner in SINGLE_TURN and wf.get("turns"))):
                why = f"no {runner} runner in this checkout" if runner not in available else f"{runner} takes one request; this workflow has turns"
                results["workflows"][f"{wf['name']}/{runner}"] = {"passed": 0, "of": 0, "skipped": why, "rows": [], "task_hash": wf["task_hash"]}
                continue
            rows = []
            bar = {"pass": 1, "of": 1} if runner == REFERENCE else parse_bar(wf.get("bar"))
            for i in range(1 if runner == REFERENCE else bar["of"] if args.n == "bar" else int(args.n)):
                try:
                    await ensure_healthy(live, baseline)
                    row = await run_once(live, wf, runner, available.get(runner, ""), baseline)
                except Exception as exc:  # setup or stack failure recorded; the suite goes on
                    row = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "infra_error": isinstance(exc, InfraError),
                           "seconds": 0, "checks": {}, "tool_names": [], "sleepwalking": []}
                rows.append(row)
                failed = [k for k, c in row["checks"].items() if not c["ok"]]
                verdict = "INFRA" if row.get("infra_error") else ("PASS" if row["ok"] else "FAIL")
                print(f"{wf['name']:32s} {runner:8s} #{i + 1} {verdict} {row['seconds']:6.1f}s {failed or ''} {row.get('error') or ''}", flush=True)
            counted = [r for r in rows if not r.get("infra_error")]
            results["workflows"][f"{wf['name']}/{runner}"] = {
                "passed": sum(r["ok"] for r in counted), "of": len(counted), "infra_errors": len(rows) - len(counted),
                "capability": wf["capability"], "kind": wf.get("kind", KINDS[0]), "held_out": bool(wf.get("held_out")),
                "bar": bar, "rows": rows, "task_hash": wf["task_hash"],
            }
            out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    try:
        await live.restore(baseline)
    except InfraError as exc:  # a stage left away from home is reported, not fatal
        print(f"warning: the instrument was not put back as the suite found it: {exc}", flush=True)
    results["passed"] = sum(v["passed"] for v in results["workflows"].values())
    results["of"] = sum(v["of"] for v in results["workflows"].values())
    results["sleepwalking_calls"] = sum(len(r.get("sleepwalking", [])) for v in results["workflows"].values() for r in v["rows"])
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))
    history = Path(args.history or HISTORY)
    board = board_report(results, previous_run(results, history))
    kept = save_history(results, history)
    kept.with_suffix(".md").write_text(board, encoding="utf-8")
    print(board)
    print(f"kept as {kept}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflows", nargs="+", type=Path, default=[WORKFLOW_DIR], help="YAML files or directories of them.")
    parser.add_argument("--only", nargs="*", default=None, help="Workflow names to run.")
    parser.add_argument("--runners", nargs="*", default=None,
                        help="Runners to use (default: each workflow's own list). 'reference' replays each workflow's reference calls with no model.")
    parser.add_argument("--stack", default="twin", help="Which stack is running: a name from benchmarks/stacks.yaml (twin, large, xrd) or a workflow file's stacks:.")
    parser.add_argument("--graph", default=None, help="An extra runner: module:function returning a chat graph, run as 'custom'.")
    parser.add_argument("--n", default="1", help="Trials per workflow and runner: a number (default 1), or 'bar' for as many as each workflow's bar counts (5 unless it sets its own).")
    parser.add_argument("--no-held-out", action="store_true", help="Leave out workflows marked held_out: use it while tuning the agent.")
    parser.add_argument("--allow-hardware", action="store_true", help="Run even when the endpoint's stack does not say it is simulated.")
    parser.add_argument("--skills", default=None,
                        help="Agent skills for this run: on, off, or skill names separated by commas (only those). Sets ASYNCROSCOPY_SKILLS "
                        "and ASYNCROSCOPY_SKILLS_OFF; left out, they stay as the environment has them (off unless set).")
    parser.add_argument("--skills-off", default=None, help="Skill names or globs to hide, separated by commas (ASYNCROSCOPY_SKILLS_OFF).")
    parser.add_argument("--history", default=None, help="Directory finished runs are kept in (default: outputs/benchmarks/history).")
    parser.add_argument("--model", default=settings.get().model or load_profile().model)
    parser.add_argument("--out", default=None, help="Results JSON (a .md table is written next to it).")
    parser.add_argument("--list", action="store_true", help="List the workflows that would run and exit (no stack needed).")
    parser.add_argument("--evaluators", default=",".join(EVALUATORS),
                        help="benchmarks.evaluate evaluators scored on each run's trace (extra columns, never pass/fail); "
                        "information, resolution and dose read the run's images from Tiled.")
    parser.add_argument("--score-columns", default=",".join(SCORE_COLUMNS), help="evaluator.objective medians added to the Markdown table.")
    sys.exit(asyncio.run(main(parser.parse_args())))
