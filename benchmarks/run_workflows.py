"""Workflow benchmarks: realistic requests to an agent on the digital twin, checked from facts.

Each workflow in benchmarks/workflows/*.yaml is a request (or several turns) a STEM user
or a hackathon team would make, the runners to try it with, and checks computed from the
Instrument Trace, readings taken before and after, and the stored data (see
benchmarks/workflow_checks.py). Nothing is judged from what the model says alone.

Needs a running stack, both MCP endpoints and the pyTEMlib server (docs/Operation/quickstart.md):

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
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# benchmarks/ for _common, which also puts the project on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import yaml
from _common import (
    endpoint_risks,
    find_tool,
    is_infra_error,
    read_trace,
    unique_caller,
    warm_up,
)
from workflow_checks import CHECKS, Call, Run, evaluate, unrequested, workflow_allow

from asyncroscopy.agents.goals import MAP_KEY, trace_from_messages
from asyncroscopy.agents.skill_middleware import recursion_limit
from asyncroscopy.evaluate import RunData
from asyncroscopy.evaluate import score_run as evaluator_scores

HERE = Path(__file__).resolve().parent
WORKFLOW_DIR = HERE / "workflows"
WORKFLOW_KEYS = {"name", "description", "request", "turns", "runners", "timeout_s", "stack", "setup", "approval", "checks"}
TURN_KEYS = {"say", "wait_for"}
SETUPS = {"image", "survey_image", "map", "map_plain", "large_map"}
SURVEY_FOV_M = 6.0e-7  # survey_image: room for two zoom steps (16x then 8x) down to atomic columns
# Setups taken where the run starts: re-homed before them, not after (a stage move after the survey would put the
# stage's landing error between the survey and the run, which no real zoom has), and never reused from a cache.
IN_PLACE_SETUPS = {"survey_image"}
STACKS_FILE = HERE / "stacks.yaml"
STAGE_SETTLE_S = 600.0  # the longest a re-homing stage move may take (the mapping twin moves 5 um/s in real time)
TRANSCRIPTS = HERE.parent / "outputs" / "live" / "transcripts"  # a failed run's messages and calls, one file per run
TRANSCRIPT_TEXT_CHARS = 4000
STACK_KEYS = {"config", "mcp", "operator", "tiled"}
STACK_DEFAULTS = {"mcp": "http://127.0.0.1:8000/mcp", "operator": "http://127.0.0.1:8001/mcp", "tiled": "http://127.0.0.1:9091"}
# The stack being run (set by main): its run_servers config and the addresses it serves.
STACK: dict[str, str] = {**STACK_DEFAULTS, "name": "twin"}
# Setup maps: (side, field of view, per-tile actions, scored). large_map has enough tiles to need zooming.
SETUP_MAPS = {
    "map": (4.8e-7, 1.6e-7, ["eds_point"], True),
    "map_plain": (6.0e-7, 1.6e-7, [], False),
    "large_map": (6.0e-6, 4.0e-7, [], True),
}
TEAM_MODULE = "asyncroscopy.agents.graphs.team"  # planner + specialists + reporter
GOAL_MODULE = "asyncroscopy.agents.graphs.goal"  # check the goal from facts, retry with feedback
SINGLE_TURN = {"mapping", "team", "goal"}  # runners that take one request, not a conversation
CHAT_TOOL_ROUNDS = 40  # tool calls one turn of a chat graph may make
# Evaluators (asyncroscopy.evaluate) scored on every run's trace, as extra columns; they never change pass or fail.
# The default reads no data; information, resolution and dose read the run's images from Tiled (set by --evaluators).
EVALUATORS = ["time"]
SCORE_COLUMNS = ["time.clock_s", "time.idle_s"]


# ------------------------------------------------------------------ workflow files


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
            if wf.get("approval") not in (None, "approve", "deny"):
                raise WorkflowError(f"{where}: approval is approve or deny")
            for spec in wf.get("checks") or []:
                if not isinstance(spec, dict) or spec.get("check") not in CHECKS:
                    raise WorkflowError(f"{where}: unknown check {spec.get('check') if isinstance(spec, dict) else spec!r}; known: {sorted(CHECKS)}")
                _check_params(where, spec)
            if not wf.get("checks"):
                raise WorkflowError(f"{where}: no checks")
            wf.setdefault("runners", ["general"])
            wf.setdefault("timeout_s", 600)
            wf.setdefault("stack", "twin")
            wf["file"] = file.name
            workflows.append(wf)
    return workflows


def _fact(facts: dict, name: str) -> Any:
    value = facts.get(name)
    if value is None:  # "x = None nm" in a request is a broken setup, not a test of the agent
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


# ------------------------------------------------------------------ the live stack


class Live:
    """The stack's endpoints, with tool names found from what they offer (never hard-coded)."""

    def __init__(self, config):
        self.config = config
        self.instrument = config.instrument()
        self.analysis = config.analysis()
        from asyncroscopy.agents.mcp_tools import McpServer

        self.operator = McpServer(STACK["operator"], {"X-Asyncroscopy-Caller": "workflow-operator"})
        self.names: set[str] = set()
        self.operator_names: set[str] = set()
        self.risks: dict[str, str] | None = None  # the endpoint's own tool risks (endpoint_info)
        self.setup_cache: dict[str, dict] = {}

    async def discover(self) -> None:
        self.names = {t.name for t in await self.instrument.list_tools()}
        self.risks = await endpoint_risks(self.config)
        try:
            self.operator_names = {t.name for t in await self.operator.list_tools()}
        except Exception as exc:
            print(f"[workflows] operator endpoint {STACK['operator']} unavailable ({exc}); approval workflows will fail", flush=True)

    def tool(self, *suffixes: str, prefix: str = "") -> str | None:
        return find_tool(self.names, *suffixes, prefix=prefix)

    async def text(self, name: str | None, args: dict | None = None, server=None) -> str | None:
        if not name:
            return None
        try:
            return await (server or self.instrument).call_text(name, args or {})
        except Exception:
            return None

    async def snapshot(self) -> dict:
        stage = await self.text(self.tool("_get_stage"))
        fov = await self.text(self.tool("_get_fov"))
        defocus = await self.text(self.tool("_get_defocus"))
        return {"stage": _floats(stage), "fov": _float(fov), "defocus": _float(defocus)}

    async def restore(self, state: dict) -> None:
        """Put the stage, field of view and defocus back as they were when the suite started; InfraError if the
        instrument does not answer (a run after it would test a wedged device, not the agent)."""
        steps = [("_move_stage", {"position": state.get("stage")}, state.get("stage")),
                 ("_set_fov", {"fov": state.get("fov")}, state.get("fov")), ("_set_defocus", {"defocus": state.get("defocus")}, state.get("defocus"))]
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
                return  # parameters that do not answer or parse: nothing to wait on
            if not (isinstance(parameters, dict) and parameters.get("stage_moving")):
                return
            await asyncio.sleep(1.0)
        raise InfraError(f"the stage was still moving {timeout_s:.0f} s after re-homing")

    # ---------------------------------------------------------- jobs, used by setups

    async def run_map(self, plan: dict, timeout_s: float = 1800) -> dict:
        """Run a Mapper job to the end with whichever job tools the endpoint offers; returns its results."""
        mapper = next((n[: -len("_reserve")] for n in sorted(self.names) if n.startswith("Mapper") and n.endswith("_reserve")), None)
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
        tool = find_tool(self.operator_names, "_export_ground_truth")
        text = await self.text(tool, server=self.operator) if tool else None
        try:
            return json.loads(text) if text else None
        except json.JSONDecodeError:
            return None

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

    # ---------------------------------------------------------- setups (run by the benchmark, not the agent)

    async def setup(self, name: str) -> dict:
        if name in self.setup_cache and name not in IN_PLACE_SETUPS:
            return self.setup_cache[name]
        if name in ("image", "survey_image"):  # image: at the field of view the instrument has; survey_image: at SURVEY_FOV_M
            if name == "survey_image":
                await self.text(self.tool("_set_fov"), {"fov": SURVEY_FOV_M})
            await self.text(self.tool("_unblank_beam"), {})  # a blanked beam images noise
            try:
                key = await self.text(self.tool("_acquire_scanned_image"), {"detector_list": ["haadf"]})
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

    # ---------------------------------------------------------- approval mode and the scripted operator

    async def set_mode(self, mode: str) -> str | None:
        get_mode, set_mode = find_tool(self.operator_names, "_get_mode"), find_tool(self.operator_names, "_set_mode")
        before = (await self.text(get_mode, server=self.operator) or "").strip().strip('"') or None
        await self.text(set_mode, {"request": json.dumps({"mode": mode, "who": "workflow-benchmark"})}, server=self.operator)
        return before

    async def operate(self, caller: str, decision: str, stop: asyncio.Event) -> None:
        """Approve or deny every request ``caller`` makes until ``stop`` is set."""
        pending_tool = find_tool(self.operator_names, "_list_pending")
        decide_tool = find_tool(self.operator_names, "_approve" if decision == "approve" else "_deny")
        while not stop.is_set():
            listed = await self.text(pending_tool, server=self.operator)
            for request in json.loads(listed or "[]"):
                if request.get("caller") == caller:
                    body = {"request_id": request["request_id"], "who": "workflow-operator", "reason": "workflow benchmark"}
                    await self.text(decide_tool, {"request": json.dumps(body)}, server=self.operator)
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except TimeoutError:
                pass


class InfraError(RuntimeError):
    """The stack, not the agent, failed: a wedged device, a setup that produced nothing. Such rows are not counted."""


def _poi_items(listed: Any) -> list | None:
    """The points of a list_pois reply: {"total", "pois"} from pyTEMlib, {"result": [...]} or a plain list."""
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


# ------------------------------------------------------------------ truth from the data


def data_truth(keys: list[str]) -> dict:
    """Peaks and candidate elements of each 1-D dataset among ``keys``, read with the data layer."""
    from asyncroscopy.data import tiled_client, views

    uri = STACK["tiled"]
    spectra, strongest = {}, None
    # The newest: a map with a spectrum per tile came before the spectrum the answer is about, and the first six were all tiles.
    for key in keys[-6:]:
        try:
            node = tiled_client.node_for_key(key, uri)
            ds = views.pick(node, key)
            if ds.ndim != 1:
                continue
            values = np.asarray(ds.node.read(), dtype=np.float64)
            axis = views.energy_axis({**ds.parent_attrs, **ds.attrs})
        except Exception as exc:  # the truth is missing, so checks that need it fail with the reason
            spectra[key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        line_axis = axis or views.Axis(0.0, 1.0, "channel", "")
        positions = line_axis.offset + line_axis.scale * np.arange(len(values), dtype=np.float64)
        xray = line_axis.units == "eV" and views.is_xray(ds)
        # view_data's own peak function (same widths, same strongest-first order), so truth and what the agent read agree.
        rows = views.line_peaks(positions, values, line_axis, xray)
        strongest = rows[0][0] if rows else float(positions[int(np.argmax(values))])
        entry = {"strongest": strongest, "units": line_axis.units, "peaks": [round(row[0], 4) for row in rows]}
        if xray:
            entry["peaks_ev"] = [round(row[0], 1) for row in rows]
            likely = views.likely_elements(rows)  # the elements view_data names, not every line that could fit a peak
            entry["elements"] = sorted(part.split(" ")[0] for part in likely.split(", ")) if likely else []
            entry["could_be"] = sorted({el for row in rows for el in views.note_elements(row[4])})
        spectra[key] = entry
    return {"spectra": {k: v for k, v in spectra.items() if "elements" in v or "error" not in v}, "strongest_peak": strongest}


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
    common offset removed, as tests/test_map_end_to_end.py measures it; and the same for stage readings alone."""
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
        true_centres.append([match["center_true_m"][1] / pixel, match["center_true_m"][0] / pixel])  # (row, col)
        reported.append([read[1] / pixel, read[0] / pixel])
        estimated.append(positions[index] + np.asarray(tile_shape, dtype=np.float64) / 2.0)
    if len(true_centres) < 2:
        return {"error": f"matched only {len(true_centres)} of {len(steps)} tiles to the twin's ground truth"}
    true_centres = np.asarray(true_centres)

    def median_error(centres) -> float:
        diff = np.asarray(centres) - true_centres
        diff -= np.median(diff, axis=0)
        return float(np.median(np.hypot(diff[:, 0], diff[:, 1])))

    stage = median_error(reported)
    return {"median_px": median_error(estimated), "stage_median_px": stage, "stage_error_fov": stage / float(tile_shape[1]), "tiles": len(true_centres)}


def map_metadata_and_tile_shape(map_key: str) -> tuple[dict, tuple[int, int]]:
    """A stitched map's metadata and the shape of its run's tiles, read from Tiled."""
    from asyncroscopy.data import tiled_client

    uri = STACK["tiled"]
    meta = dict(tiled_client.node_for_key(map_key, uri).metadata)
    tiles = tiled_client.node_for_key(meta["run"], uri)["tiles"]
    first = next(iter(tiles))
    return meta, tuple(tiles[first].structure().shape[:2])


# ------------------------------------------------------------------ runners


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


async def run_chat_graph(factory: str, config, wf: dict, facts: dict, caller: str, started: float) -> tuple[list, str, list[str]]:
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


async def run_once(live: Live, wf: dict, runner: str, factory: str, baseline: dict) -> dict:
    from asyncroscopy.agents.graphs.mapping import MappingGraph

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
    caller = unique_caller(f"wf-{wf['name']}-{runner}")
    config = live.config.for_run(caller=caller)
    stop, operator, mode_before = asyncio.Event(), None, None
    if wf.get("approval"):
        mode_before = await live.set_mode("approve")
        operator = asyncio.create_task(live.operate(caller, wf["approval"], stop))
    facts["before"] = await live.snapshot()
    started = time.time()
    run = Run(facts=facts, has_calls=runner != "mapping", risks=live.risks)
    error, transcript = None, []
    try:
        request = render(wf.get("request") or "", facts)
        if runner == "mapping":
            state = await asyncio.wait_for(MappingGraph(config).build().ainvoke({"request": request}), wf["timeout_s"])
            run.final = str(state.get("report") or state.get("error") or "")
        elif runner == "team":
            from asyncroscopy.agents.graphs.team import TeamGraph

            state = await asyncio.wait_for(TeamGraph(config).build().ainvoke({"request": request}, {"recursion_limit": 100}), wf["timeout_s"])
            run.final, run.calls = str(state.get("answer", "")), calls_from_trace(state.get("trace"))
            run.skills = skills_of(state.get("skills_applied"))
        elif runner == "goal":
            from asyncroscopy.agents.goals import Goal
            from asyncroscopy.agents.graphs.goal import GoalGraph

            goal = Goal.from_text(request, max_seconds=float(wf["timeout_s"]))
            state = await asyncio.wait_for(GoalGraph(goal, "team", config).build().ainvoke({"goal": goal.to_dict()}, {"recursion_limit": 4 * goal.budget.max_attempts + 4}), wf["timeout_s"] * 1.2)
            run.final, run.calls = str(state.get("report") or state.get("answer", "")), calls_from_trace(state.get("trace"))
            run.skills = skills_of(state.get("skills_applied"))
        else:
            messages, run.final, run.skills = await asyncio.wait_for(run_chat_graph(factory, config, wf, facts, caller, started), wf["timeout_s"])
            run.calls = trace_from_messages(messages)
            transcript = [transcript_entry(m) for m in messages]
            if not (run.final or "").strip():  # what the model sent last, to see why it gave no answer
                facts["last_messages"] = [message_summary(m) for m in messages[-3:]]
    except TimeoutError:
        error = f"timed out after {wf['timeout_s']} s"
    except InfraError:
        raise
    except Exception as exc:  # a crash or the recursion limit counts as a failed run
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        if operator:
            await operator
        if mode_before:
            await live.set_mode(mode_before)
    seconds = round(time.time() - started, 1)
    if error and len(config.ledger) > len(run.calls):  # stopped part way: keep every call whose reply arrived
        run.calls = [Call.from_dict(c) for c in config.ledger]
    await asyncio.sleep(1.0)  # let the last trace lines land
    facts["after"] = await live.snapshot()
    run.events = read_trace(started, [caller])
    facts["all_job_events"] = [e for e in read_trace(started) if e.get("source") == "job"]
    mine = {e.get("run_id") for e in run.events if e.get("source") == "job"}
    await live.tidy([e for e in facts["all_job_events"] if e.get("run_id") in mine])
    # The maps this run touched: named in its replies or answer, stitched from its runs, or set up for it.
    mine_runs = run_keys([e for e in facts["all_job_events"] if e.get("run_id") in mine])
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
    stitched_here = [k for k in facts["poi_records"] if any(k == f"{Path(r).stem}_map" for r in mine_runs)]
    stitch = {}
    if stitched_here and any(c.get("check") == "stitch_accuracy" for c in wf["checks"]):
        truth_doc = await live.ground_truth()
        # (run id, step result) for every tile; a run file is map_<run id>.h5, its map map_<run id>_map.
        steps = [(e.get("run_id"), json.loads(e["result"])) for e in run.events if str(e.get("action", "")).endswith(".step") and e.get("ok") and e.get("result")]
        for key in stitched_here if truth_doc is not None else []:
            try:
                meta, shape = map_metadata_and_tile_shape(key)
                stitch[key] = stitch_truth(meta, shape, [step for run_id, step in steps if f"map_{run_id}" == Path(meta["run"]).stem], truth_doc)
            except Exception as exc:  # recorded, so the check fails with the reason
                stitch[key] = {"error": f"{type(exc).__name__}: {exc}"}
    truth_keys = produced_keys(run) + [k for k in (facts.get("spectrum_key"),) if k]
    truth = data_truth(truth_keys) if truth_keys else {}
    facts["truth"] = {**facts.get("truth", {}), **truth, **({"stitch": stitch} if stitch else {})}
    checks = evaluate(run, render(wf["checks"], {**facts, **facts.get("truth", {})}))
    if runner != "mapping" and not (run.final or "").strip():
        checks["answered"] = {"ok": False, "detail": "the agent gave no final answer"}
    names = [c.name for c in run.calls] or [str(e["action"]) for e in run.events if e.get("source") == "mcp"]
    try:  # extra columns only: an evaluator that fails is recorded, the row's verdict stays as the checks gave it
        scores = evaluator_scores(RunData(caller, run.events, tiled_uri=STACK.get("tiled")), EVALUATORS)
    except Exception as exc:
        scores = {"error": f"{type(exc).__name__}: {exc}"}
    ok = error is None and all(c["ok"] for c in checks.values())
    if not ok:
        TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
        (TRANSCRIPTS / f"{caller}.json").write_text(json.dumps({
            "workflow": wf["name"], "runner": runner, "error": error, "checks": checks, "final": run.final,
            "messages": transcript, "calls": [c.to_dict() for c in run.calls],
        }, indent=1, default=str), encoding="utf-8")
    return {
        "ok": ok,
        "error": error,
        "infra_error": is_infra_error(error),
        "seconds": seconds,
        "caller": caller,
        "checks": checks,
        "tool_calls": len(names),
        "tool_names": names,
        "sleepwalking": unrequested(run, workflow_allow(wf)),  # with this workflow's own allow list
        "skills": run.skills,
        "final": (run.final or "")[-600:],
        "scores": scores,
        **({"last_messages": facts["last_messages"]} if facts.get("last_messages") else {}),
    }


def transcript_entry(message) -> dict:
    """A message as a failed run's transcript keeps it: type, tool name, text, the calls it made with their arguments,
    and why the model stopped (an empty reply right after the model server reloaded says "load", not "stop")."""
    content = message.content if isinstance(message.content, str) else json.dumps(message.content, default=str)
    metadata = getattr(message, "response_metadata", None) or {}
    return {
        "type": getattr(message, "type", ""), "name": getattr(message, "name", None), "text": content[:TRANSCRIPT_TEXT_CHARS],
        "tool_calls": [{"name": c.get("name"), "args": c.get("args")} for c in getattr(message, "tool_calls", None) or []],
        "stop": metadata.get("done_reason") or metadata.get("finish_reason"),
    }


def message_summary(message) -> dict:
    """A message's type, the end of its text and the tools it called: enough to tell an empty answer's cause."""
    content = message.content if isinstance(message.content, str) else json.dumps(message.content, default=str)
    return {
        "type": getattr(message, "type", ""), "text": content[-300:],
        "tool_calls": [c.get("name") for c in getattr(message, "tool_calls", None) or []],
        "stop": (getattr(message, "response_metadata", None) or {}).get("done_reason") or (getattr(message, "response_metadata", None) or {}).get("finish_reason"),
    }


# ------------------------------------------------------------------ reports


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


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE.parent, check=False).stdout.strip() or "?"
    except OSError:
        return "?"


async def main(args) -> int:
    from asyncroscopy.agents.config import AgentConfig

    stacks = load_stacks(args.workflows)
    if args.stack not in stacks:
        print(f"no stack {args.stack!r}; declared: {sorted(stacks)} (benchmarks/stacks.yaml, or stacks: in a workflow file)")
        return 2
    STACK.update(stacks[args.stack])
    # Environment overrides win over the stack file (configs/mcp_*_operator.yaml comments use them).
    for field, variable in (("operator", "ASYNCROSCOPY_OPERATOR_URL"), ("tiled", "ASYNCROSCOPY_TILED_URI")):
        if os.environ.get(variable):
            STACK[field] = os.environ[variable]
    on_stack = [wf for wf in load_workflows(args.workflows, stacks) if wf["stack"] == args.stack]
    unknown = sorted(set(args.only or ()) - {wf["name"] for wf in on_stack})
    if unknown:
        print(f"no workflow {unknown} on stack {args.stack!r} (names are separated by spaces, not commas); --list shows them")
        return 2
    workflows = [wf for wf in on_stack if not args.only or wf["name"] in args.only]
    if not (HERE.parent / STACK["config"]).exists() and not Path(STACK["config"]).exists():
        print(f"stack {args.stack} needs {STACK['config']}, which this checkout does not have: fix config: in its stacks: entry")
        return 2
    available = runners_available()
    if args.graph:
        available["custom"] = args.graph
    if args.list:
        for wf in workflows:
            runners = runners_for(wf, available)
            print(f"{wf['name']:32s} {','.join(runners):22s} timeout {wf['timeout_s']:5d} s  {wf['file']}" + ("  [approval]" if wf.get("approval") else ""))
        print(f"{len(workflows)} workflows on stack {args.stack}; runners here: {sorted(available)}")
        return 0
    config = AgentConfig.from_env(model=args.model, caller="workflows")
    if not os.environ.get("ASYNCROSCOPY_MCP_URL"):
        config.mcp_url = STACK["mcp"]
    live = Live(config)
    await live.discover()
    await warm_up(config)
    baseline = await live.snapshot()
    EVALUATORS[:] = [n.strip() for n in args.evaluators.split(",") if n.strip()]
    SCORE_COLUMNS[:] = [c.strip() for c in args.score_columns.split(",") if c.strip()]
    results = {"model": args.model, "stack": args.stack, "git": git_head(), "n": args.n, "started": time.time(), "runners": available,
               "evaluators": EVALUATORS, "score_columns": SCORE_COLUMNS, "workflows": {}}
    out = Path(args.out or HERE / "results" / f"workflows_{args.stack}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    for wf in workflows:
        for runner in runners_for(wf, available):
            if args.runners and runner not in args.runners:
                continue
            if runner not in available or (runner in SINGLE_TURN and wf.get("turns")):
                why = f"no {runner} runner in this checkout" if runner not in available else f"{runner} takes one request; this workflow has turns"
                results["workflows"][f"{wf['name']}/{runner}"] = {"passed": 0, "of": 0, "skipped": why, "rows": []}
                continue
            rows = []
            for i in range(args.n):
                try:
                    await ensure_healthy(live, baseline)
                    row = await run_once(live, wf, runner, available[runner], baseline)
                except Exception as exc:  # a setup or stack failure: recorded, the suite goes on
                    row = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "infra_error": isinstance(exc, InfraError),
                           "seconds": 0, "checks": {}, "tool_names": [], "sleepwalking": []}
                rows.append(row)
                failed = [k for k, c in row["checks"].items() if not c["ok"]]
                verdict = "INFRA" if row.get("infra_error") else ("PASS" if row["ok"] else "FAIL")
                print(f"{wf['name']:32s} {runner:8s} #{i + 1} {verdict} {row['seconds']:6.1f}s {failed or ''} {row.get('error') or ''}", flush=True)
            counted = [r for r in rows if not r.get("infra_error")]
            results["workflows"][f"{wf['name']}/{runner}"] = {"passed": sum(r["ok"] for r in counted), "of": len(counted), "infra_errors": len(rows) - len(counted), "rows": rows}
            out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    await live.restore(baseline)
    results["passed"] = sum(v["passed"] for v in results["workflows"].values())
    results["of"] = sum(v["of"] for v in results["workflows"].values())
    results["sleepwalking_calls"] = sum(len(r.get("sleepwalking", [])) for v in results["workflows"].values() for r in v["rows"])
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflows", nargs="+", type=Path, default=[WORKFLOW_DIR], help="YAML files or directories of them.")
    parser.add_argument("--only", nargs="*", default=None, help="Workflow names to run.")
    parser.add_argument("--runners", nargs="*", default=None, help="Runners to use (default: each workflow's own list).")
    parser.add_argument("--stack", default="twin", help="Which stack is running: a name from benchmarks/stacks.yaml (twin, large, xrd) or a workflow file's stacks:.")
    parser.add_argument("--graph", default=None, help="An extra runner: module:function returning a chat graph, run as 'custom'.")
    parser.add_argument("--n", type=int, default=1, help="Runs per workflow and runner.")
    parser.add_argument("--model", default=os.environ.get("ASYNCROSCOPY_MODEL", "qwen3:8b"))
    parser.add_argument("--out", default=None, help="Results JSON (a .md table is written next to it).")
    parser.add_argument("--list", action="store_true", help="List the workflows that would run and exit (no stack needed).")
    parser.add_argument("--evaluators", default=",".join(EVALUATORS),
                        help="asyncroscopy.evaluate evaluators scored on each run's trace (extra columns, never pass/fail); "
                        "information, resolution and dose read the run's images from Tiled.")
    parser.add_argument("--score-columns", default=",".join(SCORE_COLUMNS), help="evaluator.objective medians added to the Markdown table.")
    sys.exit(asyncio.run(main(parser.parse_args())))
