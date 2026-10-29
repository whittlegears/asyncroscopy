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
    find_tool,
    is_infra_error,
    read_trace,
    sleepwalking,
    unique_caller,
    warm_up,
)
from workflow_checks import CHECKS, Call, Run, evaluate

HERE = Path(__file__).resolve().parent
WORKFLOW_DIR = HERE / "workflows"
WORKFLOW_KEYS = {"name", "description", "request", "turns", "runners", "timeout_s", "stack", "setup", "approval", "checks"}
TURN_KEYS = {"say", "wait_for"}
SETUPS = {"image", "map", "map_plain", "large_map"}
# Which config each stack runs; a stack whose config this checkout lacks is skipped.
STACKS = {"twin": "configs/digital_twin_mapping.yaml", "large": "configs/digital_twin_large_map.yaml", "xrd": "configs/yaml_sim_xrd.yaml"}
# Setup maps: (side, field of view, per-tile actions, scored). large_map has enough tiles to need zooming.
SETUP_MAPS = {
    "map": (4.8e-7, 1.6e-7, ["eds_point"], True),
    "map_plain": (6.0e-7, 1.6e-7, [], False),
    "large_map": (6.0e-6, 4.0e-7, [], True),
}
TEAM_MODULE = "asyncroscopy.agents.graphs.team"  # planner + specialists + reporter
GOAL_MODULE = "asyncroscopy.agents.graphs.goal"  # check the goal from facts, retry with feedback
SINGLE_TURN = {"mapping", "team", "goal"}  # runners that take one request, not a conversation
OPERATOR_URL = os.environ.get("ASYNCROSCOPY_OPERATOR_URL", "http://127.0.0.1:8001/mcp")


# ------------------------------------------------------------------ workflow files


class WorkflowError(ValueError):
    pass


def load_workflows(paths: list[Path]) -> list[dict]:
    """Every workflow in the given YAML files (or directories of them), checked for typos."""
    files = []
    for path in paths:
        files += sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    workflows, seen = [], set()
    for file in files:
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
            if wf.get("stack", "twin") not in STACKS:
                raise WorkflowError(f"{where}: stack must be one of {sorted(STACKS)}")
            if wf.get("approval") not in (None, "approve", "deny"):
                raise WorkflowError(f"{where}: approval is approve or deny")
            for spec in wf.get("checks") or []:
                if spec.get("check") not in CHECKS:
                    raise WorkflowError(f"{where}: unknown check {spec.get('check')!r}; known: {sorted(CHECKS)}")
            if not wf.get("checks"):
                raise WorkflowError(f"{where}: no checks")
            wf.setdefault("runners", ["general"])
            wf.setdefault("timeout_s", 600)
            wf.setdefault("stack", "twin")
            wf["file"] = file.name
            workflows.append(wf)
    return workflows


def render(value: Any, facts: dict) -> Any:
    """Fill {name} placeholders from setup facts; a string that is only a placeholder keeps the fact's type."""
    if isinstance(value, str):
        whole = re.fullmatch(r"\{(\w+)\}", value)
        if whole and whole.group(1) in facts:
            return facts[whole.group(1)]
        return re.sub(r"\{(\w+)\}", lambda m: str(facts.get(m.group(1), m.group(0))), value)
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

        self.operator = McpServer(OPERATOR_URL, {"X-Asyncroscopy-Caller": "workflow-operator"})
        self.names: set[str] = set()
        self.operator_names: set[str] = set()
        self.setup_cache: dict[str, dict] = {}

    async def discover(self) -> None:
        self.names = {t.name for t in await self.instrument.list_tools()}
        try:
            self.operator_names = {t.name for t in await self.operator.list_tools()}
        except Exception as exc:
            print(f"[workflows] operator endpoint {OPERATOR_URL} unavailable ({exc}); approval workflows will fail", flush=True)

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
        """Put the stage, field of view and defocus back as they were when the suite started."""
        if state.get("stage") and self.tool("_move_stage"):
            await self.text(self.tool("_move_stage"), {"position": state["stage"]})
        if state.get("fov") is not None:
            await self.text(self.tool("_set_fov"), {"fov": state["fov"]})
        if state.get("defocus") is not None:
            await self.text(self.tool("_set_defocus"), {"defocus": state["defocus"]})

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
        items = listed.get("result", listed) if isinstance(listed, dict) else listed
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
        try:
            listed = await self.analysis.call_json("list_pois", {"map": map_key})
        except Exception:
            return []
        items = listed.get("result", listed) if isinstance(listed, dict) else listed
        return [[p["x_m"], p["y_m"]] for p in items or [] if isinstance(p, dict) and "x_m" in p]

    # ---------------------------------------------------------- setups (run by the benchmark, not the agent)

    async def setup(self, name: str) -> dict:
        if name in self.setup_cache:
            return self.setup_cache[name]
        if name == "image":
            key = await self.text(self.tool("_acquire_scanned_image"), {"detector_list": ["haadf"]})
            facts = {"image_key": key, "keys": [key] if key else []}
        else:
            size, fov, per_tile, scored = SETUP_MAPS[name]
            plan = {"size_m": [size, size], "fov_m": fov, "per_tile": per_tile, "eds_exposure_s": 0.2}
            results = await self.run_map(plan, timeout_s=7200)
            run_key = results.get("result_keys", {}).get("run")
            stitched = await self.analysis_job("assemble_map", {"run": run_key}, timeout_s=3600)
            map_key = stitched["result_keys"]["map"]
            if scored:
                await self.analysis_job("score_map", {"map": map_key, "top_k": 3})
            pois = await self.pois(map_key)
            info = await self.map_info(map_key)
            spectra = [s["key"] for s in results.get("summary", {}).get("spectra", []) if s.get("key")]
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

    uri = os.environ.get("ASYNCROSCOPY_TILED_URI", tiled_client.DEFAULT_URI)
    spectra, strongest = {}, None
    for key in keys[:6]:
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
        positions = axis.values(0, len(values)) if axis else np.arange(len(values), dtype=np.float64)
        strongest = float(positions[int(np.argmax(values))])
        entry = {"strongest": strongest, "units": axis.units if axis else ""}
        if axis and axis.units == "eV":
            fwhm = views.eds_fwhm_ev(positions)
            peaks = views.find_peaks(values, fwhm / abs(axis.scale))
            entry["peaks_ev"] = [round(float(positions[p.index]), 1) for p in peaks]
            entry["elements"] = sorted({c.split()[0] for p in peaks for c in views.xray_candidates(float(positions[p.index]), float(fwhm[p.index]) / 2)})
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


MAP_KEY = re.compile(r"map_[\w.]+?_map")


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

    uri = os.environ.get("ASYNCROSCOPY_TILED_URI", tiled_client.DEFAULT_URI)
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
    return [Call(c.get("name", ""), dict(c.get("args") or {}), not c.get("error"), str(c.get("result", ""))[:2000]) for c in trace or []]


def calls_from_messages(messages) -> list[Call]:
    replies = {getattr(m, "tool_call_id", None): m for m in messages if getattr(m, "type", "") == "tool"}
    calls = []
    for m in messages:
        for c in getattr(m, "tool_calls", None) or []:
            reply = replies.get(c.get("id"))
            content = getattr(reply, "content", "") if reply is not None else ""
            text = content if isinstance(content, str) else json.dumps(content, default=str)
            ok = reply is not None and getattr(reply, "status", "success") != "error"
            calls.append(Call(c["name"], dict(c.get("args") or {}), ok, text[:2000]))
    return calls


async def run_chat_graph(factory: str, config, wf: dict, facts: dict, caller: str, started: float) -> tuple[list, str]:
    module, _, function = factory.partition(":")
    make_graph = getattr(importlib.import_module(module), function or "make_graph")
    graph = await make_graph({"configurable": {"model": config.model, "caller": caller}})
    from langchain_core.messages import HumanMessage

    turns = wf.get("turns") or [{"say": wf["request"]}]
    messages: list = []
    for turn in turns:
        await wait_for(turn.get("wait_for"), started, caller)
        messages = messages + [HumanMessage(render(turn["say"], facts))]
        state = await graph.ainvoke({"messages": messages}, {"recursion_limit": 80})
        messages = state["messages"]
    final = messages[-1].content if messages else ""
    return messages, final if isinstance(final, str) else json.dumps(final, default=str)


async def wait_for(condition: dict | None, started: float, caller: str) -> None:
    """Hold the next turn until the run's job has done ``job_steps_at_least`` tiles (or the timeout)."""
    if not condition:
        return
    deadline = time.time() + float(condition.get("timeout_s", 300))
    need = int(condition.get("job_steps_at_least", 0))
    while time.time() < deadline:
        steps = sum(1 for e in read_trace(started, [caller]) if str(e.get("action", "")).endswith(".step"))
        if steps >= need:
            return
        await asyncio.sleep(2)


async def run_once(live: Live, wf: dict, runner: str, factory: str, baseline: dict) -> dict:
    from asyncroscopy.agents.graphs.mapping import MappingGraph

    facts: dict = {}
    for name in wf.get("setup") or []:
        facts.update(await live.setup(name))
    await live.restore(baseline)
    caller = unique_caller(f"wf-{wf['name']}-{runner}")
    config = type(live.config)(**{**live.config.__dict__, "caller": caller})
    stop, operator, mode_before = asyncio.Event(), None, None
    if wf.get("approval"):
        mode_before = await live.set_mode("approve")
        operator = asyncio.create_task(live.operate(caller, wf["approval"], stop))
    facts["before"] = await live.snapshot()
    started = time.time()
    run = Run(facts=facts, has_calls=runner != "mapping")
    error = None
    try:
        request = render(wf.get("request") or "", facts)
        if runner == "mapping":
            state = await asyncio.wait_for(MappingGraph(config).build().ainvoke({"request": request}), wf["timeout_s"])
            run.final = str(state.get("report") or state.get("error") or "")
        elif runner == "team":
            from asyncroscopy.agents.graphs.team import TeamGraph

            state = await asyncio.wait_for(TeamGraph(config).build().ainvoke({"request": request}, {"recursion_limit": 100}), wf["timeout_s"])
            run.final, run.calls = str(state.get("answer", "")), calls_from_trace(state.get("trace"))
        elif runner == "goal":
            from asyncroscopy.agents.goals import Goal
            from asyncroscopy.agents.graphs.goal import GoalGraph

            goal = Goal.from_text(request, max_seconds=float(wf["timeout_s"]))
            state = await asyncio.wait_for(GoalGraph(goal, "team", config).build().ainvoke({"goal": goal.to_dict()}, {"recursion_limit": 4 * goal.budget.max_attempts + 4}), wf["timeout_s"] * 1.2)
            run.final, run.calls = str(state.get("report") or state.get("answer", "")), calls_from_trace(state.get("trace"))
        else:
            messages, run.final = await asyncio.wait_for(run_chat_graph(factory, config, wf, facts, caller, started), wf["timeout_s"])
            run.calls = calls_from_messages(messages)
    except TimeoutError:
        error = f"timed out after {wf['timeout_s']} s"
    except Exception as exc:  # a crash or the recursion limit counts as a failed run
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        if operator:
            await operator
        if mode_before:
            await live.set_mode(mode_before)
    seconds = round(time.time() - started, 1)
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
    names = [c.name for c in run.calls] or [str(e["action"]) for e in run.events if e.get("source") == "mcp"]
    return {
        "ok": error is None and all(c["ok"] for c in checks.values()),
        "error": error,
        "infra_error": is_infra_error(error),
        "seconds": seconds,
        "caller": caller,
        "checks": checks,
        "tool_calls": len(names),
        "tool_names": names,
        "sleepwalking": sleepwalking(names),
        "final": (run.final or "")[-600:],
    }


# ------------------------------------------------------------------ reports


def markdown(results: dict) -> str:
    lines = [
        f"# Workflow benchmark: {results['model']}, stack {results['stack']}, {results.get('git', '?')}",
        "",
        "| workflow | runner | passed | median s | failed checks |",
        "|---|---|---|---|---|",
    ]
    for key, value in results["workflows"].items():
        name, runner = key.rsplit("/", 1)
        failed = sorted({label for r in value["rows"] for label, c in r["checks"].items() if not c["ok"]} | {"error" for r in value["rows"] if r.get("error")})
        seconds = sorted(r["seconds"] for r in value["rows"]) or [0]
        lines.append(f"| {name} | {runner} | {value['passed']}/{value['of']} | {seconds[len(seconds) // 2]:.0f} | {', '.join(failed) or '-'} |")
    lines += ["", f"Passed {results['passed']} of {results['of']} runs; sleepwalking calls: {results['sleepwalking_calls']}."]
    return "\n".join(lines) + "\n"


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE.parent, check=False).stdout.strip() or "?"
    except OSError:
        return "?"


async def main(args) -> int:
    from asyncroscopy.agents.config import AgentConfig

    workflows = [wf for wf in load_workflows(args.workflows) if wf["stack"] == args.stack and (not args.only or wf["name"] in args.only)]
    if not (HERE.parent / STACKS[args.stack]).exists():
        print(f"stack {args.stack} needs {STACKS[args.stack]}, which this checkout does not have; nothing to run")
        return 0
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
    live = Live(config)
    await live.discover()
    await warm_up(config)
    baseline = await live.snapshot()
    results = {"model": args.model, "stack": args.stack, "git": git_head(), "n": args.n, "started": time.time(), "runners": available, "workflows": {}}
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
                    row = await run_once(live, wf, runner, available[runner], baseline)
                except Exception as exc:  # a setup or stack failure: recorded, the suite goes on
                    row = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "infra_error": False, "seconds": 0, "checks": {}, "tool_names": [], "sleepwalking": []}
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
    parser.add_argument("--stack", default="twin", choices=sorted(STACKS), help="Which stack is running: twin (mapping), large or xrd.")
    parser.add_argument("--graph", default=None, help="An extra runner: module:function returning a chat graph, run as 'custom'.")
    parser.add_argument("--n", type=int, default=1, help="Runs per workflow and runner.")
    parser.add_argument("--model", default=os.environ.get("ASYNCROSCOPY_MODEL", "qwen3:8b"))
    parser.add_argument("--out", default=None, help="Results JSON (a .md table is written next to it).")
    parser.add_argument("--list", action="store_true", help="List the workflows that would run and exit (no stack needed).")
    sys.exit(asyncio.run(main(parser.parse_args())))
