"""Seeded agent trials on the digital twin: does the agent map what was asked?

Needs the twin stack, both MCP endpoints and the pyTEMlib server running
(see docs/Operation/windows_quickstart.md). Each trial sends one request to
the ``mapping`` graph (the model writes the plan; code runs it) or the
``general`` agent (the model makes every call), then checks the outcome
against what the request asked for:

* the run finished and a stitched map exists,
* the mapped box covers the requested square (within 25 %),
* EDS spectra were taken when asked, and not when not asked,
* interesting regions were proposed when asked,
* every Mapper tile stayed inside its plan box (from the Instrument Trace).

    uv run python benchmarks/run_mapping_trials.py --n 10 --model qwen3:8b

Results go to benchmarks/results/mapping_trials.json. Pass bar: 8 of 10 per task.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import HumanMessage

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.graphs.mapping import MappingGraph

HERE = Path(__file__).resolve().parent
SIDE_M = 6e-7

TASKS = {
    "map_square": {"request": "Map a 600 nm square around the current stage position.", "eds": False, "score": False},
    "map_with_eds": {"request": "Map a 600 nm square around the current position and take an EDS spectrum with each image.", "eds": True, "score": False},
    "map_and_explore": {"request": "Map a 600 nm square around the current position and find the 3 most interesting regions.", "eds": False, "score": True},
}


def check(task: dict, plan: dict | None, state: dict, centre: list[float]) -> dict:
    checks = {}
    checks["completed"] = bool(state.get("map_key")) and not state.get("error")
    if plan:
        x0, x1, y0, y1 = plan["box_m"]
        covers = x0 <= centre[0] - SIDE_M / 2 * 0.75 and x1 >= centre[0] + SIDE_M / 2 * 0.75 and y0 <= centre[1] - SIDE_M / 2 * 0.75 and y1 >= centre[1] + SIDE_M / 2 * 0.75
        not_huge = (x1 - x0) <= SIDE_M * 1.6 and (y1 - y0) <= SIDE_M * 1.6
        checks["box_matches_request"] = covers and not_huge
        checks["eds_as_asked"] = ("eds_point" in plan.get("per_tile", [])) == task["eds"]
    else:
        checks["box_matches_request"] = checks["eds_as_asked"] = False
    candidates = [p for p in state.get("pois", []) if p.get("kind") == "candidate"]
    checks["explored_as_asked"] = (len(candidates) > 0) == task["score"]
    return checks


async def graph_trial(config: AgentConfig, task: dict) -> dict:
    graph = MappingGraph(config)
    microscope = await graph._microscope()
    centre = json.loads(await graph.instrument.call_text(f"{microscope}_get_stage"))[:2]
    started = time.time()
    state = await graph.build().ainvoke({"request": task["request"]})
    checks = check(task, state.get("plan"), state, centre)
    return {"seconds": round(time.time() - started, 1), "model_calls": 1, "checks": checks, "ok": all(checks.values()), "error": state.get("error"), "plan": state.get("plan"), "report": state.get("report")}


async def general_trial(config: AgentConfig, task: dict) -> dict:
    from asyncroscopy.agents.graphs.general import make_graph

    graph = await make_graph({"configurable": {"model": config.model, "caller": config.caller}})
    instrument = config.instrument()
    microscope = next(t.name[: -len("_move_stage")] for t in await instrument.list_tools() if t.name.endswith("_move_stage"))
    centre = json.loads(await instrument.call_text(f"{microscope}_get_stage"))[:2]
    started = time.time()
    try:
        state = await graph.ainvoke({"messages": [HumanMessage(task["request"] + " Tell me the map key at the end.")]}, {"recursion_limit": 60})
        messages = state["messages"]
    except Exception as exc:  # recursion limit or a crash counts as a failed trial
        return {"seconds": round(time.time() - started, 1), "ok": False, "error": f"{type(exc).__name__}: {exc}", "checks": {}}
    calls = [c for m in messages for c in (getattr(m, "tool_calls", None) or [])]
    reserve = next((c for c in reversed(calls) if c["name"] == "Mapper_reserve"), None)
    plan = None
    if reserve:
        args = reserve["args"]
        body = args.get("request", args)
        body = json.loads(body) if isinstance(body, str) else body
        plan = body.get("plan")
    final = messages[-1].content if messages else ""
    map_key = re.search(r"map_\w+_map", final)
    state_like = {"map_key": map_key.group(0) if map_key else None, "pois": [{"kind": "candidate"}] if any(c["name"] == "score_map" for c in calls) else []}
    checks = check(task, plan, state_like, centre)
    checks["used_mapper_not_a_loop"] = sum(1 for c in calls if c["name"].endswith("_move_stage")) <= 2
    return {"seconds": round(time.time() - started, 1), "model_calls": len([m for m in messages if getattr(m, "tool_calls", None) is not None and m.type == "ai"]), "tool_calls": len(calls), "checks": checks, "ok": all(checks.values()), "plan": plan, "final": final[-600:]}


def out_of_box_moves(trace_path: Path, since: float) -> int:
    """Mapper steps whose reported stage position lies outside their plan box (should always be 0)."""
    if not trace_path.exists():
        return -1
    return sum(1 for line in trace_path.read_text(encoding="utf-8").splitlines() if '"Mapper.step"' in line and '"ok": false' in line and json.loads(line).get("ts", 0) >= since and "outside the plan box" in line)


async def main(args) -> None:
    config = AgentConfig.from_env(model=args.model, caller="trials")
    results = {"model": args.model, "n": args.n, "started": time.time(), "tasks": {}}
    tasks = {k: v for k, v in TASKS.items() if not args.tasks or k in args.tasks}
    for name, task in tasks.items():
        for runner_name in args.runners:
            rows = []
            for i in range(args.n if runner_name == "graph" else max(1, args.n // 2)):
                runner = graph_trial if runner_name == "graph" else general_trial
                try:
                    row = await runner(config, task)
                except Exception as exc:
                    row = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "checks": {}}
                rows.append(row)
                print(f"{name:16s} {runner_name:8s} #{i + 1:2d} {'PASS' if row['ok'] else 'FAIL'} {row.get('seconds', 0):6.1f}s  {json.dumps(row.get('checks'))}  {row.get('error') or ''}", flush=True)
            passed = sum(r["ok"] for r in rows)
            results["tasks"][f"{name}/{runner_name}"] = {"passed": passed, "of": len(rows), "pass_bar_met": passed >= 0.8 * len(rows), "rows": rows}
            out = HERE / "results" / "mapping_trials.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    results["out_of_box_moves"] = out_of_box_moves(Path("outputs/trace/jobs.jsonl"), results["started"])
    (HERE / "results" / "mapping_trials.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    for key, value in results["tasks"].items():
        print(f"== {key}: {value['passed']}/{value['of']}")
    print(f"== out-of-box Mapper moves: {results['out_of_box_moves']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--runners", nargs="+", default=["graph", "general"], choices=["graph", "general"])
    parser.add_argument("--tasks", nargs="*", default=None, choices=list(TASKS))
    asyncio.run(main(parser.parse_args()))
