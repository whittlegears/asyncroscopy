"""Which local model reliably turns instrument requests into valid tool calls?

Binds the live MCP tools (instrument endpoint plus the pyTEMlib server) to
each candidate model, asks a fixed set of requests, and counts a call as
valid when the model picks the expected tool and its arguments satisfy that
tool's JSON schema. No tool is executed.

    uv run python benchmarks/run_tool_json.py --models qwen3:4b qwen3:8b qwen3-agent

Results go to benchmarks/results/tool_json.json. The pass bar is 90 % valid.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import time
from pathlib import Path

import jsonschema
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from asyncroscopy.agents.config import AgentConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
SYSTEM = (
    "You control a scanning transmission electron microscope through tools. "
    "Answer every request with exactly one tool call. Lengths are in metres; tilts in degrees."
)
ALLOW = [
    "*_get_stage", "*_move_stage", "*_acquire_scanned_image", "*_set_fov", "*_place_beam", "*_acquire_spectrum",
    "*_blank_beam", "*_set_defocus", "Approvals_get_mode", "approval_status", "Mapper_*", "list_devices",
    "assemble_map", "score_map", "add_poi", "get_region", "list_scorers", "analysis_job_status", "analysis_job_results",
]


async def run(models: list[str], num_ctx: int, out: Path) -> dict:
    base = AgentConfig.from_env(caller="benchmark")
    tools = await base.tools(allow=ALLOW)
    schemas = {t.name: t.args_schema for t in tools}
    prompts = [json.loads(line) for line in (HERE / "tool_json" / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {"tools": len(tools), "prompts": len(prompts), "num_ctx": num_ctx, "models": {}}
    for model in models:
        chat = AgentConfig.from_env(model=model, num_ctx=num_ctx).chat_model().bind_tools(tools)
        rows, start = [], time.time()
        for item in prompts:
            t0 = time.time()
            try:
                reply = await chat.ainvoke([SystemMessage(SYSTEM), HumanMessage(item["prompt"])])
                calls = reply.tool_calls
            except Exception as exc:
                calls, reply = [], f"error: {exc}"
            ok, why = False, "no tool call"
            if calls:
                name, args = calls[0]["name"], calls[0]["args"]
                if not fnmatch.fnmatch(name, item["tool"]):
                    why = f"picked {name}"
                else:
                    try:
                        jsonschema.validate(args, schemas[name])
                        ok, why = True, "ok"
                    except jsonschema.ValidationError as exc:
                        why = f"bad args: {exc.message}"
            rows.append({"prompt": item["prompt"], "expected": item["tool"], "ok": ok, "why": why, "seconds": round(time.time() - t0, 2), "call": calls[0] if calls else None})
            print(f"{model:14s} {'OK ' if ok else 'BAD'} {rows[-1]['seconds']:6.1f}s  {item['prompt'][:60]:60s}  {why}", flush=True)
        valid = sum(r["ok"] for r in rows) / len(rows)
        report["models"][model] = {"valid_fraction": valid, "passes": valid >= 0.9, "total_seconds": round(time.time() - start, 1), "rows": rows}
        print(f"== {model}: {valid:.0%} valid in {report['models'][model]['total_seconds']} s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["qwen3:4b", "qwen3-agent", "qwen3:8b"])
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "tool_json.json")
    args = parser.parse_args()
    asyncio.run(run(args.models, args.num_ctx, args.out))
