"""Check a running asyncroscopy stack end to end, the way an agent and the GUI use it.

Start the twin and both MCP endpoints first:

    uv run startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml
    uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
    uv run startup_scripts/run_mcp.py --yaml configs/mcp_operator.yaml

then:

    uv run startup_scripts/check_stack.py

Each check prints PASS or FAIL with the reason; the exit code is the number of
failures. Safe on the digital twin; on a real instrument it moves the stage
and acquires, so read it first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)
    return ok


def text(result) -> str:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and isinstance(structured.get("result"), str):
        return structured["result"]
    return "".join(getattr(block, "text", "") or "" for block in (result.content or []))


async def call(client: Client, tool: str, args: dict | None = None, raise_on_error: bool = True):
    return await client.call_tool(tool, args or {}, raise_on_error=raise_on_error)


async def check_analysis(args: argparse.Namespace, run_key: str) -> None:
    """Stitch the run on the pyTEMlib MCP server, look at it, score it and mark a point."""
    analysis = Client(StreamableHttpTransport(args.pytemlib))
    async with analysis:

        async def structured(tool, payload):
            result = await analysis.call_tool(tool, payload, raise_on_error=False)
            if result.is_error:
                return {"error": text(result)}
            return result.structured_content or json.loads(text(result) or "{}")

        async def finish(job):
            deadline = time.time() + args.timeout
            while time.time() < deadline:
                status = await structured("analysis_job_status", {"job_id": job["job_id"]})
                if status.get("state") in {"done", "failed", "aborted"}:
                    break
                await asyncio.sleep(0.5)
            return status, await structured("analysis_job_results", {"job_id": job["job_id"]})

        job = await structured("assemble_map", {"run": run_key})
        status, stitched = await finish(job) if "job_id" in job else (job, {})
        detail = stitched.get("summary") if status.get("state") == "done" else status.get("last_error", status)
        if not report("pyTEMlib stitches the run from Tiled", status.get("state") == "done", json.dumps(detail)[:300]):
            return
        map_key = stitched["result_keys"]["map"]
        info = await structured("map_info", {"map": map_key})
        x0, x1, y0, y1 = info["stage_box_m"]
        region = await analysis.call_tool("get_region", {"map": map_key, "box_m": [x0, (x0 + x1) / 2, y0, (y0 + y1) / 2]}, raise_on_error=False)
        report("map region preview comes back as an image", not region.is_error and any(getattr(b, "type", "") == "image" for b in region.content))
        scored = await structured("score_map", {"map": map_key, "top_k": 2})
        _status, ranked = await finish(scored)
        added = ranked.get("summary", {}).get("added_pois", [])
        report("scorer adds points of interest inside the map", len(added) == 2 and all(x0 <= p["x_m"] <= x1 and y0 <= p["y_m"] <= y1 for p in added))
        poi = await structured("add_poi", {"map": map_key, "x_m": (x0 + x1) / 2, "y_m": (y0 + y1) / 2, "kind": "check", "reason": "check_stack"})
        report("agent can mark a point on the map", "id" in poi, json.dumps(poi)[:160])


async def main(args: argparse.Namespace) -> int:
    agent = Client(StreamableHttpTransport(args.agent, headers={"X-Asyncroscopy-Caller": "check_stack-agent"}))
    operator = Client(StreamableHttpTransport(args.operator, headers={"X-Asyncroscopy-Caller": "check_stack-operator"}))
    async with agent, operator:
        agent_tools = {t.name for t in await agent.list_tools()}
        operator_tools = {t.name for t in await operator.list_tools()}
        report("agent endpoint lists tools", len(agent_tools) > 10, f"{len(agent_tools)} tools")
        report("agent endpoint hides operator-only commands", not ({"Approvals_approve", "Approvals_set_mode"} & agent_tools) and not any(t.endswith("export_ground_truth") for t in agent_tools))
        report("operator endpoint offers approvals", {"Approvals_approve", "Approvals_set_mode", "Approvals_list_pending"} <= operator_tools)
        microscope = next((t.rsplit("_move_stage", 1)[0] for t in agent_tools if t.endswith("_move_stage")), None)
        if not report("a microscope with move_stage is exported", microscope is not None,
                      "" if microscope else f"only {len(agent_tools)} agent tools; were the MCP servers started before run_servers.py printed ready? Restart them"):
            return len(FAILURES)

        # A small map as one background job.
        plan = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": 1.6e-7, "overlap": 0.2, "imsize": 128, "dwell_s": 1e-6, "per_tile": ["eds_point"], "eds_exposure_s": 0.2}
        reserved = json.loads(text(await call(agent, "Mapper_reserve", {"request": json.dumps({"plan": plan, "caller": "check_stack"})})))
        if report("Mapper reserves a plan", "reservation_id" in reserved, json.dumps(reserved.get("estimate", reserved.get("error")))):
            started = json.loads(text(await call(agent, "Mapper_start", {"request": json.dumps({"reservation_id": reserved["reservation_id"]})})))
            job_id = started.get("job_id")
            deadline = time.time() + args.timeout
            status = {}
            while job_id and time.time() < deadline:
                # wait_for_job waits on the server side, so a model needs one call, not dozens.
                status = json.loads(text(await call(agent, "wait_for_job", {"device": "Mapper", "job_id": job_id, "max_wait_s": 25})))
                if status.get("state") in {"done", "failed", "aborted"}:
                    break
            report("map job finishes (via wait_for_job)", status.get("state") == "done", json.dumps(status))
            results = json.loads(text(await call(agent, "Mapper_results", {"request": json.dumps({"job_id": job_id})})))
            report("map run file is registered in Tiled", "run" in results.get("result_keys", {}), json.dumps(results.get("result_keys")))
            if "run" in results.get("result_keys", {}):
                data = await call(agent, "get_data_from_key", {"key": results["result_keys"]["run"], "max_values": 4})
                report("agent can read the map back by key", "tiles" in text(data) or bool(getattr(data, "structured_content", None)))
                if args.pytemlib:
                    await check_analysis(args, results["result_keys"]["run"])
            await call(agent, "Mapper_release", {"request": json.dumps({"reservation_id": reserved["reservation_id"]})})

        # Approval mode: hold, approve, retry once; then put back the mode the operator had.
        mode_before = text(await call(operator, "Approvals_get_mode")).strip().strip('"')
        await call(operator, "Approvals_set_mode", {"request": json.dumps({"mode": "approve", "who": "check_stack"})})
        try:
            target = {"position": [1e-8, -1e-8, 0.0, 0.0, 0.0]}
            held = await call(agent, f"{microscope}_move_stage", target, raise_on_error=False)
            message = text(held)
            request_id = message.split("request_id=")[1].split(")")[0] if "request_id=" in message else None
            report("approve mode holds a high-risk agent call", held.is_error and request_id is not None, message[:120])
            low = await call(agent, f"{microscope}_get_stage", raise_on_error=False)
            report("approve mode lets a low-risk call through", not low.is_error)
            if request_id:
                pending = json.loads(text(await call(operator, "Approvals_list_pending")))
                report("operator sees the pending request", any(p["request_id"] == request_id for p in pending))
                await call(operator, "Approvals_approve", {"request": json.dumps({"request_id": request_id, "who": "check_stack"})})
                retried = await call(agent, f"{microscope}_move_stage", target, raise_on_error=False)
                report("the approved call runs once", not retried.is_error, text(retried)[:120])
                again = await call(agent, f"{microscope}_move_stage", target, raise_on_error=False)
                report("the approval is single use", again.is_error)
            # A held job start shows the operator the plan it would run, not just an id.
            small = {"size_m": [1.6e-7, 1.6e-7], "fov_m": 1.6e-7, "imsize": 64}
            reserved = json.loads(text(await call(agent, "Mapper_reserve", {"request": json.dumps({"plan": small})})))
            if "reservation_id" in reserved:
                held = await call(agent, "Mapper_start", {"request": json.dumps({"reservation_id": reserved["reservation_id"]})}, raise_on_error=False)
                pending = json.loads(text(await call(operator, "Approvals_list_pending")))
                shown = [p for p in pending if p["action"] == "Mapper_start" and reserved["reservation_id"] in (p.get("reason") or "")]
                report("a held map start shows the operator its plan", held.is_error and bool(shown) and "size_m" in shown[0]["reason"], (shown[0]["reason"][:120] if shown else text(held)[:120]))
                await call(agent, "Mapper_release", {"request": json.dumps({"reservation_id": reserved["reservation_id"]})})
        finally:
            # Deny anything this check left pending (the single-use retry files a new request),
            # so the operator's approvals list is not left holding test calls.
            for leftover in json.loads(text(await call(operator, "Approvals_list_pending"))):
                if leftover.get("caller") == "check_stack-agent":
                    await call(operator, "Approvals_deny", {"request": json.dumps({"request_id": leftover["request_id"], "who": "check_stack", "reason": "left over from check_stack"})})
            await call(operator, "Approvals_set_mode", {"request": json.dumps({"mode": mode_before, "who": "check_stack"})})
        report(f"mode is back to {mode_before}", text(await call(agent, "Approvals_get_mode")) == mode_before)
    print(f"\n{len(FAILURES)} failure(s)" + (": " + ", ".join(FAILURES) if FAILURES else ""))
    return len(FAILURES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--operator", default="http://127.0.0.1:8001/mcp")
    parser.add_argument("--timeout", type=float, default=300.0, help="Seconds to wait for each job.")
    parser.add_argument("--pytemlib", default="", help="pyTEMlib MCP server URL (e.g. http://127.0.0.1:8003/mcp) to also check stitching and scoring.")
    sys.exit(asyncio.run(main(parser.parse_args())))
