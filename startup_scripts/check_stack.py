"""Check a running asyncroscopy stack end to end, the way an agent and the GUI use it.

Start the twin and its MCP endpoint first (the second command starts the agent endpoint, which the GUI uses too):

    uv run startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml
    uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml

then:

    uv run startup_scripts/check_stack.py

Each check prints PASS or FAIL with the reason; the exit code is the number of
failures. It moves the stage and acquires, so it runs only on a simulated
stack: it asks the agent endpoint (endpoint_info's ``simulated``, the stack's
top-level ``simulated:``) first, and refuses when the answer is not true.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import socket
import sys
import time
import urllib.parse

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)
    return ok


def text(result: CallToolResult) -> str:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and isinstance(structured.get("result"), str):
        return structured["result"]
    return "".join(getattr(block, "text", "") or "" for block in (result.content or []))


async def call(client: Client, tool: str, args: dict | None = None, raise_on_error: bool = True) -> CallToolResult:
    return await client.call_tool(tool, args or {}, raise_on_error=raise_on_error)


async def check_analysis(args: argparse.Namespace, run_key: str, agent: Client) -> None:
    """Stitch the run on the analysis MCP server, look at it, score it and mark a point."""
    analysis = Client(StreamableHttpTransport(args.analysis))
    async with analysis:

        async def structured(tool, payload):
            result = await analysis.call_tool(tool, payload, raise_on_error=False)
            if result.is_error:
                return {"error": text(result)}
            return result.structured_content or json.loads(text(result) or "{}")

        async def finish(job):
            deadline = time.time() + args.timeout
            status = {"state": "not checked", "last_error": f"--timeout {args.timeout} left no time to wait"}
            while time.time() < deadline:
                status = await structured("analysis_job_status", {"job_id": job["job_id"]})
                if status.get("state") in {"done", "failed", "aborted"}:
                    break
                await asyncio.sleep(0.5)
            return status, await structured("analysis_job_results", {"job_id": job["job_id"]})

        job = await structured("assemble_map", {"run": run_key})
        status, stitched = await finish(job) if "job_id" in job else (job, {})
        detail = stitched.get("summary") if status.get("state") == "done" else status.get("last_error", status)
        if not report("the analysis server stitches the run from Tiled", status.get("state") == "done", json.dumps(detail)[:300]):
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
        # One wait covers every kind of job. The agent endpoint's wait_for_job also waits on analysis jobs.
        again = await structured("assemble_map", {"run": run_key})
        status = {}
        deadline = time.time() + args.timeout
        while "job_id" in again and time.time() < deadline:
            status = json.loads(text(await call(agent, "wait_for_job", {"job_id": again["job_id"]})))
            if status.get("state") in {"done", "failed", "aborted"}:
                break
        report("wait_for_job waits on an analysis job and returns its map key", "_map" in status.get("result_keys", {}).get("map", ""), json.dumps(status)[:200])


async def simulated_stack(agent: Client) -> tuple[bool, str]:
    """Whether the stack drives a simulator, as the agent endpoint's endpoint_info says (run_servers'
    top-level ``simulated:``). Anything but a clear true counts as hardware."""
    try:
        result = await agent.call_tool("endpoint_info", {}, raise_on_error=False)
    except Exception as exc:
        return False, f"could not ask endpoint_info ({type(exc).__name__}: {exc})"
    if result.is_error:
        return False, f"endpoint_info failed: {text(result)[:200]}"
    info = getattr(result, "structured_content", None) or json.loads(text(result) or "{}")
    answer = info.get("simulated") if isinstance(info, dict) else None
    return answer is True, f"endpoint_info says simulated={answer!r}"


def unreachable(*urls: str) -> list[str]:
    """The URLs nothing is listening on (a stack that was not started yet)."""
    down = []
    for url in urls:
        parts = urllib.parse.urlsplit(url)
        try:
            socket.create_connection((parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)), timeout=2).close()
        except (OSError, ValueError):  # ValueError means a port out of range
            down.append(url)
    return down


async def check_endpoints(agent: Client, operator: Client | None) -> str | None:
    """Check the tools the endpoints list and give the name of the microscope. The operator endpoint is optional."""
    agent_tools = {t.name for t in await agent.list_tools()}
    report("agent endpoint lists tools", len(agent_tools) > 10, f"{len(agent_tools)} tools")
    report("agent endpoint hides operator-only commands", not any(t.endswith("export_ground_truth") for t in agent_tools))
    report("agent endpoint offers stop_all", "stop_all" in agent_tools)
    if operator is not None:
        operator_tools = {t.name for t in await operator.list_tools()}
        report("operator endpoint lists tools", len(operator_tools) > 10, f"{len(operator_tools)} tools")
    microscope = next((t.rsplit("_move_stage", 1)[0] for t in agent_tools if t.endswith("_move_stage")), None)
    report("a microscope with move_stage is exported", microscope is not None,
           "" if microscope else f"only {len(agent_tools)} agent tools; were the MCP servers started before run_servers.py printed ready? Restart them")
    return microscope


async def check_map_results(args: argparse.Namespace, agent: Client, results: dict) -> None:
    """Read the map run and the newest image back as an agent would."""
    data = await call(agent, "view_data", {"key": results["result_keys"]["run"], "format": "text"})
    report("agent can read the map run back by key (view_data text)", "tiles" in text(data), text(data)[:200])
    listed = json.loads(text(await call(agent, "list_data", {"kind": "image", "limit": 1})))
    newest = (listed.get("data") or [{}])[0].get("key")
    picture = await call(agent, "view_data", {"key": newest, "format": "png"}) if newest else None
    report("agent can find the newest image (list_data) and see it (view_data png)",
           any(getattr(block, "type", "") == "image" for block in getattr(picture, "content", []) or []), str(newest))
    if args.analysis:
        await check_analysis(args, results["result_keys"]["run"], agent)


async def check_map_job(args: argparse.Namespace, agent: Client) -> None:
    """Run a small map as one background job and check its results."""
    plan = {"box_m": [-1.5e-7, 1.5e-7, -1.5e-7, 1.5e-7], "fov_m": 1.6e-7, "overlap": 0.2, "imsize": 128, "dwell_s": 1e-6, "per_tile": ["eds_point"], "eds_exposure_s": 0.2}
    reserved = json.loads(text(await call(agent, "Mapper_reserve", {"request": json.dumps({"plan": plan, "caller": "check_stack"})})))
    if report("Mapper reserves a plan", "reservation_id" in reserved, json.dumps(reserved.get("estimate", reserved.get("error")))):
        started = json.loads(text(await call(agent, "Mapper_start", {"request": json.dumps({"reservation_id": reserved["reservation_id"]})})))
        job_id = started.get("job_id")
        deadline = time.time() + args.timeout
        status = {}
        while job_id and time.time() < deadline:
            # wait_for_job waits on the server side so a model needs one call and not dozens.
            status = json.loads(text(await call(agent, "wait_for_job", {"device": "Mapper", "job_id": job_id, "max_wait_s": 25})))
            if status.get("state") in {"done", "failed", "aborted"}:
                break
        report("map job finishes (via wait_for_job)", status.get("state") == "done", json.dumps(status))
        results = json.loads(text(await call(agent, "Mapper_results", {"request": json.dumps({"job_id": job_id})})))
        report("map run file is registered in Tiled", "run" in results.get("result_keys", {}), json.dumps(results.get("result_keys")))
        if "run" in results.get("result_keys", {}):
            await check_map_results(args, agent, results)
        await call(agent, "Mapper_release", {"request": json.dumps({"reservation_id": reserved["reservation_id"]})})


async def check_run_job(args: argparse.Namespace, agent: Client) -> None:
    """Run a small map as one run_job call and wait for its run key."""
    tile = {"size_m": [2.4e-7, 2.4e-7], "fov_m": 1.6e-7, "imsize": 64, "dwell_s": 1e-6}
    ran = json.loads(text(await call(agent, "run_job", {"device": "Mapper", "plan": tile})))
    status = {}
    deadline = time.time() + args.timeout
    while "job_id" in ran and time.time() < deadline:
        status = json.loads(text(await call(agent, "wait_for_job", {"job_id": ran["job_id"]})))
        if status.get("state") in {"done", "failed", "aborted"}:
            break
    report("run_job then wait_for_job maps and returns the run key", "run" in status.get("result_keys", {}), json.dumps(status)[:200])


async def main(args: argparse.Namespace) -> int:
    down = unreachable(*[url for url in (args.agent, args.operator) if url])
    if down:
        report("MCP endpoints reachable", False, f"nothing at {', '.join(down)}: start the stack (quickstart section 3) and wait for 'MCP ready'")
        return 1
    agent = Client(StreamableHttpTransport(args.agent, headers={"X-Asyncroscopy-Caller": "check_stack-agent"}))
    operator = Client(StreamableHttpTransport(args.operator, headers={"X-Asyncroscopy-Caller": "check_stack-operator"})) if args.operator else None
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(agent)
        if operator is not None:
            await stack.enter_async_context(operator)
        simulated, why = await simulated_stack(agent)
        if not report("the stack is simulated", simulated,
                      why if simulated else f"{why}. This check moves the stage and acquires: it runs only on a simulated stack"):
            return len(FAILURES)
        microscope = await check_endpoints(agent, operator)
        if microscope is None:
            return len(FAILURES)

        # A small map as one background job.
        await check_map_job(args, agent)

        # The same as one run_job call. The wait's reply carries the results.
        await check_run_job(args, agent)
    print(f"\n{len(FAILURES)} failure(s)" + (": " + ", ".join(FAILURES) if FAILURES else ""))
    return len(FAILURES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--operator", default="", help="Optional operator endpoint (configs/mcp_operator.yaml, e.g. http://127.0.0.1:8001/mcp) to also check.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Seconds to wait for each job.")
    parser.add_argument("--analysis", default="", help="Analysis MCP server URL (e.g. http://127.0.0.1:8003/mcp) to also check stitching and scoring.")
    sys.exit(asyncio.run(main(parser.parse_args())))
