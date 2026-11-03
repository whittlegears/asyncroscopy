"""Print what an agent sees on a running agent endpoint: every tool, its risk, what auto mode does with it,
and which tools a small model is given up front. It only reads; nothing on the instrument changes.

Start the stack first (see docs/Operation/quickstart.md), then:

    uv run startup_scripts/show_agent_view.py
    uv run startup_scripts/show_agent_view.py --url http://127.0.0.1:8000/mcp --json

Columns: risk (low: never held; high: held in approve mode), auto (what auto mode does with a high-risk
tool: runs, held until an operator switches it on, or a job start that runs when every command it drives
runs), model (yes: in the tool list a small model is given; no: reachable through find_tools and call_tool).
A new command on your device shows up as high and held: that is on purpose. Mark a read-only command low
with mcp.safety.risk in your MCP config, or switch it on for auto mode in SciAgentGUI (Settings, Agent
permissions) or with Approvals.set_auto_allowed.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import os
import sys
from typing import Any

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport


def payload(result: CallToolResult) -> Any:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        value = structured.get("result", structured) if set(structured) == {"result"} else structured
    else:
        value = "".join(getattr(block, "text", "") or "" for block in (result.content or []))
    if isinstance(value, str):  # device commands return their JSON as a string
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def rows_from(info: dict, names: list[str], switched_on: list[str]) -> list[dict]:
    risks, auto, patterns = info.get("risks") or {}, info.get("auto") or {}, info.get("agent_tools") or []
    rows = []
    for name in sorted(names):
        risk = risks.get(name, "low")
        if risk == "low":
            treatment = "runs"
        elif any(fnmatch.fnmatchcase(name.lower(), p.lower()) for p in switched_on):
            treatment = "runs (switched on)"
        else:
            treatment = {"routine": "runs (routine)", "job": "runs if its job drives only commands that run", "held": "held until switched on"}.get(auto.get(name, "held"), "held until switched on")
        rows.append({"tool": name, "risk": risk, "auto": treatment, "model": "yes" if any(fnmatch.fnmatchcase(name, p) for p in patterns) else "no"})
    return rows


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp", help="The agent endpoint (default: %(default)s).")
    parser.add_argument("--json", action="store_true", help="Print the rows as JSON instead of a table.")
    parser.add_argument("--token", default=os.environ.get("ASYNCROSCOPY_MCP_TOKEN", ""), help="Bearer token for an endpoint with mcp.auth (default: $ASYNCROSCOPY_MCP_TOKEN).")
    args = parser.parse_args(argv)
    headers = {"X-Asyncroscopy-Caller": "show_agent_view", **({"Authorization": f"Bearer {args.token}"} if args.token else {})}
    client = Client(StreamableHttpTransport(args.url, headers=headers))
    async with client:
        names = [t.name for t in await client.list_tools()]
        info = payload(await client.call_tool("endpoint_info", {}))
        switched_on: list[str] = []
        mode = None
        if "Approvals_get_auto_allowed" in names:
            allowed = payload(await client.call_tool("Approvals_get_auto_allowed", {}))
            switched_on = allowed.get("switched_on", []) if isinstance(allowed, dict) else []
            mode = payload(await client.call_tool("Approvals_get_mode", {}))
    if not isinstance(info, dict) or "risks" not in info:
        print(f"{args.url} is not an asyncroscopy agent endpoint (no endpoint_info risks)", file=sys.stderr)
        return 1
    rows = rows_from(info, names, switched_on)
    if not rows:
        print(f"{args.url} offers no tools to this caller", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"mode": mode, "switched_on": switched_on, "tools": rows}, indent=2))
        return 0
    width = max(len(r["tool"]) for r in rows)
    print(f"{args.url}: role {info.get('role')}, approval mode {mode or 'none (no Approvals device)'}, switched on: {', '.join(switched_on) or 'none'}")
    print(f"{'tool'.ljust(width)}  risk  model  auto mode")
    for r in rows:
        print(f"{r['tool'].ljust(width)}  {r['risk'].ljust(4)}  {r['model'].ljust(5)}  {r['auto']}")
    held = [r["tool"] for r in rows if r["auto"].startswith("held")]
    print(f"\n{len(rows)} tools; {sum(r['model'] == 'yes' for r in rows)} given to a small model up front; {len(held)} held in auto mode until switched on.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
