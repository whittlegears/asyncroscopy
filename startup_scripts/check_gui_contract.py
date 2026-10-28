"""Check the stack answers the way SciAgentGUI's thin client expects.

Replays the GUI's operator-endpoint calls (approval mode, pending requests,
approve/deny with JSON-string requests) and its Tiled reads (stitched-map
search, level arrays, region slices, linked EDS spectra), and checks each reply
against what sciagentgui/src/lib/asyncroscopy.ts and tiledClient.ts parse.
Run against a live stack (see docs/Operation/quickstart.md):

    uv run startup_scripts/check_gui_contract.py

It switches approval mode to approve and back, and denies the requests it causes.
"""
import asyncio
import json
import socket
import sys
import urllib.parse
import urllib.request

import numpy as np
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

TILED = "http://localhost:9091"
HEADERS = {"Authorization": "Apikey secret"}
CALLER = "gui-contract-agent"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  -- {detail}" if detail else ""))
    return ok


def get(path, accept="application/json"):
    req = urllib.request.Request(TILED + path, headers={**HEADERS, "Accept": accept})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
    return json.loads(body) if accept == "application/json" else body


def text(result):
    return "".join(b.text for b in result.content if getattr(b, "type", "") == "text")


async def mcp_side():
    async with Client("http://127.0.0.1:8001/mcp") as op:
        names = {t.name for t in await op.list_tools()}
        check("operator role detected (Approvals_approve + Approvals_set_mode)", {"Approvals_approve", "Approvals_set_mode"} <= names)
        mode = text(await op.call_tool("Approvals_get_mode", {}))
        mode = mode.strip().strip('"')
        if not check("parseMode accepts get_mode reply", mode in ("auto", "approve"), mode):
            return  # without the current mode there is nothing safe to restore
        reply = text(await op.call_tool("Approvals_set_mode", {"request": json.dumps({"mode": "approve", "who": "gui-probe"})}))
        check("set_mode with a JSON-string request", json.loads(reply)["mode"] == "approve", reply)
        try:
            async with Client(StreamableHttpTransport("http://127.0.0.1:8000/mcp", headers={"X-Asyncroscopy-Caller": CALLER})) as agent:
                microscope = next((t.name[: -len("_unblank_beam")] for t in await agent.list_tools() if t.name.endswith("_unblank_beam")), None)
                if not check("agent endpoint offers a microscope (<name>_unblank_beam)", microscope is not None,
                             "" if microscope else "none found: were the MCP servers started before run_servers.py printed ready? Restart them"):
                    return
                held_text = text(await agent.call_tool(f"{microscope}_unblank_beam", {}, raise_on_error=False))
            check("agent high-risk call is held in approve mode", "APPROVAL REQUIRED" in held_text, held_text[:120])
            pending = json.loads(text(await op.call_tool("Approvals_list_pending", {})))
            ok = isinstance(pending, list) and pending and all(isinstance(p.get("request_id"), str) and isinstance(p.get("action"), str) and isinstance(p.get("created"), (int, float)) for p in pending)
            check("parsePending fields (request_id, action, created, state)", bool(ok), json.dumps(pending[:1])[:200])
            for p in [p for p in pending if p.get("caller") == CALLER]:
                denied = text(await op.call_tool("Approvals_deny", {"request": json.dumps({"request_id": p["request_id"], "who": "gui-probe", "reason": "probe"})}))
                check("deny with JSON-string request", json.loads(denied)["state"] == "denied")
        finally:
            # Whatever failed above, do not leave the instrument holding every call.
            reply = text(await op.call_tool("Approvals_set_mode", {"request": json.dumps({"mode": mode, "who": "gui-probe"})}, raise_on_error=False))
            try:
                restored = json.loads(reply)["mode"] == mode
            except (ValueError, TypeError, KeyError):
                restored = False
            check(f"mode restored to {mode}", restored, "" if restored else reply[:160])


def tiled_side():
    query = "&".join([
        "filter[eq][condition][key]=kind",
        "filter[eq][condition][value]=" + urllib.parse.quote('"stitched_map"'),
        "fields=metadata", "page[offset]=0", "page[limit]=100",
    ])
    page = get(f"/api/v1/search/?{query}")
    maps = page.get("data") or []
    check("listStitchedMaps finds kind=stitched_map nodes", len(maps) > 0, f"{len(maps)} maps")
    if not maps:
        return
    with_eds = [m for m in maps if any(p.get("kind") == "eds_point" for p in m["attributes"]["metadata"].get("pois", []))]
    node = (with_eds or maps)[-1]
    key, meta = node["id"], node["attributes"]["metadata"]
    ok = isinstance(meta.get("pixel_size_m"), (int, float)) and meta["pixel_size_m"] > 0 and len(meta.get("origin_xy_m", [])) == 2 and len(meta.get("canvas_shape", [])) == 2
    check("parseMapMetadata required fields", ok, key)
    levels_page = get(f"/api/v1/search/{urllib.parse.quote(key)}?fields=structure&fields=metadata&page[limit]=100")
    levels = []
    for child in levels_page.get("data") or []:
        s = child["attributes"].get("structure") or {}
        if child["id"].startswith("level_") and len(s.get("shape", [])) == 2 and s.get("data_type"):
            levels.append((child["id"], s["shape"], s["data_type"]))
    check("getLevels finds level_N 2-D arrays with data_type", len(levels) >= 1, str([(l[0], l[1]) for l in levels]))
    stored = meta.get("level_shapes")
    if stored:
        check("level_shapes metadata matches the arrays", [list(l[1]) for l in sorted(levels)] == [list(s) for s in stored], f"{stored}")
    name, shape, dtype = sorted(levels)[0]
    r1, c1 = min(64, shape[0]), min(64, shape[1])
    buf = get(f"/api/v1/array/full/{urllib.parse.quote(key)}/{name}?slice=0:{r1},0:{c1}", "application/octet-stream")
    kind = dtype.get("kind"); size = dtype.get("itemsize")
    np_dtype = np.dtype(("<" if dtype.get("endianness", "little") == "little" else ">") + kind + str(size))
    values = np.frombuffer(buf, dtype=np_dtype)
    check("fetchRegion returns rows*cols values", values.size == r1 * c1, f"{values.size} values, dtype {np_dtype}")
    eds = [p for p in meta.get("pois", []) if p.get("kind") == "eds_point" and p.get("links")]
    if eds:
        link = eds[0]["links"][0]
        n = get(f"/api/v1/metadata/{urllib.parse.quote(link)}")["data"]
        family = n["attributes"]["structure_family"]
        if family == "container":
            kids = get(f"/api/v1/search/{urllib.parse.quote(link)}?fields=structure&fields=metadata&fields=structure_family&page[limit]=100")["data"]
            spec = next((k for k in kids if "energy_offset_ev" in k["attributes"]["metadata"] and len(k["attributes"]["structure"]["shape"]) == 1), None)
        else:
            spec = n if "energy_offset_ev" in n["attributes"]["metadata"] else None
        check("resolveSpectrum finds a 1-D array with energy_offset_ev/energy_dispersion_ev", spec is not None, link)
    else:
        print("SKIP spectrum check: newest map has no linked EDS points")


def unreachable(*urls: str) -> list[str]:
    """The URLs nothing is listening on (a stack that was not started yet)."""
    down = []
    for url in urls:
        parts = urllib.parse.urlsplit(url)
        try:
            socket.create_connection((parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)), timeout=2).close()
        except (OSError, ValueError):  # ValueError: a port out of range
            down.append(url)
    return down


down = unreachable("http://127.0.0.1:8000/mcp", "http://127.0.0.1:8001/mcp", TILED)
if down:
    check("stack reachable", False, f"nothing at {', '.join(down)}: start the stack (quickstart section 3) and wait for 'MCP ready'")
    sys.exit(1)
asyncio.run(mcp_side())
tiled_side()
print(f"\n{results.count(False)} failure(s) of {len(results)}")
sys.exit(1 if False in results else 0)
