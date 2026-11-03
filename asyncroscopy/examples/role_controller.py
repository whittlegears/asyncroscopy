"""A controller that knows roles, not command names: it runs unchanged on a STEM and on an AFM.

It reads the endpoint's action space (``action_space()``; see asyncroscopy.mcp.action_space) and,
using only canonical roles:

1. reads the stage (``stage.read``),
2. moves it by a small offset (``stage.move``), lifting a probe first if there is one (``probe.retract``),
3. acquires the instrument's main observable (``acquire.image``, else ``acquire.spectrum``), with the
   beam let through (``beam.unblank``) or the probe brought down (``probe.approach``) first, and
   both undone after (``beam.blank``, ``probe.retract``),
4. reads what it acquired back (``data.read``) and the stage again.

On the digital twin that is move_stage, acquire_scanned_image and unblank_beam; on the AFM YAML twin
move_xy, acquire_topography and approach_tip. Neither name appears below.

Against a running stack (the calls are real: it moves the stage and acquires)::

    python -m asyncroscopy.examples.role_controller --url http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import TYPE_CHECKING, Any

from asyncroscopy.mcp.action_space import DEFAULT_URL, actions_with, structured

if TYPE_CHECKING:
    from fastmcp import Client

MAIN_OBSERVABLES = ("acquire.image", "acquire.spectrum", "acquire.other")


class RoleController:
    """Calls tools by role on one MCP client (a fastmcp Client, already open)."""

    def __init__(self, client: Client, space: dict) -> None:
        self.client, self.space = client, space
        self.calls: list[tuple[str, str]] = []  # (role, tool) in call order
        # The main observable (an acquisition marked primary, else the first image, spectrum, ...); every
        # other role is looked up on its instrument.
        candidates = [a for role in MAIN_OBSERVABLES for a in actions_with(space, role)]
        acquire = next((a for a in candidates if a.get("primary")), candidates[0] if candidates else None)
        if acquire is None:
            raise RuntimeError("this endpoint has no acquire.* action")
        self.acquire, self.device = acquire, acquire["device"]

    def find(self, role: str) -> list[dict]:
        return actions_with(self.space, role, self.device)

    async def call(self, action: dict, arguments: dict | None = None) -> Any:
        self.calls.append((action["role"], action["tool"]))
        return structured(await self.client.call_tool(action["tool"], arguments or {}))

    async def each(self, role: str) -> None:
        """Call every no-argument action with ``role`` (blank the beam, retract the probe)."""
        for action in self.find(role):
            if not any(a.get("required") for a in action["args"]):
                await self.call(action)

    async def position(self) -> dict[str, float]:
        """The stage position by axis, from every stage.read action (one array, or one value per axis)."""
        found: dict[str, float] = {}
        for action in self.find("stage.read"):
            returns = action.get("returns") or {}
            if not returns.get("elements") and not returns.get("axis"):
                continue
            value = await self.call(action)
            if returns.get("elements"):
                found.update(zip(returns["elements"], (float(v) for v in value)))
            else:
                found[returns["axis"]] = float(value)
        return found

    async def move(self, target: dict[str, float], current: dict[str, float]) -> None:
        """Move with the one stage.move action that sets every axis in ``target``, filling the rest from ``current``."""
        for action in self.find("stage.move"):
            arguments = {}
            for arg in action["args"]:
                if arg.get("elements"):
                    arguments[arg["name"]] = [target.get(e, current.get(e)) for e in arg["elements"]]
                elif arg.get("axis"):
                    arguments[arg["name"]] = target.get(arg["axis"], current.get(arg["axis"]))
            given = {e for a in action["args"] for e in (a.get("elements") or [a.get("axis")]) if e}
            values = [v for value in arguments.values() for v in (value if isinstance(value, list) else [value])]
            if set(target) <= given and None not in values:
                await self.call(action, arguments)
                return
        raise RuntimeError(f"no stage.move action on {self.device} sets {sorted(target)}")

    async def observe(self) -> Any:
        """Acquire the main observable, with whatever must be on (beam, probe) only for the acquisition."""
        await self.each("probe.approach")
        await self.each("beam.unblank")
        try:
            # Required arguments the controller knows nothing about get their empty value: [] for a list.
            arguments = {a["name"]: [] for a in self.acquire["args"] if a.get("required") and a.get("type") == "array"}
            return await self.call(self.acquire, arguments)
        finally:
            await self.each("beam.blank")
            await self.each("probe.retract")

    async def read_back(self, key: str) -> Any:
        """What an acquisition saved, through the data.read action that takes a key."""
        reader = next((a for a in actions_with(self.space, "data.read") if any(arg["name"] == "key" for arg in a["args"])), None)
        if reader is None:
            raise RuntimeError("this endpoint has no data.read action that takes a key")
        extra = {"format": "text"} if any(arg["name"] == "format" for arg in reader["args"]) else {}
        return await self.call(reader, {"key": key, **extra})


async def run(client: Client, offset_m: tuple[float, float] = (1e-7, 1e-7)) -> dict:
    """Move the stage by ``offset_m`` (x, y), acquire, read it back. Returns what happened."""
    space = structured(await client.call_tool("action_space", {}))
    controller = RoleController(client, space)
    start = await controller.position()
    if not {"x", "y"} <= set(start):
        raise RuntimeError(f"stage.read on {controller.device} gives no x and y (got {sorted(start)})")
    target = {"x": start["x"] + offset_m[0], "y": start["y"] + offset_m[1]}
    await controller.each("probe.retract")  # a probe never drags over the sample
    await controller.move(target, start)
    reached = await controller.position()
    key = await controller.observe()
    readback = await controller.read_back(str(key))
    return {
        "device": controller.device, "start": start, "target": target, "reached": reached, "key": key,
        "readback": readback, "calls": controller.calls,
    }


async def _main(url: str, offset_m: float) -> dict:
    from fastmcp import Client

    async with Client(url) as client:
        return await run(client, (offset_m, offset_m))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Move, acquire and read back on any instrument, by canonical role.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--offset-m", type=float, default=1e-7, help="stage move in x and y, metres (default 1e-7)")
    args = parser.parse_args(argv)
    report = asyncio.run(_main(args.url, args.offset_m))
    for role, tool in report["calls"]:
        print(f"{role:<16} {tool}")
    print(json.dumps({k: v for k, v in report.items() if k != "calls"}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
