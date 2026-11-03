"""Helpers shared by the benchmark runners: model warm-up, infrastructure errors, the
Instrument Trace, live tool lookup and the "sleepwalking" count."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from asyncroscopy.safety.policy import tool_risk
from asyncroscopy.skills.registry import list_skills

if TYPE_CHECKING:
    from asyncroscopy.agents.config import AgentConfig

TRACE_DIR = Path("outputs/trace")

# A crash of the model server itself (seen with Ollama on a 6 GB GPU on the first load after a
# restart) says nothing about the agent; such rows are reported apart and left out of the pass rate.
# Failures of the stack, not the agent: a crashed model server, a device that stopped answering.
INFRA_ERRORS = ("llama-server process has terminated", "model runner has unexpectedly stopped", "CUDA error",
                "API_DeviceTimedOut", "Not able to acquire serialization")


def is_infra_error(error: str | None) -> bool:
    return any(marker in (error or "") for marker in INFRA_ERRORS)


async def warm_up(config: AgentConfig) -> None:
    """Load the model once before the first trial, retrying if the first load crashes the server."""
    for attempt in range(3):
        try:
            await config.chat_model().ainvoke("Reply with OK.")
            return
        except Exception as exc:
            print(f"warm-up {attempt + 1} failed: {exc}", flush=True)
            await asyncio.sleep(5)


def skill_tools(name: str = "mapping") -> set[str]:
    """The tools a skill lists in its front matter (empty if there is no such skill)."""
    return set(next((s.tools for s in list_skills() if s.name == name), []))


def matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def sleepwalking(tool_names: list[str], allowed: Iterable[str] | None = None, risks: dict[str, str] | None = None) -> list[str]:
    """High-risk calls outside ``allowed`` (globs): actions nobody asked for.

    The term is from AILA (Nat. Commun. 2025). Risk is the endpoint's own table (``risks``,
    from endpoint_info: overrides and STEM settings counted as the approval gate counts them),
    else the safety policy applied to the name. ``allowed`` defaults to the mapping skill's
    tools. The job device's reserve and start that run_job makes are its own: the Instrument
    Trace marks them (via "run_job"), and the workflow checks leave them out when run_job is
    allowed; a direct ``Mapper_start`` still counts.
    """
    allowed = skill_tools() if allowed is None else set(allowed)
    return [name for name in tool_names if not matches(name, allowed) and tool_risk(name, risks) == "high"]


async def endpoint_risks(config: AgentConfig) -> dict[str, str] | None:
    """The live endpoint's tool risk table (endpoint_info), or None if it has none."""
    return (await config.info()).get("risks") or None


def read_trace(since: float, callers: Iterable[str] | None = None, directory: Path = TRACE_DIR) -> list[dict]:
    """Instrument Trace events (every outputs/trace/*.jsonl) at or after ``since``, oldest first,
    from ``callers`` when given (an event's caller, or for an approval decision the request's caller)."""
    events = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                if number == len(lines):  # the last line may still be being written
                    continue
                raise ValueError(f"{path} line {number} is not JSON; the trace checks cannot be trusted") from exc
            if event.get("ts", 0) < since:
                continue
            event.setdefault("file", path.stem)
            events.append(event)
    events.sort(key=lambda e: e.get("ts", 0))
    if callers is None:
        return events
    callers = set(callers)
    held = {e.get("request_id") for e in events if e.get("caller") in callers and e.get("request_id")}
    return [e for e in events if e.get("caller") in callers or (e.get("request_id") in held and e.get("source") == "approvals")]


def out_of_box_moves(trace_path: Path, since: float) -> int:
    """Mapper steps whose reported stage position lies outside their plan box (should always be 0)."""
    if not trace_path.exists():
        return -1
    return sum(1 for line in trace_path.read_text(encoding="utf-8").splitlines() if '"Mapper.step"' in line and '"ok": false' in line and json.loads(line).get("ts", 0) >= since and "outside the plan box" in line)


def find_tool(names: Iterable[str], *suffixes: str, prefix: str = "") -> str | None:
    """The first live tool whose name starts with ``prefix`` and ends with one of ``suffixes``
    (tool names follow the devices exported, so benchmarks never hard-code them)."""
    for suffix in suffixes:
        for name in sorted(names):
            if name.startswith(prefix) and name.endswith(suffix):
                return name
    return None


def unique_caller(prefix: str) -> str:
    """A caller name no earlier run used, so its trace events can be told apart."""
    return f"{prefix}-{time.strftime('%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
