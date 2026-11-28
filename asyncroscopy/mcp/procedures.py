"""Calibration procedures as a small MCP server that only talks to other MCP servers.

Each tool here is a complete, bounded procedure (focus on stage Z, sweep defocus,
tune one aberration). It drives the instrument exclusively through the STEM MCP
server's tools and does all image analysis through the pyTEMlib MCP server, so it
needs neither pytango nor the analysis libraries. What it adds is what an agent
calling the low-level tools one by one cannot guarantee:

- hard limits from the YAML config that no tool argument can exceed,
- the starting setting is restored if a run fails part-way,
- one call instead of dozens of model turns.

Instrument tool names differ between instruments (``DigitalTwin_move_stage`` vs
``AutoScriptMicroscope_move_stage``), so each role is found by command suffix and
can be pinned in the config when more than one tool matches.
"""
from __future__ import annotations

import argparse
import json
import math
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastmcp import Client, FastMCP

INSTRUMENT_ROLES = {
    "get_stage": "get_stage",
    "move_stage": "move_stage",
    "acquire_image": "acquire_scanned_image",
    "get_defocus": "get_defocus",
    "set_defocus": "set_defocus",
    "correct_aberration": "correct_aberration",
}

ANALYSIS_TOOLS = ("compute_image_metrics", "propose_next_setting", "fit_parabola_vertex")

METRICS = ("contrast", "entropy", "high_frequency_power", "edge_sharpness")


@dataclass
class Limits:
    """Hard bounds; a request outside them is refused, never clipped silently."""

    stage_z_max_offset_m: float = 2e-6
    stage_z_max_step_m: float = 200e-9
    defocus_max_range_m: float = 100e-9
    aberration_max_offset_m: dict[str, float] = field(default_factory=lambda: {"C1": 20e-9, "A1": 20e-9})
    aberration_max_step_m: float = 5e-9
    max_acquisitions: int = 40


@dataclass
class ProceduresConfig:
    instrument_url: str = "http://127.0.0.1:8000/mcp"
    analysis_url: str = "http://127.0.0.1:8003/mcp"
    analysis_transport: str = "sse"
    tool_names: dict[str, str] = field(default_factory=dict)
    limits: Limits = field(default_factory=Limits)
    name: str = "Calibration_Procedures"
    http_host: str = "127.0.0.1"
    http_port: int = 8010

    @classmethod
    def from_yaml(cls, path: Path) -> "ProceduresConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        section = raw.get("procedures", raw)
        limits = Limits(**(section.pop("limits", None) or {}))
        known = {key: value for key, value in section.items() if key in cls.__dataclass_fields__}
        unknown = sorted(set(section) - set(known))
        if unknown:
            raise ValueError(f"unknown procedures config keys: {unknown}")
        return cls(limits=limits, **known)


class ProcedureError(RuntimeError):
    pass


def _transport(url: str, transport: str):
    if transport == "sse":
        from fastmcp.client.transports import SSETransport
        return SSETransport(url)
    if transport == "streamable-http":
        from fastmcp.client.transports import StreamableHttpTransport
        return StreamableHttpTransport(url)
    raise ValueError(f"unsupported transport {transport!r}")


def components_of(aberration: str) -> int:
    """Round aberrations (C1, C3, C5) have one coefficient; all others are two-fold (x, y)."""
    return 1 if aberration[:1].upper() == "C" else 2


class Session:
    """Open connections to both MCP servers for one procedure run."""

    def __init__(self, config: ProceduresConfig, instrument: Client, analysis: Client):
        self.config = config
        self.instrument = instrument
        self.analysis = analysis
        self.instrument_tools: dict[str, Any] = {}
        self.analysis_tools: dict[str, Any] = {}
        self.acquisitions = 0

    async def discover(self) -> None:
        self.instrument_tools = {tool.name: tool for tool in await self.instrument.list_tools()}
        self.analysis_tools = {tool.name: tool for tool in await self.analysis.list_tools()}
        missing = [name for name in ANALYSIS_TOOLS if name not in self.analysis_tools]
        if missing:
            raise ProcedureError(f"the analysis server does not offer {missing}; is it the pyTEMlib MCP server?")

    def tool_for(self, role: str) -> Any:
        pinned = self.config.tool_names.get(role)
        if pinned:
            if pinned not in self.instrument_tools:
                raise ProcedureError(f"configured tool {pinned!r} for {role} is not offered by the instrument server")
            return self.instrument_tools[pinned]
        suffix = "_" + INSTRUMENT_ROLES[role]
        matches = [tool for name, tool in self.instrument_tools.items() if name.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ProcedureError(f"the instrument server offers no *{suffix} tool, needed to {role.replace('_', ' ')}")
        raise ProcedureError(
            f"several instrument tools match *{suffix}: {sorted(tool.name for tool in matches)}; "
            f"set tool_names.{role} in the procedures config"
        )

    async def _call(self, client: Client, tool: Any, value: Any = None, arguments: dict | None = None) -> Any:
        if arguments is None:
            properties = list((tool.input_schema or {}).get("properties", {}))
            if value is None:
                arguments = {}
            elif len(properties) == 1:
                arguments = {properties[0]: value}
            else:
                raise ProcedureError(f"{tool.name} takes {properties}; cannot pass a single value")
        result = await client.call_tool(tool.name, arguments, raise_on_error=False)
        text = next((item.text for item in result.content if getattr(item, "text", None) is not None), None)
        if result.is_error:
            raise ProcedureError(f"{tool.name} failed: {text}")
        structured = result.structured_content or {}
        if "result" in structured:
            return structured["result"]
        return text

    async def instrument_call(self, role: str, value: Any = None) -> Any:
        return await self._call(self.instrument, self.tool_for(role), value)

    async def analysis_call(self, name: str, arguments: dict) -> dict:
        reply = await self._call(self.analysis, self.analysis_tools[name], arguments=arguments)
        return json.loads(reply) if isinstance(reply, str) else reply

    async def measure(self, metric: str, detector: str) -> tuple[float, str]:
        if self.acquisitions >= self.config.limits.max_acquisitions:
            raise ProcedureError(f"stopped at the acquisition limit ({self.config.limits.max_acquisitions})")
        key = await self.instrument_call("acquire_image", [detector])
        self.acquisitions += 1
        result = await self.analysis_call(
            "compute_image_metrics", {"source": key, "dataset_key": detector.upper(), "metrics": [metric]}
        )
        return float(result["metrics"][metric]), key


class TrustRegion:
    """Grow the search radius after an improvement, shrink it otherwise; stop when it is tiny."""

    def __init__(self, step: list[float], max_step: list[float], min_step: list[float]):
        self.radius = list(step)
        self.max_step = max_step
        self.min_step = min_step

    def update(self, improved: bool) -> None:
        factor = 1.2 if improved else 0.65
        self.radius = [min(radius * factor, cap) for radius, cap in zip(self.radius, self.max_step)]

    @property
    def exhausted(self) -> bool:
        return all(radius < minimum for radius, minimum in zip(self.radius, self.min_step))


def local_points(history: list[dict], radius: list[float], reach: float = 3.0) -> list[dict]:
    """Measurements near the current best, for a local quadratic model.

    Focus-type responses are sharply peaked with flat tails, and a single quadratic fitted
    to far-away points misplaces the peak. Keep points within ``reach`` trust radii of the
    best one, but never fewer than a full quadratic in this many dimensions needs.
    """
    best = max(history, key=lambda item: item["score"])["point"]
    dimensions = len(best)
    needed = (dimensions + 1) * (dimensions + 2) // 2

    def distance(item: dict) -> float:
        return math.sqrt(sum(((a - b) / r) ** 2 for a, b, r in zip(item["point"], best, radius)))

    ordered = sorted(history, key=distance)
    near = [item for item in ordered if distance(item) <= reach]
    return near if len(near) >= needed else ordered[:needed]


async def optimize(
    session: Session,
    apply,
    start: list[float],
    step: list[float],
    bounds: list[tuple[float, float]],
    metric: str,
    detector: str,
    max_iterations: int,
) -> dict:
    """Shared measure / model / propose loop. ``apply(point)`` sets the instrument to ``point``."""
    history: list[dict] = []

    async def measure_at(point: list[float], trial: str) -> float:
        await apply(point)
        score, key = await session.measure(metric, detector)
        history.append({"point": list(point), "score": score, "data_key": key, "trial": trial})
        return score

    await measure_at(start, "baseline")
    for axis in range(len(start)):
        for sign in (-1, 1):
            point = list(start)
            point[axis] += sign * step[axis]
            if bounds[axis][0] <= point[axis] <= bounds[axis][1]:
                await measure_at(point, f"initial {'-+'[sign > 0]}{axis}")

    region = TrustRegion(step, [2 * value for value in step], [value / 16 for value in step])
    for iteration in range(max_iterations):
        if region.exhausted:
            break
        previous_best = max(item["score"] for item in history)
        local = local_points(history, region.radius)
        proposal = await session.analysis_call(
            "propose_next_setting",
            {"points": [item["point"] for item in local], "scores": [item["score"] for item in local],
             "trust_radius": region.radius, "bounds": [list(pair) for pair in bounds], "maximize": True},
        )
        if proposal["next_point"] is None:
            break
        score = await measure_at(list(proposal["next_point"]), f"model {iteration}")
        region.update(score > previous_best + 1e-3 * abs(previous_best))

    best = max(history, key=lambda item: item["score"])
    baseline = history[0]["score"]
    improved = best["score"] > baseline + 1e-3 * abs(baseline)
    bracketed = all(
        any(item["point"][axis] < best["point"][axis] and item["score"] < best["score"] for item in history)
        and any(item["point"][axis] > best["point"][axis] and item["score"] < best["score"] for item in history)
        for axis in range(len(start))
    )
    if not improved:
        outcome = "no improvement over the starting setting; the instrument is left as it started"
    elif not bracketed:
        outcome = ("improved, but the best setting is at the edge of what was explored; the optimum may lie "
                   "further out, so consider running again from here")
    else:
        outcome = "optimum found inside the explored range"
    return {"best_point": best["point"] if improved else list(start), "best_score": best["score"],
            "best_data_key": best["data_key"], "baseline_score": baseline, "improved": improved,
            "peak_bracketed": bracketed, "outcome": outcome, "history": history}


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ProcedureError(f"metric must be one of {list(METRICS)}")


def _within(name: str, value: float, limit: float) -> None:
    if not 0 < value <= limit:
        raise ProcedureError(f"{name} must be > 0 and at most {limit:g} (configured limit); got {value:g}")


class ProceduresServer:
    """``instrument`` / ``analysis`` override the configured URLs with any target a fastmcp
    Client accepts, such as an in-process FastMCP server."""

    def __init__(self, config: ProceduresConfig, instrument: Any = None, analysis: Any = None):
        self.config = config
        self.instrument_target = instrument or config.instrument_url
        self.analysis_target = analysis or _transport(config.analysis_url, config.analysis_transport)
        self.mcp = FastMCP(config.name)
        self._register()

    async def _open(self, stack: AsyncExitStack) -> Session:
        instrument = await stack.enter_async_context(Client(self.instrument_target))
        analysis = await stack.enter_async_context(Client(self.analysis_target))
        session = Session(self.config, instrument, analysis)
        await session.discover()
        return session

    def _register(self) -> None:
        limits = self.config.limits

        @self.mcp.tool()
        async def focus_stage_z(metric: str = "contrast", step_m: float = 50e-9, max_offset_m: float = 1e-6,
                                max_iterations: int = 8, detector: str = "haadf") -> str:
            """Focus by moving the stage height (z) to maximise an image metric, then leave it at the best z.

            Starts at the current z, never moves further than max_offset_m from it, and moves the stage
            back to the starting z if anything fails. metric: contrast | entropy | high_frequency_power |
            edge_sharpness. Returns the best z, the score gain and every measurement's data key.
            """
            _check_metric(metric)
            _within("step_m", step_m, limits.stage_z_max_step_m)
            _within("max_offset_m", max_offset_m, limits.stage_z_max_offset_m)
            if step_m > max_offset_m:
                raise ProcedureError("step_m must not exceed max_offset_m")
            async with AsyncExitStack() as stack:
                session = await self._open(stack)
                start = [float(value) for value in await session.instrument_call("get_stage")]

                async def apply(point: list[float]) -> None:
                    target = list(start)
                    target[2] = point[0]
                    await session.instrument_call("move_stage", target)

                completed = False
                try:
                    result = await optimize(session, apply, [start[2]], [step_m],
                                            [(start[2] - max_offset_m, start[2] + max_offset_m)],
                                            metric, detector, max_iterations)
                    await apply(result["best_point"])
                    completed = True
                finally:
                    if not completed:
                        await session.instrument_call("move_stage", start)
                final = await session.instrument_call("get_stage")
                return json.dumps({"procedure": "focus_stage_z", "metric": metric, "start_z_m": start[2],
                                   "best_z_m": result["best_point"][0], "final_stage": final,
                                   "baseline_score": result["baseline_score"], "best_score": result["best_score"],
                                   "improved": result["improved"], "peak_bracketed": result["peak_bracketed"],
                                   "outcome": result["outcome"],
                                   "acquisitions": session.acquisitions, "history": result["history"]})

        @self.mcp.tool()
        async def sweep_defocus(half_range_m: float = 20e-9, steps: int = 9, metric: str = "contrast",
                                apply_best: bool = True, detector: str = "haadf") -> str:
            """Acquire images across a defocus range centred on the current defocus and find the best focus.

            A parabola is fitted to the (up to) five measurements around the best one. With apply_best,
            sets the fitted best defocus when that fit has a maximum inside those points, otherwise the
            best measured one; without it, restores the starting defocus. The starting defocus is
            restored if anything fails.
            """
            _check_metric(metric)
            _within("half_range_m", half_range_m, limits.defocus_max_range_m)
            if not 3 <= steps <= limits.max_acquisitions:
                raise ProcedureError(f"steps must be between 3 and {limits.max_acquisitions}")
            async with AsyncExitStack() as stack:
                session = await self._open(stack)
                start = float(await session.instrument_call("get_defocus"))
                values = [start - half_range_m + 2 * half_range_m * index / (steps - 1) for index in range(steps)]
                history = []
                completed = False
                try:
                    for value in values:
                        await session.instrument_call("set_defocus", value)
                        score, key = await session.measure(metric, detector)
                        history.append({"defocus_m": value, "score": score, "data_key": key})
                    best_index = max(range(steps), key=lambda index: history[index]["score"])
                    low = min(max(best_index - 2, 0), steps - 5) if steps >= 5 else 0
                    neighbourhood = history[low:low + 5]
                    fit = await session.analysis_call("fit_parabola_vertex", {
                        "x": [(item["defocus_m"] - start) * 1e9 for item in neighbourhood],
                        "y": [item["score"] for item in neighbourhood]})
                    vertex_m = start + fit["vertex_x"] * 1e-9
                    fit_usable = (fit["curvature"] < 0
                                  and neighbourhood[0]["defocus_m"] <= vertex_m <= neighbourhood[-1]["defocus_m"])
                    best_measured = history[best_index]["defocus_m"]
                    chosen = vertex_m if fit_usable else best_measured
                    await session.instrument_call("set_defocus", chosen if apply_best else start)
                    completed = True
                finally:
                    if not completed:
                        await session.instrument_call("set_defocus", start)
                return json.dumps({"procedure": "sweep_defocus", "metric": metric, "start_defocus_m": start,
                                   "fit_vertex_m": vertex_m, "fit_has_maximum_in_range": fit_usable,
                                   "best_measured_defocus_m": best_measured, "chosen_defocus_m": chosen,
                                   "applied": apply_best,
                                   "final_defocus_m": await session.instrument_call("get_defocus"),
                                   "history": history})

        @self.mcp.tool()
        async def tune_aberration(name: str, step_m: float = 1e-9, max_offset_m: float = 10e-9,
                                  metric: str = "contrast", max_iterations: int = 8, detector: str = "haadf") -> str:
            """Tune one corrector aberration (e.g. C1, A1, A2, B2) to maximise an image metric.

            Moves the aberration by increments relative to its current value, never more than
            max_offset_m in total (per component), and undoes the whole change if anything fails.
            Round aberrations (C1, C3) have one coefficient, the others an x and a y component.
            """
            _check_metric(metric)
            cap = limits.aberration_max_offset_m.get(name)
            if cap is None:
                raise ProcedureError(f"no limit is configured for {name!r}; allowed: {sorted(limits.aberration_max_offset_m)}")
            _within("max_offset_m", max_offset_m, cap)
            _within("step_m", step_m, min(limits.aberration_max_step_m, max_offset_m))
            dimensions = components_of(name)
            async with AsyncExitStack() as stack:
                session = await self._open(stack)
                applied = [0.0] * dimensions

                async def apply(point: list[float]) -> None:
                    increment = [target - current for target, current in zip(point, applied)]
                    if any(increment):
                        await session.instrument_call(
                            "correct_aberration", " ".join([name, *(f"{value:.12g}" for value in increment)]))
                        applied[:] = point

                completed = False
                try:
                    result = await optimize(session, apply, [0.0] * dimensions, [step_m] * dimensions,
                                            [(-max_offset_m, max_offset_m)] * dimensions,
                                            metric, detector, max_iterations)
                    await apply(result["best_point"])
                    completed = True
                finally:
                    if not completed:
                        await apply([0.0] * dimensions)
                return json.dumps({"procedure": "tune_aberration", "aberration": name, "metric": metric,
                                   "applied_correction_m": result["best_point"],
                                   "baseline_score": result["baseline_score"], "best_score": result["best_score"],
                                   "improved": result["improved"], "peak_bracketed": result["peak_bracketed"],
                                   "outcome": result["outcome"],
                                   "acquisitions": session.acquisitions, "history": result["history"],
                                   "note": "correction values follow the corrector's correct_aberration sign "
                                           "convention, relative to the starting state"})

        @self.mcp.tool()
        def procedure_limits() -> str:
            """The hard limits every procedure enforces, from the server's configuration."""
            return json.dumps(self.config.limits.__dict__)

    def run(self) -> None:
        self.mcp.run(transport="streamable-http", host=self.config.http_host, port=self.config.http_port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yaml", type=Path, required=True, help="procedures config (see configs/procedures.yaml)")
    args = parser.parse_args(argv)
    ProceduresServer(ProceduresConfig.from_yaml(args.yaml)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
