"""Map a sample from one sentence: plan -> reserve -> start -> wait -> points -> score -> report.

The language model only turns the request into a map plan. Every instrument
and analysis call is made by graph code through the MCP endpoints, so a
small local model cannot drop a step, invent a tool or lose count of tiles.
The plan is checked against map_plan.schema.json and completed with
defaults before anything moves.

Inputs (state): ``request`` (text), optionally ``plan`` (skip the model) and
``score`` (rank regions afterwards). Output: ``report`` plus keys and points.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TypedDict

import jsonschema
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from asyncroscopy import settings
from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.goals import DOSE
from asyncroscopy.agents.mcp_tools import McpServer, result_text
from asyncroscopy.contracts import load, validate
from asyncroscopy.core.names import devices_of_kind
from asyncroscopy.core.map_analysis import AnalysisFailed, score_map, stitch_map

ABORT_WAIT_S = 60.0  # wait for an aborted map to stop before releasing
MAX_POLL_MISSES = 5  # consecutive failed polls before giving up
DEFAULT_FOV_M = 200e-9
load("map_plan")  # file I/O in async nodes blocks the event loop

log = logging.getLogger(__name__)


class MapState(TypedDict, total=False):
    request: str
    plan: dict
    score: bool
    context: dict
    reservation_id: str
    estimate: dict
    job_id: str
    released: bool  # already released: error paths must not release or abort again
    run_key: str
    map_key: str
    eds_points: int
    pois: list
    report: str
    error: str


PLAN_PROMPT = """You turn a microscopist's request into a JSON map plan for {instrument}. Reply with JSON only.

Fields (lengths in metres):
- "box_m": [x_min, x_max, y_min, y_max], the stage region to cover.
- "fov_m": field of view of each tile.
{per_tile}- "score": true if the request asks to find interesting regions / features / where to look next.
Optional: "overlap" (0-0.5, default 0.15), "imsize" (pixels, default 512), "dwell_s" (default 1e-6).

Instrument now: {context}
Use the sample bounds above only if the request asks for the whole sample (or is just "map the sample"); otherwise give the area the request states.
If it gives a size like "a 500 nm square around the centre", centre it on the current stage position.
Convert units: 1 mm = 1e-3 m, 1 um = 1e-6 m, 1 nm = 1e-9 m."""
EDS_LINE = '- "per_tile": [] for images only, ["eds_point"] for one EDS spectrum at each tile, ["eds_map"] only if a full spectrum image of every tile is explicitly asked for.\n'


def plan_prompt(context: dict) -> str:
    """The planner's system prompt, naming what this endpoint's instrument offers: EDS only when it can take spectra."""
    microscope = context.get("microscope")
    instrument = f"the {microscope}, a scanning electron microscope" if microscope else "a scanning electron microscope"
    return PLAN_PROMPT.format(instrument=instrument, per_tile=EDS_LINE if context.get("eds", True) else "", context=json.dumps(context))


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    return json.loads(match.group(0))


def _to_metres(value: float, metres_below: float) -> float:
    """Repair a length a model wrote in the wrong unit: above 1 is nm, from ``metres_below`` to 1 is um."""
    if abs(value) > 1.0:
        return value * 1e-9
    if abs(value) > metres_below:
        return value * 1e-6
    return value


def _plausible_limits(context: dict) -> tuple[float, float]:
    """(largest field of view, largest stage coordinate) plausible in metres, scaled from the current field and stage."""
    fov = context.get("fov_m")
    stage = [abs(float(v)) for v in (context.get("stage_position_m") or [])[:2] if isinstance(v, (int, float))]
    bounds = [abs(float(v)) for v in (context.get("sample_bounds_m") or []) if isinstance(v, (int, float))]
    reach = max(stage + bounds + [0.0])
    fov_max = max(1e-3, 1e3 * float(fov) if fov else 0.0, 0.1 * reach)
    coord_max = max(1e-2, 1e3 * float(fov) if fov else 0.0, 10 * reach)
    return min(fov_max, 1.0), min(coord_max, 1.0)


def _length(metres: float) -> str:
    """A length for a report: nm, um or mm, whichever reads best."""
    for unit, scale in (("mm", 1e-3), ("um", 1e-6)):
        if abs(metres) >= scale:
            return f"{metres / scale:.3g} {unit}"
    return f"{metres * 1e9:.0f} nm"

WHOLE_SAMPLE = re.compile(r"\b(?:whole|entire|full)\s+(?:sample|specimen|grid)\b|\ball of the sample\b|^\s*map\s+the\s+sample\s*[.!]?\s*$", re.I)


class MappingGraph:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env(caller=settings.get().caller or "mapping-graph")
        self.instrument: McpServer = self.config.instrument()
        self.analysis: McpServer | None = self.config.analysis()
        self._tool_names: set[str] | None = None
        self._mapper_prefix: str | None = None

    async def _mapper(self) -> str:
        """The Mapper's tool prefix: Mapper, or Mapper_<key> when an endpoint exports several."""
        if self._mapper_prefix is None:
            configured = settings.get().mapper
            candidates = devices_of_kind(await self.instrument.call_json("endpoint_info"), "map")
            if configured:
                if configured not in candidates:
                    raise RuntimeError(f"ASYNCROSCOPY_MAPPER={configured!r} is not one of {candidates}")
                self._mapper_prefix = configured
            elif len(candidates) == 1:
                self._mapper_prefix = candidates[0]
            elif not candidates:
                raise RuntimeError(f"no Mapper job device on {self.config.mcp_url}")
            else:
                raise RuntimeError(f"several Mappers {candidates}; set ASYNCROSCOPY_MAPPER to the one to use")
        return self._mapper_prefix

    async def _microscope(self) -> str:
        """The exported microscope's class name, e.g. DigitalTwin or AutoScriptMicroscope."""
        if self._tool_names is None:
            self._tool_names = {t.name for t in await self.instrument.list_tools()}
        configured = settings.get().microscope
        if configured:
            if f"{configured}_move_stage" not in self._tool_names:
                raise RuntimeError(f"ASYNCROSCOPY_MICROSCOPE={configured!r} has no {configured}_move_stage tool on {self.config.mcp_url}")
            return configured
        candidates = sorted(
            name[: -len("_move_stage")]
            for name in self._tool_names
            if name.endswith("_move_stage") and f"{name[: -len('_move_stage')]}_acquire_scanned_image" in self._tool_names
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise RuntimeError("no device with both move_stage and acquire_scanned_image on the MCP endpoint")
        raise RuntimeError(f"several microscopes {candidates}; set ASYNCROSCOPY_MICROSCOPE to the one to map with")

    async def gather_context(self, state: MapState) -> MapState:
        microscope = await self._microscope()
        await self._mapper()  # zero or several Mappers end the run before the model is asked
        params = await self.instrument.call_json(f"{microscope}_get_parameters")
        stage = params.get("stage_position")
        if stage is None:
            stage = json.loads(await self.instrument.call_text(f"{microscope}_get_stage"))
        fov = params.get("fov_m")
        if fov is None and f"{microscope}_get_fov" in (self._tool_names or set()):
            fov = float(await self.instrument.call_text(f"{microscope}_get_fov"))
        eds = all(f"{microscope}_{c}" in (self._tool_names or set()) for c in ("acquire_spectrum", "place_beam"))
        context = {"stage_position_m": stage, "fov_m": fov, "microscope": microscope, "eds": eds}
        bounds = params.get("world_bounds_angstrom")
        if bounds:
            context["sample_bounds_m"] = [bounds["x_min"] * 1e-10, bounds["x_max"] * 1e-10, bounds["y_min"] * 1e-10, bounds["y_max"] * 1e-10]
        return {"context": context}

    async def make_plan(self, state: MapState) -> MapState:
        context = state.get("context", {})
        from_model = not state.get("plan")
        if not from_model:
            plan = dict(state["plan"])
        else:
            model = self.config.chat_model()
            reply = await model.ainvoke([SystemMessage(plan_prompt(context)), HumanMessage(state.get("request", "map the sample"))])
            try:
                plan = _parse_json(reply.content if isinstance(reply.content, str) else str(reply.content))
            except (ValueError, json.JSONDecodeError):
                plan = {}
        score = bool(plan.pop("score", state.get("score", False)))
        try:
            plan = self._complete_plan(plan, context, state.get("request", ""), from_model=from_model)
        except ValueError as exc:
            return {"error": f"could not make a map plan: {exc}"}
        try:
            validate(plan, "map_plan")
        except jsonschema.ValidationError as exc:
            return {"error": f"could not make a valid map plan: {exc.message}", "plan": plan}
        return {"plan": plan, "score": score or bool(re.search(r"interesting|feature|score|where to look|regions? of interest", state.get("request", ""), re.I))}

    @staticmethod
    def _complete_plan(plan: dict, context: dict, request: str, from_model: bool = True) -> dict:
        """Fill gaps and repair slips (units, swapped corners) in a model's plan; a plan passed in keeps its size and per_tile."""
        allowed = {"box_m", "size_m", "centre_m", "fov_m", "overlap", "imsize", "dwell_s", "detectors", "per_tile", "eds_exposure_s", "eds_map_points", "dose_budget_e_per_a2", "order", "z_m", "label"}
        plan = {k: v for k, v in plan.items() if k in allowed}
        fov_max_m, coord_max_m = _plausible_limits(context)
        fov = float(plan.get("fov_m") or context.get("fov_m") or DEFAULT_FOV_M)
        fov = _to_metres(fov, metres_below=fov_max_m)
        plan["fov_m"] = fov
        size_m, centre_m = plan.pop("size_m", None), plan.pop("centre_m", None)
        stage_xy = (context.get("stage_position_m") or [0.0, 0.0])[:2]
        box = plan.get("box_m")
        if isinstance(size_m, list) and len(size_m) == 2 and not isinstance(box, list):
            width, height = (_to_metres(float(v), metres_below=coord_max_m) for v in size_m)
            x, y = (_to_metres(float(v), metres_below=coord_max_m) for v in (centre_m or stage_xy)[:2])
            box = [x - width / 2, x + width / 2, y - height / 2, y + height / 2]
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            box = [_to_metres(float(v), metres_below=coord_max_m) for v in box]
            x0, x1, y0, y1 = box
            box = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
        else:
            box = None
        # The stated size is always the size mapped, because small models fall back to the whole sample.
        size = MappingGraph._requested_size(request) if from_model else None
        if size is not None:
            width, height = size
            if box is not None and abs((box[1] - box[0]) - width) <= 0.3 * width and abs((box[3] - box[2]) - height) <= 0.3 * height:
                x, y = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
            else:
                x, y = stage_xy
            box = [x - width / 2, x + width / 2, y - height / 2, y + height / 2]
        if box is None:
            # The whole sample is hours of beam time, so map it only when the request asks.
            if not WHOLE_SAMPLE.search(request):
                raise ValueError("the plan gives no area (box_m or size_m) and the request does not ask for the whole sample")
            box = context.get("sample_bounds_m")
            if box is None:
                raise ValueError("the request asks for the whole sample, but the instrument reports no sample bounds; give box_m or size_m")
            box = list(box)
        if box[1] - box[0] <= fov and box[3] - box[2] <= fov:
            raise ValueError(f"the area fits in one {_length(fov)} field of view: that is one image, not a map; give a larger area or a smaller fov_m")
        for lo, hi in ((0, 1), (2, 3)):
            if box[hi] - box[lo] < fov:
                mid = (box[lo] + box[hi]) / 2
                box[lo], box[hi] = mid - fov / 2, mid + fov / 2
        plan["box_m"] = box
        # A dose limit the user states always goes in the plan, so the Mapper refuses a plan above it.
        stated = DOSE.search(request) if from_model else None
        if stated:
            limit = float(stated.group(1))
            given = plan.get("dose_budget_e_per_a2")
            plan["dose_budget_e_per_a2"] = min(limit, float(given)) if isinstance(given, (int, float)) and given > 0 else limit
        per_tile = [p for p in plan.get("per_tile") or [] if p in ("eds_point", "eds_map")]
        if not from_model:
            return MappingGraph._set_per_tile(plan, per_tile)
        asks_for_eds = re.search(r"\beds\b|\bedx\b|spectr|composition|element", request, re.I)
        if not per_tile and asks_for_eds and context.get("eds", True):
            per_tile = ["eds_point"]
        if not asks_for_eds or not context.get("eds", True):
            per_tile = []  # EDS adds dose and time: only when asked and available
        if "eds_map" in per_tile and not re.search(r"spectrum image|eds map|spectral map|eds_map", request, re.I):
            per_tile = ["eds_point"]  # a full spectrum image must be asked for by name
        return MappingGraph._set_per_tile(plan, per_tile)

    @staticmethod
    def _set_per_tile(plan: dict, per_tile: list) -> dict:
        """EDS per tile only when asked for: a profile without EDS refuses even an empty per_tile."""
        if per_tile:
            plan["per_tile"] = per_tile[:1]
        else:
            for key in ("per_tile", "eds_exposure_s", "eds_map_points"):
                plan.pop(key, None)
        return plan

    @staticmethod
    def _requested_size(request: str) -> tuple[float, float] | None:
        """(width, height) in metres of an area stated in the request, or None."""
        units = {"nm": 1e-9, "um": 1e-6, "µm": 1e-6, "μm": 1e-6, "micron": 1e-6, "microns": 1e-6, "mm": 1e-3}
        number = r"(\d+(?:\.\d+)?)"
        unit = r"(nm|um|µm|μm|microns?|mm)"
        area = r"(?:square|area|region|box|field|patch|map)"
        two = re.search(rf"{number}\s*{unit}?\s*(?:x|by|×)\s*{number}\s*{unit}\b[^.]*?{area}", request, re.I)
        if two:
            u2 = units[two.group(4).lower()]
            u1 = units[two.group(2).lower()] if two.group(2) else u2
            return float(two.group(1)) * u1, float(two.group(3)) * u2
        one = re.search(rf"{number}\s*{unit}\b(?:\s*-?\s*(?:wide|side|across))?\s*{area}", request, re.I)
        if one:
            side = float(one.group(1)) * units[one.group(2).lower()]
            return side, side
        return None

    async def reserve(self, state: MapState) -> MapState:
        reply = await self.instrument.call_json(f"{await self._mapper()}_reserve", {"request": json.dumps({"plan": state["plan"], "caller": self.config.caller})})
        if "error" in reply:
            return {"error": f"Mapper refused the plan: {reply['error']['message']}"}
        return {"reservation_id": reply["reservation_id"], "estimate": reply["estimate"]}

    async def start(self, state: MapState) -> MapState:
        arguments = {"request": json.dumps({"reservation_id": state["reservation_id"]})}
        result = await self.instrument.call(f"{await self._mapper()}_start", arguments)
        text = result_text(result)
        if result.is_error:
            return {"error": f"the map start failed: {text[:300]}"}
        reply = json.loads(text)
        if "error" in reply:
            return {"error": f"Mapper could not start: {reply['error']['message']}"}
        return {"job_id": reply["job_id"]}

    async def wait(self, state: MapState) -> MapState:
        deadline = time.time() + 3.0 * float(state.get("estimate", {}).get("duration_s", 600)) + 300.0
        misses = 0
        while True:
            try:
                status = await self.instrument.call_json(f"{await self._mapper()}_job_status", {"request": json.dumps({"job_id": state["job_id"]})})
            except Exception as exc:  # one dropped poll must not end the watch
                status = {"error": {"message": str(exc)}}
            if "error" in status:
                misses += 1
                if misses >= MAX_POLL_MISSES:
                    return {"error": f"lost track of the map job: {status['error'].get('message', '')}"}
            else:
                misses = 0
            if status.get("state") in {"done", "failed", "aborted", "released"}:
                break
            if time.time() > deadline:
                return {"error": "the map run took more than three times its estimate"}
            await asyncio.sleep(settings.get().poll_s)
        results = await self.instrument.call_json(f"{await self._mapper()}_results", {"request": json.dumps({"job_id": state["job_id"]})})
        await self.instrument.call(f"{await self._mapper()}_release", {"request": json.dumps({"reservation_id": state["reservation_id"]})})
        if status["state"] != "done":
            return {"error": f"map run {status['state']}: {(status.get('last_error') or {}).get('message', '')}", "released": True}
        keys = results.get("result_keys", {})
        summary = results.get("summary", {})
        return {"run_key": keys.get("run") or keys.get("run_file"), "map_key": keys.get("map") or summary.get("map"), "released": True}

    async def points(self, state: MapState) -> MapState:
        map_key = state.get("map_key")
        if not map_key:
            try:
                map_key = (await stitch_map(self.analysis.call_json, state["run_key"]))["map"]
            except AnalysisFailed as exc:
                return {"error": f"stitching {state['run_key']} failed: {exc}"}
        eds = await self.analysis.call_json("list_pois", {"map": map_key, "kind": "eds_point", "limit": 0})
        candidates = await self.analysis.call_json("list_pois", {"map": map_key, "kind": "candidate"})
        if "total" not in eds or "pois" not in candidates:
            return {"map_key": map_key, "error": f"the map {map_key} was stitched, but its points of interest could not be read"}
        return {"map_key": map_key, "eds_points": int(eds["total"]), "pois": [p for p in candidates["pois"] if isinstance(p, dict)]}

    async def score(self, state: MapState) -> MapState:
        try:
            scored = await score_map(self.analysis.call_json, state["map_key"], 3)
        except AnalysisFailed as exc:
            return {"error": f"the map {state['map_key']} was stitched, but scoring failed: {exc}"}
        return {"pois": list(state.get("pois", [])) + scored["pois"]}

    async def report(self, state: MapState) -> MapState:
        if state.get("error"):
            # Free the instrument so the next reserve does not find it busy.
            await self._release(state)
            return {"report": f"Map not completed: {state['error']}"}
        plan = state["plan"]
        box = plan["box_m"]
        lines = [
            f"Mapped {_length(box[1] - box[0])} x {_length(box[3] - box[2])} with {state.get('estimate', {}).get('tiles', state.get('estimate', {}).get('steps', '?'))} tiles "
            f"of {_length(plan['fov_m'])}" + (" and an EDS spectrum at each tile" if "eds_point" in plan.get("per_tile", []) else "") + ".",
            f"Run data: {state.get('run_key')}.",
        ]
        if state.get("map_key"):
            lines.append(f"Stitched map: {state['map_key']}.")
        candidates = [p for p in state.get("pois", []) if p.get("kind") == "candidate"]
        for p in candidates:
            lines.append(f"Interesting region at ({p['x_m'] * 1e9:.0f} nm, {p['y_m'] * 1e9:.0f} nm), score {p['score']:.2f}: {p['reason']}")
        eds = state.get("eds_points", 0)
        if eds:
            lines.append(f"{eds} EDS spectra are marked on the map.")
        return {"report": "\n".join(lines)}

    async def _release(self, state: MapState) -> None:
        """Abort the job if it may still run, then release it; errors are ignored, as the endpoint may be why we fail."""
        if not state.get("reservation_id") or state.get("released"):
            return
        try:
            if state.get("job_id"):
                await self.instrument.call(f"{await self._mapper()}_abort", {"request": json.dumps({"job_id": state["job_id"]})})
                deadline = time.time() + ABORT_WAIT_S
                while time.time() < deadline:
                    status = await self.instrument.call_json(f"{await self._mapper()}_job_status", {"request": json.dumps({"job_id": state["job_id"]})})
                    if status.get("state") not in {"running", "aborting"}:
                        break
                    await asyncio.sleep(settings.get().poll_s)
            await self.instrument.call(f"{await self._mapper()}_release", {"request": json.dumps({"reservation_id": state["reservation_id"]})})
        except Exception:
            log.warning("could not stop or release reservation %s", state["reservation_id"], exc_info=True)

    @staticmethod
    def _guarded(node):
        """Turn an exception in a node into an error in the state, so the run still reaches report."""
        async def run(state: MapState) -> MapState:
            try:
                return await node(state)
            except Exception as exc:
                return {"error": f"{node.__name__} failed: {type(exc).__name__}: {exc}"}

        run.__name__ = node.__name__
        return run

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(MapState)
        for name in ("gather_context", "make_plan", "reserve", "start", "wait", "points", "score", "report"):
            graph.add_node(name, self._guarded(getattr(self, name)))
        ok = lambda nxt: (lambda s: "report" if s.get("error") else nxt)
        graph.add_edge(START, "gather_context")
        graph.add_conditional_edges("gather_context", ok("make_plan"), ["make_plan", "report"])
        graph.add_conditional_edges("make_plan", ok("reserve"), ["reserve", "report"])
        graph.add_conditional_edges("reserve", ok("start"), ["start", "report"])
        graph.add_conditional_edges("start", ok("wait"), ["wait", "report"])
        graph.add_conditional_edges("wait", ok("points" if self.analysis else "report"), ["points", "report"])
        graph.add_conditional_edges("points", lambda s: "report" if s.get("error") or not s.get("score") else "score", ["score", "report"])
        graph.add_edge("score", "report")
        graph.add_edge("report", END)
        return graph.compile()


def build_graph(config: AgentConfig | None = None) -> CompiledStateGraph:
    return MappingGraph(config).build()


# Used by langgraph.json and LangGraph Studio.
graph = build_graph()
