"""Map a sample from one sentence: plan -> reserve -> start (approval) -> wait -> stitch -> score -> report.

The language model only turns the request into a map plan. Every instrument
and analysis call is made by graph code through the MCP endpoints, so a
small local model cannot drop a step, invent a tool or lose count of tiles.
The plan is checked against map_plan.schema.json and completed with
defaults before anything moves.

Approval: in the instrument's ``approve`` mode, ``Mapper_start`` is held. This
graph then waits (polling ``approval_status``) while the operator decides in
SciAgentGUI; a denial or expiry ends the run with that reason.

Inputs (state): ``request`` (text), optionally ``plan`` (skip the model) and
``score`` (rank regions afterwards). Output: ``report`` plus keys and points.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import TypedDict

import jsonschema
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.mcp_tools import McpServer, result_text
from asyncroscopy.contracts import load, validate

POLL_S = float(os.environ.get("ASYNCROSCOPY_POLL_S", "1.0"))
APPROVAL_WAIT_S = float(os.environ.get("ASYNCROSCOPY_APPROVAL_WAIT_S", "1800"))
ANALYSIS_WAIT_S = float(os.environ.get("ASYNCROSCOPY_ANALYSIS_WAIT_S", "3600"))
ABORT_WAIT_S = 60.0  # how long an error path waits for an aborted map to stop before releasing
MAX_POLL_MISSES = 5  # consecutive failed status polls before the graph gives up on a job
DEFAULT_FOV_M = 200e-9
load("map_plan")  # read the schema now: file I/O inside async nodes blocks the event loop


class MapState(TypedDict, total=False):
    request: str
    plan: dict
    score: bool
    context: dict
    reservation_id: str
    estimate: dict
    job_id: str
    approval_request_id: str
    released: bool  # the reservation is already released; error paths must not release (or abort) it again
    run_key: str
    map_key: str
    eds_points: int
    pois: list
    report: str
    error: str


PLAN_PROMPT = """You turn a microscopist's request into a JSON map plan for a STEM. Reply with JSON only.

Fields (lengths in metres):
- "box_m": [x_min, x_max, y_min, y_max], the stage region to cover.
- "fov_m": field of view of each tile.
- "per_tile": [] for images only, ["eds_point"] for one EDS spectrum at each tile, ["eds_map"] only if a full spectrum image of every tile is explicitly asked for.
- "score": true if the request asks to find interesting regions / features / where to look next.
Optional: "overlap" (0-0.5, default 0.15), "imsize" (pixels, default 512), "dwell_s" (default 1e-6).

Instrument now: {context}
Use the sample bounds above only if the request asks for the whole sample (or is just "map the sample"); otherwise give the area the request states.
If it gives a size like "a 500 nm square around the centre", centre it on the current stage position.
Convert units: 1 um = 1e-6 m, 1 nm = 1e-9 m."""


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    return json.loads(match.group(0))



def _to_metres(value: float, metres_below: float) -> float:
    """Repair a length a model wrote in the wrong unit.

    Above 1 it can only be nanometres; between ``metres_below`` and 1 it is
    micrometres (0.2 meant 0.2 um, not 20 cm); below that it is already metres.
    """
    if abs(value) > 1.0:
        return value * 1e-9
    if abs(value) > metres_below:
        return value * 1e-6
    return value

# Only an explicit whole-area request: "a patch near the edge of the sample" is not one.
WHOLE_SAMPLE = re.compile(r"\b(?:whole|entire|full)\s+(?:sample|specimen|grid)\b|\ball of the sample\b|^\s*map\s+the\s+sample\s*[.!]?\s*$", re.I)


class MappingGraph:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env(caller=os.environ.get("ASYNCROSCOPY_CALLER", "mapping-graph"))
        self.instrument: McpServer = self.config.instrument()
        self.analysis: McpServer | None = self.config.analysis()
        self._tool_names: set[str] | None = None
        self._mapper_prefix: str | None = None

    async def _mapper(self) -> str:
        """The Mapper's tool prefix: Mapper, or Mapper_<key> when an endpoint exports several."""
        if self._mapper_prefix is None:
            if self._tool_names is None:
                self._tool_names = {t.name for t in await self.instrument.list_tools()}
            configured = os.environ.get("ASYNCROSCOPY_MAPPER")
            candidates = sorted(n[: -len("_reserve")] for n in self._tool_names if n.startswith("Mapper") and n.endswith("_reserve"))
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
        configured = os.environ.get("ASYNCROSCOPY_MICROSCOPE")
        if configured:
            if f"{configured}_move_stage" not in self._tool_names:
                raise RuntimeError(f"ASYNCROSCOPY_MICROSCOPE={configured!r} has no {configured}_move_stage tool on {self.config.mcp_url}")
            return configured
        # A microscope here is a device that both moves a stage and acquires scanned images;
        # an SPM or indenter with move_stage alone is not a candidate.
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

    # ------------------------------------------------------------------ nodes
    async def gather_context(self, state: MapState) -> MapState:
        microscope = await self._microscope()
        await self._mapper()  # a missing or ambiguous Mapper ends the run here, before the model is asked
        # get_parameters keys differ by vendor (the twin reports the most); fall back to plain commands.
        params = await self.instrument.call_json(f"{microscope}_get_parameters")
        stage = params.get("stage_position")
        if stage is None:
            stage = json.loads(await self.instrument.call_text(f"{microscope}_get_stage"))
        fov = params.get("fov_m")
        if fov is None and f"{microscope}_get_fov" in (self._tool_names or set()):
            fov = float(await self.instrument.call_text(f"{microscope}_get_fov"))
        context = {"stage_position_m": stage, "fov_m": fov, "microscope": microscope}
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
            reply = await model.ainvoke([SystemMessage(PLAN_PROMPT.format(context=json.dumps(context))), HumanMessage(state.get("request", "map the sample"))])
            try:
                plan = _parse_json(reply.content if isinstance(reply.content, str) else str(reply.content))
            except (ValueError, json.JSONDecodeError):
                # Small models often answer in prose. The request text then decides: a stated size
                # ("a 600 nm square") or "the whole sample"; with neither, _complete_plan refuses.
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
        """Fill gaps and repair obvious slips (units, swapped corners) so a small model's plan is usable.

        The request text overrides size and EDS only for a plan the model wrote; a plan passed in
        directly is the caller's decision and keeps its per_tile and size.
        """
        allowed = {"box_m", "size_m", "centre_m", "fov_m", "overlap", "imsize", "dwell_s", "detectors", "per_tile", "eds_exposure_s", "eds_map_points", "dose_budget_e_per_a2", "order", "z_m", "label"}
        plan = {k: v for k, v in plan.items() if k in allowed}
        fov = float(plan.get("fov_m") or context.get("fov_m") or DEFAULT_FOV_M)
        fov = _to_metres(fov, metres_below=1e-3)  # a STEM field of view is far below a millimetre
        plan["fov_m"] = fov
        size_m, centre_m = plan.pop("size_m", None), plan.pop("centre_m", None)
        stage_xy = (context.get("stage_position_m") or [0.0, 0.0])[:2]
        box = plan.get("box_m")
        if isinstance(size_m, list) and len(size_m) == 2 and not isinstance(box, list):
            width, height = (_to_metres(float(v), metres_below=1e-2) for v in size_m)
            x, y = (_to_metres(float(v), metres_below=1e-2) for v in (centre_m or stage_xy)[:2])
            box = [x - width / 2, x + width / 2, y - height / 2, y + height / 2]
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            box = [_to_metres(float(v), metres_below=1e-2) for v in box]  # STEM stage travel is a few mm
            x0, x1, y0, y1 = box
            box = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
        else:
            box = None
        # A size stated in the request ("a 600 nm square", "2 x 1 um area") is read here, not left to
        # the model: small models often fall back to the whole sample. Centred on the stage unless the
        # model's box already has that size.
        size = MappingGraph._requested_size(request) if from_model else None
        if size is not None:
            width, height = size
            if box is None or not (abs((box[1] - box[0]) - width) <= 0.3 * width and abs((box[3] - box[2]) - height) <= 0.3 * height):
                x, y = stage_xy
                box = [x - width / 2, x + width / 2, y - height / 2, y + height / 2]
        if box is None:
            # The whole sample is hours of beam time: only when the request asks for it.
            if not WHOLE_SAMPLE.search(request):
                raise ValueError("the plan gives no area (box_m or size_m) and the request does not ask for the whole sample")
            box = context.get("sample_bounds_m")
            if box is None:
                raise ValueError("the request asks for the whole sample, but the instrument reports no sample bounds; give box_m or size_m")
            box = list(box)
        # An area that fits in one tile is an image, not a map; a side shorter than a tile becomes one tile (a strip).
        if box[1] - box[0] <= fov and box[3] - box[2] <= fov:
            raise ValueError(f"the area fits in one {fov * 1e9:.0f} nm field of view: that is one image, not a map; give a larger area or a smaller fov_m")
        for lo, hi in ((0, 1), (2, 3)):
            if box[hi] - box[lo] < fov:
                mid = (box[lo] + box[hi]) / 2
                box[lo], box[hi] = mid - fov / 2, mid + fov / 2
        plan["box_m"] = box
        per_tile = [p for p in plan.get("per_tile") or [] if p in ("eds_point", "eds_map")]
        if not from_model:
            plan["per_tile"] = per_tile[:1]
            return plan
        asks_for_eds = re.search(r"\beds\b|\bedx\b|spectr|composition|element", request, re.I)
        if not per_tile and asks_for_eds:
            per_tile = ["eds_point"]
        if not asks_for_eds:
            per_tile = []  # EDS adds dose and time: only when the request asks for it
        if "eds_map" in per_tile and not re.search(r"spectrum image|eds map|spectral map|eds_map", request, re.I):
            per_tile = ["eds_point"]  # a full spectrum image must be asked for by name
        plan["per_tile"] = per_tile[:1]
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
        deadline = time.time() + APPROVAL_WAIT_S
        request_id = None
        while True:
            result = await self.instrument.call(f"{await self._mapper()}_start", arguments)
            text = result_text(result)
            if not result.is_error:
                reply = json.loads(text)
                if "error" in reply:
                    return {"error": f"Mapper could not start: {reply['error']['message']}"}
                return {"job_id": reply["job_id"], "approval_request_id": request_id or ""}
            match = re.search(r"request_id=(\w+)", text)
            if not match:
                return {"error": f"the map start failed: {text[:300]}"}
            request_id = match.group(1)
            # Held in approve mode: wait for the operator, then repeat exactly the same call.
            while True:
                status = await self.instrument.call_json("approval_status", {"request_id": request_id})
                if status.get("state") == "approved":
                    break
                if status.get("state") in {"denied", "expired"}:
                    await self._release(state)
                    reason = status.get("reason") or status["state"]
                    return {"error": f"the operator did not approve the map ({reason})", "approval_request_id": request_id, "released": True}
                if time.time() > deadline:
                    await self._release(state)
                    return {"error": "no operator decision before the approval wait ran out", "approval_request_id": request_id, "released": True}
                await asyncio.sleep(POLL_S)

    async def wait(self, state: MapState) -> MapState:
        # Give the job three times its estimate (plus 5 minutes), then abort it rather than wait forever.
        deadline = time.time() + 3.0 * float(state.get("estimate", {}).get("duration_s", 600)) + 300.0
        misses = 0
        while True:
            try:
                status = await self.instrument.call_json(f"{await self._mapper()}_job_status", {"request": json.dumps({"job_id": state["job_id"]})})
            except Exception as exc:  # one dropped call in an hours-long poll must not end the watch
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
            await asyncio.sleep(POLL_S)
        results = await self.instrument.call_json(f"{await self._mapper()}_results", {"request": json.dumps({"job_id": state["job_id"]})})
        await self.instrument.call(f"{await self._mapper()}_release", {"request": json.dumps({"reservation_id": state["reservation_id"]})})
        if status["state"] != "done":
            return {"error": f"map run {status['state']}: {(status.get('last_error') or {}).get('message', '')}", "released": True}
        keys = results.get("result_keys", {})
        summary = results.get("summary", {})
        # A failed stitch in the job leaves no map key, so the stitch node tries again here.
        return {"run_key": keys.get("run") or keys.get("run_file"), "map_key": keys.get("map") or summary.get("map"), "released": True}

    async def stitch(self, state: MapState) -> MapState:
        # The Mapper job stitches its own run and links every EDS spectrum as a point of interest;
        # stitch here only when the job had no analysis endpoint to do it.
        map_key = state.get("map_key")
        if not map_key:
            job = await self.analysis.call_json("assemble_map", {"run": state["run_key"]})
            if "error" in job:
                return {"error": f"stitching was refused: {job['error']['message']}"}
            results = await self._analysis_job(job["job_id"])
            if results.get("state") != "done":
                return {"error": f"stitching {results.get('state')}"}
            map_key = results["result_keys"]["map"]
        # The report counts EDS points and lists candidates; a large map has thousands of EDS points.
        eds = await self.analysis.call_json("list_pois", {"map": map_key, "kind": "eds_point", "limit": 0})
        candidates = await self.analysis.call_json("list_pois", {"map": map_key, "kind": "candidate"})
        if "total" not in eds or "pois" not in candidates:
            return {"map_key": map_key, "error": f"the map {map_key} was stitched, but its points of interest could not be read"}
        return {"map_key": map_key, "eds_points": int(eds["total"]), "pois": [p for p in candidates["pois"] if isinstance(p, dict)]}

    async def score(self, state: MapState) -> MapState:
        job = await self.analysis.call_json("score_map", {"map": state["map_key"], "top_k": 3})
        if "error" in job:
            return {"error": f"scoring was refused: {job['error']['message']}"}
        results = await self._analysis_job(job["job_id"])
        if results.get("state") != "done":
            return {"error": f"the map {state['map_key']} was stitched, but scoring {results.get('state')}"}
        return {"pois": list(state.get("pois", [])) + results.get("summary", {}).get("added_pois", [])}

    async def report(self, state: MapState) -> MapState:
        if state.get("error"):
            # Every failed path ends here; free the instrument so the next reserve is not "busy".
            await self._release(state)
            return {"report": f"Map not completed: {state['error']}"}
        plan = state["plan"]
        box = plan["box_m"]
        lines = [
            f"Mapped {(box[1] - box[0]) * 1e6:.2f} x {(box[3] - box[2]) * 1e6:.2f} um with {state.get('estimate', {}).get('steps', '?')} tiles "
            f"of {plan['fov_m'] * 1e9:.0f} nm" + (" and an EDS spectrum at each tile" if "eds_point" in plan["per_tile"] else "") + ".",
            f"Run data: {state.get('run_key')}.",
        ]
        if state.get("approval_request_id"):
            lines.append(f"Started after operator approval (request {state['approval_request_id']}).")
        if state.get("map_key"):
            lines.append(f"Stitched map: {state['map_key']}.")
        candidates = [p for p in state.get("pois", []) if p.get("kind") == "candidate"]
        for p in candidates:
            lines.append(f"Interesting region at ({p['x_m'] * 1e9:.0f} nm, {p['y_m'] * 1e9:.0f} nm), score {p['score']:.2f}: {p['reason']}")
        eds = state.get("eds_points", 0)
        if eds:
            lines.append(f"{eds} EDS spectra are marked on the map.")
        return {"report": "\n".join(lines)}

    # ------------------------------------------------------------------ helpers
    async def _analysis_job(self, job_id: str) -> dict:
        deadline = time.time() + ANALYSIS_WAIT_S
        while True:
            status = await self.analysis.call_json("analysis_job_status", {"job_id": job_id})
            if "error" in status:  # e.g. unknown_job after the analysis server restarted
                return {"state": "lost", "error": status["error"]}
            if status.get("state") in {"done", "failed", "aborted"}:
                break
            if time.time() > deadline:
                await self.analysis.call_json("analysis_job_abort", {"job_id": job_id})
                return {"state": "timed out"}
            await asyncio.sleep(POLL_S)
        results = await self.analysis.call_json("analysis_job_results", {"job_id": job_id})
        results["state"] = status["state"]
        return results

    async def _release(self, state: MapState) -> None:
        """Stop the job if it may still be running, then release the reservation.

        Release refuses a running job, so on an error path a started map is aborted
        first (abort blanks the beam) rather than left running with nobody watching.
        Errors are ignored: the endpoint may be the reason we are failing.
        """
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
                    await asyncio.sleep(POLL_S)
            await self.instrument.call(f"{await self._mapper()}_release", {"request": json.dumps({"reservation_id": state["reservation_id"]})})
        except Exception:
            pass

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

    # ------------------------------------------------------------------ graph
    def build(self):
        graph = StateGraph(MapState)
        for name in ("gather_context", "make_plan", "reserve", "start", "wait", "stitch", "score", "report"):
            graph.add_node(name, self._guarded(getattr(self, name)))
        ok = lambda nxt: (lambda s: "report" if s.get("error") else nxt)
        graph.add_edge(START, "gather_context")
        graph.add_conditional_edges("gather_context", ok("make_plan"), ["make_plan", "report"])
        graph.add_conditional_edges("make_plan", ok("reserve"), ["reserve", "report"])
        graph.add_conditional_edges("reserve", ok("start"), ["start", "report"])
        graph.add_conditional_edges("start", ok("wait"), ["wait", "report"])
        graph.add_conditional_edges("wait", ok("stitch" if self.analysis else "report"), ["stitch", "report"])
        graph.add_conditional_edges("stitch", lambda s: "report" if s.get("error") or not s.get("score") else "score", ["score", "report"])
        graph.add_edge("score", "report")
        graph.add_edge("report", END)
        return graph.compile()


def build_graph(config: AgentConfig | None = None):
    return MappingGraph(config).build()


# For langgraph.json / LangGraph Studio.
graph = build_graph()
