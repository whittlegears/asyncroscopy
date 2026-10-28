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
from typing import Any, TypedDict

import jsonschema
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from asyncroscopy.agents.config import AgentConfig
from asyncroscopy.agents.mcp_tools import McpServer, result_text
from asyncroscopy.contracts import load, validate

POLL_S = float(os.environ.get("ASYNCROSCOPY_POLL_S", "1.0"))
APPROVAL_WAIT_S = float(os.environ.get("ASYNCROSCOPY_APPROVAL_WAIT_S", "1800"))
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
    run_key: str
    spectra: list
    map_key: str
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
If the request says "the sample", "whole sample" or "everything", use the sample bounds above.
If it gives a size like "a 500 nm square around the centre", centre it on the current stage position.
Convert units: 1 um = 1e-6 m, 1 nm = 1e-9 m."""


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    return json.loads(match.group(0))


class MappingGraph:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env(caller=os.environ.get("ASYNCROSCOPY_CALLER", "mapping-graph"))
        self.instrument: McpServer = self.config.instrument()
        self.analysis: McpServer | None = self.config.analysis()
        self._tool_names: set[str] | None = None

    async def _microscope(self) -> str:
        """The exported microscope's class name, e.g. DigitalTwin or AutoScriptMicroscope."""
        if self._tool_names is None:
            self._tool_names = {t.name for t in await self.instrument.list_tools()}
        for name in sorted(self._tool_names):
            if name.endswith("_move_stage"):
                return name[: -len("_move_stage")]
        raise RuntimeError("no microscope with move_stage on the MCP endpoint")

    # ------------------------------------------------------------------ nodes
    async def gather_context(self, state: MapState) -> MapState:
        microscope = await self._microscope()
        params = await self.instrument.call_json(f"{microscope}_get_parameters")
        context = {
            "stage_position_m": params.get("stage_position"),
            "fov_m": params.get("fov_m"),
            "microscope": microscope,
        }
        bounds = params.get("world_bounds_angstrom")
        if bounds:
            context["sample_bounds_m"] = [bounds["x_min"] * 1e-10, bounds["x_max"] * 1e-10, bounds["y_min"] * 1e-10, bounds["y_max"] * 1e-10]
        return {"context": context}

    async def make_plan(self, state: MapState) -> MapState:
        context = state.get("context", {})
        if state.get("plan"):
            plan = dict(state["plan"])
        else:
            model = self.config.chat_model()
            reply = await model.ainvoke([SystemMessage(PLAN_PROMPT.format(context=json.dumps(context))), HumanMessage(state.get("request", "map the sample"))])
            try:
                plan = _parse_json(reply.content if isinstance(reply.content, str) else str(reply.content))
            except (ValueError, json.JSONDecodeError):
                plan = {}
        score = bool(plan.pop("score", state.get("score", False)))
        plan = self._complete_plan(plan, context, state.get("request", ""))
        try:
            validate(plan, "map_plan")
        except jsonschema.ValidationError as exc:
            return {"error": f"could not make a valid map plan: {exc.message}", "plan": plan}
        return {"plan": plan, "score": score or bool(re.search(r"interesting|feature|score|where to look|regions? of interest", state.get("request", ""), re.I))}

    @staticmethod
    def _complete_plan(plan: dict, context: dict, request: str) -> dict:
        """Fill gaps and repair obvious slips (units, swapped corners) so a small model's plan is usable."""
        allowed = {"box_m", "fov_m", "overlap", "imsize", "dwell_s", "detectors", "per_tile", "eds_exposure_s", "eds_map_points", "dose_budget_e_per_a2", "order", "z_m", "label"}
        plan = {k: v for k, v in plan.items() if k in allowed}
        fov = float(plan.get("fov_m") or context.get("fov_m") or DEFAULT_FOV_M)
        if fov > 1e-3:  # the model wrote nanometres
            fov *= 1e-9
        plan["fov_m"] = fov
        box = plan.get("box_m")
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            box = context.get("sample_bounds_m")
            if box is None:
                x, y = (context.get("stage_position_m") or [0.0, 0.0])[:2]
                box = [x - 2 * fov, x + 2 * fov, y - 2 * fov, y + 2 * fov]
        box = [float(v) * (1e-9 if abs(float(v)) > 1e-3 else 1.0) for v in box]
        x0, x1, y0, y1 = box
        box = [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
        # A size stated in the request ("a 600 nm square", "2 x 1 um area") is read here, not left to
        # the model: small models often fall back to the whole sample. Centred on the stage unless the
        # model's box already has that size.
        size = MappingGraph._requested_size(request)
        if size is not None:
            width, height = size
            if not (abs((box[1] - box[0]) - width) <= 0.3 * width and abs((box[3] - box[2]) - height) <= 0.3 * height):
                x, y = (context.get("stage_position_m") or [0.0, 0.0])[:2]
                box = [x - width / 2, x + width / 2, y - height / 2, y + height / 2]
        # A box smaller than one tile becomes one tile.
        for lo, hi in ((0, 1), (2, 3)):
            if box[hi] - box[lo] < fov:
                mid = (box[lo] + box[hi]) / 2
                box[lo], box[hi] = mid - fov / 2, mid + fov / 2
        plan["box_m"] = box
        per_tile = [p for p in plan.get("per_tile") or [] if p in ("eds_point", "eds_map")]
        if not per_tile and re.search(r"\beds\b|spectr", request, re.I):
            per_tile = ["eds_point"]
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
        reply = await self.instrument.call_json("Mapper_reserve", {"request": json.dumps({"plan": state["plan"], "caller": self.config.caller})})
        if "error" in reply:
            return {"error": f"Mapper refused the plan: {reply['error']['message']}"}
        return {"reservation_id": reply["reservation_id"], "estimate": reply["estimate"]}

    async def start(self, state: MapState) -> MapState:
        arguments = {"request": json.dumps({"reservation_id": state["reservation_id"]})}
        deadline = time.time() + APPROVAL_WAIT_S
        request_id = None
        while True:
            result = await self.instrument.call("Mapper_start", arguments)
            text = result_text(result)
            if not result.is_error:
                reply = json.loads(text)
                if "error" in reply:
                    return {"error": f"Mapper could not start: {reply['error']['message']}"}
                return {"job_id": reply["job_id"], "approval_request_id": request_id or ""}
            match = re.search(r"request_id=(\w+)", text)
            if not match:
                return {"error": f"Mapper_start failed: {text[:300]}"}
            request_id = match.group(1)
            # Held in approve mode: wait for the operator, then repeat exactly the same call.
            while True:
                status = await self.instrument.call_json("approval_status", {"request_id": request_id})
                if status.get("state") == "approved":
                    break
                if status.get("state") in {"denied", "expired"}:
                    await self._release(state)
                    reason = status.get("reason") or status["state"]
                    return {"error": f"the operator did not approve the map ({reason})", "approval_request_id": request_id}
                if time.time() > deadline:
                    await self._release(state)
                    return {"error": "no operator decision before the approval wait ran out", "approval_request_id": request_id}
                await asyncio.sleep(POLL_S)

    async def wait(self, state: MapState) -> MapState:
        while True:
            status = await self.instrument.call_json("Mapper_job_status", {"request": json.dumps({"job_id": state["job_id"]})})
            if status.get("state") in {"done", "failed", "aborted"}:
                break
            await asyncio.sleep(POLL_S)
        results = await self.instrument.call_json("Mapper_results", {"request": json.dumps({"job_id": state["job_id"]})})
        await self._release(state)
        if status["state"] != "done":
            return {"error": f"map run {status['state']}: {(status.get('last_error') or {}).get('message', '')}"}
        keys = results.get("result_keys", {})
        return {"run_key": keys.get("run") or keys.get("run_file"), "spectra": results.get("summary", {}).get("spectra", [])}

    async def stitch(self, state: MapState) -> MapState:
        job = await self.analysis.call_json("assemble_map", {"run": state["run_key"]})
        if "error" in job:
            return {"error": f"stitching was refused: {job['error']['message']}"}
        results = await self._analysis_job(job["job_id"])
        if results.get("state") != "done":
            return {"error": f"stitching {results.get('state')}"}
        map_key = results["result_keys"]["map"]
        pois = []
        for spectrum in state.get("spectra", []):
            x, y = spectrum["stage_xy_m"]
            poi = await self.analysis.call_json("add_poi", {"map": map_key, "x_m": x, "y_m": y, "kind": "eds_point", "reason": f"EDS point spectrum, tile {spectrum['index']}", "links": [spectrum["key"]]})
            if "id" in poi:
                pois.append(poi)
        return {"map_key": map_key, "pois": pois}

    async def score(self, state: MapState) -> MapState:
        job = await self.analysis.call_json("score_map", {"map": state["map_key"], "top_k": 3})
        if "error" in job:
            return {"error": f"scoring was refused: {job['error']['message']}"}
        results = await self._analysis_job(job["job_id"])
        return {"pois": list(state.get("pois", [])) + results.get("summary", {}).get("added_pois", [])}

    async def report(self, state: MapState) -> MapState:
        if state.get("error"):
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
        eds = sum(1 for p in state.get("pois", []) if p.get("kind") == "eds_point")
        if eds:
            lines.append(f"{eds} EDS spectra are marked on the map.")
        return {"report": "\n".join(lines)}

    # ------------------------------------------------------------------ helpers
    async def _analysis_job(self, job_id: str) -> dict:
        while True:
            status = await self.analysis.call_json("analysis_job_status", {"job_id": job_id})
            if status.get("state") in {"done", "failed", "aborted"}:
                break
            await asyncio.sleep(POLL_S)
        results = await self.analysis.call_json("analysis_job_results", {"job_id": job_id})
        results["state"] = status["state"]
        return results

    async def _release(self, state: MapState) -> None:
        if state.get("reservation_id"):
            await self.instrument.call("Mapper_release", {"request": json.dumps({"reservation_id": state["reservation_id"]})})

    # ------------------------------------------------------------------ graph
    def build(self):
        graph = StateGraph(MapState)
        for name in ("gather_context", "make_plan", "reserve", "start", "wait", "stitch", "score", "report"):
            graph.add_node(name, getattr(self, name))
        ok = lambda nxt: (lambda s: "report" if s.get("error") else nxt)  # noqa: E731
        graph.add_edge(START, "gather_context")
        graph.add_edge("gather_context", "make_plan")
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
