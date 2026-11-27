"""Return to a saved site, verified: a LangGraph workflow over MCP tools.

The same procedure as ``skills/return-to-site/SKILL.md``, as a deterministic
graph, so a swarm (or any caller) can run it without an LLM deciding each
step::

    goto ──► survey ──match──► centre ──► verify ──match, close──► record ──► END
               │                            │  ▲
               └─no_match / ambiguous──►    │  └── correct ◄── match, still off (≤ 3 tries)
                                record      └─ ambiguous / no_match ──► record

Every instrument action is an MCP tool call through *call(tool, args)*: the
instrument's ``*_get_stage``, ``*_move_stage``, ``*_set_fov``, ``*_get_fov``,
``*_acquire_scanned_image``, and ``register_images``. Sites, the measured
``stage_to_image`` mapping and every visit (match or not) live in a JSON
notes file, the same one the skill describes, so an agent following the
skill and this workflow share their memory.

``status`` is ``match`` only when the final native-magnification registration
is a match within tolerance; ``ambiguous``, ``no_match`` and ``not_converged``
are reported as such and never retried away.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, TypedDict

Call = Callable[[str, dict], Awaitable[Any]]


# ----------------------------------------------------------------------------
# Notes file
# ----------------------------------------------------------------------------

class SiteBook:
    """JSON notes: {"stage_to_image": {...}, "mount": "...", "sites": {name: {...}}}."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            "stage_to_image": {}, "mount": "original", "sites": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)

    def site(self, name: str) -> dict:
        try:
            return self.data["sites"][name]
        except KeyError:
            raise KeyError(f"no saved site {name!r} (have: {', '.join(self.data['sites']) or 'none'})") from None

    def mapping(self, fov_m: float) -> list[list[float]] | None:
        """The stage_to_image measured at this field of view (it is per magnification)."""
        entries = self.data.get("stage_to_image") or {}
        best = min(entries.items(), key=lambda kv: abs(math.log(float(kv[0]) / fov_m)), default=None)
        if best is None or abs(math.log(float(best[0]) / fov_m)) > math.log(1.5):
            return None
        return best[1]


# ----------------------------------------------------------------------------
# Instrument over MCP
# ----------------------------------------------------------------------------

def _parse(result: Any) -> Any:
    """MCP tool results arrive as data, JSON text, or lists of text blocks; normalise to Python values."""
    if isinstance(result, list) and result and all(isinstance(r, dict) and "text" in r for r in result):
        result = "".join(r["text"] for r in result)
    if hasattr(result, "content") and not isinstance(result, (str, bytes)):
        result = result.content
        return _parse(result)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (ValueError, TypeError):
            return result
    return result


class McpInstrument:
    """Resolves the instrument's class-prefixed tool names (``DigitalTwin_move_stage`` ...)."""

    def __init__(self, call: Call, tool_names: list[str], prefix: str | None = None):
        self.call = call
        if prefix is None:
            prefixes = {n.rsplit("_move_stage", 1)[0] for n in tool_names if n.endswith("_move_stage")}
            if len(prefixes) != 1:
                raise ValueError(f"cannot pick the instrument: tools ending in _move_stage come from {prefixes or 'none'}")
            prefix = prefixes.pop()
        self.prefix = prefix

    async def _c(self, command: str, args: dict | None = None) -> Any:
        return _parse(await self.call(f"{self.prefix}_{command}", args or {}))

    async def get_stage(self) -> list[float]:
        return [float(v) for v in await self._c("get_stage")]

    async def move_stage(self, position: list[float]) -> None:
        await self._c("move_stage", {"position": [float(v) for v in position]})

    async def get_fov(self) -> float:
        return float(await self._c("get_fov"))

    async def set_fov(self, fov: float) -> None:
        await self._c("set_fov", {"fov": float(fov)})

    async def acquire(self) -> str:
        key = await self._c("acquire_scanned_image", {"detector_list": ["haadf"]})
        return key if isinstance(key, str) else json.dumps(key)

    async def register(self, reference_key: str, image_key: str, **kw) -> dict:
        return _parse(await self.call("register_images", {"reference_key": reference_key, "image_key": image_key,
                                                          **{k: v for k, v in kw.items() if v is not None}}))


# ----------------------------------------------------------------------------
# Save and calibrate (plain coroutines)
# ----------------------------------------------------------------------------

async def calibrate(inst: McpInstrument, book: SiteBook, step_fraction: float = 0.15) -> list[list[float]]:
    """Measure stage_to_image at the current field of view: ± steps per axis from a fresh start image."""
    fov = await inst.get_fov()
    start = await inst.get_stage()
    step = step_fraction * fov
    cols = []
    try:
        for axis in (0, 1):
            offsets = []
            for sign in (1, -1):
                await inst.move_stage(start)
                a = await inst.acquire()
                moved = list(start)
                moved[axis] += sign * step
                await inst.move_stage(moved)
                r = await inst.register(a, await inst.acquire(), image_fov_m=fov)
                if r["status"] != "match":
                    raise RuntimeError(f"calibration step could not be registered ({r['status']}: "
                                       f"{'; '.join(r['reasons'])}); choose a spot with more texture")
                offsets.append(r["offset_m"])
            cols.append([(offsets[0][i] - offsets[1][i]) / (2 * step) for i in (0, 1)])
    finally:
        await inst.move_stage(start)
    mapping = [[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]]
    book.data.setdefault("stage_to_image", {})[repr(float(fov))] = mapping
    book.save()
    return mapping


async def save_site(inst: McpInstrument, book: SiteBook, name: str, landmark: bool = False,
                    notes: str = "") -> dict:
    """Image the current spot and record everything needed to come back to it."""
    key = await inst.acquire()
    site = {"name": name, "key": key, "stage": await inst.get_stage(), "fov_m": await inst.get_fov(),
            "landmark": bool(landmark), "notes": notes, "saved_at": time.time(),
            "mount": book.data.get("mount", "original"), "visits": []}
    book.data["sites"][name] = site
    book.save()
    return site


# ----------------------------------------------------------------------------
# The return graph
# ----------------------------------------------------------------------------

class ReturnState(TypedDict, total=False):
    site: str
    predicted: list[float]
    mapping: list[list[float]] | None
    survey_factor: float
    max_rotation_deg: float
    tolerance_px: float
    max_corrections: int
    corrections: int
    survey: dict
    verify: dict
    status: str
    reasons: list[str]
    keys: dict
    final_stage: list[float]
    visit: dict


def build_return_graph(inst: McpInstrument, book: SiteBook):
    """Compile the return-to-site StateGraph bound to *inst* and *book*."""
    from langgraph.graph import END, START, StateGraph

    async def goto(state: ReturnState) -> dict:
        site = book.site(state["site"])
        predicted = state.get("predicted") or site["stage"]
        await inst.move_stage(predicted)
        mapping = book.mapping(site["fov_m"])
        reasons = [] if mapping else ["no stage_to_image measured at this magnification: run calibrate first"]
        return {"predicted": predicted, "mapping": mapping, "corrections": 0, "keys": {}, "reasons": reasons}

    async def survey(state: ReturnState) -> dict:
        site = book.site(state["site"])
        survey_fov = site["fov_m"] * state.get("survey_factor", 4.0)
        await inst.set_fov(survey_fov)
        key = await inst.acquire()
        r = await inst.register(site["key"], key, reference_fov_m=site["fov_m"], image_fov_m=survey_fov,
                                max_rotation_deg=state.get("max_rotation_deg", 0.0),
                                stage_to_image=state.get("mapping"))
        return {"survey": r, "keys": {**state.get("keys", {}), "survey": key}}

    def after_survey(state: ReturnState) -> str:
        if state["survey"]["status"] != "match" or not state.get("mapping"):
            return "record"
        return "centre"

    async def centre(state: ReturnState) -> dict:
        await _apply(state["survey"]["stage_correction_m"])
        await inst.set_fov(book.site(state["site"])["fov_m"])
        return {}

    async def _apply(correction: list[float]) -> None:
        pos = await inst.get_stage()
        pos[0] += float(correction[0])
        pos[1] += float(correction[1])
        await inst.move_stage(pos)

    async def verify(state: ReturnState) -> dict:
        site = book.site(state["site"])
        key = await inst.acquire()
        r = await inst.register(site["key"], key, image_fov_m=site["fov_m"], stage_to_image=state.get("mapping"),
                                max_rotation_deg=min(1.0, state.get("max_rotation_deg", 0.0)))
        return {"verify": r, "keys": {**state.get("keys", {}), "native": key}}

    def after_verify(state: ReturnState) -> str:
        r = state["verify"]
        if r["status"] != "match":
            return "record"
        if math.hypot(*r["offset_px"]) <= state.get("tolerance_px", 2.0):
            return "record"
        return "correct" if state.get("corrections", 0) < state.get("max_corrections", 3) else "record"

    async def correct(state: ReturnState) -> dict:
        await _apply(state["verify"]["stage_correction_m"])
        return {"corrections": state.get("corrections", 0) + 1}

    async def record(state: ReturnState) -> dict:
        site = book.site(state["site"])
        await inst.set_fov(site["fov_m"])          # every exit leaves the site's magnification, not the survey's
        reasons = list(state.get("reasons", []))
        if "verify" in state:
            r = state["verify"]
            status = r["status"]
            reasons += r.get("reasons", [])
            if status == "match" and math.hypot(*r["offset_px"]) > state.get("tolerance_px", 2.0):
                status = "not_converged"
                reasons.append(f"still {math.hypot(*r['offset_px']):.1f} px off after "
                               f"{state.get('corrections', 0)} corrections")
        else:
            r = state.get("survey") or {}
            status = r.get("status", "no_match") if state.get("mapping") else "not_calibrated"
            reasons += [f"survey: {x}" for x in r.get("reasons", [])]
        final_stage = await inst.get_stage()
        visit = {"t": time.time(), "status": status, "reasons": reasons, "keys": state.get("keys", {}),
                 "predicted": state.get("predicted"), "final_stage": final_stage,
                 "offset_m": (state.get("verify") or {}).get("offset_m"),
                 "sigma_m": (state.get("verify") or {}).get("sigma_m"),
                 "ncc": (state.get("verify") or state.get("survey") or {}).get("ncc"),
                 "mount": book.data.get("mount", "original")}
        site.setdefault("visits", []).append(visit)
        book.save()
        return {"status": status, "reasons": reasons, "final_stage": final_stage, "visit": visit}

    g = StateGraph(ReturnState)
    for name, fn in (("goto", goto), ("survey", survey), ("centre", centre), ("verify", verify),
                     ("correct", correct), ("record", record)):
        g.add_node(name, fn)
    g.add_edge(START, "goto")
    g.add_edge("goto", "survey")
    g.add_conditional_edges("survey", after_survey, {"centre": "centre", "record": "record"})
    g.add_edge("centre", "verify")
    g.add_conditional_edges("verify", after_verify, {"correct": "correct", "record": "record"})
    g.add_edge("correct", "verify")
    g.add_edge("record", END)
    return g.compile()


async def return_to_site(inst: McpInstrument, book: SiteBook, name: str, *, predicted: list[float] | None = None,
                         max_rotation_deg: float = 0.0, survey_factor: float = 4.0) -> dict:
    """Run the graph for one site and return its final state (``status`` is the answer)."""
    graph = build_return_graph(inst, book)
    state: ReturnState = {"site": name, "max_rotation_deg": max_rotation_deg, "survey_factor": survey_factor}
    if predicted is not None:
        state["predicted"] = predicted
    return await graph.ainvoke(state)
