"""Goals: what an agent must achieve, checked from facts rather than from what the model says.

A ``Goal`` is a statement, a list of criteria and a budget. After each attempt
the checker builds ``Facts`` from the tool calls the agent made and the
servers' replies (never from the model's own text), verifies the keys it
found against the live servers, and either accepts the attempt or returns
the unmet criteria as feedback for a bounded retry.

Only calls that make something count as producing it: an acquisition, a job
the run started (run_job, ``<device>_start``, assemble_map, score_map) and the
waits and results of those jobs. Listing or reading data (list_data,
view_data, map_info, ...) never produces it: an old spectrum found by
list_data does not meet "take a spectrum".

Criteria (``{"kind": ..., ...}``):

==================== ============================================== ==========================================
kind                 met when                                       options
==================== ============================================== ==========================================
``job_done``         a job of ``device`` this run started finished  ``device`` (default ``Mapper``)
``map``              a stitched map this run made opens (map_info)  ``min_pois``, ``poi_kind`` (list_pois)
``data``             >= ``count`` keys of ``data_kind`` this run    ``data_kind`` (image, spectrum, ...),
                     made open with view_data                       ``count`` (default 1)
``no_unrequested``   no high-risk call outside ``allow`` patterns   ``allow`` (default: running jobs)
``grounded_answer``  every data key the answer names came back      (none)
                     from a server in this run
==================== ============================================== ==========================================

Risk is the endpoint's own classification (``endpoint_info`` on the MCP server,
so deployment overrides and STEM settings count as the gate counts them),
or the safety policy applied to the name when the endpoint cannot be asked.

A goal written as text (``--goal "..."``) gets criteria inferred from its
words: a map, interesting regions, spectra or an image, and an ``allow`` list
for ``no_unrequested`` that fits what it asks (an image allows the settings
and acquisition an image needs; "go to" allows a stage move). A request that
only asks about data gets no production criterion and allows no action.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from asyncroscopy.safety.policy import tool_risk
from asyncroscopy.skills.registry import read_only_request

KINDS = ("job_done", "map", "data", "no_unrequested", "grounded_answer")
DATA_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]*\.(?:h5|hdf5|hdf|nxs|tiff?)\b")
# A stitched map key ends in _map and carries its run id (so tool names such as assemble_map do not match).
MAP_KEY = re.compile(r"\b(?=[A-Za-z0-9_\-.]*\d)[A-Za-z0-9][A-Za-z0-9_\-.]*_map\b")
HEX_ID = re.compile(r"\b[0-9a-f]{12}\b")  # job, reservation and request ids
# High-risk calls a goal file allows when it names none: running (and stopping) the job it asks for.
DEFAULT_ALLOW = ["run_job", "*_start", "*_reserve", "*_abort", "*_release"]
JOB_TOOLS = ["run_job", "Mapper*_start", "Mapper*_reserve", "Mapper*_abort", "Mapper*_release"]
IMAGE_TOOLS = ["*_acquire_scanned_image", "*_acquire_image", "*_acquire_camera_image", "*_set_fov", "*_set_defocus", "*_unblank_beam", "*_blank_beam", "*_place_beam"]
SPECTRUM_TOOLS = ["*_acquire_spectrum", "*_acquire_eds*", "*_place_beam", "*_unblank_beam", "*_blank_beam", "*_set_fov"]
MOVE_TOOLS = ["*_move_stage"]
# Tools whose replies are what the run made; with ``_job_id`` below, waits and results of its own jobs.
STARTERS = ("run_job", "assemble_map", "score_map")
JOB_READERS = ("wait_for_job", "analysis_job_status", "analysis_job_results")
ACTIVE = {"reserved", "queued", "running", "aborting", "paused_awaiting_approval", "paused_suspended"}


@dataclass
class Budget:
    max_seconds: float = 1800.0
    max_tool_calls: int = 60
    max_dose_e_per_a2: float | None = None
    max_attempts: int = 2


@dataclass
class Goal:
    statement: str
    criteria: list[dict] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)

    @classmethod
    def from_text(cls, statement: str, **budget) -> Goal:
        """A goal with criteria inferred from its words (always grounded_answer and no_unrequested)."""
        text = statement.lower()
        criteria: list[dict] = []
        allow: list[str] = []
        if not read_only_request(statement):
            if re.search(r"\bmap\b|\bmaps\b|\bmapping\b|\bsurvey\b", text):
                entry: dict[str, Any] = {"kind": "map"}
                wanted = re.search(r"(\d+)\s+(?:most\s+)?(?:interesting|notable|unusual)", text)
                if wanted or re.search(r"interesting|notable|unusual|features?\b|regions? of interest", text):
                    entry.update(min_pois=int(wanted.group(1)) if wanted else 1, poi_kind="candidate")
                criteria.append(entry)
                allow += JOB_TOOLS
            if re.search(r"\beds\b|\bedx\b|spectr|composition|element", text):
                criteria.append({"kind": "data", "data_kind": "spectrum", "count": 1})
                if not allow:  # a spectrum on its own, not one per map tile
                    allow += SPECTRUM_TOOLS
            imaging = re.search(r"\b(?:take|acquire|capture|record)\b.*\bimage\b|\bimage (?:it|them|that|this|there|each)\b|high(?:er)?[- ]magnification", text)
            if imaging:
                if not any(c["kind"] == "map" for c in criteria):
                    criteria.append({"kind": "data", "data_kind": "image", "count": 1})
                allow += IMAGE_TOOLS
            if re.search(r"\bgo(?: back)? to\b|\bmove\b|\brevisit\b|\bdrive\b|\bat x\b|\bposition\b|high(?:er)?[- ]magnification|\bimage them\b", text):
                allow += MOVE_TOOLS
        criteria += [{"kind": "no_unrequested", "allow": sorted(set(allow))}, {"kind": "grounded_answer"}]
        return cls(statement, criteria, Budget(**budget))

    @classmethod
    def load(cls, source: str | Path) -> Goal:
        """A goal from a YAML or JSON file: ``statement``, optional ``criteria`` and ``budget``."""
        data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Goal:
        if not isinstance(data, dict) or not data.get("statement"):
            raise ValueError("a goal needs a 'statement'")
        unknown = set(data) - {"statement", "criteria", "budget"}
        if unknown:
            raise ValueError(f"unknown goal fields {sorted(unknown)}; allowed: statement, criteria, budget")
        budget = Budget(**(data.get("budget") or {}))
        if data.get("criteria") is None:
            goal = cls.from_text(str(data["statement"]))
            goal.budget = budget
            return goal
        criteria = list(data["criteria"])
        for c in criteria:
            if not isinstance(c, dict) or c.get("kind") not in KINDS:
                raise ValueError(f"criterion {c!r}: 'kind' must be one of {KINDS}")
        return cls(str(data["statement"]), criteria, budget)

    def describe(self) -> str:
        """The criteria in words, for the agent's prompt."""
        words = {
            "job_done": lambda c: f"a {c.get('device', 'Mapper')} job finishes (state done)",
            "map": lambda c: "a stitched map exists" + (f" with at least {c['min_pois']} {c.get('poi_kind', '')} points of interest".replace("  ", " ") if c.get("min_pois") else ""),
            "data": lambda c: f"at least {c.get('count', 1)} {c.get('data_kind', 'data')} recorded",
            "no_unrequested": lambda c: "no instrument actions beyond what the goal needs",
            "grounded_answer": lambda c: "the answer names only keys the tools returned",
        }
        return "; ".join(words[c["kind"]](c) for c in self.criteria)

    def to_dict(self) -> dict:
        return {"statement": self.statement, "criteria": self.criteria, "budget": asdict(self.budget)}


# ---------------------------------------------------------------------- calls
@dataclass
class Call:
    """One tool call and the server's reply. The one record of a call for goals, graphs and benchmarks."""

    name: str
    args: dict
    result: str = ""
    error: bool = False

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict:
        return {"name": self.name, "args": self.args, "result": self.result, "error": self.error}

    @classmethod
    def from_dict(cls, data: dict) -> Call:
        error = data.get("error")
        if error is None and "ok" in data:
            error = not data["ok"]
        return cls(str(data.get("name", "")), dict(data.get("args") or {}), str(data.get("result", "")), bool(error))


def trace_from_messages(messages) -> list[Call]:
    """The tool calls in a LangChain message list, each with the server's reply.

    A call failed when it has no reply, its reply has status "error", or the reply text is a
    tool error ("Error...", "APPROVAL REQUIRED...").
    """
    replies = {getattr(m, "tool_call_id", None): m for m in messages if getattr(m, "type", "") == "tool"}
    calls = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            reply = replies.get(call.get("id"))
            content = getattr(reply, "content", "") if reply is not None else ""
            text = content if isinstance(content, str) else json.dumps(content, default=str)
            error = reply is None or getattr(reply, "status", "") == "error" or text.startswith(("Error", "APPROVAL REQUIRED"))
            calls.append(Call(call["name"], dict(call.get("args") or {}), text, error))
    return calls


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def _job_id(call: Call) -> str | None:
    """The job a wait/status/results call asks about (an argument, or inside a JSON "request")."""
    if isinstance(call.args.get("job_id"), str):
        return call.args["job_id"]
    request = call.args.get("request")
    body = _json(request) if isinstance(request, str) else request
    return body.get("job_id") if isinstance(body, dict) and isinstance(body.get("job_id"), str) else None


def _device(call: Call) -> str:
    if call.args.get("device"):
        return str(call.args["device"])
    device, _, command = call.name.partition("_")
    if command and device[:1].isupper():
        return device
    if call.name in ("assemble_map", "score_map") or call.name.startswith("analysis_"):
        return "analysis"
    return "unknown"


def _is_starter(name: str) -> bool:
    return name in STARTERS or (name.partition("_")[0][:1].isupper() and name.endswith("_start"))


def _is_job_reader(name: str) -> bool:
    return name in JOB_READERS or name.endswith(("_job_status", "_results"))


def _is_acquisition(name: str) -> bool:
    device, _, command = name.partition("_")
    return bool(command) and device[:1].isupper() and command.startswith("acquire")


@dataclass
class Facts:
    data_keys: set[str] = field(default_factory=set)  # data this run made
    map_keys: set[str] = field(default_factory=set)  # stitched maps this run made
    seen_keys: set[str] = field(default_factory=set)  # every key a server returned in this run (the answer may name them)
    jobs: dict[str, dict] = field(default_factory=dict)  # job_id -> {"device", "state"}, for jobs this run started
    dose_e_per_a2: float = 0.0
    high_risk: list[str] = field(default_factory=list)
    errors: int = 0
    calls: int = 0

    @classmethod
    def from_calls(cls, calls: list[Call], risks: dict[str, str] | None = None) -> Facts:
        """Facts from the calls of one run. ``risks``: the endpoint's tool risk table (endpoint_info)."""
        facts = cls()
        for call in calls:
            facts.calls += 1
            if call.error:
                facts.errors += 1
            if tool_risk(call.name, risks) == "high":
                facts.high_risk.append(call.name)
            if call.error:
                continue
            facts.seen_keys.update(DATA_KEY.findall(call.result))
            facts.seen_keys.update(MAP_KEY.findall(call.result))
            payload = _json(call.result)
            if (call.name.endswith("_reserve") or call.name == "run_job") and isinstance(payload, dict):
                estimate = payload.get("estimate")
                if isinstance(estimate, dict) and isinstance(estimate.get("max_dose_e_per_a2"), (int, float)):
                    facts.dose_e_per_a2 += float(estimate["max_dose_e_per_a2"])
            asked = _job_id(call)
            produced = _is_acquisition(call.name) or _is_starter(call.name) or (_is_job_reader(call.name) and asked in facts.jobs)
            if not produced:
                continue
            if payload is None or isinstance(payload, str):
                text = payload if isinstance(payload, str) else call.result
                facts.data_keys.update(DATA_KEY.findall(text))
                facts.map_keys.update(MAP_KEY.findall(text))
                continue
            for node in _walk(payload):
                for key, value in node.items():
                    if not isinstance(value, str):
                        continue
                    if key == "map" or MAP_KEY.fullmatch(value):
                        facts.map_keys.add(value)
                    elif DATA_KEY.fullmatch(value) and key in {"run", "key", "result", "spectrum_key", "image_key"}:
                        facts.data_keys.add(value)
                job_id = node.get("job_id")
                if isinstance(job_id, str) and (_is_starter(call.name) or job_id in facts.jobs):
                    previous = facts.jobs.get(job_id, {})
                    state = node.get("state") if isinstance(node.get("state"), str) else previous.get("state", "queued")
                    facts.jobs[job_id] = {"device": previous.get("device") or _device(call), "state": state}
        facts.seen_keys |= facts.data_keys | facts.map_keys
        return facts

    def active_jobs(self) -> dict[str, dict]:
        """Jobs this run started that have not reached done, failed or aborted (as last reported)."""
        return {j: v for j, v in self.jobs.items() if v["state"] in ACTIVE}

    def produced(self) -> list[str]:
        return sorted(self.map_keys | self.data_keys)

    def summary(self) -> str:
        parts = []
        if self.map_keys:
            parts.append(f"maps {sorted(self.map_keys)}")
        if self.data_keys:
            keys = sorted(self.data_keys)
            parts.append(f"data {keys[:6]}" + (f" and {len(keys) - 6} more" if len(keys) > 6 else ""))
        if self.jobs:
            parts.append("jobs " + ", ".join(f"{j} {v['device']} {v['state']}" for j, v in list(self.jobs.items())[-4:]))
        parts.append(f"{self.calls} tool calls ({self.errors} failed)")
        return "; ".join(parts)


# ---------------------------------------------------------------------- checking
@dataclass
class Verdict:
    met: bool
    results: list[dict]  # {"criterion", "met", "why"}
    over_budget: str = ""

    def feedback(self) -> str:
        unmet = [r for r in self.results if not r["met"]]
        return "\n".join(f"- not yet: {r['why']}" for r in unmet)


def _text_of(result) -> str:
    return "".join(getattr(b, "text", "") for b in getattr(result, "content", []) or [])


class GoalChecker:
    """Checks a goal's criteria against Facts, verifying keys on the live servers.

    ``instrument`` and ``analysis`` are McpServer-like objects (``call(name, args)``
    returning a fastmcp result); either may be None, in which case criteria that
    need it are judged from the facts alone and say so.
    """

    def __init__(self, instrument=None, analysis=None):
        self.instrument = instrument
        self.analysis = analysis
        self._risks: dict[str, str] | None = None

    async def _ok(self, server, name: str, args: dict):
        if server is None:
            return None
        try:
            result = await server.call(name, args)
        except Exception:
            return False
        return None if result is None else (not result.is_error, result)

    async def risks(self) -> dict[str, str] | None:
        """The endpoint's tool risk table (endpoint_info), or None when it cannot be asked."""
        if self._risks is None and self.instrument is not None:
            opened = await self._ok(self.instrument, "endpoint_info", {})
            if opened and opened[0]:
                payload = getattr(opened[1], "structured_content", None) or _json(_text_of(opened[1])) or {}
                self._risks = dict(payload.get("risks") or {}) or None
        return self._risks

    async def facts(self, calls: list[Call]) -> Facts:
        return Facts.from_calls(calls, await self.risks())

    async def check(self, goal: Goal, facts: Facts, answer: str, seconds: float = 0.0) -> Verdict:
        results = []
        for criterion in goal.criteria:
            met, why = await getattr(self, f"_check_{criterion['kind']}")(criterion, facts, answer)
            results.append({"criterion": criterion, "met": met, "why": why})
        over = ""
        b = goal.budget
        if b.max_dose_e_per_a2 is not None and facts.dose_e_per_a2 > b.max_dose_e_per_a2:
            over = f"dose {facts.dose_e_per_a2:.1f} e/A^2 is over the budget of {b.max_dose_e_per_a2}"
        elif facts.calls > b.max_tool_calls:
            over = f"{facts.calls} tool calls is over the budget of {b.max_tool_calls}"
        elif seconds > b.max_seconds:
            over = f"{seconds:.0f} s is over the budget of {b.max_seconds:.0f} s"
        return Verdict(all(r["met"] for r in results) and not over, results, over)

    async def stop_jobs(self, facts: Facts) -> list[Call]:
        """Abort the jobs this run started that are still active (the run is over). Returns the calls made."""
        made = []
        for job_id, job in facts.active_jobs().items():
            if job["device"] in ("analysis", "unknown") or self.instrument is None:
                continue
            args = {"request": json.dumps({"job_id": job_id})}
            opened = await self._ok(self.instrument, f"{job['device']}_abort", args)
            ok = bool(opened and opened[0])
            made.append(Call(f"{job['device']}_abort", args, _text_of(opened[1]) if opened else "", not ok))
        return made

    async def _check_job_done(self, c: dict, facts: Facts, answer: str):
        device = c.get("device", "Mapper")
        done = [j for j, v in facts.jobs.items() if v["device"].startswith(device) and v["state"] == "done"]
        if done:
            return True, f"{device} job {done[-1]} finished"
        seen = [f"{j} {v['state']}" for j, v in facts.jobs.items() if v["device"].startswith(device)]
        return False, f"no {device} job this run started reached state done ({', '.join(seen) or 'none started'})"

    async def _check_map(self, c: dict, facts: Facts, answer: str):
        if not facts.map_keys:
            return False, "this run made no stitched map (the map job, or assemble_map on its run, gives one)"
        for key in sorted(facts.map_keys):
            opened = await self._ok(self.analysis, "map_info", {"map": key})
            if opened is False or (opened and not opened[0]):
                continue
            if not c.get("min_pois"):
                return True, f"map {key} exists" + ("" if opened else " (not verified: no analysis server)")
            if not opened:
                return False, f"map {key}: cannot count its points of interest without the analysis server"
            listed = await self._ok(self.analysis, "list_pois", {"map": key, **({"kind": c["poi_kind"]} if c.get("poi_kind") else {})})
            pois = []
            if listed and listed[0]:
                payload = getattr(listed[1], "structured_content", None)
                payload = payload.get("result", payload) if isinstance(payload, dict) else payload
                if payload is None:
                    payload = _json(_text_of(listed[1]))
                pois = payload if isinstance(payload, list) else []
            if len(pois) >= int(c["min_pois"]):
                return True, f"map {key} has {len(pois)} {c.get('poi_kind', '')} points of interest".replace("  ", " ")
            return False, f"map {key} has {len(pois)} {c.get('poi_kind', '')} points of interest, {c['min_pois']} wanted (score_map adds them)".replace("  ", " ")
        return False, f"the maps this run made do not open: {sorted(facts.map_keys)}"

    async def _check_data(self, c: dict, facts: Facts, answer: str):
        kind, wanted = c.get("data_kind"), int(c.get("count", 1))
        found = []
        for key in sorted(facts.data_keys):
            opened = await self._ok(self.instrument, "view_data", {"key": key, "format": "text"})
            if opened is None:
                found.append(key)  # no instrument to ask: trust the server reply that named it
                continue
            if not opened or not opened[0]:
                continue
            text = _text_of(opened[1])
            structured = getattr(opened[1], "structured_content", None) or {}
            kinds = {structured.get("kind")} | set(re.findall(r":\s*(\w+), shape", text)) | set(re.findall(r"\b(\w+)/:", text))
            if kind is None or kind in kinds or (kind == "spectrum" and ("spectra" in kinds or "spectrum_image" in kinds)) or (kind == "image" and ("tiles" in kinds or "image" in kinds)):
                found.append(key)
        if len(found) >= wanted:
            return True, f"{len(found)} {kind or 'data'} keys this run made: {found[:3]}"
        return False, f"{len(found)} {kind or 'data'} recorded in this run, {wanted} wanted (data found by listing or reading does not count)"

    async def _check_no_unrequested(self, c: dict, facts: Facts, answer: str):
        allow = c["allow"] if "allow" in c else DEFAULT_ALLOW
        extra = [n for n in facts.high_risk if not any(fnmatch.fnmatchcase(n, p) for p in allow)]
        if extra:
            return False, f"instrument actions the goal did not need: {sorted(set(extra))}"
        return True, "no unrequested instrument actions"

    async def _check_grounded_answer(self, c: dict, facts: Facts, answer: str):
        named = set(DATA_KEY.findall(answer)) | set(MAP_KEY.findall(answer))
        invented = sorted(k for k in named if k not in facts.seen_keys)
        if invented:
            return False, f"the answer names keys no server returned in this run: {invented[:5]}"
        return True, "the answer names only keys the servers returned"
