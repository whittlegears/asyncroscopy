"""Goals: what an agent must achieve, checked from facts rather than from what the model says.

A ``Goal`` is a statement, a list of criteria and a budget. After each attempt
the checker builds ``Facts`` from the tool calls the agent made and the
servers' replies (never from the model's own text), verifies the keys it
found against the live servers, and either accepts the attempt or returns
the unmet criteria as feedback for a bounded retry.

Criteria (``{"kind": ..., ...}``):

==================== ============================================== ==========================================
kind                 met when                                       options
==================== ============================================== ==========================================
``job_done``         a job of ``device`` reported state ``done``    ``device`` (default ``Mapper``)
``map``              a stitched map from this run opens (map_info)  ``min_pois``, ``poi_kind`` (list_pois)
``data``             >= ``count`` keys of ``data_kind`` from this   ``data_kind`` (image, spectrum, ...),
                     run open with view_data                        ``count`` (default 1)
``no_unrequested``   no high-risk call outside ``allow`` patterns   ``allow`` (default: job tools)
``grounded_answer``  every data key the answer names was produced   (none)
                     or read in this run
==================== ============================================== ==========================================

A goal written as text (``--goal "..."``) gets criteria inferred from its
words: a map, interesting regions, spectra or an image.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from asyncroscopy.safety.policy import risk_of

KINDS = ("job_done", "map", "data", "no_unrequested", "grounded_answer")
DATA_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]*\.(?:h5|hdf5|hdf|nxs|tiff?)\b")
# A stitched map key ends in _map and carries its run id (so tool names such as assemble_map do not match).
MAP_KEY = re.compile(r"\b(?=[A-Za-z0-9_\-.]*\d)[A-Za-z0-9][A-Za-z0-9_\-.]*_map\b")
# High-risk calls a goal allows by default: starting (and stopping) the job the request asked for
# (run_job is a native tool and is not counted; the envelope's own tools are allowed when used directly).
DEFAULT_ALLOW = ["*_start", "*_abort", "*_reserve", "*_release"]


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
        if re.search(r"\bmap\b|\bmaps\b|\bmapping\b|\bsurvey\b", text):
            entry: dict[str, Any] = {"kind": "map"}
            wanted = re.search(r"(\d+)\s+(?:most\s+)?(?:interesting|notable|unusual)", text)
            if wanted or re.search(r"interesting|notable|unusual|features?\b|regions? of interest", text):
                entry.update(min_pois=int(wanted.group(1)) if wanted else 1, poi_kind="candidate")
            criteria.append(entry)
        if re.search(r"\beds\b|\bedx\b|spectr|composition|element", text):
            criteria.append({"kind": "data", "data_kind": "spectrum", "count": 1})
        if re.search(r"\b(?:take|acquire|capture)\b.*\bimage\b", text) and not any(c["kind"] == "map" for c in criteria):
            criteria.append({"kind": "data", "data_kind": "image", "count": 1})
        criteria += [{"kind": "no_unrequested"}, {"kind": "grounded_answer"}]
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


# ---------------------------------------------------------------------- facts
@dataclass
class Call:
    name: str
    args: dict
    result: str
    error: bool = False


def trace_from_messages(messages) -> list[Call]:
    """The tool calls in a LangChain message list, each with the server's reply."""
    replies = {getattr(m, "tool_call_id", None): m for m in messages if getattr(m, "type", "") == "tool"}
    calls = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            reply = replies.get(call.get("id"))
            text = reply.content if reply is not None and isinstance(reply.content, str) else (json.dumps(reply.content) if reply is not None else "")
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


def _job_device(call: Call, job_id: str, jobs: dict) -> str:
    """The device a job runs on: known from its start, else from the call that reported it."""
    if job_id in jobs:
        return jobs[job_id]["device"]
    if call.args.get("device"):
        return str(call.args["device"])
    device, _, command = call.name.partition("_")
    if command and device[:1].isupper():
        return device
    if call.name.startswith(("analysis_", "assemble_", "score_")):
        return "analysis"
    return "unknown"


@dataclass
class Facts:
    data_keys: set[str] = field(default_factory=set)
    map_keys: set[str] = field(default_factory=set)
    read_keys: set[str] = field(default_factory=set)  # keys the agent looked at (it may name them)
    jobs: dict[str, dict] = field(default_factory=dict)  # job_id -> {"device", "state"}
    dose_e_per_a2: float = 0.0
    high_risk: list[str] = field(default_factory=list)
    errors: int = 0
    calls: int = 0

    @classmethod
    def from_calls(cls, calls: list[Call]) -> Facts:
        facts = cls()
        for call in calls:
            facts.calls += 1
            if call.error:
                facts.errors += 1
            device, _, command = call.name.partition("_")
            if command and device[:1].isupper() and risk_of(device, command, envelope=command in ("reserve", "start", "job_status", "abort", "results", "release")) == "high":
                facts.high_risk.append(call.name)
            for value in call.args.values():
                if isinstance(value, str):
                    facts.read_keys.update(DATA_KEY.findall(value))
                    facts.read_keys.update(MAP_KEY.findall(value))
            if call.error:
                continue
            payload = _json(call.result)
            if payload is None:
                facts.data_keys.update(DATA_KEY.findall(call.result))
                facts.map_keys.update(MAP_KEY.findall(call.result))
                continue
            for node in _walk(payload):
                for key, value in node.items():
                    if not isinstance(value, str):
                        continue
                    if key == "map" or MAP_KEY.fullmatch(value):
                        facts.map_keys.add(value)
                    elif DATA_KEY.fullmatch(value) and key in {"run", "key", "result", "spectrum_key"}:
                        facts.data_keys.add(value)
                if "job_id" in node and isinstance(node.get("state"), str):
                    facts.jobs[node["job_id"]] = {"device": _job_device(call, node["job_id"], facts.jobs), "state": node["state"]}
                elif "job_id" in node and call.name in ("assemble_map", "score_map"):
                    facts.jobs[node["job_id"]] = {"device": "analysis", "state": "queued"}
                estimate = node.get("estimate")
                if (call.name.endswith("_reserve") or call.name == "run_job") and isinstance(estimate, dict) and isinstance(estimate.get("max_dose_e_per_a2"), (int, float)):
                    facts.dose_e_per_a2 += float(estimate["max_dose_e_per_a2"])
            if isinstance(payload, str):
                facts.data_keys.update(DATA_KEY.findall(payload))
        facts.read_keys |= facts.data_keys | facts.map_keys
        return facts

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


class GoalChecker:
    """Checks a goal's criteria against Facts, verifying keys on the live servers.

    ``instrument`` and ``analysis`` are McpServer-like objects (``call(name, args)``
    returning a fastmcp result); either may be None, in which case criteria that
    need it are judged from the facts alone and say so.
    """

    def __init__(self, instrument=None, analysis=None):
        self.instrument = instrument
        self.analysis = analysis

    async def _ok(self, server, name: str, args: dict):
        if server is None:
            return None
        try:
            result = await server.call(name, args)
        except Exception:
            return False
        return None if result is None else (not result.is_error, result)

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

    async def _check_job_done(self, c: dict, facts: Facts, answer: str):
        device = c.get("device", "Mapper")
        done = [j for j, v in facts.jobs.items() if v["device"].startswith(device) and v["state"] == "done"]
        if done:
            return True, f"{device} job {done[-1]} finished"
        seen = [f"{j} {v['state']}" for j, v in facts.jobs.items() if v["device"].startswith(device)]
        return False, f"no {device} job reached state done ({', '.join(seen) or 'none started'})"

    async def _check_map(self, c: dict, facts: Facts, answer: str):
        if not facts.map_keys:
            return False, "no stitched map was made (assemble_map on the run gives one)"
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
                    payload = _json("".join(getattr(b, "text", "") for b in getattr(listed[1], "content", []) or []))
                pois = payload if isinstance(payload, list) else []
            if len(pois) >= int(c["min_pois"]):
                return True, f"map {key} has {len(pois)} {c.get('poi_kind', '')} points of interest".replace("  ", " ")
            return False, f"map {key} has {len(pois)} {c.get('poi_kind', '')} points of interest, {c['min_pois']} wanted (score_map adds them)".replace("  ", " ")
        return False, f"the map keys from this run do not open: {sorted(facts.map_keys)}"

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
            text = "".join(getattr(b, "text", "") for b in getattr(opened[1], "content", []) or [])
            structured = getattr(opened[1], "structured_content", None) or {}
            kinds = {structured.get("kind")} | set(re.findall(r":\s*(\w+), shape", text)) | set(re.findall(r"\b(\w+)/:", text))
            if kind is None or kind in kinds or (kind == "spectrum" and ("spectra" in kinds or "spectrum_image" in kinds)) or (kind == "image" and ("tiles" in kinds or "image" in kinds)):
                found.append(key)
        if len(found) >= wanted:
            return True, f"{len(found)} {kind or 'data'} keys: {found[:3]}"
        return False, f"{len(found)} {kind or 'data'} recorded in this run, {wanted} wanted"

    async def _check_no_unrequested(self, c: dict, facts: Facts, answer: str):
        allow = c.get("allow") or DEFAULT_ALLOW
        extra = [n for n in facts.high_risk if not any(fnmatch.fnmatchcase(n, p) for p in allow)]
        if extra:
            return False, f"instrument actions the goal did not need: {sorted(set(extra))}"
        return True, "no unrequested instrument actions"

    async def _check_grounded_answer(self, c: dict, facts: Facts, answer: str):
        named = set(DATA_KEY.findall(answer)) | set(MAP_KEY.findall(answer))
        invented = sorted(k for k in named if k not in facts.read_keys)
        if invented:
            return False, f"the answer names keys no tool returned: {invented[:5]}"
        return True, "the answer names only real keys"
