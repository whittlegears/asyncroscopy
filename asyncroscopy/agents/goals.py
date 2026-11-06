"""Goals: what an agent must achieve, checked from facts rather than from what the model says.

A ``Goal`` is a statement, a list of criteria and a budget. After each attempt
the checker builds ``Facts`` from the tool calls the agent made and the
servers' replies (never from the model's own text), verifies the keys it
found against the live servers, and either accepts the attempt or returns
the unmet criteria as feedback for a bounded retry.

Only calls that make something count as producing it: an acquisition, a job
the run started (run_job, ``<device>_start``) and the waits and results of
those jobs, and the stitching of a run this run acquired. Listing or reading
data (list_data, view_data, map_info, ...) never produces it, and neither does
scoring or re-stitching an existing map: an old spectrum found by list_data
does not meet "take a spectrum".

Criteria (``{"kind": ..., ...}``):

==================== ============================================== ==========================================
kind                 met when                                       options
==================== ============================================== ==========================================
``job_done``         a job of ``device`` this run started finished  ``device`` (default ``Mapper``)
``map``              a stitched map this run made opens (map_info)  ``min_pois``, ``poi_kind`` (list_pois)
``data``             >= ``count`` keys of ``data_kind`` this run    ``data_kind`` (image, spectrum, ...),
                     made open with view_data                       ``count`` (default 1)
``no_unrequested``   no high-risk call outside what was asked       ``categories`` (job, acquire, move,
                                                                    settings, beam) or ``allow`` globs
``grounded_answer``  every data key the answer names came back      (none)
                     from a server in this run
==================== ============================================== ==========================================

What each tool is comes from the endpoint itself (``endpoint_info``'s ``tools``, read once per run:
``asyncroscopy.agents.catalog``), never from its name: its risk as the approval gate sees it (so
deployment overrides and STEM settings count as the gate counts them), the device it belongs to, and
its canonical role, from which its category follows (acquire.* is acquire, stage.move move, setting.set
and optics.* settings, beam.* beam, probe.* probe, the job envelope and run_job job). Without the
endpoint's catalog ``no_unrequested`` is reported as not checked rather than guessed, and no call counts
as an acquisition or a job.

A goal written as text (``--goal "..."``) gets only the criteria its words
make plain: a new map ("map a 600 nm square", "make an overview map"), never
one for a map it names; spectra or images from an acquisition verb on them,
unless "without EDS"; ``no_unrequested`` categories for what it asks (an
image allows acquiring, settings and the beam; "go to" a stage move). Unsure
means fewer criteria: the retry must never push work the request did not ask
for. A goal file is the precise way to state criteria. "Do not wait" (or
``leave_running``) keeps the goal from aborting the job it started.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from asyncroscopy.agents.catalog import ToolCatalog
from asyncroscopy.agents.skill_runs import skill_run_calls
from asyncroscopy.core.names import ACTIVE_JOB_STATES, DATA_KEY_IN_TEXT
from asyncroscopy.skills.registry import read_only_request

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from asyncroscopy.agents.mcp_tools import McpServer

KINDS = ("job_done", "map", "data", "no_unrequested", "grounded_answer")
# A stitched map key ends in _map and carries its run id (so tool names such as assemble_map do not match).
# A stitched map: <run>_map (stitched before map versions), <run>_map_<YYYYmmddTHHMMSS>_<hex4> (a re-stitch before
# versions), or a version <run>_map_<YYYYmmddTHHMMSSmmm>Z_<hex4> (pyTEMlib keeps every version).
MAP_KEY = re.compile(r"\b(?=[A-Za-z0-9_\-.]*\d)[A-Za-z0-9][A-Za-z0-9_\-.]*?_map(?:_\d{8}T\d{6}(?:\d{3}Z)?_[0-9a-f]{4})?\b")
HEX_ID = re.compile(r"\b[0-9a-f]{12}\b")  # job, reservation and request ids
# What kinds of instrument action a goal may take, from each tool's canonical role (any instrument, not only a STEM).
CATEGORIES = ("job", "acquire", "move", "settings", "beam", "probe")
# A goal file's no_unrequested with neither allow nor categories: running (and stopping) the job it asks for.
DEFAULT_CATEGORIES = ["job"]

# Inferring a goal from a sentence: only what the words say plainly. Unsure means fewer criteria, never
# wrong ones: a criterion the request did not ask for would make the retry push actions nobody wanted.
NEW_MAP = re.compile(
    r"\b(?:map|survey)\s+(?:a|an|the\s+(?:area|sample|region|square|whole|surface|flake)|this\s+(?:area|sample|region)|around|here|\d)"
    r"|\b(?:make|take|acquire|record|do|perform|run|start|get)\s+(?:a|an|another|one|new)\s+(?:[\w-]+\s+){0,3}?(?:map|survey)\b"
    r"|\bmapping\b",
    re.I,
)
NO_EDS = re.compile(r"\b(?:without|no|not|don't|do not|skip)\s+(?:any\s+|an?\s+|the\s+)?(?:eds|edx|x-ray|spectr\w*)", re.I)
SPECTRUM = re.compile(
    r"\b(?:take|taking|acquire|acquiring|measure|measuring|record|recording|collect|collecting|get)\b[^.?!]{0,40}\b(?:spectrum|spectra|eds|edx)\b"
    r"|\b(?:with|and)\s+(?:an?\s+)?(?:eds|edx)\b|\b(?:eds|edx)(?:\s+spectrum|\s+spectra)?\s+(?:at|on|with|for)\s+each\b",
    re.I,
)
IMAGE = re.compile(
    r"\bimage\s+(?:a|an)\s+(?:[\w.-]+\s+){0,3}?(?:square|area|region|field|patch)\b"
    r"|\b(?:take|taking|acquire|acquiring|capture|capturing|record|recording|get)\b[^.?!]{0,40}\b(?:image|images|picture|pictures|micrograph)s?\b"
    r"|\bimage\s+(?:it|them|that|this|there|each|the\s+region)\b|\bimaging\b|\bhigh(?:er)?[- ]magnification\b",
    re.I,
)
DOSE = re.compile(
    r"\b(?:at most|no more than|under|below|less than|max(?:imum)?(?: of)?|up to|limit(?:ed)? to|budget of)\s+(\d+(?:\.\d+)?)\s*"
    r"(?:e(?:lectrons?)?\s*(?:/|per)\s*(?:square\s+)?(?:a(?:ngstr[oö]m)?|Å)(?:\^?2|²)?|e/Å²|e/a\^?2)",
    re.I,
)
ACQUIRE_ANY = re.compile(r"\b(?:acquire|take|record|measure|collect|capture)\s+(?:a|an|another|some|one|\d+)\s+(?:[\w-]+\s+){0,2}?[\w-]+", re.I)
MOVE = re.compile(r"\bgo(?:\s+back)?\s+to\b|\bmove\b|\brevisit\b|\bdrive\b|\bat\s+x\s*=|\bto\s+(?:the\s+)?position\b|\bimage\s+them\b|\bhigh(?:er)?[- ]magnification\b|\bfind\s+(?:a|an|the)\s+(?:particle|feature|cluster|grain)", re.I)
REGIONS = re.compile(r"(\d+|one|two|three|four|five)?\s*(?:most\s+)?(?:interesting|notable|unusual)\s+(?:regions?|areas?|features?|spots?)", re.I)
LEAVE_RUNNING = re.compile(r"\b(?:do not|don't)\s+wait\b|\bleave\s+it\s+running\b|\bkeep\s+(?:it\s+)?running\b|\bin\s+the\s+background\b", re.I)
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def action_category(tool_name: str, catalog: ToolCatalog) -> str:
    """What a tool does: job, acquire, move, probe, beam, settings, or other (from its role in ``catalog``)."""
    return catalog.category(tool_name)


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
    leave_running: bool = False  # the request said to start a job and not wait for it: never abort it

    @classmethod
    def from_text(cls, statement: str, **budget: Any) -> Goal:
        """A goal with only the criteria its words make plain (see the module docstring).

        A new map is asked for by "map a ...", "make a ... map" and the like, never by a mention of an
        existing map ("on the map X", a key); spectra and images by an acquisition verb on them, and
        "without EDS" takes spectra away. When nothing is plain, the goal is checked only for
        unrequested actions and a grounded answer: a goal file is the precise way to say more.
        """
        criteria: list[dict] = []
        categories: set[str] = set()
        leave_running = bool(LEAVE_RUNNING.search(statement))
        if not read_only_request(statement):
            new_map = bool(NEW_MAP.search(statement))
            if new_map:
                categories |= {"job"}
                if not leave_running:  # a job left running produces no map before the goal ends
                    entry: dict[str, Any] = {"kind": "map"}
                    regions = REGIONS.search(statement)
                    if regions:
                        number = regions.group(1)
                        entry.update(min_pois=int(WORDS.get(number.lower(), number)) if number else 1, poi_kind="candidate")
                    criteria.append(entry)
            if SPECTRUM.search(statement) and not NO_EDS.search(statement):
                if not new_map:  # a map's EDS spectra come with the map job
                    criteria.append({"kind": "data", "data_kind": "spectrum", "count": 1})
                categories |= {"acquire", "settings", "beam"}
            if IMAGE.search(statement):
                if not new_map:
                    criteria.append({"kind": "data", "data_kind": "image", "count": 1})
                categories |= {"acquire", "settings", "beam"}
            acquire_any = ACQUIRE_ANY.search(statement)
            if acquire_any and "acquire" not in categories and not new_map:
                # Some other acquisition (a diffractogram, a force curve): data of any kind, as many as a number says.
                number = re.search(r"\b(\d+)\b", acquire_any.group(0))
                criteria.append({"kind": "data", "count": int(number.group(1)) if number else 1})
                categories |= {"acquire", "settings", "beam"}
            if MOVE.search(statement):
                categories |= {"move", "acquire", "settings", "beam"} if re.search(r"\bfind\b", statement, re.I) else {"move"}
            if re.search(r"\b(?:set|change|adjust)\b[^.?!]{0,30}\b(?:defocus|focus|field of view|fov|magnification|exposure|power)\b", statement, re.I):
                categories |= {"settings"}
            if re.search(r"\b(?:approach|engage|land|retract|withdraw|lift)\w*\b[^.?!]{0,20}\b(?:tip|probe|cantilever|needle)\b|\b(?:tip|probe)\s+(?:up|down)\b", statement, re.I):
                categories |= {"probe"}
            if re.search(r"\b(?:un)?blank\b|\blaser\s+(?:on|off)\b|\bswitch\s+(?:the\s+)?\w+\s+(?:on|off)\b|\bturn\s+(?:the\s+)?\w+\s+(?:on|off)\b", statement, re.I):
                categories |= {"beam"}
        criteria += [{"kind": "no_unrequested", "categories": sorted(categories)}, {"kind": "grounded_answer"}]
        stated = DOSE.search(statement)
        if stated and "max_dose_e_per_a2" not in budget:  # a dose the user states is a limit the run must keep
            budget["max_dose_e_per_a2"] = float(stated.group(1))
        return cls(statement, criteria, Budget(**budget), leave_running)

    @classmethod
    def load(cls, source: str | Path) -> Goal:
        """A goal from a YAML or JSON file: ``statement``, optional ``criteria``, ``budget`` and ``leave_running``."""
        data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Goal:
        if not isinstance(data, dict) or not data.get("statement"):
            raise ValueError("a goal needs a 'statement'")
        unknown = set(data) - {"statement", "criteria", "budget", "leave_running"}
        if unknown:
            raise ValueError(f"unknown goal fields {sorted(unknown)}; allowed: statement, criteria, budget, leave_running")
        budget = Budget(**(data.get("budget") or {}))
        if data.get("criteria") is None:
            goal = cls.from_text(str(data["statement"]))
            goal.budget = budget
            if "leave_running" in data:
                goal.leave_running = bool(data["leave_running"])
            return goal
        criteria = list(data["criteria"])
        for c in criteria:
            if not isinstance(c, dict) or c.get("kind") not in KINDS:
                raise ValueError(f"criterion {c!r}: 'kind' must be one of {KINDS}")
            bad = set(c.get("categories") or []) - set(CATEGORIES)
            if bad:
                raise ValueError(f"criterion {c!r}: categories must be among {CATEGORIES}")
        return cls(str(data["statement"]), criteria, budget, bool(data.get("leave_running", False)))

    def describe(self) -> str:
        """The criteria in words, for the agent's prompt."""
        words = {
            "job_done": lambda c: f"a {c.get('device', 'Mapper')} job finishes (state done)",
            "map": lambda c: "a stitched map exists" + (f" with at least {c['min_pois']} {c.get('poi_kind', '')} points of interest".replace("  ", " ") if c.get("min_pois") else ""),
            "data": lambda c: f"at least {c.get('count', 1)} {c.get('data_kind', 'data')} recorded",
            "no_unrequested": lambda c: "no instrument actions beyond what the request asks for",
            "grounded_answer": lambda c: "the answer names only keys the tools returned",
        }
        return "; ".join(words[c["kind"]](c) for c in self.criteria)

    def to_dict(self) -> dict:
        return {"statement": self.statement, "criteria": self.criteria, "budget": asdict(self.budget), "leave_running": self.leave_running}


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


def trace_from_messages(messages: Sequence[BaseMessage]) -> list[Call]:
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
            # A skill run's calls were made for the agent in the background: they count as its own.
            calls += [Call(c["tool"], dict(c.get("args") or {}), str(c.get("result", "")), bool(c.get("error"))) for c in skill_run_calls(text)]
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


def _device(call: Call, catalog: ToolCatalog) -> str:
    """The job device a call is about: run_job's device argument, the device of an envelope command (a
    multi-Mapper endpoint's "Mapper_key_start" is device "Mapper_key"), "analysis", or "unknown"."""
    if call.args.get("device"):
        return str(call.args["device"])
    if call.name in ("assemble_map", "score_map") or call.name.startswith("analysis_"):
        return "analysis"
    return catalog.device(call.name) or "unknown"


def _starts(call: Call, facts: Facts, catalog: ToolCatalog) -> str | None:
    """The kind of job a call starts in this run: "instrument" (run_job, a job device's start), "assemble"
    (stitching a run this run acquired), "score" (scoring makes points of interest, not a map), or None."""
    if call.name == "run_job" or (catalog.is_device_tool(call.name) and catalog.role(call.name) == "job.start"):
        return "instrument"
    if call.name == "assemble_map":
        return "assemble" if call.args.get("run") in facts.data_keys else None
    if call.name == "score_map":
        return "score"
    return None


def _reads_job(name: str, catalog: ToolCatalog) -> bool:
    return name in ("wait_for_job", "analysis_job_status", "analysis_job_results") or (
        catalog.is_device_tool(name) and catalog.role(name) in ("job.status", "job.results"))


def _is_acquisition(name: str, catalog: ToolCatalog) -> bool:
    return catalog.category(name) == "acquire"


@dataclass
class Facts:
    data_keys: set[str] = field(default_factory=set)  # data this run made
    map_keys: set[str] = field(default_factory=set)  # stitched maps this run made (its map jobs, or stitching its runs)
    seen_keys: set[str] = field(default_factory=set)  # every key a server returned in this run (the answer may name them)
    jobs: dict[str, dict] = field(default_factory=dict)  # job_id -> {"device", "state", "kind"}, for jobs this run started
    dose_e_per_a2: float = 0.0
    high_risk: list[str] = field(default_factory=list)
    risks_known: bool = False  # high_risk comes from the endpoint's own catalog (without one: not checked)
    catalog: ToolCatalog = field(default_factory=ToolCatalog.unknown, repr=False, compare=False)  # what the tools are
    errors: int = 0
    calls: int = 0

    @classmethod
    def from_calls(cls, calls: list[Call], catalog: ToolCatalog | None = None) -> Facts:
        """Facts from the calls of one run. ``catalog``: what the endpoint's tools are (endpoint_info); without
        it no call counts as an instrument action and high-risk calls are not known.

        Only what this run made counts: acquisitions, jobs it started and their results. A map counts
        when this run's instrument job made it, or this run stitched a run it acquired; scoring an
        existing map adds points of interest to it but does not make it this run's.
        """
        catalog = catalog if catalog is not None else ToolCatalog.unknown()
        facts = cls(risks_known=catalog.known, catalog=catalog)
        for call in calls:
            facts.calls += 1
            if call.error:
                facts.errors += 1
            if catalog.risk(call.name) == "high":
                facts.high_risk.append(call.name)
            if call.error:
                continue
            facts.seen_keys.update(DATA_KEY_IN_TEXT.findall(call.result))
            facts.seen_keys.update(MAP_KEY.findall(call.result))
            payload = _json(call.result)
            if (call.name == "run_job" or catalog.role(call.name) == "job.reserve") and isinstance(payload, dict):
                estimate = payload.get("estimate")
                if isinstance(estimate, dict) and isinstance(estimate.get("max_dose_e_per_a2"), (int, float)):
                    facts.dose_e_per_a2 += float(estimate["max_dose_e_per_a2"])
            started = _starts(call, facts, catalog)
            asked = _job_id(call)
            if _is_acquisition(call.name, catalog):
                kind = "acquisition"
            elif started:
                kind = started
            elif _reads_job(call.name, catalog) and asked in facts.jobs:
                kind = facts.jobs[asked]["kind"]
            else:
                continue
            if payload is None or isinstance(payload, str):
                text = payload if isinstance(payload, str) else call.result
                facts.data_keys.update(DATA_KEY_IN_TEXT.findall(text))
                facts.map_keys.update(MAP_KEY.findall(text))
                continue
            for node in _walk(payload):
                job_id = node.get("job_id")
                if isinstance(job_id, str) and (started or job_id in facts.jobs):
                    previous = facts.jobs.get(job_id, {})
                    state = node.get("state") if isinstance(node.get("state"), str) else previous.get("state", "queued")
                    facts.jobs[job_id] = {"device": previous.get("device") or _device(call, catalog), "state": state, "kind": previous.get("kind") or kind}
                if kind == "score":  # its result_keys name the map it scored, which this run did not make
                    continue
                for key, value in node.items():
                    if not isinstance(value, str):
                        continue
                    if key == "map" or MAP_KEY.fullmatch(value):
                        facts.map_keys.add(value)
                    elif DATA_KEY_IN_TEXT.fullmatch(value) and key in {"run", "key", "result", "spectrum_key", "image_key"}:
                        facts.data_keys.add(value)
        facts.seen_keys |= facts.data_keys | facts.map_keys
        return facts

    def active_jobs(self) -> dict[str, dict]:
        """Jobs this run started that have not reached done, failed or aborted (as last reported)."""
        return {j: v for j, v in self.jobs.items() if v["state"] in ACTIVE_JOB_STATES}

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

    def __init__(self, instrument: McpServer | None = None, analysis: McpServer | None = None) -> None:
        self.instrument = instrument
        self.analysis = analysis
        self._catalog: ToolCatalog | None = None

    async def _ok(self, server, name: str, args: dict):
        if server is None:
            return None
        try:
            result = await server.call(name, args)
        except Exception:
            return False
        return None if result is None else (not result.is_error, result)

    async def catalog(self) -> ToolCatalog:
        """What the endpoint's tools are (endpoint_info, asked once); an unknown catalog when it cannot be asked."""
        if self._catalog is None and self.instrument is not None:
            opened = await self._ok(self.instrument, "endpoint_info", {})
            if opened and opened[0]:
                payload = getattr(opened[1], "structured_content", None) or _json(_text_of(opened[1])) or {}
                self._catalog = ToolCatalog.from_info(payload)
        return self._catalog if self._catalog is not None else ToolCatalog.unknown()

    async def facts(self, calls: list[Call]) -> Facts:
        return Facts.from_calls(calls, await self.catalog())

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
        if self.analysis is None and not c.get("min_pois"):
            # No analysis server (a Raman, an AFM): nothing stitches, so the map is the job's own result.
            done = [j for j, v in facts.jobs.items() if v["device"] not in ("analysis", "unknown") and v["state"] == "done"]
            if done:
                return True, f"job {done[-1]} finished (this instrument has no analysis server, so its map is the job's result)"
            return False, "no job this run started reached state done (this instrument has no analysis server to stitch a map)"
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
        """No high-risk call outside what the request asked for: ``allow`` globs (a goal file), else
        ``categories`` (job, acquire, move, settings, beam, probe; from each tool's role). Not checked when the
        endpoint's catalog is unavailable: a guess from names would misjudge (set_fov on a STEM is low)."""
        if not facts.risks_known:
            return True, "not checked: the endpoint's tool catalog (endpoint_info) was not available"
        if "allow" in c:
            extra = [n for n in facts.high_risk if not any(fnmatch.fnmatchcase(n, p) for p in c["allow"])]
        else:
            allowed = set(c.get("categories", DEFAULT_CATEGORIES))
            extra = [n for n in facts.high_risk if action_category(n, facts.catalog) not in allowed]
        if extra:
            return False, f"instrument actions the request did not ask for: {sorted(set(extra))}"
        return True, "no unrequested instrument actions"

    async def _check_grounded_answer(self, c: dict, facts: Facts, answer: str):
        named = set(DATA_KEY_IN_TEXT.findall(answer)) | set(MAP_KEY.findall(answer))
        invented = sorted(k for k in named if k not in facts.seen_keys)
        if invented:
            return False, f"the answer names keys no server returned in this run: {invented[:5]}"
        return True, "the answer names only keys the servers returned"
