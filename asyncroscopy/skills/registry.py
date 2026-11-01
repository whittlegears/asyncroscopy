"""Find and load SKILL.md files.

A skill is a folder holding ``SKILL.md``: YAML front matter (name,
description, version, min_asyncroscopy, envelope_schema, tools, requires,
keywords, acts, run) followed by instructions in Markdown. ``acts`` says whether
following the skill moves or exposes the instrument; when it is left out it
is true if any of its ``tools`` is high risk by the safety policy. A request
that only asks about data (``read_only_request``) never gets a skill that acts.
``matched: false`` keeps a skill out of request matching: it is reached only through another skill that
names it (a run another skill sends the agent to). ``run`` lists a fixed procedure's tool calls, which ``run_skill`` sends one
after another in the background (see ``asyncroscopy.agents.skill_runs``).

Agents do not have to look skills up: ``select_skills`` matches a request's
words against each skill's keywords, name and description (no model call,
so the same request always gets the same skills), and ``skills_text`` writes
the chosen ones into a prompt within a size budget. The agent graphs do this
for every request and again as the task develops (see
``asyncroscopy.agents.skill_middleware``); ``load_skill`` remains for an
agent whose matched skill did not fit, and for MCP clients.

Skills come from this package's folders and from any directory listed in
``$ASYNCROSCOPY_SKILLS_PATH`` (``os.pathsep``-separated), so a lab can keep
its own skills outside the repository; an outside skill replaces a packaged
one with the same name. Skills agents learned from past runs
(``<agent memory>/skills``, see ``asyncroscopy.agents.learning``) are added
only once an operator has promoted them, never replace a packaged or lab
skill of the same name, and are usable only by an agent offered every tool
they list. A skill whose minimum versions are newer than what is
installed, or whose ``requires`` tools (glob patterns) the endpoint does not
offer, is listed as unavailable rather than loaded: an indenter agent is not
offered the STEM mapping skill.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from asyncroscopy.contracts import ENVELOPE_VERSION
from asyncroscopy.safety.policy import tool_risk

HERE = Path(__file__).resolve().parent


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    version: str = "0.0.0"
    tools: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)  # phrases that select it for a request
    min_asyncroscopy: str | None = None
    envelope_schema: str | None = None
    problem: str | None = None  # why it cannot be used here, if it cannot
    learned: bool = False  # drafted by an agent from a past run (promoted by an operator)
    status: str | None = None  # a learned skill's review state: draft or promoted
    acts: bool = False  # following it moves or exposes the instrument
    run: dict | None = None  # its fixed list of tool calls, for run_skill
    matched: bool = True  # chosen for a request by its words (false: only another skill sends the agent to it)

    @property
    def available(self) -> bool:
        return self.problem is None


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def _installed_version() -> str | None:
    """Installed asyncroscopy version; from pyproject.toml when running from a checkout; None if unknown."""
    try:
        return version("asyncroscopy")
    except PackageNotFoundError:
        pass
    pyproject = HERE.parents[1] / "pyproject.toml"
    try:
        import tomllib

        return str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"])
    except (OSError, KeyError, ValueError):
        return None


def read_front_matter(path: Path) -> tuple[dict, str]:
    """(front matter, Markdown body) of a SKILL.md; ValueError if it has no front matter."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path} has no front matter")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path} front matter is not a mapping")
    return meta, re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()  # notes to the skill's author never reach a prompt


def _parse(path: Path) -> Skill:
    meta, _body = read_front_matter(path)
    for key in ("name", "description"):
        if not meta.get(key):
            raise ValueError(f"{path} front matter is missing '{key}'")
    skill = Skill(
        name=str(meta["name"]),
        description=" ".join(str(meta["description"]).split()),
        path=path,
        version=str(meta.get("version", "0.0.0")),
        tools=list(meta.get("tools") or []),
        requires=list(meta.get("requires") or []),
        keywords=[str(k) for k in meta.get("keywords") or []],
        min_asyncroscopy=meta.get("min_asyncroscopy"),
        envelope_schema=meta.get("envelope_schema"),
        status=meta.get("status"),
        matched=bool(meta.get("matched", True)),
    )
    if meta.get("run") is not None:
        run = meta["run"]
        steps = run.get("steps") if isinstance(run, dict) else None
        if not isinstance(steps, list) or not steps or not all(isinstance(s, dict) and bool(s.get("tool")) != bool(s.get("skill")) for s in steps):
            raise ValueError(f"{path} run needs steps: a list of {{tool, args}} or {{skill, inputs}}")
        skill.run = run
    tools = skill.tools
    skill.acts = bool(meta["acts"]) if "acts" in meta else any(tool_risk(str(t)) == "high" for t in tools)
    installed = _installed_version()
    if skill.min_asyncroscopy and installed and _version_tuple(installed) < _version_tuple(str(skill.min_asyncroscopy)):
        skill.problem = f"needs asyncroscopy >= {skill.min_asyncroscopy}, installed {installed}"
    elif skill.envelope_schema and _version_tuple(str(skill.envelope_schema))[:1] != _version_tuple(ENVELOPE_VERSION)[:1]:
        skill.problem = f"written for envelope schema {skill.envelope_schema}, installed {ENVELOPE_VERSION}"
    return skill


def skill_dirs() -> list[Path]:
    extra = [Path(p) for p in os.environ.get("ASYNCROSCOPY_SKILLS_PATH", "").split(os.pathsep) if p]
    return [HERE, *extra]


def learned_dir() -> Path:
    """Where agents' learned skills are kept: ``<$ASYNCROSCOPY_AGENT_MEMORY or outputs/agent_memory>/skills``."""
    return Path(os.environ.get("ASYNCROSCOPY_AGENT_MEMORY") or "outputs/agent_memory") / "skills"


def list_skills(tool_names: set[str] | None = None) -> list[Skill]:
    """Every skill found; with ``tool_names``, those whose ``requires`` are not all offered are unavailable."""
    skills: dict[str, Skill] = {}
    for base in skill_dirs():
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/SKILL.md")):
            try:
                skill = _parse(path)
            except (ValueError, yaml.YAMLError) as exc:
                skill = Skill(name=path.parent.name, description="(unreadable skill file)", path=path, problem=str(exc))
            skills[skill.name] = skill  # later folders (a lab's own) replace same-named packaged skills
    learned = learned_dir()
    if learned.is_dir():
        for path in sorted(learned.glob("*/SKILL.md")):
            try:
                skill = _parse(path)
            except (ValueError, yaml.YAMLError):
                continue
            if skill.status == "promoted" and skill.name not in skills:  # drafts are never offered; shipped skills win
                skill.learned = True
                skills[skill.name] = skill
    if tool_names is not None:
        for skill in skills.values():
            missing = [p for p in skill.requires if not any(fnmatch.fnmatchcase(t, p) for t in tool_names)]
            if missing and skill.problem is None:
                skill.problem = f"needs tools this endpoint does not offer: {', '.join(missing)}"
    return sorted(skills.values(), key=lambda s: s.name)


def load_skill(name: str, tool_names: set[str] | None = None) -> str:
    """Full instructions of one skill. Raises KeyError if unknown, RuntimeError if not usable here."""
    for skill in list_skills(tool_names):
        if skill.name == name:
            if not skill.available:
                raise RuntimeError(f"skill {name!r} cannot be used: {skill.problem}")
            return read_front_matter(skill.path)[1]
    raise KeyError(f"no skill named {name!r}; available: {[s.name for s in list_skills()]}")


# ---------------------------------------------------------------------- matching a request
STOP = frozenset(
    "a an and are as at be by do for from how i in into is it its me my of on or our please so that the their them then this to "
    "up us use using we what when where which with you your can could would should will".split()
)
MAX_MATCHED = 2  # skills chosen from the request; a small model follows one or two well
PROMPT_BUDGET_CHARS = 7000  # about 1800 tokens of skill text: inside an 8k context with the tools
MATCH_THRESHOLD = 2.0  # one keyword or name word, or four description words


def _stem(word: str) -> str:
    """Crude English stem: "maps", "mapping", "mapped" -> "map"; "failing", "failed" -> "fail"."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if suffix in ("ing", "ed") and len(word) >= 4 and word[-1] == word[-2] and word[-1] not in "ls":
                word = word[:-1]
            break
    return word


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z][a-z0-9]+", text.lower()) if w not in STOP}


# The negated word and the one after it ("without EDS spectra"): both name what is not wanted.
NEGATED_WORD = re.compile(r"\b(?:without|no|not|don't|do not|skip)\s+(?:any\s+|an?\s+|the\s+)?[\w-]+(?:\s+[\w-]+)?", re.I)


def match_score(skill: Skill, request: str) -> float:
    """How well a skill matches a request: a keyword phrase counts 3, a word of its name 2, of its description 0.5.
    Words the request negates ("without EDS", "no spectra") do not count; a skill with ``matched: false`` scores 0."""
    if not skill.matched:
        return 0.0
    asked = content_words(NEGATED_WORD.sub(" ", request))
    hits = sum(3.0 for k in skill.keywords if content_words(k) and content_words(k) <= asked)
    hits += 2.0 * len(asked & content_words(skill.name.replace("_", " ")))
    hits += 0.5 * len(asked & content_words(skill.description))
    return hits


def usable_by(skill: Skill, agent_tools: set[str]) -> bool:
    """Usable on the endpoint and, when it lists tools, the agent holds at least one of them.

    A learned skill needs every tool it lists: it can never lead an agent to a tool it was not given.
    """
    if not skill.available:
        return False
    if skill.learned:
        return all(any(fnmatch.fnmatchcase(t, p) for t in agent_tools) for p in skill.tools)
    return not skill.tools or any(fnmatch.fnmatchcase(t, p) for p in skill.tools for t in agent_tools)


# A request acts on the instrument unless it is clearly read-only. The costs are not equal: taking an
# acting request for read-only keeps the skills it needs away from the agent, while taking a read-only
# request for acting is still held by the risk policy and the approval gate.
READ_CUE = re.compile(
    r"\?\s*$|^\s*(?:is|are|was|were|does|do|did|where|when|why|who|show|list|count|read)\b"
    r"|\b(?:what|which|how many|how much|is there|are there|tell me|show me|describe|list the|list all|look at|summari[sz]e|compare|explain)\b",
    re.I | re.M,
)
# An instruction to act, anywhere in the request (past tenses such as "acquired" describe data, not actions).
ACTION = re.compile(
    r"\b(?:take|taking|acquire|acquiring|capture|capturing|measure|measuring|record|recording|collect|collecting|"
    r"move|moving|go|going|goto|drive|revisit|start|starting|resume|perform|expose|exposing|(?:un)?blank|"
    r"imaging|scanning|surveying|mapping|refocus|autofocus|focus(?:ing)? on|higher magnification|high magnification|"
    r"(?:map|image|photograph|scan|survey|record)\s+(?:a|an|the|this|that|it|them|each|every|around|here|out|over|\d)|"
    r"(?:do|run|make|get|give me|acquire)\s+(?:a|an|another|one|some)\s+(?:[\w-]+\s+){0,3}?(?:map|image|spectrum|spectra|scan|survey|measurement)s?|"
    r"(?:new|another|fresh)\s+(?:area|image|map|spectrum|region|measurement|scan|data))\b",
    re.I,
)
# An explicit instruction not to act ("do not acquire anything", "without moving", "no new data").
NO_ACTION = re.compile(
    r"\b(?:do not|don't|dont|never|without|no need to)\s+(?:\w+\s+){0,2}?"
    r"(?:acquir\w*|tak\w*|mov\w*|scan\w*|map\w*|measur\w*|imag\w*|expos\w*|touch\w*|start\w*|record\w*|collect\w*)(?:\s+anything)?"
    r"|\bno new (?:data|acquisitions?|measurements?|images?|maps?)\b",
    re.I,
)
# A reference to data that already exist: a key, "the latest", "the stitched map", "stored" ...
EXISTING = re.compile(
    r"\b(?:latest|last|most recent(?:ly)?|recently|stored|existing|earlier|previous|already|acquired|recorded|saved)\b"
    r"|\b[\w\-.]+\.(?:h5|hdf5|hdf|nxs|tiff?)\b|\b\w*\d\w*_map\b|\b(?:the|this|that)\s+(?:stitched\s+)?map\b",
    re.I,
)


def read_only_request(request: str) -> bool:
    """True only when a request clearly does not act: it says not to ("do not acquire anything"), or it
    is a question about data that already exist ("the latest spectrum", a key, "the stitched map") with
    no instruction to act anywhere in it. Everything else counts as acting."""
    if ACTION.search(NO_ACTION.sub(" ", request)):
        return False
    if NO_ACTION.search(request):
        return True
    return bool(READ_CUE.search(request)) and bool(EXISTING.search(request))


def select_skills(request: str, agent_tools: set[str], endpoint_tools: set[str] | None = None, limit: int = MAX_MATCHED) -> list[Skill]:
    """The skills matching ``request`` for an agent holding ``agent_tools``, best first.

    ``requires`` is checked against ``endpoint_tools`` (the endpoint's full tool list; default
    ``agent_tools``), so an agent holding part of an endpoint still gets a skill whose other
    steps another agent does. A read-only request never gets a skill that acts.
    """
    read_only = read_only_request(request)
    skills = [s for s in list_skills(endpoint_tools or agent_tools) if usable_by(s, agent_tools) and not (read_only and s.acts)]
    ranked = sorted(skills, key=lambda s: match_score(s, request), reverse=True)
    return [s for s in ranked if match_score(s, request) >= MATCH_THRESHOLD][:limit]


def skills_text(names: list[str], endpoint_tools: set[str] | None = None, budget: int = PROMPT_BUDGET_CHARS) -> tuple[str, list[str]]:
    """The named skills written out for a prompt, in order, while they fit in ``budget`` characters.

    Returns (text, names cut short): a skill that does not fit is listed by its description
    with "call load_skill", so the agent needs load_skill only then.
    """
    parts, cut, used = [], [], 0
    known = {s.name: s for s in list_skills(endpoint_tools)}
    for name in names:
        skill = known.get(name)
        if skill is None or not skill.available:
            continue
        body = read_front_matter(skill.path)[1]
        if used + len(body) <= budget:
            # Not "skill: <name>": qwen3:8b called the name as a tool, which Ollama drops, so the run got an empty reply.
            parts.append(f"=== instructions (not a tool) ===\n{body}")
            used += len(body)
        else:
            parts.append(f"=== skill: {name} (call load_skill for its instructions) ===\n{skill.description}")
            cut.append(name)
    return "\n\n".join(parts), cut
