"""Agent profiles: the model-specific settings of the agent graphs, kept as data (``agents/profiles/*.yaml``).

Everything that depends on how capable the model is lives in a profile, not in code: the default model, its
provider and context window, the middleware the graphs wrap the model in, which tools a model is given (and
which tools each specialist of the team graph holds), how much skill text and memory a prompt carries, which
tools are held back unless the request asks for them, and whether the servers' structured replies are shown or
replaced by sentences.

``small`` is tuned for a small local model (qwen3:8b, 8k context). ``capable`` is for any model that follows
tool descriptions and longer instructions well: it extends ``small`` and turns the small-model aids off.
Choose one with ASYNCROSCOPY_AGENT_PROFILE (a name in agents/profiles, or a path to a YAML file); without it,
``capable`` (gemma4:31b).
A profile may start from another with ``extends: <name>`` and change only what differs.

In REPLIES, notes replaces a reply's structured fields with sentences; facts_and_notes keeps both.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from asyncroscopy.mcp.agent_tools import DEFAULT_PROFILE_NAME, PROFILES, ToolLists, profile_data, tool_lists

MIDDLEWARE = ("skills", "user_words", "job_wait", "empty_reply", "claim_check")
REPLIES = ("notes", "facts_and_notes")


@dataclass(frozen=True)
class TeamTools:
    """The team graph's tool split (a profile's ``team``): tool-name globs for each specialist, the globs both
    get, and those neither gets. The first match decides, hidden then shared then the roles in order."""

    roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    shared: tuple[str, ...] = ()
    hidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillLimits:
    """How much a prompt carries (a profile's ``skills``).

    ``from_request``: skills chosen from the request's words. ``at_most``: skills one run may carry, those that
    events add (a failed call, a held call, ...) included. ``prompt_chars``: the most skill text in a prompt; it is
    also kept under a quarter of the context. ``notes_chars``: the most memory notes in a prompt.
    ``passages``: skill passages found for each step (``asyncroscopy.skills.index``), beyond the skills in whole;
    ``passage_chars``: the most text they add, within ``prompt_chars``.
    """

    from_request: int = 2
    at_most: int = 4
    prompt_chars: int = 7000
    notes_chars: int = 1500
    passages: int = 3
    passage_chars: int = 2000


def _cue(text: str) -> re.Pattern:
    """A request cue as a pattern: ``re:`` a regular expression, else whole words ignoring case, ``*`` a prefix."""
    if text.startswith("re:"):
        return re.compile(text[3:], re.I)
    words = [re.escape(w[:-1]) + r"\w*" if w.endswith("*") else re.escape(w) for w in text.split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"\b", re.I)


@dataclass(frozen=True)
class ToolGate:
    """Tools held back from the model unless the request asks for what they do (one entry of ``tool_gates``).

    A small model calls a tool it can see even when the request does not need it (it annotates a map it was only
    asked to describe), so its profile holds such tools back. ``offer_if``: the tools are offered only when the
    request has one of these cues. ``hide_if``: they are held back when it has one. A held-back tool is still on
    the endpoint; it is only left out of this model call.
    """

    tools: tuple[str, ...]
    offer_if: tuple[str, ...] = ()
    hide_if: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_offer", tuple(_cue(c) for c in self.offer_if))
        object.__setattr__(self, "_hide", tuple(_cue(c) for c in self.hide_if))

    def holds_back(self, request: str) -> bool:
        if self._offer and not any(p.search(request) for p in self._offer):
            return True
        return any(p.search(request) for p in self._hide)

    def covers(self, tool: str) -> bool:
        return any(fnmatch.fnmatchcase(tool, p) for p in self.tools)


def held_back(gates: tuple[ToolGate, ...], request: str, tools: list[str]) -> set[str]:
    """The tools of ``tools`` that ``gates`` hold back for ``request``."""
    closed = [g for g in gates if g.holds_back(request)]
    return {t for t in tools if any(g.covers(t) for g in closed)}


@dataclass(frozen=True)
class AgentProfile:
    name: str
    model: str
    model_provider: str = "ollama"
    num_ctx: int = 8192
    middleware: tuple[str, ...] = MIDDLEWARE
    tools: ToolLists = field(default_factory=ToolLists)
    team: TeamTools = field(default_factory=TeamTools)
    skills: SkillLimits = field(default_factory=SkillLimits)
    tool_gates: tuple[ToolGate, ...] = ()
    replies: str = "notes"
    images: bool = False


def _team(raw: object, path: Path) -> TeamTools:
    """A profile's ``team`` section, checked: roles maps each specialist to globs, shared and hidden are globs."""
    if raw is None:
        return TeamTools()
    globs = lambda v: isinstance(v, list) and all(isinstance(p, str) and p for p in v)  # noqa: E731
    roles = raw.get("roles") if isinstance(raw, dict) else None
    if not isinstance(raw, dict) or set(raw) - {"roles", "shared", "hidden"} or not isinstance(roles, dict) \
            or not all(globs(v) for v in roles.values()) or not all(globs(raw.get(k, [])) for k in ("shared", "hidden")):
        raise ValueError(f"{path}: team must have roles (each a list of tool-name globs), and shared and hidden lists of globs")
    return TeamTools(roles={str(k): tuple(v) for k, v in roles.items()}, shared=tuple(raw.get("shared", [])), hidden=tuple(raw.get("hidden", [])))


def _skills(raw: object, path: Path) -> SkillLimits:
    """A profile's ``skills`` section, checked: whole numbers above 0 for the keys SkillLimits has."""
    if raw is None:
        return SkillLimits()
    known = set(SkillLimits.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) - known or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in raw.values()):
        raise ValueError(f"{path}: skills takes whole numbers above 0 for {sorted(known)}")
    return SkillLimits(**raw)


def _gates(raw: object, path: Path) -> tuple[ToolGate, ...]:
    """A profile's ``tool_gates``, checked: each has tools and offer_if or hide_if, all lists of text."""
    if raw is None:
        return ()
    texts = lambda v: isinstance(v, list) and all(isinstance(p, str) and p.strip() for p in v)  # noqa: E731
    gates = []
    for entry in raw if isinstance(raw, list) else [None]:
        ok = isinstance(entry, dict) and not set(entry) - {"tools", "offer_if", "hide_if"} and texts(entry.get("tools")) \
            and ("offer_if" in entry or "hide_if" in entry) and all(texts(entry.get(k, [])) for k in ("offer_if", "hide_if"))
        if not ok:
            raise ValueError(f"{path}: each tool_gates entry has tools (globs) and offer_if or hide_if (lists of cues)")
        try:
            gates.append(ToolGate(tuple(entry["tools"]), tuple(entry.get("offer_if", [])), tuple(entry.get("hide_if", []))))
        except re.error as exc:
            raise ValueError(f"{path}: a tool_gates cue is not a valid pattern: {exc}") from exc
    return tuple(gates)


def profile_path(name_or_path: str = DEFAULT_PROFILE_NAME) -> Path:
    """The YAML file of a profile: a name in agents/profiles, or a path."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml"):
        return path
    return PROFILES / f"{name_or_path}.yaml"


@cache
def load_profile(name_or_path: str = DEFAULT_PROFILE_NAME) -> AgentProfile:
    """A profile, checked: a typo must not quietly drop the claim check or hand a model every tool."""
    path = profile_path(name_or_path)
    raw = profile_data(path)
    if not raw.get("model"):
        raise ValueError(f"{path}: an agent profile needs at least a model")
    middleware = tuple(raw.get("middleware", MIDDLEWARE))
    unknown = [m for m in middleware if m not in MIDDLEWARE]
    if unknown:
        raise ValueError(f"{path}: unknown middleware {unknown}; known: {list(MIDDLEWARE)}")
    replies = str(raw.get("replies", "notes"))
    if replies not in REPLIES:
        raise ValueError(f"{path}: replies is one of {list(REPLIES)}, not {replies!r}")
    return AgentProfile(
        name=path.stem, model=str(raw["model"]), model_provider=str(raw.get("model_provider", "ollama")),
        num_ctx=int(raw.get("num_ctx", 8192)), middleware=middleware, tools=tool_lists(path), team=_team(raw.get("team"), path),
        skills=_skills(raw.get("skills"), path), tool_gates=_gates(raw.get("tool_gates"), path), replies=replies,
        images=bool(raw.get("images", False)),
    )
