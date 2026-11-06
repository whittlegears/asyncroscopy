"""Agent profiles: the model-specific settings of the agent graphs, kept as data (``agents/profiles/*.yaml``).

A profile names the default model, its provider and context window, the middleware the graphs wrap the
model in, which tools a model is given, and which tools each specialist of the team graph holds. ``default`` is tuned for qwen3:8b. Choose another with
ASYNCROSCOPY_AGENT_PROFILE (a name in agents/profiles, or a path to a YAML file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from asyncroscopy.mcp.agent_tools import PROFILES, ToolLists, tool_lists

MIDDLEWARE = ("skills", "job_wait", "empty_reply", "claim_check")


@dataclass(frozen=True)
class TeamTools:
    """The team graph's tool split (a profile's ``team``): tool-name globs for each specialist, the globs both
    get, and those neither gets. The first match decides, hidden then shared then the roles in order."""

    roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    shared: tuple[str, ...] = ()
    hidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentProfile:
    name: str
    model: str
    model_provider: str = "ollama"
    num_ctx: int = 8192
    middleware: tuple[str, ...] = MIDDLEWARE
    tools: ToolLists = field(default_factory=ToolLists)
    team: TeamTools = field(default_factory=TeamTools)


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


def profile_path(name_or_path: str = "default") -> Path:
    """The YAML file of a profile: a name in agents/profiles, or a path."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml"):
        return path
    return PROFILES / f"{name_or_path}.yaml"


@cache
def load_profile(name_or_path: str = "default") -> AgentProfile:
    """A profile, checked: a typo must not quietly drop the claim check or hand a model every tool."""
    path = profile_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not raw.get("model"):
        raise ValueError(f"{path}: an agent profile needs at least a model")
    middleware = tuple(raw.get("middleware", MIDDLEWARE))
    unknown = [m for m in middleware if m not in MIDDLEWARE]
    if unknown:
        raise ValueError(f"{path}: unknown middleware {unknown}; known: {list(MIDDLEWARE)}")
    return AgentProfile(
        name=path.stem, model=str(raw["model"]), model_provider=str(raw.get("model_provider", "ollama")),
        num_ctx=int(raw.get("num_ctx", 8192)), middleware=middleware, tools=tool_lists(path), team=_team(raw.get("team"), path),
    )
