"""Which of an endpoint's tools a tool-using model is given.

One rule set for every client: the general agent (``asyncroscopy.agents.graphs.general``)
builds its tool list with it, and the agent MCP endpoint reports the same globs as
``endpoint_info.agent_tools``, so another harness (e.g. SciAgentGUI) gives a small model the
same tools. The lists themselves are model-specific data in an agent profile
(``asyncroscopy/agents/profiles/*.yaml``, its ``tools`` section); this module holds only the rules.
It reads the profile file as data and imports nothing from ``asyncroscopy.agents`` (the MCP server uses it).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

# The packaged agent profiles (data files; see asyncroscopy.agents.profile).
PROFILES = Path(__file__).resolve().parents[1] / "agents" / "profiles"
DEFAULT_PROFILE = PROFILES / "default.yaml"


@dataclass(frozen=True)
class ToolLists:
    """A profile's tool selection: globs for a STEM, the generic job and map tools, the analysis server's
    waits (only when the endpoint cannot wait on analysis jobs itself), and tools never given to a model."""

    stem: tuple[str, ...] = ()
    generic: tuple[str, ...] = ()
    analysis_waits: tuple[str, ...] = ()
    not_for_models: tuple[str, ...] = field(default=("list_skills", "endpoint_info", "action_space"))


def tool_lists(path: str | Path | None = None) -> ToolLists:
    """The ``tools`` section of an agent profile file (the packaged default profile without ``path``)."""
    return _tool_lists(str(Path(path or DEFAULT_PROFILE).resolve()))


@cache
def _tool_lists(path: str) -> ToolLists:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tools = raw.get("tools") or {}
    if not isinstance(tools, dict) or not all(isinstance(v, list) for v in tools.values()):
        raise ValueError(f"{path}: tools must map stem, generic, analysis_waits and not_for_models to lists of globs")
    unknown = set(tools) - {"stem", "generic", "analysis_waits", "not_for_models"}
    if unknown:
        raise ValueError(f"{path}: unknown tools lists {sorted(unknown)}")
    return ToolLists(**{k: tuple(str(p) for p in v) for k, v in tools.items()})


def _matches(name: str, pattern: str) -> bool:
    """fnmatch ignoring case on every OS, as the approval policy and SciAgentGUI match (fnmatch.fnmatch follows the OS)."""
    return fnmatch.fnmatchcase(name.lower(), pattern.lower())


def agent_tool_patterns(names: Iterable[str], has_analysis_url: bool, chosen: list[str] | None = None, lists: ToolLists | None = None,
                        *, stem: bool = False, maps: bool = False) -> list[str]:
    """Tool-name globs a model is given, for an instrument endpoint offering ``names``.

    ``names``: the instrument endpoint's own tools (not the analysis server's). ``stem``: the endpoint has
    an electron microscope, ``maps``: a Mapper (endpoint_info's stem_devices and map_devices, as discovery
    found them). ``chosen``: globs a caller picked (ASYNCROSCOPY_AGENT_TOOLS); otherwise the profile's STEM
    set on a STEM (a Mapper alone does not make one: it maps an AFM or a YAML twin too, and the STEM set
    would hide that instrument's own tools; with the analysis server's waits when the endpoint has no
    analysis server to wait through), else every tool of the endpoint but its State/Status attributes,
    plus the generic job and map tools when it has a Mapper. ``lists``: the profile's tool lists (the
    default profile's when None). load_skill is always included, the profile's not_for_models never.
    """
    lists = lists or tool_lists()
    names = set(names)
    waits = [] if has_analysis_url else list(lists.analysis_waits)
    if chosen:
        patterns = list(chosen)
    elif stem:
        patterns = list(lists.stem) + list(lists.generic) + waits
    else:
        patterns = sorted(n for n in names if not n.endswith(("_State", "_Status")))
        if maps:
            patterns += [p for p in list(lists.generic) + waits if p not in patterns]
    patterns = [p for p in patterns if p not in lists.not_for_models]
    return patterns if "load_skill" in patterns else patterns + ["load_skill"]


def for_model(name: str, patterns: Iterable[str], lists: ToolLists | None = None) -> bool:
    """Whether a tool named ``name`` goes to the model under ``patterns`` (never one of the profile's not_for_models)."""
    hidden = (lists or tool_lists()).not_for_models
    return name not in hidden and any(_matches(name, p) for p in patterns)
