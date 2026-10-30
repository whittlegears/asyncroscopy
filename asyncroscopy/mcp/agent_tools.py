"""Which of an endpoint's tools a tool-using model is given.

One rule set for every client: the general agent (``asyncroscopy.agents.graphs.general``)
builds its tool list with it, and the agent MCP endpoint reports the same globs as
``endpoint_info.agent_tools``, so another harness (e.g. SciAgentGUI) gives a small model the
same tools. No LangChain imports here: the MCP server imports this module.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable

# Tools every instrument endpoint and the pyTEMlib analysis server offer. Left out, from the trial
# traces (benchmarks/results/tool_usage_analysis.md): calls that always come as a pair or are always
# made first. run_job is reserve+start, wait_for_job also waits on analysis jobs and returns results,
# and the skills are put in the prompt (load_skill is shown to the model only while one does not fit).
# find_tools and call_tool reach every other tool of the endpoint, so a curated list never hides one.
GENERIC_TOOLS = [
    "find_tools", "call_tool", "approval_status", "run_job", "wait_for_job", "list_devices", "list_data", "view_data",
    "assemble_map", "score_map", "map_info", "get_region", "list_pois", "add_poi", "list_scorers",
    "annotate", "list_annotations", "edit_annotation", "to_stage", "to_pixel",
]
# A curated STEM set: a 4-8 B model loses track with all ~80 microscope tools.
STEM_TOOLS = [
    "*_get_stage", "*_get_sample", "*_move_stage", "*_get_parameters", "*_acquire_scanned_image", "*_acquire_spectrum",
    "*_set_fov", "*_get_fov", "*_set_defocus", "*_place_beam", "*_blank_beam", "*_unblank_beam", "Mapper*_abort",
]
DEFAULT_TOOLS = STEM_TOOLS + GENERIC_TOOLS
# Only when the instrument endpoint has no mcp.analysis_url (so wait_for_job cannot wait on stitching
# and scoring jobs) does an agent need the analysis server's own wait and results tools.
ANALYSIS_WAITS = ["analysis_job_status", "analysis_job_results"]
# For agent programs: skills come to the agent in its prompt, and endpoint_info is not a model's business;
# action_space is a long document a controller program reads once, far too big for a small model's context.
NOT_FOR_MODELS = ("list_skills", "endpoint_info", "action_space")


def is_stem_endpoint(names: Iterable[str]) -> bool:
    """A STEM: one device both scans images and blanks a beam (an AFM scans too), the same test as
    ``asyncroscopy.safety.policy.is_stem_device``.

    A Mapper alone does not make one: it maps any instrument (an AFM, or a YAML twin through the
    ``yaml`` profile), and the curated STEM set would hide that instrument's own tools (an AFM's
    retract_tip). move_stage, set_fov and the like are common to many instruments.
    """
    names = set(names)
    return any(
        n.endswith("_acquire_scanned_image") and f"{n[: -len('_acquire_scanned_image')]}_blank_beam" in names for n in names
    )


def has_mapper(names: Iterable[str]) -> bool:
    """The endpoint offers a Mapper (whose maps the analysis server stitches and scores)."""
    return any(fnmatch.fnmatch(n, "Mapper*_reserve") for n in names)


def agent_tool_patterns(names: Iterable[str], has_analysis_url: bool, chosen: list[str] | None = None) -> list[str]:
    """Tool-name globs a model is given, for an instrument endpoint offering ``names``.

    ``names``: the instrument endpoint's own tools (not the analysis server's). ``chosen``: globs a
    caller picked (ASYNCROSCOPY_AGENT_TOOLS); otherwise the curated STEM set on a STEM (with the
    analysis server's waits when the endpoint has no analysis server to wait through), else every
    tool of the endpoint but its State/Status attributes, plus the generic job and map tools when it
    has a Mapper (the analysis tools stitch and score its maps). load_skill is always included,
    NOT_FOR_MODELS never.
    """
    names = set(names)
    waits = [] if has_analysis_url else ANALYSIS_WAITS
    if chosen:
        patterns = list(chosen)
    elif is_stem_endpoint(names):
        patterns = DEFAULT_TOOLS + waits
    else:
        patterns = sorted(n for n in names if not n.endswith(("_State", "_Status")))
        if has_mapper(names):
            patterns += [p for p in GENERIC_TOOLS + waits if p not in patterns]
    patterns = [p for p in patterns if p not in NOT_FOR_MODELS]
    return patterns if "load_skill" in patterns else patterns + ["load_skill"]


def for_model(name: str, patterns: Iterable[str]) -> bool:
    """Whether a tool named ``name`` goes to the model under ``patterns`` (never one of NOT_FOR_MODELS)."""
    return name not in NOT_FOR_MODELS and any(fnmatch.fnmatch(name, p) for p in patterns)
