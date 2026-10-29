"""Canonical roles: what a tool does, in words that do not depend on what a twin calls it.

Two instruments name the same action differently (a STEM's ``move_stage``, an AFM file's
``move_xy``). Each tool in an endpoint's action space (``asyncroscopy.mcp.action_space``) carries
one role from ROLES, so a controller written against roles runs on both. A tool's role comes from,
in order:

1. an annotation: a device's ``action_hints`` attribute (a YAML instrument's ``role:`` key);
2. the rules in ``role_for`` below, on the command's words (the safety policy's word lists);
3. ``other``.

The vocabulary is kept short on purpose: add a role only when a controller needs to tell two
tools apart that the existing roles lump together.
"""

from __future__ import annotations

from asyncroscopy.safety.policy import ENVELOPE_VERBS, OFF_OBJECTS, PROBE_WORDS, RETREAT_VERBS, _word_list, is_read_only, is_safe_stop

# The Tango attribute through which any device annotates its commands, as JSON
# {command: {"role": ..., "args": [{"name", "type", "units", "min", "max", "enum", "axis", "elements"}], "returns": {...}}}.
# A YAML instrument has one (from its file); a Python device may add one.
ACTION_HINTS = "action_hints"

# role -> what a tool with that role does (shown in the action space and the docs).
ROLES: dict[str, str] = {
    "stage.move": "move the sample stage (all axes, or one)",
    "stage.read": "read the stage position",
    "beam.blank": "stop the probe beam (electrons, laser, X-rays) reaching the sample",
    "beam.unblank": "let the probe beam reach the sample",
    "beam.place": "park the beam on one point of the field of view",
    "optics.focus": "set the focus or defocus, or run an autofocus",
    "optics.fov": "set the field of view (scan size, magnification)",
    "acquire.image": "acquire a 2-D image; returns a data key",
    "acquire.spectrum": "acquire a 1-D spectrum or curve; returns a data key",
    "acquire.other": "acquire anything else (a value, a 4-D dataset)",
    "probe.approach": "bring a probe (tip, indenter) onto the sample",
    "probe.retract": "lift a probe off the sample",
    "job.run": "run a whole job in one call (reserve, start, wait)",
    "job.reserve": "check a job plan and hold the device",
    "job.start": "start a reserved job",
    "job.status": "read a job's state, or wait for it",
    "job.abort": "stop a running job, leaving the instrument safe",
    "job.results": "read a finished job's results",
    "job.release": "free a reserved job device",
    "safe_stop": "stop or switch off something (always low risk)",
    "setting.read": "read a setting or the instrument's status",
    "setting.set": "change a setting",
    "data.read": "read stored data (by key)",
    "other": "anything the rules do not place",
}

# Native tools of the MCP server (asyncroscopy.mcp.mcp_server), which have no device behind them.
NATIVE_ROLES: dict[str, str] = {
    "run_job": "job.run",
    "wait_for_job": "job.status",
    "list_data": "data.read",
    "view_data": "data.read",
}

_JOB_ROLES = {"reserve": "job.reserve", "start": "job.start", "job_status": "job.status", "abort": "job.abort",
              "results": "job.results", "release": "job.release", "get_reservation": "job.status"}
_STAGE = frozenset({"stage", "xy", "xyz", "position", "xy_m"})
_MOVE = frozenset({"move", "goto", "go", "jog", "translate"})
_IMAGE = frozenset({"image", "images", "topography", "camera", "frame", "micrograph", "brightfield"})
_SPECTRUM = frozenset({"spectrum", "spectra", "curve", "diffractogram", "pattern", "profile", "eels", "eds", "edx"})
_FOV = frozenset({"fov", "magnification", "mag", "zoom"})
_FOCUS = frozenset({"focus", "defocus", "autofocus"})


def check_role(role: str, where: str = "role") -> str:
    """``role`` if it is in the vocabulary; otherwise ValueError naming the roles there are."""
    if role not in ROLES:
        raise ValueError(f"{where}: {role!r} is not a role; roles: {', '.join(ROLES)}")
    return role


def role_for(dev_class: str, command: str, *, envelope: bool = False, probe: bool = False) -> str:
    """The role the rules give ``command`` (no annotation): job verbs on a job device, stops, then words.

    ``envelope``: the device runs jobs (``policy.is_envelope_device``); ``probe``: it brings a probe to
    the sample, so a bare retract lifts it (``policy.is_probe_device``).
    """
    name = command.lower()
    if envelope and (name in ENVELOPE_VERBS or name == "get_reservation"):
        return _JOB_ROLES[name]
    words = _word_list(command)
    w = set(words)
    if "unblank" in w:
        return "beam.unblank"
    if is_safe_stop(command, probe=probe):
        if "blank" in w or (words and words[-1] == "off" and w & OFF_OBJECTS):
            return "beam.blank"
        if w & RETREAT_VERBS:
            return "probe.retract"
        return "safe_stop"
    reads = is_read_only(dev_class, command)
    if reads:
        if w & _STAGE:
            return "stage.read"
        if "data" in w and "image" in w:  # get_image_data_cached
            return "data.read"
        return "setting.read"
    if words and words[0].startswith("acquire"):
        if w & _SPECTRUM:
            return "acquire.spectrum"
        if w & _IMAGE:
            return "acquire.image"
        return "acquire.other"
    if "place" in w and "beam" in w:
        return "beam.place"
    if w & PROBE_WORDS - {"indent"} and not w & {"abort", "stop"}:
        return "probe.approach"
    if w & _MOVE and (w & _STAGE or len(w) == 1):
        return "stage.move"
    if w & _FOCUS:
        return "optics.focus"
    if w & _FOV or ("scan" in w and "size" in w):
        return "optics.fov"
    if words and words[0] == "set":
        return "stage.move" if "stage" in w else "setting.set"
    return "other"
