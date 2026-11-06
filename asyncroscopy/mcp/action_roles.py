"""Canonical roles: what a tool does, in words that do not depend on what a twin calls it.

Two instruments name the same action differently (a STEM's ``move_stage``, an AFM file's
``move_xy``). Each tool in an endpoint's action space (``asyncroscopy.mcp.action_space``) carries
one role from ``asyncroscopy.core.names.ROLES``, so a controller written against roles runs on both. A tool's role comes from,
in order:

1. an annotation: a device's ``action_hints`` attribute (a YAML instrument's ``role:`` key, or a Python
   device's ``asyncroscopy.utils.commands.command(role=...)``);
2. the rules in ``role_for`` below, on the command's words (the safety policy's word lists);
3. ``other``.

The vocabulary is kept short on purpose: add a role only when a controller needs to tell two
tools apart that the existing roles lump together.
"""

from __future__ import annotations

from asyncroscopy.core.names import ENVELOPE_VERBS
from asyncroscopy.safety.policy import OFF_OBJECTS, PROBE_WORDS, RETREAT_VERBS, _word_list, is_read_only, is_safe_stop

# What switching off stops the beam reaching the sample, without the hours a gun, filament, high tension
# or emission source takes to come back: their "off" is a safe stop, not a beam blank (stop_all never calls it).
_BEAM_OFF = OFF_OBJECTS - {"gun", "filament", "hv", "emission", "source"}
_JOB_ROLES = {"reserve": "job.reserve", "start": "job.start", "job_status": "job.status", "abort": "job.abort",
              "results": "job.results", "release": "job.release", "get_reservation": "job.status"}
_STAGE = frozenset({"stage", "xy", "xyz", "position", "xy_m"})
_MOVE = frozenset({"move", "goto", "go", "jog", "translate"})
_IMAGE = frozenset({"image", "images", "topography", "camera", "frame", "micrograph", "brightfield"})
_SPECTRUM = frozenset({"spectrum", "spectra", "curve", "diffractogram", "pattern", "profile", "eels", "eds", "edx"})
_FOV = frozenset({"fov", "magnification", "mag", "zoom"})
_FOCUS = frozenset({"focus", "defocus", "autofocus"})


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
        if "blank" in w or {"close", "shutter"} <= w or (words and words[-1] == "off" and w & _BEAM_OFF):
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
