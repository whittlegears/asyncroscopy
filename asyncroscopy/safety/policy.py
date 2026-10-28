"""Which commands count as high risk, and which only an operator may call.

Risk is decided by ``Class.command`` patterns, first match wins:

1. the deployment's own patterns (the MCP config's ``safety.risk``), so a
   team can mark e.g. ``Opentrons.get_tip`` high or ``MyStage.jog`` high;
2. safe-stop commands, always low: an agent must never wait for approval to
   abort, stop, retract, switch off or close a shutter. This is an allow-list,
   like the rest of the policy: a command's name (split on "_" and CamelCase)
   has exactly one stop verb (STOP_VERBS, RETREAT_VERBS, or "off" as the last
   word) and every other word is something that verb may stop (STOP_OBJECTS;
   MOTION_OBJECTS too after abort/stop/halt/pause), or it is a stop pair
   (STOP_PAIRS) with only STOP_OBJECTS beside it. So ``Off``, ``stop_scan``,
   ``AbortScan``, ``abort_approach``, ``beam_off`` and ``shutter_close`` are
   low; ``brake_off``, ``vacuum_off``, ``stop_retract``, ``retract_approach``
   and ``estop_ack`` are not. Mark other stops low with ``mcp.safety.risk``;
3. the job-envelope verbs other than ``start``, low on devices that implement
   the envelope (the MCP server knows which those are);
4. Tango's ``State`` and ``Status``, and the read-only naming convention of
   this code base (``get_*``, ``list_*``), plus, on a device that acquires
   scanned images, a few STEM commands that neither move hardware nor expose
   the sample (STEM_LOW);
5. everything else is high: a new or renamed command is held in approval
   mode rather than waved through.

The convention in (4) is a promise instrument authors keep: a command named
``get_...`` or ``list_...`` must not move or change anything (an SPM
``get_image`` that scans the tip breaks it). Name commands that act
otherwise, or list them as high in ``safety.risk``.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase

# A safe stop is recognised by an allow-list (anything else is high, like any unknown name):
# exactly one stop verb, and every other word one of the objects that verb may take.
STOP_VERBS = frozenset({"abort", "stop", "halt", "pause", "estop"})  # may also stop a motion or a job
RETREAT_VERBS = frozenset({"retract", "withdraw", "disengage"})  # move away from the sample; take hardware objects only
STOP_OBJECTS = frozenset({
    "beam", "emission", "lamp", "light", "laser", "shutter", "gun", "filament", "hv",
    "tip", "probe", "stage", "all", "x", "y", "z", "xy", "xyz", "motor", "motors", "now",
    "indent", "indenter", "head", "objective", "imaging", "exposure", "robot", "protocol",
})
# What "off" may switch off: a source of energy. "stage_off", "z_off" or "shutter_off" are not stops
# (a de-energised shutter may open; close it with close_shutter).
OFF_OBJECTS = frozenset({"beam", "emission", "lamp", "light", "led", "illumination", "laser", "gun", "filament", "hv", "xray", "source"})
MOTION_OBJECTS = frozenset({"approach", "move", "motion", "movement", "scan", "scanning", "acquisition", "acquire", "job", "run", "travel", "focus", "autofocus", "live"})
STOP_PAIRS = (frozenset({"close", "shutter"}), frozenset({"blank", "beam"}), frozenset({"safe", "state"}))
# "emergency" qualifies a stop verb (emergency_stop) and is never one by itself.
MODIFIERS = frozenset({"emergency"})

ENVELOPE_VERBS = frozenset({"reserve", "start", "job_status", "abort", "results", "release"})
ENVELOPE_LOW = ("reserve", "job_status", "results", "release")

DEFAULT_RISK: list[tuple[str, str]] = [
    # Read-only by naming convention.
    ("*.get_*", "low"),
    ("*.list_*", "low"),
    ("*.ping", "low"),
    ("*.state", "low"),
    ("*.status", "low"),
    # Reading the approval state; the deciding commands are operator-only below.
    ("approvals.request_status", "low"),
]
# STEM settings that neither move hardware nor expose the sample. Low only on a device that
# acquires scanned images: on an optical microscope set_defocus drives the objective. place_beam
# is not here: with the beam unblanked it parks the probe on one spot.
STEM_LOW = frozenset({"set_fov", "set_defocus"})

# Never offered on the agent endpoint, whatever the config says: an agent must
# not approve its own requests, switch the mode, or read a twin's answers.
OPERATOR_ONLY: list[str] = [
    "approvals.approve",
    "approvals.deny",
    "approvals.set_mode",
    # The MCP server reaches these through Tango itself; offering them to agents would let
    # an agent consume a grant (check) or file requests under any caller name (request).
    "approvals.check",
    "approvals.request",
    "approvals.list_grants",
    "*.export_ground_truth",
    # Restarting a device resets its state (the approval mode among it).
    "*.init",
    "*.kill",
    "*.restartserver",
    # Running the data service is the operator's job; the launcher calls these over Tango.
    "data.configure",
    "data.start_tiled_server",
    "data.stop_tiled_server",
]


def _key(dev_class: str, command: str) -> str:
    return f"{dev_class}.{command}".lower()


def _word_list(name: str) -> list[str]:
    """Lower-case words of a command name: "AbortScan" and "abort_scan" both give ["abort", "scan"]."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return [w for w in re.split(r"[_\W]+", spaced.lower()) if w]


def looks_like_stop(command: str) -> bool:
    """A name with a stop word in it: if it is still high, the author probably meant it as a stop."""
    return bool(set(_word_list(command)) & (STOP_VERBS | RETREAT_VERBS | {"off", "unload", "park", "estop"}))


def is_safe_stop(command: str) -> bool:
    """Exactly one stop verb and nothing else but what it may stop (see STOP_OBJECTS), or a stop pair.

    "off" counts only as the last word (beam_off, lamp_off); "off_axis" and "kick_off" are not stops.
    """
    ordered = [w for w in _word_list(command) if w not in MODIFIERS]
    words = set(ordered)
    if not ordered:
        return False
    for pair in STOP_PAIRS:
        if pair <= words:
            return words - pair <= STOP_OBJECTS
    verbs = [w for w in ordered if w in STOP_VERBS | RETREAT_VERBS or w == "off"]
    if len(verbs) != 1 or (verbs[0] == "off" and ordered[-1] != "off"):
        return False
    verb_at = ordered.index(verbs[0])
    before, after = set(ordered[:verb_at]), set(ordered[verb_at + 1:])
    # A motion is stopped only when it follows the verb: "abort_approach" stops, "approach_stop" approaches.
    if verbs[0] == "off":
        return before <= OFF_OBJECTS
    allowed_after = STOP_OBJECTS | (MOTION_OBJECTS if verbs[0] in STOP_VERBS else frozenset())
    return before <= STOP_OBJECTS and after <= allowed_after


def check_overrides(overrides: dict[str, str] | None) -> dict[str, str]:
    """A deployment's risk patterns with their levels lower-cased; any level but low or high is refused.

    A typo such as "Hgih" must not quietly make a command unheld.
    """
    checked = {}
    for pattern, level in (overrides or {}).items():
        normalised = str(level).strip().lower()
        if normalised not in ("low", "high"):
            raise ValueError(f"risk level for {pattern!r} must be 'low' or 'high', got {level!r}")
        checked[pattern] = normalised
    return checked


def risk_of(dev_class: str, command: str, overrides: dict[str, str] | None = None, envelope: bool = False, stem: bool = False) -> str:
    """"low" or "high" for one command. ``envelope``: the device implements the job envelope;
    ``stem``: it acquires scanned images (see ``is_stem_device``)."""
    key = _key(dev_class, command)
    for pattern, level in check_overrides(overrides).items():
        if fnmatchcase(key, pattern.lower()):
            return level
    name = command.lower()
    if is_safe_stop(command):
        return "low"
    if envelope and name in ENVELOPE_LOW:
        return "low"
    if stem and name in STEM_LOW:
        return "low"
    for pattern, level in DEFAULT_RISK:
        if fnmatchcase(key, pattern):
            return level
    return "high"


def is_stem_device(command_names) -> bool:
    """True for an electron microscope: it scans images and blanks a beam, so the STEM_LOW
    commands are beam settings. An AFM's acquire_scanned_image alone does not count."""
    return {"acquire_scanned_image", "blank_beam"} <= {c.lower() for c in command_names}


def is_envelope_device(command_names) -> bool:
    """True when a device exposes the whole job envelope (reserve ... release)."""
    return ENVELOPE_VERBS <= {c.lower() for c in command_names}


def operator_only(dev_class: str, command: str) -> bool:
    key = _key(dev_class, command)
    return any(fnmatchcase(key, pattern) for pattern in OPERATOR_ONLY)
