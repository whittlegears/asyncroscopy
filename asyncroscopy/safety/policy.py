"""Which commands count as high risk, and which only an operator may call.

Risk is decided by ``Class.command`` patterns, first match wins:

1. the deployment's own patterns (the MCP config's ``safety.risk``), so a
   team can mark e.g. ``Opentrons.get_tip`` high or ``MyStage.jog`` high;
2. safe-stop commands, always low: an agent must never wait for approval to
   abort, stop, retract, switch off or close a shutter. This is an allow-list,
   like the rest of the policy: a command's name (split on "_" and CamelCase)
   has exactly one stop verb (STOP_VERBS, RETREAT_VERBS, or "off" as the last
   word) and every other word is something that verb may stop (STOP_OBJECTS;
   MOTION_OBJECTS too after abort/stop/halt/pause; a named RETREAT_OBJECTS word
   for retract/withdraw (a bare one only on a probe device, ``is_probe_device``:
   a syringe pump's withdraw aspirates), a named DISENGAGE_OBJECTS word for disengage, and an
   OFF_OBJECTS word before "off"), or it is a stop pair (STOP_PAIRS) with only
   STOP_OBJECTS beside it; "beam stop" is the beamstop, not a stop. So ``stop_scan``,
   ``AbortScan``, ``abort_approach``, ``beam_off`` and ``shutter_close`` are
   low; ``Off``, ``brake_off``, ``vacuum_off``, ``disengage_motor``,
   ``retract_shutter``, ``BeamStopMove``, ``stop_retract``, ``retract_approach``
   and ``estop_ack`` are not. Mark other stops low with ``mcp.safety.risk``;
3. the job-envelope verbs other than ``start``, low on devices that implement
   the envelope (the MCP server knows which those are);
4. Tango's ``State`` and ``Status``, and the read-only naming convention of
   this code base (``get_*``, ``list_*``), plus, on a device that acquires
   scanned images, a few STEM commands that neither move hardware nor expose
   the sample (STEM_LOW);
5. everything else is high: a new or renamed command is held in approval
   mode rather than waved through, and in auto mode too unless it is
   routine (AUTO_DEFAULT) or an operator switched it on.

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
    "indenter", "head", "objective", "robot", "xray", "source",
})
# What "retract"/"withdraw" may move: a probe, needle or Z away from the sample. It must name one: a bare
# Retract lifts the tip on an SPM, but a syringe pump's Withdraw aspirates. A bare one (or "retract_now")
# is low only on a probe device (``probe=True``, see ``is_probe_device``). retract_shutter opens a beam
# path, retract_stage or retract_robot moves an axis or an arm, and retract_screen uncovers the detectors.
RETREAT_OBJECTS = frozenset({"tip", "probe", "head", "z", "indenter", "objective", "needle", "now"})
# Words that make a device a probe instrument, whose bare retract/withdraw lifts the probe off the sample.
PROBE_WORDS = frozenset({"approach", "engage", "indent"})
# What "disengage" may take, and it must name one: disengage_motor or a bare Disengage can free an axis.
DISENGAGE_OBJECTS = frozenset({"tip", "probe", "head"})
# What "off" may switch off: a source of energy. "stage_off", "z_off" or "shutter_off" are not stops
# (a de-energised shutter may open; close it with close_shutter).
OFF_OBJECTS = frozenset({"beam", "emission", "lamp", "light", "led", "illumination", "laser", "gun", "filament", "hv", "xray", "source"})
MOTION_OBJECTS = frozenset({"approach", "move", "motion", "movement", "scan", "scanning", "acquisition", "acquire", "job", "run", "travel", "focus", "autofocus", "live", "indent", "imaging", "exposure", "protocol"})
STOP_PAIRS = (frozenset({"close", "shutter"}), frozenset({"blank", "beam"}), frozenset({"safe", "state"}))
# "emergency" qualifies a stop verb (emergency_stop) and is never one by itself.
MODIFIERS = frozenset({"emergency"})

ENVELOPE_VERBS = frozenset({"reserve", "start", "job_status", "abort", "results", "release"})
ENVELOPE_LOW = ("reserve", "job_status", "results", "release", "renew_lease")

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
    "approvals.set_auto_allowed",
    "approvals.revoke_grants",
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
    # Which holder is fitted sets the stage's Z and tilt limits: an agent must not be able to widen them.
    "*.set_holder",
    "*.revise",
    "*.load_sample",
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
    return bool(set(_word_list(command)) & (STOP_VERBS | RETREAT_VERBS | {"off", "unload", "park", "estop", "cancel", "shutdown", "close", "blank", "lift", "safe"}))


def is_safe_stop(command: str, probe: bool = False) -> bool:
    """Exactly one stop verb and nothing else but what it may stop (see STOP_OBJECTS), or a stop pair.

    "off" counts only as the last word (beam_off, lamp_off); "off_axis" and "kick_off" are not stops.
    A bare retract or withdraw counts only on a probe device (``probe``; see ``is_probe_device``).
    """
    ordered = [w for w in _word_list(command) if w not in MODIFIERS]
    # "beam" then "stop" names the beamstop (BeamStopMove moves it out of the direct beam): one noun.
    for i in range(len(ordered) - 1, 0, -1):
        if ordered[i - 1] == "beam" and ordered[i] == "stop":
            ordered[i - 1:i + 1] = ["beamstop"]
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
    if verbs[0] == "off":  # a bare Off is a de-energise (on a stage it drops holding torque)
        return bool(before) and before <= OFF_OBJECTS
    if verbs[0] == "disengage":
        return bool(before | after) and (before | after) <= DISENGAGE_OBJECTS
    if verbs[0] in RETREAT_VERBS:
        named = (before | after) - {"now"}
        return (bool(named) or probe) and (before | after) <= RETREAT_OBJECTS
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


def risk_of(dev_class: str, command: str, overrides: dict[str, str] | None = None, envelope: bool = False, stem: bool = False, probe: bool = False) -> str:
    """"low" or "high" for one command. ``envelope``: the device implements the job envelope;
    ``stem``: it acquires scanned images (see ``is_stem_device``); ``probe``: it approaches a sample
    with a probe, so a bare retract lifts it away (see ``is_probe_device``)."""
    key = _key(dev_class, command)
    matched = [level for pattern, level in check_overrides(overrides).items() if fnmatchcase(key, pattern.lower())]
    if matched:  # overrides that disagree (a broad "*": low beside an exact high) never lower a command
        return "high" if "high" in matched else matched[0]
    name = command.lower()
    if is_safe_stop(command, probe=probe):
        return "low"
    if envelope and name in ENVELOPE_LOW:
        return "low"
    if stem and name in STEM_LOW:
        return "low"
    for pattern, level in DEFAULT_RISK:
        if fnmatchcase(key, pattern):
            return level
    return "high"


READ_ONLY_NAMES = frozenset({"state", "status", "ping"})
READ_ONLY_PREFIXES = ("get_", "list_")


def is_read_only(dev_class: str, command: str, overrides: dict[str, str] | None = None) -> bool:
    """True when the policy counts ``command`` as a pure read: the naming convention of (4) above
    (``get_*``, ``list_*``, ``State``, ``Status``, ``ping``) and no deployment pattern marks it high.

    Low risk is not the same thing: a safe stop (``blank_beam``) or a STEM_LOW setting is low but
    writes. A dry run passes a command through only when this is true (jobs/recording_proxy.py).
    """
    name = command.lower()
    if not (name in READ_ONLY_NAMES or name.startswith(READ_ONLY_PREFIXES)):
        return False
    key = _key(dev_class, command)
    return not any(level == "high" for pattern, level in check_overrides(overrides).items() if fnmatchcase(key, pattern.lower()))


# Native MCP tools (lower-case names, not device commands) that act on the instrument: run_job starts a job.
NATIVE_HIGH = frozenset({"run_job"})

# High-risk commands that auto mode runs without asking: routine imaging, spectra, stage moves (the
# device refuses one outside its limits) and placing the beam. Every other high-risk command (column
# valves, screen, apertures, corrector, spectrometer set-up, calibration, a camera exposure, or any
# unknown or renamed command) is held in auto too, until an operator switches it on
# (Approvals.set_auto_allowed). A job's start runs in auto when every command it drives would.
AUTO_DEFAULT = frozenset({
    "move_stage", "acquire_scanned_image", "acquire_scanned_data_advanced", "acquire_spectrum", "acquire_eels_spectrum",
    "place_beam", "place_beam_list", "unblank_beam", "auto_focus", "set_stigmator", "set_beam_tilt", "set_image_shift",
})


def auto_allows(tool_name: str, switched_on=(), stem_devices=()) -> bool:
    """True when auto mode may run the high-risk tool ``tool_name`` ("<Prefix>_<command>") without a
    person: an operator switched it on (``switched_on``: tool-name globs, case ignored), or it is a
    routine command (AUTO_DEFAULT) of an electron microscope (``stem_devices``: the tool prefixes of
    devices that scan images and blank a beam, see ``is_stem_device``). On any other instrument
    (an AFM's move_stage drags an engaged tip) nothing is routine until an operator switches it on."""
    if any(fnmatchcase(tool_name.lower(), str(pattern).lower()) for pattern in switched_on):
        return True
    for prefix in sorted(stem_devices, key=len, reverse=True):  # "EELS_Cam" before "EELS"
        if tool_name.startswith(f"{prefix}_"):
            return tool_name[len(prefix) + 1:].lower() in AUTO_DEFAULT
    return False


def auto_allows_start(tool_name: str, drives, switched_on=(), stem_devices=()) -> bool:
    """True when auto mode may start a job without a person: its start was switched on, or ``drives``
    (the high-risk tools the reserved plan drives) is known and auto mode allows every one."""
    if any(fnmatchcase(tool_name.lower(), str(pattern).lower()) for pattern in switched_on):
        return True
    return drives is not None and all(auto_allows(name, switched_on, stem_devices) for name in drives)


def tool_risk(tool_name: str, risks: dict[str, str] | None = None) -> str:
    """"low" or "high" for an MCP tool name ("<Class>_<command>" or a native tool).

    ``risks`` is the endpoint's own table (the MCP server's ``endpoint_info``), which knows each
    device's envelope, STEM and override rules; without it, or for a name it lacks, the policy
    is applied to the name alone: envelope verbs count as an envelope device's, and STEM_LOW
    settings stay high (unknown is high).
    """
    if risks is not None and tool_name in risks:
        return risks[tool_name]
    if any(c in tool_name for c in "*?["):
        return _glob_risk(tool_name, risks)
    device, _, command = tool_name.partition("_")
    if not command or not device[:1].isupper():
        return "high" if tool_name in NATIVE_HIGH else "low"
    return risk_of(device, command, envelope=command.lower() in ENVELOPE_VERBS)


def _glob_risk(pattern: str, risks: dict[str, str] | None) -> str:
    """A tool-name glob (a skill's ``tools: ["*_indent"]``) is high when any tool it may name is.

    With the endpoint's table: high if a high tool there matches. Otherwise from the command part
    alone: "*_indent" or "*_acquire_*" act, "*_get_*" and "Mapper*_abort" do not, a bare "*" may.
    """
    if risks is not None and any(level == "high" for name, level in risks.items() if fnmatchcase(name, pattern)):
        return "high"
    device, _, command = pattern.partition("_")
    if not command or not command.strip("*?[]_"):  # "*" or "Mapper_*": every tool, or every command of a device
        return "high" if not pattern.islower() or any(fnmatchcase(n, pattern) for n in NATIVE_HIGH) else "low"
    if device[:1].islower():  # "analysis_job_*": a native tool family
        return "high" if any(fnmatchcase(n, pattern) for n in NATIVE_HIGH) else "low"
    # "*_indent" or "Mapper*_abort": the command decides, on any device of that name.
    return risk_of(device.strip("*?[]") or "Any", command, envelope=command.lower() in ENVELOPE_VERBS)


def is_stem_device(command_names) -> bool:
    """True for an electron microscope: it scans images and blanks a beam, so the STEM_LOW
    commands are beam settings. An AFM's acquire_scanned_image alone does not count."""
    return {"acquire_scanned_image", "blank_beam"} <= {c.lower() for c in command_names}


def is_probe_device(command_names) -> bool:
    """True for a device that brings a probe to the sample (an SPM's approach, an indenter's indent):
    there a bare retract or withdraw lifts it away. A syringe pump's withdraw does not count."""
    return any(set(_word_list(str(c))) & PROBE_WORDS for c in command_names)


def is_envelope_device(command_names) -> bool:
    """True when a device exposes the whole job envelope (reserve ... release)."""
    return ENVELOPE_VERBS <= {c.lower() for c in command_names}


def operator_only(dev_class: str, command: str) -> bool:
    key = _key(dev_class, command)
    return any(fnmatchcase(key, pattern) for pattern in OPERATOR_ONLY)
