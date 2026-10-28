"""Which commands count as high risk, and which only an operator may call.

Risk is decided by ``Class.command`` patterns, first match wins:

1. the deployment's own patterns (the MCP config's ``safety.risk``), so a
   team can mark e.g. ``Opentrons.get_tip`` high or ``MyStage.jog`` high;
2. safe-stop commands, always low: an agent must never wait for approval to
   abort, stop, retract, switch off or close a shutter. A command is a safe
   stop when its name (split on "_" and CamelCase) starts or ends with a stop
   word (STOP_WORDS) or contains a stop pair (STOP_PAIRS), and has no word that
   undoes a safeguard (UNSAFE_WORDS) or does more than stop (ACTION_WORDS). So
   ``Off``, ``stop_scan``, ``AbortScan`` and ``shutter_close`` are low;
   ``disable_stop``, ``interlock_off`` and ``withdraw_and_move`` are not;
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

# Words that make a command a safe stop, and words that disqualify it (see the module docstring).
STOP_WORDS = frozenset({"abort", "stop", "halt", "retract", "withdraw", "off", "estop", "emergency", "pause"})
STOP_PAIRS = (frozenset({"close", "shutter"}), frozenset({"blank", "beam"}), frozenset({"safe", "state"}))
UNSAFE_WORDS = frozenset({"disable", "override", "bypass", "ignore", "reset", "clear", "release", "unblank", "open", "force"})
# Words that mean the command does something besides stopping, or touches a safeguard:
# "interlock_off", "withdraw_and_move" and "set_stop_height" are not safe stops.
ACTION_WORDS = frozenset({
    "interlock", "safety", "limit", "limits", "guard", "lock", "unlock", "feedback",
    "move", "approach", "set", "drop", "resume", "start", "run", "and", "then", "enable",
})

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
# STEM commands that neither move hardware nor expose the sample. Low only on a device that
# acquires scanned images: on an optical microscope set_defocus drives the objective, and on a
# laser system place_beam points a live beam.
STEM_LOW = frozenset({"place_beam", "place_beam_list", "set_fov", "set_defocus"})

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


def is_safe_stop(command: str) -> bool:
    """A stop word as the first or last word (or a stop pair), and nothing that does more than stop."""
    ordered = _word_list(command)
    words = set(ordered)
    if not ordered or words & (UNSAFE_WORDS | ACTION_WORDS):
        return False
    return ordered[0] in STOP_WORDS or ordered[-1] in STOP_WORDS or any(pair <= words for pair in STOP_PAIRS)


def risk_of(dev_class: str, command: str, overrides: dict[str, str] | None = None, envelope: bool = False, stem: bool = False) -> str:
    """"low" or "high" for one command. ``envelope``: the device implements the job envelope;
    ``stem``: it acquires scanned images (see ``is_stem_device``)."""
    key = _key(dev_class, command)
    for pattern, level in (overrides or {}).items():
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
    """True when a device acquires scanned images, so the STEM_LOW commands are beam settings."""
    return "acquire_scanned_image" in {c.lower() for c in command_names}


def is_envelope_device(command_names) -> bool:
    """True when a device exposes the whole job envelope (reserve ... release)."""
    return ENVELOPE_VERBS <= {c.lower() for c in command_names}


def operator_only(dev_class: str, command: str) -> bool:
    key = _key(dev_class, command)
    return any(fnmatchcase(key, pattern) for pattern in OPERATOR_ONLY)
