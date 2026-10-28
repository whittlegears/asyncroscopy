"""Which commands count as high risk, and which only an operator may call.

Risk is decided by ``Class.command`` patterns, first match wins:

1. the deployment's own patterns (the MCP config's ``safety.risk``), so a
   team can mark e.g. ``Opentrons.get_tip`` high or ``MyStage.jog`` high;
2. safe-stop verbs, always low: an agent must never wait for approval to
   abort, stop, retract or close a shutter;
3. the job-envelope verbs other than ``start``, low on devices that implement
   the envelope (the MCP server knows which those are);
4. the read-only naming convention of this code base (``get_*``, ``list_*``)
   and a few STEM commands that neither move hardware nor expose the sample;
5. everything else is high: a new or renamed command is held in approval
   mode rather than waved through.

The convention in (4) is a promise instrument authors keep: a command named
``get_...`` or ``list_...`` must not move or change anything. Name commands
that act otherwise, or list them as high in ``safety.risk``.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

SAFE_STOP_VERBS = ("abort", "stop", "retract", "close_shutter", "safe_state", "blank_beam", "halt", "emergency_stop")

ENVELOPE_VERBS = frozenset({"reserve", "start", "job_status", "abort", "results", "release"})
ENVELOPE_LOW = ("reserve", "job_status", "results", "release")

DEFAULT_RISK: list[tuple[str, str]] = [
    # Read-only by naming convention.
    ("*.get_*", "low"),
    ("*.list_*", "low"),
    ("*.ping", "low"),
    ("approvals.*", "low"),
    ("data.*", "low"),
    # STEM commands that neither move hardware nor expose the sample.
    ("*.place_beam", "low"),
    ("*.place_beam_list", "low"),
    ("*.set_fov", "low"),
    ("*.set_defocus", "low"),
]

# Never offered on the agent endpoint, whatever the config says: an agent must
# not approve its own requests, switch the mode, or read a twin's answers.
OPERATOR_ONLY: list[str] = [
    "approvals.approve",
    "approvals.deny",
    "approvals.set_mode",
    "approvals.revoke",
    # The MCP server reaches these through Tango itself; offering them to agents would let
    # an agent consume a grant (check) or file requests under any caller name (request).
    "approvals.check",
    "approvals.request",
    "approvals.list_grants",
    "*.export_ground_truth",
]


def _key(dev_class: str, command: str) -> str:
    return f"{dev_class}.{command}".lower()


def risk_of(dev_class: str, command: str, overrides: dict[str, str] | None = None, envelope: bool = False) -> str:
    """"low" or "high" for one command. ``envelope``: the device implements the job envelope."""
    key = _key(dev_class, command)
    for pattern, level in (overrides or {}).items():
        if fnmatchcase(key, pattern.lower()):
            return level
    name = command.lower()
    if name in SAFE_STOP_VERBS:
        return "low"
    if envelope and name in ENVELOPE_LOW:
        return "low"
    for pattern, level in DEFAULT_RISK:
        if fnmatchcase(key, pattern):
            return level
    return "high"


def is_envelope_device(command_names) -> bool:
    """True when a device exposes the whole job envelope (reserve ... release)."""
    return ENVELOPE_VERBS <= {c.lower() for c in command_names}


def operator_only(dev_class: str, command: str) -> bool:
    key = _key(dev_class, command)
    return any(fnmatchcase(key, pattern) for pattern in OPERATOR_ONLY)
