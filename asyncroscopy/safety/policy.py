"""Which commands count as high risk, and which only an operator may call.

Risk is decided by ``Class.command`` patterns, first match wins. Anything no
pattern matches is high risk: a new or renamed command is held in approval
mode rather than waved through. A deployment can put its own patterns in
front of these with the MCP config's ``safety.risk`` mapping.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

DEFAULT_RISK: list[tuple[str, str]] = [
    # Reading state, planning and stopping are always safe.
    ("*.get_*", "low"),
    ("*.list*", "low"),
    ("*.read*", "low"),
    ("*.status", "low"),
    ("*.job_status", "low"),
    ("*.results", "low"),
    ("*.reserve", "low"),
    ("*.release", "low"),
    ("*.abort", "low"),
    ("*.blank_beam", "low"),
    ("*.ping", "low"),
    ("approvals.*", "low"),
    # Changes that do not move hardware or expose the sample.
    ("*.place_beam", "low"),
    ("*.place_beam_list", "low"),
    ("*.set_fov", "low"),
    ("*.set_defocus", "low"),
    ("data.*", "low"),
    # Moving hardware, exposing the sample, or starting a job.
    ("*.move_stage", "high"),
    ("*.acquire*", "high"),
    ("*.unblank_beam", "high"),
    ("*.start", "high"),
    ("*.set_column_valves", "high"),
    ("*.set_screen*", "high"),
    ("*.calibrate*", "high"),
    ("*.correct*", "high"),
    ("*.auto_focus", "high"),
]

# Never offered on the agent endpoint, whatever the config says: an agent must
# not approve its own requests, switch the mode, or read the twin's answers.
OPERATOR_ONLY: list[str] = [
    "approvals.approve",
    "approvals.deny",
    "approvals.set_mode",
    "approvals.revoke",
    "*.export_ground_truth",
]


def _key(dev_class: str, command: str) -> str:
    return f"{dev_class}.{command}".lower()


def risk_of(dev_class: str, command: str, overrides: dict[str, str] | None = None) -> str:
    key = _key(dev_class, command)
    for pattern, level in list((overrides or {}).items()) + DEFAULT_RISK:
        if fnmatchcase(key, pattern.lower()):
            return level
    return "high"


def operator_only(dev_class: str, command: str) -> bool:
    key = _key(dev_class, command)
    return any(fnmatchcase(key, pattern) for pattern in OPERATOR_ONLY)
