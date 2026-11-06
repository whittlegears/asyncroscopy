"""What the MCP server knows about each tool, decided once at discovery: ``ToolDescriptor``.

The approval gate, ``find_tools``, ``endpoint_info``, ``action_space``, the token role check
(``endpoint_auth``) and ``stop_all`` all read the descriptor; none of them parses a tool's name again.

A device command's facts come from, in order:

1. the MCP config (``mcp.safety.risk`` overrides; ``mcp.safety.auto_routine``);
2. what the device declares in its ``action_hints`` attribute, per command: ``risk`` ("low" or "high"),
   ``operator_only``, ``stop`` (a no-argument command stop_all calls), ``routine`` (auto mode runs it
   without a person), ``role`` (``asyncroscopy.core.names.ROLES``). A Python device declares them with
   ``asyncroscopy.utils.commands.command(risk=..., ...)``; a YAML instrument from its file;
3. the safety policy's name rules (``asyncroscopy.safety.policy``), only for what the device leaves out.
   Unknown is high. The server prints, at startup, which commands fell back to the rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

import tango

from asyncroscopy.core.names import ACTION_HINTS, check_role, tool_name
from asyncroscopy.mcp.action_roles import role_for
from asyncroscopy.safety import policy

# Native tools of the MCP server (no device behind them): role, risk, whether it acts, whether it starts a job.
NATIVE: dict[str, tuple[str, str, bool, bool]] = {
    "run_job": ("job.run", "high", True, True),
    "wait_for_job": ("job.status", "low", False, False),
    "list_data": ("data.read", "low", False, False),
    "view_data": ("data.read", "low", False, False),
}
# Envelope verbs that only read a job (an observe, not an act).
JOB_READS = frozenset({"job_status", "results"})


@dataclass(frozen=True)
class ToolDescriptor:
    """One tool as the server decided it at discovery.

    ``device``/``command``: the tool prefix and Tango command (None for a native tool). ``risk``: "low" or
    "high" as the approval gate sees it, and ``risk_source`` where that came from ("override",
    "declared", "rule", "native"). ``acts``: it changes something (else it only observes).
    ``routine``: auto mode runs it without a person. ``stop``: stop_all calls it (no argument).
    ``job_start``: it starts a job.
    """

    name: str
    device: str | None
    command: str | None
    native: bool
    action_role: str
    risk: str
    acts: bool
    operator_only: bool
    dev_class: str | None = None
    role_source: str = "rule"
    risk_source: str = "rule"
    routine: bool = False
    stop: bool = False
    job_start: bool = False
    void: bool = False  # takes no argument


@dataclass
class DeviceInfo:
    """One device behind a tool prefix, as found at discovery."""

    prefix: str
    dev_class: str
    proxy: Any = None
    stem: bool = False  # scans images and blanks a beam (policy.is_stem_device)
    probe: bool = False  # brings a probe to the sample (policy.is_probe_device)
    envelope: bool = False  # runs jobs (the whole job envelope)
    job_kind: str | None = None
    plan_example: str | None = None
    in_types: dict[str, Any] = field(default_factory=dict)  # command -> Tango input type
    hints: dict[str, dict] = field(default_factory=dict)  # command -> its action_hints entry

    def kinds(self) -> set[str]:
        return {k for k in ("stem", "probe", "envelope") if getattr(self, k)}


def read_hints(proxy: Any) -> tuple[dict[str, dict], str | None]:
    """A device's ``action_hints`` ({command: {...}}) and a warning when it has one that cannot be read."""
    try:
        names = {a.lower() for a in proxy.get_attribute_list()}
        if ACTION_HINTS not in names:
            return {}, None
        raw = json.loads(str(proxy.read_attribute(ACTION_HINTS).value) or "{}")
    except Exception as exc:
        return {}, f"could not read its {ACTION_HINTS} ({type(exc).__name__}: {exc})"
    if not isinstance(raw, dict):
        return {}, f"its {ACTION_HINTS} is not a JSON object"
    return {str(c): h for c, h in raw.items() if isinstance(h, dict)}, None


def matched_overrides(name: str, key: str, overrides: dict[str, str]) -> list[str]:
    """The levels of the mcp.safety.risk patterns matching a tool name or its "Class.command"."""
    return [level for pattern, level in overrides.items()
            if fnmatchcase(name.lower(), pattern.lower()) or fnmatchcase(key.lower(), pattern.lower())]


def is_routine(device: DeviceInfo, command: str, name: str, routine: dict[str, list[str]]) -> bool:
    """Whether the auto-routine data (``mcp.safety.auto_routine`` or the packaged profile) lists this command:
    a device-kind key ("stem", "probe", "envelope") with command globs, or "tools" with tool-name or
    "Class.command" globs."""
    key = f"{device.dev_class}.{command}".lower()
    for kind, patterns in routine.items():
        if kind == "tools":
            if any(fnmatchcase(name.lower(), p.lower()) or fnmatchcase(key, p.lower()) for p in patterns):
                return True
        elif kind in device.kinds() and any(fnmatchcase(command.lower(), p.lower()) for p in patterns):
            return True
    return False


def describe_command(device: DeviceInfo, command: str, *, overrides: dict[str, str], routine: dict[str, list[str]],
                     warnings: list[str]) -> ToolDescriptor:
    """The descriptor of one device command (see the module docstring for the order of sources)."""
    name = tool_name(device.prefix, command)
    key = f"{device.dev_class}.{command}"
    hint = device.hints.get(command, {})
    matched = matched_overrides(name, key, overrides)
    declared = str(hint.get("risk", "")).strip().lower()
    if declared and declared not in ("low", "high"):
        warnings.append(f"{device.prefix}.{command}: declared risk {hint.get('risk')!r} is not low or high; the rules decide")
        declared = ""
    if matched:  # overrides that disagree (a broad "*": low and an exact high) never lower a tool: high wins
        risk, source = ("high" if "high" in matched else matched[0]), "override"
    elif declared:
        risk, source = declared, "declared"
    else:
        risk = policy.risk_of(device.dev_class, command, envelope=device.envelope, stem=device.stem, probe=device.probe)
        source = "rule"
    role, role_source = None, "rule"
    if hint.get("role"):
        try:
            role, role_source = check_role(str(hint["role"]), f"{device.prefix}.{command}"), "annotation"
        except ValueError as exc:
            warnings.append(f"{exc}; the rules decide instead")
    role = role or role_for(device.dev_class, command, envelope=device.envelope, probe=device.probe)
    if role == "other" and role_source == "rule":
        role_source = "default"
    reads = policy.is_read_only(device.dev_class, command, overrides) or (device.envelope and command.lower() in JOB_READS)
    marked_high = "high" in matched
    return ToolDescriptor(
        name=name, device=device.prefix, command=command, native=False, action_role=role, risk=risk, acts=not reads,
        operator_only=hint.get("operator_only") is True or policy.operator_only(device.dev_class, command),
        dev_class=device.dev_class, role_source=role_source, risk_source=source,
        routine=not marked_high and (hint.get("routine") is True or is_routine(device, command, name, routine)),
        stop=hint.get("stop") is True, job_start=device.envelope and command == "start",
        void=_is_void(device.in_types.get(command)),
    )


def rule_risk(device: DeviceInfo, command: str) -> str:
    """The risk the declarations and rules give a command before any mcp.safety.risk override."""
    declared = str(device.hints.get(command, {}).get("risk", "")).strip().lower()
    if declared in ("low", "high"):
        return declared
    return policy.risk_of(device.dev_class, command, envelope=device.envelope, stem=device.stem, probe=device.probe)


def describe_native(name: str) -> ToolDescriptor:
    """The descriptor of a native tool (one the server itself offers): low and observing unless NATIVE says otherwise."""
    role, risk, acts, job_start = NATIVE.get(name, ("other", "low", False, False))
    return ToolDescriptor(name=name, device=None, command=None, native=True, action_role=role, risk=risk, acts=acts,
                          operator_only=False, role_source="rule" if role != "other" else "default", risk_source="native",
                          job_start=job_start)


def _is_void(in_type: Any) -> bool:
    return in_type == tango.CmdArgType.DevVoid
