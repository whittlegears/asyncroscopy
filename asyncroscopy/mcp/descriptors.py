"""What the MCP server knows about each tool, decided once at discovery: ``ToolDescriptor``.

``find_tools``, ``endpoint_info``, ``action_space``, the token role check
(``endpoint_auth``) and ``stop_all`` all read the descriptor; none of them parses a tool's name again.

A device command's facts come from, in order:

1. the MCP config (``mcp.safety.risk`` overrides; ``mcp.safety.auto_routine``, which, when the config
   gives it, replaces every device's ``routine`` declaration);
2. what the device declares in its ``action_hints`` attribute, per command: ``risk`` ("low" or "high"),
   ``operator_only``, ``stop`` (a no-argument command stop_all calls), ``routine`` (auto mode runs it
   without a person), ``role`` (``asyncroscopy.core.names.ROLES``; a read role also says the command
   only observes). A Python device declares them with ``asyncroscopy.utils.commands.command(risk=..., ...)``
   on its base class, so every vendor and twin subclass inherits them; a YAML instrument from its file.
   Tango's own State, Status (low, read) and Init (high, operator only) need no declaration;
3. the safety policy's name rules (``asyncroscopy.safety.policy``), only for what a device leaves out (a
   third-party device). Unknown is high. The server prints, at startup, which commands fell back to the rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

import tango

from asyncroscopy.core.names import ACTION_HINTS, READ_ROLES, TANGO_ADMIN, TANGO_READS, check_role, tool_name
from asyncroscopy.mcp.action_roles import role_for
from asyncroscopy.safety import policy

# Native tools of the MCP server, which have no device behind them.
# Each gives role, risk, whether it acts and whether it starts a job.
NATIVE: dict[str, tuple[str, str, bool, bool]] = {
    "run_job": ("job.run", "high", True, True),
    "wait_for_job": ("job.status", "low", False, False),
    "list_data": ("data.read", "low", False, False),
    "view_data": ("data.read", "low", False, False),
    # They change what agents are told, not the instrument, and are never given to a model (not_for_models).
    "save_skill": ("other", "low", True, False),
    "delete_skill": ("other", "low", True, False),
    "switch_skill": ("other", "low", True, False),
    "list_pending_approvals": ("setting.read", "low", False, False),
    "approve_call": ("other", "high", True, False),
    "deny_call": ("other", "low", True, False),
    "get_permissions": ("setting.read", "low", False, False),
    "set_permission": ("setting.set", "high", True, False),
    "set_group_permission": ("setting.set", "high", True, False),
}
# Envelope verbs that only read a job.
JOB_READS = frozenset({"job_status", "results"})


@dataclass(frozen=True)
class ToolDescriptor:
    """One tool as the server decided it at discovery.

    ``device``/``command``: the tool prefix and Tango command (None for a native tool). ``risk``: "low" or
    "high", and ``risk_source`` where that came from ("override",
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
    stem: bool = False  # scans images and blanks a beam
    probe: bool = False  # brings a probe to the sample
    envelope: bool = False  # runs jobs
    job_kind: str | None = None
    plan_example: str | None = None
    in_types: dict[str, Any] = field(default_factory=dict)  # Tango input type of each command
    hints: dict[str, dict] = field(default_factory=dict)  # action_hints entry of each command
    hints_unreadable: bool = False  # it has action_hints that could not be read: agents get only State and Status

    def kinds(self) -> set[str]:
        return {k for k in ("stem", "probe", "envelope") if getattr(self, k)}


def read_hints(proxy: Any) -> tuple[dict[str, dict] | None, str | None]:
    """A device's ``action_hints`` ({command: {...}}; {} for a device without the attribute) and None with a
    warning when they cannot be read. None is not "nothing declared": the declarations that keep a command
    from agents (``operator_only``) are unknown, so the server must not offer that device's commands to agents."""
    try:
        names = {a.lower() for a in proxy.get_attribute_list()}
        if ACTION_HINTS not in names:
            return {}, None
        raw = json.loads(str(proxy.read_attribute(ACTION_HINTS).value) or "{}")
    except Exception as exc:
        return None, f"could not read its {ACTION_HINTS} ({type(exc).__name__}: {exc})"
    if not isinstance(raw, dict):
        return None, f"its {ACTION_HINTS} is not a JSON object"
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


def tango_builtin(command: str) -> dict[str, Any] | None:
    """What a Tango built-in command is, as a declaration: State and Status read, Init is the operator's."""
    if command in TANGO_READS:
        return {"risk": "low", "role": "setting.read"}
    if command in TANGO_ADMIN:
        return {"risk": "high", "operator_only": True}
    return None


def describe_command(device: DeviceInfo, command: str, *, overrides: dict[str, str], routine: dict[str, list[str]] | None,
                     warnings: list[str]) -> ToolDescriptor:
    """The descriptor of one device command (see the module docstring for the order of sources).

    ``routine``: the config's mcp.safety.auto_routine, which replaces the devices' routine declarations;
    None when the config gives none (the declarations decide).
    """
    name = tool_name(device.prefix, command)
    key = f"{device.dev_class}.{command}"
    builtin = tango_builtin(command)
    hint = builtin if builtin is not None and command not in device.hints else device.hints.get(command, {})
    matched = matched_overrides(name, key, overrides)
    declared = str(hint.get("risk", "")).strip().lower()
    if declared and declared not in ("low", "high"):
        warnings.append(f"{device.prefix}.{command}: declared risk {hint.get('risk')!r} is not low or high; the rules decide")
        declared = ""
    if matched:  # when overrides disagree the high one wins
        risk, source = ("high" if "high" in matched else matched[0]), "override"
    elif declared:
        risk, source = declared, ("tango" if hint is builtin else "declared")
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
    marked_high = "high" in matched
    if role_source == "annotation":  # a declared role says whether the command only observes
        reads = role in READ_ROLES and not marked_high
    else:
        reads = policy.is_read_only(device.dev_class, command, overrides) or (device.envelope and command.lower() in JOB_READS)
    if routine is None:
        listed = hint.get("routine") is True
    else:
        listed = is_routine(device, command, name, routine)
    return ToolDescriptor(
        name=name, device=device.prefix, command=command, native=False, action_role=role, risk=risk, acts=not reads,
        operator_only=hint.get("operator_only") is True or policy.operator_only(device.dev_class, command),
        dev_class=device.dev_class, role_source=role_source, risk_source=source,
        routine=not marked_high and listed,
        stop=hint.get("stop") is True, job_start=device.envelope and command == "start",
        void=_is_void(device.in_types.get(command)),
    )


def rule_risk(device: DeviceInfo, command: str) -> str:
    """The risk the declarations and rules give a command before any mcp.safety.risk override."""
    hint = device.hints.get(command) or tango_builtin(command) or {}
    declared = str(hint.get("risk", "")).strip().lower()
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


def with_declared_limits(parameters: dict, hint: dict) -> dict:
    """A tool's JSON input schema with the ranges, choices and units its command declares in ``action_hints``
    (``args``: name, min, max, enum, units). Ranges go in ``minimum`` and ``maximum`` (on an array's items when
    one range holds for every item) and text choices in ``enum``; units, number choices and ranges that differ
    element by element go in the property's description. Only keywords every model provider accepts are used.
    The device still enforces every limit; this only lets a client see them before it calls."""
    properties = (parameters or {}).get("properties") or {}
    args = [a for a in (hint or {}).get("args") or [] if isinstance(a, dict)]
    if not properties or not args:
        return parameters
    out = {**parameters, "properties": dict(properties)}
    for arg in args:
        name = _property_for(arg, properties, len(args))
        if name is None:
            continue
        prop = dict(properties[name])
        value = _value_schema(prop)
        if value is None:
            continue
        target = value
        if value.get("type") == "array":
            target = value["items"] = dict(value.get("items") or {})
        for key, keyword in (("min", "minimum"), ("max", "maximum")):
            if _is_number(arg.get(key)):
                target[keyword] = arg[key]
        choices = arg.get("enum")
        if isinstance(choices, list) and choices and all(isinstance(c, str) for c in choices):
            target["enum"] = choices
        said = _limits_in_words(arg)
        if said and said not in (prop.get("description") or ""):
            prop["description"] = f"{prop['description'].rstrip('. ')}. {said}" if prop.get("description") else said
        out["properties"][name] = prop
    return out


def _property_for(arg: dict, properties: dict, count: int) -> str | None:
    """The schema property an argument describes: the one of its name, else the only one there is."""
    if arg.get("name") in properties:
        return arg["name"]
    if len(properties) == 1 and count == 1:
        return next(iter(properties))
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _limits_in_words(arg: dict) -> str:
    """Units, number choices and element-by-element ranges as a sentence."""
    units, low, high, elements, choices = (arg.get(k) for k in ("units", "min", "max", "elements", "enum"))
    words = []
    if isinstance(choices, list) and choices and not all(isinstance(c, str) for c in choices):
        words.append("One of " + ", ".join(str(c) for c in choices) + ".")
    lists = [v for v in (units, low, high) if isinstance(v, list)]
    if not lists:
        return " ".join(words + ([f"Units: {units}."] if isinstance(units, str) and units else []))
    n = max(len(v) for v in lists + ([elements] if isinstance(elements, list) else []))
    labels = _per_element(elements, n) if isinstance(elements, list) else [f"element {i}" for i in range(n)]
    parts = []
    for label, lo, hi, unit in zip(labels, _per_element(low, n), _per_element(high, n), _per_element(units, n)):
        span = f"{lo}..{hi}" if lo is not None or hi is not None else ""
        parts.append(f"{label}: {' '.join(p for p in (span, unit) if p)}" if span or unit else str(label))
    return " ".join(words + ["Elements " + ", ".join(parts) + "."])


def _per_element(value: Any, n: int) -> list:
    """``value`` for each of ``n`` elements: a list padded with None, or one value repeated."""
    return list(value) + [None] * (n - len(value)) if isinstance(value, list) else [value] * n


def _value_schema(prop: dict) -> dict | None:
    """The part of a property's schema that holds its value: itself, or its one non-null branch of an anyOf
    (copied into ``prop``, so it can be changed)."""
    if "anyOf" not in prop:
        return prop
    branches = [i for i, b in enumerate(prop["anyOf"]) if b.get("type") != "null"]
    if len(branches) != 1:
        return None
    prop["anyOf"] = list(prop["anyOf"])
    prop["anyOf"][branches[0]] = dict(prop["anyOf"][branches[0]])
    return prop["anyOf"][branches[0]]
