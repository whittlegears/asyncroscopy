"""An endpoint's action space: what can be controlled, what each command does, and what can be observed.

The MCP server (``asyncroscopy.mcp.mcp_server``) offers it as the native tool ``action_space()``;
``build(server)`` makes it. One JSON document (``asyncroscopy/contracts/action_space.schema.json``):

- ``actions``: every tool, with its arguments (type, units, range, choices, and the stage ``axis``
  or array ``elements`` where known), a one-line purpose, its risk as the approval gate sees it,
  whether it ``act``s or only ``observe``s, and a canonical ``role`` (``asyncroscopy.mcp.roles``);
- ``state``: what can be read, through a tool (``get_*``, ``State``) or as a Tango attribute, with units;
- ``jobs``: each job device's plan schema and example plan;
- ``instruments``: each device's kind (stem, probe, envelope).

Units and ranges come from, in order: the device's ``action_hints`` attribute (a YAML instrument
derives it from its file), an electron microscope's limits (``limits.action_hints`` with the device's
min_fov, max_fov and max_defocus properties), the tool's input schema (Tango types), and the units
named in the command's doc. A controller that picks tools by role, not by name, runs unchanged on
instruments whose commands are named differently (``asyncroscopy/examples/role_controller.py``).

From a shell::

    python -m asyncroscopy.mcp.action_space --url http://127.0.0.1:8000/mcp [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from typing import Any

from asyncroscopy.mcp.roles import ACTION_HINTS, NATIVE_ROLES, ROLES, check_role, role_for
from asyncroscopy.safety.policy import ENVELOPE_VERBS, NATIVE_HIGH, is_read_only

SCHEMA_VERSION = "1.0.0"
DEFAULT_URL = "http://127.0.0.1:8000/mcp"
# Attributes that describe a device to the MCP server rather than the instrument's state.
META_ATTRIBUTES = frozenset({"state", "status", "job_kind", "plan_example", "plan_schema", ACTION_HINTS})
NATIVE_ACTS = frozenset(NATIVE_HIGH)  # every other native tool only reads
# Unit words in command docs ("Field of view in metres", "(pA)") -> the unit.
UNIT_WORDS = {
    "metres": "m", "meters": "m", "metre": "m", "meter": "m", "nanometres": "nm", "nanometers": "nm",
    "degrees": "deg", "radians": "rad", "radian": "rad", "seconds": "s", "picoamperes": "pA",
}
_UNIT_IN_TEXT = re.compile(r"\b(?:in|of)\s+(" + "|".join(UNIT_WORDS) + r")\b|\((m|nm|s|pA|eV|V|Hz|deg|rad|W|mW|K)\)", re.IGNORECASE)


def _units(text: str) -> str | None:
    """The one unit a doc names ("in meters", "(pA)"); None when it names none or several."""
    found = {UNIT_WORDS.get((a or "").lower(), b) for a, b in _UNIT_IN_TEXT.findall(text or "")}
    return found.pop() if len(found) == 1 else None


def _json_type(schema: dict) -> dict:
    """{"type": ..., "items": ...} from a JSON Schema fragment (anyOf joined with |, null left out)."""
    if "type" in schema:
        found = {"type": schema["type"]}
        if schema["type"] == "array" and isinstance(schema.get("items"), dict) and schema["items"].get("type"):
            found["items"] = schema["items"]["type"]
        return found
    types = [s.get("type") for s in schema.get("anyOf", []) if s.get("type") not in (None, "null")]
    return {"type": "|".join(dict.fromkeys(types))} if types else {}


def _purpose(description: str, prefix: str | None, command: str | None = None) -> str:
    """The first line of a tool's description that says what it does (not the Tango plumbing lines).

    A command declared with plain tango.server.command has no doc on the wire; its name, and what
    its argument is, stand in.
    """
    skip = ("Tango Device Class:", "Tango Command:", "Input Type:", "Input Description:", "Output Type:", "Example request:")
    lines = [line.strip() for line in (description or "").splitlines()]
    for line in lines:
        if line and not line.startswith(skip) and not (prefix and line.startswith(f"{prefix} job:")):
            return line
    if not command:
        return ""
    argument = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Input Description:")), "")
    return command.replace("_", " ") + (f": {argument}" if argument else "")


def _arguments(tool, hinted_args: list[dict] | None) -> list[dict]:
    """A tool's arguments from its input schema, with the hints' units, ranges and axes laid over them.

    A JSON-request command whose fields are hinted (a YAML command with several inputs) is shown by its
    fields alone: the MCP tool takes them by name, so the request parameter is only another way in.
    """
    params = tool.parameters or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    hinted = {a["name"]: a for a in hinted_args or [] if isinstance(a, dict) and a.get("name")}
    fields_only = bool(hinted) and set(hinted) <= set(props) and any(
        n not in hinted and str(s.get("description", "")).upper().startswith("JSON") for n, s in props.items())
    args = []
    for name, schema in props.items():
        if fields_only and name not in hinted:
            continue
        arg = {"name": name, **_json_type(schema), "required": name in required or (fields_only and name in hinted)}
        doc = str(schema.get("description") or "")
        if doc and not doc.startswith("Same as "):
            arg["doc"] = doc
        arg.update({k: v for k, v in hinted.get(name, {}).items() if k != "name"})
        if "units" not in arg and doc and _units(doc):
            arg["units"] = _units(doc)
        args.append(arg)
    args += [{**h, "required": True} for name, h in hinted.items() if name not in props]
    return args


def _returns(tool, hint: dict | None, purpose: str) -> dict | None:
    out = tool.output_schema or {}
    props = out.get("properties") or {}
    found = _json_type(props["result"]) if out.get("x-fastmcp-wrap-result") and "result" in props else ({"type": out["type"]} if out.get("type") else {})
    if found.get("type") == "null":  # a command that returns nothing
        found = {}
    found.update(hint or {})
    if found and "units" not in found and _units(purpose):
        found["units"] = _units(purpose)
    return found or None


def _stem_hints(device) -> dict:
    """An electron microscope's argument ranges (limits.py), at the device's own min_fov, max_fov and max_defocus."""
    from asyncroscopy.instruments.electron_microscope import limits

    values = {"min_fov": limits.DEFAULT_MIN_FOV, "max_fov": limits.DEFAULT_MAX_FOV, "max_defocus": limits.DEFAULT_MAX_DEFOCUS}
    try:
        stored = device.get_property(list(values))
        for name in values:
            if stored.get(name):
                values[name] = float(list(stored[name])[0])
    except Exception:  # no database (a bare test context) or not a number: the backends' defaults apply
        pass
    return limits.action_hints(**values)


def _device_hints(device, stem: bool, warnings: list[str], prefix: str) -> dict:
    """Hints for one device: the limits of an electron microscope, then its own action_hints attribute."""
    hints = _stem_hints(device) if stem else {}
    try:
        names = {a.lower() for a in device.get_attribute_list()}
        own = json.loads(str(device.read_attribute(ACTION_HINTS).value)) if ACTION_HINTS in names else {}
    except Exception as exc:
        warnings.append(f"{prefix}: could not read its {ACTION_HINTS} ({type(exc).__name__}: {exc})")
        own = {}
    for command, hint in (own if isinstance(own, dict) else {}).items():
        if isinstance(hint, dict):
            hints[command] = {**hints.get(command, {}), **hint}
    return hints


def _attributes(device, prefix: str, warnings: list[str]) -> list[dict]:
    """Readable Tango attributes of a device (not the MCP server's own metadata), with type, units and range."""
    import tango

    try:
        infos = device.attribute_list_query()
    except Exception as exc:
        warnings.append(f"{prefix}: could not list its attributes ({type(exc).__name__}: {exc})")
        return []
    found = []
    for info in infos:
        if info.name.lower() in META_ATTRIBUTES:
            continue
        entry = {"name": info.name, "device": prefix, "source": "attribute", "type": str(tango.CmdArgType.values.get(info.data_type, info.data_type)),
                 "writable": info.writable != tango.AttrWriteType.READ}
        unit = str(info.unit or "")
        if unit and unit.lower() not in ("no unit", "none"):
            entry["units"] = unit
        description = str(info.description or "")
        if description and description.lower() not in ("no description", "none"):
            entry["doc"] = description
        for key in ("min_value", "max_value"):
            value = str(getattr(info, key, "") or "")
            try:
                entry[key[:3]] = float(value)
            except ValueError:
                pass  # "Not specified"
        found.append(entry)
    return found


def build(server) -> dict[str, Any]:
    """The action space of an MCPServer after setup()."""
    warnings: list[str] = []
    actions: list[dict] = []
    instruments: list[dict] = []
    state: list[dict] = []
    jobs: list[dict] = []
    overrides = server.risk_overrides
    for prefix in sorted(server.tools):
        device = server._devices.get(prefix)
        dev_class = server._prefix_class.get(prefix, prefix)
        kind = {"stem": prefix in server._stem_prefixes, "probe": prefix in server._probe_prefixes, "envelope": prefix in server._envelope_prefixes}
        try:
            address = str(device.dev_name()) if device is not None else None
        except Exception:
            address = None
        instruments.append({"device": prefix, "class": dev_class, "address": address, "kind": kind})
        hints = _device_hints(device, kind["stem"], warnings, prefix) if device is not None else {}
        for command in sorted(server.tools[prefix]):
            name = f"{prefix}_{command}".replace("/", "_").replace("-", "_")
            tool = server._registered.get(name)
            if tool is None:
                continue
            hint = hints.get(command, {})
            role, source = None, "rule"
            if hint.get("role"):
                try:
                    role, source = check_role(hint["role"], f"{prefix}.{command}"), "annotation"
                except ValueError as exc:
                    warnings.append(f"{exc}; the rules decide instead")
            role = role or role_for(dev_class, command, envelope=kind["envelope"], probe=kind["probe"])
            if role == "other":
                source = "default"
            purpose = _purpose(tool.description, prefix, command)
            observes = is_read_only(dev_class, command, overrides) or (kind["envelope"] and command.lower() in ("job_status", "results"))
            action = {
                "tool": name, "device": prefix, "command": command, "role": role, "role_source": source,
                "purpose": purpose, "risk": server.tool_risks.get(name, "high"), "effect": "observe" if observes else "act",
                "args": _arguments(tool, hint.get("args")),
            }
            returns = _returns(tool, hint.get("returns"), purpose)
            if returns:
                action["returns"] = returns
            if hint.get("primary") is True:
                action["primary"] = True
            actions.append(action)
            if observes and not role.startswith("job."):
                read = {"name": command[4:] if command.lower().startswith("get_") else command, "device": prefix, "source": "tool", "tool": name, "role": role}
                read.update({k: v for k, v in (returns or {}).items() if k in ("type", "items", "units", "axis", "elements", "kind")})
                state.append(read)
        if device is not None:
            state += _attributes(device, prefix, warnings)
        if kind["envelope"]:
            jobs.append(_job(server, prefix, device))
    for name in sorted(server._native_names):
        tool = server._registered.get(name)
        if tool is None:
            continue
        purpose = _purpose(tool.description, None)
        role = NATIVE_ROLES.get(name, "other")
        returns = _returns(tool, None, purpose)
        actions.append({
            "tool": name, "device": None, "command": None, "role": role, "role_source": "rule" if role != "other" else "default",
            "purpose": purpose, "risk": "high" if name in NATIVE_HIGH else "low", "effect": "act" if name in NATIVE_ACTS else "observe",
            "args": _arguments(tool, None), **({"returns": returns} if returns else {}),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "endpoint": {"name": server.mcp.name, "role": server.endpoint_role},
        "roles": dict(ROLES),
        "instruments": instruments,
        "actions": actions,
        "state": state,
        "jobs": jobs,
        "warnings": warnings,
    }


def _job(server, prefix: str, device) -> dict:
    def attribute(name: str) -> Any:
        try:
            text = str(device.read_attribute(name).value)
            return json.loads(text) if text else None
        except Exception:  # an older job device without it, or not JSON
            return None

    example = server._plan_examples.get(prefix)
    try:
        example = json.loads(example) if example else None
    except json.JSONDecodeError:
        pass
    verbs = {verb: f"{prefix}_{verb}" for verb in server.tools.get(prefix, {}) if verb.lower() in ENVELOPE_VERBS | {"get_reservation"}}
    return {
        "device": prefix, "kind": server._job_kinds.get(prefix), "plan_schema": attribute("plan_schema") if device is not None else None,
        "plan_example": example, "tools": verbs, "run_with": "run_job" if "run_job" in server._native_names else None,
    }


def actions_with(space: dict, role: str, device: str | None = None) -> list[dict]:
    """The actions with ``role`` (on ``device``, a tool prefix, when given), in the document's order."""
    return [a for a in space.get("actions", []) if a["role"] == role and (device is None or a["device"] == device)]


# ---------------------------------------------------------------------- command line
def _arg_text(arg: dict) -> str:
    text = arg["name"]
    if arg.get("elements"):
        text += "[" + ",".join(arg["elements"]) + "]"
    if arg.get("axis"):
        text += f"<{arg['axis']}>"
    units = arg.get("units")
    if units:
        text += " " + (",".join(units) if isinstance(units, list) else str(units))
    if arg.get("min") is not None or arg.get("max") is not None:
        text += f" [{arg.get('min', '')}..{arg.get('max', '')}]"
    if arg.get("enum"):
        text += " {" + "|".join(str(v) for v in arg["enum"]) + "}"
    return text


def format_table(space: dict) -> str:
    """A readable summary of an action space: instruments, actions by role, state, jobs."""
    endpoint = space.get("endpoint", {})
    lines = [f"Action space of {endpoint.get('name')} ({endpoint.get('role')} endpoint), schema {space.get('schema_version')}", ""]
    lines.append("Instruments:")
    for inst in space.get("instruments", []):
        flags = [k for k, v in inst["kind"].items() if v]
        lines.append(f"  {inst['device']:<24} {inst['class']:<22} {','.join(flags) or '-'}")
    lines += ["", "Actions:", f"  {'ROLE':<16} {'TOOL':<40} {'RISK':<5} {'EFFECT':<8} ARGS"]
    order = {role: i for i, role in enumerate(space.get("roles", ROLES))}
    for a in sorted(space.get("actions", []), key=lambda a: (order.get(a["role"], 99), a["tool"])):
        args = "; ".join(_arg_text(arg) for arg in a.get("args", []))
        tool = a["tool"] + (" *" if a.get("primary") else "")
        lines.append(f"  {a['role']:<16} {tool:<40} {a['risk']:<5} {a['effect']:<8} {args}")
    lines.append("  (* the instrument's main observable)")
    lines += ["", "State:"]
    for s in space.get("state", []):
        where = s.get("tool") or f"{s['device']}.{s['name']} (Tango attribute)"
        units = s.get("units")
        units = ",".join(units) if isinstance(units, list) else units
        lines.append(f"  {s['device']:<24} {s['name']:<28} {units or '':<10} {where}")
    if space.get("jobs"):
        lines += ["", "Jobs:"]
        for job in space["jobs"]:
            schema = job.get("plan_schema")
            fields = sorted((schema or {}).get("properties", {}))
            lines.append(f"  {job['device']}: {job.get('kind') or ''}")
            lines.append(f"    plan fields: {', '.join(fields) if fields else '(no schema; see the example)'}")
            lines.append(f"    example: {json.dumps(job.get('plan_example'))}")
    if space.get("warnings"):
        lines += ["", "Warnings:"] + [f"  {w}" for w in space["warnings"]]
    return "\n".join(lines)


def structured(result) -> Any:
    """The value of an MCP tool result (fastmcp CallToolResult): its structured content, unwrapped."""
    content = getattr(result, "structured_content", None)
    if isinstance(content, dict):
        return content["result"] if set(content) == {"result"} else content
    text = "".join(getattr(b, "text", "") for b in getattr(result, "content", []) or [])
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


async def fetch(url: str) -> dict:
    """The action space of the MCP endpoint at ``url``."""
    from fastmcp import Client

    async with Client(url) as client:
        return structured(await client.call_tool("action_space", {}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print an MCP endpoint's action space: tools by canonical role, state and jobs.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"MCP endpoint (default {DEFAULT_URL})")
    parser.add_argument("--json", action="store_true", help="print the JSON document instead of a table")
    args = parser.parse_args(argv)
    try:
        space = asyncio.run(fetch(args.url))
    except Exception as exc:
        print(f"could not read the action space at {args.url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(space, indent=2) if args.json else format_table(space))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
