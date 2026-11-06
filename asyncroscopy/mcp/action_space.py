"""An endpoint's action space: what can be controlled, what each command does, and what can be observed.

The MCP server (``asyncroscopy.mcp.mcp_server``) offers it as the native tool ``action_space()``;
``build(server)`` makes it. One JSON document (``asyncroscopy/contracts/action_space.schema.json``):

- ``actions``: every tool, with its arguments (type, units, range, choices, and the stage ``axis``
  or array ``elements`` where known), a one-line purpose, its risk as the approval gate sees it,
  whether it ``act``s or only ``observe``s, and a canonical ``role`` (``asyncroscopy.mcp.action_roles``);
- ``state``: what can be read, through a tool (``get_*``, ``State``) or as a Tango attribute, with units;
- ``jobs``: each job device's plan schema and example plan;
- ``instruments``: each device's kind (stem, probe, envelope).

Units and ranges come from, in order: the device's ``action_hints`` attribute (a YAML instrument
derives it from its file; an electron microscope gives its limits at its own min_fov, max_fov and
max_defocus), the tool's input schema (Tango types), and the units named in the command's doc. A controller that picks tools by role, not by name, runs unchanged on
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
from typing import TYPE_CHECKING, Any

from asyncroscopy.core.names import ACTION_HINTS, DEFAULT_MCP_URL, ENVELOPE_VERBS, ROLES, tool_name

if TYPE_CHECKING:
    from asyncroscopy.mcp.mcp_server import MCPServer

SCHEMA_VERSION = "1.0.0"
# Attributes that describe a device to the MCP server rather than the instrument's state.
META_ATTRIBUTES = frozenset({"state", "status", "job_kind", "plan_example", "plan_schema", ACTION_HINTS})
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


# Declaration keys of action_hints that the descriptors already carry (risk, role): not argument hints.
_DECLARED = ("risk", "operator_only", "stop", "routine", "role")


def build(server: MCPServer) -> dict[str, Any]:
    """The action space of an MCPServer after setup(), from its tool descriptors."""
    warnings: list[str] = list(server.discovery_warnings)
    actions: list[dict] = []
    instruments: list[dict] = []
    state: list[dict] = []
    jobs: list[dict] = []
    for prefix in sorted(server.tools):
        device = server.devices[prefix]
        kind = {"stem": device.stem, "probe": device.probe, "envelope": device.envelope}
        try:
            address = str(device.proxy.dev_name()) if device.proxy is not None else None
        except Exception:
            address = None
        instruments.append({"device": prefix, "class": device.dev_class, "address": address, "kind": kind})
        hints = device.hints
        for command in sorted(server.tools[prefix]):
            tool = server.descriptors.get(tool_name(prefix, command))
            registered = server.registered.get(tool.name) if tool is not None else None
            if registered is None:
                continue
            hint = {k: v for k, v in hints.get(command, {}).items() if k not in _DECLARED}
            purpose = _purpose(registered.description, prefix, command)
            action = {
                "tool": tool.name, "device": prefix, "command": command, "role": tool.action_role, "role_source": tool.role_source,
                "purpose": purpose, "risk": tool.risk, "effect": "act" if tool.acts else "observe",
                "args": _arguments(registered, hint.get("args")),
            }
            returns = _returns(registered, hint.get("returns"), purpose)
            if returns:
                action["returns"] = returns
            if hint.get("primary") is True:
                action["primary"] = True
            actions.append(action)
            if not tool.acts and not tool.action_role.startswith("job."):
                read = {"name": command[4:] if command.lower().startswith("get_") else command, "device": prefix, "source": "tool", "tool": tool.name, "role": tool.action_role}
                read.update({k: v for k, v in (returns or {}).items() if k in ("type", "items", "units", "axis", "elements", "kind")})
                state.append(read)
        if device.proxy is not None:
            state += _attributes(device.proxy, prefix, warnings)
        if device.envelope:
            jobs.append(_job(server, prefix, device))
    for name in sorted(server.native_names):
        tool, registered = server.descriptors.get(name), server.registered.get(name)
        if tool is None or registered is None:
            continue
        purpose = _purpose(registered.description, None)
        returns = _returns(registered, None, purpose)
        actions.append({
            "tool": name, "device": None, "command": None, "role": tool.action_role, "role_source": tool.role_source,
            "purpose": purpose, "risk": tool.risk, "effect": "act" if tool.acts else "observe",
            "args": _arguments(registered, None), **({"returns": returns} if returns else {}),
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
            text = str(device.proxy.read_attribute(name).value)
            return json.loads(text) if text else None
        except Exception:  # an older job device without it, or not JSON
            return None

    example = device.plan_example
    try:
        example = json.loads(example) if example else None
    except json.JSONDecodeError:
        pass
    verbs = {verb: tool_name(prefix, verb) for verb in server.tools.get(prefix, {}) if verb.lower() in ENVELOPE_VERBS | {"get_reservation"}}
    return {
        "device": prefix, "kind": device.job_kind, "plan_schema": attribute("plan_schema") if device.proxy is not None else None,
        "plan_example": example, "tools": verbs, "run_with": "run_job" if "run_job" in server.native_names else None,
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


def structured(result: Any) -> Any:
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
    parser.add_argument("--url", default=DEFAULT_MCP_URL, help=f"MCP endpoint (default {DEFAULT_MCP_URL})")
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
