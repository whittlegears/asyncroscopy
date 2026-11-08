"""Tool catalogs for agent tests: what endpoint_info would say the tools are, built the way the MCP server builds
them (each device's declared commands through the server's descriptors), without starting a stack."""

from __future__ import annotations

from pathlib import Path

from asyncroscopy.agents.catalog import ToolCatalog, ToolInfo
from asyncroscopy.instruments.yaml_sim.spec import action_hints, load_spec
from asyncroscopy.mcp.descriptors import NATIVE, DeviceInfo, describe_command, describe_native
from asyncroscopy.safety.policy import is_envelope_device, is_probe_device, is_stem_device
from asyncroscopy.utils.commands import declared_hints

EXAMPLES = Path(__file__).resolve().parents[1] / "asyncroscopy" / "instruments" / "yaml_sim" / "examples"
NATIVES = sorted(set(NATIVE) | {"approval_status", "find_tools", "call_tool", "endpoint_info", "action_space", "load_skill", "list_devices"})


def _commands(device_class: type) -> list[str]:
    return list(device_class.TangoClassClass.cmd_list) + ["State", "Status"]


def catalog(devices: dict[str, type | str], natives: list[str] = NATIVES) -> ToolCatalog:
    """The catalog of an endpoint with ``devices`` ({tool prefix: Tango device class, or a YAML example file name
    such as "raman.yaml"}) and the server's native tools."""
    tools = [ToolInfo(n, None, None, d.action_role, d.risk, d.acts) for n in natives for d in [describe_native(n)]]
    entries: list[tuple[str, dict, list[str]]] = []
    for prefix, source in devices.items():
        if isinstance(source, str):
            from asyncroscopy.instruments.yaml_sim.device import YamlInstrument
            from asyncroscopy.jobs.job_device import JobDevice

            spec = load_spec(EXAMPLES / source)
            hints = {name: hint for name, hint in declared_hints(YamlInstrument).items() if not hint.get("operator_only")} | action_hints(spec)
            entries.append((prefix, hints, list(hints) + ["State", "Status"]))
            if spec.jobs:  # the file's jobs are a device of their own, as run_servers starts them
                entries.append((f"{prefix}Job", declared_hints(JobDevice), _commands(JobDevice)))
        else:
            entries.append((prefix, declared_hints(source), _commands(source)))
    for prefix, hints, names in entries:
        device = DeviceInfo(prefix, prefix, stem=is_stem_device(names), probe=is_probe_device(names), envelope=is_envelope_device(names), hints=hints)
        for command in dict.fromkeys(names):
            d = describe_command(device, command, overrides={}, routine=None, warnings=[])
            tools.append(ToolInfo(d.name, prefix, command, d.action_role, d.risk, d.acts))
    return ToolCatalog(tools)


def twin_catalog() -> ToolCatalog:
    """The STEM twin stack's endpoint: the twin, a Mapper and a Tuner, and the Raman YAML twin beside it."""
    from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
    from asyncroscopy.jobs.mapping.mapper import Mapper
    from asyncroscopy.jobs.tuning.tuner import Tuner

    return catalog({"DigitalTwin": DigitalTwin, "Mapper": Mapper, "Tuner": Tuner, "RamanTwin": "raman.yaml", "XRDTwin": "xrd.yaml"})


def info_of(tools: ToolCatalog) -> dict:
    """An endpoint_info reply carrying ``tools`` (what a fake server answers)."""
    return {"tools": {t.name: {"device": t.device, "command": t.command, "role": t.role, "risk": t.risk, "acts": t.acts}
                      for t in tools.tools.values()}}
