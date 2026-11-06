"""FastMCP bridge for asyncroscopy Tango devices.

The server discovers the devices in the Tango database, decides once what each command is
(``asyncroscopy.mcp.descriptors``), and offers every command as an MCP tool. A tool call passes an
ordered list of interceptors (``asyncroscopy.mcp.interceptors``): the approval gate, the job checks,
logging and the trace, an acquisition's preview. The native tools (data, jobs, discovery, skills,
the trace and the emergency stop) come from the components in ``asyncroscopy.mcp.components``.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import inspect
import json
import re
import socket
import traceback
import warnings
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Annotated, Any, Callable

from fastmcp import FastMCP
from fastmcp.server.server import Transport
from fastmcp.tools import Tool, tool
from pydantic import Field
from tango import CmdArgType, CommandInfo, Database, DeviceProxy
from tango.utils import TO_TANGO_TYPE, is_array_type, is_bool_type, is_float_type, is_int_type, is_scalar_type, is_str_type

from asyncroscopy.core.names import CALLER_HEADER, ENVELOPE_VERBS, RUN_HEADER, tool_name
from asyncroscopy.core.net import bind_host, is_remote_host
from asyncroscopy.data.data_reader import numpy_to_python
from asyncroscopy.mcp import endpoint_auth
from asyncroscopy.mcp.components import data_tools, discovery, estop, gate, jobs, skills_tools, trace_tools
from asyncroscopy.mcp.descriptors import DeviceInfo, ToolDescriptor, describe_command, describe_native, read_hints, rule_risk
from asyncroscopy.mcp.interceptors import Call, Interceptor, run_call
from asyncroscopy.safety.policy import check_overrides, is_envelope_device, is_probe_device, is_read_only, is_stem_device, looks_like_stop, operator_only
from asyncroscopy.trace.clock import InstrumentClock
from asyncroscopy.trace.writer import ToolCallLog, TraceWriter

ENDPOINT_ROLES = ("agent", "operator", "monitor")
# Tango's client default of 3000 ms is shorter than a real acquisition: the digital twin's first
# acquire_camera_image takes over 3 s cold. SciAgentGUI allows 60 s per MCP HTTP request, so 30 s lets
# slow commands finish while still failing inside the client's window with a readable Tango error.
COMMAND_TIMEOUT_MILLIS = 30_000
JOB_LIMITS = ("max_duration_s", "max_dose_e_per_a2")


def _without_param_name(desc: str) -> str:
    """A parameter's description without its ":param name:" prefix, which only names it."""
    return re.sub(r"^(?::param|@param)\s+\w+:\s*", "", desc.strip())


def _placeholder_doc(text: str) -> bool:
    """True for the text PyTango fills in when a command has no doc_in/doc_out.

    Such text must never lead a tool description: models read the first line to
    pick a tool. Use asyncroscopy.utils.commands.command to carry the docstring.
    """
    # A plain tango.server.command gives ":param x: (not documented)\n:type x: DevDouble".
    t = re.sub(r"^(?::param|@param)\s+\w+:\s*", "", text.strip())
    t = "\n".join(line for line in t.splitlines() if not line.strip().startswith(":type")).strip().lower()
    return t in ("", "uninitialised", "uninitialized", "none", "(not documented)") or t.startswith((":return: (not documented)", "no input parameter", "no output parameter"))


@dataclass
class ServerConfig:
    """What the components are configured with (the MCP config's mcp section, resolved)."""

    name: str
    endpoint_role: str
    data_device_address: str
    approvals_device_address: str | None = None
    risk_overrides: dict[str, str] = field(default_factory=dict)
    auto_routine: dict[str, list[str]] | None = None  # mcp.safety.auto_routine; None: the devices' routine declarations
    analysis_url: str | None = None
    job_limits: dict[str, float] = field(default_factory=dict)
    trace_follow: list[str] = field(default_factory=list)
    verbose: bool = True
    auth: Any = None
    agent_profile: str | None = None  # agent profile file whose tool selection endpoint_info reports

    def high_overrides(self) -> dict[str, str]:
        """The mcp.safety.risk patterns that mark commands high: what a job's dry run must record, not pass through."""
        return {pattern: level for pattern, level in self.risk_overrides.items() if level == "high"}


class MCPServer:
    def __init__(
        self,
        name: str,
        tango_host: str,
        tango_port: int,
        blocked_functions: dict[str, list[str]],
        blocked_classes: list[str],
        data_device_address: str,
        include_only_functions: list[str] | None = None,
        verbose: bool = True,
        tool_log_path: str | None = None,
        endpoint_role: str = "agent",
        approvals_device_address: str | None = None,
        risk_overrides: dict[str, str] | None = None,
        trace_path: str | None = None,
        database: Any = None,
        analysis_url: str | None = None,
        job_limits: dict[str, float] | None = None,
        auth: dict | None = None,
        clock_device: str | None = None,
        trace_follow: list[str] | None = None,
        auto_routine: dict[str, list[str]] | None = None,
        agent_profile: str | None = None,
    ) -> None:
        """
        Args:
            name: Display name for the MCP server instance.
            tango_host, tango_port: The Tango database server (e.g. "localhost", 9094).
            blocked_functions: Command names to exclude, keyed by Tango class name. Use "*" for global blocks.
            blocked_classes: Tango device class names to skip entirely.
            data_device_address: Tango DATA device whose Tiled server list_data and view_data read.
            include_only_functions: Command names/patterns to allow exclusively.
            verbose: Print device discovery and tool registration progress.
            tool_log_path: JSONL file with one line per tool call. Off when empty.
            endpoint_role: "agent" (default) hides operator-only commands such as Approvals.approve and
                export_ground_truth; "operator" is for the GUI; "monitor" offers only pure reads (get_*,
                list_*, State, Status) and the trace, for a status panel or an agent that must not act.
            approvals_device_address: Approvals device holding the approval mode. Unset: every call runs.
            risk_overrides: {"Class.command" or tool-name glob: "low"|"high"}, checked before what devices declare.
            trace_path: Instrument Trace JSONL file. Off when empty.
            database: Stands in for the Tango database (anything with get_device_exported);
                asyncroscopy.testing uses it to serve the devices of a test context.
            analysis_url: MCP URL of the analysis server, so wait_for_job also waits on its jobs.
            job_limits: {"max_duration_s": ..., "max_dose_e_per_a2": ...}: run_job refuses a plan whose estimate is over either.
            auth: the MCP config's ``auth`` section (asyncroscopy.mcp.endpoint_auth); None: no token check.
            clock_device: Instrument whose clock_s stamps this server's trace events. Empty: this computer's time.
            trace_follow: Trace files, directories or globs the operator endpoint's trace_since reads.
                Empty: the directory of trace_path (every *.jsonl there).
            auto_routine: High-risk commands auto mode runs without a person, as {device kind ("stem", "probe",
                "envelope"): [command globs], "tools": [tool-name or Class.command globs]}; it replaces what the
                devices declare routine. None: the devices' declarations (an electron microscope's imaging,
                spectra, stage moves and beam placement; asyncroscopy.utils.commands.command(routine=True)).
            agent_profile: Agent profile YAML whose tool selection endpoint_info.agent_tools reports. None: the
                packaged default (asyncroscopy/agents/profiles/default.yaml).
        """
        if endpoint_role not in ENDPOINT_ROLES:
            raise ValueError(f"endpoint_role must be one of {ENDPOINT_ROLES}, got {endpoint_role!r}")
        limits = {k: float(v) for k, v in (job_limits or {}).items() if v not in (None, "")}
        unknown_limits = set(limits) - set(JOB_LIMITS)
        if unknown_limits:
            raise ValueError(f"unknown job_limits {sorted(unknown_limits)}; allowed: {', '.join(JOB_LIMITS)}")
        self.config = ServerConfig(
            name=name, endpoint_role=endpoint_role, data_device_address=data_device_address,
            approvals_device_address=approvals_device_address or None, risk_overrides=check_overrides(risk_overrides),
            auto_routine=None if auto_routine is None else dict(auto_routine), analysis_url=analysis_url or None,
            job_limits=limits, trace_follow=list(trace_follow or ([str(Path(trace_path).parent)] if trace_path else [])),
            verbose=verbose, auth=auth, agent_profile=agent_profile or None,
        )
        # ``database`` replaces the Tango database lookup (asyncroscopy.testing passes a device list).
        self.database = database if database is not None else Database(tango_host, tango_port)
        self.mcp = FastMCP(name)
        self.auth = None  # set at the end of __init__
        self.blocked_functions = {key: list(value) for key, value in blocked_functions.items()}
        self.blocked_classes = list(blocked_classes)
        self._blocked_classes_normalized = {cls_name.lower() for cls_name in self.blocked_classes}
        self.include_only_functions = list(include_only_functions) if include_only_functions else []
        self.verbose = verbose
        self.endpoint_role = endpoint_role
        self.tool_log = ToolCallLog(tool_log_path) if tool_log_path else None
        self.trace = TraceWriter(
            trace_path, source="mcp", writer=f"mcp:{name}:{endpoint_role}", clock=InstrumentClock(clock_device or ""),
        ) if trace_path else None
        self.devices: dict[str, DeviceInfo] = {}  # tool prefix -> its device
        self.descriptors: dict[str, ToolDescriptor] = {}  # tool name -> what it is (devices' and native tools)
        self.tools: dict[str, dict[str, Callable]] = {}  # tool prefix -> command -> its wrapper
        self.registered: dict[str, Tool] = {}  # tool name -> the MCP tool (description, input schema)
        self.native_names: set[str] = set()
        self.interceptors: list[Interceptor] = []
        self._lowered: set[str] = set()  # tools an override made low that the declarations or rules would hold
        self.risk_warnings: list[str] = []  # what setup() printed as [safety] lines
        self.risk_fallbacks: dict[str, list[str]] = {}  # tool prefix -> commands whose risk only the name rules decided
        self.discovery_warnings: list[str] = []
        # Whether the current caller's token may call a tool (endpoint_auth.attach replaces it); find_tools and call_tool ask it.
        self.may_call: Callable[[str], bool] = lambda name: True
        cfg = self.config
        self.data = data_tools.register(self, cfg)
        self.skills = skills_tools.register(self, cfg)
        self.discovery = discovery.register(self, cfg)
        self.estop = estop.register(self, cfg)
        self.jobs = jobs.register(self, cfg)
        self.gate = gate.register(self, cfg)
        self.traces = trace_tools.register(self, cfg)
        self.components = [self.data, self.skills, self.discovery, self.estop, self.jobs, self.gate, self.traces]
        self.auth = endpoint_auth.attach(self, auth)  # None without mcp.auth

    # ------------------------------------------------------------------ registries
    def add_interceptor(self, interceptor: Interceptor) -> None:
        """Add a step every applicable tool call passes through; the list stays sorted by ``order``."""
        self.interceptors.append(interceptor)
        self.interceptors.sort(key=lambda i: i.order)

    def add_device(self, prefix: str, proxy: Any = None, dev_class: str | None = None, **facts: Any) -> DeviceInfo:
        """Register the device behind a tool prefix (discovery does this; so can a test). ``facts``: DeviceInfo fields."""
        device = DeviceInfo(prefix=prefix, dev_class=dev_class or prefix, proxy=proxy, **facts)
        self.devices[prefix] = device
        return device

    @property
    def tool_risks(self) -> dict[str, str]:
        """Each device tool's risk as the approval gate sees it."""
        return {name: t.risk for name, t in self.descriptors.items() if not t.native}

    def tool_names(self) -> set[str]:
        """Names of every tool this endpoint offers (device and native)."""
        return {tool_name(prefix, command) for prefix, commands in self.tools.items() for command in commands} | self.native_names

    def caller(self) -> str | None:
        """Who is calling: with mcp.auth, the verified token's subject; without, the self-reported
        X-Asyncroscopy-Caller header. None outside an HTTP request."""
        if self.auth is not None:
            return endpoint_auth.token_subject()
        return self._header(CALLER_HEADER) or None

    def run_id(self) -> str:
        """The agent run of the current request (X-Asyncroscopy-Run; "" without one). Agents open a new MCP
        session for every call, so a held run_job is found again by caller and run, not by session."""
        return self._header(RUN_HEADER)

    @staticmethod
    def _header(name: str) -> str:
        try:
            from fastmcp.server.dependencies import get_http_headers

            return get_http_headers().get(name.lower()) or ""
        except Exception:
            return ""

    # ------------------------------------------------------------------ discovery
    def _is_blocked_class(self, class_name: str) -> bool:
        return class_name.lower() in self._blocked_classes_normalized

    def _list_all_devices(self) -> list[str]:
        return list(self.database.get_device_exported("*").value_string)

    @staticmethod
    def _is_admin_device(device_name: str) -> bool:
        return device_name.lower().startswith("dserver/")

    @tool()
    def list_devices(self) -> list[str]:
        """List available devices filtered by blocked classes."""
        available = []
        for device_name in self._list_all_devices():
            if self._is_admin_device(device_name):
                continue
            try:
                if not self._is_blocked_class(DeviceProxy(device_name).info().dev_class):
                    available.append(device_name)
            except Exception as exc:
                warnings.warn(f"list_devices: left out {device_name}: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
        return available

    def _find_tools(self) -> dict[str, dict[str, tuple[Callable, CommandInfo]]]:
        """Discover devices and their commands in the Tango database; registers each device (``add_device``).

        Returns tool prefix (the class name, or <Class>_<member> when several devices share a class)
        -> command name -> (func, cmd_info).
        """
        opened: list[tuple[str, DeviceProxy, str]] = []
        for device_name in self._list_all_devices():
            if self._is_admin_device(device_name):
                continue
            try:
                dev = DeviceProxy(device_name)
                dev.set_timeout_millis(COMMAND_TIMEOUT_MILLIS)
                dev_class = dev.info().dev_class
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {device_name}: failed to open proxy ({exc})")
                continue
            if not self._is_blocked_class(dev_class):
                opened.append((device_name, dev, dev_class))
                self._warn_if_fault(device_name, dev)
        # Tools are named <Class>_<command>. When several devices share a class (two robots, two job devices)
        # each gets its own prefix, <Class>_<member>, so neither hides the other and an approval for one never
        # covers the other.
        members: dict[str, list[str]] = {}
        families: dict[str, list[str]] = {}
        for name, _dev, dev_class in opened:
            members.setdefault(dev_class, []).append(name.split("/")[-1])
            families.setdefault(dev_class, []).append(name.split("/")[1] if name.count("/") == 2 else name)
        self.devices.clear()
        self.descriptors.clear()
        self._lowered.clear()
        self.discovery_warnings = []
        tools: dict[str, dict[str, tuple[Callable, CommandInfo]]] = {}
        for device_name, dev, dev_class in opened:
            prefix = dev_class
            if len(members[dev_class]) > 1:
                # The member ("lab/mapper/2" -> 2); else the family, which the launcher sets from the config key
                # ("asyncroscopy/grid2/default" -> grid2); else the whole device name.
                parts = device_name.split("/")
                member, family = parts[-1], (parts[1] if len(parts) == 3 else device_name)
                suffix = member if members[dev_class].count(member) == 1 else family if families[dev_class].count(family) == 1 else "_".join(parts)
                prefix = f"{dev_class}_{suffix}".replace("/", "_").replace("-", "_")
            try:
                commands = dev.command_list_query()
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {device_name}: failed to query commands ({exc})")
                continue
            names = [c.cmd_name for c in commands]
            hints, problem = read_hints(dev)
            if problem:
                self.discovery_warnings.append(f"{prefix}: {problem}")
            device = self.add_device(prefix, dev, dev_class, stem=is_stem_device(names), probe=is_probe_device(names),
                                     envelope=is_envelope_device(names), hints=hints)
            if device.envelope:
                device.job_kind = self._attribute(dev, "job_kind")
                device.plan_example = self._attribute(dev, "plan_example") or None
            for cmd in commands:
                command_name = cmd.cmd_name
                declared_operator = hints.get(command_name, {}).get("operator_only") is True
                if self.endpoint_role != "operator" and (declared_operator or operator_only(dev_class, command_name)):
                    continue
                if self.endpoint_role == "monitor":
                    name = tool_name(prefix, command_name).lower()
                    marked_high = any(level == "high" and fnmatchcase(name, pattern.lower()) for pattern, level in self.config.risk_overrides.items())
                    if marked_high or not (is_read_only(dev_class, command_name, self.config.risk_overrides) or command_name in ("job_status", "results")):
                        continue
                global_blocks = self.blocked_functions.get("*", [])
                if command_name in global_blocks or f"{dev_class}.{command_name}" in global_blocks or command_name in self.blocked_functions.get(dev_class, []):
                    continue
                if self.include_only_functions and not (
                    command_name in self.include_only_functions
                    or f"{dev_class}.{command_name}" in self.include_only_functions
                    or any(item.endswith(f".{command_name}") for item in self.include_only_functions)
                ):
                    continue
                try:
                    func = getattr(dev, command_name)
                except Exception as exc:
                    if self.verbose:
                        print(f"Skipping {device_name}.{command_name}: failed to resolve command ({exc})")
                    continue
                tools.setdefault(prefix, {})[command_name] = (func, cmd)
                device.in_types[command_name] = getattr(cmd, "in_type", None)
        return tools

    @staticmethod
    def _attribute(dev: DeviceProxy, name: str) -> str | None:
        try:
            return str(getattr(dev, name))
        except Exception:  # an older job device without it
            return None

    def _warn_if_fault(self, device_name: str, dev: DeviceProxy) -> None:
        """A device in FAULT (an instrument file that does not load) offers only what it could set up, often just the
        job envelope: say so, rather than serve its few tools as if the instrument were there."""
        try:
            state, status = str(dev.state()), str(dev.status())
        except Exception:
            return
        if state == "FAULT":
            print(f"[safety] {device_name} is in FAULT ({status.strip()[:300]}); its tools are registered but it cannot run them. Fix it and restart this server", flush=True)

    # ------------------------------------------------------------------ wrapping
    @staticmethod
    def _tango_type_to_python(cmd_type: CmdArgType) -> Any:
        if cmd_type == CmdArgType.DevVoid:
            return type(None)
        if cmd_type == CmdArgType.DevEncoded:
            return dict
        if is_scalar_type(cmd_type):
            if is_bool_type(cmd_type):
                return bool
            if is_float_type(cmd_type):
                return float
            if is_int_type(cmd_type):
                return int
            if is_str_type(cmd_type):
                return str
            candidates = [py_type for py_type, tango_type in TO_TANGO_TYPE.items() if tango_type == cmd_type and isinstance(py_type, type)]
            if not candidates:
                return Any
            return next((t for t in candidates if t.__module__ == "builtins"), candidates[0])
        if is_array_type(cmd_type):
            if is_bool_type(cmd_type, inc_array=True):
                return list[bool]
            if is_float_type(cmd_type, inc_array=True):
                return list[float]
            if is_int_type(cmd_type, inc_array=True):
                return list[int]
            if is_str_type(cmd_type, inc_array=True):
                return list[str]
            return list
        return Any

    @staticmethod
    def _normalize_command_result(out_type: CmdArgType, result: Any) -> Any:
        """Convert Tango command output into JSON-safe data for MCP transport."""
        result = numpy_to_python(result)  # numpy types, nested too, as plain Python
        if out_type != CmdArgType.DevEncoded or not isinstance(result, tuple) or len(result) != 2:
            return result
        metadata_raw, payload_raw = result
        metadata_str = metadata_raw.decode("utf-8", errors="replace") if isinstance(metadata_raw, bytes) else str(metadata_raw)
        try:
            metadata = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            metadata = metadata_str
        if isinstance(payload_raw, memoryview):
            payload_bytes = payload_raw.tobytes()
        elif isinstance(payload_raw, (bytes, bytearray)):
            payload_bytes = bytes(payload_raw)
        else:
            payload_bytes = str(payload_raw).encode("utf-8")
        return {"encoding": "base64", "metadata": metadata, "payload": base64.b64encode(payload_bytes).decode("ascii")}

    def _describe(self, prefix: str, dev_class: str, command_name: str, in_type: Any) -> ToolDescriptor:
        """Decide what one device command is, once, and keep it in ``descriptors``."""
        device = self.devices.get(prefix) or self.add_device(prefix, None, dev_class)
        device.in_types.setdefault(command_name, in_type)
        descriptor = describe_command(device, command_name, overrides=self.config.risk_overrides, routine=self.config.auto_routine,
                                      warnings=self.discovery_warnings)
        if descriptor.risk == "low" and descriptor.risk_source == "override" and rule_risk(device, command_name) == "high":
            self._lowered.add(descriptor.name)
        self.descriptors[descriptor.name] = descriptor
        return descriptor

    def _create_wrapper(self, func: Callable, cmd_info: CommandInfo, command_name: str, dev_class: str, tool_prefix: str | None = None) -> Callable:
        """A tool function, with a proper signature and description, for one Tango command.

        Args:
            func: The raw Tango device command method.
            cmd_info: The CommandInfo object from Tango.
            command_name: The name of the command.
            dev_class: The Tango device class name.
            tool_prefix: Tool-name prefix; defaults to the class name.
        """
        prefix = tool_prefix or dev_class
        in_type, in_desc, out_type = cmd_info.in_type, cmd_info.in_type_desc, cmd_info.out_type
        py_type = self._tango_type_to_python(in_type)
        descriptor = self._describe(prefix, dev_class, command_name, in_type)
        unique_name = descriptor.name
        device = self.devices[prefix]
        chain = [i for i in self.interceptors if i.applies(descriptor)]

        def _run(arguments: dict, *call_args: Any) -> Any:
            call = Call(tool=descriptor, arguments=arguments, call_args=call_args, caller=self.caller(), run=self.run_id() or None)
            return run_call(call, chain, lambda args: self._normalize_command_result(out_type, func(*args)))

        py_return_type = self._tango_type_to_python(out_type)
        doc_lines = [f"Tango Device Class: {dev_class}", f"Tango Command: {command_name}"]
        # A DevVoid type line tells a model nothing; small models read every line.
        if in_type != CmdArgType.DevVoid:
            doc_lines.append(f"Input Type: {in_type.name}")
        if in_desc and not _placeholder_doc(in_desc) and _without_param_name(in_desc):
            doc_lines.append(f"Input Description: {_without_param_name(in_desc)}")
        if out_type != CmdArgType.DevVoid:
            doc_lines.append(f"Output Type: {out_type.name}")
        out_desc = cmd_info.out_type_desc
        if out_desc and not _placeholder_doc(out_desc):
            # Lead with what the command is for (once): models weigh the first line most when choosing a tool.
            doc_lines.insert(0, out_desc)
        if device.envelope and device.job_kind and command_name.lower() in ENVELOPE_VERBS:
            doc_lines.insert(0, f"{prefix} job: {device.job_kind}.")
        if device.envelope and command_name == "reserve" and device.plan_example:
            doc_lines.insert(1, f'Example request: {{"plan": {device.plan_example}}}')
        doc = "\n".join(doc_lines)

        # The parameter's name comes from the description's ":param <name>:"; otherwise "arg".
        match = re.search(r"(?::param|@param)\s+(\w+):", in_desc or "")
        param_name = match.group(1) if match else "arg"
        if in_desc and not _placeholder_doc(in_desc):
            clean_desc = _without_param_name(in_desc.replace("\n", " "))
            arg_type = Annotated[py_type, Field(description=clean_desc)]
        else:
            clean_desc = ""
            arg_type = py_type
        # A string parameter documented as JSON also accepts the object itself: models pass
        # {"job_id": "..."} far more reliably than a JSON-encoded string of it.
        json_string = py_type is str and clean_desc.upper().startswith("JSON")
        if json_string:
            arg_type = Annotated[str | dict[str, Any] | list[Any], Field(description=clean_desc)]

        if in_type == CmdArgType.DevVoid:
            def wrapper():
                return _run({})

            params = []
        elif py_type is dict:
            # For commands taking a dictionary (like DevEncoded), allow arbitrary keyword arguments.
            def wrapper(**kwargs):
                arg_input = kwargs["arg"] if "arg" in kwargs and len(kwargs) == 1 and isinstance(kwargs["arg"], dict) else kwargs
                return _run({"arg": arg_input}, arg_input)

            params = [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)]
        elif json_string:
            # JSON-object commands (job envelope, approvals): accept {"request": {...}}, {"request": "<json>"} or the
            # object's fields given directly, e.g. {"plan": {...}}. The fields come from the description's example
            # object; FastMCP does not allow **kwargs.
            fields = [f for f in dict.fromkeys(re.findall(r'"(\w+)"\s*:', clean_desc)) if f != param_name]

            def wrapper(*args, **kwargs):
                arg = args[0] if args else kwargs.get(param_name)
                if arg is None:
                    arg = {k: v for k, v in kwargs.items() if k in fields and v is not None}
                if isinstance(arg, str):
                    try:
                        arg = json.loads(arg)
                    except json.JSONDecodeError:
                        pass  # not JSON: the device reports the error
                if isinstance(arg, (dict, list)):
                    # One canonical text for one call, whatever form it came in, so the approval fingerprint of a
                    # retried call matches its grant.
                    arg = json.dumps(arg, sort_keys=True, separators=(",", ":"))
                return _run({param_name: numpy_to_python(arg)}, arg)

            params = [inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[arg_type | None, Field(description=clean_desc)], default=None)]
            params += [inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[Any, Field(description=f"Same as {param_name}.{f}.")], default=None) for f in fields]
        else:
            def wrapper(*args, **kwargs):
                arg = args[0] if args else kwargs.get(param_name)
                return _run({param_name: numpy_to_python(arg)}, arg)

            params = [inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=arg_type)]

        # Annotations without VAR_KEYWORD parameters, so Pydantic allows dynamic extra fields.
        wrapper.__annotations__ = {p.name: p.annotation for p in params if p.kind != inspect.Parameter.VAR_KEYWORD}
        wrapper.__annotations__["return"] = py_return_type
        wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=py_return_type)
        wrapper.__doc__ = doc
        wrapper.__name__ = unique_name
        wrapper.__qualname__ = unique_name
        return wrapper

    # ------------------------------------------------------------------ setup
    def _warn_about_risks(self) -> None:
        """Print what an instrument author most likely got wrong in the risk settings: an override that matches no
        command, a stop-like command that approve mode holds, and a command an override made low."""
        found = []
        devices = [t for t in self.descriptors.values() if not t.native]
        example = next((t.name for t in devices if not t.name.endswith(("_State", "_Status"))), "Class_command")
        for pattern in self.config.risk_overrides:
            p = pattern.lower()
            # An operator-only command (declared, or Tango's admin commands) is not on this endpoint, but real.
            hidden = any(fnmatchcase(f"{d.dev_class}.{c}".lower(), p) or fnmatchcase(tool_name(d.prefix, c).lower(), p)
                         for d in self.devices.values() for c, h in d.hints.items() if h.get("operator_only") is True)
            hidden = hidden or ("." in pattern and operator_only(*pattern.split(".", 1)))
            if not hidden and not any(fnmatchcase(t.name.lower(), p) or fnmatchcase(f"{t.dev_class}.{t.command}".lower(), p) for t in devices):
                found.append(f"risk override {pattern!r} matches no command; write it as Class.command or a tool name such as {example}")
        for t in devices:
            if self.endpoint_role == "agent" and t.risk == "high" and looks_like_stop(t.command or ""):
                found.append(f"{t.name} looks like a stop but is high: approve mode holds it. If it only stops, switches off a source or lifts a probe "
                             f"away from the sample (not if it opens, moves or frees anything), add {{\"{t.name}\": \"low\"}} to mcp.safety.risk")
        for name in sorted(self._lowered if self.endpoint_role == "agent" else ()):
            found.append(f"{name} would be held by default; an mcp.safety.risk override makes it low, so approve mode lets agents run it")
        self.risk_warnings = found
        for line in found + self.discovery_warnings:
            print(f"[safety] {line}")

    def _report_fallbacks(self) -> None:
        """Print which commands declare no risk, so the name rules decided it: a device should declare them."""
        fallbacks: dict[str, list[str]] = {}
        for t in self.descriptors.values():
            if t.risk_source == "rule":
                fallbacks.setdefault(t.device or "", []).append(t.command or "")
        self.risk_fallbacks = {prefix: sorted(commands) for prefix, commands in sorted(fallbacks.items())}
        for prefix, commands in self.risk_fallbacks.items():
            print(f"[safety] {prefix} declares no risk for {len(commands)} command(s); the name rules decided: {', '.join(commands)}")

    def _natives(self) -> list[tuple[Any, Callable]]:
        """(component, tool) for each native tool this endpoint offers, in registration order."""
        found: list[tuple[Any, Callable]] = []
        for component in self.components:
            found += [(component, t) for t in component.natives()]
            if component is self.data:
                found.append((self, self.list_devices))
        return found

    def setup(self, print_summary: bool = True) -> None:
        """Discover the devices, decide what each command is, and add every tool to the MCP instance."""
        raw_tools = self._find_tools()
        exported = {str(device.proxy.dev_name()).lower() for device in self.devices.values() if device.proxy is not None}
        address = self.config.approvals_device_address
        if address and address.lower() not in exported:
            print(f"[safety] approvals device {address} is not among the exported devices; while it is not running, "
                  "gated calls are refused. Start this server after run_servers.py prints ready, or check safety.approvals_device_address", flush=True)
        self.tools = {
            prefix: {command: self._create_wrapper(func, cmd_info, command, self.devices[prefix].dev_class, tool_prefix=prefix)
                     for command, (func, cmd_info) in raw_tools[prefix].items()}
            for prefix in raw_tools
        }
        if print_summary and self.verbose:
            print("Discovered tools by Tango class:")
            for prefix in sorted(raw_tools):
                print(f"- {prefix}: {len(raw_tools[prefix])}")
                for command_name in sorted(raw_tools[prefix]):
                    print(f"    - {command_name}")
        if self.trace is not None and "simulated" not in self.trace.defaults and address and address.lower() in exported:
            try:  # the stack's simulated flag, as its Approvals device holds it (run_servers' top-level simulated:)
                self.trace.defaults["simulated"] = bool(self.gate.approvals().get_simulated())
            except Exception:
                pass  # unknown: the events leave it out
        natives = self._natives()
        self.native_names = {getattr(t, "__name__", getattr(t, "name", "")) for _c, t in natives}
        for component, native_tool in natives:
            name = getattr(native_tool, "__name__", getattr(native_tool, "name", ""))
            description = getattr(component, "description_of", lambda _n: None)(name)
            added = self.mcp.add_tool(Tool.from_function(native_tool, description=description) if description else native_tool)
            if isinstance(added, Tool):
                self.registered[added.name] = added
            self.descriptors[name] = describe_native(name)
            if self.verbose:
                print(f"Registered native tool: {name}")
        num_device_tools = 0
        for prefix, commands in self.tools.items():
            for command_name, wrapped in commands.items():
                try:
                    tool_obj = Tool.from_function(wrapped)
                    self.mcp.add_tool(tool_obj)
                    self.registered[tool_obj.name] = tool_obj
                    num_device_tools += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to wrap {prefix}.{command_name}: {e}")
                        traceback.print_exc()
        self._warn_about_risks()
        self._report_fallbacks()
        # Printed unconditionally so GUIs can parse the final count from stdout even when quiet mode is on.
        print(f"MCP ready: {len(natives) + num_device_tools} tool(s) registered", flush=True)
        if print_summary and self.verbose:
            print(f"\nRegistered {len(natives)} native tool(s)\nRegistered {num_device_tools} Tango device command tool(s)\nTotal: {len(natives) + num_device_tools} tools")
            print("\nAll MCP tools available:")
            for prefix in sorted(self.tools):
                for command_name in sorted(self.tools[prefix]):
                    wrapped = self.tools[prefix][command_name]
                    print(f"  - {prefix}.{command_name}{inspect.signature(wrapped)}")
                    for line in (wrapped.__doc__ or "").split("\n"):
                        if line.strip():
                            print(line.strip())
                    print("")
                print("")

    def start(self, transport: Transport | None = None, **kwargs: Any) -> None:
        """Synchronize with the Tango database and serve the MCP protocol ("stdio", "http", "sse" or "streamable-http")."""
        self.setup()
        self.mcp.run(transport=transport, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tango-host", required=True)
    parser.add_argument("--tango-port", type=int, required=True)
    parser.add_argument("--transport", required=True)
    parser.add_argument("--http-host", required=True)
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--blocked-classes-json", required=True)
    parser.add_argument("--blocked-functions-json", required=True)
    parser.add_argument("--include-only-functions-json", default="[]")
    parser.add_argument("--data-device-address", required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--tool-log", default="", help="JSONL file for per-call tool logging; off when empty.")
    parser.add_argument("--endpoint-role", default="agent", choices=ENDPOINT_ROLES)
    parser.add_argument("--approvals-device-address", default="", help="Approvals device; empty disables the approval mode.")
    parser.add_argument("--risk-json", default="{}", help='{"Class.command" glob: "low"|"high"} overrides.')
    parser.add_argument("--auto-routine-json", default="", help='{"stem": [command globs], "tools": [tool globs]}: what auto mode runs; empty: the packaged profile.')
    parser.add_argument("--agent-profile", default="", help="Agent profile YAML whose tool selection endpoint_info reports; empty: the default.")
    parser.add_argument("--trace", default="", help="Instrument Trace JSONL file; off when empty.")
    parser.add_argument("--clock-device", default="", help="Instrument whose clock_s stamps trace events; empty: this computer's time.")
    parser.add_argument("--trace-follow-json", default="[]", help="Trace files, directories or globs trace_since reads (operator); default: the --trace file's directory.")
    parser.add_argument("--data-views-json", default="[]", help="Modules to import first; each registers data kinds with asyncroscopy.data.views.")
    parser.add_argument("--analysis-url", default="", help="MCP URL of the analysis server, so wait_for_job also waits on its jobs.")
    parser.add_argument("--job-limits-json", default="{}", help='{"max_duration_s": ..., "max_dose_e_per_a2": ...}: run_job refuses plans over these.')
    parser.add_argument("--allow-remote-bind", action="store_true", help="Bind 0.0.0.0 even when the stack is not simulated (its Approvals device says so); otherwise such a stack binds 127.0.0.1.")
    parser.add_argument("--auth-json", default="", help="The MCP config's auth section (bearer-token verification); empty: none.")
    return parser.parse_args(argv)


def check_port_free(host: str, port: int) -> str | None:
    """An error string if (host, port) cannot be bound, else None: a previous MCP server may still hold the port."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        return str(exc)
    finally:
        probe.close()
    return None


def stack_simulated(tango_host: str, tango_port: int, approvals_address: str | None) -> bool:
    """Whether the stack drives a simulator, as its Approvals device says (get_simulated).

    No approvals device, or one that cannot be read, counts as real hardware: the safe answer when it is not known.
    """
    if not approvals_address:
        return False
    try:
        proxy = DeviceProxy(f"tango://{tango_host}:{tango_port}/{approvals_address}")
        proxy.set_timeout_millis(3000)
        return bool(proxy.get_simulated())
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    auth = json.loads(args.auth_json) if args.auth_json else None
    if args.transport == "streamable-http":
        simulated = is_remote_host(args.http_host) and not args.allow_remote_bind and stack_simulated(args.tango_host, args.tango_port, args.approvals_device_address)
        args.http_host, note = bind_host(args.http_host, args.allow_remote_bind, simulated)
        if note:
            print(note, flush=True)
        if not auth and is_remote_host(args.http_host):  # run_mcp checks this too; a direct start must not skip it
            refused = endpoint_auth.remote_without_auth(args.http_host, stack_simulated(args.tango_host, args.tango_port, args.approvals_device_address), auth)
            if refused:
                print(f"MCP ERROR: {refused}", flush=True)
                return 1
        bind_error = check_port_free(args.http_host, args.http_port)
        if bind_error:
            # "MCP ERROR:" is parsed by startup GUIs; keep the prefix stable.
            print(f"MCP ERROR: http://{args.http_host}:{args.http_port} is already in use - another MCP server is likely still running "
                  f"with a different config. Stop it or choose a different port. ({bind_error})", flush=True)
            return 1
    for module in json.loads(args.data_views_json):
        try:
            importlib.import_module(module)  # registers its data kinds for view_data and previews
        except Exception as exc:
            print(f"MCP ERROR: mcp.data_views lists {module!r}, which could not be imported: {type(exc).__name__}: {exc}", flush=True)
            return 1
    server = MCPServer(
        name=args.name, tango_host=args.tango_host, tango_port=args.tango_port,
        blocked_classes=json.loads(args.blocked_classes_json), blocked_functions=json.loads(args.blocked_functions_json),
        include_only_functions=json.loads(args.include_only_functions_json), data_device_address=args.data_device_address,
        verbose=not args.quiet, tool_log_path=args.tool_log or None, endpoint_role=args.endpoint_role,
        approvals_device_address=args.approvals_device_address or None, risk_overrides=json.loads(args.risk_json),
        trace_path=args.trace or None, analysis_url=args.analysis_url or None, job_limits=json.loads(args.job_limits_json),
        auth=auth, clock_device=args.clock_device or None, trace_follow=json.loads(args.trace_follow_json) or None,
        auto_routine=json.loads(args.auto_routine_json) if args.auto_routine_json else None, agent_profile=args.agent_profile or None,
    )
    if args.transport == "streamable-http":
        print(f"Starting {args.name} at http://{args.http_host}:{args.http_port}/mcp for Tango DB {args.tango_host}:{args.tango_port}", flush=True)
        server.start(transport="streamable-http", host=args.http_host, port=args.http_port)
    else:
        server.start(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
