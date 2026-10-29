"""FastMCP bridge for asyncroscopy Tango devices."""

import argparse
import asyncio
import base64
import importlib
import inspect
import re
import json
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Annotated, Any, Callable, Literal

import httpx
from pydantic import Field

import tango
from tango import Database, DeviceProxy, CommandInfo, CmdArgType
from tango.utils import (
    TO_TANGO_TYPE,
    is_array_type,
    is_scalar_type,
    is_bool_type,
    is_float_type,
    is_int_type,
    is_str_type,
)

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import tool, Tool, ToolResult
from fastmcp.server.server import Transport
from fastmcp.utilities.types import Image as MCPImage
from mcp.types import TextContent

from asyncroscopy.data import tiled_client, views
from asyncroscopy.data.data import REGISTRABLE_SUFFIXES
from asyncroscopy.data.data_reader import numpy_to_python
from asyncroscopy.mcp.tool_log import ToolCallLog, TraceWriter
from fnmatch import fnmatchcase

from asyncroscopy.safety.policy import ENVELOPE_VERBS, check_overrides, is_envelope_device, is_stem_device, looks_like_stop, operator_only, risk_of

ENDPOINT_ROLES = ("agent", "operator")
CALLER_HEADER = "x-asyncroscopy-caller"

def _without_param_name(desc: str) -> str:
    """A parameter's description without its ":param name:" prefix, which only names it."""
    return re.sub(r"^(?::param|@param)\s+\w+:\s*", "", desc.strip())


# acquire* commands whose result is a data key (a file name such as
# "spectrum_eds_20261028T101500.h5") get a PNG of what they saved, drawn by
# asyncroscopy.data.views, so clients that render MCP image blocks (e.g. SciAgentGUI)
# show the capture without another call. A preview never fails the acquisition: for
# PREVIEW_COMMANDS the result says why there is none; other commands just return.
PREVIEW_COMMANDS = {"acquire_camera_image", "acquire_scanned_image", "acquire_spectrum"}
DATA_KEY = re.compile(r"^[\w.\-/]+(" + "|".join(re.escape(s) for s in REGISTRABLE_SUFFIXES) + r")$", re.IGNORECASE)
# Data tools answer within this, so a slow or stuck Tiled cannot hang the agent.
DATA_TOOL_DEADLINE_S = 20.0
PREVIEW_DEADLINE_S = 10.0
LIST_DATA_WINDOW = 300  # newest keys list_data looks through; Tiled refuses pages larger than 300
# Job states wait_for_job keeps waiting through, and what it tells the model when its wait ends first.
WAIT_NOTES = {
    "queued": "Still working, which is normal for a long job: call wait_for_job again. Abort only if the user asked to stop.",
    "running": "Still working, which is normal for a long job: call wait_for_job again. Abort only if the user asked to stop.",
    "aborting": "Abort in progress: call wait_for_job again to see the job stop.",
    "paused_suspended": "Paused by a safety condition ({reason}); it resumes by itself when the condition clears. Call wait_for_job again, or abort if the user asked to stop.",
    "paused_awaiting_approval": "Waiting for an operator to approve the next step: call wait_for_job again. Do not retry or start another job.",
}
LIST_DATA_BUDGET_S = 12.0  # list_data stops looking after this and says how far it got


class _Busy(RuntimeError):
    """Every slot of a lane is taken by calls still waiting for Tiled."""


class _Lane:
    """A small thread pool with a slot count: work that outlives its deadline keeps its slot until it ends,
    so a stuck Tiled fills one lane with a clear "busy" answer instead of starving every data tool."""

    def __init__(self, name: str, slots: int):
        self.name, self.slots = name, slots
        self.pool = ThreadPoolExecutor(max_workers=slots, thread_name_prefix=f"data-{name}")
        self.free = threading.BoundedSemaphore(slots)

    def run(self, fn: Callable[[], Any], deadline_s: float) -> Any:
        if not self.free.acquire(blocking=False):
            raise _Busy(f"{self.slots} earlier {self.name} calls are still waiting for Tiled")
        future = self.pool.submit(fn)
        future.add_done_callback(lambda _f: self.free.release())
        return future.result(timeout=deadline_s)


_lanes = {"view": _Lane("view_data", 4), "list": _Lane("list_data", 2), "preview": _Lane("preview", 2)}
# What to do next when a lane is busy or Tiled is slow. A preview never suggests trying again:
# that could re-run the acquisition.
_NEXT_STEP = {
    "view": "Try again shortly, or ask for less (a region, or format text).",
    "list": "Try again shortly, or narrow it with search or kind.",
    "preview": "",
}

# Tango's client default of 3000 ms is shorter than a real acquisition: the digital
# twin's first acquire_camera_image takes over 3 s cold, so the call died with
# API_DeviceTimedOut while the command kept running server-side. SciAgentGUI allows
# 60 s per MCP HTTP request, so 30 s lets slow commands finish while still failing
# inside the client's window with a readable Tango error rather than an HTTP timeout.
COMMAND_TIMEOUT_MILLIS = 30_000
JOB_WAIT_MAX_S = 25.0  # wait_for_job stays under MCP client timeouts
RUN_JOB_WAIT_S = 15.0  # run_job waits this long after starting, so a short job needs no wait_for_job
RUN_JOB_BUDGET_S = 25.0  # run_job as a whole (reserve, gate, start, wait) stays under the 30 s client timeout
HELD_SWEEP_S = 5.0  # how often held run_job reservations are checked against their approval requests
HELD_MAX_S = 2 * 3600.0  # a held reservation nobody came back for is released after this long
MAX_HELD_RUNS = 32  # held reservations kept at once; the oldest is released beyond this

# From this many identical failures in a row, tell the model to stop and report.
REPEAT_FAILURE_WARN_AT = 2
# Identical failures further apart than this don't count as a loop.
REPEAT_FAILURE_WINDOW_S = 120.0


def describe_error(exc: Exception) -> str:
    """One-line summary of a failure; for DevFailed, the first (cause) and last (device) errors, no stack."""
    if isinstance(exc, tango.DevFailed) and exc.args:
        errors = [exc.args[0]] + ([exc.args[-1]] if len(exc.args) > 1 else [])
        return " | ".join(f"{e.reason}: {' '.join(str(e.desc).split())}" for e in errors)
    return f"{type(exc).__name__}: {exc}".strip()



def _placeholder_doc(text: str) -> bool:
    """True for the text PyTango fills in when a command has no doc_in/doc_out.

    Such text must never lead a tool description: models read the first line to
    pick a tool. Use asyncroscopy.utils.commands.command to carry the docstring.
    """
    # A plain tango.server.command gives ":param x: (not documented)\n:type x: DevDouble".
    t = re.sub(r"^(?::param|@param)\s+\w+:\s*", "", text.strip())
    t = "\n".join(line for line in t.splitlines() if not line.strip().startswith(":type")).strip().lower()
    return t in ("", "uninitialised", "uninitialized", "none", "(not documented)") or t.startswith((":return: (not documented)", "no input parameter", "no output parameter"))

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
        database=None,
        analysis_url: str | None = None,
        job_limits: dict[str, float] | None = None,
    ):
        """
        Args:
            name (str): Display name for the MCP server instance.
            tango_host (str): Hostname of the Tango database server (e.g. "localhost").
            tango_port (int): Port of the Tango database server (e.g. 9094).
            blocked_functions: Command names to exclude, keyed by Tango class name.
                Use "*" for global blocks.
            blocked_classes: Tango device class names to skip entirely.
            data_device_address: Tango DATA device whose Tiled server list_data and view_data read.
            include_only_functions (list[str], optional): Command names/patterns to allow exclusively.
            verbose (bool, optional): If True, print device discovery and tool registration
                progress to stdout. Defaults to True.
            tool_log_path (str, optional): JSONL file with one line per tool call. Off when empty.
            endpoint_role: "agent" (default) hides operator-only commands such as
                Approvals.approve and export_ground_truth; "operator" is for the GUI.
            approvals_device_address: Approvals device holding the approval mode.
                When unset, every call runs (there is no mode to consult).
            risk_overrides: {"Class.command" or tool-name ("Class_command") glob: "low"|"high"},
                checked before the defaults.
            trace_path: Instrument Trace JSONL file. Off when empty.
            database: Stands in for the Tango database (anything with get_device_exported);
                asyncroscopy.testing uses it to serve the devices of a test context.
            analysis_url: MCP URL of the analysis server (pyTEMlib), so wait_for_job also
                waits on its stitching and scoring jobs. Off when empty. Agents read it from
                endpoint_info, so it is configured once, here.
            job_limits: {"max_duration_s": ..., "max_dose_e_per_a2": ...}: run_job refuses a plan
                whose estimate is over either (the agent must ask the user; an operator raises the
                limit in the config). Empty: no limit beyond the job device's own.
        """
        if endpoint_role not in ENDPOINT_ROLES:
            raise ValueError(f"endpoint_role must be one of {ENDPOINT_ROLES}, got {endpoint_role!r}")
        # ``database`` replaces the Tango database lookup (asyncroscopy.testing passes a device list).
        self.database = database if database is not None else Database(tango_host, tango_port)
        self.mcp = FastMCP(name)

        self.blocked_functions = {key: list(value) for key, value in blocked_functions.items()}
        self.blocked_classes = list(blocked_classes)
        self._blocked_classes_normalized = {cls_name.lower() for cls_name in self.blocked_classes}
        self.data_device_address = data_device_address
        self.include_only_functions = list(include_only_functions) if include_only_functions else []
        self.verbose = verbose
        self.tool_log = ToolCallLog(tool_log_path) if tool_log_path else None
        self.endpoint_role = endpoint_role
        self.approvals_device_address = approvals_device_address or None
        self.risk_overrides = check_overrides(risk_overrides)
        self.trace = TraceWriter(trace_path, source="mcp") if trace_path else None
        # tool name -> (arguments + error signature, consecutive count, time of last failure)
        self._failures: dict[str, tuple[str, int, float]] = {}
        self._failures_lock = threading.Lock()
        self.analysis_url = analysis_url or None
        self.job_limits = {k: float(v) for k, v in (job_limits or {}).items() if v not in (None, "")}
        unknown_limits = set(self.job_limits) - {"max_duration_s", "max_dose_e_per_a2"}
        if unknown_limits:
            raise ValueError(f"unknown job_limits {sorted(unknown_limits)}; allowed: max_duration_s, max_dose_e_per_a2")
        # (caller, MCP session, device, plan) -> {reservation_id, request_id, device, created, summary} of a
        # run_job whose start is held, so repeating the same run_job after approval starts that reservation
        # instead of a new one. A denied or expired request, or one nobody came back for, frees it
        # (_sweep_held runs on approval_status, run_job and wait_for_job).
        self._held_runs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._held_runs_lock = threading.Lock()
        self._run_job_lock = threading.Lock()  # one reserve-and-start at a time: identical retries never file twice
        self._last_sweep = 0.0
        self._native_names: set[str] = set()
        self.tools: dict[str, dict[str, Callable]] = {}
        self._devices: dict[str, DeviceProxy] = {}  # tool prefix -> its device
        # Filled by _find_tools; set here so a wrapper built before it sees empty sets, not an AttributeError.
        self._prefix_class: dict[str, str] = {}
        self._envelope_prefixes: set[str] = set()
        self._stem_prefixes: set[str] = set()
        self._job_kinds: dict[str, str] = {}
        self._plan_examples: dict[str, str] = {}
        self.tool_risks: dict[str, str] = {}  # tool name -> "low" | "high", as the approval gate sees it
        self._risk_keys: dict[str, str] = {}  # tool name -> "Class.command", the other name overrides match
        self._lowered: set[str] = set()  # tools an override made low that the policy would hold
        self.risk_warnings: list[str] = []  # what setup() printed as [safety] lines

    def _is_blocked_class(self, class_name: str) -> bool:
        """Return True when a Tango class should be filtered out."""
        return class_name.lower() in self._blocked_classes_normalized

    def _list_all_devices(self) -> list[str]:
        """List all devices exported in the Tango DB."""
        devices = self.database.get_device_exported("*")
        return list(devices.value_string)

    @staticmethod
    def _is_admin_device(device_name: str) -> bool:
        """Return True for Tango admin (dserver) devices."""
        return device_name.lower().startswith("dserver/")

    @tool()
    def list_devices(self) -> list[str]:
        """List available devices filtered by blocked classes."""
        all_devices = self._list_all_devices()
        available = []
        for device_name in all_devices:
            if self._is_admin_device(device_name):
                continue
            try:
                dev = DeviceProxy(device_name)
                dev_class = dev.info().dev_class
                if not self._is_blocked_class(dev_class):
                    available.append(device_name)
            except Exception:
                pass
        return available

    def _tiled_node(self, key: str, hints: bool = True) -> Any:
        """The Tiled node for a DATA/Tiled key, through the DATA device's Tiled address.

        With hints, a missing key's error names keys that start with it (a scan of the
        catalog); without, a missing key is a plain KeyError.
        """
        uri = tiled_client.uri_from_data_device(self.data_device_address)
        return tiled_client.node_for_key(key, uri) if hints else tiled_client.connect(uri)[key]

    @staticmethod
    def _with_deadline(fn: Callable[[], Any], lane: str = "view", deadline_s: float | None = None) -> Any:
        """Run a data read in its lane within the deadline; every failure becomes an error the agent can act on."""
        deadline_s = DATA_TOOL_DEADLINE_S if deadline_s is None else deadline_s
        next_step = _NEXT_STEP[lane]
        try:
            return _lanes[lane].run(fn, deadline_s)
        except ToolError:
            raise
        except _Busy as exc:
            raise ToolError(f"{exc}. {next_step}".strip()) from None
        except FutureTimeout:
            raise ToolError(f"Tiled did not answer within {deadline_s:.0f} s. {next_step}".strip()) from None
        except (tiled_client.TiledUnavailable, tiled_client.DataKeyNotFound, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        except (httpx.TransportError, OSError) as exc:
            raise ToolError(f"Tiled stopped answering ({type(exc).__name__}: {exc}).") from exc
        except Exception as exc:  # an HTTP error status, a plug-in kind's bug, data this view cannot draw
            raise ToolError(f"could not read the data ({type(exc).__name__}: {exc})") from exc

    @tool()
    def list_data(self, kind: str | None = None, search: str | None = None, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """Stored data, newest registered first: key, kind (image, spectrum, spectrum_image, stack, map, table, ...), shape, detector. Filter by kind or by part of the key; look at one with view_data. Looks through the newest 300 keys (files registered after a Tiled outage count as new); any older key can still be opened with view_data. If more is true, call again with offset = next_offset."""
        if kind is not None and kind not in views.kind_names():
            raise ToolError(f"unknown kind {kind!r}; kinds are {views.kind_names()}")
        started = time.monotonic()
        return self._with_deadline(lambda: self._list_data(kind, search, max(1, min(int(limit), 100)), max(0, int(offset)), started), lane="list")

    def _list_data(self, kind: str | None, search: str | None, limit: int, offset: int, started: float) -> dict[str, Any]:
        """One page of matches. ``offset`` counts keys from the newest, so a page cut short by the time budget
        continues where it stopped (next_offset) instead of rescanning."""
        uri = tiled_client.uri_from_data_device(self.data_device_address)
        client = tiled_client.connect(uri)
        items = client.items()
        tail = getattr(items, "tail", None)
        window = list(reversed(list(tail(LIST_DATA_WINDOW)) if callable(tail) else list(items)[-LIST_DATA_WINDOW:]))
        total = views._count(client)
        deadline = started + LIST_DATA_BUDGET_S
        found: list[dict[str, Any]] = []
        position = offset
        stopped = None
        while position < len(window):
            if position > offset and time.monotonic() >= deadline:  # each call moves on by at least one key
                stopped = f"stopped after {LIST_DATA_BUDGET_S:.0f} s; continue with offset={position}, or narrow it with search"
                break
            key, node = window[position]
            if search and search.lower() not in key.lower():
                position += 1
                continue
            try:
                entry = views.summarize(node, key)
            except Exception as exc:  # one unreadable entry must not hide the rest
                entry = {"key": key, "kind": "unreadable", "note": f"{type(exc).__name__}: {exc}"[:200]}
            if kind and entry.get("kind") != kind:
                position += 1
                continue
            if len(found) == limit:  # a match past this page: the next page starts here
                break
            found.append(entry)
            position += 1
        more = position < len(window)
        looked = f"looked at keys {offset} to {position} of the newest {len(window)} (of {total if total is not None else '?'} in all)"
        if stopped:
            looked += f"; {stopped}"
        elif not more and total is not None and total > len(window):
            looked += "; older keys are not listed: open them by key with view_data"
        return numpy_to_python({"uri": uri, "data": found, "more": more, "next_offset": position if more else None, "looked": looked, "kinds": views.kind_names()})

    @tool()
    def view_data(
        self,
        key: str,
        format: Literal["png", "text"] = "png",
        dataset: str | None = None,
        region: list[float] | None = None,
        index: list[int] | int | None = None,
        max_side: int = views.PREVIEW_MAX_SIDE,
    ) -> ToolResult:
        """Look at stored data by key (from an acquisition, a job's results, or list_data). png: a picture (images contrast-stretched, spectra plotted against energy with peaks marked) plus a short text summary (statistics, peaks). text: shape, units, statistics, peaks and a small CSV. dataset: one array or group inside a file (the text lists them). region: an image's [row0, row1, col0, col1] in pixels, or a spectrum's [min, max] in its axis units (eV). index: which frame of a stack or 4-D dataset, or a spectrum image's [row, col] position to show the spectrum there. Never returns raw arrays."""
        options = views.ViewOptions(region=region, index=index, max_side=max_side)
        summary, png = self._with_deadline(lambda: views.view(self._tiled_node(key), key, format, dataset, options), lane="view")
        # A TextContent block, not a bare str: FastMCP joins a list of only strings into one JSON array.
        content: list[Any] = [TextContent(type="text", text=summary["text"])]
        if png is not None:
            content.append(MCPImage(data=png, format="png"))
        return ToolResult(content=content, structured_content=numpy_to_python(summary))

    def _augment_with_preview(self, command_name: str, result: Any) -> Any:
        """Attach a PNG of what an acquire command saved; never fails the acquisition.

        Returns a ToolResult with the Tiled key as text (unchanged for existing
        callers) and the image, plus structured_content matching the tool's
        {"result": str} output schema; a bare (text, Image) tuple would fail
        client-side output-schema validation. For PREVIEW_COMMANDS a failed preview
        is reported in a text block, so a Tiled outage is visible, not silent.
        """
        if not isinstance(result, str) or not command_name.startswith("acquire") or not DATA_KEY.match(result.strip()):
            return result

        try:
            _summary, png = self._with_deadline(
                lambda: views.view(self._tiled_node(result, hints=False), result, "png"), lane="preview", deadline_s=PREVIEW_DEADLINE_S
            )
        except Exception as exc:  # the hardware already acquired: report, never raise
            if self.verbose:
                print(f"[preview] could not build a preview for {result!r}: {exc}")
            if command_name not in PREVIEW_COMMANDS:
                return result
            note = f"image preview unavailable ({exc}). The data were saved; look at them later with view_data(key={result!r}) instead of acquiring again."
            return ToolResult(content=[TextContent(type="text", text=result), TextContent(type="text", text=note)], structured_content={"result": result})
        return ToolResult(content=[result, MCPImage(data=png, format="png")], structured_content={"result": result})

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
            for py_type in candidates:
                if py_type.__module__ == "builtins":
                    return py_type
            return candidates[0]

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

        # Convert numpy types (including nested containers) to native Python types
        result = numpy_to_python(result)
        if out_type != CmdArgType.DevEncoded:
            return result

        if not isinstance(result, tuple) or len(result) != 2:
            return result

        metadata_raw, payload_raw = result
        if isinstance(metadata_raw, bytes):
            metadata_str = metadata_raw.decode("utf-8", errors="replace")
        else:
            metadata_str = str(metadata_raw)

        try:
            metadata = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            metadata = metadata_str

        if isinstance(payload_raw, memoryview):
            payload_bytes = payload_raw.tobytes()
        elif isinstance(payload_raw, bytearray):
            payload_bytes = bytes(payload_raw)
        elif isinstance(payload_raw, bytes):
            payload_bytes = payload_raw
        else:
            payload_bytes = str(payload_raw).encode("utf-8")

        payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
        return {
            "encoding": "base64",
            "metadata": metadata,
            "payload": payload_b64,
        }

    def _create_wrapper(
        self,
        func: Callable,
        cmd_info: CommandInfo,
        command_name: str,
        dev_class: str,
        tool_prefix: str | None = None,
    ) -> Callable:
        """Create a wrapper function with a proper signature for a Tango command.

        Args:
            func: The raw Tango device command method
            cmd_info: The CommandInfo object from Tango
            command_name: The name of the command
            dev_class: The Tango device class name (used by the risk and blocking rules)
            tool_prefix: Tool-name prefix; defaults to the class name

        Returns:
            A wrapper function with a proper signature
        """

        in_type = cmd_info.in_type
        py_type = self._tango_type_to_python(in_type)
        in_desc = cmd_info.in_type_desc

        out_type = cmd_info.out_type
        unique_name = f"{tool_prefix or dev_class}_{command_name}".replace("/", "_").replace("-", "_")

        envelope = (tool_prefix or dev_class) in self._envelope_prefixes
        stem = (tool_prefix or dev_class) in self._stem_prefixes
        key = f"{dev_class}.{command_name}"
        matched = [level for pattern, level in self.risk_overrides.items() if fnmatchcase(unique_name.lower(), pattern.lower()) or fnmatchcase(key.lower(), pattern.lower())]
        default = risk_of(dev_class, command_name, envelope=envelope, stem=stem)
        risk = matched[0] if matched else default
        if risk == "low" and default == "high":
            self._lowered.add(unique_name)
        self.tool_risks[unique_name] = risk
        self._risk_keys[unique_name] = key

        def _run(arguments: dict, *call_args):
            started, t0 = time.time(), time.perf_counter()
            caller = self._caller()
            describe = None
            if envelope and command_name == "start" and call_args:
                device = self._devices.get(tool_prefix or dev_class)
                describe = (lambda: "Reserved plan: " + device.get_reservation(str(call_args[0]))) if device is not None else None
            gate = self._gate(unique_name, risk, arguments, caller, describe)
            if envelope and command_name == "start" and call_args:
                call_args = (self._with_grant(call_args[0], gate, caller), *call_args[1:])
            try:
                normalized = self._normalize_command_result(out_type, func(*call_args))
                result = self._augment_with_preview(command_name, normalized)
            except Exception as exc:
                if self.tool_log is not None:
                    self.tool_log.record(unique_name, arguments, started, time.perf_counter() - t0, error=exc)
                if self.trace is not None:
                    self.trace.record(unique_name, False, started, arguments=arguments, duration_s=time.perf_counter() - t0, error=describe_error(exc), caller=caller, **gate)
                message = describe_error(exc)
                repeats = self._note_failure(unique_name, arguments, message)
                if repeats >= REPEAT_FAILURE_WARN_AT:
                    message += (
                        f"\nThis call has failed {repeats} times in a row. "
                        "Do not call it again with the same arguments; report the error to the user."
                    )
                raise RuntimeError(message) from exc
            with self._failures_lock:
                self._failures.pop(unique_name, None)
            if self.tool_log is not None:
                self.tool_log.record(unique_name, arguments, started, time.perf_counter() - t0, result=normalized)
            if self.trace is not None:
                key = normalized if isinstance(normalized, str) and command_name.startswith("acquire") else None
                self.trace.record(unique_name, True, started, arguments=arguments, duration_s=time.perf_counter() - t0, result=normalized, result_key=key, caller=caller, **gate)
            return result
        py_return_type = self._tango_type_to_python(out_type)
        doc_lines = [f"Tango Device Class: {dev_class}", f"Tango Command: {command_name}"]
        # A DevVoid type line tells a model nothing; small models read every line.
        if in_type != CmdArgType.DevVoid:
            doc_lines.append(f"Input Type: {in_type.name}")
        if in_desc and not _placeholder_doc(in_desc):
            if _without_param_name(in_desc):
                doc_lines.append(f"Input Description: {_without_param_name(in_desc)}")
        if out_type != CmdArgType.DevVoid:
            doc_lines.append(f"Output Type: {out_type.name}")
        out_desc = cmd_info.out_type_desc
        if out_desc and not _placeholder_doc(out_desc):
            # Lead with what the command is for (once): models weigh the first line most when choosing a tool.
            doc_lines.insert(0, out_desc)
        job_kind = self._job_kinds.get(tool_prefix or dev_class)
        if envelope and job_kind and command_name.lower() in ENVELOPE_VERBS:
            doc_lines.insert(0, f"{tool_prefix or dev_class} job: {job_kind}.")
        plan_example = self._plan_examples.get(tool_prefix or dev_class)
        if envelope and command_name == "reserve" and plan_example:
            doc_lines.insert(1, f'Example request: {{"plan": {plan_example}}}')
        doc = "\n".join(doc_lines)

        # Get parameter name from docstring description text
        param_name = "arg"
        if in_desc:
            match = re.search(r'(?::param|@param)\s+(\w+):', in_desc)
            if match:
                param_name = match.group(1)

        if in_desc and not _placeholder_doc(in_desc):
            # Sanitize description
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
                if "arg" in kwargs and len(kwargs) == 1 and isinstance(kwargs["arg"], dict):
                    arg_input = kwargs["arg"]
                else:
                    arg_input = kwargs
                return _run({"arg": arg_input}, arg_input)

            # Use VAR_KEYWORD (**kwargs) to make Pydantic accept any incoming fields
            params = [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)]
            
        elif json_string:
            # JSON-object commands (job envelope, approvals): accept {"request": {...}},
            # {"request": "<json>"} or the object's fields given directly, e.g. {"plan": {...}}.
            # The fields come from the description's example object; FastMCP does not allow **kwargs.
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
                    # One canonical text for one call, whatever form it came in, so the
                    # approval fingerprint of a retried call matches its grant.
                    arg = json.dumps(arg, sort_keys=True, separators=(",", ":"))
                return _run({param_name: numpy_to_python(arg)}, arg)

            params = [inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[arg_type | None, Field(description=clean_desc)], default=None)]
            params += [inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[Any, Field(description=f"Same as {param_name}.{f}.")], default=None) for f in fields]

        else:
            # Scalars and standard arrays
            def wrapper(*args, **kwargs):
                # Get first positional arg or parameter name out of kwargs
                arg = args[0] if args else kwargs.get(param_name)
                return _run({param_name: numpy_to_python(arg)}, arg)

            params = [inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=arg_type)]

        # Build annotations, omitting VAR_KEYWORD parameters so Pydantic safely allows dynamic extra fields
        wrapper.__annotations__ = {p.name: p.annotation for p in params if p.kind != inspect.Parameter.VAR_KEYWORD}
        wrapper.__annotations__["return"] = py_return_type

        wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=py_return_type)
        wrapper.__doc__ = doc
        wrapper.__name__ = unique_name
        wrapper.__qualname__ = unique_name

        return wrapper

    @staticmethod
    def _with_grant(request: Any, gate: dict, caller: str | None) -> Any:
        """Tell a starting job who asked and which approval covers it, so its trace links to the grant.

        Added after the approval check, so it never changes the call's fingerprint.
        """
        if not isinstance(request, str):
            return request
        try:
            body = json.loads(request)
        except json.JSONDecodeError:
            return request
        if not isinstance(body, dict):
            return request
        # Only the gate may say which approval covers a job; an agent-supplied grant_id is dropped.
        body.pop("grant_id", None)
        if gate.get("grant_id"):
            body["grant_id"] = gate["grant_id"]
        if caller:
            body["caller"] = caller
        return json.dumps(body, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _caller() -> str | None:
        """Who is calling, from the X-Asyncroscopy-Caller header; None outside an HTTP request."""
        try:
            from fastmcp.server.dependencies import get_http_headers

            return get_http_headers().get(CALLER_HEADER) or None
        except Exception:
            return None

    def _approvals(self) -> DeviceProxy:
        proxy = DeviceProxy(self.approvals_device_address)
        proxy.set_timeout_millis(5_000)
        return proxy

    def _gate(self, tool_name: str, risk: str, arguments: dict, caller: str | None, describe: Callable[[], str] | None = None) -> dict:
        """Apply the approval mode to one call. Returns trace fields, or raises ToolError to hold the call.

        Low-risk calls and servers without an Approvals device always pass. In
        approve mode a high-risk call passes only if an operator approved this
        exact call; otherwise a request is filed and the call is refused with
        the request id, so the agent can wait and retry the same call.
        """
        if self.endpoint_role == "operator":
            # The operator endpoint is the person at the GUI; nothing they do is held.
            return {"mode": None, "approved_by": "operator"}
        if self.approvals_device_address is None or risk == "low":  # anything not plainly low is gated
            return {"mode": None, "approved_by": None}
        try:
            approvals = self._approvals()
            mode = approvals.get_mode()
            if mode == "auto":
                return {"mode": "auto", "approved_by": "auto"}
            decision = json.loads(approvals.check(json.dumps({"action": tool_name, "arguments": arguments})))
            if decision.get("allowed"):
                return {"mode": mode, "approved_by": decision.get("approved_by"), "grant_id": decision.get("grant_id"), "request_id": decision.get("request_id")}
            reason = None
            if describe is not None:
                try:  # what the operator is approving; outside the fingerprint, so the retry still matches
                    reason = describe()[:2000]
                except Exception as exc:
                    reason = f"(could not read what this call will do: {describe_error(exc)})"
            filed = json.loads(approvals.request(json.dumps({"action": tool_name, "arguments": arguments, "caller": caller, "reason": reason})))
        except tango.DevFailed as exc:
            # Fail closed: without the Approvals device we cannot tell whether a person must approve.
            raise ToolError(f"{tool_name} is high risk and the approval service is unreachable ({describe_error(exc)}); not run.") from exc
        if self.trace is not None:
            self.trace.record(tool_name, False, time.time(), arguments=arguments, error="approval_required", caller=caller, mode=mode, request_id=filed["request_id"])
        raise ToolError(
            f"APPROVAL REQUIRED: {tool_name} is held for an operator (request_id={filed['request_id']}). "
            "Call approval_status with this request_id until it is approved, then repeat exactly the same call. "
            "If it is denied, do not retry; report to the user."
        )

    def _tool_names(self) -> set[str]:
        """Names of the device tools this endpoint offers (skills that need others are not offered)."""
        return {f"{prefix}_{command}".replace("/", "_").replace("-", "_") for prefix, commands in self.tools.items() for command in commands}

    def _all_tool_names(self) -> set[str]:
        return self._tool_names() | self._native_names

    def _skills(self):
        """Skills this endpoint offers: a learned skill only when every tool it lists is offered here."""
        from asyncroscopy.skills.registry import list_skills, usable_by

        names = self._all_tool_names()
        return [s for s in list_skills(names) if not s.learned or usable_by(s, names)]

    @tool()
    def list_skills(self) -> list[dict[str, Any]]:
        """Skills: procedures for kinds of experiment (e.g. mapping). Load one with load_skill before doing that kind of task."""
        return [{"name": s.name, "description": s.description, "version": s.version, "available": s.available, **({"problem": s.problem} if s.problem else {})} for s in self._skills()]

    @tool()
    def load_skill(self, name: str) -> str:
        """Full instructions of one skill from list_skills. Follow them for that kind of task."""
        from asyncroscopy.skills.registry import read_front_matter

        skill = next((s for s in self._skills() if s.name == name), None)
        if skill is None:
            raise ToolError(f"no skill named {name!r}; available: {[s.name for s in self._skills() if s.available]}")
        if not skill.available:
            raise ToolError(f"skill {name!r} cannot be used: {skill.problem}")
        return read_front_matter(skill.path)[1]

    @tool()
    def endpoint_info(self) -> dict[str, Any]:
        """For agent programs, not models: each tool's risk as the approval gate sees it, the job devices and the analysis server."""
        from asyncroscopy.safety.policy import NATIVE_HIGH

        risks = dict(self.tool_risks)
        risks.update({name: "high" if name in NATIVE_HIGH else "low" for name in self._native_names})
        return {
            "role": self.endpoint_role,
            "analysis_url": self.analysis_url,
            "job_devices": sorted(self._envelope_prefixes),
            "stem_devices": sorted(self._stem_prefixes),
            "job_limits": self.job_limits,
            "risks": risks,
        }

    @tool()
    def approval_status(self, request_id: str) -> dict[str, Any]:
        """State of an approval request: pending, approved, denied or expired. After 'approved', repeat the held call unchanged."""
        if self.approvals_device_address is None:
            return {"request_id": request_id, "state": "approved", "note": "this server has no approval mode"}
        status = json.loads(self._approvals().request_status(request_id))
        if status.get("state") in {"denied", "expired"}:
            with self._held_runs_lock:
                held = next(((k, v) for k, v in self._held_runs.items() if v["request_id"] == request_id), None)
                if held is not None:
                    del self._held_runs[held[0]]
            if held is not None:
                # A run_job held on this request reserved the device; nothing will start it now.
                status["note"] = self._release(held[1]["device"], held[1]["reservation_id"])
        self._sweep_held()
        return status

    # ------------------------------------------------------------------ run_job's reservations
    @staticmethod
    def _session() -> str:
        """The MCP session of the current request ("" outside one): two agents with one caller name stay apart."""
        try:
            from fastmcp.server.dependencies import get_http_headers

            return get_http_headers(include_all=True).get("mcp-session-id") or ""
        except Exception:
            return ""

    def _release(self, device: str, reservation_id: str) -> str:
        """Release a reservation run_job made (housekeeping, low risk); returns what happened, for the reply."""
        proxy = self._devices.get(device)
        started = time.time()
        if proxy is None:
            return f"could not release {device} reservation {reservation_id}: no such job device"
        try:
            reply = json.loads(proxy.release(json.dumps({"reservation_id": reservation_id})))
        except Exception as exc:
            note = f"could not release {device} reservation {reservation_id} ({describe_error(exc)}); an operator can release it"
        else:
            failed = isinstance(reply, dict) and "error" in reply
            note = (f"{device} reservation {reservation_id} was not released: {reply['error'].get('message', reply['error'])}" if failed
                    else f"{device} reservation {reservation_id} was released")
        if self.trace is not None:
            self.trace.record(f"{device}_release", "was released" in note, started, arguments={"reservation_id": reservation_id}, caller="run_job", result=note)
        return note

    def _sweep_held(self, force: bool = False) -> list[str]:
        """Release held run_job reservations whose request was denied or expired, that nobody came back
        for within HELD_MAX_S, or beyond MAX_HELD_RUNS (oldest first). Returns what was released."""
        now = time.monotonic()
        with self._held_runs_lock:
            if not force and now - self._last_sweep < HELD_SWEEP_S:
                return []
            self._last_sweep = now
            entries = list(self._held_runs.items())
        stale = []
        for key, held in entries:
            state = None
            if held["request_id"] and self.approvals_device_address:
                try:
                    state = json.loads(self._approvals().request_status(held["request_id"])).get("state")
                except Exception:
                    state = None
            if state in ("denied", "expired") or now - held["created"] > HELD_MAX_S:
                stale.append(key)
        with self._held_runs_lock:
            ordered = sorted(self._held_runs, key=lambda k: self._held_runs[k]["created"])
            stale += [k for k in ordered[: max(0, len(ordered) - MAX_HELD_RUNS)] if k not in stale]
            removed = [self._held_runs.pop(k) for k in stale if k in self._held_runs]
        return [self._release(h["device"], h["reservation_id"]) for h in removed]

    def _over_limits(self, reserved: dict) -> list[dict]:
        estimate = reserved.get("estimate") if isinstance(reserved.get("estimate"), dict) else {}
        checks = (("max_duration_s", "duration_s", "s"), ("max_dose_e_per_a2", "max_dose_e_per_a2", "e/A^2"))
        return [
            {"what": field, "value": estimate[field], "limit": self.job_limits[limit], "units": units}
            for limit, field, units in checks
            if limit in self.job_limits and isinstance(estimate.get(field), (int, float)) and estimate[field] > self.job_limits[limit]
        ]

    @staticmethod
    def _held_message(device: str, request_id: str, reservation_id: str, summary: dict) -> str:
        return (
            f"APPROVAL REQUIRED: {device}_start is held for an operator (request_id={request_id}). "
            "Call approval_status with this request_id until it is approved, then repeat this run_job call unchanged. "
            f"If it is denied, do not retry; report to the user. Reserved as {reservation_id}; estimate: {json.dumps(summary.get('estimate'), default=str)}"
        )

    def _job_device(self, device: str) -> DeviceProxy:
        proxy = self._devices.get(device)
        if proxy is None or device not in self._envelope_prefixes:
            raise ToolError(f"no job device {device!r}; job devices here: {sorted(self._envelope_prefixes)}")
        return proxy

    @tool()
    def run_job(self, device: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Run a job on a job device in one call: the plan is checked (a refused plan comes back with the reason and an example), the job starts, and the call waits up to 15 s. device: the job device's tool prefix, e.g. "Mapper". Every reply has the estimate. If the reply's state is done it already carries the results (result_keys, summary); otherwise call wait_for_job with its job_id until it is done. A plan over this endpoint's limits is refused: tell the user the estimate and ask."""
        began = time.monotonic()
        tools = self.tools.get(device) if device in self._envelope_prefixes else None
        if not tools or "reserve" not in tools or "start" not in tools:
            raise ToolError(f"no job device {device!r}; job devices here: {sorted(self._envelope_prefixes)}")
        self._sweep_held()
        key = (self._caller() or "", self._session(), device, json.dumps(plan, sort_keys=True, default=str))
        with self._run_job_lock:
            with self._held_runs_lock:
                held = self._held_runs.get(key)
            if held is not None and held["request_id"] and self.approvals_device_address is not None:
                try:
                    state = json.loads(self._approvals().request_status(held["request_id"])).get("state")
                except Exception:
                    state = None
                if state == "pending":  # asking again before the operator decided must not file a second request
                    raise ToolError(self._held_message(device, held["request_id"], held["reservation_id"], held["summary"]))
                if state in ("denied", "expired"):
                    with self._held_runs_lock:
                        self._held_runs.pop(key, None)
                    note = self._release(device, held["reservation_id"])
                    raise ToolError(f"the approval request {held['request_id']} for this run was {state}; {note}. Do not retry; tell the user.")
            if held is None:
                reserved = json.loads(tools["reserve"]({"plan": plan}))
                if "error" in reserved:
                    return reserved
                summary = {k: reserved[k] for k in ("reservation_id", "plan_id", "resolved_plan", "estimate", "limits") if k in reserved}
                violations = self._over_limits(reserved)
                if violations:
                    note = self._release(device, reserved["reservation_id"])
                    return {**summary, "error": {"code": "plan_rejected", "violations": violations, "message": (
                        "the estimate is over this endpoint's job limits (mcp.job_limits): "
                        + "; ".join(f"{v['what']} {v['value']:.4g} {v['units']} > {v['limit']:.4g}" for v in violations)
                        + ". Tell the user the estimate and ask before trying a smaller plan; only an operator can raise the limit.")}, "note": note}
            else:
                summary = held["summary"]
            reservation_id = summary["reservation_id"]
            try:
                # The start goes through the approval gate like <device>_start; the operator sees the reserved plan.
                started = json.loads(tools["start"]({"reservation_id": reservation_id}))
            except Exception as exc:
                request_id = re.search(r"request_id=(\w+)", str(exc)) if "APPROVAL REQUIRED" in str(exc) else None
                if request_id is not None:
                    with self._held_runs_lock:
                        self._held_runs[key] = {"reservation_id": reservation_id, "request_id": request_id.group(1), "device": device,
                                                "created": time.monotonic(), "summary": summary}
                    raise ToolError(self._held_message(device, request_id.group(1), reservation_id, summary)) from exc
                with self._held_runs_lock:
                    self._held_runs.pop(key, None)
                note = self._release(device, reservation_id)
                raise ToolError(f"{device} did not start ({describe_error(exc)}); {note}. Plan: {json.dumps(summary, default=str)[:1500]}") from exc
            with self._held_runs_lock:
                self._held_runs.pop(key, None)
            if "error" in started:
                return {**summary, **started, "note": self._release(device, reservation_id)}
        reply = {**summary, **{k: v for k, v in started.items() if k != "schema_version"}}
        wait_s = max(0.0, min(RUN_JOB_WAIT_S, RUN_JOB_BUDGET_S - (time.monotonic() - began)))
        if started.get("job_id"):
            proxy = self._job_device(device)
            status = json.loads(proxy.job_status(json.dumps({"job_id": started["job_id"]})))
            reply.update(self._wait_on_device(proxy, started["job_id"], status, time.monotonic() + wait_s))
        return reply

    @tool()
    def wait_for_job(self, job_id: str, device: str | None = None, max_wait_s: float = JOB_WAIT_MAX_S) -> dict[str, Any]:
        """Wait for any background job (a map from run_job, or assemble_map / score_map on the analysis server) to finish, at most max_wait_s (25 s) per call, and return its state. When it is done the reply also has its results (result_keys, summary). device (e.g. "Mapper") is optional. Call again while the state is still active."""
        budget = min(max(float(max_wait_s), 0.0), JOB_WAIT_MAX_S)
        self._sweep_held()
        if device and device not in self._envelope_prefixes:
            raise ToolError(f"no job device {device!r}; job devices here: {sorted(self._envelope_prefixes)}"
                            + ("; leave device out for analysis jobs" if self.analysis_url else ""))
        unreachable = []
        for candidate in [device] if device else sorted(self._envelope_prefixes):
            proxy = self._job_device(candidate)
            try:
                status = json.loads(proxy.job_status(json.dumps({"job_id": job_id})))
            except tango.DevFailed as exc:
                if device:  # the one device asked for is down: say so
                    raise ToolError(f"{candidate} did not answer ({describe_error(exc)}); try wait_for_job again shortly") from exc
                unreachable.append(candidate)  # one dead job device must not stop waiting on the others' jobs
                continue
            if status.get("error", {}).get("code") != "unknown_job":
                return self._wait_on_device(proxy, job_id, status, time.monotonic() + budget)
        if self.analysis_url:
            return self._wait_on_analysis(job_id, budget)
        raise ToolError(
            f"no job {job_id!r} on {device or 'any job device here'}"
            + (f" (not answering: {unreachable})" if unreachable else "")
            + ". Jobs started by assemble_map or score_map run on the analysis server; this endpoint can wait on them "
            "only when its MCP config sets mcp.analysis_url."
        )

    def _wait_on_device(self, proxy: DeviceProxy, job_id: str, status: dict, deadline: float) -> dict[str, Any]:
        while True:
            if "error" in status:
                return status
            state = status.get("state")
            if state not in WAIT_NOTES:
                if state == "done":
                    results = json.loads(proxy.results(json.dumps({"job_id": job_id})))
                    status.update({k: results[k] for k in ("result_keys", "summary") if k in results})
                return status
            if time.monotonic() >= deadline:
                reason = (status.get("suspended") or {}).get("reason", "")
                return {**status, "note": WAIT_NOTES[state].format(reason=reason)}
            time.sleep(1.0)
            # Polls job_status (low risk) rather than blocking in the device, so abort stays immediate.
            status = json.loads(proxy.job_status(json.dumps({"job_id": job_id})))

    def _wait_on_analysis(self, job_id: str, budget: float) -> dict[str, Any]:
        """Wait on a stitching or scoring job of the analysis server; when it is done, add its results."""
        from fastmcp import Client

        def text(reply) -> str:
            return "".join(getattr(b, "text", "") for b in getattr(reply, "content", []) or [])

        async def call() -> dict[str, Any]:
            async with Client(self.analysis_url, timeout=budget + 15) as client:
                reply = await client.call_tool("analysis_job_status", {"job_id": job_id, "wait_s": budget}, raise_on_error=False)
                if reply.is_error and "wait_s" in text(reply):  # an analysis server without server-side waits: poll it
                    deadline = time.monotonic() + budget
                    while True:
                        reply = await client.call_tool("analysis_job_status", {"job_id": job_id}, raise_on_error=False)
                        state = (reply.structured_content or {}).get("state")
                        if reply.is_error or state not in ("queued", "running") or time.monotonic() >= deadline:
                            break
                        await asyncio.sleep(1.0)
                if reply.is_error:
                    raise RuntimeError(text(reply)[:500] or "an error with no message")
                status = dict(reply.structured_content or {})
                if not status:
                    raise RuntimeError(f"an empty reply: {text(reply)[:300]!r}")
                if status.get("state") == "done":
                    results = await client.call_tool("analysis_job_results", {"job_id": job_id}, raise_on_error=False)
                    if results.is_error:
                        status["note"] = f"done, but its results could not be read: {text(results)[:300]}"
                    else:
                        status.update({k: v for k, v in (results.structured_content or {}).items() if k in ("result_keys", "summary")})
                return status

        try:
            # Sync tools run in FastMCP's worker threads, which have no event loop of their own.
            status = asyncio.run(call())
        except Exception as exc:
            raise ToolError(f"the analysis server at {self.analysis_url} did not answer ({type(exc).__name__}: {exc}); call wait_for_job again shortly") from exc
        if status.get("error", {}).get("code") == "unknown_job":
            raise ToolError(f"no job {job_id!r} on any job device here or on the analysis server")
        if status.get("state") in {"queued", "running"}:
            status["note"] = WAIT_NOTES["running"]
        return status

    def _note_failure(self, tool_name: str, arguments: dict, error: str) -> int:
        """Count consecutive identical failures of a tool; a success, new error or long pause resets it."""
        signature = json.dumps(arguments, sort_keys=True, default=str) + "\n" + error
        now = time.monotonic()
        with self._failures_lock:
            previous, count, last = self._failures.get(tool_name, ("", 0, 0.0))
            repeat = previous == signature and now - last <= REPEAT_FAILURE_WINDOW_S
            count = count + 1 if repeat else 1
            self._failures[tool_name] = (signature, count, now)
        return count

    def _find_tools(self) -> dict[str, dict[str, tuple[Callable, CommandInfo]]]:
        """Discover tools by querying Tango DB for devices and their commands.

        Returns a dict mapping tool prefix (the class name, or <Class>_<member> when several
        devices share a class) -> command_name -> (func, cmd_info)
        """
        devices = self._list_all_devices()
        tools: dict[str, dict[str, tuple[Callable, CommandInfo]]] = {}
        opened: list[tuple[str, DeviceProxy, str]] = []
        for device_name in devices:
            if self._is_admin_device(device_name):
                continue
            try:
                dev = DeviceProxy(device_name)
                dev.set_timeout_millis(COMMAND_TIMEOUT_MILLIS)
                info = dev.info()
                dev_class = info.dev_class
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {device_name}: failed to open proxy ({exc})")
                continue
            if not self._is_blocked_class(dev_class):
                opened.append((device_name, dev, dev_class))

        # Tools are named <Class>_<command>. When several devices share a class (two robots,
        # two job devices) each gets its own prefix, <Class>_<member>, so neither hides the
        # other and an approval for one never covers the other.
        per_class: dict[str, int] = {}
        members: dict[str, list[str]] = {}
        families: dict[str, list[str]] = {}
        for name, _dev, dev_class in opened:
            per_class[dev_class] = per_class.get(dev_class, 0) + 1
            members.setdefault(dev_class, []).append(name.split("/")[-1])
            families.setdefault(dev_class, []).append(name.split("/")[1] if name.count("/") == 2 else name)
        for table in (self._prefix_class, self._envelope_prefixes, self._stem_prefixes, self._job_kinds, self._plan_examples, self.tool_risks, self._risk_keys, self._lowered):
            table.clear()
        for device_name, dev, dev_class in opened:
            prefix = dev_class
            if per_class[dev_class] > 1:
                # The member ("lab/mapper/2" -> 2); else the family, which the launcher sets from the
                # config key ("asyncroscopy/grid2/default" -> grid2); else the whole device name.
                parts = device_name.split("/")
                member, family = parts[-1], (parts[1] if len(parts) == 3 else device_name)
                if members[dev_class].count(member) == 1:
                    suffix = member
                elif families[dev_class].count(family) == 1:
                    suffix = family
                else:
                    suffix = "_".join(parts)
                prefix = f"{dev_class}_{suffix}".replace("/", "_").replace("-", "_")
            self._prefix_class[prefix] = dev_class

            try:
                commands = dev.command_list_query()
                if is_stem_device(c.cmd_name for c in commands):
                    self._stem_prefixes.add(prefix)
                if is_envelope_device(c.cmd_name for c in commands):
                    self._envelope_prefixes.add(prefix)
                    try:
                        self._job_kinds[prefix] = str(dev.job_kind)
                    except Exception:  # an envelope device without the job_kind attribute
                        pass
                    try:
                        if str(dev.plan_example):
                            self._plan_examples[prefix] = str(dev.plan_example)
                    except Exception:  # older job devices have no plan_example
                        pass
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {device_name}: failed to query commands ({exc})")
                continue

            for cmd in commands:
                command_name = cmd.cmd_name
                if self.endpoint_role == "agent" and operator_only(dev_class, command_name):
                    continue
                global_blocks = self.blocked_functions.get("*", [])
                if command_name in global_blocks or f"{dev_class}.{command_name}" in global_blocks or command_name in self.blocked_functions.get(dev_class, []):
                    continue

                if self.include_only_functions:
                    allowed = (
                        command_name in self.include_only_functions
                        or f"{dev_class}.{command_name}" in self.include_only_functions
                        or any(item.endswith(f".{command_name}") for item in self.include_only_functions)
                    )
                    if not allowed:
                        continue
                try:
                    func = getattr(dev, command_name)
                except Exception as exc:
                    if self.verbose:
                        print(
                            f"Skipping {device_name}.{command_name}: "
                            f"failed to resolve command ({exc})"
                        )
                    continue
                tools.setdefault(prefix, {})[command_name] = (func, cmd)
                self._devices[prefix] = dev
        return tools

    def _warn_about_risks(self) -> None:
        """Print what an instrument author most likely got wrong in the risk settings: an override
        that matches no command (it does nothing), and a stop-like command that approve mode holds."""
        warnings = []
        example = next((t for t in self._risk_keys if not t.endswith(("_State", "_Status"))), "Class_command")
        for pattern in self.risk_overrides:
            p = pattern.lower()
            hidden = "." in pattern and operator_only(*pattern.split(".", 1))  # not on this endpoint, but real
            if not hidden and not any(fnmatchcase(name.lower(), p) or fnmatchcase(key.lower(), p) for name, key in self._risk_keys.items()):
                warnings.append(f"risk override {pattern!r} matches no command; write it as Class.command or a tool name such as {example}")
        for name, risk in self.tool_risks.items():
            if self.endpoint_role == "agent" and risk == "high" and looks_like_stop(self._risk_keys[name].split(".", 1)[1]):
                warnings.append(f"{name} looks like a stop but is high: approve mode holds it. If it only stops, switches off a source or lifts a probe away from the sample (not if it opens, moves or frees anything), add {{\"{name}\": \"low\"}} to mcp.safety.risk")
        for name in sorted(self._lowered if self.endpoint_role == "agent" else ()):
            warnings.append(f"{name} would be held by default; an mcp.safety.risk override makes it low, so approve mode lets agents run it")
        self.risk_warnings = warnings
        for line in warnings:
            print(f"[safety] {line}")

    def setup(self, print_summary: bool = True):
        """Configure tools and add them to the MCP instance.

        Args:
            print_summary: If True, print tool discovery and registration summary.
        """
        raw_tools = self._find_tools()
        exported = {str(dev.dev_name()).lower() for dev in self._devices.values()}
        if self.approvals_device_address and self.approvals_device_address.lower() not in exported:
            print(f"[safety] approvals device {self.approvals_device_address} is not among the exported devices; while it is not running, "
                  "gated calls are refused. Start this server after run_servers.py prints ready, or check safety.approvals_device_address", flush=True)

        wrapped_tools: dict[str, dict[str, Callable]] = {}
        prefix_class = self._prefix_class
        for prefix in raw_tools:
            wrapped_tools[prefix] = {}
            for command_name, (func, cmd_info) in raw_tools[prefix].items():
                wrapped = self._create_wrapper(func, cmd_info, command_name, prefix_class.get(prefix, prefix), tool_prefix=prefix)
                wrapped_tools[prefix][command_name] = wrapped

        self.tools = wrapped_tools
        if print_summary and self.verbose:
            print("Discovered tools by Tango class:")
            for dev_class in sorted(raw_tools):
                command_names = sorted(raw_tools[dev_class].keys())
                print(f"- {dev_class}: {len(command_names)}")
                for command_name in command_names:
                    print(f"    - {command_name}")

        native_tools = [self.list_data, self.view_data, self.list_devices, self.list_skills, self.load_skill, self.endpoint_info]
        if self._envelope_prefixes:
            native_tools.extend([self.run_job, self.wait_for_job])
        if self.approvals_device_address is not None:
            native_tools.append(self.approval_status)
        self._native_names = {getattr(t, "__name__", getattr(t, "name", "")) for t in native_tools}
        for native_tool in native_tools:
            self.mcp.add_tool(native_tool)
            if self.verbose:
                print(f"Registered native tool: {native_tool.__name__}")

        num_device_tools = 0
        for dev_class in wrapped_tools:
            for command_name, wrapped_func in wrapped_tools[dev_class].items():
                try:
                    tool_obj = Tool.from_function(wrapped_func)
                    self.mcp.add_tool(tool_obj)
                    num_device_tools += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to wrap {dev_class}.{command_name}: {e}")
                        traceback.print_exc()

        self._warn_about_risks()
        # Printed unconditionally (unlike the verbose summary below) so GUIs can
        # parse the final count from stdout even when quiet mode is on.
        print(f"MCP ready: {len(native_tools) + num_device_tools} tool(s) registered", flush=True)

        if print_summary and self.verbose:
            print(f"\nRegistered {len(native_tools)} native tool(s)")
            print(f"Registered {num_device_tools} Tango device command tool(s)")
            print(f"Total: {len(native_tools) + num_device_tools} tools")
            print("\nAll MCP tools available:")
            for dev_class in sorted(self.tools.keys()):
                command_names = sorted(self.tools[dev_class].keys())
                for command_name in command_names:
                    wrapped_func = self.tools[dev_class][command_name]
                    sig = inspect.signature(wrapped_func)
                    print(f"  - {dev_class}.{command_name}{sig}")
                    if wrapped_func.__doc__:
                        for line in wrapped_func.__doc__.split("\n"):
                            stripped = line.strip()
                            if stripped:
                                print(f"{stripped}")
                    print("")
                print("")

    def start(self, transport: Transport | None = None, **kwargs):
        """
        Synchronizes with Tango DB and begins serving the MCP protocol.

        Args:
            transport: Transport protocol to use ("stdio", "http", "sse", or "streamable-http").
                       Defaults to None, which uses stdio for local piping to agents.
            **kwargs: Additional keyword arguments to pass to the MCP server
        """
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
    parser.add_argument("--trace", default="", help="Instrument Trace JSONL file; off when empty.")
    parser.add_argument("--data-views-json", default="[]", help="Modules to import first; each registers data kinds with asyncroscopy.data.views.")
    parser.add_argument("--analysis-url", default="", help="MCP URL of the analysis server, so wait_for_job also waits on its jobs.")
    parser.add_argument("--job-limits-json", default="{}", help='{"max_duration_s": ..., "max_dose_e_per_a2": ...}: run_job refuses plans over these.')
    return parser.parse_args(argv)


def check_port_free(host: str, port: int) -> str | None:
    """Return an error string if (host, port) cannot be bound, else None.

    Catches the common failure where a previous MCP server (possibly started
    from a different config) is still holding the port, before setup() prints
    the "MCP ready" line that GUIs parse as success.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        return str(exc)
    finally:
        probe.close()
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.transport == "streamable-http":
        bind_error = check_port_free(args.http_host, args.http_port)
        if bind_error:
            # "MCP ERROR:" is parsed by startup GUIs; keep the prefix stable.
            print(
                f"MCP ERROR: http://{args.http_host}:{args.http_port} is already in use - "
                f"another MCP server is likely still running with a different config. "
                f"Stop it or choose a different port. ({bind_error})",
                flush=True,
            )
            return 1

    for module in json.loads(args.data_views_json):
        try:
            importlib.import_module(module)  # registers its data kinds for view_data and previews
        except Exception as exc:
            print(f"MCP ERROR: mcp.data_views lists {module!r}, which could not be imported: {type(exc).__name__}: {exc}", flush=True)
            return 1
    server = MCPServer(
        name=args.name,
        tango_host=args.tango_host,
        tango_port=args.tango_port,
        blocked_classes=json.loads(args.blocked_classes_json),
        blocked_functions=json.loads(args.blocked_functions_json),
        include_only_functions=json.loads(args.include_only_functions_json),
        data_device_address=args.data_device_address,
        verbose=not args.quiet,
        tool_log_path=args.tool_log or None,
        endpoint_role=args.endpoint_role,
        approvals_device_address=args.approvals_device_address or None,
        risk_overrides=json.loads(args.risk_json),
        trace_path=args.trace or None,
        analysis_url=args.analysis_url or None,
        job_limits=json.loads(args.job_limits_json),
    )
    if args.transport == "streamable-http":
        print(
            f"Starting {args.name} at http://{args.http_host}:{args.http_port}/mcp "
            f"for Tango DB {args.tango_host}:{args.tango_port}",
            flush=True,
        )
        server.start(transport="streamable-http", host=args.http_host, port=args.http_port)
    else:
        server.start(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
