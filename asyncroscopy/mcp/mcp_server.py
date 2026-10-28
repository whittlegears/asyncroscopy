"""FastMCP bridge for asyncroscopy Tango devices."""

import argparse
import base64
import inspect
import io
import re
import json
import socket
import threading
import time
import traceback
from typing import Annotated, Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from pydantic import Field
from tiled.client import from_uri

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

from asyncroscopy.data.data_reader import describe_tiled_node
from asyncroscopy.mcp.tool_log import ToolCallLog, TraceWriter
from asyncroscopy.safety.policy import is_envelope_device, is_stem_device, operator_only, risk_of

ENDPOINT_ROLES = ("agent", "operator")
CALLER_HEADER = "x-asyncroscopy-caller"

# Tango commands that acquire and save data, returning its DATA/Tiled key as a
# plain string. Their tool wrappers additionally fetch the array back from Tiled and
# attach an inline PNG preview, so chat clients that already render MCP image content
# blocks (e.g. SciAgentGUI) display the capture without any extra tool call.
# Image commands are previewed as a grayscale rendering of the first 2D dataset;
# spectrum commands as a plot of the first 1D dataset.
IMAGE_PREVIEW_COMMANDS = {"acquire_camera_image", "acquire_scanned_image"}
SPECTRUM_PREVIEW_COMMANDS = {"acquire_spectrum"}

# Tango's client default of 3000 ms is shorter than a real acquisition: the digital
# twin's first acquire_camera_image takes over 3 s cold, so the call died with
# API_DeviceTimedOut while the command kept running server-side. SciAgentGUI allows
# 60 s per MCP HTTP request, so 30 s lets slow commands finish while still failing
# inside the client's window with a readable Tango error rather than an HTTP timeout.
COMMAND_TIMEOUT_MILLIS = 30_000
JOB_WAIT_MAX_S = 25.0  # wait_for_job stays under MCP client timeouts

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
    t = text.strip().lower()
    return t in ("", "uninitialised", "uninitialized", "none") or t.startswith((":return: (not documented)", "no input parameter", "no output parameter"))

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
    ):
        """
        Args:
            name (str): Display name for the MCP server instance.
            tango_host (str): Hostname of the Tango database server (e.g. "localhost").
            tango_port (int): Port of the Tango database server (e.g. 9094).
            blocked_functions: Command names to exclude, keyed by Tango class name.
                Use "*" for global blocks.
            blocked_classes: Tango device class names to skip entirely.
            data_device_address: Tango DATA device used by get_data_from_key.
            include_only_functions (list[str], optional): Command names/patterns to allow exclusively.
            verbose (bool, optional): If True, print device discovery and tool registration
                progress to stdout. Defaults to True.
            tool_log_path (str, optional): JSONL file with one line per tool call. Off when empty.
            endpoint_role: "agent" (default) hides operator-only commands such as
                Approvals.approve and export_ground_truth; "operator" is for the GUI.
            approvals_device_address: Approvals device holding the approval mode.
                When unset, every call runs (there is no mode to consult).
            risk_overrides: {"Class.command" glob: "low"|"high"} checked before the defaults.
            trace_path: Instrument Trace JSONL file. Off when empty.
        """
        if endpoint_role not in ENDPOINT_ROLES:
            raise ValueError(f"endpoint_role must be one of {ENDPOINT_ROLES}, got {endpoint_role!r}")
        self.database = Database(tango_host, tango_port)
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
        self.risk_overrides = dict(risk_overrides or {})
        self.trace = TraceWriter(trace_path, source="mcp") if trace_path else None
        # tool name -> (arguments + error signature, consecutive count, time of last failure)
        self._failures: dict[str, tuple[str, int, float]] = {}
        self._failures_lock = threading.Lock()
        self.tools: dict[str, dict[str, Callable]] = {}
        self._devices: dict[str, DeviceProxy] = {}  # tool prefix -> its device

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

    @tool()
    def get_data_from_key(
        self,
        key: str,
        max_values: int = 64,
        data_device_address: str | None = None,
    ) -> dict[str, Any]:
        """Read a remote DATA/Tiled key and return dataset metadata plus small previews."""
        address = data_device_address or self.data_device_address
        data = DeviceProxy(address)
        config = json.loads(data.get_config())
        uri = config.get("uri")
        if not uri:
            raise RuntimeError(f"DATA device {address!r} did not provide a Tiled URI")

        client = from_uri(uri)
        try:
            node = client[key]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Could not resolve data key {key!r} from Tiled server {uri!r}"
            ) from exc

        return describe_tiled_node(key, uri, node, max_values=max_values)

    @staticmethod
    def _find_first_2d_array(node: Any, prefer_key: str | None = None) -> np.ndarray | None:
        """Recursively search a Tiled node for the first readable 2D dataset.

        Acquisition writers (see asyncroscopy/data/data_writer.py) group image
        datasets under an "image/<detector>" path, so ``prefer_key`` lets callers
        check that group first before falling back to a full walk.
        """

        def search(current: Any) -> np.ndarray | None:
            read = getattr(current, "read", None)
            if callable(read):
                try:
                    array = np.asarray(read())
                except Exception:
                    return None
                return array if array.ndim == 2 else None

            keys = getattr(current, "keys", None)
            if callable(keys):
                for child_name in keys():
                    try:
                        found = search(current[child_name])
                    except Exception:
                        continue
                    if found is not None:
                        return found
            return None

        if prefer_key is not None:
            try:
                found = search(node[prefer_key])
                if found is not None:
                    return found
            except Exception:
                pass
        return search(node)

    @staticmethod
    def _array_to_png_bytes(array: np.ndarray, max_side: int = 1024) -> bytes:
        """Normalize a 2D array to 8-bit grayscale and encode it as PNG."""
        values = np.asarray(array, dtype=np.float64)
        finite = values[np.isfinite(values)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        normalized = (values - low) / (high - low) if high > low else np.zeros_like(values)
        pixels = np.clip(normalized, 0.0, 1.0)
        image = PILImage.fromarray((pixels * 255).astype(np.uint8), mode="L")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(new_size, PILImage.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _resolve_tiled_node(self, key: str) -> Any:
        """Resolve a DATA/Tiled key to its Tiled node via the DATA device's config."""
        data = DeviceProxy(self.data_device_address)
        config = json.loads(data.get_config())
        uri = config.get("uri")
        if not uri:
            raise RuntimeError("the DATA device's config carries no Tiled uri")
        return from_uri(uri)[key]

    def _fetch_image_preview(self, key: str) -> tuple[MCPImage | None, str | None]:
        """Fetch a captured image from Tiled and render it as a PNG preview.

        Returns (preview, None) on success and (None, reason) on any failure
        (unreachable Tiled server, key not yet registered, no 2D dataset found),
        so the acquisition tool can still return its text key and state why no
        preview accompanies it. Swallowing the reason made a Tiled outage
        indistinguishable from a command that never produces images.
        """
        try:
            node = self._resolve_tiled_node(key)
            array = self._find_first_2d_array(node, prefer_key="image")
            if array is None:
                return None, f"no 2D dataset was found under key {key!r} in Tiled"
            return MCPImage(data=self._array_to_png_bytes(array), format="png"), None
        except Exception as exc:
            if self.verbose:
                print(f"[image preview] could not build preview for {key!r}: {exc}")
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _find_first_1d_array(
        node: Any, prefer_key: str | None = None
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Recursively search a Tiled node for the first readable 1D dataset.

        Returns (array, metadata) so the caller can label the plot from the
        dataset's HDF5 attributes (see data_writer.save_acquisition, which
        stores spectra under a "spectrum" dataset name).
        """

        def search(current: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
            read = getattr(current, "read", None)
            if callable(read):
                try:
                    array = np.asarray(read())
                except Exception:
                    return None
                if array.ndim != 1 or array.size == 0:
                    return None
                metadata = dict(getattr(current, "metadata", {}) or {})
                return array, metadata

            keys = getattr(current, "keys", None)
            if callable(keys):
                for child_name in keys():
                    try:
                        found = search(current[child_name])
                    except Exception:
                        continue
                    if found is not None:
                        return found
            return None

        if prefer_key is not None:
            try:
                found = search(node[prefer_key])
                if found is not None:
                    return found
            except Exception:
                pass
        return search(node)

    @staticmethod
    def _element_labels(metadata: dict[str, Any]) -> list[str] | None:
        """Extract per-channel element labels from a spectrum dataset's attrs.

        data_writer json-encodes non-scalar HDF5 attrs, so "elements" may arrive
        as a JSON string or as a list. Returns None when absent or malformed,
        in which case the preview falls back to an unlabeled channel axis.
        """
        raw = metadata.get("elements")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if isinstance(raw, (list, tuple)) and raw and all(isinstance(item, str) for item in raw):
            return list(raw)
        return None

    @staticmethod
    def _spectrum_to_png_bytes(values: np.ndarray, labels: list[str] | None = None) -> bytes:
        """Render a 1D spectrum as a PNG plot.

        With one label per value (the digital twin's per-element composition
        spectra) this draws a labeled bar chart; otherwise a line plot against
        channel index, which is the only axis honestly known without calibration.
        """
        counts = np.asarray(values, dtype=np.float64)
        figure, axes = plt.subplots(figsize=(6.0, 3.5), dpi=120)
        try:
            if labels is not None and len(labels) == len(counts):
                positions = np.arange(len(counts))
                axes.bar(positions, counts)
                axes.set_xticks(positions)
                axes.set_xticklabels(labels)
                axes.set_xlabel("element")
                axes.set_ylabel("relative intensity")
            else:
                axes.plot(np.arange(counts.size), counts, linewidth=1.0)
                axes.set_xlabel("channel")
                axes.set_ylabel("counts")
            figure.tight_layout()
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png")
            return buffer.getvalue()
        finally:
            plt.close(figure)

    def _fetch_spectrum_preview(self, key: str) -> tuple[MCPImage | None, str | None]:
        """Fetch an acquired spectrum from Tiled and render it as a PNG plot.

        Same contract as _fetch_image_preview: (preview, None) on success,
        (None, reason) on failure so the tool result states why no preview
        accompanies the key.
        """
        try:
            node = self._resolve_tiled_node(key)
            found = self._find_first_1d_array(node, prefer_key="spectrum")
            if found is None:
                return None, f"no 1D dataset was found under key {key!r} in Tiled"
            array, metadata = found
            labels = self._element_labels(metadata)
            return MCPImage(data=self._spectrum_to_png_bytes(array, labels), format="png"), None
        except Exception as exc:
            if self.verbose:
                print(f"[spectrum preview] could not build preview for {key!r}: {exc}")
            return None, f"{type(exc).__name__}: {exc}"

    def _augment_with_preview(self, command_name: str, result: Any) -> Any:
        """Attach an inline image preview to known acquisition commands.

        Returns a ToolResult with both a text content block (the Tiled key,
        unchanged for existing callers) and an image content block, plus
        structured_content matching the tool's auto-generated {"result": str}
        output schema — a bare (text, Image) tuple would leave structured_content
        empty and fail client-side output-schema validation. When the preview
        cannot be built, a text block states the reason instead of the image, so
        the failure is visible to the operator and the model rather than silent.
        """
        if not isinstance(result, str) or not result:
            return result
        if command_name in IMAGE_PREVIEW_COMMANDS:
            preview, failure = self._fetch_image_preview(result)
        elif command_name in SPECTRUM_PREVIEW_COMMANDS:
            preview, failure = self._fetch_spectrum_preview(result)
        else:
            return result
        if preview is None:
            return ToolResult(
                content=[result, f"image preview unavailable: {failure}"],
                structured_content={"result": result},
            )
        return ToolResult(content=[result, preview], structured_content={"result": result})

    @staticmethod
    def _hdf5_attrs_to_json(attrs: Any) -> dict[str, Any]:
        return {key: MCPServer._numpy_to_python(value) for key, value in attrs.items()}

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
    def _numpy_to_python(obj: Any) -> Any:
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return MCPServer._numpy_to_python(obj.tolist())
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: MCPServer._numpy_to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            conv = [MCPServer._numpy_to_python(v) for v in obj]
            return tuple(conv) if isinstance(obj, tuple) else conv
        return obj

    @staticmethod
    def _normalize_command_result(out_type: CmdArgType, result: Any) -> Any:
        """Convert Tango command output into JSON-safe data for MCP transport."""

        # Convert numpy types (including nested containers) to native Python types
        result = MCPServer._numpy_to_python(result)
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

        envelope = (tool_prefix or dev_class) in getattr(self, "_envelope_prefixes", set())
        stem = (tool_prefix or dev_class) in getattr(self, "_stem_prefixes", set())
        risk = risk_of(dev_class, command_name, self.risk_overrides, envelope=envelope, stem=stem)

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
        doc_lines = [
            f"Tango Device Class: {dev_class}",
            f"Tango Command: {command_name}",
            f"Input Type: {in_type.name}",
        ]
        if in_desc and not _placeholder_doc(in_desc):
            doc_lines.append(f"Input Description: {in_desc}")
        doc_lines.append(f"Output Type: {out_type.name}")
        out_desc = cmd_info.out_type_desc
        if out_desc and not _placeholder_doc(out_desc):
            doc_lines.append(f"Output Description: {out_desc}")
            # Lead with what the command is for: models weigh the first line most when choosing a tool.
            doc_lines.insert(0, out_desc)
        job_kind = getattr(self, "_job_kinds", {}).get(tool_prefix or dev_class)
        if envelope and job_kind:
            doc_lines.insert(0, f"{tool_prefix or dev_class} job: {job_kind}.")
        plan_example = getattr(self, "_plan_examples", {}).get(tool_prefix or dev_class)
        if envelope and command_name == "reserve" and plan_example:
            doc_lines.insert(1, f'Example request: {{"plan": {plan_example}}}')
        doc = "\n".join(doc_lines)

        # Get parameter name from docstring description text
        param_name = "arg"
        if in_desc:
            match = re.search(r'(?::param|@param)\s+(\w+):', in_desc)
            if match:
                param_name = match.group(1)

        if in_desc and in_desc.lower() not in (
            "uninitialised",
            "none",
            "",
            "uninitialized",
        ):
            # Sanitize description
            clean_desc = in_desc.replace("\n", " ").strip()
            # ":param name:" only names the parameter; the model should see just the description.
            clean_desc = re.sub(r"^(?::param|@param)\s+\w+:\s*", "", clean_desc)
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
                return _run({param_name: self._numpy_to_python(arg)}, arg)

            params = [inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[arg_type | None, Field(description=clean_desc)], default=None)]
            params += [inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[Any, Field(description=f"Same as {param_name}.{f}.")], default=None) for f in fields]

        else:
            # Scalars and standard arrays
            def wrapper(*args, **kwargs):
                # Get first positional arg or parameter name out of kwargs
                arg = args[0] if args else kwargs.get(param_name)
                return _run({param_name: self._numpy_to_python(arg)}, arg)

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
        if self.approvals_device_address is None or risk != "high":
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

    @tool()
    def list_skills(self) -> list[dict[str, Any]]:
        """Skills: procedures for kinds of experiment (e.g. mapping). Load one with load_skill before doing that kind of task."""
        from asyncroscopy.skills import list_skills

        return [{"name": s.name, "description": s.description, "version": s.version, "available": s.available, **({"problem": s.problem} if s.problem else {})} for s in list_skills(self._tool_names())]

    @tool()
    def load_skill(self, name: str) -> str:
        """Full instructions of one skill from list_skills. Follow them for that kind of task."""
        from asyncroscopy.skills import load_skill

        try:
            return load_skill(name, self._tool_names())
        except (KeyError, RuntimeError) as exc:
            raise ToolError(str(exc).strip("'\"")) from exc

    @tool()
    def approval_status(self, request_id: str) -> dict[str, Any]:
        """State of an approval request: pending, approved, denied or expired. After 'approved', repeat the held call unchanged."""
        if self.approvals_device_address is None:
            return {"request_id": request_id, "state": "approved", "note": "this server has no approval mode"}
        return json.loads(self._approvals().request_status(request_id))

    @tool()
    def wait_for_job(self, device: str, job_id: str, max_wait_s: float = 25.0) -> dict[str, Any]:
        """Wait until a background job finishes (or max_wait_s, at most 25 s, passes) and return its status. device is the tool prefix, e.g. "Mapper". Use this instead of calling <device>_job_status over and over."""
        proxy = self._devices.get(device)
        if proxy is None or device not in getattr(self, "_envelope_prefixes", set()):
            raise ToolError(f"no job device {device!r}; job devices here: {sorted(getattr(self, '_envelope_prefixes', set()))}")
        deadline = time.monotonic() + min(max(float(max_wait_s), 0.0), JOB_WAIT_MAX_S)
        while True:
            # Polls job_status (low risk) rather than blocking in the device, so abort stays immediate.
            status = json.loads(proxy.job_status(json.dumps({"job_id": job_id})))
            if status.get("error", {}).get("code") == "unknown_job":
                raise ToolError(
                    f"{job_id!r} is not a {device} job. Jobs started by assemble_map or score_map run on the analysis "
                    "server: wait for them with analysis_job_status and wait_s=25."
                )
            if "error" in status or status.get("state") not in {"reserved", "queued", "running", "aborting"} or time.monotonic() >= deadline:
                return status
            time.sleep(1.0)

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
        self._prefix_class: dict[str, str] = {}
        self._envelope_prefixes: set[str] = set()
        self._stem_prefixes: set[str] = set()
        self._job_kinds: dict[str, str] = {}
        self._plan_examples: dict[str, str] = {}
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

    def setup(self, print_summary: bool = True):
        """Configure tools and add them to the MCP instance.

        Args:
            print_summary: If True, print tool discovery and registration summary.
        """
        raw_tools = self._find_tools()

        wrapped_tools: dict[str, dict[str, Callable]] = {}
        prefix_class = getattr(self, "_prefix_class", {})
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

        native_tools = [self.get_data_from_key, self.list_devices, self.list_skills, self.load_skill]
        if getattr(self, "_envelope_prefixes", None):
            native_tools.append(self.wait_for_job)
        if self.approvals_device_address is not None:
            native_tools.append(self.approval_status)
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
