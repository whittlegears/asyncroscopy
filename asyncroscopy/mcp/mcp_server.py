"""FastMCP bridge for asyncroscopy Tango devices."""

import argparse
import base64
import functools
import inspect
import io
import re
import json
import socket
import time
import traceback
from dataclasses import dataclass
from typing import Annotated, Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from pydantic import Field

import tango
from tango import Database, DeviceProxy, CommandInfo, CmdArgType, AttrDataFormat, AttrWriteType
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
from fastmcp.tools import tool, Tool, ToolResult
from fastmcp.server.server import Transport
from fastmcp.utilities.types import Image as MCPImage
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from asyncroscopy.data.data_reader import describe_tiled_node
from asyncroscopy.data.tiled_client import list_acquisitions as _list_tiled_acquisitions
from asyncroscopy.data.tiled_client import allowed_origins, open_client, uri_from_data_proxy

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

# Commands known to outlive the default cap get their own proxy with a longer
# timeout (DATA registration walks the Tiled catalog file by file). The MCP HTTP
# client's own request timeout still applies; the point is that Tango does not
# abandon a call that is legitimately still running.
COMMAND_TIMEOUT_OVERRIDES_MILLIS = {
    "register_save_path": 3_600_000,
    "register_path": 120_000,
    "start_tiled_server": 120_000,
}

# Attributes every Tango device carries that duplicate the State/Status commands.
SKIPPED_ATTRIBUTES = {"state", "status"}

# A device can fail discovery transiently (its server still initializing, a
# slow first connect) and was previously skipped silently, so the registered
# tool count came up short with no explanation. Retry each device briefly, and
# report anything still skipped in an unconditional "MCP WARNING" line.
DISCOVERY_ATTEMPTS = 3
DISCOVERY_RETRY_DELAY_SECONDS = 2.0

# Failure signatures after which the rebuild-and-retry in _create_wrapper must NOT
# re-execute the command. Timeouts and monitor contention mean the first attempt is
# likely still running server-side: a retry then executes the command a second time
# and queues behind the device's serialization monitor (observed live: a retried
# DATA.register_save_path raced its own first attempt and deleted Tiled catalog
# entries). PyDs_PythonError means the device's own code raised — deterministic, so
# a retry only re-runs a command that already executed and failed.
NON_RETRYABLE_SIGNATURES = {
    "API_DeviceTimedOut": (
        "the call timed out, so the command is likely still running on the device; "
        "invoking it again would execute it a second time. Wait, then check the "
        "device's state before calling again"
    ),
    "TRANSIENT_CallTimedout": (
        "the call timed out, so the command is likely still running on the device; "
        "invoking it again would execute it a second time. Wait, then check the "
        "device's state before calling again"
    ),
    "not able to acquire serialization": (
        "another command is still executing on this device (its serialization "
        "monitor is held); wait for that command to finish before calling again"
    ),
    "PyDs_PythonError": (
        "the device's own code rejected the command, so retrying would just "
        "execute it again with the same outcome"
    ),
}


def command_timeout_millis(command_name: str) -> int:
    return COMMAND_TIMEOUT_OVERRIDES_MILLIS.get(command_name, COMMAND_TIMEOUT_MILLIS)


_SPECTRUM_TYPES = {
    CmdArgType.DevDouble: CmdArgType.DevVarDoubleArray,
    CmdArgType.DevFloat: CmdArgType.DevVarFloatArray,
    CmdArgType.DevLong: CmdArgType.DevVarLongArray,
    CmdArgType.DevLong64: CmdArgType.DevVarLong64Array,
    CmdArgType.DevShort: CmdArgType.DevVarShortArray,
    CmdArgType.DevULong: CmdArgType.DevVarULongArray,
    CmdArgType.DevULong64: CmdArgType.DevVarULong64Array,
    CmdArgType.DevUShort: CmdArgType.DevVarUShortArray,
    CmdArgType.DevUChar: CmdArgType.DevVarCharArray,
    CmdArgType.DevString: CmdArgType.DevVarStringArray,
    CmdArgType.DevBoolean: CmdArgType.DevVarBooleanArray,
}


@dataclass
class AttributeToolInfo:
    """CommandInfo look-alike for a tool that reads or writes one Tango attribute.

    Attribute devices (SCAN, STAGE, CAMERA, EDS) carry their settings as
    attributes and define no commands, so without these the bridge could
    trigger an acquisition but never change dwell time, image size, exposure or
    a stage axis. ``mode`` is "read" (a get_<attr> tool) or "write" (set_<attr>).
    """

    attribute_name: str
    mode: str
    in_type: CmdArgType
    out_type: CmdArgType
    in_type_desc: str
    out_type_desc: str


def attribute_value_type(info: Any) -> CmdArgType:
    """Map an AttributeInfoEx to the CmdArgType its value travels as."""
    data_type = CmdArgType(info.data_type)
    if info.data_format == AttrDataFormat.SCALAR:
        return data_type
    if info.data_format == AttrDataFormat.SPECTRUM:
        return _SPECTRUM_TYPES.get(data_type, CmdArgType.DevVarDoubleArray)
    return CmdArgType.DevEncoded  # IMAGE: nested list, schema left open


def attribute_tool_infos(info: Any) -> list[AttributeToolInfo]:
    """Build the get_/set_ tool descriptions for one attribute."""
    name = str(info.name)
    value_type = attribute_value_type(info)
    parts = [str(info.description or "").strip()]
    if info.unit and info.unit.lower() not in ("", "no unit"):
        parts.append(f"unit: {info.unit}")
    if info.min_value and info.min_value.lower() != "not specified":
        parts.append(f"min: {info.min_value}")
    if info.max_value and info.max_value.lower() != "not specified":
        parts.append(f"max: {info.max_value}")
    description = "; ".join(part for part in parts if part)
    tools = [
        AttributeToolInfo(
            attribute_name=name, mode="read", in_type=CmdArgType.DevVoid, out_type=value_type,
            in_type_desc="", out_type_desc=description,
        )
    ]
    if info.writable in (AttrWriteType.READ_WRITE, AttrWriteType.WRITE, AttrWriteType.READ_WITH_WRITE):
        tools.append(
            AttributeToolInfo(
                attribute_name=name, mode="write", in_type=value_type, out_type=CmdArgType.DevVoid,
                in_type_desc=f":param {name}: {description}" if description else f":param {name}:",
                out_type_desc="",
            )
        )
    return tools


def plain_attribute_value(value: Any) -> Any:
    """Make an attribute value JSON-safe (DevState and enums become strings)."""
    if isinstance(value, tango.DevState):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [plain_attribute_value(item) for item in value]
    return value


def bind_tool(dev: Any, info: Any, name: str) -> Callable:
    """Bind a tool to a (possibly rebuilt) DeviceProxy.

    Commands go through command_inout, not getattr: DeviceProxy defines
    client-side methods (reconnect, ping, ...) that shadow same-named Tango
    commands, and getattr silently returns the client method instead.
    """
    attribute_name = getattr(info, "attribute_name", None)
    if attribute_name is None:
        return functools.partial(dev.command_inout, name)
    if info.mode == "read":
        return lambda: plain_attribute_value(dev.read_attribute(attribute_name).value)
    return lambda value: dev.write_attribute(attribute_name, value)


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
        """
        self.database = Database(tango_host, tango_port)
        self.mcp = FastMCP(name)

        self.blocked_functions = {key: list(value) for key, value in blocked_functions.items()}
        self.blocked_classes = list(blocked_classes)
        self._blocked_classes_normalized = {cls_name.lower() for cls_name in self.blocked_classes}
        self.data_device_address = data_device_address
        self.include_only_functions = list(include_only_functions) if include_only_functions else []
        self.verbose = verbose
        self.tools: dict[str, dict[str, Callable]] = {}
        # Devices whose tools could not be discovered, name -> failure reason.
        self.skipped_devices: dict[str, str] = {}

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
        uri = uri_from_data_proxy(DeviceProxy(address))

        client = open_client(uri)
        try:
            node = client[key]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Could not resolve data key {key!r} from Tiled server {uri!r}"
            ) from exc

        return describe_tiled_node(key, uri, node, max_values=max_values)

    @tool()
    def list_acquisitions(
        self,
        acquisition_type: str | None = None,
        since: str | None = None,
        limit: int = 20,
        with_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """List recent acquisitions on the DATA/Tiled server, newest first.

        acquisition_type filters by key prefix (stem_image, camera_image,
        spectrum, stem_data, diffraction); since is an ISO-8601 timestamp.
        Each entry carries the key to pass to get_data_from_key and, when
        with_metadata is true, the instrument state recorded at acquisition.
        """
        uri = uri_from_data_proxy(DeviceProxy(self.data_device_address))
        return _list_tiled_acquisitions(
            open_client(uri), acquisition_type=acquisition_type, since=since, limit=limit, with_metadata=with_metadata
        )

    @tool()
    def refresh_devices(self) -> dict[str, Any]:
        """Re-run Tango device discovery and register tools for devices started after this bridge.

        Existing tools are replaced with fresh bindings. Returns the tool count
        per Tango class and any devices that still failed discovery.
        """
        registered = self._register_device_tools(print_summary=False)
        return {
            "tools": {dev_class: sorted(names) for dev_class, names in registered.items()},
            "skipped_devices": dict(self.skipped_devices),
        }

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
        uri = uri_from_data_proxy(DeviceProxy(self.data_device_address))
        return open_client(uri)[key]

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
        if command_name not in IMAGE_PREVIEW_COMMANDS and command_name not in SPECTRUM_PREVIEW_COMMANDS:
            return result
        # TIFF acquisitions return a JSON list of keys (one file per detector);
        # preview the first one and keep the full list as the text result.
        preview_key = result
        if result.startswith("["):
            try:
                keys = json.loads(result)
            except json.JSONDecodeError:
                keys = []
            if keys and all(isinstance(item, str) for item in keys):
                preview_key = keys[0]
        if command_name in IMAGE_PREVIEW_COMMANDS:
            preview, failure = self._fetch_image_preview(preview_key)
        else:
            preview, failure = self._fetch_spectrum_preview(preview_key)
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
        device_name: str,
    ) -> Callable:
        """Create a wrapper function with a proper signature for a Tango command.

        Args:
            func: The raw Tango device command method
            cmd_info: The CommandInfo object from Tango
            command_name: The name of the command
            dev_class: The Tango device class name
            device_name: The Tango device address, used to rebuild a stale proxy

        Returns:
            A wrapper function with a proper signature
        """

        # The bound command method captures the DeviceProxy built at discovery
        # time. If the device server restarts after this bridge starts, that
        # proxy goes stale and every call fails (often as an opaque pybind11
        # "Caught an unknown exception!"). _invoke rebuilds the proxy and
        # retries once, and turns any remaining failure into a readable error
        # naming the device and command. Failures matching
        # NON_RETRYABLE_SIGNATURES are never retried: re-executing a command
        # that timed out or that the device itself rejected is not recovery.
        proxy_state = {"func": func}

        def _describe(exc: Exception) -> str:
            return f"{type(exc).__name__}: {exc}".strip()

        def _no_retry_reason(exc: Exception) -> str | None:
            description = _describe(exc)
            for signature, reason in NON_RETRYABLE_SIGNATURES.items():
                if signature in description:
                    return reason
            return None

        def _invoke(*call_args):
            try:
                return proxy_state["func"](*call_args)
            except Exception as first_exc:
                no_retry_reason = _no_retry_reason(first_exc)
                if no_retry_reason is not None:
                    raise RuntimeError(
                        f"Tango command {device_name}.{command_name} failed: "
                        f"{_describe(first_exc)}\nNot retrying: {no_retry_reason}."
                    ) from first_exc
                try:
                    dev = DeviceProxy(device_name)
                    dev.set_timeout_millis(command_timeout_millis(command_name))
                    proxy_state["func"] = bind_tool(dev, cmd_info, command_name)
                except Exception as rebuild_exc:
                    raise RuntimeError(
                        f"Tango command {device_name}.{command_name} failed "
                        f"({_describe(first_exc)}) and the device proxy could not "
                        f"be rebuilt ({_describe(rebuild_exc)}). Is the device "
                        f"server running?"
                    ) from first_exc
                try:
                    return proxy_state["func"](*call_args)
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"Tango command {device_name}.{command_name} failed even "
                        f"after rebuilding the device proxy: {_describe(retry_exc)}"
                    ) from retry_exc

        in_type = cmd_info.in_type
        py_type = self._tango_type_to_python(in_type)
        in_desc = cmd_info.in_type_desc

        out_type = cmd_info.out_type
        py_return_type = self._tango_type_to_python(out_type)
        attribute_name = getattr(cmd_info, "attribute_name", None)
        doc_lines = [f"Tango Device Class: {dev_class}"]
        if attribute_name is None:
            doc_lines.append(f"Tango Command: {command_name}")
        else:
            verb = "Reads" if cmd_info.mode == "read" else "Writes"
            doc_lines.append(f"Tango Attribute: {attribute_name} ({verb} the attribute)")
        doc_lines.append(f"Input Type: {in_type.name}")
        if in_desc:
            doc_lines.append(f"Input Description: {in_desc}")
        doc_lines.append(f"Output Type: {out_type.name}")
        if cmd_info.out_type_desc:
            doc_lines.append(f"Output Description: {cmd_info.out_type_desc}")
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
            # Sanitize description; the ":param name:" prefix only names the parameter.
            clean_desc = re.sub(r"^(?::param|@param)\s+\w+:\s*", "", in_desc.replace("\n", " ").strip())
            arg_type = Annotated[py_type, Field(description=clean_desc)] if clean_desc else py_type
        else:
            arg_type = py_type

        if in_type == CmdArgType.DevVoid:
            def wrapper():
                result = _invoke()
                normalized = self._normalize_command_result(out_type, result)
                return self._augment_with_preview(command_name, normalized)

            params = []
            
        elif py_type is dict:
            # For commands taking a dictionary (like DevEncoded), allow arbitrary keyword arguments.
            def wrapper(**kwargs):
                if "arg" in kwargs and len(kwargs) == 1 and isinstance(kwargs["arg"], dict):
                    arg_input = kwargs["arg"]
                else:
                    arg_input = kwargs
                
                result = _invoke(arg_input)
                normalized = self._normalize_command_result(out_type, result)
                return self._augment_with_preview(command_name, normalized)

            # Use VAR_KEYWORD (**kwargs) to make Pydantic accept any incoming fields
            params = [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)]
            
        else:
            # Scalars and standard arrays
            def wrapper(*args, **kwargs):
                # Get first positional arg or parameter name out of kwargs
                arg = args[0] if args else kwargs.get(param_name)
                result = _invoke(arg)
                normalized = self._normalize_command_result(out_type, result)
                return self._augment_with_preview(command_name, normalized)

            params = [inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=arg_type)]

        # Build annotations, omitting VAR_KEYWORD parameters so Pydantic safely allows dynamic extra fields
        wrapper.__annotations__ = {p.name: p.annotation for p in params if p.kind != inspect.Parameter.VAR_KEYWORD}
        wrapper.__annotations__["return"] = py_return_type

        wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=py_return_type)
        wrapper.__doc__ = doc

        unique_name = f"{dev_class}_{command_name}".replace("/", "_").replace("-", "_")
        wrapper.__name__ = unique_name
        wrapper.__qualname__ = unique_name

        return wrapper

    def _is_tool_allowed(self, dev_class: str, name: str) -> bool:
        global_blocks = self.blocked_functions.get("*", [])
        if name in global_blocks or f"{dev_class}.{name}" in global_blocks or name in self.blocked_functions.get(dev_class, []):
            return False
        if self.include_only_functions:
            return (
                name in self.include_only_functions
                or f"{dev_class}.{name}" in self.include_only_functions
                or any(item.endswith(f".{name}") for item in self.include_only_functions)
            )
        return True

    def _find_tools(self) -> dict[str, dict[str, tuple[Callable, CommandInfo, str]]]:
        """Discover tools by querying Tango DB for devices, their commands and attributes.

        Returns a dict mapping dev_class -> tool_name -> (func, info, device_name),
        where info is a CommandInfo or an AttributeToolInfo.
        """
        devices = self._list_all_devices()
        tools: dict[str, dict[str, tuple[Callable, CommandInfo]]] = {}
        self.skipped_devices = {}
        for device_name in devices:
            if self._is_admin_device(device_name):
                continue
            dev = None
            dev_class = None
            commands = None
            attributes: list = []
            last_error = ""
            for attempt in range(1, DISCOVERY_ATTEMPTS + 1):
                try:
                    dev = DeviceProxy(device_name)
                    dev.set_timeout_millis(COMMAND_TIMEOUT_MILLIS)
                    dev_class = dev.info().dev_class
                    if self._is_blocked_class(dev_class):
                        break
                    commands = dev.command_list_query()
                    query = getattr(dev, "attribute_list_query_ex", None)
                    attributes = list(query()) if callable(query) else []
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}".strip()
                    commands = None
                    if attempt < DISCOVERY_ATTEMPTS:
                        if self.verbose:
                            print(
                                f"Discovery attempt {attempt}/{DISCOVERY_ATTEMPTS} failed for "
                                f"{device_name} ({last_error}); retrying in "
                                f"{DISCOVERY_RETRY_DELAY_SECONDS}s"
                            )
                        time.sleep(DISCOVERY_RETRY_DELAY_SECONDS)

            if dev_class is not None and self._is_blocked_class(dev_class):
                continue
            if commands is None:
                self.skipped_devices[device_name] = last_error
                if self.verbose:
                    print(f"Skipping {device_name}: {last_error}")
                continue

            for cmd in commands:
                command_name = cmd.cmd_name
                if not self._is_tool_allowed(dev_class, command_name):
                    continue
                proxy = dev
                if command_name in COMMAND_TIMEOUT_OVERRIDES_MILLIS:
                    proxy = DeviceProxy(device_name)
                    proxy.set_timeout_millis(command_timeout_millis(command_name))
                tools.setdefault(dev_class, {})[command_name] = (bind_tool(proxy, cmd, command_name), cmd, device_name)

            for attr in attributes:
                if str(attr.name).lower() in SKIPPED_ATTRIBUTES:
                    continue
                for info in attribute_tool_infos(attr):
                    tool_name = f"{'get' if info.mode == 'read' else 'set'}_{info.attribute_name}"
                    if not self._is_tool_allowed(dev_class, tool_name):
                        continue
                    tools.setdefault(dev_class, {})[tool_name] = (bind_tool(dev, info, tool_name), info, device_name)
        return tools

    def _register_device_tools(self, print_summary: bool) -> dict[str, dict[str, Callable]]:
        """Discover device tools and (re)register them on the MCP instance."""
        raw_tools = self._find_tools()

        wrapped_tools: dict[str, dict[str, Callable]] = {}
        for dev_class in raw_tools:
            wrapped_tools[dev_class] = {}
            for command_name, (func, cmd_info, device_name) in raw_tools[dev_class].items():
                wrapped = self._create_wrapper(func, cmd_info, command_name, dev_class, device_name)
                wrapped_tools[dev_class][command_name] = wrapped

        if print_summary and self.verbose:
            print("Discovered tools by Tango class:")
            for dev_class in sorted(raw_tools):
                command_names = sorted(raw_tools[dev_class].keys())
                print(f"- {dev_class}: {len(command_names)}")
                for command_name in command_names:
                    print(f"    - {command_name}")

        registered: dict[str, dict[str, Callable]] = {}
        for dev_class in wrapped_tools:
            for command_name, wrapped_func in wrapped_tools[dev_class].items():
                try:
                    tool_obj = Tool.from_function(wrapped_func)
                    provider = getattr(self.mcp, "local_provider", self.mcp)
                    try:
                        provider.remove_tool(tool_obj.name)
                    except Exception:
                        pass
                    self.mcp.add_tool(tool_obj)
                    registered.setdefault(dev_class, {})[command_name] = wrapped_func
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to wrap {dev_class}.{command_name}: {e}")
                        traceback.print_exc()
        self.tools = registered
        return registered

    def setup(self, print_summary: bool = True):
        """Configure tools and add them to the MCP instance.

        Args:
            print_summary: If True, print tool discovery and registration summary.
        """
        native_tools = [self.get_data_from_key, self.list_acquisitions, self.list_devices, self.refresh_devices]
        for native_tool in native_tools:
            self.mcp.add_tool(native_tool)
            if self.verbose:
                print(f"Registered native tool: {native_tool.__name__}")

        registered = self._register_device_tools(print_summary)
        num_device_tools = sum(len(names) for names in registered.values())

        # Printed unconditionally (unlike the verbose summary below) so GUIs and
        # operators see missing tools even in quiet mode: a device that failed
        # discovery means every one of its commands is absent from the tool list.
        if self.skipped_devices:
            print(
                f"MCP WARNING: {len(self.skipped_devices)} device(s) failed tool "
                f"discovery after {DISCOVERY_ATTEMPTS} attempts - their commands "
                f"are NOT registered as tools:",
                flush=True,
            )
            for name, reason in sorted(self.skipped_devices.items()):
                print(f"  - {name}: {reason}", flush=True)

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
    # Binding 0.0.0.0 succeeds even while another server holds 127.0.0.1 on the
    # same port, and that loopback bind then wins every local connection: clients
    # dialing 127.0.0.1 silently reach the other server (observed: a stale bridge
    # for the real instrument shadowing the digital twin's). Connecting tells.
    for candidate in {host, "127.0.0.1"}:
        if candidate == "0.0.0.0":
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
            check.settimeout(0.5)
            if check.connect_ex((candidate, port)) == 0:
                return f"something already accepts connections on {candidate}:{port}"
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
    )
    if args.transport == "streamable-http":
        print(
            f"Starting {args.name} at http://{args.http_host}:{args.http_port}/mcp "
            f"for Tango DB {args.tango_host}:{args.tango_port}",
            flush=True,
        )
        # Browser clients (the SciAgentGUI vite preview) need CORS headers; fastmcp 3 sends
        # none by default. Same loopback list the DATA device hands to Tiled — a hostile
        # web page must never be able to drive instrument tools cross-origin, so never "*".
        browser_dev_origins = allowed_origins()
        cors = Middleware(
            CORSMiddleware,
            allow_origins=browser_dev_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=[
                "content-type",
                "accept",
                "authorization",
                "mcp-session-id",
                "mcp-protocol-version",
                "last-event-id",
            ],
            expose_headers=["mcp-session-id"],
        )
        server.start(
            transport="streamable-http",
            host=args.http_host,
            port=args.http_port,
            middleware=[cors],
        )
    else:
        server.start(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
