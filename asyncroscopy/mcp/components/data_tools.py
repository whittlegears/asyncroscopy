"""Stored data: list_data, view_data, and the picture an acquisition returns with its key.

acquire* commands whose result is a data key (a file name such as "spectrum_eds_20261028T101500.h5")
get a PNG of what they saved, drawn by asyncroscopy.data.views, so clients that render MCP image blocks
(SciAgentGUI) show the capture without another call. A preview never fails the acquisition, except
when it shows the beam was blanked (the file is noise). Reads of Tiled run in lanes: small thread pools
with a slot count, so a stuck Tiled fills one lane with a clear "busy" answer instead of starving every
data tool.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import TYPE_CHECKING, Any, Callable, Literal

import httpx
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult, tool
from fastmcp.utilities.types import Image as MCPImage
from mcp.types import TextContent

from asyncroscopy.core.names import is_data_key
from asyncroscopy.data import tiled_client, views
from asyncroscopy.data.data_reader import numpy_to_python
from asyncroscopy.mcp.interceptors import Call, Interceptor

if TYPE_CHECKING:
    from asyncroscopy.mcp.descriptors import ToolDescriptor
    from asyncroscopy.mcp.mcp_server import MCPServer, ServerConfig

PREVIEW_COMMANDS = {"acquire_camera_image", "acquire_scanned_image", "acquire_spectrum"}  # a failed preview is reported, not silent
PREVIEW_TEXT_CHARS = 1500  # the summary sent with an acquisition's preview (a spectrum's peak table fits)
DATA_TOOL_DEADLINE_S = 20.0  # data tools answer within this, so a slow or stuck Tiled cannot hang the agent
PREVIEW_DEADLINE_S = 10.0
LIST_DATA_WINDOW = 300  # newest keys list_data looks through; Tiled refuses pages larger than 300
LIST_DATA_BUDGET_S = 12.0  # list_data stops looking after this and says how far it got
# What to do next when a lane is busy or Tiled is slow. A preview never suggests trying again: that could re-run the acquisition.
NEXT_STEP = {
    "view": "Try again shortly, or ask for less (a region, or format text).",
    "list": "Try again shortly, or narrow it with search or kind.",
    "preview": "",
}


class Busy(RuntimeError):
    """Every slot of a lane is taken by calls still waiting for Tiled."""


class Lane:
    """A small thread pool with a slot count: work that outlives its deadline keeps its slot until it ends."""

    def __init__(self, name: str, slots: int) -> None:
        self.name, self.slots = name, slots
        self.pool = ThreadPoolExecutor(max_workers=slots, thread_name_prefix=f"data-{name}")
        self.free = threading.BoundedSemaphore(slots)

    def run(self, fn: Callable[[], Any], deadline_s: float) -> Any:
        if not self.free.acquire(blocking=False):
            raise Busy(f"{self.slots} earlier {self.name} calls are still waiting for Tiled")
        future = self.pool.submit(fn)
        future.add_done_callback(lambda _f: self.free.release())
        return future.result(timeout=deadline_s)


class DataTools:
    def __init__(self, server: MCPServer, cfg: ServerConfig) -> None:
        self.server = server
        self.data_device_address = cfg.data_device_address
        self.verbose = cfg.verbose
        self.lanes = {"view": Lane("view_data", 4), "list": Lane("list_data", 2), "preview": Lane("preview", 2)}

    def natives(self) -> list:
        return [self.list_data, self.view_data]

    def tiled_node(self, key: str, hints: bool = True) -> Any:
        """The Tiled node for a DATA/Tiled key, through the DATA device's Tiled address.

        With hints, a missing key's error names keys that start with it (a scan of the catalog);
        without, a missing key is a plain KeyError.
        """
        uri = tiled_client.uri_from_data_device(self.data_device_address)
        return tiled_client.node_for_key(key, uri) if hints else tiled_client.connect(uri)[key]

    def with_deadline(self, fn: Callable[[], Any], lane: str = "view", deadline_s: float | None = None) -> Any:
        """Run a data read in its lane within the deadline; every failure becomes an error the agent can act on."""
        deadline_s = DATA_TOOL_DEADLINE_S if deadline_s is None else deadline_s
        next_step = NEXT_STEP[lane]
        try:
            return self.lanes[lane].run(fn, deadline_s)
        except ToolError:
            raise
        except Busy as exc:
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
        return self.with_deadline(lambda: self._list_data(kind, search, max(1, min(int(limit), 100)), max(0, int(offset)), started), lane="list")

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
        summary, png = self.with_deadline(lambda: views.view(self.tiled_node(key), key, format, dataset, options), lane="view")
        # A TextContent block, not a bare str: FastMCP joins a list of only strings into one JSON array.
        content: list[Any] = [TextContent(type="text", text=summary["text"])]
        if png is not None:
            content.append(MCPImage(data=png, format="png"))
        return ToolResult(content=content, structured_content=numpy_to_python(summary))

    def augment_with_preview(self, command_name: str, result: Any, device: str = "", arguments: dict | None = None) -> Any:
        """Attach a PNG of what an acquire command saved; fails the acquisition only when it was taken with the beam
        blanked (``beam_blanked``).

        Returns a ToolResult with the Tiled key as text and the image, plus structured_content matching the
        tool's {"result": str} output schema; a bare (text, Image) tuple would fail client-side output-schema
        validation. For PREVIEW_COMMANDS a failed preview is reported in a text block, so a Tiled outage is
        visible, not silent.
        """
        if not isinstance(result, str) or not command_name.startswith("acquire") or not is_data_key(result):
            return result
        try:
            summary, png = self.with_deadline(
                lambda: views.view(self.tiled_node(result, hints=False), result, "png"), lane="preview", deadline_s=PREVIEW_DEADLINE_S
            )
        except Exception as exc:  # the hardware already acquired: report, never raise
            if self.verbose:
                print(f"[preview] could not build a preview for {result!r}: {exc}")
            if command_name not in PREVIEW_COMMANDS:
                return result
            note = f"image preview unavailable ({exc}). The data were saved; look at them later with view_data(key={result!r}) instead of acquiring again."
            return ToolResult(content=[TextContent(type="text", text=result), TextContent(type="text", text=note)], structured_content={"result": result})
        if summary.get("beam_blanked"):
            raise beam_blanked(result, command_name, device, arguments)
        # The text summary (statistics; for a spectrum its peaks and candidate elements) goes with the picture,
        # so a model that cannot see images still learns what it acquired without another call.
        described = str(summary.get("text") or "")[:PREVIEW_TEXT_CHARS]
        content = [TextContent(type="text", text=result)] + ([TextContent(type="text", text=described)] if described else []) + [MCPImage(data=png, format="png")]
        return ToolResult(content=content, structured_content={"result": result})


def beam_blanked(key: str, command_name: str, device: str, arguments: dict | None) -> ToolError:
    """The error for an acquisition taken with the beam blanked, as JSON: the calls that retake it (unblank, then
    the same acquisition again) and the noise-only file that was saved, which is not a result."""
    unblank = f"{device}_unblank_beam" if device else "unblank_beam"
    retake = f"{device}_{command_name}" if device else command_name
    return ToolError(json.dumps({
        "error": {"code": "beam_blanked", "message": "Not acquired: the beam is blanked."},
        "state": "not_acquired",
        "next_actions": [{"tool": unblank, "args": {}}, {"tool": retake, "args": dict(arguments or {})}],
        "warnings": [{"code": "noise_only_file", "key": key}],
    }, default=str))


class PreviewInterceptor(Interceptor):
    """Adds the picture to an acquisition's reply (acquire* commands only). Outermost: it runs after the call is logged."""

    order = 10

    def __init__(self, data: DataTools) -> None:
        self.data = data

    def applies(self, tool: ToolDescriptor) -> bool:
        return bool(tool.command and tool.command.startswith("acquire"))

    def after_call(self, call: Call, result: Any) -> Any:
        return self.data.augment_with_preview(call.tool.command or "", result, call.tool.device or "", call.arguments)


def register(server: MCPServer, cfg: ServerConfig) -> DataTools:
    data = DataTools(server, cfg)
    server.add_interceptor(PreviewInterceptor(data))
    return data
