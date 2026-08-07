"""
Standalone MCP server for grid stitching and multi-scale sample mapping.

Deliberately decoupled from the instrument bridge
(:mod:`asyncroscopy.mcp.mcp_server`): it has no Tango dependency, so it can
run on a different (and more powerful) machine than the one driving the
microscope, and it keeps opencv/pillow-heavy image processing off the
safety-critical instrument process. Connect it to the GUI as a second MCP
server alongside the instrument bridge.

Typical agent-driven workflow
------------------------------
1. ``start_map`` on this server.
2. Loop on the instrument server: ``move_stage``, ``acquire_scanned_image``,
   resolve the returned key to a path with ``get_data_from_key``; call
   ``add_tiles`` here with that path (one or several tiles per batch - call
   it again as more batches complete, they accumulate into the same map).
3. Optionally ``preview_map`` to sanity-check progress visually.
4. To add spatial context, increase the microscope's field of view, acquire
   an overview image, and call ``register_overview`` here to place it in the
   same coordinate system as the fine tiles.
5. ``finalize_map`` to export the zoomable map bundle.
"""

from __future__ import annotations

import argparse
import base64
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastmcp import FastMCP
from fastmcp.tools import Tool, tool
from fastmcp.server.server import Transport

from asyncroscopy.mapping.grid_stitcher import GridStitcher, GridStitchResult
from asyncroscopy.mapping.map_export import export_map
from asyncroscopy.mapping.montage import load_acquisition_image
from asyncroscopy.mapping.overview import register_overview as _register_overview


@dataclass
class _TileEntry:
    image: np.ndarray
    stage_xy_m: Optional[tuple[float, float]] = None
    source: Optional[str] = None


@dataclass
class _OverviewEntry:
    image: np.ndarray
    registration: Any
    stage_xy_m: Optional[tuple[float, float]] = None
    source: Optional[str] = None


@dataclass
class MapSession:
    name: str
    overlap: float = 0.15
    pixel_size_nm: Optional[float] = None
    stage_origin_m: Optional[list[float]] = None
    tiles: dict[tuple[int, int], _TileEntry] = field(default_factory=dict)
    overviews: list[_OverviewEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class MappingServer:
    """Session-based grid stitching + multi-scale registration, exposed over MCP."""

    def __init__(self, name: str, output_root: str | Path, verbose: bool = True):
        self.mcp = FastMCP(name)
        self.output_root = Path(output_root)
        self.verbose = verbose
        self._sessions: dict[str, MapSession] = {}

    @staticmethod
    def _load_image(image_path: str | None = None, image_b64: str | None = None) -> np.ndarray:
        if image_path:
            return load_acquisition_image(image_path)
        if image_b64:
            raw = base64.b64decode(image_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not decode image_b64 as an image.")
            return img
        raise ValueError("Provide image_path or image_b64.")

    def _get_session(self, map_id: str) -> MapSession:
        session = self._sessions.get(map_id)
        if session is None:
            raise ValueError(f"Unknown map_id {map_id!r}. Call start_map first.")
        return session

    def _stitch(self, session: MapSession) -> GridStitchResult:
        grid = {rc: entry.image for rc, entry in session.tiles.items()}
        return GridStitcher(overlap=session.overlap).stitch(grid)

    @tool()
    def start_map(
        self,
        name: str = "sample_map",
        overlap: float = 0.15,
        pixel_size_nm: float | None = None,
        stage_origin_m: list[float] | None = None,
    ) -> dict[str, Any]:
        """Start a new incremental sample-map session and return its map_id.

        Pass the returned map_id to add_tiles, preview_map, register_overview
        and finalize_map. overlap is the nominal fractional overlap between
        neighbouring grid tiles (0-0.5).
        """
        if not 0.0 < overlap < 0.6:
            raise ValueError("overlap must be between 0 and 0.6")
        map_id = uuid.uuid4().hex[:12]
        self._sessions[map_id] = MapSession(
            name=name, overlap=overlap, pixel_size_nm=pixel_size_nm, stage_origin_m=stage_origin_m
        )
        return {"map_id": map_id, "name": name}

    @tool()
    def add_tiles(self, map_id: str, tiles: list[dict[str, Any]]) -> dict[str, Any]:
        """Add a batch of acquired tiles to a map session; call repeatedly as batches complete.

        Each tile: {row, col, image_path (local file: .h5/.tiff/.png/...) or
        image_b64 (base64-encoded PNG/JPEG bytes), stage_xy_m (optional [x,y]
        in metres)}. Tiles accumulate across calls and are re-registered as a
        whole grid each time, so later batches correct/extend earlier ones.
        Returns stitch-quality diagnostics so far.
        """
        session = self._get_session(map_id)
        if not tiles:
            raise ValueError("tiles must be non-empty.")
        for t in tiles:
            if "row" not in t or "col" not in t:
                raise ValueError(f"Tile missing row/col: {t}")
            image = self._load_image(t.get("image_path"), t.get("image_b64"))
            rc = (int(t["row"]), int(t["col"]))
            session.tiles[rc] = _TileEntry(
                image=image,
                stage_xy_m=tuple(t["stage_xy_m"]) if t.get("stage_xy_m") else None,
                source=t.get("image_path"),
            )
        result = self._stitch(session)
        return {
            "map_id": map_id,
            "tiles_total": len(session.tiles),
            "edges_measured": sum(e.used for e in result.edges),
            "edges_total": len(result.edges),
            "mean_edge_residual_px": result.mean_residual,
            "canvas_px": list(result.canvas.shape[:2][::-1]),
        }

    @tool()
    def preview_map(self, map_id: str, max_dim: int = 640) -> dict[str, Any]:
        """Return a small preview PNG (base64) of the session's current stitched canvas.

        Use this to visually check acquisition progress before finalizing.
        """
        session = self._get_session(map_id)
        if not session.tiles:
            raise ValueError("No tiles added yet.")
        canvas = self._stitch(session).canvas
        h, w = canvas.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            canvas = cv2.resize(canvas, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            raise ValueError("Failed to encode preview image.")
        return {
            "encoding": "base64",
            "mime_type": "image/png",
            "payload": base64.b64encode(buf.tobytes()).decode("ascii"),
            "canvas_px": [w, h],
        }

    @tool()
    def register_overview(
        self,
        map_id: str,
        image_path: str | None = None,
        image_b64: str | None = None,
        pixel_size_nm: float | None = None,
        stage_xy_m: list[float] | None = None,
    ) -> dict[str, Any]:
        """Register a lower-magnification 'zoom out' image against the session's fine mosaic.

        Acquire this image at a larger field of view over roughly the same
        area as the fine tiles, then pass it here. Uses scale-invariant
        feature matching, not phase correlation, since the two images differ
        in scale (and possibly rotation). Returns matched=false rather than
        raising if registration fails, so the agent can retry with a
        different position or a smaller magnification gap.
        """
        session = self._get_session(map_id)
        if not session.tiles:
            raise ValueError("Add fine tiles before registering an overview.")
        overview_img = self._load_image(image_path, image_b64)
        canvas = self._stitch(session).canvas
        reg = _register_overview(overview_img, canvas)
        if reg is None:
            return {
                "matched": False,
                "message": "Registration failed: too few reliable feature matches. "
                "Try a smaller magnification gap between the overview and the fine "
                "tiles, or a different overview position with more contrast/texture.",
            }
        session.overviews.append(
            _OverviewEntry(
                image=overview_img,
                registration=reg,
                stage_xy_m=tuple(stage_xy_m) if stage_xy_m else None,
                source=image_path,
            )
        )
        return {
            "matched": True,
            "scale": reg.scale,
            "rotation_deg": reg.rotation_deg,
            "offset_px": list(reg.offset_px),
            "inliers": reg.inliers,
            "match_count": reg.match_count,
        }

    @tool()
    def get_map_status(self, map_id: str) -> dict[str, Any]:
        """Report progress on a map session: tile count, grid extent, overview count."""
        session = self._get_session(map_id)
        rows = [rc[0] for rc in session.tiles]
        cols = [rc[1] for rc in session.tiles]
        return {
            "map_id": map_id,
            "name": session.name,
            "tiles_total": len(session.tiles),
            "grid_extent": [max(rows, default=-1) + 1, max(cols, default=-1) + 1],
            "overviews": len(session.overviews),
        }

    @tool()
    def finalize_map(
        self,
        map_id: str,
        name: str | None = None,
        out_dir: str | None = None,
        tile_size: int = 256,
    ) -> dict[str, Any]:
        """Stitch every tile added so far and export the final zoomable map bundle.

        Writes web-map style PNG tiles (tiles/{z}/{x}/{y}.png), a
        self-contained zoomable index.html viewer, and an agent-readable
        map.json manifest carrying source-tile provenance and any overviews
        registered via register_overview. The session is closed afterward -
        call start_map again to build another map.
        """
        session = self._get_session(map_id)
        if not session.tiles:
            raise ValueError("No tiles added yet.")
        result = self._stitch(session)
        map_name = name or session.name
        if out_dir is None:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", map_name).strip("_") or "sample_map"
            out_dir = str(self.output_root / safe_name)

        source_tiles = []
        for rc, entry in sorted(session.tiles.items()):
            h, w = result.tile_shapes[rc]
            x, y = result.positions[rc]
            source_tiles.append(
                {
                    "grid": list(rc),
                    "canvas_xy": [x, y],
                    "size_px": [w, h],
                    "stage_xy_m": list(entry.stage_xy_m) if entry.stage_xy_m else None,
                    "source": entry.source,
                }
            )

        overviews_payload = []
        for ov in session.overviews:
            overviews_payload.append(
                {
                    "image": ov.image,
                    "matrix": ov.registration.matrix,
                    "scale": ov.registration.scale,
                    "rotation_deg": ov.registration.rotation_deg,
                    "offset_px": list(ov.registration.offset_px),
                    "inliers": ov.registration.inliers,
                    "stage_xy_m": list(ov.stage_xy_m) if ov.stage_xy_m else None,
                    "source": ov.source,
                }
            )

        bundle = export_map(
            result.canvas,
            out_dir,
            name=map_name,
            tile_size=tile_size,
            pixel_size_nm=session.pixel_size_nm,
            stage_origin=session.stage_origin_m,
            source_tiles=source_tiles,
            overviews=overviews_payload,
        )
        del self._sessions[map_id]
        return {
            "bundle_path": str(bundle),
            "viewer": str(bundle / "index.html"),
            "manifest": str(bundle / "map.json"),
            "canvas_px": list(result.canvas.shape[:2][::-1]),
            "tiles_stitched": len(session.tiles),
            "edges_measured": sum(e.used for e in result.edges),
            "edges_total": len(result.edges),
            "mean_edge_residual_px": result.mean_residual,
            "overview_count": len(overviews_payload),
        }

    @tool()
    def list_maps(self) -> list[dict[str, Any]]:
        """List active (not yet finalized or discarded) map sessions."""
        return [self.get_map_status(map_id) for map_id in self._sessions]

    @tool()
    def discard_map(self, map_id: str) -> None:
        """Discard a map session without exporting a bundle."""
        self._get_session(map_id)
        del self._sessions[map_id]

    def setup(self) -> None:
        native_tools = [
            self.start_map,
            self.add_tiles,
            self.preview_map,
            self.register_overview,
            self.get_map_status,
            self.finalize_map,
            self.list_maps,
            self.discard_map,
        ]
        for native_tool in native_tools:
            tool_obj = Tool.from_function(native_tool)
            self.mcp.add_tool(tool_obj)
            if self.verbose:
                print(f"Registered tool: {native_tool.__name__}")

    def start(self, transport: Transport | None = None, **kwargs: Any) -> None:
        self.setup()
        self.mcp.run(transport=transport, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Mapping")
    parser.add_argument("--transport", default="streamable-http")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8001)
    parser.add_argument("--output-root", default="./outputs/maps")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server = MappingServer(name=args.name, output_root=args.output_root, verbose=not args.quiet)
    if args.transport == "streamable-http":
        print(f"Starting {args.name} at http://{args.http_host}:{args.http_port}/mcp", flush=True)
        server.start(transport="streamable-http", host=args.http_host, port=args.http_port)
    else:
        server.start(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
