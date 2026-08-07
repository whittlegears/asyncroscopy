"""
Sample-map montage helpers.

Pure functions that turn a set of acquired grid tiles into a zoomable
"google-maps" style map bundle (see :mod:`asyncroscopy.mapping.map_export`).
Nothing in this module talks to Tango; acquisition is driven by the
``acquire_map_grid`` command on :class:`ElectronMicroscope` and the bundle is
built by the ``build_sample_map`` MCP tool, both of which delegate here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .grid_stitcher import GridStitcher
from .map_export import export_map

logger = logging.getLogger(__name__)


def load_acquisition_image(path: str | Path) -> np.ndarray:
    """Load a 2-D image array from an acquisition file (.h5, .tiff or plain image).

    For HDF5 files the ``image/HAADF`` dataset is preferred, then any other
    dataset under ``image/``, then the first 2-D dataset in the file.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(p, "r") as h5:
            names: List[str] = []

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                    names.append(name)

            h5.visititems(visit)
            if not names:
                raise ValueError(f"No 2-D dataset found in {p}")
            pick = None
            for candidate in names:
                if candidate.lower() == "image/haadf":
                    pick = candidate
                    break
            if pick is None:
                image_names = [n for n in names if n.lower().startswith("image/")]
                pick = image_names[0] if image_names else names[0]
            return np.asarray(h5[pick][()])
    if suffix in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(p)
        if arr.ndim > 2:
            arr = arr[0]
        return np.asarray(arr)
    import cv2

    arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Could not read image {p}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def build_sample_map(
    tiles: List[Dict[str, Any]],
    resolve_key: Callable[[str], Path],
    out_dir: str | Path,
    name: str = "sample_map",
    overlap: float = 0.15,
    pixel_size_nm: Optional[float] = None,
    stage_origin_m: Optional[List[float]] = None,
    tile_size: int = 256,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stitch acquired grid tiles and export a zoomable map bundle.

    Parameters
    ----------
    tiles:
        One entry per acquired tile:
        ``{"row": r, "col": c, "key": <DATA key>,
        "stage_xy_m": [x, y] (optional), "eds_key": <DATA key> (optional)}``.
    resolve_key:
        Callable mapping a DATA key to a local file path.
    out_dir:
        Directory for the map bundle.
    overlap:
        Nominal fractional overlap between neighbouring tiles.
    pixel_size_nm:
        Physical pixel size of a tile at native resolution, if known.

    Returns
    -------
    dict
        Summary with bundle paths and stitch quality diagnostics.
    """
    grid: Dict[Tuple[int, int], np.ndarray] = {}
    by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for entry in tiles:
        rc = (int(entry["row"]), int(entry["col"]))
        path = resolve_key(str(entry["key"]))
        grid[rc] = load_acquisition_image(path)
        by_key[rc] = entry

    if not grid:
        raise ValueError("No tiles supplied.")

    stitcher = GridStitcher(overlap=overlap)
    result = stitcher.stitch(grid)

    source_tiles = []
    annotations = []
    for rc, entry in sorted(by_key.items()):
        h, w = result.tile_shapes[rc]
        x, y = result.positions[rc]
        source_tiles.append(
            {
                "grid": list(rc),
                "canvas_xy": [x, y],
                "size_px": [w, h],
                "stage_xy_m": entry.get("stage_xy_m"),
                "data_key": entry["key"],
                "signals": {"eds": entry["eds_key"]} if entry.get("eds_key") else {},
            }
        )
        if entry.get("eds_key"):
            annotations.append(
                {
                    "type": "eds_point",
                    "label": f"EDS ({rc[0]}, {rc[1]})",
                    "canvas_xy": [x + w / 2.0, y + h / 2.0],
                    "data": {"data_key": entry["eds_key"]},
                }
            )

    bundle = export_map(
        result.canvas,
        out_dir,
        name=name,
        tile_size=tile_size,
        pixel_size_nm=pixel_size_nm,
        stage_origin=stage_origin_m,
        source_tiles=source_tiles,
        annotations=annotations,
        extra_metadata=extra_metadata,
    )

    return {
        "bundle_path": str(bundle),
        "viewer": str(bundle / "index.html"),
        "manifest": str(bundle / "map.json"),
        "canvas_px": list(result.canvas.shape[:2][::-1]),
        "tiles_stitched": len(grid),
        "edges_measured": sum(e.used for e in result.edges),
        "edges_total": len(result.edges),
        "mean_edge_residual_px": result.mean_residual,
    }
