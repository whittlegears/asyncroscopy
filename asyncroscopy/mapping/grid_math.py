"""
Pure grid/spec helpers for sample mapping - no image processing dependencies.

Kept separate from :mod:`grid_stitcher` / :mod:`map_export` / :mod:`montage`
(which pull in opencv and pillow) so that instrument-side code such as
:meth:`ElectronMicroscope.acquire_map_grid` can plan a stage grid without
dragging heavyweight CV dependencies into the Tango device process. The
actual stitching lives in the standalone mapping MCP server
(``asyncroscopy.mcp.mapping_mcp_server``), not on the instrument bridge.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def plan_grid(
    rows: int,
    cols: int,
    step_x_m: float,
    step_y_m: float,
    snake: bool = True,
    flip_x: bool = False,
    flip_y: bool = False,
) -> List[Dict[str, Any]]:
    """Return the acquisition order for a rows x cols stage grid.

    Each entry is ``{"row": r, "col": c, "offset_m": [dx, dy]}`` relative to
    the starting stage position.  With ``snake=True`` (default) alternate rows
    are traversed right-to-left to minimise stage travel.  ``flip_x`` /
    ``flip_y`` invert the stage step direction for instruments whose stage
    axes are mirrored with respect to the image axes.
    """
    sx = -step_x_m if flip_x else step_x_m
    sy = -step_y_m if flip_y else step_y_m
    order: List[Dict[str, Any]] = []
    for r in range(rows):
        cs = range(cols) if (not snake or r % 2 == 0) else range(cols - 1, -1, -1)
        for c in cs:
            order.append({"row": r, "col": c, "offset_m": [c * sx, r * sy]})
    return order


def parse_map_grid_result(payload: str | Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise the JSON returned by ``acquire_map_grid``."""
    spec = json.loads(payload) if isinstance(payload, str) else dict(payload)
    if "tiles" not in spec or not spec["tiles"]:
        raise ValueError("Grid result contains no tiles.")
    for t in spec["tiles"]:
        if "row" not in t or "col" not in t or "key" not in t:
            raise ValueError(f"Malformed tile entry: {t}")
    return spec
