"""Tile grids for maps: where the tiles of a box go, in what order, and how often a point is covered.

Pure numbers in metres; no instrument. The Mapper (any instrument profile) and the
plan checks use these; see docs/designs/concepts_brief.md, decision D1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EDGE_TOLERANCE_M = 1e-12


def grid_axes(box: list[float], fov: float, overlap: float, limit: int | None = None, fov_y: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tile-centre x and y coordinates covering ``box`` with every tile inside it; more than ``limit`` tiles is refused.

    ``fov`` is the tile's width; ``fov_y`` its height when the tiles are not square (default ``fov``).
    """
    x_min, x_max, y_min, y_max = box

    def axis(lo: float, hi: float, side: float) -> np.ndarray:
        step = side * (1.0 - overlap)
        span = hi - lo
        if span < side - EDGE_TOLERANCE_M:
            raise ValueError(f"box side {span:.3e} m is smaller than the field of view {side:.3e} m")
        n = max(1, int(math.ceil((span - side) / step - 1e-9)) + 1)
        if limit is not None and n > limit:
            raise ValueError(f"plan needs more than {limit} tiles; this Mapper allows at most {limit}")
        if n == 1:
            return np.array([(lo + hi) / 2.0])
        # Spread the tiles evenly so the last one ends exactly at the box edge.
        return np.linspace(lo + side / 2.0, hi - side / 2.0, n)

    xs, ys = axis(x_min, x_max, fov), axis(y_min, y_max, fov if fov_y is None else fov_y)
    if limit is not None and len(xs) * len(ys) > limit:
        raise ValueError(f"plan needs {len(xs) * len(ys)} tiles; this Mapper allows at most {limit}")
    return xs, ys


def compact_grid(xs: Sequence[float] | np.ndarray, ys: Sequence[float] | np.ndarray) -> dict:
    """A grid as origin, step and count per axis (the tiles are evenly spaced)."""
    return {
        "rows": len(ys), "cols": len(xs),
        "x0": float(xs[0]), "dx": float(xs[1] - xs[0]) if len(xs) > 1 else 0.0,
        "y0": float(ys[0]), "dy": float(ys[1] - ys[0]) if len(ys) > 1 else 0.0,
    }


def tile_at(grid: dict, order: str, index: int) -> dict:
    """Tile ``index`` of the grid, in acquisition order (serpentine reverses every other row)."""
    cols = grid["cols"]
    row, k = divmod(int(index), cols)
    col = cols - 1 - k if order == "serpentine" and row % 2 == 1 else k
    return {"index": int(index), "row": row, "col": col, "x": grid["x0"] + col * grid["dx"], "y": grid["y0"] + row * grid["dy"]}


def index_at(grid: dict, order: str, row: int, col: int) -> int:
    """The acquisition index of the tile at (row, col): the inverse of ``tile_at``."""
    cols = grid["cols"]
    k = cols - 1 - col if order == "serpentine" and row % 2 == 1 else col
    return int(row) * cols + int(k)


def tile_centres(box: list[float], fov: float, overlap: float, order: str, limit: int | None = None) -> list[dict]:
    """Every tile of a (small) grid as a list; plans use ``compact_grid`` and ``tile_at`` instead."""
    grid = compact_grid(*grid_axes(box, fov, overlap, limit))
    return [tile_at(grid, order, k) for k in range(grid["rows"] * grid["cols"])]


def max_exposures(grid: dict, fov: float, fov_y: float | None = None) -> int:
    """Most tiles that cover any one point, from the actual tile spacing.

    The grid spreads tiles evenly, so the real overlap can exceed the
    requested one; a dose (or any per-area) check must use the real spacing.
    """

    def per_axis(n: int, step: float, side: float) -> int:
        return 1 if n < 2 or step <= 0 else min(n, int(math.ceil(side / step - 1e-9)))

    return per_axis(grid["cols"], grid["dx"], fov) * per_axis(grid["rows"], grid["dy"], fov if fov_y is None else fov_y)


def inside_box(box: list[float], x: float, y: float, half_x: float, half_y: float | None = None, tol: float = EDGE_TOLERANCE_M) -> bool:
    """Whether an image of half-width ``half_x`` (half-height ``half_y``) centred on (x, y) lies inside ``box``."""
    x_min, x_max, y_min, y_max = box
    hy = half_x if half_y is None else half_y
    return x_min + half_x - tol <= x <= x_max - half_x + tol and y_min + hy - tol <= y <= y_max - hy + tol
