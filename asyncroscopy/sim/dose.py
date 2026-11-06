"""Where the beam has been, and what that did.

Dose is recorded in square cells in sample coordinates, kept only where the
beam has been. Each frame records its dose in cells sized to it (about an
eighth of its field and at least four of its pixels, never smaller than
``min_cell_m``), so a 50 nm frame on
a 20 um sample keeps its dose on the 50 nm it hit instead of spreading it
over a cell sized for the whole sample. The dose at a point is the sum over
every cell size of the cell holding that point. Memory grows with the
number of frames, not with the sample's area.

Dose drives a carbon contamination layer that builds up and saturates, and
dims particles past a critical dose, so re-imaging the same area has a
visible cost.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

ANGSTROM2_PER_M2 = 1e20
CELLS_PER_FIELD = 8  # a frame's dose is kept in cells about an eighth of its field...
MIN_PIXELS_PER_CELL = 4  # ...but at least this many pixels wide, so a small frame's cells are not mostly empty
KEY_SPAN = 2**31  # cell (row, col) -> one int64 key; cell indices stay well inside +-KEY_HALF
KEY_HALF = 2**30
COARSEST_LEVEL = -30  # cells up to 2^30 times cell_m: no frame is that large


@dataclass
class DoseParams:
    enabled: bool = False
    cell_m: float = 20e-9  # the reference cell; finer frames use cells half as large, coarser ones twice, again and again
    min_cell_m: float = 0.5e-9
    spot_spread_m: float = 1e-8  # a parked beam's dose is counted over a square this wide (match the Mapper's eds_dose_spread_m)
    contamination_angstrom_max: float = 0.0
    contamination_dose_e_per_a2: float = 1e4
    damage_critical_dose_e_per_a2: float = 0.0


class DoseMap:
    def __init__(self, params: DoseParams, bounds: tuple[float, float, float, float], margin_m: float = 0.0) -> None:
        if not (params.cell_m > 0 and 0 < params.min_cell_m <= params.cell_m):
            raise ValueError("dose cells need 0 < min_cell_m <= cell_m")
        self.params = params
        x_min, _x_max, y_min, _y_max = bounds
        self.x0 = x_min - margin_m
        self.y0 = y_min - margin_m
        self.finest = int(math.floor(math.log2(params.cell_m / params.min_cell_m)))
        self.levels: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)  # electrons per square angstrom

    def _cell_m(self, level: int) -> float:
        return self.params.cell_m / 2**level

    def _level_for(self, size_m: float) -> int:
        """The level whose cells are the largest no bigger than ``size_m`` / CELLS_PER_FIELD (at most ``finest``;
        levels below 0 are cells larger than ``cell_m``, for frames whose pixels are larger than it)."""
        if not (np.isfinite(size_m) and size_m > 0):
            return 0
        wanted = size_m / CELLS_PER_FIELD
        return int(np.clip(math.ceil(math.log2(self.params.cell_m / wanted)), COARSEST_LEVEL, self.finest))

    def _cells(self, level: int, x, y) -> tuple[np.ndarray, np.ndarray]:
        cell = self._cell_m(level)
        return (np.floor((np.asarray(y, dtype=np.float64) - self.y0) / cell).astype(np.int64),
                np.floor((np.asarray(x, dtype=np.float64) - self.x0) / cell).astype(np.int64))

    def add_frame(self, xs: np.ndarray, ys: np.ndarray, electrons_per_pixel: float, pixel_m: float) -> None:
        if not self.params.enabled:
            return
        if not (np.isfinite(electrons_per_pixel) and np.isfinite(pixel_m) and pixel_m > 0):
            return
        xs, ys = np.asarray(xs, dtype=np.float64).ravel(), np.asarray(ys, dtype=np.float64).ravel()
        finite = np.isfinite(xs) & np.isfinite(ys)  # one NaN would otherwise poison the whole map
        xs, ys = xs[finite], ys[finite]
        if xs.size == 0:
            return
        level = self._level_for(max(float(np.ptp(xs)), float(np.ptp(ys)), CELLS_PER_FIELD * MIN_PIXELS_PER_CELL * pixel_m))
        cell = self._cell_m(level)
        per_a2 = electrons_per_pixel / (pixel_m * pixel_m * ANGSTROM2_PER_M2)
        rows, cols = self._cells(level, xs, ys)
        # One integer per cell (a 1-D unique is far faster than a unique over (row, col) pairs).
        keys, hits = np.unique((rows + KEY_HALF) * KEY_SPAN + (cols + KEY_HALF), return_counts=True)
        # Each cell gets the frame's dose times its pixels over the pixels a full cell holds: every electron is
        # counted once (a cell the grid lines up badly with may hold a few more pixel centres than that).
        pixels_per_cell = (cell / pixel_m) ** 2
        cells = self.levels[level]
        for key, n in zip(keys.tolist(), hits.tolist()):
            r, c = key // KEY_SPAN - KEY_HALF, key % KEY_SPAN - KEY_HALF
            cells[(r, c)] = cells.get((r, c), 0.0) + per_a2 * n / pixels_per_cell

    def add_point(self, x: float, y: float, electrons: float, probe_area_m2: float) -> None:
        if not self.params.enabled:
            return
        size = max(math.sqrt(max(probe_area_m2, 0.0)), self.params.spot_spread_m)
        level = int(np.clip(math.floor(math.log2(self.params.cell_m / size)), COARSEST_LEVEL, self.finest))  # the finest cell at least that wide
        cell = self._cell_m(level)
        r, c = (int(v) for v in self._cells(level, x, y))
        cells = self.levels[level]
        cells[(r, c)] = cells.get((r, c), 0.0) + electrons / (cell * cell * ANGSTROM2_PER_M2)

    def dose_at(self, x: float | list[float], y: float | list[float]) -> np.ndarray:
        xs, ys = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        total = np.zeros(xs.shape, dtype=np.float64)
        for level, cells in self.levels.items():
            if not cells:
                continue
            rows, cols = self._cells(level, xs, ys)
            total += np.array([cells.get((r, c), 0.0) for r, c in zip(rows.ravel().tolist(), cols.ravel().tolist())]).reshape(xs.shape)
        return total

    def contamination_angstrom(self, x: float, y: float) -> float:
        p = self.params
        if not p.enabled or p.contamination_angstrom_max <= 0:
            return 0.0
        d = float(self.dose_at(x, y))
        return p.contamination_angstrom_max * (1.0 - np.exp(-d / p.contamination_dose_e_per_a2))

    def damage_factors(self, centers: np.ndarray, indices: np.ndarray) -> dict[int, float]:
        p = self.params
        if not p.enabled or p.damage_critical_dose_e_per_a2 <= 0:
            return {}
        out = {}
        for i in indices:
            d = float(self.dose_at(centers[i, 0], centers[i, 1]))
            excess = max(0.0, d - p.damage_critical_dose_e_per_a2)
            if excess > 0:
                out[int(i)] = float(np.exp(-excess / p.damage_critical_dose_e_per_a2))
        return out

    def total_electrons(self) -> float:
        """Every electron recorded, summed over cells (dose times cell area)."""
        return float(sum(d * self._cell_m(level) ** 2 * ANGSTROM2_PER_M2 for level, cells in self.levels.items() for d in cells.values()))

    def summary(self) -> dict:
        centres = [
            (self.x0 + (c + 0.5) * self._cell_m(level), self.y0 + (r + 0.5) * self._cell_m(level))
            for level, cells in self.levels.items() for (r, c) in cells
        ]
        peak = float(self.dose_at([x for x, _ in centres], [y for _, y in centres]).max()) if centres else 0.0
        return {
            "cell_m": self.params.cell_m,
            "finest_cell_m": self._cell_m(self.finest),
            "max_dose_e_per_a2": peak,
            "exposed_cells": sum(len(cells) for cells in self.levels.values()),
        }
