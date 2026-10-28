"""Where the beam has been, and what that did.

Dose is recorded on a coarse grid in sample coordinates. It drives a carbon
contamination layer that builds up and saturates, and dims particles past a
critical dose, so re-imaging the same area has a visible cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANGSTROM2_PER_M2 = 1e20


@dataclass
class DoseParams:
    enabled: bool = False
    cell_m: float = 20e-9
    contamination_angstrom_max: float = 0.0
    contamination_dose_e_per_a2: float = 1e4
    damage_critical_dose_e_per_a2: float = 0.0


class DoseMap:
    def __init__(self, params: DoseParams, bounds: tuple[float, float, float, float], margin_m: float = 0.0):
        self.params = params
        x_min, x_max, y_min, y_max = bounds
        self.x0 = x_min - margin_m
        self.y0 = y_min - margin_m
        nx = int(np.ceil((x_max - x_min + 2 * margin_m) / params.cell_m)) + 1
        ny = int(np.ceil((y_max - y_min + 2 * margin_m) / params.cell_m)) + 1
        self.dose = np.zeros((ny, nx), dtype=np.float64)  # electrons per square angstrom

    def _cells(self, x, y):
        c = np.clip(((np.asarray(x) - self.x0) / self.params.cell_m).astype(int), 0, self.dose.shape[1] - 1)
        r = np.clip(((np.asarray(y) - self.y0) / self.params.cell_m).astype(int), 0, self.dose.shape[0] - 1)
        return r, c

    def add_frame(self, xs: np.ndarray, ys: np.ndarray, electrons_per_pixel: float, pixel_m: float) -> None:
        if not self.params.enabled:
            return
        if not (np.isfinite(electrons_per_pixel) and np.isfinite(pixel_m) and pixel_m > 0):
            return
        xs, ys = xs.ravel(), ys.ravel()
        finite = np.isfinite(xs) & np.isfinite(ys)  # one NaN would otherwise poison the whole map
        xs, ys = xs[finite], ys[finite]
        per_a2 = electrons_per_pixel / (pixel_m * pixel_m * ANGSTROM2_PER_M2)
        r, c = self._cells(xs, ys)
        hits = np.zeros_like(self.dose)
        np.add.at(hits, (r, c), 1.0)
        # Spread each frame's dose over the cells it touched, weighted by pixel count.
        pixels_per_cell = (self.params.cell_m / pixel_m) ** 2
        self.dose += per_a2 * np.minimum(hits / pixels_per_cell, 1.0)

    def add_point(self, x: float, y: float, electrons: float, probe_area_m2: float) -> None:
        if not self.params.enabled:
            return
        r, c = self._cells(x, y)
        cell_area = self.params.cell_m ** 2
        self.dose[r, c] += electrons / (cell_area * ANGSTROM2_PER_M2)

    def dose_at(self, x, y) -> np.ndarray:
        r, c = self._cells(x, y)
        return self.dose[r, c]

    def contamination_angstrom(self, x: float, y: float) -> float:
        p = self.params
        if not p.enabled or p.contamination_angstrom_max <= 0:
            return 0.0
        d = float(self.dose_at(x, y))
        return p.contamination_angstrom_max * (1.0 - np.exp(-d / p.contamination_dose_e_per_a2))

    def damage_factors(self, centers: np.ndarray, indices) -> dict[int, float]:
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

    def summary(self) -> dict:
        return {
            "cell_m": self.params.cell_m,
            "max_dose_e_per_a2": float(self.dose.max()) if self.dose.size else 0.0,
            "exposed_cells": int((self.dose > 0).sum()),
        }
