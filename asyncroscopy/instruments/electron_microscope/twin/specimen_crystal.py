"""The ``crystal`` specimen: a polycrystalline film built from an ASE unit cell.

A film of one element (``element``, default Au; its structure and lattice
constant from ``ELEMENTS``) covers the sample area. It is cut into
``grain_count`` Voronoi grains, each with its own seeded in-plane rotation,
thickness (``thickness_m`` +/- ``thickness_variation``) and small tilt off
the [001] zone axis (``mosaic_spread_mrad``). Nothing here runs a multislice
simulation, so it stays cheap at any field of view:

* survey: the film's mass thickness, brighter where a grain is close to the
  zone axis (``channeling`` sets how much), so grains show up in an overview;
* atomic: the projected atomic columns of the cubic cell along [001], each
  weighted by the atoms it holds through the grain's thickness, rotated per
  grain and cut at the grain boundaries.

Ground truth is the grain table (seed point, rotation, tilt, thickness) plus
the lattice: what a grain-mapping or lattice-measuring workflow is scored on.
There are no particles.
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk
from scipy.spatial import cKDTree

from .elements import ELEMENTS
from .sample_model import thermal_smear
from .specimen import Specimen, SpecimenContext, check_options, register_specimen

ANGSTROM = 1e-10
OPTIONS = {
    "element", "lattice_constant_m", "thickness_m", "thickness_variation", "grain_count", "mosaic_spread_mrad",
    "channeling", "size_xy_m", "support_thickness_m", "height_gradient", "seed",
}


def projected_columns(element: str, lattice_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Columns of the cubic cell seen along [001]: (fractional xy positions, atoms per cell height)."""
    _z, _rho, structure, lattice_a = ELEMENTS[element]
    cell = bulk(element, structure, a=lattice_m / ANGSTROM, cubic=True)
    xy = np.round(cell.get_scaled_positions()[:, :2] % 1.0, 6) % 1.0
    positions, counts = np.unique(xy, axis=0, return_counts=True)
    return positions, counts.astype(np.float64)


@register_specimen
class CrystalSpecimen(Specimen):
    source = "crystal"

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        element: str = "Au",
        lattice_constant_m: float | None = None,
        thickness_m: float = 5e-9,
        thickness_variation: float = 0.15,
        grain_count: int = 12,
        mosaic_spread_mrad: float = 20.0,
        channeling: float = 0.3,
        size_xy_m: float = 4e-7,
        support_thickness_m: float = 0.0,
        height_gradient: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if element not in ELEMENTS:
            raise ValueError(f"crystal element {element!r} unknown; known: {sorted(ELEMENTS)}")
        if size_xy_m <= 0 or thickness_m <= 0 or grain_count < 1:
            raise ValueError("crystal specimen needs size_xy_m > 0, thickness_m > 0 and grain_count >= 1")
        self.element = element
        self.z, self.rho = float(ELEMENTS[element][0]), float(ELEMENTS[element][1])
        self.lattice_m = float(lattice_constant_m or ELEMENTS[element][3] * ANGSTROM)
        self.columns, self.column_atoms = projected_columns(element, self.lattice_m)
        half = size_xy_m / 2.0
        self.bounds = (-half, half, -half, half)
        self.support_thickness_m = float(support_thickness_m)
        self.height_gradient = tuple(float(v) for v in height_gradient)
        n = int(grain_count)
        self.seeds = rng.uniform(-half, half, size=(n, 2))
        self.rotation_rad = rng.uniform(0.0, np.pi / 2.0, size=n)  # [001] cubic: 90 degrees is a full turn
        self.thickness = thickness_m * (1.0 + thickness_variation * rng.uniform(-1.0, 1.0, size=n))
        self.tilt_mrad = np.abs(rng.normal(0.0, mosaic_spread_mrad, size=n))
        # Close to the zone axis the beam channels down the columns and scatters more to HAADF.
        self.brightness = 1.0 + channeling * np.exp(-((self.tilt_mrad / 10.0) ** 2))
        self._tree = cKDTree(self.seeds)
        self.options = {"element": element, "lattice_constant_m": self.lattice_m, "thickness_m": thickness_m, "grain_count": n}

    @classmethod
    def from_options(cls, options: dict, context: SpecimenContext) -> "CrystalSpecimen":
        check_options(cls.source, options, OPTIONS)
        sample = context.sample
        kwargs = {k: v for k, v in options.items() if k != "seed"}
        kwargs.setdefault("size_xy_m", sample.size_xy_m if sample else 4e-7)
        kwargs.setdefault("support_thickness_m", sample.support_thickness_m if sample else 0.0)
        kwargs.setdefault("height_gradient", sample.height_gradient if sample else (0.0, 0.0))
        return cls(context.rng_for(options).for_event("crystal"), **kwargs)

    # ------------------------------------------------------------------
    def grain_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Grain index at sample points; -1 off the film."""
        xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        _d, grain = self._tree.query(np.stack([xs.ravel(), ys.ravel()], axis=-1))
        x_min, x_max, y_min, y_max = self.bounds
        on = (xs.ravel() >= x_min) & (xs.ravel() <= x_max) & (ys.ravel() >= y_min) & (ys.ravel() <= y_max)
        return np.where(on, grain, -1).reshape(xs.shape)

    def projected_weight(self, x_min: float, y_min: float, n: int, pixel_m: float, level: str, z_exponent: float, damage: dict[int, float]) -> np.ndarray:
        coords = x_min + (np.arange(n) + 0.5) * pixel_m
        ycoords = y_min + (np.arange(n) + 0.5) * pixel_m
        gx, gy = np.meshgrid(coords, ycoords)
        grains = self.grain_at(gx, gy)
        left = np.array([damage.get(i, 1.0) for i in range(len(self.seeds))])
        zn = self.z ** z_exponent
        if level == "survey":
            per_grain = zn * self.rho * self.thickness / ANGSTROM * self.brightness * left
            return np.where(grains >= 0, per_grain[np.clip(grains, 0, None)], 0.0)
        weight = np.zeros((n, n), dtype=np.float64)
        pixel_area_a2 = (pixel_m / ANGSTROM) ** 2
        a = self.lattice_m
        corners = np.array([[x_min, y_min], [x_min + n * pixel_m, y_min], [x_min, y_min + n * pixel_m], [x_min + n * pixel_m, y_min + n * pixel_m]])
        for g in np.unique(grains[grains >= 0]):
            c, s = np.cos(self.rotation_rad[g]), np.sin(self.rotation_rad[g])
            to_world = np.array([[c, -s], [s, c]])
            local = (corners - self.seeds[g]) @ to_world  # world -> grain frame (rotation by -angle)
            i0, j0 = np.floor(local.min(axis=0) / a).astype(int) - 1
            i1, j1 = np.ceil(local.max(axis=0) / a).astype(int) + 1
            ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
            cells = np.stack([ii.ravel(), jj.ravel()], axis=-1).astype(np.float64)
            frac = (cells[:, None, :] + self.columns[None, :, :]).reshape(-1, 2)
            atoms = np.tile(self.column_atoms, len(cells)) * (self.thickness[g] / a) * left[g] * self.brightness[g]
            world = frac * a @ to_world.T + self.seeds[g]
            cols = np.floor((world[:, 0] - x_min) / pixel_m).astype(np.int64)
            rows = np.floor((world[:, 1] - y_min) / pixel_m).astype(np.int64)
            ok = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
            ok[ok] = grains[rows[ok], cols[ok]] == g  # cut at the grain boundary
            np.add.at(weight, (rows[ok], cols[ok]), atoms[ok] * zn / pixel_area_a2)
        return thermal_smear(weight, pixel_m)

    def areal_density(self, x: float, y: float) -> dict[str, float]:
        g = int(self.grain_at(np.array([x]), np.array([y]))[0])
        return {self.element: float(self.rho * self.thickness[g] / ANGSTROM)} if g >= 0 else {}

    @property
    def feature_centers(self) -> np.ndarray:
        return self.seeds

    def in_box(self, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        """Grains whose region may overlap the box: those nearest to any corner or the centre, and those seeded in it."""
        probe = np.array([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max], [(x_min + x_max) / 2, (y_min + y_max) / 2]])
        near = set(int(g) for g in self._tree.query(probe)[1])
        inside = (self.seeds[:, 0] >= x_min) & (self.seeds[:, 0] <= x_max) & (self.seeds[:, 1] >= y_min) & (self.seeds[:, 1] <= y_max)
        return np.array(sorted(near | set(np.nonzero(inside)[0].tolist())), dtype=np.int64)

    @property
    def element_symbols(self) -> tuple[str, ...]:
        return (self.element,)

    def ground_truth(self) -> dict:
        return {
            "known": True,
            "particles": [],
            "lattice": {"element": self.element, "structure": ELEMENTS[self.element][2], "lattice_constant_m": self.lattice_m, "zone_axis": [0, 0, 1]},
            "grains": [
                {
                    "id": i,
                    "seed_m": [float(v) for v in self.seeds[i]],
                    "rotation_deg": float(np.rad2deg(self.rotation_rad[i])),
                    "tilt_mrad": float(self.tilt_mrad[i]),
                    "thickness_m": float(self.thickness[i]),
                }
                for i in range(len(self.seeds))
            ],
        }

    def describe(self) -> dict:
        return {**super().describe(), **self.options}
