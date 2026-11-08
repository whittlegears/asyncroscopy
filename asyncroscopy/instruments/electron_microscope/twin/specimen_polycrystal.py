"""The ``polycrystal`` specimen: an atomistic slab of FCC grains, each near a low-index zone axis.

The sample is a table of grains; ``view`` builds atoms only for the grains near the requested field, for the
polycrystalline gold twin's pseudo-potential renderer. A Voronoi cut (each atom belongs to its nearest grain centre)
gives the boundaries. Coordinates are angstrom in the slab's own frame; the field is centred on the slab's middle
plus the stage position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .elements import ELEMENTS, atomic_number, bulk_cell
from .specimen import FieldView, Specimen, SpecimenContext, ViewRequest, check_options, register_specimen


@dataclass(frozen=True)
class GrainParameters:
    label: int
    occupied: bool
    center_angstrom: tuple[float, float, float]
    zone_axis: tuple[int, int, int]
    zone_offset_deg: float
    orientation_euler_deg: tuple[float, float, float]
    composition: dict[str, float]


@register_specimen
class Polycrystal(Specimen):
    """FCC grains in a slab, with a fraction of the grain sites left empty."""

    source = "polycrystal"
    render_modes = ("grain_potential",)
    OPTIONS = (
        "element", "volume_size_xy_nm", "volume_thickness_nm", "grain_size_angstrom", "empty_grain_fraction",
        "lattice_constant_angstrom", "zone_axis_max_index", "zone_axis_max_deviation_deg", "seed",
    )

    def __init__(
        self,
        seed: int,
        element: str = "Au",
        volume_size_xy_nm: float = 200.0,
        volume_thickness_nm: float = 20.0,
        grain_size_angstrom: float = 10.0,
        empty_grain_fraction: float = 0.15,
        lattice_constant_angstrom: float | None = None,
        zone_axis_max_index: int = 2,
        zone_axis_max_deviation_deg: float = 2.0,
    ) -> None:
        if element not in ELEMENTS:
            raise ValueError(f"polycrystal element {element!r} unknown; known: {sorted(ELEMENTS)}")
        sample_xy_angstrom = float(volume_size_xy_nm) * 10.0
        sample_z_angstrom = float(volume_thickness_nm) * 10.0
        grain_size = float(grain_size_angstrom)
        empty_fraction = float(empty_grain_fraction)
        max_index = int(zone_axis_max_index)
        max_deviation = float(zone_axis_max_deviation_deg)
        if sample_xy_angstrom <= 0 or sample_z_angstrom <= 0 or grain_size <= 0:
            raise ValueError("Sample dimensions and grain size must be positive")
        if not 0.0 <= empty_fraction < 1.0:
            raise ValueError("empty_grain_fraction must be in [0, 1)")
        if max_index < 1 or max_deviation < 0.0:
            raise ValueError("Zone-axis index must be positive and deviation must be nonnegative")

        self.element = element
        self.volume_size_xy_nm, self.volume_thickness_nm = float(volume_size_xy_nm), float(volume_thickness_nm)
        self.sample_xy_angstrom, self.sample_z_angstrom = sample_xy_angstrom, sample_z_angstrom
        self.grain_size_angstrom = grain_size
        self.lattice_constant_angstrom = float(ELEMENTS[element][3] if lattice_constant_angstrom is None else lattice_constant_angstrom)
        self.bounds = (0.0, sample_xy_angstrom * 1e-10, 0.0, sample_xy_angstrom * 1e-10)
        self.world_bounds_ang = {"x_min": 0.0, "x_max": sample_xy_angstrom, "y_min": 0.0, "y_max": sample_xy_angstrom, "z_min": 0.0, "z_max": sample_z_angstrom}

        rng = np.random.default_rng(int(seed))
        grain_count = max(1, int(sample_xy_angstrom**2 // (4.0 / 3.0 * np.pi * grain_size**2)))
        sample_size = np.array([sample_xy_angstrom, sample_xy_angstrom, sample_z_angstrom])
        self.grain_centers = rng.random((grain_count, 3)) * sample_size
        empty_count = int(grain_count * empty_fraction)
        self.empty_grain_indices = {int(index) for index in rng.choice(grain_count, empty_count, replace=False)}
        zone_axes = [
            (n, m, ell)
            for n in range(max_index + 1)
            for m in range(n, max_index + 1)
            for ell in range(m, max_index + 1)
            if (n, m, ell) != (0, 0, 0) and np.gcd.reduce((n, m, ell)) == 1
        ]
        self.grain_zone_axes = [zone_axes[int(rng.integers(len(zone_axes)))] for _index in range(grain_count)]
        self.grain_zone_offsets: list[float] = []
        self.grain_rotations: list[np.ndarray] = []
        self.grain_angles: list[tuple[float, float, float]] = []
        for zone_axis in self.grain_zone_axes:
            zone_direction = np.asarray(zone_axis, dtype=float)
            zone_direction /= np.linalg.norm(zone_direction)
            while True:
                tilt_xy_deg = rng.normal(0.0, max_deviation / 3.0, 2) if max_deviation > 0.0 else np.zeros(2)
                if np.linalg.norm(tilt_xy_deg) <= max_deviation:
                    break
            alignment, _error = Rotation.align_vectors([[0.0, 0.0, 1.0]], [zone_direction])
            tilt = Rotation.from_rotvec(np.radians([tilt_xy_deg[0], tilt_xy_deg[1], 0.0]))
            in_plane = Rotation.from_rotvec([0.0, 0.0, rng.uniform(0.0, 2.0 * np.pi)])
            rotation = in_plane * tilt * alignment
            self.grain_zone_offsets.append(float(np.linalg.norm(tilt_xy_deg)))
            self.grain_rotations.append(rotation.as_matrix())
            self.grain_angles.append(tuple(float(value) for value in rotation.as_euler("ZYX", degrees=True)))
        self.grain_tree = cKDTree(self.grain_centers)
        self.region_parameters = [
            GrainParameters(
                index + 1,
                index not in self.empty_grain_indices,
                tuple(float(value) for value in self.grain_centers[index]),
                self.grain_zone_axes[index],
                self.grain_zone_offsets[index],
                self.grain_angles[index],
                {element: 1.0} if index not in self.empty_grain_indices else {},
            )
            for index in range(grain_count)
        ]
        self._metadata = {
            "volume_size_xy_nm": self.volume_size_xy_nm,
            "volume_thickness_nm": self.volume_thickness_nm,
            "grain_size_angstrom": grain_size,
            "lattice_constant_angstrom": self.lattice_constant_angstrom,
            "grain_count": grain_count,
            "occupied_grain_count": grain_count - empty_count,
            "empty_grain_count": empty_count,
            "zone_axis_families": [list(axis) for axis in zone_axes],
            "zone_axis_max_deviation_deg": max_deviation,
        }

    @classmethod
    def from_options(cls, options: dict, context: SpecimenContext) -> "Polycrystal":
        check_options(cls.source, options, cls.OPTIONS)
        kwargs = {k: options[k] for k in cls.OPTIONS if k in options and k != "seed"}
        return cls(context.rng_for(options).seed, **kwargs)

    @property
    def element_symbols(self) -> tuple[str, ...]:
        return (self.element,)

    def regions(self) -> list[GrainParameters]:
        return list(self.region_parameters)

    def view(self, request: ViewRequest) -> FieldView:
        """atoms: every atom in the field (slab coordinates, angstrom); info: {"window_angstrom": (x_min, y_min), "atom_count": n}."""
        field = request.fov_angstrom
        center_x = self.sample_xy_angstrom / 2.0 + request.stage[0] * 1e10
        center_y = self.sample_xy_angstrom / 2.0 + request.stage[1] * 1e10
        x_min, x_max = center_x - field / 2.0, center_x + field / 2.0
        y_min, y_max = center_y - field / 2.0, center_y + field / 2.0

        grid_x = np.linspace(x_min, x_max, 12)
        grid_y = np.linspace(y_min, y_max, 12)
        grid_z = np.linspace(0.0, self.sample_z_angstrom, 8)
        query_points = np.stack(np.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), axis=-1).reshape(-1, 3)
        neighbor_count = min(8, len(self.grain_centers))
        neighbor_distances, neighbor_indices = self.grain_tree.query(query_points, k=neighbor_count)
        candidate_indices = np.unique(np.asarray(neighbor_indices).reshape(-1))
        local_radius = max(float(np.max(neighbor_distances)) * 2.0, self.grain_size_angstrom * 2.0)
        cell = bulk_cell(self.element, self.lattice_constant_angstrom, cubic=True)
        repeat = int(np.ceil(2.0 * local_radius / self.lattice_constant_angstrom)) + 2
        supercell = cell.repeat((repeat, repeat, repeat))
        base_positions = supercell.get_positions()
        base_positions -= base_positions.mean(axis=0)
        all_positions = []
        for grain_index in candidate_indices:
            if int(grain_index) in self.empty_grain_indices:
                continue
            positions = base_positions @ self.grain_rotations[int(grain_index)].T + self.grain_centers[int(grain_index)]
            inside = (positions[:, 0] >= x_min) & (positions[:, 0] < x_max) & (positions[:, 1] >= y_min) & (positions[:, 1] < y_max) & (positions[:, 2] >= 0.0) & (positions[:, 2] < self.sample_z_angstrom)
            positions = positions[inside]
            if len(positions) == 0:
                continue
            nearest_grains = self.grain_tree.query(positions)[1]
            grain_positions = positions[nearest_grains == grain_index]
            if len(grain_positions):
                all_positions.append(grain_positions)
        atom_positions = np.vstack(all_positions) if all_positions else np.empty((0, 3))
        atoms = Atoms(numbers=np.full(len(atom_positions), atomic_number(self.element)), positions=atom_positions)
        return FieldView(atoms=atoms, info={"window_angstrom": (x_min, y_min), "atom_count": len(atom_positions)})

    def describe(self) -> dict:
        return {**super().describe(), **self._metadata, "element": self.element}
