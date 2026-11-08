"""The ``nanoparticle_map`` specimen: a fixed 2-D field of strained Au nanoparticles for the diffraction twin.

Round particles have a compressed rim and oval ones an expanded rim; a thin amorphous shell has rattled atoms. The
map is fixed in sample coordinates, and the stage only selects where to look: ``view`` samples it onto an image grid
and ``atoms_at`` builds the local Au cell under a parked beam (or None over vacuum).
"""

from __future__ import annotations

import numpy as np

from .elements import ELEMENTS, atomic_number, bulk_cell
from .specimen import FieldView, Specimen, SpecimenContext, ViewRequest, check_options, register_specimen

FOV_NM = 500.0
FOV_M = FOV_NM * 1e-9
DEFAULT_MAP_SIZE = 1024
TARGET_COVERAGE = 0.22
ROUND_CENTER_LATTICE_MEAN_ANGSTROM = 4.078
OVAL_CENTER_LATTICE_MEAN_ANGSTROM = 4.678
CENTER_LATTICE_STD_ANGSTROM = 0.015
EDGE_LATTICE_CHANGE_FRACTION = 0.25
AMORPHOUS_EDGE_WIDTH_PX = 5.0


def local_lattice_parameter(center_value: float, edge_change: float, normalized_radius):
    radial_fraction = np.clip(normalized_radius, 0.0, 1.0)
    smooth_radial_fraction = radial_fraction**2 * (3.0 - 2.0 * radial_fraction)
    return center_value * (1.0 + edge_change * smooth_radial_fraction)


def rattle_stdev(rattle_value: float) -> float:
    return 0.03 + 0.7 * float(rattle_value)


@register_specimen
class NanoparticleMap(Specimen):
    """A dense field of round and oval particles, on one map of intensity, particle id, rattle and local lattice parameter."""

    source = "nanoparticle_map"
    render_modes = ("diffraction_map",)
    OPTIONS = ("map_size", "element", "seed")

    def __init__(self, seed: int, map_size: int = DEFAULT_MAP_SIZE, element: str = "Au") -> None:
        if element not in ELEMENTS:
            raise ValueError(f"nanoparticle_map element {element!r} unknown; known: {sorted(ELEMENTS)}")
        self.element = element
        self.map_size = int(map_size)
        self.bounds = (-FOV_M * 5.0, FOV_M * 5.0, -FOV_M * 5.0, FOV_M * 5.0)
        self.world_bounds_ang = {"x_min": -FOV_NM * 5.0, "x_max": FOV_NM * 5.0, "y_min": -FOV_NM * 5.0, "y_max": FOV_NM * 5.0, "z_min": -50.0, "z_max": 50.0}
        self._generate(int(seed))

    @classmethod
    def from_options(cls, options: dict, context: SpecimenContext) -> "NanoparticleMap":
        check_options(cls.source, options, cls.OPTIONS)
        kwargs = {k: options[k] for k in ("map_size", "element") if k in options}
        return cls(context.rng_for(options).seed, **kwargs)

    def _generate(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        size = max(128, self.map_size)
        particle_id = np.zeros((size, size), dtype=np.int16)
        haadf = np.zeros((size, size), dtype=np.float32)
        rattle = np.zeros((size, size), dtype=np.float32)
        lattice_parameter_map = np.zeros((size, size), dtype=np.float32)
        particle_records: list[dict] = []
        yy, xx = np.mgrid[:size, :size]
        coverage = 0.0
        attempts = 0
        kind_counts = {"round": 0, "oval": 0}

        while coverage < TARGET_COVERAGE and attempts < 4000:
            attempts += 1
            if kind_counts["round"] == kind_counts["oval"]:
                kind = "round" if rng.random() < 0.5 else "oval"
            else:
                kind = "round" if kind_counts["round"] < kind_counts["oval"] else "oval"
            radius = float(rng.uniform(18.0, 34.0))
            if kind == "round":
                axis_x = radius * float(rng.uniform(0.97, 1.03))
                axis_y = radius * float(rng.uniform(0.97, 1.03))
                center_lattice_parameter = float(rng.normal(ROUND_CENTER_LATTICE_MEAN_ANGSTROM, CENTER_LATTICE_STD_ANGSTROM))
                edge_lattice_change = -EDGE_LATTICE_CHANGE_FRACTION
            else:
                axis_x = radius * float(rng.uniform(1.25, 1.65))
                axis_y = radius * float(rng.uniform(0.75, 0.95))
                center_lattice_parameter = float(rng.normal(OVAL_CENTER_LATTICE_MEAN_ANGSTROM, CENTER_LATTICE_STD_ANGSTROM))
                edge_lattice_change = EDGE_LATTICE_CHANGE_FRACTION

            margin = int(max(axis_x, axis_y) * 1.2 + 4)
            if margin * 2 >= size:
                continue
            cx = float(rng.uniform(margin, size - margin))
            cy = float(rng.uniform(margin, size - margin))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            x0 = max(0, int(np.floor(cx - margin)))
            x1 = min(size, int(np.ceil(cx + margin + 1)))
            y0 = max(0, int(np.floor(cy - margin)))
            y1 = min(size, int(np.ceil(cy + margin + 1)))
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            dx = xx[y0:y1, x0:x1] - cx
            dy = yy[y0:y1, x0:x1] - cy
            xr = dx * cos_a + dy * sin_a
            yr = -dx * sin_a + dy * cos_a
            theta = np.arctan2(yr / max(axis_y, 1e-6), xr / max(axis_x, 1e-6))
            blob_scale = 0.35 if kind == "round" else 1.0
            blob = 1.0 + 0.09 * blob_scale * np.sin(3.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
            blob += 0.07 * blob_scale * np.sin(5.0 * theta + rng.uniform(0.0, 2.0 * np.pi))
            normalized_radius = np.sqrt((xr / axis_x) ** 2 + (yr / axis_y) ** 2) / blob
            mask = normalized_radius <= 1.0
            particle_view = particle_id[y0:y1, x0:x1]
            if np.any(mask & (particle_view > 0)):
                continue
            particle_pixels = int(np.count_nonzero(mask))
            if particle_pixels / mask.size < 0.002:
                continue

            record_id = len(particle_records) + 1
            distance_to_edge_px = (1.0 - normalized_radius) * min(axis_x, axis_y)
            edge_rattle = np.clip(1.0 - distance_to_edge_px / AMORPHOUS_EDGE_WIDTH_PX, 0.0, 1.0) ** 0.6
            edge_rattle = (edge_rattle * mask).astype(np.float32)
            local_lattice = local_lattice_parameter(center_lattice_parameter, edge_lattice_change, normalized_radius)
            local_intensity = float(rng.uniform(0.72, 1.0))
            texture = rng.normal(0.0, 0.035, size=mask.shape).astype(np.float32)
            rattle_view = rattle[y0:y1, x0:x1]
            lattice_view = lattice_parameter_map[y0:y1, x0:x1]
            haadf_view = haadf[y0:y1, x0:x1]
            particle_view[mask] = record_id
            rattle_view[mask] = edge_rattle[mask]
            lattice_view[mask] = local_lattice[mask]
            haadf_view[mask] = local_intensity - 0.18 * edge_rattle[mask] + texture[mask]
            particle_records.append(
                {
                    "id": record_id,
                    "kind": kind,
                    "pixel_count": particle_pixels,
                    "center_nm": [float((cx / (size - 1) - 0.5) * FOV_NM), float((cy / (size - 1) - 0.5) * FOV_NM)],
                    "axes_nm": [float(axis_x / size * FOV_NM), float(axis_y / size * FOV_NM)],
                    "angle_rad": angle,
                    "rotation_degrees": float(np.degrees(angle)),
                    "center_lattice_parameter": center_lattice_parameter,
                    "lattice_parameter": center_lattice_parameter,
                    "edge_lattice_change_fraction": edge_lattice_change,
                }
            )
            kind_counts[kind] += 1
            coverage = np.count_nonzero(particle_id) / particle_id.size

        haadf += rng.normal(0.015, 0.01, size=(size, size)).astype(np.float32)
        haadf -= float(haadf.min())
        max_val = float(haadf.max())
        self.overview_map = np.clip(haadf / max_val if max_val > 0.0 else haadf, 0.0, 1.0)
        self.particle_id_map = particle_id
        self.rattle_map = rattle
        self.lattice_parameter_map = lattice_parameter_map
        self.records = particle_records

    @property
    def element_symbols(self) -> tuple[str, ...]:
        return (self.element,)

    def particle_records(self) -> list[dict]:
        """The particles as angstrom circles in sample coordinates, the form the atomistic twins' spectrum reads."""
        composition = {self.element: 1.0}
        return [
            {
                "center": np.array([record["center_nm"][0] * 10.0, record["center_nm"][1] * 10.0, 0.0]),
                "radius": float(max(record["axes_nm"])) * 10.0,
                "btype": self.element,
                "composition": composition,
            }
            for record in self.records
        ]

    def view(self, request: ViewRequest) -> FieldView:
        """maps: {"haadf": the map sampled on an imsize x imsize grid centred where the stage points}."""
        imsize = int(request.imsize)
        source = self.overview_map
        source_size = source.shape[0]
        stage_nm = np.asarray(request.stage[:2], dtype=np.float64) * 1e9
        axis = (np.arange(imsize) + 0.5) / imsize - 0.5
        x_nm = axis * FOV_NM + stage_nm[0]
        y_nm = axis * FOV_NM + stage_nm[1]
        xi = np.rint((x_nm / FOV_NM + 0.5) * (source_size - 1)).astype(int)
        yi = np.rint((y_nm / FOV_NM + 0.5) * (source_size - 1)).astype(int)
        image = np.zeros((imsize, imsize), dtype=np.float32)
        valid_x = (xi >= 0) & (xi < source_size)
        valid_y = (yi >= 0) & (yi < source_size)
        image[np.ix_(valid_y, valid_x)] = source[np.ix_(yi[valid_y], xi[valid_x])]
        return FieldView(maps={"haadf": image})

    def atoms_at(self, x_nm: float, y_nm: float, request: ViewRequest, seed: int, thermal_scale: float = 1.0):
        """The local Au cell (<100> along the beam) at sample point (x_nm, y_nm), or None over vacuum; info describes the particle."""
        size = self.particle_id_map.shape[0]
        ix = int(round((x_nm / FOV_NM + 0.5) * (size - 1)))
        iy = int(round((y_nm / FOV_NM + 0.5) * (size - 1)))
        if ix < 0 or ix >= size or iy < 0 or iy >= size:
            return None
        particle_id = int(self.particle_id_map[iy, ix])
        if particle_id <= 0:
            return None
        particle = self.records[particle_id - 1]
        center = float(particle["center_lattice_parameter"])
        local = float(self.lattice_parameter_map[iy, ix])
        rattle = float(self.rattle_map[iy, ix])
        # A conventional cubic cell puts the beam along the <100> zone-axis family.
        atoms = bulk_cell(self.element, local, cubic=True).repeat((4, 4, 4))
        atoms.center(vacuum=2.0)
        atoms.rotate(float(particle["rotation_degrees"]), "z", center="COP")
        if rattle > 0.0:
            atoms.rattle(stdev=rattle_stdev(rattle) * thermal_scale, seed=seed)
        atoms.info.update(
            {
                "particle_id": int(particle["id"]),
                "particle_kind": str(particle["kind"]),
                "lattice_parameter": local,
                "center_lattice_parameter": center,
                "lattice_strain_fraction": local / center - 1.0,
                "rotation_degrees": float(particle["rotation_degrees"]),
                "rattle_value": rattle,
            }
        )
        return atoms

    def describe(self) -> dict:
        return {**super().describe(), "map_size": self.map_size, "element": self.element, "particles": len(self.records), "atomic_number": atomic_number(self.element)}
