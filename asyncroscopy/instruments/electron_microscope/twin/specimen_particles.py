"""The ``particles`` specimen: a voxel-backed world of oriented nanoparticles for the particle twin.

The world spans the field of view it was built for, so changing the field rebuilds it (``set_field``). ``view`` moves
the world under the stage (translation with wrap-around, then tilt) and gives the atoms and projected label maps for
the twin's pseudo-potential renderer, and, for an image, the particles visible in it (positions in pixels).

All random numbers come from one ``numpy.random.RandomState``, kept with the specimen as ``random``; the twin draws its
image and spectrum noise from the same stream, so a rebuild (which reseeds it) and every acquisition after it repeat.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from .elements import ELEMENTS, bulk_cell
from .specimen import FieldView, Specimen, SpecimenContext, ViewRequest, check_options, register_specimen

DESIRED_ANGLES = [(0, 0, 0), (60, 0, 0), (45, 45, 45)]  # the first three particles' orientations
DEFAULT_FOV_M = 20e-9  # the twin's starting field of view


def rotation_matrix(alpha, beta, gamma):
    a, b, g = np.radians([alpha, beta, gamma])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
    return Rz @ Ry @ Rx


@register_specimen
class ParticleWorld(Specimen):
    """Oriented ellipsoidal particles in a thin world, as ASE atoms and a 3-D label map."""

    source = "particles"
    render_modes = ("particle_potential",)
    OPTIONS = ("world_imsize", "particle_count", "elements", "seed")

    PARTICLE_RADIUS_ANGSTROM = 16.0
    RADIUS_STD_ANGSTROM = 2.0
    ASPECT_RATIO = 0.4
    MIN_SEPARATION_ANGSTROM = 3.0
    MAX_ATTEMPTS = 500

    def __init__(self, seed: int, fov_m: float, world_imsize: int = 128, particle_count: int = 40, elements: tuple[str, ...] = ("Au", "Pt", "Fe")) -> None:
        unknown = [e for e in elements if e not in ELEMENTS]
        if unknown:
            raise ValueError(f"particles element(s) {unknown} unknown; known: {sorted(ELEMENTS)}")
        self.seed = int(seed)
        self.world_imsize = int(world_imsize)
        self.particle_count = int(particle_count)
        self.elements = tuple(elements)
        self.random = np.random.RandomState(self.seed)
        self._build(float(fov_m))

    @classmethod
    def from_options(cls, options: dict, context: SpecimenContext) -> "ParticleWorld":
        check_options(cls.source, options, cls.OPTIONS)
        kwargs: dict = {}
        if "world_imsize" in options:
            kwargs["world_imsize"] = int(options["world_imsize"])
        if "particle_count" in options or context.sample is not None:
            kwargs["particle_count"] = int(options.get("particle_count", context.sample.particle_count if context.sample is not None else 40))
        if "elements" in options:
            kwargs["elements"] = tuple(e.strip() for e in (options["elements"].split(",") if isinstance(options["elements"], str) else options["elements"]) if e.strip())
        return cls(context.rng_for(options).seed, float(context.fov_m or DEFAULT_FOV_M), **kwargs)

    # ------------------------------------------------------------------
    # The world
    # ------------------------------------------------------------------
    def set_field(self, fov_m: float) -> None:
        """Rebuild the world for a new field of view, from the start of the specimen's random stream."""
        self.random.seed(self.seed)
        self._build(float(fov_m))

    def _build(self, fov_m: float) -> None:
        random = self.random
        fov_ang = fov_m * 1e10  # Angstroms, lateral
        world_z_ang = fov_ang * 0.5  # thin slab
        nx = ny = max(16, self.world_imsize)
        vox_size = fov_ang / nx  # Angstroms per voxel (isotropic)
        nz = max(1, int(round(world_z_ang / vox_size)))

        n_particles = max(1, self.particle_count)
        radius_mean, radius_std = self.PARTICLE_RADIUS_ANGSTROM, self.RADIUS_STD_ANGSTROM
        aspect_ratio = self.ASPECT_RATIO
        min_separation = self.MIN_SEPARATION_ANGSTROM
        bulk_types = {element: bulk_cell(element) for element in self.elements}
        bulk_names = list(bulk_types.keys())

        placed_centers: list[tuple[float, float, float]] = []
        placed_particles: list[tuple] = []
        particle_lookup: dict[int, dict] = {}  # label (1-based) -> metadata

        for _ in range(self.MAX_ATTEMPTS * n_particles):
            if len(placed_particles) >= n_particles:
                break
            radius = np.clip(random.normal(radius_mean, radius_std), 3.0, None)
            margin = radius + 2.0
            cx = random.uniform(margin, fov_ang - margin)
            cy = random.uniform(margin, fov_ang - margin)
            cz = world_z_ang * 0.5
            if any(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) < radius + pr + min_separation for px, py, pr in placed_centers):
                continue
            placed_centers.append((cx, cy, radius))
            btype = random.choice(bulk_names)
            i = len(placed_particles)
            angles = DESIRED_ANGLES[i] if i < len(DESIRED_ANGLES) else tuple(random.rand(3) * 360)
            placed_particles.append((btype, np.array([cx, cy, cz]), radius, angles))
            symbols_in_bulk = bulk_types[btype].get_chemical_symbols()
            counts_dict: dict[str, int] = {}
            for symbol in symbols_in_bulk:
                counts_dict[symbol] = counts_dict.get(symbol, 0) + 1
            total = sum(counts_dict.values())
            composition = {symbol: count / total for symbol, count in counts_dict.items()}
            label = i + 1  # 1-based so 0 stays vacuum
            particle_lookup[label] = {
                "btype": btype,
                "composition": composition,
                "center_ang": np.array([cx, cy, cz]),
                "center_vox": np.array([cx / vox_size, cy / vox_size, cz / vox_size]),
                "radius_ang": radius,
                "radius_vox": radius / vox_size,
                "angles": angles,
            }

        # 3-D label map: each voxel gets the label of the first particle that claims it.
        label_map = np.zeros((nx, ny, nz), dtype=np.uint8)
        xv = (np.arange(nx) + 0.5) * vox_size
        yv = (np.arange(ny) + 0.5) * vox_size
        zv = (np.arange(nz) + 0.5) * vox_size
        XX, YY, ZZ = np.meshgrid(xv, yv, zv, indexing="ij")
        for label, (_btype, center, radius, angles) in enumerate(placed_particles, start=1):
            z_radius = radius * aspect_ratio
            Rinv = rotation_matrix(*angles).T
            dX, dY, dZ = XX - center[0], YY - center[1], ZZ - center[2]
            dXr = Rinv[0, 0] * dX + Rinv[0, 1] * dY + Rinv[0, 2] * dZ
            dYr = Rinv[1, 0] * dX + Rinv[1, 1] * dY + Rinv[1, 2] * dZ
            dZr = Rinv[2, 0] * dX + Rinv[2, 1] * dY + Rinv[2, 2] * dZ
            inside = (dXr / radius) ** 2 + (dYr / radius) ** 2 + (dZr / z_radius) ** 2 <= 1.0
            label_map[inside & (label_map == 0)] = label

        all_positions = []
        all_symbols: list[str] = []
        for btype, center, radius, angles in placed_particles:
            this_bulk = bulk_types[btype]
            a_lat = this_bulk.cell.lengths()[0]
            z_radius = radius * aspect_ratio
            rep = int(radius * 2 / a_lat) + 3
            supercell = this_bulk.repeat((rep, rep, rep))
            R = rotation_matrix(*angles)
            positions = supercell.get_positions().copy()
            positions -= positions.mean(axis=0)
            positions = positions @ R.T
            r_scaled = np.sqrt((positions[:, 0] / radius) ** 2 + (positions[:, 1] / radius) ** 2 + (positions[:, 2] / z_radius) ** 2)
            mask = r_scaled <= 1.0
            all_positions.append(positions[mask] + center)
            all_symbols.extend(s for s, m in zip(supercell.get_chemical_symbols(), mask) if m)

        self._atoms = Atoms(symbols=all_symbols, positions=np.vstack(all_positions), cell=[fov_ang, fov_ang, world_z_ang], pbc=False)
        self.vox_size = vox_size
        self.world_shape = (nx, ny, nz)
        self.particle_lookup = particle_lookup
        self._label_map = label_map
        self.bounds = (0.0, fov_ang * 1e-10, 0.0, fov_ang * 1e-10)
        self.world_bounds_ang = {"x_min": 0.0, "x_max": fov_ang, "y_min": 0.0, "y_max": fov_ang, "z_min": 0.0, "z_max": world_z_ang}

    @property
    def element_symbols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(meta["btype"] for meta in self.particle_lookup.values()))

    # ------------------------------------------------------------------
    # One view: the world under the stage
    # ------------------------------------------------------------------
    def view(self, request: ViewRequest) -> FieldView:
        """atoms: the world's atoms where the stage has put them (wrapped into the field, angstrom);
        maps: {"projected_label", "projected_thickness"}; records: the particles visible in an image of request.imsize pixels."""
        x_m, y_m, z_m, alpha_deg, beta_deg = (float(v) for v in request.stage)
        x_ang, y_ang, z_ang = x_m * 1e10, y_m * 1e10, z_m * 1e10
        nx, ny, nz = self.world_shape
        vox = self.vox_size

        def _Ry(deg):
            t = np.radians(deg)
            return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

        def _Rx(deg):
            t = np.radians(deg)
            return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])

        R_stage = _Rx(beta_deg) @ _Ry(alpha_deg)  # alpha about y, beta about x, around the world's centre

        positions = self._atoms.get_positions().copy()
        world_centre_ang = np.array([nx * vox / 2.0, ny * vox / 2.0, nz * vox / 2.0])
        positions[:, 0] -= x_ang
        positions[:, 1] -= y_ang
        positions[:, 2] -= z_ang
        positions -= world_centre_ang
        positions = positions @ R_stage.T
        positions += world_centre_ang
        positions[:, 0] = np.mod(positions[:, 0], nx * vox)  # wrapped, so atoms never run out of the field
        positions[:, 1] = np.mod(positions[:, 1], ny * vox)
        atoms = self._atoms.copy()
        atoms.set_positions(positions)

        # Translate the label map by whole voxels (a wrap), then tilt it by inverse-mapping each output voxel.
        rolled = np.roll(self._label_map, -int(round(x_ang / vox)), axis=0)
        rolled = np.roll(rolled, -int(round(y_ang / vox)), axis=1)
        xv = np.arange(nx) - nx / 2.0
        yv = np.arange(ny) - ny / 2.0
        zv = np.arange(nz) - nz / 2.0
        XX, YY, ZZ = np.meshgrid(xv, yv, zv, indexing="ij")
        Rinv = R_stage.T
        src_x = Rinv[0, 0] * XX + Rinv[0, 1] * YY + Rinv[0, 2] * ZZ + nx / 2.0
        src_y = Rinv[1, 0] * XX + Rinv[1, 1] * YY + Rinv[1, 2] * ZZ + ny / 2.0
        src_z = Rinv[2, 0] * XX + Rinv[2, 1] * YY + Rinv[2, 2] * ZZ + nz / 2.0
        src_xi = np.clip(np.round(src_x).astype(int), 0, nx - 1)
        src_yi = np.clip(np.round(src_y).astype(int), 0, ny - 1)
        src_zi = np.clip(np.round(src_z).astype(int), 0, nz - 1)
        rotated_map = rolled[src_xi, src_yi, src_zi]

        # The beam travels along z: each column shows the first (shallowest) particle it meets.
        occupancy = rotated_map > 0
        projected_label = np.zeros((nx, ny), dtype=np.uint8)
        for iz in range(nz):
            slice_label = rotated_map[:, :, iz]
            unfilled = projected_label == 0
            projected_label[unfilled] = slice_label[unfilled]
        projected_thickness = occupancy.sum(axis=2).astype(np.float32) * vox

        return FieldView(
            atoms=atoms,
            records=self._visible_particles(projected_label, request),
            maps={"projected_label": projected_label, "projected_thickness": projected_thickness},
            info={"world_shape": list(self.world_shape), "voxel_size_angstrom": float(self.vox_size), "particle_count": len(self.particle_lookup)},
        )

    def _visible_particles(self, projected_label: np.ndarray, request: ViewRequest) -> list[dict]:
        """The particles an image of ``request.imsize`` pixels shows, positioned in its pixels (with edge_crop pixels of padding around it)."""
        size, pad = int(request.imsize), int(request.edge_crop)
        if size <= 0:
            return []
        pixel_size = request.fov_angstrom / size
        world_nx, world_ny = projected_label.shape
        # The image sits at the centre of the world map; only the padded image region counts.
        img_origin_x = (world_nx - size) // 2
        img_origin_y = (world_ny - size) // 2
        x0 = max(img_origin_x - pad, 0)
        y0 = max(img_origin_y - pad, 0)
        x1 = min(img_origin_x + size + pad, world_nx)
        y1 = min(img_origin_y + size + pad, world_ny)
        label_crop = projected_label[x0:x1, y0:y1]
        records = []
        for label, meta in self.particle_lookup.items():
            yx = np.argwhere(label_crop == label)
            if yx.size == 0:
                continue  # not in this field
            centre_crop = yx.mean(axis=0)
            records.append(
                {
                    "center": np.array([centre_crop[0] - pad, centre_crop[1] - pad]),
                    "radius": meta["radius_ang"] / pixel_size,
                    "btype": meta["btype"],
                    "composition": meta["composition"],
                }
            )
        return records

    def describe(self) -> dict:
        return {**super().describe(), "world_imsize": self.world_imsize, "particle_count": len(self.particle_lookup), "elements": list(self.elements)}
