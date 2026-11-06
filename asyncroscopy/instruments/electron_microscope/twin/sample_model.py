"""A seeded sample stored as a table of particles, not atoms: the ``procedural`` specimen.

A 2 um sample with hundreds of nanoparticles would be ~10^7 atoms; as a
particle table it is a few kilobytes. Atoms are only built (in
``projected_weight``) for the handful of particles inside an
atomic-resolution field of view.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields, replace

import numpy as np
from ase.build import bulk
from scipy import ndimage

from asyncroscopy.sim.rng import EventRng
from .specimen import Specimen, SpecimenContext, check_options, register_specimen

ANGSTROM = 1e-10

# Z, atoms per cubic angstrom, ASE bulk structure and lattice constant (angstrom).
ELEMENTS: dict[str, tuple[int, float, str, float]] = {
    "C": (6, 0.100, "diamond", 3.567),
    "Si": (14, 0.050, "diamond", 5.431),
    "Fe": (26, 0.085, "bcc", 2.87),
    "Ni": (28, 0.091, "fcc", 3.52),
    "Cu": (29, 0.085, "fcc", 3.615),
    "Pd": (46, 0.068, "fcc", 3.89),
    "Ag": (47, 0.059, "fcc", 4.09),
    "Pt": (78, 0.066, "fcc", 3.92),
    "Au": (79, 0.059, "fcc", 4.08),
}



PLACEMENT_TIME_LIMIT_S = 15.0  # longest the twin spends placing particles at start-up
TEXTURE_FINEST_M = 0.5e-9  # the finest structure of the support film's texture
TEXTURE_WAVES = 96  # plane waves summed for the texture, spread evenly over the octaves of length it spans

@dataclass
class SampleParams:
    size_xy_m: float = 6e-9
    particle_count: int = 40
    radius_mean_m: float = 1.6e-9
    radius_std_m: float = 0.2e-9
    aspect_ratio: float = 0.4
    elements: tuple[str, ...] = ("Au", "Pt", "Fe")
    min_separation_m: float = 0.3e-9
    support_element: str = "C"
    support_thickness_m: float = 0.0
    height_gradient: tuple[float, float] = (0.0, 0.0)
    # Clusters give a large map structure at every zoom: bright aggregates in the overview,
    # single particles at full resolution. cluster_fraction of the particles fall within
    # about cluster_radius_m of one of cluster_count centres; 0 keeps the sample uniform.
    cluster_count: int = 0
    cluster_radius_m: float = 0.0
    cluster_fraction: float = 0.0
    # The support film's thickness varies by about this fraction (standard deviation; 0: a uniform film), with
    # structure at every length from support_texture_scale_m down to TEXTURE_FINEST_M, so overlapping tiles can be
    # registered on the film itself at any pixel size. Drawn from its own seeded stream: the particles do not change.
    support_texture: float = 0.0
    support_texture_scale_m: float = 2e-6
    # Atoms are drawn at their sub-pixel positions (each spread over the four nearest pixels) rather than at the
    # pixel they fall in, so one column lands at the same place in two overlapping atomic frames.
    subpixel_atoms: bool = False


def _euler_matrix(e1_deg: float, e2_deg: float, e3_deg: float) -> np.ndarray:
    """Rotation about x by e3, then y by e2, then z by e1 (the order the original twin used)."""
    def rot(axis: int, deg: float) -> np.ndarray:
        t = np.deg2rad(deg)
        c, s = np.cos(t), np.sin(t)
        m = np.eye(3)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
        if axis == 1:
            m[i, j], m[j, i] = s, -s
        return m
    return rot(2, e1_deg) @ rot(1, e2_deg) @ rot(0, e3_deg)


def thermal_smear(weight: np.ndarray, pixel_m: float) -> np.ndarray:
    """Thermal smearing of atomic columns (~0.08 A rms), for any source that places atoms."""
    return ndimage.gaussian_filter(weight, sigma=max(0.3, 0.08e-10 / pixel_m))


class ParticleTable:
    """Ellipsoidal particles as arrays: ``ids``, ``centers`` (N, 2), ``semi_axes`` (N, 3), ``theta``, ``elements``.

    Shared by the procedural specimen and a recorded specimen's labels.
    """

    ids: np.ndarray
    centers: np.ndarray
    semi_axes: np.ndarray
    theta: np.ndarray
    elements: list

    def __len__(self) -> int:
        return len(self.ids)

    def in_box(self, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        """Indices of particles whose extent overlaps the box."""
        if len(self) == 0:
            return np.zeros(0, dtype=np.int64)
        reach = self.semi_axes[:, :2].max(axis=1)
        mask = (
            (self.centers[:, 0] + reach >= x_min)
            & (self.centers[:, 0] - reach <= x_max)
            & (self.centers[:, 1] + reach >= y_min)
            & (self.centers[:, 1] - reach <= y_max)
        )
        return np.nonzero(mask)[0]

    def thickness_at(self, index: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Projected thickness (m) of one ellipsoidal particle at sample points."""
        a, b, c = self.semi_axes[index]
        dx = np.asarray(x) - self.centers[index, 0]
        dy = np.asarray(y) - self.centers[index, 1]
        cos_t, sin_t = np.cos(self.theta[index]), np.sin(self.theta[index])
        u = (dx * cos_t + dy * sin_t) / a
        v = (-dx * sin_t + dy * cos_t) / b
        inside = 1.0 - u * u - v * v
        return 2.0 * c * np.sqrt(np.clip(inside, 0.0, None))

    @property
    def feature_centers(self) -> np.ndarray:
        return self.centers

    def particle_table(self) -> list[dict]:
        """The particles in export_ground_truth's format (also the format of a recorded specimen's labels file)."""
        return [
            {
                "id": int(self.ids[i]),
                "center_m": [float(v) for v in self.centers[i]],
                "semi_axes_m": [float(v) for v in self.semi_axes[i]],
                "theta_rad": float(self.theta[i]),
                "element": self.elements[i],
            }
            for i in range(len(self))
        ]


@register_specimen
class SampleModel(ParticleTable, Specimen):
    """Seeded ellipsoidal particles on a support film; options are ``SampleParams`` fields."""

    source = "procedural"

    def __init__(self, params: SampleParams, rng: EventRng) -> None:
        if params.size_xy_m <= 0:
            raise ValueError("sample size must be > 0")
        if params.cluster_count < 0 or not 0.0 <= params.cluster_fraction <= 1.0 or params.cluster_radius_m < 0:
            raise ValueError("cluster_count must be >= 0, cluster_fraction in [0, 1] and cluster_radius_m >= 0")
        if params.support_texture < 0 or params.support_texture_scale_m <= TEXTURE_FINEST_M:
            raise ValueError(f"support_texture must be >= 0 and support_texture_scale_m above {TEXTURE_FINEST_M} m")
        unknown = [e for e in (*params.elements, params.support_element) if e not in ELEMENTS]
        if unknown:
            raise ValueError(f"Unknown sample elements {unknown}; known: {sorted(ELEMENTS)}")
        self.params = params
        self._atoms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._generate(rng.for_event("sample"))
        self._texture_waves = self._draw_texture(rng.for_event("support_texture")) if params.support_texture > 0 else None

    @classmethod
    def from_options(cls, options: dict, context: SpecimenContext) -> "SampleModel":
        """The twin's flat sample properties, with any ``SampleParams`` field (and ``seed``) overridden."""
        names = {f.name for f in fields(SampleParams)}
        check_options(cls.source, options, names | {"seed"})
        overrides = {k: (tuple(v) if isinstance(v, list) else v) for k, v in options.items() if k in names}
        return cls(replace(context.sample or SampleParams(), **overrides), context.rng_for(options))

    # Specimen interface ------------------------------------------------
    @property
    def support_element(self) -> str:
        return self.params.support_element

    @property
    def support_thickness_m(self) -> float:
        return self.params.support_thickness_m

    @property
    def element_symbols(self) -> tuple[str, ...]:
        return tuple(self.params.elements)

    def warnings(self) -> list[str]:
        if len(self) >= self.requested_count:
            return []
        return [
            f"Placed {len(self)} of {self.requested_count} particles: the rest do not fit in "
            f"{self.params.size_xy_m:.2e} m. Increase sample_size_xy or reduce the count or radius."
        ]

    def describe(self) -> dict:
        p = self.params
        return {
            **super().describe(),
            "particles": len(self),
            "elements": list(p.elements),
            "radius_mean_m": p.radius_mean_m,
            "clusters": p.cluster_count,
            "support_texture": p.support_texture,
            "subpixel_atoms": p.subpixel_atoms,
        }

    def _draw_texture(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Wave vectors (kx, ky), phases and amplitudes of the support texture: lengths spread evenly in log from
        support_texture_scale_m to TEXTURE_FINEST_M, random directions, equal amplitude per wave, scaled so the
        sum has standard deviation ``support_texture``."""
        lengths = np.geomspace(self.params.support_texture_scale_m, TEXTURE_FINEST_M, TEXTURE_WAVES)
        angles = rng.uniform(0.0, np.pi, TEXTURE_WAVES)
        phases = rng.uniform(0.0, 2.0 * np.pi, TEXTURE_WAVES)
        k = 2.0 * np.pi / lengths
        amplitude = np.full(TEXTURE_WAVES, self.params.support_texture * np.sqrt(2.0 / TEXTURE_WAVES))
        return k * np.cos(angles), k * np.sin(angles), phases, amplitude

    def support_texture(self, x_min: float, y_min: float, n: int, pixel_m: float) -> np.ndarray | None:
        if self._texture_waves is None:
            return None
        kx, ky, phases, amplitude = self._texture_waves
        # Waves shorter than two pixels are left out: on a grid of this pixel they would alias into a pattern that
        # depends on where the grid happens to start, so two overlapping frames would not see the same film.
        seen = 2.0 * np.pi / np.hypot(kx, ky) >= 2.0 * pixel_m
        kx, ky, phases, amplitude = kx[seen], ky[seen], phases[seen], amplitude[seen]
        xs = x_min + (np.arange(n) + 0.5) * pixel_m
        ys = y_min + (np.arange(n) + 0.5) * pixel_m
        ax = np.outer(xs, kx) + phases
        ay = np.outer(ys, ky)
        texture = (np.cos(ay) * amplitude) @ np.cos(ax).T - (np.sin(ay) * amplitude) @ np.sin(ax).T
        return np.maximum(texture, -1.0)

    def projected_weight(self, x_min: float, y_min: float, n: int, pixel_m: float, level: str, z_exponent: float, damage: dict[int, float]) -> np.ndarray:
        if level == "survey":
            return self._weight_survey(x_min, y_min, n, pixel_m, z_exponent, damage)
        return self._weight_atomic(x_min, y_min, n, pixel_m, z_exponent, damage)

    def areal_density(self, x: float, y: float) -> dict[str, float]:
        density: dict[str, float] = {}
        for i in self.in_box(x, x, y, y):
            t = float(self.thickness_at(int(i), np.array(x), np.array(y)))
            if t > 0:
                element = self.elements[int(i)]
                density[element] = density.get(element, 0.0) + ELEMENTS[element][1] * t / ANGSTROM
        return density

    # Rendering -----------------------------------------------------------
    def _weight_survey(self, x_min, y_min, n, pixel_m, z_exponent, damage) -> np.ndarray:
        """Each particle's analytic projected thickness times Z^n."""
        coords = x_min + (np.arange(n) + 0.5) * pixel_m
        ycoords = y_min + (np.arange(n) + 0.5) * pixel_m
        weight = np.zeros((n, n), dtype=np.float64)
        x_max, y_max = x_min + n * pixel_m, y_min + n * pixel_m
        zn = self.z ** z_exponent
        for i in self.in_box(x_min, x_max, y_min, y_max):
            reach = self.semi_axes[i, :2].max()
            c0 = max(0, int((self.centers[i, 0] - reach - x_min) / pixel_m))
            c1 = min(n, int((self.centers[i, 0] + reach - x_min) / pixel_m) + 2)
            r0 = max(0, int((self.centers[i, 1] - reach - y_min) / pixel_m))
            r1 = min(n, int((self.centers[i, 1] + reach - y_min) / pixel_m) + 2)
            if c0 >= c1 or r0 >= r1:
                continue
            gx, gy = np.meshgrid(coords[c0:c1], ycoords[r0:r1])
            t_angstrom = self.thickness_at(i, gx, gy) / ANGSTROM
            weight[r0:r1, c0:c1] += zn[i] * self.density[i] * t_angstrom * damage.get(int(i), 1.0)
        return weight

    def _particle_atoms(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Projected (x, y) atom positions of one particle, relative to its centre; built once, cached."""
        if i not in self._atoms:
            element = self.elements[i]
            z, _rho, structure, lattice = ELEMENTS[element]
            a, b, c = self.semi_axes[i] / ANGSTROM
            basis = bulk(element, structure, a=lattice, cubic=True).get_scaled_positions()
            reach = int(np.ceil(max(a, b, c) / lattice)) + 1
            steps = np.arange(-reach, reach, dtype=np.float32)
            cells = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 1, 3)
            pos = ((cells + basis[None, :, :].astype(np.float32)) * np.float32(lattice)).reshape(-1, 3)
            pos = pos @ _euler_matrix(*self.euler[i]).T.astype(np.float32)
            cos_t, sin_t = np.cos(self.theta[i]), np.sin(self.theta[i])
            u = (pos[:, 0] * cos_t + pos[:, 1] * sin_t) / a
            v = (-pos[:, 0] * sin_t + pos[:, 1] * cos_t) / b
            keep = u * u + v * v + (pos[:, 2] / c) ** 2 <= 1.0
            self._atoms[i] = (pos[keep, :2].astype(np.float64) * ANGSTROM, np.full(int(keep.sum()), float(z)))
        return self._atoms[i]

    def _weight_atomic(self, x_min, y_min, n, pixel_m, z_exponent, damage) -> np.ndarray:
        """Atoms built with ASE for the particles in view only."""
        weight = np.zeros((n, n), dtype=np.float64)
        pixel_area_a2 = (pixel_m / ANGSTROM) ** 2
        for i in self.in_box(x_min, x_min + n * pixel_m, y_min, y_min + n * pixel_m):
            positions, zs = self._particle_atoms(int(i))
            value = zs ** z_exponent / pixel_area_a2 * damage.get(int(i), 1.0)
            fx = (positions[:, 0] + self.centers[i, 0] - x_min) / pixel_m
            fy = (positions[:, 1] + self.centers[i, 1] - y_min) / pixel_m
            if not self.params.subpixel_atoms:
                cols, rows = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
                ok = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
                np.add.at(weight, (rows[ok], cols[ok]), value[ok])
                continue
            # Pixel (r, c) is centred at (c + 0.5, r + 0.5) in these units: share each atom with the four nearest centres.
            fx, fy = fx - 0.5, fy - 0.5
            c0, r0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
            wx, wy = fx - c0, fy - r0
            for dr, dc, share in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx), (1, 0, wy * (1 - wx)), (1, 1, wy * wx)):
                rows, cols = r0 + dr, c0 + dc
                ok = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
                np.add.at(weight, (rows[ok], cols[ok]), value[ok] * share[ok])
        return thermal_smear(weight, pixel_m)

    def _generate(self, rng: np.random.Generator) -> None:
        p = self.params
        half = p.size_xy_m / 2.0
        target = max(1, int(p.particle_count))
        cell = max(4.0 * p.radius_mean_m, p.size_xy_m / 64.0)
        grid: dict[tuple[int, int], list[int]] = {}
        rows: list[tuple] = []
        # Stop after a run of failed placements: asking for more particles than fit must not
        # stall device start-up (it would otherwise retry 500 x count times).
        misses, max_misses = 0, 2000
        deadline = time.monotonic() + PLACEMENT_TIME_LIMIT_S  # a crowded sample keeps succeeding just often enough to never stop
        clustered = p.cluster_count > 0 and p.cluster_fraction > 0 and p.cluster_radius_m > 0
        # Drawn only when clusters are on, so a uniform sample with the same seed is unchanged.
        centres = rng.uniform(-half, half, size=(p.cluster_count, 2)) if clustered else None
        for attempt in range(500 * target):
            if len(rows) >= target or misses >= max_misses:
                break
            if attempt % 1000 == 0 and time.monotonic() > deadline:
                break
            radius = float(max(0.3e-9, rng.normal(p.radius_mean_m, p.radius_std_m)))
            margin = min(radius, half * 0.9)
            if clustered and rng.random() < p.cluster_fraction:
                centre = centres[rng.integers(p.cluster_count)]
                cx, cy = (float(v) for v in rng.normal(centre, p.cluster_radius_m))
                if not (-half + margin <= cx <= half - margin and -half + margin <= cy <= half - margin):
                    misses += 1
                    continue
            else:
                cx = float(rng.uniform(-half + margin, half - margin))
                cy = float(rng.uniform(-half + margin, half - margin))
            gx, gy = int(np.floor(cx / cell)), int(np.floor(cy / cell))
            reach = int(np.ceil((2 * radius + p.min_separation_m) / cell)) + 1
            clash = False
            for ix in range(gx - reach, gx + reach + 1):
                for iy in range(gy - reach, gy + reach + 1):
                    for j in grid.get((ix, iy), ()):
                        ox, oy, orad = rows[j][1], rows[j][2], rows[j][3]
                        if np.hypot(cx - ox, cy - oy) < radius + orad + p.min_separation_m:
                            clash = True
                            break
                    if clash:
                        break
                if clash:
                    break
            if clash:
                misses += 1
                continue
            misses = 0
            element = str(rng.choice(list(p.elements)))
            elong = float(rng.uniform(0.8, 1.2))
            theta = float(rng.uniform(0.0, np.pi))
            euler = tuple(float(v) for v in rng.uniform(0.0, 360.0, size=3))
            grid.setdefault((gx, gy), []).append(len(rows))
            rows.append((len(rows), cx, cy, radius, radius * elong, radius * p.aspect_ratio, theta, element, euler))

        n = len(rows)
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.centers = np.array([[r[1], r[2]] for r in rows], dtype=np.float64).reshape(n, 2)
        self.semi_axes = np.array([[r[3], r[4], r[5]] for r in rows], dtype=np.float64).reshape(n, 3)
        self.theta = np.array([r[6] for r in rows], dtype=np.float64)
        self.elements = [r[7] for r in rows]
        self.euler = [r[8] for r in rows]
        self.z = np.array([ELEMENTS[e][0] for e in self.elements], dtype=np.float64)
        self.density = np.array([ELEMENTS[e][1] for e in self.elements], dtype=np.float64)
        self.bounds = (-half, half, -half, half)
        self.requested_count = target  # may exceed len(self) when the particles do not fit

    def height_at(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        gx, gy = self.params.height_gradient
        return gx * np.asarray(x) + gy * np.asarray(y)

    def ground_truth(self) -> dict:
        particles = self.particle_table()
        for record, euler in zip(particles, self.euler):  # each particle's lattice orientation (ZXZ Euler angles)
            record["euler_deg"] = [float(v) for v in euler]
        return {"known": True, "particles": particles}
