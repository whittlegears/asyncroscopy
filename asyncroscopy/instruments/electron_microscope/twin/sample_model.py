"""A seeded sample stored as a table of particles, not atoms.

A 2 um sample with hundreds of nanoparticles would be ~10^7 atoms; as a
particle table it is a few kilobytes. Atoms are only built (in ``render``) for
the handful of particles inside an atomic-resolution field of view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import EventRng

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


class SampleModel:
    def __init__(self, params: SampleParams, rng: EventRng):
        if params.size_xy_m <= 0:
            raise ValueError("sample size must be > 0")
        unknown = [e for e in (*params.elements, params.support_element) if e not in ELEMENTS]
        if unknown:
            raise ValueError(f"Unknown sample elements {unknown}; known: {sorted(ELEMENTS)}")
        self.params = params
        self._generate(rng.for_event("sample"))

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
        for _ in range(500 * target):
            if len(rows) >= target or misses >= max_misses:
                break
            radius = float(max(0.3e-9, rng.normal(p.radius_mean_m, p.radius_std_m)))
            margin = min(radius, half * 0.9)
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

    def height_at(self, x, y):
        gx, gy = self.params.height_gradient
        return gx * np.asarray(x) + gy * np.asarray(y)

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

    def ground_truth(self) -> list[dict]:
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
