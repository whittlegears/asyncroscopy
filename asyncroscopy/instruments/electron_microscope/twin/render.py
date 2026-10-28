"""Turn the particle table into the signal a detector sees.

Two levels of detail, picked by pixel size:

* survey (pixel >= ``atomic_pixel_m``): each particle contributes its
  analytic projected thickness times Z^n; the probe is a Gaussian blur.
* atomic (pixel < ``atomic_pixel_m``): atoms are built with ASE for the
  particles in view only, and the image is convolved with the pyTEMlib probe.

The ideal object is computed on a regular grid in sample coordinates and then
sampled at the (distorted, drifting) pixel coordinates from
``scan_distortion``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.build import bulk
from scipy import ndimage

from .optics import atomic_probe, survey_blur_sigma_m
from .sample_model import ELEMENTS, SampleModel

ANGSTROM = 1e-10


@dataclass
class RenderParams:
    z_exponent: float = 1.7
    # Fraction scattered to HAADF is 1 - exp(-k * sum(Z^n * rho * t)), t in angstrom.
    # 2.25e-5 gives ~30% for a 16 nm Au particle and ~0.5% for 10 nm of carbon.
    haadf_scale: float = 2.25e-5
    atomic_pixel_m: float = 0.2e-9
    voltage_v: float = 300e3
    convergence_mrad: float = 30.0


DETECTOR_SCALE = {"HAADF": 1.0, "ADF": 2.5, "BF": 4.0}


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


def object_signal(weight: np.ndarray, detector: str, params: RenderParams) -> np.ndarray:
    """Fraction of incident electrons reaching the detector, from the Z^n-weighted mass thickness."""
    detector = detector.upper()
    if detector not in DETECTOR_SCALE:
        raise ValueError(f"Unknown scanning detector {detector!r}; the twin simulates {sorted(DETECTOR_SCALE)}")
    k = params.haadf_scale * DETECTOR_SCALE[detector]
    if detector == "BF":
        return np.exp(-k * weight)
    if detector == "ADF":
        # Medium-angle signal: scattered, but not so strongly that it goes to HAADF.
        return np.exp(-params.haadf_scale * weight) - np.exp(-k * weight)
    return 1.0 - np.exp(-k * weight)


class Renderer:
    def __init__(self, sample: SampleModel, params: RenderParams):
        self.sample = sample
        self.params = params
        self._atoms: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def level_for(self, pixel_m: float) -> str:
        return "atomic" if pixel_m < self.params.atomic_pixel_m else "survey"

    def _grid(self, xs: np.ndarray, ys: np.ndarray, pixel_m: float, pad_m: float):
        x_min, x_max = float(xs.min()) - pad_m, float(xs.max()) + pad_m
        y_min, y_max = float(ys.min()) - pad_m, float(ys.max()) + pad_m
        n = int(np.ceil(max(x_max - x_min, y_max - y_min) / pixel_m)) + 1
        return x_min, y_min, n

    def _weight_survey(self, x_min, y_min, n, pixel_m, damage) -> np.ndarray:
        s = self.sample
        coords = x_min + (np.arange(n) + 0.5) * pixel_m
        ycoords = y_min + (np.arange(n) + 0.5) * pixel_m
        weight = np.zeros((n, n), dtype=np.float64)
        x_max, y_max = x_min + n * pixel_m, y_min + n * pixel_m
        zn = s.z ** self.params.z_exponent
        for i in s.in_box(x_min, x_max, y_min, y_max):
            reach = s.semi_axes[i, :2].max()
            c0 = max(0, int((s.centers[i, 0] - reach - x_min) / pixel_m))
            c1 = min(n, int((s.centers[i, 0] + reach - x_min) / pixel_m) + 2)
            r0 = max(0, int((s.centers[i, 1] - reach - y_min) / pixel_m))
            r1 = min(n, int((s.centers[i, 1] + reach - y_min) / pixel_m) + 2)
            if c0 >= c1 or r0 >= r1:
                continue
            gx, gy = np.meshgrid(coords[c0:c1], ycoords[r0:r1])
            t_angstrom = s.thickness_at(i, gx, gy) / ANGSTROM
            weight[r0:r1, c0:c1] += zn[i] * s.density[i] * t_angstrom * damage.get(int(i), 1.0)
        return weight

    def _particle_atoms(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Projected (x, y) atom positions of one particle, relative to its centre; built once, cached."""
        if i not in self._atoms:
            s = self.sample
            element = s.elements[i]
            z, _rho, structure, lattice = ELEMENTS[element]
            a, b, c = s.semi_axes[i] / ANGSTROM
            basis = bulk(element, structure, a=lattice, cubic=True).get_scaled_positions()
            reach = int(np.ceil(max(a, b, c) / lattice)) + 1
            steps = np.arange(-reach, reach, dtype=np.float32)
            cells = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 1, 3)
            pos = ((cells + basis[None, :, :].astype(np.float32)) * np.float32(lattice)).reshape(-1, 3)
            pos = pos @ _euler_matrix(*s.euler[i]).T.astype(np.float32)
            cos_t, sin_t = np.cos(s.theta[i]), np.sin(s.theta[i])
            u = (pos[:, 0] * cos_t + pos[:, 1] * sin_t) / a
            v = (-pos[:, 0] * sin_t + pos[:, 1] * cos_t) / b
            keep = u * u + v * v + (pos[:, 2] / c) ** 2 <= 1.0
            self._atoms[i] = (pos[keep, :2].astype(np.float64) * ANGSTROM, np.full(int(keep.sum()), float(z)))
        return self._atoms[i]

    def _weight_atomic(self, x_min, y_min, n, pixel_m, damage) -> np.ndarray:
        s = self.sample
        weight = np.zeros((n, n), dtype=np.float64)
        pixel_area_a2 = (pixel_m / ANGSTROM) ** 2
        for i in s.in_box(x_min, x_min + n * pixel_m, y_min, y_min + n * pixel_m):
            positions, zs = self._particle_atoms(int(i))
            cols = np.floor((positions[:, 0] + s.centers[i, 0] - x_min) / pixel_m).astype(np.int64)
            rows = np.floor((positions[:, 1] + s.centers[i, 1] - y_min) / pixel_m).astype(np.int64)
            ok = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
            np.add.at(weight, (rows[ok], cols[ok]), zs[ok] ** self.params.z_exponent / pixel_area_a2 * damage.get(int(i), 1.0))
        # Thermal smearing of the atomic columns (~0.08 A rms).
        return ndimage.gaussian_filter(weight, sigma=max(0.3, 0.08e-10 / pixel_m))

    def render(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        pixel_m: float,
        detector: str,
        ab_nm: dict,
        support_angstrom: float,
        damage: dict[int, float] | None = None,
    ) -> np.ndarray:
        """Expected detector fraction (0-1) at each pixel coordinate."""
        damage = damage or {}
        level = self.level_for(pixel_m)
        center_x, center_y = float(xs.mean()), float(ys.mean())
        local_height = float(self.sample.height_at(center_x, center_y))
        sigma_m = survey_blur_sigma_m(ab_nm, self.params.convergence_mrad, local_height)
        pad = 4.0 * sigma_m + 2.0 * pixel_m
        x_min, y_min, n = self._grid(xs, ys, pixel_m, pad)
        if level == "survey":
            weight = self._weight_survey(x_min, y_min, n, pixel_m, damage)
        else:
            weight = self._weight_atomic(x_min, y_min, n, pixel_m, damage)
        support_z, support_rho = ELEMENTS[self.sample.params.support_element][:2]
        weight += support_z ** self.params.z_exponent * support_rho * support_angstrom
        obj = object_signal(weight, detector, self.params)
        if level == "survey":
            obj = ndimage.gaussian_filter(obj, sigma=sigma_m / pixel_m, mode="nearest")
        else:
            ab = dict(ab_nm)
            ab["C10"] = ab.get("C10", 0.0) - local_height * 1e9
            probe = atomic_probe(ab, n, n * pixel_m, self.params.voltage_v, self.params.convergence_mrad)
            obj = np.real(np.fft.ifft2(np.fft.fft2(obj) * np.fft.fft2(probe)))
        rows = (ys - y_min) / pixel_m - 0.5
        cols = (xs - x_min) / pixel_m - 0.5
        return ndimage.map_coordinates(obj, [rows, cols], order=1, mode="nearest").clip(0.0, 1.0)
