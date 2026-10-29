"""Turn a specimen into the signal a detector sees.

Two levels of detail, picked by pixel size:

* survey (pixel >= ``atomic_pixel_m``): the specimen gives a smooth projected
  Z^n-weighted mass thickness; the probe is a Gaussian blur.
* atomic (pixel < ``atomic_pixel_m``): the specimen places atoms (the
  procedural one builds them with ASE for the particles in view only), and the
  image is convolved with the pyTEMlib probe.

What the specimen is (particles, a recorded image, a crystal film) is the
specimen's business (``specimen.py``). The detector response, the probe, the
specimen height's defocus and the support and contamination layers are the
instrument's, and are applied here.

The ideal object is computed on a regular grid in sample coordinates and then
sampled at the (distorted, drifting) pixel coordinates from
``scan_distortion``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .optics import atomic_probe, survey_blur_sigma_m
from .sample_model import ELEMENTS
from .specimen import Specimen

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
    def __init__(self, sample: Specimen, params: RenderParams):
        self.sample = sample  # any Specimen: it gives the projected weight, the renderer applies the instrument
        self.params = params

    def level_for(self, pixel_m: float) -> str:
        return "atomic" if pixel_m < self.params.atomic_pixel_m else "survey"

    def _grid(self, xs: np.ndarray, ys: np.ndarray, pixel_m: float, pad_m: float):
        x_min, x_max = float(xs.min()) - pad_m, float(xs.max()) + pad_m
        y_min, y_max = float(ys.min()) - pad_m, float(ys.max()) + pad_m
        n = int(np.ceil(max(x_max - x_min, y_max - y_min) / pixel_m)) + 1
        return x_min, y_min, n

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
        weight = self.sample.projected_weight(x_min, y_min, n, pixel_m, level, self.params.z_exponent, damage)
        support_z, support_rho = ELEMENTS[self.sample.support_element][:2]
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
