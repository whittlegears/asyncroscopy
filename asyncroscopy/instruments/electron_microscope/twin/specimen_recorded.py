"""The ``recorded`` specimen: a sample taken from an image someone already acquired.

The image (``.npy``, ``.npz``, ``.png``, ``.tif``, ``.h5``) is laid on the
sample plane with its pixel size, centred on ``center_m``. Its intensity,
scaled to 0..1 between robust limits, stands in for the projected
Z^n-weighted mass thickness: ``contrast_weight`` is the weight of the
brightest pixel (the default gives about 30 % HAADF scattering, a 16 nm gold
particle). The twin then images it like any other specimen, so its noise,
dose, drift, scan distortion and defocus blur still come from the instrument
model; the twin cannot resolve detail finer than the recorded pixel.

Ground truth is unknown unless a labels file is given: JSON in
export_ground_truth's particle format, ``[{"center_m": [x, y],
"semi_axes_m": [a, b, c], "theta_rad": 0, "element": "Au"}, ...]`` (or
``{"particles": [...]}``; ``radius_m`` stands for ``semi_axes_m = [r, r,
r/2]``). Labelled particles also give EDS its composition and the dose map
something to damage. Without labels, ``composition`` (``{"Au": 1.0}``) spreads
an EDS signal in proportion to the image intensity; with neither, EDS sees
only the support film.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from .sample_model import ELEMENTS, ParticleTable
from .specimen import Specimen, SpecimenContext, check_options, register_specimen

ANGSTROM = 1e-10
HAADF_30_PERCENT_WEIGHT = 1.6e4  # -ln(0.7) / 2.25e-5: the default renderer's 30 % HAADF scattering
OPTIONS = {
    "path", "pixel_size_m", "dataset", "index", "labels_path", "center_m", "invert", "contrast_weight",
    "composition", "eds_areal_density", "support_thickness_m", "height_gradient", "seed",
}


def load_image(path: Path, dataset: str | None = None, index: int = 0) -> tuple[np.ndarray, float | None]:
    """(2D float image, pixel size in metres from the file or None)."""
    suffix = path.suffix.lower()
    pixel = None
    if suffix == ".npy":
        image = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            key = dataset or ("image" if "image" in data.files else next(k for k in data.files if data[k].ndim >= 2))
            image = data[key]
            if "pixel_size_m" in data.files:
                pixel = float(data["pixel_size_m"])
    elif suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as h5:
            if dataset is None:
                found: list[str] = []
                h5.visititems(lambda name, obj: found.append(name) if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 else None)
                if not found:
                    raise ValueError(f"{path} has no 2D dataset; give the specimen's dataset option")
                dataset = found[0]
            dset = h5[dataset]
            image = dset[index] if dset.ndim == 3 else dset[()]
            for attrs in (dset.attrs, h5.attrs):
                if "pixel_size_m" in attrs:
                    pixel = float(attrs["pixel_size_m"])
                    break
    else:
        from PIL import Image

        with Image.open(path) as img:
            image = np.asarray(img)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3 and image.shape[-1] in (3, 4):  # RGB(A): mean of the colour channels
        image = image[..., :3].mean(axis=-1)
    elif image.ndim == 3:
        image = image[index]
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError(f"{path}: need a 2D image, got shape {image.shape}")
    return image, pixel


class Labels(ParticleTable):
    """Labelled particles from a JSON file, in sample coordinates."""

    def __init__(self, entries: list[dict]) -> None:
        rows = []
        for k, e in enumerate(entries):
            if "semi_axes_m" in e:
                a, b, c = (float(v) for v in e["semi_axes_m"])
            else:
                r = float(e["radius_m"])
                a, b, c = r, r, r / 2.0
            element = e.get("element")
            if element is not None and element not in ELEMENTS:
                raise ValueError(f"label {k}: unknown element {element!r}; known: {sorted(ELEMENTS)}")
            rows.append((int(e.get("id", k)), [float(v) for v in e["center_m"][:2]], [a, b, c], float(e.get("theta_rad", 0.0)), element))
        n = len(rows)
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.centers = np.array([r[1] for r in rows], dtype=np.float64).reshape(n, 2)
        self.semi_axes = np.array([r[2] for r in rows], dtype=np.float64).reshape(n, 3)
        self.theta = np.array([r[3] for r in rows], dtype=np.float64)
        self.elements = [r[4] for r in rows]

    @classmethod
    def load(cls, path: Path) -> "Labels":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["particles"] if isinstance(data, dict) else data)


@register_specimen
class RecordedSpecimen(Specimen):
    source = "recorded"

    def __init__(
        self,
        image: np.ndarray,
        pixel_size_m: float,
        *,
        center_m: tuple[float, float] = (0.0, 0.0),
        invert: bool = False,
        contrast_weight: float = HAADF_30_PERCENT_WEIGHT,
        labels: Labels | None = None,
        composition: dict[str, float] | None = None,
        eds_areal_density: float = 5.0,
        support_thickness_m: float = 0.0,
        height_gradient: tuple[float, float] = (0.0, 0.0),
        path: str = "",
    ) -> None:
        if not pixel_size_m or pixel_size_m <= 0:
            raise ValueError("recorded specimen needs pixel_size_m > 0 (an option, or pixel_size_m in the file)")
        if contrast_weight <= 0:
            raise ValueError("contrast_weight must be > 0")
        unknown = [e for e in (composition or {}) if e not in ELEMENTS]
        if unknown:
            raise ValueError(f"recorded specimen composition: unknown elements {unknown}; known: {sorted(ELEMENTS)}")
        lo, hi = np.percentile(image, [0.5, 99.9])
        norm = np.clip((image - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(image)
        self.image = 1.0 - norm if invert else norm
        self.pixel_m = float(pixel_size_m)
        rows, cols = self.image.shape
        cx, cy = (float(v) for v in center_m)
        self.x0, self.y0 = cx - cols * self.pixel_m / 2.0, cy - rows * self.pixel_m / 2.0
        self.bounds = (self.x0, self.x0 + cols * self.pixel_m, self.y0, self.y0 + rows * self.pixel_m)
        self.contrast_weight = float(contrast_weight)
        self.labels = labels
        self.composition = {k: float(v) for k, v in (composition or {}).items()}
        self.eds_areal_density = float(eds_areal_density)
        self.support_thickness_m = float(support_thickness_m)
        self.height_gradient = tuple(float(v) for v in height_gradient)
        self.path = path
        self._smoothed: dict[float, np.ndarray] = {}

    @classmethod
    def from_options(cls, options: dict, context: SpecimenContext) -> "RecordedSpecimen":
        check_options(cls.source, options, OPTIONS)
        if not options.get("path"):
            raise ValueError("recorded specimen needs a path to the image (specimen_path, or path in specimen)")
        path = context.resolve(options["path"])
        image, file_pixel = load_image(path, options.get("dataset"), int(options.get("index", 0)))
        labels = Labels.load(context.resolve(options["labels_path"])) if options.get("labels_path") else None
        sample = context.sample
        return cls(
            image,
            float(options.get("pixel_size_m") or file_pixel or 0.0),
            center_m=options.get("center_m", (0.0, 0.0)),
            invert=bool(options.get("invert", False)),
            contrast_weight=float(options.get("contrast_weight", HAADF_30_PERCENT_WEIGHT)),
            labels=labels,
            composition=options.get("composition"),
            eds_areal_density=float(options.get("eds_areal_density", 5.0)),
            support_thickness_m=float(options.get("support_thickness_m", sample.support_thickness_m if sample else 0.0)),
            height_gradient=options.get("height_gradient", sample.height_gradient if sample else (0.0, 0.0)),
            path=str(options["path"]),
        )

    # ------------------------------------------------------------------
    def _image_for(self, pixel_m: float) -> np.ndarray:
        """The image, low-passed when the field's pixels are larger than the recorded ones (no aliasing)."""
        ratio = round(pixel_m / self.pixel_m, 2)
        if ratio <= 1.5:
            return self.image
        if ratio not in self._smoothed:
            if len(self._smoothed) > 8:
                self._smoothed.clear()
            self._smoothed[ratio] = ndimage.gaussian_filter(self.image, sigma=0.4 * ratio, mode="constant")
        return self._smoothed[ratio]

    def intensity(self, xs: np.ndarray, ys: np.ndarray, pixel_m: float | None = None) -> np.ndarray:
        """Normalised recorded intensity (0..1) at sample points; 0 off the image."""
        image = self._image_for(pixel_m or self.pixel_m)
        cols = (np.asarray(xs) - self.x0) / self.pixel_m - 0.5
        rows = (np.asarray(ys) - self.y0) / self.pixel_m - 0.5
        return ndimage.map_coordinates(image, [rows, cols], order=1, mode="constant", cval=0.0)

    def projected_weight(self, x_min: float, y_min: float, n: int, pixel_m: float, level: str, z_exponent: float, damage: dict[int, float]) -> np.ndarray:
        coords = x_min + (np.arange(n) + 0.5) * pixel_m
        ycoords = y_min + (np.arange(n) + 0.5) * pixel_m
        gx, gy = np.meshgrid(coords, ycoords)
        weight = self.intensity(gx, gy, pixel_m) * self.contrast_weight
        if damage and self.labels is not None:
            for i, left in damage.items():
                weight = np.where(self.labels.thickness_at(int(i), gx, gy) > 0, weight * left, weight)
        return weight

    def areal_density(self, x: float, y: float) -> dict[str, float]:
        density: dict[str, float] = {}
        if self.labels is not None:
            for i in self.labels.in_box(x, x, y, y):
                element = self.labels.elements[int(i)]
                t = float(self.labels.thickness_at(int(i), np.array(x), np.array(y)))
                if element and t > 0:
                    density[element] = density.get(element, 0.0) + ELEMENTS[element][1] * t / ANGSTROM
            return density
        if self.composition:
            level = float(self.intensity(np.array([x]), np.array([y]))[0])
            total = sum(self.composition.values()) or 1.0
            for element, fraction in self.composition.items():
                density[element] = self.eds_areal_density * level * fraction / total
        return density

    @property
    def feature_centers(self) -> np.ndarray:
        return self.labels.centers if self.labels is not None else super().feature_centers

    def in_box(self, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        return self.labels.in_box(x_min, x_max, y_min, y_max) if self.labels is not None else super().in_box(x_min, x_max, y_min, y_max)

    @property
    def element_symbols(self) -> tuple[str, ...]:
        labelled = [e for e in (self.labels.elements if self.labels is not None else []) if e]
        return tuple(sorted(set(labelled) | set(self.composition)))

    def ground_truth(self) -> dict:
        if self.labels is None:
            return {"known": False, "particles": []}
        return {"known": True, "particles": self.labels.particle_table()}

    def describe(self) -> dict:
        return {
            **super().describe(),
            "path": self.path,
            "shape": list(self.image.shape),
            "pixel_size_m": self.pixel_m,
            "labelled": self.labels is not None,
        }
