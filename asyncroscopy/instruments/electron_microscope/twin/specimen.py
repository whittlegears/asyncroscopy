"""What the instrument model needs from a sample, and where samples come from.

The twin is split in two. The *instrument* (stage, drift, clock, scan
distortion, optics, detector noise, dose bookkeeping, EDS detector) never
looks inside the sample; it asks a ``Specimen`` these questions, all in sample
coordinates (metres, rows along +y, columns along +x):

* ``projected_weight``: the Z^n-weighted mass thickness on a regular grid,
  which the renderer turns into a detector signal and blurs with the probe;
* ``areal_density``: atoms per square angstrom of each element under the
  beam, which the EDS model turns into a spectrum;
* ``height_at``: the specimen height, which adds to the defocus;
* ``support_texture``: how the support film's thickness varies (None: a uniform film);
* ``in_box`` / ``feature_centers``: the features the dose map may damage.

``ground_truth`` is for evaluators only (export_ground_truth, scorers); the
instrument never reads it. A support film (``support_element``,
``support_thickness_m``) belongs to the specimen; contamination belongs to
the instrument, because the beam deposits it.

Sources are registered by name like the Tuner's strategies and the Mapper's
profiles::

    @register_specimen
    class MySpecimen(Specimen):
        source = "mine"

        @classmethod
        def from_options(cls, options, context): ...

and chosen with the twin's ``specimen`` device property. Built in:
``procedural`` (the seeded particle table, the default), ``recorded`` (an
image file) and ``crystal`` (a polycrystalline film built from ASE unit
cells); others install with the ``asyncroscopy.specimens`` entry point.
docs/Operation/specimens.md explains the coupling and walks through a source.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .rng import EventRng

if TYPE_CHECKING:
    from .sample_model import SampleParams

REPO_ROOT = Path(__file__).resolve().parents[4]


class Specimen:
    """A sample the twin can image. Subclasses fill in the geometry; the defaults are an empty specimen."""

    source = ""  # registry name; the twin's ``specimen`` property selects it
    support_element = "C"
    support_thickness_m = 0.0
    height_gradient: tuple[float, float] = (0.0, 0.0)
    bounds: tuple[float, float, float, float] = (-1e-9, 1e-9, -1e-9, 1e-9)  # x_min, x_max, y_min, y_max (m)

    @classmethod
    def from_options(cls, options: dict, context: "SpecimenContext") -> "Specimen":
        raise NotImplementedError

    # ------------------------------------------------------------------
    # What the instrument model reads
    # ------------------------------------------------------------------
    def projected_weight(self, x_min: float, y_min: float, n: int, pixel_m: float, level: str, z_exponent: float, damage: dict[int, float]) -> np.ndarray:
        """sum(Z^n * atoms per cubic angstrom * thickness in angstrom) on an n x n grid.

        Pixel (row r, column c) is centred at (x_min + (c + 0.5) * pixel_m, y_min + (r + 0.5) * pixel_m).
        ``level`` is "survey" (pixels larger than an atom) or "atomic"; ``damage`` maps a feature index
        (from ``in_box``) to the fraction of it left. The support film is added by the renderer.
        """
        return np.zeros((n, n), dtype=np.float64)

    def support_texture(self, x_min: float, y_min: float, n: int, pixel_m: float) -> np.ndarray | None:
        """The support film's relative thickness change on the same grid as ``projected_weight`` (0: the nominal
        ``support_thickness_m``, never below -1), or None for a uniform film."""
        return None

    def areal_density(self, x: float, y: float) -> dict[str, float]:
        """Atoms per square angstrom of each element along the beam at (x, y), support film excluded."""
        return {}

    def height_at(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        gx, gy = self.height_gradient
        return gx * np.asarray(x) + gy * np.asarray(y)

    @property
    def feature_centers(self) -> np.ndarray:
        """(N, 2) centres of the features the dose map tracks, indexed as ``in_box`` returns them."""
        return np.zeros((0, 2), dtype=np.float64)

    def in_box(self, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        """Indices of features overlapping the box."""
        return np.zeros(0, dtype=np.int64)

    @property
    def element_symbols(self) -> tuple[str, ...]:
        """The elements the specimen may contain (EDS and status reports)."""
        return ()

    def warnings(self) -> list[str]:
        """Problems worth a warning at start-up (the twin logs them)."""
        return []

    # ------------------------------------------------------------------
    # What evaluators read
    # ------------------------------------------------------------------
    def ground_truth(self) -> dict:
        """{"known": bool, "particles": [...], ...}: what a scorer may compare against; never shown to an agent."""
        return {"known": False, "particles": []}

    def describe(self) -> dict:
        """Short, JSON-safe summary: the source and its main settings."""
        return {"source": self.source, "bounds_m": [float(v) for v in self.bounds]}


@dataclass
class SpecimenContext:
    """What a source may take from the twin: its seed and the flat sample properties as defaults."""

    rng: EventRng
    sample: "SampleParams | None" = None  # the twin's flat sample properties; None: the SampleParams defaults
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve(self, path: str | Path) -> Path:
        """A data path as written in a config: absolute, or relative to the working directory or the repository."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        for root in (self.base_dir, REPO_ROOT):
            if (root / p).exists():
                return root / p
        return self.base_dir / p

    def rng_for(self, options: dict) -> EventRng:
        """A source's own seed (``seed`` option) gives another sample on the same instrument."""
        return EventRng(int(options["seed"])) if options.get("seed") is not None else self.rng


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
SPECIMENS: dict[str, type[Specimen]] = {}
BUILT_IN = {
    "procedural": "asyncroscopy.instruments.electron_microscope.twin.sample_model",
    "recorded": "asyncroscopy.instruments.electron_microscope.twin.specimen_recorded",
    "crystal": "asyncroscopy.instruments.electron_microscope.twin.specimen_crystal",
}
_entry_points_loaded = False


def register_specimen(cls: type[Specimen]) -> type[Specimen]:
    """Make ``cls`` available as ``specimen: {source: cls.source}``; usable as a class decorator."""
    if not cls.source:
        raise ValueError(f"{cls.__name__} needs a source name")
    SPECIMENS[cls.source] = cls
    return cls


def specimen_sources() -> dict[str, type[Specimen]]:
    """Every source: built in, registered, and installed with the ``asyncroscopy.specimens`` entry point."""
    global _entry_points_loaded
    for module in set(BUILT_IN.values()):
        importlib.import_module(module)
    if not _entry_points_loaded:
        _entry_points_loaded = True
        from importlib.metadata import entry_points

        for point in entry_points(group="asyncroscopy.specimens"):
            try:
                register_specimen(point.load())
            except Exception as exc:  # a broken plug-in must not take the twin down
                print(f"[asyncroscopy.specimens] could not load specimen {point.name!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return SPECIMENS


def get_specimen_class(source: str) -> type[Specimen]:
    """The class registered as ``source``; ValueError listing the known ones otherwise."""
    if source not in SPECIMENS and source in BUILT_IN:
        importlib.import_module(BUILT_IN[source])
    if source not in SPECIMENS:
        known = specimen_sources()
        if source not in known:
            raise ValueError(f"unknown specimen source {source!r}; registered: {sorted(known)}")
    return SPECIMENS[source]


def build_specimen(options: dict, context: SpecimenContext) -> Specimen:
    """Build the specimen ``options["source"]`` (default procedural) describes."""
    options = dict(options)
    source = str(options.pop("source", "") or "procedural")
    return get_specimen_class(source).from_options(options, context)


def parse_specimen(text: str) -> dict:
    """The ``specimen`` property as a dict: JSON, a YAML or Python mapping, or a bare source name.

    run_servers passes a YAML mapping under ``properties:`` as Python's repr of the dict, so both
    ``{"source": "recorded"}`` and ``{'source': 'recorded'}`` are accepted.
    """
    text = (text or "").strip()
    if not text:
        return {}
    value = None
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(text)
            break
        except (ValueError, SyntaxError):
            continue
    if value is None:
        import yaml

        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"specimen property is not JSON, YAML or a source name: {text!r} ({exc})") from exc
    if isinstance(value, str):
        return {"source": value}
    if not isinstance(value, dict):
        raise ValueError(f"specimen property must be a mapping like {{source: recorded, ...}}, got {text!r}")
    return value


def check_options(source: str, options: dict, accepted: Iterable[str]) -> None:
    """Refuse an option the source does not know, naming the ones it does (a typo must not pass silently)."""
    unknown = sorted(set(options) - set(accepted))
    if unknown:
        raise ValueError(f"specimen source {source!r} has no option(s) {unknown}; it takes {sorted(accepted)}")
