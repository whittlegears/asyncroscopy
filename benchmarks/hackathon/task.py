"""The hackathon task: with a fixed time budget, learn what a sample is made of and where.

The sample (``configs/digital_twin_heterogeneous.yaml``) is cut into regions that differ in particle density, film
thickness, element mix and particle size; some are holes. One element, the *marker*, sets the particle size: the
larger its share of a region's particles, the smaller they are. Nobody tells the controller which element it is.

A controller gets a ``Microscope`` and must hand back an ``Answer``:

* ``marker_element``: which element sets the size;
* ``size_ratio``: the radius of particles where every particle is the marker, divided by the radius where none is;
* ``share_map``: the marker's share of the particles in each cell of a GRID x GRID map of the sample (0 to 1);
* ``hole_map``: which of those cells are holes.

Everything the microscope does costs simulated time: moving, imaging, spectra and autofocus. When the budget is
spent the microscope refuses, so a controller must choose where to look, where a spectrum is worth its time and when
to focus again. The specimen is not flat and the focus drifts, so an image taken without thought is often blurred.

``score.py`` marks the answer against the twin's ground truth, which the controller never sees. ``baselines.py``
has three controllers to beat. A controller is any function ``controller(microscope) -> Answer``.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "digital_twin_heterogeneous.yaml"
DEVICE = "asyncroscopy/digitaltwin/hackathon"

GRID = 16  # the answer's maps are GRID x GRID cells over the area
AREA_M = 8.0e-6  # the square area to map, centred on the stage origin (the whole sample)
FOV_M = 4.0e-7  # every image is this wide
IMSIZE = 256  # and this many pixels across
DWELL_S = 1.0e-6
SPECTRUM_S = 0.5  # exposure of one EDS spectrum
BUDGET_S = 700.0  # simulated seconds a controller may spend: not enough to image every cell and focus often
ELEMENTS = ("Au", "Pt", "Fe", "Pd")  # what the particles may be made of (the configuration's particle_elements)


class BudgetSpent(RuntimeError):
    """The time budget is used up; the microscope does nothing more."""


@dataclass
class Image:
    pixels: np.ndarray  # raw counts, [row, column]; rows run along +y and columns along +x
    x_m: float  # where the stage says the image is centred
    y_m: float
    pixel_m: float


@dataclass
class Spectrum:
    energies_ev: np.ndarray
    counts: np.ndarray


@dataclass
class Answer:
    marker_element: str
    size_ratio: float
    share_map: np.ndarray  # (GRID, GRID) floats in 0..1, [row, column] like an image
    hole_map: np.ndarray  # (GRID, GRID) booleans


def cell_of(x_m: float, y_m: float) -> tuple[int, int]:
    """The (row, column) of the map cell holding a stage point; points outside the area go to the nearest cell."""
    col = int(np.clip((x_m + AREA_M / 2) / AREA_M * GRID, 0, GRID - 1))
    row = int(np.clip((y_m + AREA_M / 2) / AREA_M * GRID, 0, GRID - 1))
    return row, col


def cell_centres() -> np.ndarray:
    """The stage coordinate of each cell's centre along one axis (the same for x and y)."""
    return (np.arange(GRID) + 0.5) / GRID * AREA_M - AREA_M / 2


class Microscope:
    """What a controller may do. Every action spends simulated time; ``remaining_s`` says how much is left."""

    def __init__(self, device, elements: tuple[str, ...], budget_s: float = BUDGET_S) -> None:
        self._device = device
        self._saved: list = []  # what the twin's last acquisition wrote
        self._start_s = device._clock.now()
        self.elements = elements  # the elements the particles may be made of
        self.budget_s = float(budget_s)
        self.autofocus_runs = 0
        self.images_taken: list[Image] = []  # every image so far, for drawing the survey afterwards

    @property
    def used_s(self) -> float:
        return self._device._clock.now() - self._start_s

    @property
    def remaining_s(self) -> float:
        return self.budget_s - self.used_s

    def _spend(self) -> None:
        if self.remaining_s <= 0:
            raise BudgetSpent(f"the {self.budget_s:.0f} s budget is spent")

    def move(self, x_m: float, y_m: float) -> None:
        """Move the stage to (x, y), kept inside the area. The stage does not land exactly where asked, and the
        sample keeps drifting for some seconds after it stops."""
        self._spend()
        limit = AREA_M / 2 - FOV_M / 2
        self._device._move_stage([float(np.clip(x_m, -limit, limit)), float(np.clip(y_m, -limit, limit)), 0.0, 0.0, 0.0])

    def image(self) -> Image:
        """One HAADF image where the stage is."""
        self._spend()
        self._device._acquire_scanned_image(IMSIZE, DWELL_S, ["haadf"])
        stage = self._device._get_stage()
        image = Image(np.asarray(self._saved[-1], dtype=np.float32), float(stage[0]), float(stage[1]), FOV_M / IMSIZE)
        self.images_taken.append(image)
        return image

    def spectrum(self, column: float, row: float) -> Spectrum:
        """One EDS spectrum with the beam parked at pixel (column, row) of the last image."""
        self._spend()
        self._device._place_beam([float(np.clip((column + 0.5) / IMSIZE, 0.0, 1.0)), float(np.clip((row + 0.5) / IMSIZE, 0.0, 1.0))])
        self._device._acquire_spectrum("eds", SPECTRUM_S)
        counts, attrs = self._saved[-1]
        energies = attrs["energy_offset_ev"] + attrs["energy_dispersion_ev"] * np.arange(len(counts))
        return Spectrum(energies, np.asarray(counts, dtype=np.float64))

    def autofocus(self) -> float:
        """Run the instrument's autofocus here (it takes several frames) and return the defocus it settled on (m)."""
        self._spend()
        self._device._auto_focus()
        self.autofocus_runs += 1
        return self.defocus_m

    @property
    def defocus_m(self) -> float:
        return float(self._device._get_defocus())

    def set_defocus(self, defocus_m: float) -> None:
        """Set the defocus directly. It costs no time, so a controller that can predict the focus need not measure it."""
        self._device._set_defocus(float(defocus_m))


@contextlib.contextmanager
def twin() -> Iterator:
    """The task's digital twin, running in this process (no servers to start). Tango allows one per Python process."""
    import tango

    from asyncroscopy.instruments.electron_microscope import digital_twin as digital_twin_module
    from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
    from asyncroscopy.testing import instrument_context
    from benchmarks.mosaic_twin import twin_and_stage, twin_properties

    (ROOT / "outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as save_dir:
        devices = twin_and_stage(DigitalTwin, DEVICE, twin_properties(CONFIG, Path(save_dir)), CONFIG)
        with instrument_context(devices, startup_timeout_s=120.0) as context:
            tango.DeviceProxy(context.get_device_access(DEVICE)).ping()
            device = tango.Util.instance().get_device_by_name(DEVICE)
            device.hackathon_module = digital_twin_module  # where acquisitions are caught, see start()
            yield device


def start(device, seed: int, budget_s: float = BUDGET_S) -> Microscope:
    """A fresh sample and a fresh instrument for ``seed``, and the microscope a controller drives."""
    # An in-process Tango context cuts a property at its first comma, so the element list is given here.
    device.particle_elements = ",".join(ELEMENTS)
    device._generate_sample(int(seed))
    device._blanked = False
    device._set_fov(FOV_M)
    microscope = Microscope(device, tuple(device._specimen.element_symbols), budget_s)

    def keep(_device, _data_server, kind, _detectors, arrays, dataset_attrs=None, **_ignored) -> str:
        """Catch what the twin would write to disk: an image's pixels, or a spectrum's counts and energy axis."""
        microscope._saved.append(np.asarray(arrays[0]) if kind == "stem_image" else (np.asarray(arrays), dataset_attrs))
        del microscope._saved[:-1]
        return f"{kind}-{seed}"

    device.hackathon_module.save_acquisition = keep
    return microscope


def ground_truth(device) -> dict:
    """The sample's truth, for the scorer only: the regions, the marker element and the particles."""
    truth = json.loads(device.export_ground_truth())
    return {**truth["specimen"], "particles": truth["particles"]}
