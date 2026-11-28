"""Three controllers to beat. They look at the same places in the same order and differ only in how they focus.

All three walk a serpentine grid with one image per map cell and, on each image, take a spectrum of a few particles
picked at random. What they learn is turned into an answer the same way (``answer_from``). The difference:

* ``raster``: autofocus once at the start and never again. All its time goes to images and spectra.
* ``periodic``: autofocus every few images, whether it is needed or not.
* ``surface``: remembers what defocus each autofocus run found and fits a smooth surface through those places.
  Where the surface reaches it sets the predicted defocus, which costs nothing, and it runs autofocus only where
  it has no measurement nearby. It is the physics-informed one: it knows a specimen's height changes smoothly.

None of them chooses where to look or where a spectrum is worth taking. That is left for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from benchmarks.hackathon.analysis import find_particles, identify, is_hole
from benchmarks.hackathon.task import GRID, Answer, BudgetSpent, Microscope, cell_centres, cell_of

SPECTRA_PER_IMAGE = 3  # taken straight after the image, before the sample drifts off the particles
PERIODIC_EVERY = 4  # the periodic controller focuses every this many images
SURFACE_REACH_M = 1.0e-6  # the surface controller trusts its model this far from the nearest place it focused
SURFACE_NEIGHBOURS = 10  # and fits its surface to this many of the nearest such places


class FocusOnce:
    def __init__(self) -> None:
        self.focused = False

    def at(self, microscope: Microscope, x_m: float, y_m: float) -> None:
        if not self.focused:
            microscope.autofocus()
            self.focused = True


class FocusPeriodically:
    def __init__(self, every: int = PERIODIC_EVERY) -> None:
        self.every, self.images = every, 0

    def at(self, microscope: Microscope, x_m: float, y_m: float) -> None:
        if self.images % self.every == 0:
            microscope.autofocus()
        self.images += 1


class FocusFromSurface:
    """Remembers where it ran autofocus and what defocus each place needed, and treats those as samples of one smooth
    surface. Near a remembered place it fits a curved surface (a quadratic in x and y) to the nearest ones and sets
    the defocus that predicts; far from all of them it runs autofocus and remembers the result.
    """

    def __init__(self) -> None:
        self.places: list[tuple[float, float]] = []
        self.defocus_m: list[float] = []

    def at(self, microscope: Microscope, x_m: float, y_m: float) -> None:
        places = np.array(self.places).reshape(-1, 2)
        distance = np.hypot(places[:, 0] - x_m, places[:, 1] - y_m)
        if len(places) < 6 or distance.min() > SURFACE_REACH_M:
            self.places.append((x_m, y_m))
            self.defocus_m.append(microscope.autofocus())
            return
        nearest = np.argsort(distance)[:SURFACE_NEIGHBOURS]
        dx, dy = (places[nearest, 0] - x_m) / SURFACE_REACH_M, (places[nearest, 1] - y_m) / SURFACE_REACH_M
        terms = np.stack([np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy], axis=1)
        fit, *_ = np.linalg.lstsq(terms, np.array(self.defocus_m)[nearest], rcond=None)
        microscope.set_defocus(fit[0])  # the surface's value here, where dx and dy are 0


@dataclass
class Notes:
    """What was learnt in each map cell."""

    visited: np.ndarray = field(default_factory=lambda: np.zeros((GRID, GRID), bool))
    hole: np.ndarray = field(default_factory=lambda: np.zeros((GRID, GRID), bool))
    radius_m: np.ndarray = field(default_factory=lambda: np.full((GRID, GRID), np.nan))  # mean particle radius
    identified: dict = field(default_factory=dict)  # element -> (GRID, GRID) count of particles identified as it


def serpentine() -> list[tuple[float, float]]:
    """The centre of every map cell, row by row, each row in the opposite direction to the one before."""
    centres = cell_centres()
    return [(x, y) for i, y in enumerate(centres) for x in (centres if i % 2 == 0 else centres[::-1])]


def survey(microscope: Microscope, focus, seed: int = 0) -> Notes:
    """Image every cell until the budget is spent, with a few spectra on each image."""
    rng = np.random.default_rng(seed)
    notes = Notes(identified={e: np.zeros((GRID, GRID)) for e in microscope.elements})
    try:
        for x_m, y_m in serpentine():
            microscope.move(x_m, y_m)
            focus.at(microscope, x_m, y_m)
            image = microscope.image()
            cell = cell_of(image.x_m, image.y_m)
            notes.visited[cell] = True
            notes.hole[cell] = is_hole(image)
            particles = find_particles(image)
            if not particles or notes.hole[cell]:
                continue
            notes.radius_m[cell] = np.mean([p.radius_m for p in particles])
            for k in rng.permutation(len(particles))[:SPECTRA_PER_IMAGE]:
                element = identify(microscope.spectrum(particles[k].column, particles[k].row), microscope.elements)
                if element is not None:
                    notes.identified[element][cell] += 1
    except BudgetSpent:
        pass
    return notes


def answer_from(notes: Notes) -> Answer:
    """The marker element, the size ratio and the two maps, from the notes.

    Three spectra say little about one cell, so each cell's shares pool the spectra of its 3 x 3 neighbourhood. The
    marker is the element whose share goes most strongly against particle size, and the size ratio comes from a
    straight line through radius against that share.
    """
    pooled = {e: ndimage.uniform_filter(counts, 3, mode="nearest") for e, counts in notes.identified.items()}
    total = sum(pooled.values())
    known = (total > 0) & np.isfinite(notes.radius_m)
    if known.sum() < 3:  # nothing learnt: answer what knows nothing
        return Answer(next(iter(pooled)), 1.0, np.full((GRID, GRID), 1.0 / len(pooled)), notes.hole)
    share = {e: np.where(total > 0, pooled[e] / np.maximum(total, 1e-12), np.nan) for e in pooled}
    against_size = {e: np.corrcoef(share[e][known], notes.radius_m[known])[0, 1] for e in pooled}
    marker = min(against_size, key=lambda e: against_size[e] if np.isfinite(against_size[e]) else 1.0)
    slope, radius_without_marker = np.polyfit(share[marker][known], notes.radius_m[known], 1)
    size_ratio = float(np.clip((radius_without_marker + slope) / radius_without_marker, 0.05, 1.5))
    share_map = np.where(np.isfinite(share[marker]), share[marker], np.nanmean(share[marker]))
    return Answer(marker, size_ratio, share_map, notes.hole)


def raster(microscope: Microscope) -> Answer:
    return answer_from(survey(microscope, FocusOnce()))


def periodic(microscope: Microscope) -> Answer:
    return answer_from(survey(microscope, FocusPeriodically()))


def surface(microscope: Microscope) -> Answer:
    return answer_from(survey(microscope, FocusFromSurface()))


BASELINES = {"raster": raster, "periodic": periodic, "surface": surface}
