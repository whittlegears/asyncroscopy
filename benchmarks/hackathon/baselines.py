"""Three controllers to beat. They look at the same places in the same order and differ only in how they focus.

All three walk a serpentine grid with one image per map cell and, on each image, take a spectrum of a few particles
picked at random. What they learn is turned into an answer the same way (``answer_from``). The difference:

* ``raster``: autofocus once at the start and never again. All its time goes to images and spectra.
* ``periodic``: autofocus every few images, whether it is needed or not.
* ``kalman``: keeps an estimate of the focus each place needs and how sure it is (a two-number Kalman filter: the
  defocus here and its slope along the row). Between autofocus runs it sets the predicted defocus, which costs
  nothing, and it runs autofocus only when it is no longer sure enough.

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
FOCUS_TOLERANCE_M = 4.0e-8  # the kalman controller focuses when its defocus estimate is less sure than this
SLOPE_PRIOR = 0.1  # how steep the specimen may be (height change per metre moved) before anything is known
SLOPE_CHANGE_PER_M = 1.0e5  # how fast that slope may change with distance (per metre, per metre)
AUTOFOCUS_ERROR_M = 2.0e-9  # how well one autofocus run finds the focus


# ----------------------------------------------------------------------------------------------------------------------
# How each controller focuses
# ----------------------------------------------------------------------------------------------------------------------
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


class FocusByKalmanFilter:
    """Tracks [the defocus this place needs, its slope along x] and their covariance.

    Moving along a row carries the defocus forward by slope * distance and makes both less certain. Stepping to the
    next row keeps the estimate but adds the unknown slope along y to its uncertainty. An autofocus run is a
    measurement of the defocus, and through the covariance it also corrects the slope.
    """

    def __init__(self) -> None:
        self.state = np.zeros(2)  # [defocus (m), slope along x]
        self.covariance = np.diag([1.0, SLOPE_PRIOR**2])  # the first image always gets an autofocus
        self.last_xy: tuple[float, float] | None = None

    def _predict(self, x_m: float, y_m: float) -> None:
        if self.last_xy is None:
            return
        dx, dy = x_m - self.last_xy[0], y_m - self.last_xy[1]
        step = np.array([[1.0, dx], [0.0, 1.0]])
        self.state = step @ self.state
        self.covariance = step @ self.covariance @ step.T
        self.covariance[0, 0] += (SLOPE_PRIOR * dy) ** 2
        self.covariance[1, 1] += (SLOPE_CHANGE_PER_M * np.hypot(dx, dy)) ** 2

    def _correct(self, measured_defocus_m: float) -> None:
        gain = self.covariance[:, 0] / (self.covariance[0, 0] + AUTOFOCUS_ERROR_M**2)
        self.state = self.state + gain * (measured_defocus_m - self.state[0])
        self.covariance = self.covariance - np.outer(gain, self.covariance[0, :])

    def at(self, microscope: Microscope, x_m: float, y_m: float) -> None:
        self._predict(x_m, y_m)
        self.last_xy = (x_m, y_m)
        if self.covariance[0, 0] > FOCUS_TOLERANCE_M**2:
            self._correct(microscope.autofocus())
        else:
            microscope.set_defocus(self.state[0])


# ----------------------------------------------------------------------------------------------------------------------
# What all three do with the microscope
# ----------------------------------------------------------------------------------------------------------------------
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


def kalman(microscope: Microscope) -> Answer:
    return answer_from(survey(microscope, FocusByKalmanFilter()))


BASELINES = {"raster": raster, "periodic": periodic, "kalman": kalman}
