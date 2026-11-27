"""Regions that make a procedural sample differ from place to place.

A uniform sample gives a controller nothing to decide: every tile costs the same and is worth the same. With
``region_count`` above 0 the sample is cut into that many regions (each point belongs to its nearest region seed),
and every region has its own:

* particle density (0 in a hole), so some tiles are crowded and some are empty;
* support film thickness (no film at all in a hole), so contrast and registration differ;
* element mix, so what a spectrum finds depends on where it is taken;
* particle size, tied to the element mix by a hidden rule (below).

The hidden rule is something for a workflow to discover. One element, the ``marker``, is picked per sample. The
larger its share of a region's particles, the smaller that region's particles are::

    radius factor = LARGEST_RADIUS - (LARGEST_RADIUS - SMALLEST_RADIUS) * marker share

Images show the sizes and spectra show the shares, so only a workflow that uses both can find the rule. The rule and
every region's values are in the ground truth, which agents never see.

Regions have their own seeded stream, so turning them on does not change any other random number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SMALLEST_RADIUS = 0.5  # radius factor of a region whose particles are all the marker element
LARGEST_RADIUS = 1.5  # radius factor of a region with none of it
DENSITY_SPREAD = 0.8  # standard deviation of the log of a region's particle density
FILM_CHANGE_RANGE = (-0.4, 1.0)  # a region's film is between 40 % thinner and 100 % thicker than nominal
ELEMENT_MIX_EVENNESS = 0.5  # below 1, most regions are dominated by one or two elements
HOLE = -1.0  # the film change that means there is no film


@dataclass
class Regions:
    """``count`` regions over a square sample, each row of every array describing one region."""

    seeds: np.ndarray  # (count, 2) the point each region grows from (m)
    density: np.ndarray  # (count,) relative particle density, 0 in a hole
    film: np.ndarray  # (count,) relative change of the support film's thickness, HOLE where there is none
    radius: np.ndarray  # (count,) factor on the particle radius
    element_shares: np.ndarray  # (count, elements) share of each element among the region's particles
    elements: tuple[str, ...]
    marker: str  # the element whose share sets the particle size

    @classmethod
    def draw(cls, rng: np.random.Generator, count: int, half_m: float, elements: tuple[str, ...], hole_fraction: float) -> "Regions":
        """Draw ``count`` regions on a sample reaching from -half_m to +half_m; ``hole_fraction`` of them are holes."""
        if count < 1 or not 0.0 <= hole_fraction < 1.0:
            raise ValueError("region_count must be >= 1 and region_hole_fraction in [0, 1)")
        seeds = rng.uniform(-half_m, half_m, size=(count, 2))
        density = np.exp(rng.normal(0.0, DENSITY_SPREAD, count))
        film = rng.uniform(*FILM_CHANGE_RANGE, count)
        element_shares = rng.dirichlet(np.full(len(elements), ELEMENT_MIX_EVENNESS), count)
        marker = int(rng.integers(len(elements)))
        radius = LARGEST_RADIUS - (LARGEST_RADIUS - SMALLEST_RADIUS) * element_shares[:, marker]
        holes = rng.random(count) < hole_fraction
        holes[0] = False  # at least one region always has film and particles
        density[holes], film[holes] = 0.0, HOLE
        return cls(seeds, density, film, radius, element_shares, tuple(elements), elements[marker])

    def index_at(self, x: float, y: float) -> int:
        """The region the sample point (x, y) belongs to: the one with the nearest seed."""
        return int(np.argmin((self.seeds[:, 0] - x) ** 2 + (self.seeds[:, 1] - y) ** 2))

    def index_on_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The region of every point of a grid with columns at ``xs`` and rows at ``ys``, shape (rows, columns).

        Most frames lie inside one region, so only the seeds that can win somewhere on the grid are compared: a seed
        further from the grid's centre than the nearest seed plus the grid's diagonal is nearest to none of its points.
        """
        centre_x, centre_y = float(np.mean(xs)), float(np.mean(ys))
        to_centre = np.hypot(self.seeds[:, 0] - centre_x, self.seeds[:, 1] - centre_y)
        diagonal = float(np.hypot(np.ptp(xs), np.ptp(ys)))
        near = np.nonzero(to_centre <= to_centre.min() + diagonal)[0]
        if len(near) == 1:
            return np.full((len(ys), len(xs)), near[0], dtype=np.int64)
        distance2 = (xs[None, :, None] - self.seeds[near, 0]) ** 2 + (ys[:, None, None] - self.seeds[near, 1]) ** 2
        return near[np.argmin(distance2, axis=2)]

    def ground_truth(self) -> dict:
        """The regions and the hidden rule, for evaluators."""
        return {
            "marker_element": self.marker,
            "rule": f"radius factor = {LARGEST_RADIUS} - {LARGEST_RADIUS - SMALLEST_RADIUS} * share of {self.marker}",
            "regions": [
                {
                    "id": i,
                    "seed_m": [float(v) for v in self.seeds[i]],
                    "hole": bool(self.film[i] <= HOLE),
                    "relative_density": float(self.density[i]),
                    "film_change": float(self.film[i]),
                    "radius_factor": float(self.radius[i]),
                    "element_shares": {e: float(s) for e, s in zip(self.elements, self.element_shares[i])},
                }
                for i in range(len(self.seeds))
            ],
        }
