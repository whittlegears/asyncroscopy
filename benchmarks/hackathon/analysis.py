"""Simple image and spectrum analysis the baselines use. Teams may use it, improve it or replace it.

Nothing here reads the ground truth: it works from an ``Image`` and a ``Spectrum`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from benchmarks.hackathon.task import Image, Spectrum

LINE_EV = {"Si": 1740.0, "Fe": 6404.0, "Ni": 7478.0, "Cu": 8048.0, "Pd": 2839.0, "Ag": 2984.0, "Pt": 9442.0, "Au": 9713.0}
LINE_HALF_WIDTH_EV = 100.0  # counts within this of the line are the element's
MIN_LINE_COUNTS = 40  # fewer than this in the best line: the beam was not on a particle
MIN_LINE_LEAD = 3.0  # and the best line must have this many times the counts of the next

MIN_PARTICLE_COUNTS = 6.0  # a particle's smoothed peak stands at least this many counts above the film
MIN_PARTICLE_PIXELS = 6
HOLE_VARIANCE = 0.5  # the film's shot noise makes a film image noisier than this; a hole shows only read noise


@dataclass
class Particle:
    column: float  # where it is in the image (pixels)
    row: float
    radius_m: float  # from the area above half its peak, which is about 0.75 of the true radius when in focus


def find_particles(image: Image) -> list[Particle]:
    """The bright blobs of an image and a size for each."""
    film = np.median(image.pixels)
    smooth = ndimage.gaussian_filter(image.pixels, 1.0)
    labels, _count = ndimage.label(smooth > film + MIN_PARTICLE_COUNTS / 2.0)
    particles = []
    for number, blob in enumerate(ndimage.find_objects(labels), start=1):
        inside = labels[blob] == number
        signal = np.where(inside, smooth[blob] - film, 0.0)
        if inside.sum() < MIN_PARTICLE_PIXELS or signal.max() < MIN_PARTICLE_COUNTS:
            continue
        above_half = signal > signal.max() / 2.0
        rows, columns = np.nonzero(above_half)
        particles.append(Particle(blob[1].start + columns.mean(), blob[0].start + rows.mean(), float(np.sqrt(above_half.sum() / np.pi)) * image.pixel_m))
    return particles


def is_hole(image: Image) -> bool:
    """Whether the image shows no film: only the detector's read noise, however its offset has drifted."""
    darker_half = image.pixels[image.pixels <= np.median(image.pixels)]
    return bool(image.pixels.var() < HOLE_VARIANCE and darker_half.var() < HOLE_VARIANCE)


def identify(spectrum: Spectrum, elements: tuple[str, ...]) -> str | None:
    """The element under the beam: the one whose line holds the most counts, or None when no line stands out
    (the beam missed the particle, which happens when the sample has drifted since the image)."""
    counts = {e: float(spectrum.counts[np.abs(spectrum.energies_ev - LINE_EV[e]) < LINE_HALF_WIDTH_EV].sum()) for e in elements}
    best, runner_up = sorted(counts, key=counts.get, reverse=True)[:2]
    if counts[best] < MIN_LINE_COUNTS or counts[best] < MIN_LINE_LEAD * counts[runner_up]:
        return None
    return best
