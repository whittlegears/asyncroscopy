"""The SEM map profile: the scanning-beam profile on a ``ScanningElectronMicroscope``.

It drives only ``ScanningBeamInstrument``'s commands and takes no per-tile spectra. Its detectors default to the
instrument's ``default_detector`` (secondary electrons). Registered through the ``asyncroscopy.map_profiles`` entry point.
"""

from __future__ import annotations

from asyncroscopy.instruments.scanning_beam_profile import ScanningBeamProfile
from asyncroscopy.jobs.mapping.profile import register_profile


@register_profile
class SEMProfile(ScanningBeamProfile):
    """A scanning electron microscope: stage moves, a blanker, an image per tile, electron dose; no EDS per tile."""

    name = "sem"
    doc = "SEM: scanned images per tile (detectors, default the instrument's default_detector; imsize, dwell_s), electron-dose budget; no per-tile EDS."
    job_kind = "map acquisition: a grid of SEM images over a stage box, stitched into a zoomable map when it finishes"
    instrument_types = ("SEM",)
