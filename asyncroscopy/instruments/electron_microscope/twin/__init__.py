"""Models behind the STEM digital twin.

The Tango device (``digital_twin.py``) stays a thin shell; everything that
decides what an acquisition looks like lives here so it can be tested without
Tango. Coordinates are metres in the sample frame. Images are indexed
``[row, column]`` with row increasing along +y and column along +x.

The modality-free primitives (the clock, random streams, stage model and dose
map) are in ``asyncroscopy.sim``. The names below load on first use, so importing
one of them does not pull in pyTEMlib.
"""

import importlib

_EXPORTS = {
    "EdsParams": ".eds_model",
    "SampleModel": ".sample_model",
    "SampleParams": ".sample_model",
    "ScanParams": ".scan_distortion",
    "Specimen": ".specimen",
    "SpecimenContext": ".specimen",
    "build_specimen": ".specimen",
    "register_specimen": ".specimen",
    "specimen_sources": ".specimen",
    "pixel_sample_coordinates": ".scan_distortion",
    "point_spectrum": ".eds_model",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
