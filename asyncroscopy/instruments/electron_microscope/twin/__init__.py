"""Models behind the STEM digital twin.

The Tango device (``digital_twin.py``) stays a thin shell; everything that
decides what an acquisition looks like lives here so it can be tested without
Tango. Coordinates are metres in the sample frame. Images are indexed
``[row, column]`` with row increasing along +y and column along +x.
"""

from .clock import SimClock
from .dose import DoseMap
from .eds_model import EdsParams, point_spectrum
from .rng import EventRng
from .sample_model import SampleModel, SampleParams
from .scan_distortion import ScanParams, pixel_sample_coordinates
from .stage_model import StageModel, StageParams

__all__ = [
    "DoseMap",
    "EdsParams",
    "EventRng",
    "SampleModel",
    "SampleParams",
    "ScanParams",
    "SimClock",
    "StageModel",
    "StageParams",
    "pixel_sample_coordinates",
    "point_spectrum",
]
