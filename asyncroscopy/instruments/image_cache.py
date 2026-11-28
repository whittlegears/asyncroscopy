"""The latest images an instrument saved, kept in memory for ``get_image_data_cached``: ``ImageCache``.

The DATA writer (``asyncroscopy.data.data_writer.save_acquisition``) offers every array it saves to the saving
device's ``image_cache``. Only 2-D arrays are kept (an image, a diffraction pattern, one detector of a scan): a
spectrum, a camera series or a 4D-STEM cube is not an image and would not fit in a bounded cache. Each detector's
image is one entry, so a two-detector scan fills two. Entry 0 is the newest; past ``size`` the oldest is dropped.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Iterable

import numpy as np


class ImageCache:
    """At most ``size`` of the latest saved 2-D images, newest first."""

    def __init__(self, size: int) -> None:
        if int(size) < 1:
            raise ValueError(f"image_cache_size must be at least 1, got {size}")
        self._images: deque[dict] = deque(maxlen=int(size))
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return int(self._images.maxlen)

    def __len__(self) -> int:
        return len(self._images)

    def add(self, acquisition_type: str, saved: Iterable[tuple[str, str, Any]]) -> None:
        """Keep each (key, detector, array) of one saved acquisition whose array is 2-D."""
        for key, detector, source in saved:
            array = source.data if hasattr(source, "data") and not isinstance(source, np.ndarray) else source
            if np.ndim(array) != 2:
                continue
            image = np.array(array, copy=True)
            with self._lock:
                self._images.appendleft({"key": str(key), "acquisition_type": str(acquisition_type), "detector": str(detector), "image": image})

    def get(self, index: int) -> dict:
        """Entry ``index`` (0 is the newest): its key, acquisition_type, detector and image."""
        with self._lock:
            count = len(self._images)
            if count == 0:
                raise LookupError("no image has been saved since this device started")
            if not 0 <= int(index) < count:
                raise IndexError(f"index {int(index)} is out of range: {count} images are cached, 0 is the newest and {count - 1} the oldest")
            return dict(self._images[int(index)])
