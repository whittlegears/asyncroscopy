"""A small get/move/acquire adapter on top of an ``ElectronMicroscope`` Tango device."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np


class TangoMicroscope:
    """Drives an ``ElectronMicroscope`` Tango device (twin or real) for the autonomy routines.

    *fetch_array(key)* turns the DATA/Tiled key an acquisition returns into
    the 2-D image array (the MCP server already knows how to do that).
    """

    def __init__(self, instrument, scan=None, fetch_array: Callable[[str], np.ndarray] | None = None,
                 detector: str = "haadf"):
        self.instrument = instrument
        self.scan = scan
        self.fetch_array = fetch_array
        self.detector = detector

    def get_stage(self) -> list[float]:
        return [float(v) for v in self.instrument.get_stage()]

    def move_stage(self, position) -> None:
        self.instrument.move_stage([float(v) for v in position])

    def current_fov(self) -> float | None:
        try:
            return float(self.instrument.get_fov())
        except Exception:
            return None

    def current_imsize(self) -> int | None:
        try:
            return int(self.scan.imsize) if self.scan is not None else None
        except Exception:
            return None

    def acquire(self, fov_m: float, imsize: int) -> np.ndarray:
        self.instrument.set_fov(float(fov_m))
        if self.scan is not None:
            self.scan.imsize = int(imsize)
        key = self.instrument.acquire_scanned_image([self.detector])
        if self.fetch_array is None:
            raise RuntimeError("no image fetcher configured: cannot read acquisitions back from DATA/Tiled")
        image = self.fetch_array(key)
        if image is None:
            raise RuntimeError(f"acquisition {key!r} has no 2-D image")
        return np.asarray(image)

    def settings(self) -> dict[str, Any]:
        out: dict[str, Any] = {"detector": self.detector, "fov_m": self.current_fov()}
        if self.scan is not None:
            for attr in ("imsize", "dwell_time"):
                try:
                    out[attr] = getattr(self.scan, attr)
                except Exception:
                    pass
        try:
            out["stage"] = self.get_stage()
        except Exception:
            pass
        try:
            out["parameters"] = json.loads(self.instrument.get_parameters())
        except Exception:
            pass
        try:
            out["instrument_class"] = self.instrument.info().dev_class
        except Exception:
            pass
        return out
