"""What an evaluator reads: a run's trace events, the data they reference, and (on a twin) the ground truth.

``load_run(run_id)`` reads the run's events from the Instrument Trace (``asyncroscopy.trace``);
images are read lazily, the first time an evaluator asks, through the existing data readers
(``asyncroscopy.data.data_reader.read_acquisition``: a local HDF5 file, or a Tiled key when a
Tiled URI is given). Ground truth is optional and only exists on a twin: the twin's
``export_ground_truth()`` document, which a bench or test passes in. Evaluators that use it say
so in their reasons; without it they fall back to a measurement on the image.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from asyncroscopy.routines.dose import ELECTRON_CHARGE

DEFAULT_TRACE_DIR = "outputs/trace"
KEY_FIELDS = ("key", "image_key", "spectrum_key", "result_key")


@dataclass
class Image:
    """One 2-D dataset of an acquisition, with what the evaluators need to place and weigh it."""

    key: str
    name: str
    array: np.ndarray
    attrs: dict

    @property
    def pixel_m(self) -> float | None:
        value = self.attrs.get("pixel_size_m")
        if value is None and self.attrs.get("fov_m"):
            value = float(self.attrs["fov_m"]) / max(self.array.shape)
        return None if value is None else float(value)

    @property
    def fov_m(self) -> tuple[float, float] | None:
        pixel = self.pixel_m
        return None if pixel is None else (pixel * self.array.shape[1], pixel * self.array.shape[0])

    @property
    def index(self) -> int | None:
        value = self.attrs.get("acquisition_index")
        return None if value is None else int(value)

    def frame_dose_e_per_a2(self) -> float | None:
        """Electrons per square angstrom this frame put on its field: current x dwell / pixel area."""
        current, dwell, pixel = self.attrs.get("beam_current_pa"), self.attrs.get("dwell_time_s"), self.pixel_m
        if current is None or dwell is None or not pixel:
            return None
        return float(current) * 1e-12 * float(dwell) / ELECTRON_CHARGE / (pixel / 1e-10) ** 2

    def frame_time_s(self) -> float | None:
        dwell = self.attrs.get("dwell_time_s")
        return None if dwell is None else float(dwell) * self.array.size


@dataclass
class RunData:
    """A run's events (in trace order), lazy access to the data they reference, and optional ground truth.

    truth: the twin's ``export_ground_truth()`` (particles, acquisitions, focus), optionally with
        ``focus_by_acquisition`` ({acquisition index: {"c1_m", "a1_abs_m"}}), which a bench that
        knows the hidden focus error at each frame adds. Absent on hardware.
    loader: key -> {dataset name: (array, attrs)}; default: ``read_acquisition`` (local file, or
        Tiled when ``tiled_uri`` is given).
    """

    run: str | None
    events: list[dict]
    truth: dict | None = None
    tiled_uri: str | None = None
    loader: Callable[[str], dict] | None = None
    missing: dict[str, str] = field(default_factory=dict)
    _cache: dict[str, list[Image]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ data
    def images(self, event: dict) -> list[Image]:
        """The 2-D images an event produced (none for a failed or dry-run event, or data that cannot be read)."""
        if not event.get("ok", True) or event.get("dry_run") or event.get("performed") is False:
            return []
        out: list[Image] = []
        for key in data_keys(event):
            out += self.read(key)
        return out

    def read(self, key: str) -> list[Image]:
        if key in self._cache:
            return self._cache[key]
        images: list[Image] = []
        try:
            datasets = (self.loader or self._default_loader)(key)
            for name, (array, attrs) in datasets.items():
                array = np.asarray(array)
                if array.ndim == 2 and array.size >= 64:
                    images.append(Image(key, name, array.astype(np.float64), dict(attrs or {})))
        except Exception as exc:  # data that cannot be read is reported, and the evaluators score what they can
            self.missing[key] = f"{type(exc).__name__}: {exc}"[:200]
        self._cache[key] = images
        return images

    def _default_loader(self, key: str) -> dict:
        from asyncroscopy.data.data_reader import read_acquisition

        proxy = _TiledConfig(self.tiled_uri) if self.tiled_uri else None
        return read_acquisition(key, proxy)

    def all_images(self) -> list[tuple[dict, Image]]:
        """(event, image) for every image the run produced, in trace order; a key named twice is read once."""
        seen, out = set(), []
        for event in self.events:
            for image in self.images(event):
                if (image.key, image.name) in seen:
                    continue
                seen.add((image.key, image.name))
                out.append((event, image))
        return out

    # ------------------------------------------------------------------ ground truth (twin only)
    def truth_record(self, image: Image) -> dict | None:
        """The twin's record of the acquisition behind ``image`` (true centre, timing), matched by acquisition index."""
        if not self.truth or image.index is None:
            return None
        records = getattr(self, "_records", None)
        if records is None:
            records = self._records = {int(r["index"]): r for r in self.truth.get("acquisitions", []) if "index" in r}
        return records.get(image.index)

    def focus_truth(self, image: Image) -> dict | None:
        """The hidden C1 and |A1| when ``image`` was taken, if the bench recorded them (``focus_by_acquisition``)."""
        if not self.truth or image.index is None:
            return None
        return (self.truth.get("focus_by_acquisition") or {}).get(str(image.index))

    def particles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Ground-truth particles: (ids, centres (n, 2) m, radii m); None off the twin."""
        if not self.truth or not self.truth.get("particles"):
            return None
        cached = getattr(self, "_particles", None)
        if cached is None:
            items = self.truth["particles"]
            ids = np.array([int(p["id"]) for p in items])
            centres = np.array([p["center_m"][:2] for p in items], dtype=np.float64)
            radii = np.array([min(p.get("semi_axes_m", [1e-9])[:2]) for p in items], dtype=np.float64)
            cached = self._particles = (ids, centres, radii)
        return cached

    def centre_m(self, image: Image) -> tuple[float, float] | None:
        """Where the field was: the twin's true centre when known, else the stage position the data reports."""
        record = self.truth_record(image)
        if record and record.get("center_true_m"):
            return float(record["center_true_m"][0]), float(record["center_true_m"][1])
        for key in ("stage_reported_m", "stage_commanded_m"):
            value = image.attrs.get(key)
            if isinstance(value, str):  # Tiled metadata keeps lists as JSON text
                try:
                    value = json.loads(value)
                except ValueError:
                    value = None
            if value is not None and len(value) >= 2:
                return float(value[0]), float(value[1])
        return None

    # ------------------------------------------------------------------ time
    def clock_span_s(self) -> float | None:
        """Instrument time from the first to the last event that carries the instrument's clock."""
        clocks = [float(e["clock_s"]) for e in self.events if e.get("clock") == "instrument" and isinstance(e.get("clock_s"), (int, float))]
        return (max(clocks) - min(clocks)) if len(clocks) >= 2 else None


class _TiledConfig:
    """Stands in for the DATA device: read_acquisition only asks it for the Tiled URI."""

    def __init__(self, uri: str):
        self.uri = uri

    def get_config(self) -> str:
        return json.dumps({"uri": self.uri})


def data_keys(event: dict) -> list[str]:
    """Data keys an event references: ``result_key``, and ``key``/``image_key``/``spectrum_key``/``keys`` in its result."""
    keys: list[str] = []
    if event.get("result_key"):
        keys.append(str(event["result_key"]))
    result = event.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            result = None
    if isinstance(result, dict):
        for name in KEY_FIELDS:
            if isinstance(result.get(name), str) and result[name]:
                keys.append(result[name])
        if isinstance(result.get("keys"), list):
            keys += [str(k) for k in result["keys"] if isinstance(k, str)]
    return list(dict.fromkeys(keys))


def event_result(event: dict) -> dict:
    """An event's result as a dict ({} when it is not JSON or was cut short)."""
    result = event.get("result")
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            value = json.loads(result)
            return value if isinstance(value, dict) else {}
        except ValueError:
            return {}
    return {}


def load_run(
    run: str,
    trace_dir: str | os.PathLike | Iterable[str | os.PathLike] | None = None,
    truth: dict | str | os.PathLike | None = None,
    tiled_uri: str | None = None,
) -> RunData:
    """A run's events from the Instrument Trace: matched by agent run (``run``), else caller, else job/reservation id."""
    from asyncroscopy.trace.stream import read_trace

    sources = [trace_dir] if isinstance(trace_dir, (str, os.PathLike)) else list(trace_dir or [DEFAULT_TRACE_DIR])
    events = read_trace(sources, run=run)
    if not events:
        events = read_trace(sources, caller=run)
    if not events:
        events = read_trace(sources, job=run)
    if isinstance(truth, (str, os.PathLike)):
        truth = json.loads(Path(truth).read_text(encoding="utf-8"))
    return RunData(run, events, truth=truth, tiled_uri=tiled_uri)


def as_run_data(data: Any) -> RunData:
    if isinstance(data, RunData):
        return data
    if isinstance(data, str):
        return load_run(data)
    if isinstance(data, list):
        return RunData(None, data)
    raise TypeError(f"expected a RunData, a run id or a list of trace events, got {type(data).__name__}")


# ----------------------------------------------------------------------
# Features found in an image, matched to the twin's particles
# ----------------------------------------------------------------------
def detect_blobs(array: np.ndarray, k: float = 5.0, smooth_px: float = 1.0, background_px: float = 8.0) -> np.ndarray:
    """(row, col) of bright compact features: local maxima of a smoothed, background-subtracted image above k x noise.

    Deliberately plain (one threshold, no model of the particles), so what it finds is easy to
    reason about; it is a stand-in for whatever finder a team trusts.
    """
    from scipy import ndimage

    data = np.asarray(array, dtype=np.float64)
    smooth = ndimage.gaussian_filter(data, smooth_px)
    high = smooth - ndimage.gaussian_filter(smooth, background_px)
    noise = 1.4826 * float(np.median(np.abs(high - np.median(high)))) + 1e-12
    peaks = (high == ndimage.maximum_filter(high, size=5)) & (high > k * noise)
    return np.argwhere(peaks)


def match_particles(data: RunData, image: Image, k: float = 5.0) -> dict | None:
    """Ground-truth particles in the field, and those a detected feature lands on (within the particle's radius).

    None when the run has no ground truth or the field cannot be placed. Assumes the twin's
    image axes (rows = +y, columns = +x, centred on the field) and no scan rotation.
    """
    particles = data.particles()
    centre, pixel = data.centre_m(image), image.pixel_m
    if particles is None or centre is None or not pixel:
        return None
    ids, centres, radii = particles
    rows, cols = image.array.shape
    half_x, half_y = cols * pixel / 2, rows * pixel / 2
    inside = (np.abs(centres[:, 0] - centre[0]) < half_x) & (np.abs(centres[:, 1] - centre[1]) < half_y)
    peaks = detect_blobs(image.array, k=k)
    found: list[int] = []
    if len(peaks) and inside.any():
        xy = np.stack([centre[0] - half_x + (peaks[:, 1] + 0.5) * pixel, centre[1] - half_y + (peaks[:, 0] + 0.5) * pixel], axis=1)
        for i in np.where(inside)[0]:
            distance = np.hypot(xy[:, 0] - centres[i, 0], xy[:, 1] - centres[i, 1]).min()
            if distance <= max(radii[i], 2 * pixel):
                found.append(int(ids[i]))
    return {"in_field": [int(v) for v in ids[inside]], "found": found, "detections": len(peaks)}


def finite_or_none(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)
