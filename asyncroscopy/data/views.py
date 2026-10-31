"""Look at stored data as a PNG or as text, without downloading whole arrays.

Agents cannot read raw arrays, so every stored dataset is shown through a *kind*:
an image, a spectrum, a stack of frames, a table... Each kind knows how to draw
itself as a small PNG and how to describe itself as short text (shape, units,
statistics, peaks, a capped CSV). Reads are sliced and strided on the Tiled server
so a 4-D dataset or a run with hundreds of tiles never comes over HTTP whole.

Pure functions over Tiled nodes (or anything with the same ``shape``, ``dtype``,
``metadata`` and ``read(slices)``); no Tango or MCP imports, so notebooks and
other servers can use them too.

A new instrument gets sensible views by writing attributes the built-in kinds
understand (``axis_offset``/``axis_scale``/``axis_label``/``axis_units`` on 1-D
data, ``pixel_size_m`` on images). For anything else, register a kind::

    from asyncroscopy.data.views import Axis, line_kind, register_kind
    register_kind(line_kind(
        "diffractogram",
        matches=lambda ds: ds.ndim == 1 and "two_theta_start_deg" in ds.attrs,
        axis=lambda ds: Axis(ds.attrs["two_theta_start_deg"], ds.attrs["two_theta_step_deg"], "2θ", "deg"),
    ), before="profile")

A kind's ``png`` returns (PNG bytes, what was read) or (PNG bytes, what was read,
a short text summary); the summary goes with the picture, so an agent that cannot
see images still learns the statistics and peaks.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import islice
from typing import Any, Callable, Iterator

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image as PILImage

from asyncroscopy.data.data_reader import numpy_to_python

MAX_READ_BYTES = 32 * 2**20  # per read from Tiled
MAX_TEXT_CHARS = 6000
PREVIEW_MAX_SIDE = 512
MAX_LISTED = 200  # datasets walked inside one container
MAX_GROUPS = 40  # entries listed per level of a container
MAX_DEPTH = 3
MAX_PEAKS = 8
PEAK_SIGMA = 5.0  # a peak's counts within its FWHM must stand this many sigmas above the background
MAX_STRINGS = 50
SI_ESCAPE_EV = 1740.0
# Silicon-drift EDS detector resolution, as the twin models it (twin/eds_model.py).
EDS_FWHM_MN_KA_EV = 130.0
MN_KA_EV = 5895.0
IMAGE_GROUPS = ("image", "tiles", "channels")  # looked at first: a run's pictures before its spectra
SPECTRUM_GROUPS = ("spectrum_image", "spectrum", "spectra")


@dataclass(frozen=True)
class Dataset:
    """One array (or table) inside a key, described from its structure, without reading its values."""

    key: str
    path: str
    node: Any
    shape: tuple[int, ...]
    dtype: np.dtype
    attrs: dict
    parent_attrs: dict = field(default_factory=dict)
    family: str = "array"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize if self.shape else self.dtype.itemsize

    @property
    def name(self) -> str:
        return f"{self.key}:{self.path}" if self.path else self.key


@dataclass(frozen=True)
class Axis:
    """A linear axis: channel i is at ``offset + i * scale``."""

    offset: float
    scale: float
    label: str
    units: str

    def values(self, start: int, stop: int, step: int = 1) -> np.ndarray:
        return self.offset + self.scale * np.arange(start, stop, step, dtype=np.float64)

    def channel(self, value: float) -> float:
        return (value - self.offset) / self.scale if self.scale else 0.0


@dataclass
class ViewOptions:
    region: list[float] | None = None  # image [row0, row1, col0, col1] in pixels; 1-D [min, max] in axis units
    index: list[int] | None = None  # a frame of a stack, or a [row, col] position of a spectrum image
    max_side: int = PREVIEW_MAX_SIDE
    max_rows: int = 200  # CSV rows in text views


@dataclass
class Kind:
    name: str
    matches: Callable[[Dataset], bool]
    png: Callable[[Dataset, ViewOptions], tuple]
    text: Callable[[Dataset, ViewOptions], tuple[str, str]]
    description: str = ""


# ----------------------------------------------------------------- finding datasets


def _decode(value: Any) -> Any:
    """data_writer json-encodes list and dict attributes; decode them back."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _attrs(node: Any) -> dict:
    return {str(k): _decode(numpy_to_python(v)) for k, v in dict(getattr(node, "metadata", None) or {}).items()}


def _family(node: Any) -> str:
    family = getattr(node, "structure_family", None)
    if family is not None:
        return str(getattr(family, "value", family))
    if callable(getattr(node, "read", None)) and not callable(getattr(node, "keys", None)):
        return "array"
    return "container" if callable(getattr(node, "keys", None)) else "unknown"


class _Loaded:
    """An array already in memory, with the node interface (for nodes that do not report their shape)."""

    def __init__(self, array: np.ndarray, metadata: dict):
        self.array, self.metadata = array, metadata
        self.shape, self.dtype = array.shape, array.dtype

    def read(self, slices: Any = None) -> np.ndarray:
        return self.array if slices is None else self.array[slices]


def as_dataset(node: Any, key: str, path: str = "", parent_attrs: dict | None = None) -> Dataset:
    family = _family(node)
    shape = getattr(node, "shape", None)
    if family == "array" and shape is None:  # a reader that cannot report its shape: read it once
        node = _Loaded(np.asarray(node.read()), dict(getattr(node, "metadata", None) or {}))
        shape = node.shape
    try:
        dtype = np.dtype(getattr(node, "dtype", None) or np.float64) if family == "array" else np.dtype(object)
    except TypeError:
        dtype = np.dtype(object)
    return Dataset(key, path, node, tuple(int(s) for s in (shape or ())), dtype, _attrs(node), dict(parent_attrs or {}), family)


def _children(node: Any) -> Iterator[tuple[str, Any]]:
    items = getattr(node, "items", None)
    if callable(items):  # Tiled pages carry each child's structure: no request per child, pages fetched as needed
        return ((str(k), v) for k, v in items())
    return ((str(k), node[k]) for k in node.keys())


def _count(node: Any) -> int | None:
    try:
        return len(node)
    except TypeError:
        return None


def _join(base: str, name: str) -> str:
    return f"{base}/{name}" if base else name


def datasets(node: Any, key: str, limit: int = MAX_LISTED) -> list[Dataset]:
    """The arrays and tables inside ``node`` (itself, if it is one), at most ``limit``, at most 3 levels deep."""
    out: list[Dataset] = []

    def walk(current: Any, path: str, depth: int, parent: dict) -> None:
        if len(out) >= limit:
            return
        family = _family(current)
        if family in ("array", "table"):
            out.append(as_dataset(current, key, path, parent))
            return
        if family != "container" or depth > MAX_DEPTH:
            return
        attrs = _attrs(current)
        for child, value in _children(current):
            walk(value, _join(path, child), depth + 1, attrs)
            if len(out) >= limit:
                return

    walk(node, "", 0, {})
    return out


def _first_dataset(node: Any, key: str, path: str = "", parent: dict | None = None, depth: int = 0) -> Dataset | None:
    """The first array or table found depth-first, reading as little as possible."""
    family = _family(node)
    if family in ("array", "table"):
        return as_dataset(node, key, path, parent)
    if family != "container" or depth > MAX_DEPTH:
        return None
    attrs = _attrs(node)
    for child, value in islice(_children(node), MAX_GROUPS):
        found = _first_dataset(value, key, _join(path, child), attrs, depth + 1)
        if found is not None:
            return found
    return None


def is_map(node: Any) -> bool:
    return _family(node) == "container" and _attrs(node).get("kind") == "stitched_map"


def resolve(node: Any, key: str, path: str) -> Any:
    """The node at ``path`` inside ``key`` (a dataset or a group), or a ValueError listing what is there."""
    current = node
    for part in path.strip("/").split("/"):
        try:
            current = current[part]
        except (KeyError, TypeError):
            raise ValueError(f"{key!r} has no dataset {path!r}; it has:\n{container_listing(node)}") from None
    return current


def pick(node: Any, key: str, dataset: str | None = None, max_side: int = PREVIEW_MAX_SIDE) -> Dataset:
    """The dataset to show for ``key``: ``dataset`` if given (inside it, if it is a group), else a picture
    (image, a run's tiles or channels), a spectrum, a map level, else the first dataset."""
    base = ""
    if dataset:
        base = dataset.strip("/")
        target = resolve(node, key, base)
        if _family(target) in ("array", "table"):
            return as_dataset(target, key, base, _attrs(node))
        node = target
    if _family(node) in ("array", "table"):
        return as_dataset(node, key)
    if is_map(node):
        levels = sorted((d for d in datasets(node, key, limit=40) if d.path.startswith("level_") and d.ndim == 2), key=lambda d: d.nbytes)
        fitting = [d for d in levels if max(d.shape) <= 2 * max_side]
        if levels:
            return fitting[-1] if fitting else levels[0]
    for name in IMAGE_GROUPS + SPECTRUM_GROUPS:
        try:
            child = node[name]
        except (KeyError, TypeError):
            continue
        found = _first_dataset(child, key, _join(base, name), _attrs(node))
        if found is not None:
            return found
    found = _first_dataset(node, key, base)
    if found is None:
        raise ValueError(f"{key!r}{' / ' + base if base else ''} holds no arrays or tables")
    return found


def container_listing(node: Any, prefix: str = "", depth: int = 0) -> str:
    """Every group of a container, breadth-first: 'tiles/: 400 x (128, 128) uint16 (shape of tile_00000)'."""
    lines: list[str] = []
    children = list(islice(_children(node), MAX_GROUPS + 1))
    for name, child in children[:MAX_GROUPS]:
        family = _family(child)
        if family in ("array", "table"):
            ds = as_dataset(child, "", name)
            lines.append(f"{prefix}{name}: {tuple(ds.shape)} {ds.dtype}" if family == "array" else f"{prefix}{name}: table")
            continue
        if family != "container":
            lines.append(f"{prefix}{name}: {family}")
            continue
        count = _count(child)
        counted = count if count is not None else "?"
        first = next(iter(_children(child)), None)
        if first is None:
            lines.append(f"{prefix}{name}/: empty")
        elif _family(first[1]) in ("array", "table"):
            ds = as_dataset(first[1], "", first[0])
            lines.append(f"{prefix}{name}/: {counted} x {tuple(ds.shape)} {ds.dtype} (shape of {first[0]})")
        elif depth < MAX_DEPTH:
            lines.append(f"{prefix}{name}/: {counted} groups")
            lines.append(container_listing(child, f"{prefix}{name}/", depth + 1))
        else:
            lines.append(f"{prefix}{name}/: {counted} groups (deeper levels not listed; open one with dataset)")
    if len(children) > MAX_GROUPS:
        total = _count(node)
        lines.append(f"{prefix}... and {(total - MAX_GROUPS) if total else 'more'} more entries")
    return "\n".join(line for line in lines if line)


# ----------------------------------------------------------------- reading


def _stride_for(shape: tuple[int, ...], itemsize: int, target: int | None) -> int:
    """A common step that keeps the read under MAX_READ_BYTES and, if given, near ``target`` per axis."""
    if not shape:
        return 1
    count = int(np.prod(shape, dtype=np.int64))
    step = math.ceil((count * itemsize / MAX_READ_BYTES) ** (1.0 / len(shape))) if count * itemsize > MAX_READ_BYTES else 1
    if target:
        step = max(step, math.ceil(max(shape) / target))
    return max(1, step)


def _read(ds: Dataset, slices: tuple[slice | int, ...]) -> np.ndarray:
    return np.asarray(ds.node.read(slices))


def _numeric(values: np.ndarray) -> np.ndarray:
    """Values that statistics and plots can use: magnitude of complex, 0/1 for bool."""
    if values.dtype.kind == "c":
        return np.abs(values)
    if values.dtype.kind == "b":
        return values.astype(np.uint8)
    return values


def _describe_slices(slices: tuple[slice | int, ...]) -> str:
    parts = []
    for s in slices:
        if isinstance(s, slice):
            parts.append(f"{s.start or 0}:{s.stop}" + (f":{s.step}" if s.step and s.step > 1 else ""))
        else:
            parts.append(str(s))
    return "[" + ", ".join(parts) + "]"


def _finite_region(region: Any, length: int, what: str) -> list[float]:
    if not isinstance(region, (list, tuple)) or len(region) != length:
        raise ValueError(f"{what}; got {region!r}")
    try:
        values = [float(v) for v in region]
    except (TypeError, ValueError):
        raise ValueError(f"{what}; got {region!r}") from None
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"region {region} must be finite numbers; {what}")
    return values


def _span(lo: float, hi: float, size: int, axis_name: str) -> tuple[int, int]:
    """Integer [start, stop) inside 0..size for a requested [lo, hi), or a clear error."""
    if hi <= lo:
        raise ValueError(f"region {axis_name}: the end ({_fmt(hi)}) must be after the start ({_fmt(lo)})")
    start, stop = max(0, int(math.floor(lo))), min(size, int(math.ceil(hi)))
    if start >= size or stop <= 0 or stop <= start:
        raise ValueError(f"region {axis_name} [{_fmt(lo)}, {_fmt(hi)}] is outside the data (0 to {size})")
    return start, stop


def read_image(ds: Dataset, opts: ViewOptions, target: int | None = None, frame: tuple[int, ...] = ()) -> tuple[np.ndarray, str]:
    """A 2-D (or H x W x 3/4 colour) picture from ``ds``, cropped to opts.region and strided on the server."""
    plane = ds.shape[len(frame):]
    rows, cols = plane[0], plane[1]
    r0, r1, c0, c1 = 0, rows, 0, cols
    if opts.region:
        a, b, c, d = _finite_region(opts.region, 4, "region for an image is [row0, row1, col0, col1] in pixels")
        r0, r1 = _span(a, b, rows, "rows")
        c0, c1 = _span(c, d, cols, "columns")
    extra = plane[2:]
    step = _stride_for((r1 - r0, c1 - c0), ds.dtype.itemsize * int(np.prod(extra, dtype=np.int64) or 1), target)
    slices = (*frame, slice(r0, r1, step), slice(c0, c1, step), *(slice(0, n) for n in extra))
    return _numeric(_read(ds, slices)), "read " + _describe_slices(slices)


def read_line(ds: Dataset, opts: ViewOptions, axis: Axis | None) -> tuple[np.ndarray, int, int, str]:
    """A 1-D dataset (cropped to opts.region in axis units); returns (values, first channel, step, what was read)."""
    n = ds.shape[0]
    start, stop = 0, n
    if opts.region:
        lo, hi = sorted(_finite_region(opts.region, 2, "region for 1-D data is [min, max] in axis units"))
        if axis is not None:
            lo, hi = sorted((axis.channel(lo), axis.channel(hi)))
        start, stop = _span(lo, hi + 1, n, "channels")
    step = _stride_for((stop - start,), ds.dtype.itemsize, None)
    slices = (slice(start, stop, step),)
    return _numeric(_read(ds, slices)), start, step, "read " + _describe_slices(slices)


# ----------------------------------------------------------------- numbers to words


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _axis_value(value: float) -> str:
    """A position on an axis as people write it: 2690 (cm-1), 28.4 (degrees), 1.49 (keV); not 2.69e+03."""
    return f"{value:.0f}" if 100 <= abs(value) < 1e6 else f"{value:.3g}"


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> str:
    finite = _finite(values)
    if finite.size == 0:
        return "no finite values"
    p1, p50, p99 = np.percentile(finite, [1, 50, 99])
    return (
        f"min {_fmt(finite.min())}, max {_fmt(finite.max())}, mean {_fmt(finite.mean())}, std {_fmt(finite.std())}, "
        f"p1 {_fmt(p1)}, median {_fmt(p50)}, p99 {_fmt(p99)}"
    )


def _histogram(values: np.ndarray, bins: int = 16) -> str:
    finite = _finite(values)
    if finite.size == 0 or finite.min() == finite.max():
        return ""
    counts, edges = np.histogram(finite, bins=bins)
    top = counts.max()
    return "\n".join(f"{_fmt(edges[i]):>10} {'#' * int(round(30 * c / top)):<30} {c}" for i, c in enumerate(counts))


def _csv(header: str, rows: list[tuple]) -> str:
    return "\n".join([header, *(",".join(_fmt(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) for row in rows)])


def _cap(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[: MAX_TEXT_CHARS - 80].rsplit("\n", 1)[0] + f"\n... (cut at {MAX_TEXT_CHARS} characters; narrow it with region or dataset)"


def _png(figure_or_pixels: Any) -> bytes:
    buffer = io.BytesIO()
    if isinstance(figure_or_pixels, Figure):
        FigureCanvasAgg(figure_or_pixels)
        figure_or_pixels.savefig(buffer, format="png")
    else:
        figure_or_pixels.save(buffer, format="png")
    return buffer.getvalue()


def to_pixels(values: np.ndarray, max_side: int) -> PILImage.Image:
    """8-bit picture with 1-99.5 percentile contrast (a hot pixel does not black it out); H x W x 3/4 stays colour."""
    colour = values.ndim == 3 and values.shape[2] in (3, 4)
    if colour and values.dtype == np.uint8:
        image = PILImage.fromarray(np.ascontiguousarray(values[..., :3]))
    else:
        data = np.asarray(_numeric(values[..., :3] if colour else values), dtype=np.float64)
        finite = data[np.isfinite(data)]
        low, high = (np.percentile(finite, [1.0, 99.5]) if finite.size else (0.0, 1.0))
        if high <= low:
            low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        scaled = np.clip((np.nan_to_num(data, nan=low) - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(data)
        image = PILImage.fromarray((scaled * 255).astype(np.uint8), mode="RGB" if colour else "L")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), PILImage.BILINEAR)
    return image


# ----------------------------------------------------------------- peaks


def energy_axis(attrs: dict) -> Axis | None:
    """The calibrated axis of a 1-D dataset, from any naming convention this repository writes."""
    try:
        if "energy_offset_ev" in attrs and "energy_dispersion_ev" in attrs:
            return Axis(float(attrs["energy_offset_ev"]), float(attrs["energy_dispersion_ev"]), "energy", "eV")
        if "dispersion" in attrs and ("start_energy" in attrs or "offset" in attrs):
            return Axis(float(attrs.get("start_energy", attrs.get("offset"))), float(attrs["dispersion"]), "energy", str(attrs.get("energy_units", "eV")))
        if "axis_offset" in attrs and "axis_scale" in attrs:
            return Axis(float(attrs["axis_offset"]), float(attrs["axis_scale"]), str(attrs.get("axis_label", "axis")), str(attrs.get("axis_units", "")))
    except (TypeError, ValueError):
        return None
    return None


def eds_fwhm_ev(energy_ev: Any) -> np.ndarray:
    """Silicon-drift detector line width at an energy (eV), at least 30 eV."""
    return np.sqrt(np.maximum(EDS_FWHM_MN_KA_EV**2 + 2.5 * (np.asarray(energy_ev, dtype=np.float64) - MN_KA_EV), 30.0**2))


@dataclass(frozen=True)
class Peak:
    index: int  # channel in the values given
    net: float  # counts within +-FWHM/2 above the background
    significance: float  # net / its noise


def find_peaks(values: np.ndarray, fwhm_channels: Any = 3.0, limit: int = MAX_PEAKS) -> list[Peak]:
    """Significant peaks, most significant first; none in pure noise.

    For each local maximum, the values within +-FWHM/2 above a running-median background
    are summed; the sum must exceed PEAK_SIGMA times its noise: sqrt(background summed over
    the same channels) for count data, else the residual's scatter times sqrt(width).
    ``fwhm_channels`` is the expected peak width in channels, one number or one per channel.
    """
    from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
    from scipy.signal import find_peaks as scipy_peaks
    from scipy.stats import norm, poisson

    data = np.nan_to_num(np.asarray(values, dtype=np.float64))
    n = data.size
    if n < 9 or data.max() == data.min():
        return []
    width = np.maximum(np.broadcast_to(np.asarray(fwhm_channels, dtype=np.float64), (n,)), 1.0)
    window = int(max(9, 4 * np.median(width), n // 40))
    window = min(window | 1, n - 1 if n % 2 == 0 else n)
    counts = data.min() >= 0 and np.allclose(data, np.round(data))
    # Background: the mean of side bands one line width away on each side, of the data clipped near
    # a running median. The median alone is biased low for sparse counts (it is a whole number), which
    # turns noise into peaks; a plain running mean would count a peak as its own background.
    rough = median_filter(data, size=window, mode="nearest")
    rough_scatter = 1.4826 * float(np.median(np.abs(data - rough))) or 1e-12
    clipped = np.minimum(data, rough + 3.0 * (np.sqrt(np.maximum(rough, 1.0)) if counts else rough_scatter))
    gap, band = int(math.ceil(float(np.median(width)))), max(8, int(3 * float(np.median(width))))
    summed = np.concatenate(([0.0], np.cumsum(clipped)))
    at = np.arange(n)
    left_lo, left_hi = np.clip(at - gap - band, 0, n), np.clip(at - gap, 0, n)
    right_lo, right_hi = np.clip(at + gap + 1, 0, n), np.clip(at + gap + 1 + band, 0, n)
    taken = (left_hi - left_lo) + (right_hi - right_lo)
    background = np.where(
        taken > 0,
        (summed[left_hi] - summed[left_lo] + summed[right_hi] - summed[right_lo]) / np.maximum(taken, 1),
        uniform_filter1d(clipped, size=window, mode="nearest"),
    )
    residual = data - background
    scatter = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-6 * float(data.max() - data.min()))
    summed_residual = np.concatenate(([0.0], np.cumsum(residual)))
    summed_background = np.concatenate(([0.0], np.cumsum(background)))
    # Candidates: maxima of the residual smoothed to the expected line width (a matched filter),
    # so a weak peak is centred by its shape, not by its noisiest channel.
    smoothed = gaussian_filter1d(residual, sigma=max(0.5, float(np.median(width)) / 2.355), mode="nearest")
    candidates, _ = scipy_peaks(smoothed, distance=max(1, int(np.median(width) // 2)))
    peaks = []
    for i in candidates:
        half = max(1, int(round(width[i] / 2)))
        lo, hi = max(0, i - half), min(n, i + half + 1)
        net = summed_residual[hi] - summed_residual[lo]
        channels, under = hi - lo, summed_background[hi] - summed_background[lo]
        # The background under the peak is itself an estimate from `taken` channels: add its variance.
        spread = 1.0 + channels / max(int(taken[i]), 1)
        if counts:  # exact Poisson tail (sparse counts have heavier tails than a Gaussian), as a z-score
            expected = max(under, 0.5)
            z = float(norm.isf(max(poisson.sf(round(expected + net) - 1, expected), 1e-300))) / math.sqrt(spread)
        else:
            z = net / (scatter * math.sqrt(channels * spread))
        if net > 0 and z > PEAK_SIGMA:
            peaks.append(Peak(int(i), float(net), z))
    return sorted(peaks, key=lambda p: p.significance, reverse=True)[:limit]


@lru_cache(maxsize=1)
def _xray_lines() -> tuple[tuple[float, str, float], ...]:
    """(energy eV, "Fe K-L3", weight) for the main X-ray lines in pyTEMlib's tables."""
    try:
        from pyTEMlib import xrpa_x_sections
    except ImportError:
        return ()
    out = []
    for entry in xrpa_x_sections.x_sections.values():
        for line, info in (entry.get("lines") or {}).items():
            if float(info.get("weight", 0)) >= 0.1 and 100.0 <= float(info["position"]) <= 40000.0:
                out.append((float(info["position"]), f"{entry['name']} {line}", float(info["weight"])))
    return tuple(sorted(out))


def xray_candidates(energy_ev: float, tolerance_ev: float) -> list[str]:
    near = [(abs(e - energy_ev), name) for e, name, _w in _xray_lines() if abs(e - energy_ev) <= tolerance_ev]
    return [name for _d, name in sorted(near)[:3]]


def is_xray(ds: Dataset) -> bool:
    words = " ".join(str(ds.attrs.get(k, "")) + " " + str(ds.parent_attrs.get(k, "")) for k in ("detector", "acquisition_type", "technique")).lower()
    return "eds" in words or "edx" in words or "x-ray" in words or "xray" in words


def line_peaks(x: np.ndarray, y: np.ndarray, axis: Axis, xray: bool, step: int = 1) -> list[tuple[float, float, float, int, str]]:
    """The significant peaks of 1-D data, strongest (net counts) first: (position, net, sigma, index, note).

    The one peak finder: view_data lists these, and the benchmarks' truth is read with it. An X-ray
    spectrum uses the detector's line width and names candidate lines (or flags escape and sum peaks);
    any other axis a width of a few channels.
    """
    fwhm = eds_fwhm_ev(x) / abs(axis.scale * step) if xray and axis.scale else 3.0
    rows: list[tuple[float, float, float, int, str]] = []
    for peak in find_peaks(y, fwhm):
        position = float(x[peak.index])
        note = ""
        if xray:
            tolerance = max(float(eds_fwhm_ev(position)) / 2, 2 * abs(axis.scale * step))
            note = _artefact(position, [row[0] for row in rows], tolerance) or ", ".join(xray_candidates(position, tolerance))
        rows.append((position, peak.net, peak.significance, peak.index, note))
    return sorted(rows, key=lambda row: row[1], reverse=True)


def _artefact(energy: float, stronger: list[float], tolerance: float) -> str:
    """A note when an EDS peak may be a Si escape or a sum (pile-up) peak of a stronger one."""
    for e in stronger:
        if abs(energy - (e - SI_ESCAPE_EV)) <= tolerance:
            return f"possible Si escape peak of the {_fmt(e)} eV peak"
        if abs(energy - 2 * e) <= tolerance:
            return f"possible sum peak (2 x {_fmt(e)} eV)"
    return ""


def _real(ds: Dataset) -> bool:
    return ds.dtype.kind in "uifcb"


# ----------------------------------------------------------------- 1-D: spectra and profiles


def likely_elements(rows, share: float = 0.25) -> str:
    """Elements ranked by the summed sigma of the peaks they could explain, each peak counted once per element, kept
    while an element explains at least ``share`` of what the best one does: "Au 85%, Pt 7%"."""
    score: dict[str, float] = {}
    for _energy, _net, sigma, _index, note in rows:
        for element in {c.strip().split(" ")[0] for c in (note or "").split(",") if c.strip() and not c.strip().startswith("possible")}:
            score[element] = score.get(element, 0.0) + float(sigma)
    if not score:
        return ""
    total = sum(float(r[2]) for r in rows) or 1.0
    best = max(score.values())
    ranked = sorted((e for e in score if score[e] >= share * best), key=lambda e: -score[e])[:4]
    return ", ".join(f"{e} {100.0 * score[e] / total:.0f}%" for e in ranked)


def line_kind(name: str, matches: Callable[[Dataset], bool], axis: Callable[[Dataset], Axis | None], description: str = "") -> Kind:
    """A kind for 1-D data plotted against a calibrated axis, with its significant peaks marked and listed."""

    def describe(ds: Dataset, opts: ViewOptions):
        y, start, step, read = read_line(ds, opts, axis(ds))
        a = axis(ds) or Axis(0.0, 1.0, "channel", "")
        x = a.offset + a.scale * (start + step * np.arange(y.size))
        xray = a.units == "eV" and is_xray(ds)
        rows = line_peaks(x, y, a, xray, step)
        unit = f" {a.units}" if a.units else ""
        total = float(np.nansum(np.asarray(y, dtype=np.float64)))
        lines = [
            f"{ds.shape[0]} channels; {a.label} {_fmt(x[0])} to {_fmt(x[-1])}{unit} ({_fmt(a.scale * step)}{unit} per value shown)" if y.size else "empty",
            f"total {_fmt(total)}; {_stats(y)}",
        ]
        if rows and xray:
            likely = likely_elements(rows)
            if likely:
                # A small model given only per-peak candidates did not conclude; the ranking says which elements the
                # peaks point to, strongest first, and stays a reading of the peaks (a light element can still hide).
                lines.append(f"elements most consistent with these peaks (the share of the peak signal each one explains): {likely}")
        if rows:
            lines += ["peaks (strongest first; net = counts within the line width above the background, sigma = how far above the noise):",
                      _csv(f"{a.label}{unit},net,sigma,note", [(e, net, sig, f'"{note}"' if note else "") for e, net, sig, _i, note in rows])]
        elif xray and total == 0:
            lines.append("0 counts: no electrons reached the detector, so a longer exposure will not help. The beam was probably "
                         "blanked (unblank_beam, then acquire again) or the detector retracted")
        elif xray:
            lines.append(f"no peaks detected above noise at this exposure (total {_fmt(total)} counts); "
                         "a longer exposure is needed to rule elements out")
        else:
            lines.append(f"no peaks stand {PEAK_SIGMA:g} sigma above the background")
        return lines, read, x, y, a, rows

    def png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str, str]:
        lines, read, x, y, a, rows = describe(ds, opts)
        factor, units = (1e-3, "keV") if a.units == "eV" and x.size and np.nanmax(np.abs(x)) > 2000 else (1.0, a.units)
        figure = Figure(figsize=(6.4, 3.6), dpi=100)
        plot = figure.add_subplot()
        plot.plot(x * factor, y, linewidth=0.9)
        for energy, _net, _sig, index, note in rows:
            label = note.split(",")[0] if note and not note.startswith("possible") else ""
            plot.annotate(_axis_value(energy * factor) + (f"\n{label}" if label else ""), (energy * factor, float(y[index])),
                          textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
        plot.set_xlabel(f"{a.label} ({units})" if units else a.label)
        plot.set_ylabel(str(ds.attrs.get("units", "counts")))
        plot.set_title(ds.name, fontsize=8)
        figure.tight_layout()
        return _png(figure), read, "\n".join(lines)

    def text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
        lines, read, x, y, a, _rows = describe(ds, opts)
        unit = f" {a.units}" if a.units else ""
        bins = max(1, math.ceil(y.size / max(1, opts.max_rows)))
        if bins > 1:  # sum neighbouring channels so counts are kept
            usable = (y.size // bins) * bins
            y_binned = np.nansum(np.asarray(y[:usable], dtype=np.float64).reshape(-1, bins), axis=1)
            x_binned = x[:usable].reshape(-1, bins).mean(axis=1)
        else:
            y_binned, x_binned = y, x
        lines += [f"data ({'sums of ' + str(bins) + ' channels' if bins > 1 else 'every channel'}):",
                  _csv(f"{a.label}{unit},value", list(zip(x_binned.tolist(), y_binned.tolist())))]
        return "\n".join(lines), read

    return Kind(name, matches, png, text, description)


def _composition_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str, str]:
    values = _read(ds, (slice(0, ds.shape[0]),))
    figure = Figure(figsize=(6.0, 3.5), dpi=100)
    plot = figure.add_subplot()
    plot.bar(np.arange(values.size), values)
    plot.set_xticks(np.arange(values.size))
    plot.set_xticklabels(ds.attrs["elements"])
    plot.set_xlabel("element")
    plot.set_ylabel("relative intensity")
    figure.tight_layout()
    return _png(figure), f"read [0:{ds.shape[0]}]", _composition_text(ds, opts)[0]


def _composition_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    values = _read(ds, (slice(0, ds.shape[0]),))
    return "\n".join(f"{element}: {_fmt(float(v))}" for element, v in zip(ds.attrs["elements"], values)), f"read [0:{ds.shape[0]}]"


def _has_elements(ds: Dataset) -> bool:
    elements = ds.attrs.get("elements")
    return ds.ndim == 1 and _real(ds) and isinstance(elements, list) and len(elements) == ds.shape[0] and all(isinstance(e, str) for e in elements)


# ----------------------------------------------------------------- strings


def _strings_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str]:
    raise ValueError(f"{ds.name} holds text ({ds.dtype}), which has no picture; use format text")


def _strings_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    if not ds.shape:
        values, read = [_read(ds, ())], "read all"
    else:
        slices = (slice(0, min(ds.shape[0], MAX_STRINGS)), *(slice(0, min(n, 4)) for n in ds.shape[1:]))
        values, read = list(_read(ds, slices).reshape(-1)), "read " + _describe_slices(slices)
    texts = [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in values]
    return "\n".join([f"{list(ds.shape)} text entries; the first {len(texts)}:"] + [t[:300] for t in texts]), read


# ----------------------------------------------------------------- 2-D and up


def image_kind(name: str, matches: Callable[[Dataset], bool], description: str = "", colormap: str | None = None) -> Kind:
    """A kind for pictures: percentile contrast, strided read, optional colour map for labels."""

    def describe(ds: Dataset, values: np.ndarray) -> list[str]:
        lines = [f"{ds.shape[0]} x {ds.shape[1]} pixels{' x ' + str(ds.shape[2]) if ds.ndim == 3 else ''}, {ds.dtype}"]
        if ds.dtype.kind == "c":
            lines.append("complex values: statistics are of the magnitude")
        pixel = ds.attrs.get("pixel_size_m")
        if isinstance(pixel, (int, float)):
            lines.append(f"pixel {_fmt(pixel)} m; field of view {_fmt(pixel * ds.shape[1])} x {_fmt(pixel * ds.shape[0])} m")
        if ds.attrs.get("beam_current_pa") == 0:
            lines.append("taken with the beam blanked (beam_current_pa 0): it shows only detector noise, not the sample; "
                         "unblank_beam and take it again")
        if colormap:
            labels = np.unique(np.asarray(values, dtype=np.int64))
            lines.append(f"{int((labels != 0).sum())} labelled areas")
            stats = ds.attrs.get("area_statistics")
            if stats:
                lines.append("area_statistics: " + json.dumps(stats)[:2000])
            return lines
        if ds.dtype.kind == "b":
            lines.append(f"true: {int(values.sum())} of {values.size} pixels read")
            return lines
        lines.append("statistics of the pixels read: " + _stats(values))
        if ds.dtype.kind in "ui":
            top = np.iinfo(ds.dtype).max
            lines.append(f"saturated (= {top}): {_fmt(100.0 * float((values == top).mean()))}%")
        histogram = _histogram(values)
        if histogram:
            lines += ["histogram (value: count):", histogram]
        return lines

    def png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str, str]:
        values, read = read_image(ds, opts, target=opts.max_side)
        brief = "\n".join(describe(ds, values))
        if colormap:
            from matplotlib import colormaps

            labels = np.asarray(values, dtype=np.int64)
            rgba = colormaps[colormap](labels % 20 / 19.0)
            rgba[labels == 0] = (0, 0, 0, 1)
            return _png(to_pixels((rgba[..., :3] * 255).astype(np.uint8), opts.max_side)), read, brief
        return _png(to_pixels(values, opts.max_side)), read, brief

    def text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
        values, read = read_image(ds, opts, target=1024)
        lines = describe(ds, values)
        if not colormap and ds.dtype.kind != "b":
            grid_step = max(1, math.ceil(max(values.shape[:2]) / 16))
            grid = np.asarray(values[::grid_step, ::grid_step], dtype=np.float64)
            if grid.ndim == 3:
                grid = grid.mean(axis=2)
            lines += [f"every {grid_step}th pixel of what was read (rows top to bottom):",
                      "\n".join(",".join(_fmt(v) for v in row) for row in grid[: opts.max_rows])]
        return "\n".join(lines), read

    return Kind(name, matches, png, text, description)


def _index(opts: ViewOptions) -> list[int] | None:
    if opts.index is None:
        return None
    raw = [opts.index] if isinstance(opts.index, (int, np.integer)) else opts.index
    try:
        return [int(i) for i in raw]
    except (TypeError, ValueError):
        raise ValueError(f"index is a list of whole numbers (positions on the leading axes); got {opts.index!r}") from None


def _frame(ds: Dataset, opts: ViewOptions) -> tuple[int, ...]:
    """Position on the leading axes of a stack: opts.index, else the middle."""
    lead = ds.shape[:-2]
    index = _index(opts)
    if index is None:
        return tuple(n // 2 for n in lead)
    if len(index) != len(lead) or any(not 0 <= i < n for i, n in zip(index, lead)):
        raise ValueError(f"index must give {len(lead)} positions within {list(lead)}; got {index}")
    return tuple(index)


def _stack_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str, str]:
    frame = _frame(ds, opts)
    values, read = read_image(ds, opts, target=opts.max_side, frame=frame)
    return _png(to_pixels(values, opts.max_side)), read, _stack_lines(ds, frame, values)


def _stack_lines(ds: Dataset, frame: tuple[int, ...], values: np.ndarray) -> str:
    lines = [f"shape {list(ds.shape)} {ds.dtype}; showing the frame at index {list(frame)} (pass index to choose another)"]
    for attr in ("scan_shape", "axes", "spectrum_image_axes"):
        if attr in ds.attrs:
            lines.append(f"{attr}: {ds.attrs[attr]}")
    lines.append(f"frame: {_stats(values)}")
    return "\n".join(lines)


def _stack_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    frame = _frame(ds, opts)
    values, read = read_image(ds, opts, target=1024, frame=frame)
    return _stack_lines(ds, frame, values), read


def _is_spectrum_image(ds: Dataset) -> bool:
    return ds.ndim == 3 and _real(ds) and (energy_axis(ds.attrs) is not None or "spectrum_image_axes" in ds.attrs)


def _read_cube(ds: Dataset, opts: ViewOptions) -> tuple[np.ndarray, str, int]:
    """The cube (cropped to region [row0, row1, col0, col1]), strided on rows and columns, then channels, under the byte cap."""
    rows, cols, channels = ds.shape
    r0, r1, c0, c1 = 0, rows, 0, cols
    if opts.region:
        a, b, c, d = _finite_region(opts.region, 4, "region for a spectrum image is [row0, row1, col0, col1] in positions")
        r0, r1 = _span(a, b, rows, "rows")
        c0, c1 = _span(c, d, cols, "columns")
    size = (r1 - r0) * (c1 - c0) * channels * ds.dtype.itemsize
    spatial = max(1, min(max(r1 - r0, c1 - c0), math.ceil(math.sqrt(size / MAX_READ_BYTES)))) if size > MAX_READ_BYTES else 1
    left = math.ceil((r1 - r0) / spatial) * math.ceil((c1 - c0) / spatial) * channels * ds.dtype.itemsize
    spectral = max(1, math.ceil(left / MAX_READ_BYTES))
    slices = (slice(r0, r1, spatial), slice(c0, c1, spatial), slice(0, channels, spectral))
    return _numeric(_read(ds, slices)), "read " + _describe_slices(slices), spectral


def _spectrum_at(ds: Dataset, opts: ViewOptions) -> Dataset:
    """The spectrum at index [row, col] of a spectrum image, as its own 1-D dataset."""
    index = _index(opts)
    if len(index) != 2 or not (0 <= index[0] < ds.shape[0] and 0 <= index[1] < ds.shape[1]):
        raise ValueError(f"index for a spectrum image is one [row, col] position within {list(ds.shape[:2])}; got {index}")
    row, col = index
    line = _Loaded(np.asarray(ds.node.read((row, col, slice(0, ds.shape[2])))), ds.attrs)
    return Dataset(ds.key, f"{ds.path}[{row},{col}]", line, (ds.shape[2],), ds.dtype, ds.attrs, ds.parent_attrs)


def _summed_lines(ds: Dataset, cube: np.ndarray, step: int) -> tuple[list[str], np.ndarray, np.ndarray, Axis]:
    axis = energy_axis(ds.attrs) or Axis(0.0, 1.0, "channel", "")
    summed = cube.sum(axis=(0, 1))
    x = axis.values(0, ds.shape[2], step)[: summed.size]
    lines = [f"spectrum image {ds.shape[0]} x {ds.shape[1]} positions x {ds.shape[2]} channels, {ds.dtype}",
             f"{axis.label} {_fmt(x[0])} to {_fmt(x[-1])} {axis.units}".rstrip(),
             "index [row, col] shows the spectrum at one position; region [row0, row1, col0, col1] sums part of it"]
    fwhm = eds_fwhm_ev(x) / abs(axis.scale * step) if axis.units == "eV" and is_xray(ds) and axis.scale else 3.0
    peaks = find_peaks(summed, fwhm)
    if peaks:
        lines += ["peaks of the summed spectrum:", _csv(f"{axis.label},net,sigma", [(float(x[p.index]), p.net, p.significance) for p in peaks])]
    return lines, summed, x, axis


def _spectrum_image_png(ds: Dataset, opts: ViewOptions) -> tuple:
    if opts.index is not None:
        single = _spectrum_at(ds, opts)
        return detect(single).png(single, ViewOptions(max_side=opts.max_side))
    cube, read, step = _read_cube(ds, opts)
    lines, summed, x, axis = _summed_lines(ds, cube, step)
    figure = Figure(figsize=(8.0, 3.4), dpi=100)
    left, right = figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)
    left.imshow(np.asarray(to_pixels(cube.sum(axis=2), opts.max_side)), cmap="gray")
    left.set_title("summed intensity", fontsize=8)
    left.axis("off")
    right.plot(x, summed, linewidth=0.9)
    right.set_xlabel(f"{axis.label} ({axis.units})" if axis.units else axis.label)
    right.set_title("summed spectrum", fontsize=8)
    figure.tight_layout()
    return _png(figure), read, "\n".join(lines)


def _spectrum_image_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    if opts.index is not None:
        single = _spectrum_at(ds, opts)
        body, read = detect(single).text(single, ViewOptions(max_rows=opts.max_rows))
        return f"spectrum at position {_index(opts)}:\n{body}", read
    cube, read, step = _read_cube(ds, opts)
    return "\n".join(_summed_lines(ds, cube, step)[0]), read


TABLE_MAX_ROWS = 1000


def _table_frame(ds: Dataset, opts: ViewOptions):
    """The first rows of a table: only its first partition when Tiled splits it."""
    read_partition = getattr(ds.node, "read_partition", None)
    frame = read_partition(0) if callable(read_partition) else ds.node.read()
    rows = min(len(frame), opts.max_rows, TABLE_MAX_ROWS)
    return frame.head(rows), f"read first {rows} rows" + (" (of the first partition)" if callable(read_partition) else "")


def _table_columns(frame) -> str:
    return "columns: " + ", ".join(f"{c} ({frame[c].dtype})" for c in frame.columns)


def _table_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str, str]:
    frame, read = _table_frame(ds, ViewOptions(max_rows=TABLE_MAX_ROWS))
    numeric = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    if not numeric:
        raise ValueError(f"{ds.name} has no numeric columns to plot; use format text")
    figure = Figure(figsize=(6.4, 3.6), dpi=100)
    plot = figure.add_subplot()
    if len(numeric) >= 2:
        for column in numeric[1:6]:
            plot.plot(frame[numeric[0]], frame[column], label=str(column), linewidth=0.9)
        plot.set_xlabel(str(numeric[0]))
        plot.legend(fontsize=7)
    else:
        plot.plot(frame[numeric[0]].to_numpy(), linewidth=0.9)
        plot.set_ylabel(str(numeric[0]))
    figure.tight_layout()
    return _png(figure), read, _table_columns(frame)


def _table_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    frame, read = _table_frame(ds, opts)
    return f"{_table_columns(frame)}\n{frame.to_csv(index=False, lineterminator=chr(10), float_format='%.6g').rstrip()}", read


KINDS: list[Kind] = [
    Kind("text", lambda ds: ds.family == "array" and ds.dtype.kind in "OSU", _strings_png, _strings_text, "strings"),
    line_kind("spectrum", lambda ds: ds.ndim == 1 and _real(ds) and energy_axis(ds.attrs) is not None, lambda ds: energy_axis(ds.attrs),
              "1-D data on a calibrated axis (EDS, EELS, Raman, a diffractogram ...); its significant peaks listed strongest first, "
              "with candidate X-ray lines for an X-ray spectrum"),
    Kind("composition", _has_elements, _composition_png, _composition_text, "one value per element"),
    line_kind("profile", lambda ds: ds.ndim == 1 and _real(ds), lambda ds: None, "any other 1-D data, against channel index"),
    image_kind("label_image", lambda ds: ds.ndim == 2 and ds.dtype.kind in "ui" and ds.attrs.get("acquisition_type") == "segmentation",
               "segmentation labels", colormap="tab20"),
    image_kind("image", lambda ds: _real(ds) and (ds.ndim == 2 or (ds.ndim == 3 and ds.shape[2] in (3, 4) and not _is_spectrum_image(ds))),
               "a picture (H x W, or H x W x 3/4 colour): contrast-stretched, downsampled"),
    Kind("spectrum_image", _is_spectrum_image, _spectrum_image_png, _spectrum_image_text, "a spectrum at each scan position"),
    Kind("stack", lambda ds: ds.ndim >= 3 and _real(ds), _stack_png, _stack_text, "frames (tilt series, 4-D STEM): one frame shown"),
    Kind("table", lambda ds: ds.family == "table", _table_png, _table_text, "rows and columns"),
]


def kind_names() -> list[str]:
    """Every kind list_data can report: the registered ones, plus map and unreadable."""
    return [k.name for k in KINDS] + ["map", "unreadable"]


def register_kind(kind: Kind, *, before: str | None = None) -> None:
    """Add (or replace) a kind; with ``before``, it is tried ahead of that existing kind."""
    names = [k.name for k in KINDS if k.name != kind.name]
    if before is not None and before not in names:
        raise ValueError(f"register_kind(before={before!r}): no such kind; kinds are {names}")
    KINDS[:] = [k for k in KINDS if k.name != kind.name]
    KINDS.insert(names.index(before) if before is not None else len(KINDS), kind)


def detect(ds: Dataset) -> Kind:
    for kind in KINDS:
        if kind.matches(ds):
            return kind
    raise ValueError(f"{ds.name}: no view for a {ds.family} of shape {list(ds.shape)} and type {ds.dtype}")


# ----------------------------------------------------------------- what the tools call


def _map_text(node: Any) -> str:
    meta = _attrs(node)
    return (
        f"stitched map of {meta.get('run')}: canvas {meta.get('canvas_shape')} pixels at {_fmt(float(meta.get('pixel_size_m') or 0))} m, "
        f"origin {meta.get('origin_xy_m')} m, {meta.get('levels')} levels. "
        "Its points of interest belong to the run (every version of the map shares them). "
        "For stage coordinates, crops, points and other versions use map_info, get_region, list_pois and list_maps."
    )


def view(node: Any, key: str, fmt: str = "png", dataset: str | None = None, opts: ViewOptions | None = None) -> tuple[dict, bytes | None]:
    """Look at ``key``: (summary with its text, PNG bytes or None). fmt is "png" or "text".

    In png format the text still carries the kind's short summary (statistics, peaks),
    so an agent that cannot see the picture learns what it shows.
    """
    if fmt not in ("png", "text"):
        raise ValueError('format is "png" or "text"')
    opts = opts or ViewOptions()
    opts.max_side = int(min(max(64, opts.max_side), 1024))
    opts.max_rows = int(min(max(1, opts.max_rows), 1000))
    group = resolve(node, key, dataset) if dataset else node
    ds = pick(node, key, dataset, opts.max_side)
    kind = detect(ds)
    summary: dict[str, Any] = {"key": key, "dataset": ds.path, "kind": kind.name, "shape": list(ds.shape), "dtype": str(ds.dtype)}
    header = [f"{key}" + (f" / {ds.path}" if ds.path else "") + f": {kind.name}, shape {list(ds.shape)}, {ds.dtype}"]
    for attr in ("acquisition_type", "detector", "units"):
        if attr in ds.attrs:
            header.append(f"{attr}: {ds.attrs[attr]}")
    if _family(group) == "container":
        where = f"{dataset.strip('/')}/" if dataset else ""
        listing = _map_text(group) if is_map(group) else f"{where or key} contains:\n" + container_listing(group, where) + \
            f"\nshowing {ds.path}; choose another with dataset"
        header.insert(0, listing)
    png = None
    if fmt == "png":
        drawn = kind.png(ds, opts)
        png, read = drawn[0], drawn[1]
        body = drawn[2] if len(drawn) > 2 else ""
    else:
        body, read = kind.text(ds, opts)
    summary["read"] = read
    summary["text"] = _cap("\n".join(header + [read] + ([body] if body else [])))
    return summary, png


def summarize(node: Any, key: str) -> dict:
    """What list_data shows for one key: its kind, shape and detector, from structure only (one short walk)."""
    if is_map(node):
        meta = _attrs(node)
        return {"key": key, "kind": "map", "shape": meta.get("canvas_shape"), "run": meta.get("run")}
    ds = pick(node, key)
    entry = {"key": key, "kind": detect(ds).name, "dataset": ds.path, "shape": list(ds.shape), "dtype": str(ds.dtype)}
    for attr in ("acquisition_type", "detector"):
        value = ds.attrs.get(attr, ds.parent_attrs.get(attr))
        if value is not None:
            entry[attr] = value
    if _family(node) == "container":
        entry["entries"] = _count(node)
    return entry
