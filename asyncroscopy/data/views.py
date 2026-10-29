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
PEAK_SIGMA = 5.0  # a peak must stand this many noise sigmas above its local background
MAX_STRINGS = 50


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
    png: Callable[[Dataset, ViewOptions], tuple[bytes, str]]
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
            walk(value, f"{path}/{child}" if path else child, depth + 1, attrs)
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
        found = _first_dataset(value, key, f"{path}/{child}" if path else child, attrs, depth + 1)
        if found is not None:
            return found
    return None


def is_map(node: Any) -> bool:
    return _family(node) == "container" and _attrs(node).get("kind") == "stitched_map"


def pick(node: Any, key: str, dataset: str | None = None, max_side: int = PREVIEW_MAX_SIDE) -> Dataset:
    """The dataset to show for ``key``: ``dataset`` if given, else an image, a spectrum, a map level, the first."""
    if _family(node) in ("array", "table"):
        return as_dataset(node, key)
    if dataset:
        current = node
        for part in dataset.strip("/").split("/"):
            try:
                current = current[part]
            except KeyError:
                raise ValueError(f"{key!r} has no dataset {dataset!r}; it has:\n{container_listing(node)}") from None
        return as_dataset(current, key, dataset.strip("/"), _attrs(node))
    if is_map(node):
        levels = sorted((d for d in datasets(node, key, limit=40) if d.path.startswith("level_") and d.ndim == 2), key=lambda d: d.nbytes)
        fitting = [d for d in levels if max(d.shape) <= 2 * max_side]
        if levels:
            return fitting[-1] if fitting else levels[0]
    for prefix in ("image", "spectrum_image", "spectrum"):
        try:
            child = node[prefix]
        except (KeyError, TypeError):
            continue
        found = _first_dataset(child, key, prefix, _attrs(node))
        if found is not None:
            return found
    found = _first_dataset(node, key)
    if found is None:
        raise ValueError(f"{key!r} holds no arrays or tables")
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
            continue
        count = _count(child)
        first = next(iter(_children(child)), None)
        if first is None:
            lines.append(f"{prefix}{name}/: empty")
        elif _family(first[1]) in ("array", "table"):
            ds = as_dataset(first[1], "", first[0])
            lines.append(f"{prefix}{name}/: {count if count is not None else '?'} x {tuple(ds.shape)} {ds.dtype} (shape of {first[0]})")
        elif depth < MAX_DEPTH:
            lines.append(f"{prefix}{name}/: {count if count is not None else '?'} groups")
            lines.append(container_listing(child, f"{prefix}{name}/", depth + 1))
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


def _finite_region(region: list[float], length: int, what: str) -> list[float]:
    if len(region) != length:
        raise ValueError(what)
    values = [float(v) for v in region]
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


# ----------------------------------------------------------------- 1-D: spectra and profiles


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


def find_peaks(values: np.ndarray, limit: int = MAX_PEAKS) -> list[int]:
    """Indices of significant peaks, most significant first; none in pure noise.

    A peak must stand PEAK_SIGMA noise sigmas above a running-median background and be
    wider than one channel. For count data the noise is at least sqrt(background).
    """
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks as scipy_peaks

    data = np.nan_to_num(np.asarray(values, dtype=np.float64))
    if data.size < 9 or data.max() == data.min():
        return []
    window = max(9, data.size // 40) | 1
    background = median_filter(data, size=window, mode="nearest")
    residual = data - background
    sigma = np.full(data.size, 1.4826 * np.median(np.abs(residual - np.median(residual))))
    if data.min() >= 0 and np.allclose(data, np.round(data)):  # counts: Poisson noise
        sigma = np.maximum(sigma, np.sqrt(np.maximum(background, 1.0)))
    sigma = np.maximum(sigma, 1e-3 * (data.max() - data.min()))
    candidates, _ = scipy_peaks(data, width=1.5, distance=max(2, data.size // 200))
    significance = residual[candidates] / sigma[candidates]
    keep = candidates[significance > PEAK_SIGMA]
    order = np.argsort(significance[significance > PEAK_SIGMA])[::-1][:limit]
    return [int(keep[i]) for i in order]


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


def _is_xray(ds: Dataset) -> bool:
    words = " ".join(str(ds.attrs.get(k, "")) + " " + str(ds.parent_attrs.get(k, "")) for k in ("detector", "acquisition_type", "technique")).lower()
    return "eds" in words or "edx" in words or "x-ray" in words or "xray" in words


def _real(ds: Dataset) -> bool:
    return ds.dtype.kind in "uifcb"


def line_kind(name: str, matches: Callable[[Dataset], bool], axis: Callable[[Dataset], Axis | None], description: str = "") -> Kind:
    """A kind for 1-D data plotted against a calibrated axis, with its significant peaks marked and listed."""

    def x_values(ds: Dataset, start: int, step: int, count: int) -> tuple[np.ndarray, Axis]:
        a = axis(ds) or Axis(0.0, 1.0, "channel", "")
        return a.offset + a.scale * (start + step * np.arange(count)), a

    def scale_for(a: Axis, x: np.ndarray) -> tuple[float, str]:
        if a.units == "eV" and x.size and np.nanmax(np.abs(x)) > 2000:
            return 1e-3, "keV"
        return 1.0, a.units

    def peak_rows(ds: Dataset, x: np.ndarray, y: np.ndarray, a: Axis) -> list[tuple[float, float, str]]:
        tolerance = max(2 * abs(a.scale), 30.0)
        rows = []
        for i in find_peaks(y):
            candidates = xray_candidates(x[i], tolerance) if a.units == "eV" and _is_xray(ds) else []
            rows.append((float(x[i]), float(y[i]), ", ".join(candidates)))
        return rows

    def png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str]:
        y, start, step, read = read_line(ds, opts, axis(ds))
        x, a = x_values(ds, start, step, y.size)
        factor, units = scale_for(a, x)
        figure = Figure(figsize=(6.4, 3.6), dpi=100)
        plot = figure.add_subplot()
        plot.plot(x * factor, y, linewidth=0.9)
        for px, py, label in peak_rows(ds, x, y, a):
            plot.annotate(f"{px * factor:.3g}" + (f"\n{label.split(',')[0]}" if label else ""), (px * factor, py),
                          textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
        plot.set_xlabel(f"{a.label} ({units})" if units else a.label)
        plot.set_ylabel(str(ds.attrs.get("units", "counts")))
        plot.set_title(ds.name, fontsize=8)
        figure.tight_layout()
        return _png(figure), read

    def text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
        y, start, step, read = read_line(ds, opts, axis(ds))
        x, a = x_values(ds, start, step, y.size)
        unit = f" {a.units}" if a.units else ""
        lines = [
            f"{ds.shape[0]} channels; {a.label} {_fmt(x[0])} to {_fmt(x[-1])}{unit} ({_fmt(a.scale * step)}{unit} per value shown)" if y.size else "empty",
            f"total {_fmt(float(np.nansum(y)))}; {_stats(y)}",
        ]
        peaks = peak_rows(ds, x, y, a)
        lines += ["peaks (strongest first):", _csv(f"{a.label}{unit},value,candidates", [(px, py, f'"{c}"' if c else "") for px, py, c in peaks])] if peaks \
            else [f"no peaks stand {PEAK_SIGMA:g} sigma above the background"]
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


def _composition_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str]:
    values = _read(ds, (slice(0, ds.shape[0]),))
    figure = Figure(figsize=(6.0, 3.5), dpi=100)
    plot = figure.add_subplot()
    plot.bar(np.arange(values.size), values)
    plot.set_xticks(np.arange(values.size))
    plot.set_xticklabels(ds.attrs["elements"])
    plot.set_xlabel("element")
    plot.set_ylabel("relative intensity")
    figure.tight_layout()
    return _png(figure), f"read [0:{ds.shape[0]}]"


def _composition_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    values = _read(ds, (slice(0, ds.shape[0]),))
    return "\n".join(f"{element}: {_fmt(float(v))}" for element, v in zip(ds.attrs["elements"], values)), f"read [0:{ds.shape[0]}]"


def _has_elements(ds: Dataset) -> bool:
    elements = ds.attrs.get("elements")
    return ds.ndim == 1 and _real(ds) and isinstance(elements, list) and len(elements) == ds.shape[0] and all(isinstance(e, str) for e in elements)


# ----------------------------------------------------------------- strings


def _strings(ds: Dataset) -> tuple[list[str], str]:
    if not ds.shape:
        return [str(_read(ds, ()))], "read all"
    slices = (slice(0, min(ds.shape[0], MAX_STRINGS)), *(slice(0, min(n, 4)) for n in ds.shape[1:]))
    values = _read(ds, slices).reshape(-1)
    return [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in values], "read " + _describe_slices(slices)


def _strings_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str]:
    raise ValueError(f"{ds.name} holds text ({ds.dtype}), which has no picture; use format text")


def _strings_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    values, read = _strings(ds)
    lines = [f"{list(ds.shape)} text entries; the first {len(values)}:"] + [v[:300] for v in values]
    return "\n".join(lines), read


# ----------------------------------------------------------------- 2-D and up


def image_kind(name: str, matches: Callable[[Dataset], bool], description: str = "", colormap: str | None = None) -> Kind:
    """A kind for pictures: percentile contrast, strided read, optional colour map for labels."""

    def png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str]:
        values, read = read_image(ds, opts, target=opts.max_side)
        if colormap:
            from matplotlib import colormaps

            labels = np.asarray(values, dtype=np.int64)
            rgba = colormaps[colormap](labels % 20 / 19.0)
            rgba[labels == 0] = (0, 0, 0, 1)
            return _png(to_pixels((rgba[..., :3] * 255).astype(np.uint8), opts.max_side)), read
        return _png(to_pixels(values, opts.max_side)), read

    def text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
        values, read = read_image(ds, opts, target=1024)
        lines = [f"{ds.shape[0]} x {ds.shape[1]} pixels{' x ' + str(ds.shape[2]) if ds.ndim == 3 else ''}, {ds.dtype}"]
        if ds.dtype.kind == "c":
            lines.append("complex values: statistics are of the magnitude")
        pixel = ds.attrs.get("pixel_size_m")
        if isinstance(pixel, (int, float)):
            lines.append(f"pixel {_fmt(pixel)} m; field of view {_fmt(pixel * ds.shape[1])} x {_fmt(pixel * ds.shape[0])} m")
        if colormap:
            labels = np.unique(np.asarray(values, dtype=np.int64))
            lines.append(f"{int((labels != 0).sum())} labelled areas")
            stats = ds.attrs.get("area_statistics")
            if stats:
                lines.append("area_statistics: " + json.dumps(stats)[:2000])
            return "\n".join(lines), read
        if ds.dtype.kind == "b":
            lines.append(f"true: {int(values.sum())} of {values.size} pixels read")
            return "\n".join(lines), read
        lines.append(_stats(values))
        if ds.dtype.kind in "ui":
            top = np.iinfo(ds.dtype).max
            lines.append(f"saturated (= {top}): {_fmt(100.0 * float((values == top).mean()))}%")
        histogram = _histogram(values)
        if histogram:
            lines += ["histogram (value: count):", histogram]
        grid_step = max(1, math.ceil(max(values.shape[:2]) / 16))
        grid = np.asarray(values[::grid_step, ::grid_step], dtype=np.float64)
        if grid.ndim == 3:
            grid = grid.mean(axis=2)
        lines += [f"every {grid_step}th pixel of what was read (rows top to bottom):",
                  "\n".join(",".join(_fmt(v) for v in row) for row in grid[: opts.max_rows])]
        return "\n".join(lines), read

    return Kind(name, matches, png, text, description)


def _frame(ds: Dataset, opts: ViewOptions) -> tuple[int, ...]:
    """Position on the leading axes of a stack: opts.index, else the middle."""
    lead = ds.shape[:-2]
    if opts.index is None:
        return tuple(n // 2 for n in lead)
    if len(opts.index) != len(lead) or any(not 0 <= int(i) < n for i, n in zip(opts.index, lead)):
        raise ValueError(f"index must give {len(lead)} positions within {list(lead)}")
    return tuple(int(i) for i in opts.index)


def _stack_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str]:
    frame = _frame(ds, opts)
    values, read = read_image(ds, opts, target=opts.max_side, frame=frame)
    return _png(to_pixels(values, opts.max_side)), read


def _stack_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    frame = _frame(ds, opts)
    values, read = read_image(ds, opts, target=1024, frame=frame)
    lines = [f"shape {list(ds.shape)} {ds.dtype}; showing the frame at index {list(frame)} (pass index to choose another)"]
    for attr in ("scan_shape", "axes", "spectrum_image_axes"):
        if attr in ds.attrs:
            lines.append(f"{attr}: {ds.attrs[attr]}")
    lines.append(f"frame: {_stats(values)}")
    return "\n".join(lines), read


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
    spatial = min(max(r1 - r0, c1 - c0), math.ceil(math.sqrt(size / MAX_READ_BYTES))) if size > MAX_READ_BYTES else 1
    spatial = max(1, spatial)
    left = math.ceil((r1 - r0) / spatial) * math.ceil((c1 - c0) / spatial) * channels * ds.dtype.itemsize
    spectral = max(1, math.ceil(left / MAX_READ_BYTES))
    slices = (slice(r0, r1, spatial), slice(c0, c1, spatial), slice(0, channels, spectral))
    return _numeric(_read(ds, slices)), "read " + _describe_slices(slices), spectral


def _spectrum_at(ds: Dataset, opts: ViewOptions) -> Dataset:
    """The spectrum at index [row, col] of a spectrum image, as its own 1-D dataset."""
    if len(opts.index) != 2 or not (0 <= int(opts.index[0]) < ds.shape[0] and 0 <= int(opts.index[1]) < ds.shape[1]):
        raise ValueError(f"index for a spectrum image is one [row, col] position within {list(ds.shape[:2])}")
    row, col = int(opts.index[0]), int(opts.index[1])
    line = _Loaded(np.asarray(ds.node.read((row, col, slice(0, ds.shape[2])))), ds.attrs)
    return Dataset(ds.key, f"{ds.path}[{row},{col}]", line, (ds.shape[2],), ds.dtype, ds.attrs, ds.parent_attrs)


def _spectrum_image_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str]:
    if opts.index is not None:
        single = _spectrum_at(ds, opts)
        return detect(single).png(single, ViewOptions(max_side=opts.max_side))
    cube, read, step = _read_cube(ds, opts)
    axis = energy_axis(ds.attrs) or Axis(0.0, 1.0, "channel", "")
    figure = Figure(figsize=(8.0, 3.4), dpi=100)
    left, right = figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)
    left.imshow(np.asarray(to_pixels(cube.sum(axis=2), opts.max_side)), cmap="gray")
    left.set_title("summed intensity", fontsize=8)
    left.axis("off")
    right.plot(axis.values(0, ds.shape[2], step)[: cube.shape[2]], cube.sum(axis=(0, 1)), linewidth=0.9)
    right.set_xlabel(f"{axis.label} ({axis.units})" if axis.units else axis.label)
    right.set_title("summed spectrum", fontsize=8)
    figure.tight_layout()
    return _png(figure), read


def _spectrum_image_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    if opts.index is not None:
        single = _spectrum_at(ds, opts)
        body, read = detect(single).text(single, ViewOptions(max_rows=opts.max_rows))
        return f"spectrum at position {list(opts.index)}:\n{body}", read
    cube, read, step = _read_cube(ds, opts)
    axis = energy_axis(ds.attrs) or Axis(0.0, 1.0, "channel", "")
    summed = cube.sum(axis=(0, 1))
    x = axis.values(0, ds.shape[2], step)[: summed.size]
    lines = [f"spectrum image {ds.shape[0]} x {ds.shape[1]} positions x {ds.shape[2]} channels, {ds.dtype}",
             f"{axis.label} {_fmt(x[0])} to {_fmt(x[-1])} {axis.units}".rstrip(),
             "index [row, col] shows the spectrum at one position; region [row0, row1, col0, col1] sums part of it"]
    peaks = find_peaks(summed)
    if peaks:
        lines += ["peaks of the summed spectrum:", _csv(f"{axis.label},value", [(float(x[i]), float(summed[i])) for i in peaks])]
    return "\n".join(lines), read


TABLE_MAX_ROWS = 1000


def _table_frame(ds: Dataset, opts: ViewOptions):
    """The first rows of a table: only its first partition when Tiled splits it."""
    read_partition = getattr(ds.node, "read_partition", None)
    frame = read_partition(0) if callable(read_partition) else ds.node.read()
    rows = min(len(frame), opts.max_rows, TABLE_MAX_ROWS)
    return frame.head(rows), f"read first {rows} rows" + (" (of the first partition)" if callable(read_partition) else "")


def _table_png(ds: Dataset, opts: ViewOptions) -> tuple[bytes, str]:
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
    return _png(figure), read


def _table_text(ds: Dataset, opts: ViewOptions) -> tuple[str, str]:
    frame, read = _table_frame(ds, opts)
    columns = ", ".join(f"{c} ({frame[c].dtype})" for c in frame.columns)
    return f"columns: {columns}\n{frame.to_csv(index=False, lineterminator=chr(10), float_format='%.6g').rstrip()}", read


KINDS: list[Kind] = [
    Kind("text", lambda ds: ds.family == "array" and ds.dtype.kind in "OSU", _strings_png, _strings_text, "strings"),
    line_kind("spectrum", lambda ds: ds.ndim == 1 and _real(ds) and energy_axis(ds.attrs) is not None, lambda ds: energy_axis(ds.attrs),
              "1-D data on a calibrated axis (EDS, EELS); significant peaks listed, with candidate X-ray lines for EDS"),
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
        f"origin {meta.get('origin_xy_m')} m, {meta.get('levels')} levels, {len(meta.get('pois') or [])} points of interest. "
        "For stage coordinates, crops and points use map_info, get_region and list_pois."
    )


def view(node: Any, key: str, fmt: str = "png", dataset: str | None = None, opts: ViewOptions | None = None) -> tuple[dict, bytes | None]:
    """Look at ``key``: (summary with its text, PNG bytes or None). fmt is "png" or "text"."""
    if fmt not in ("png", "text"):
        raise ValueError('format is "png" or "text"')
    opts = opts or ViewOptions()
    opts.max_side = int(min(max(64, opts.max_side), 1024))
    opts.max_rows = int(min(max(1, opts.max_rows), 1000))
    ds = pick(node, key, dataset, opts.max_side)
    kind = detect(ds)
    summary: dict[str, Any] = {"key": key, "dataset": ds.path, "kind": kind.name, "shape": list(ds.shape), "dtype": str(ds.dtype)}
    header = [f"{key}" + (f" / {ds.path}" if ds.path else "") + f": {kind.name}, shape {list(ds.shape)}, {ds.dtype}"]
    for attr in ("acquisition_type", "detector", "units"):
        if attr in ds.attrs:
            header.append(f"{attr}: {ds.attrs[attr]}")
    if dataset is None and _family(node) == "container":
        header.insert(0, _map_text(node) if is_map(node) else "contains:\n" + container_listing(node) + "\n(choose one with dataset)")
    png = None
    if fmt == "png":
        png, read = kind.png(ds, opts)
        body = ""
    else:
        body, read = kind.text(ds, opts)
    summary["read"] = read
    summary["text"] = _cap("\n".join(header + [read] + ([body] if body else [])))
    return summary, png


def summarize(node: Any, key: str) -> dict:
    """What list_data shows for one key: its kind, shape and detector, from structure only (one short walk)."""
    if is_map(node):
        meta = _attrs(node)
        return {"key": key, "kind": "map", "shape": meta.get("canvas_shape"), "pois": len(meta.get("pois") or [])}
    ds = pick(node, key)
    entry = {"key": key, "kind": detect(ds).name, "dataset": ds.path, "shape": list(ds.shape), "dtype": str(ds.dtype)}
    for attr in ("acquisition_type", "detector"):
        value = ds.attrs.get(attr, ds.parent_attrs.get(attr))
        if value is not None:
            entry[attr] = value
    if _family(node) == "container":
        entry["entries"] = _count(node)
    return entry
