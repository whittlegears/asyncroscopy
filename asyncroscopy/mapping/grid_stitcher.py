"""
Grid-aware stitching for rectangular STEM montages.

The classic :class:`~stem_stitcher.stitcher.ImageStitcher` registers each tile
against the *previous* tile only, which works for a horizontal strip but drifts
badly on an N x M grid where tiles also overlap vertically.

:class:`GridStitcher` instead:

1. Measures the translation between every pair of *grid neighbours*
   (left-right and top-bottom) with sub-pixel phase correlation on the
   expected overlap strips.
2. Solves a single weighted least-squares problem for the global position of
   every tile, so every overlap constraint is honoured simultaneously and
   registration errors cannot accumulate along a chain.
3. Optionally matches tile intensities in the overlap regions and blends with
   distance-transform feathering.

STEM stage montages are translation-only to a very good approximation, so a
pure-translation model is both faster and far more robust on noisy
low-contrast tiles than homography estimation from ORB features.

Usage
-----
::

    tiles = {(r, c): img for ...}          # 2-D numpy arrays
    result = GridStitcher(overlap=0.15).stitch(tiles)
    canvas = result.canvas
    result.positions[(1, 2)]               # (x, y) of that tile on the canvas
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GridKey = Tuple[int, int]


@dataclass
class EdgeMeasurement:
    """Measured translation between two neighbouring tiles.

    ``offset`` is the position of tile *b*'s origin relative to tile *a*'s
    origin, in pixels ``(dx, dy)``.  ``response`` is the phase-correlation
    peak response (higher is better); ``used`` is False when the measurement
    was rejected and the nominal grid offset was substituted.
    """

    a: GridKey
    b: GridKey
    offset: Tuple[float, float]
    response: float
    used: bool
    residual: float = 0.0


@dataclass
class GridStitchResult:
    """Output of :meth:`GridStitcher.stitch`."""

    canvas: np.ndarray
    positions: Dict[GridKey, Tuple[float, float]]
    edges: List[EdgeMeasurement] = field(default_factory=list)
    tile_shapes: Dict[GridKey, Tuple[int, int]] = field(default_factory=dict)

    @property
    def mean_residual(self) -> float:
        """Mean absolute residual (px) of used edges after the global solve."""
        used = [e.residual for e in self.edges if e.used]
        return float(np.mean(used)) if used else 0.0


class GridStitcher:
    """Stitch a rectangular grid of overlapping tiles into one mosaic.

    Parameters
    ----------
    overlap:
        Nominal fractional overlap between adjacent tiles (0-0.5).  Used to
        pick the correlation strips and as the fallback offset when a
        measurement is rejected.
    blend_mode:
        ``"feather"`` (distance-weighted average, default), ``"average"`` or
        ``"max"``.
    min_response:
        Minimum phase-correlation response to accept a measurement.
    max_deviation:
        Maximum allowed deviation (as a fraction of tile size) between a
        measured offset and the nominal grid offset before the measurement is
        rejected as an outlier.
    intensity_match:
        When True (default), each tile receives a linear gain/offset so that
        overlap statistics agree with its neighbours before blending.
    subpixel:
        When True (default), tiles are placed with sub-pixel accuracy using
        bilinear warping; otherwise positions are rounded to whole pixels.
    """

    def __init__(
        self,
        overlap: float = 0.15,
        blend_mode: str = "feather",
        min_response: float = 0.03,
        max_deviation: float = 0.35,
        intensity_match: bool = True,
        subpixel: bool = True,
    ) -> None:
        if not 0.0 < overlap < 0.6:
            raise ValueError("overlap must be between 0 and 0.6")
        self.overlap = overlap
        self.blend_mode = blend_mode
        self.min_response = min_response
        self.max_deviation = max_deviation
        self.intensity_match = intensity_match
        self.subpixel = subpixel

    # ------------------------------------------------------------------ API

    def stitch(
        self,
        tiles: Mapping[GridKey, np.ndarray] | Sequence[Sequence[np.ndarray]],
        progress_callback=None,
    ) -> GridStitchResult:
        """Stitch *tiles* and return the mosaic.

        Parameters
        ----------
        tiles:
            Either a mapping ``{(row, col): image}`` or a 2-D nested sequence
            ``tiles[row][col]``.  Rows/cols need not start at zero and gaps
            are allowed (missing grid positions are simply absent from the
            mosaic).
        progress_callback:
            Optional ``f(step, total, message)`` for GUI progress bars.
        """
        grid = self._normalise_input(tiles)
        if not grid:
            raise ValueError("No tiles provided.")

        keys = sorted(grid.keys())
        n = len(keys)
        total = n + 2

        def _report(step: int, msg: str) -> None:
            if progress_callback is not None:
                progress_callback(step, total, msg)

        greys = {k: self._to_float_grey(grid[k]) for k in keys}

        _report(0, "Measuring neighbour offsets ...")
        edges = self._measure_edges(greys, _report)

        _report(n, "Solving global positions ...")
        positions = self._solve_positions(greys, edges)

        for e in edges:
            pa, pb = positions[e.a], positions[e.b]
            e.residual = math.hypot(
                (pb[0] - pa[0]) - e.offset[0], (pb[1] - pa[1]) - e.offset[1]
            )

        gains: Dict[GridKey, Tuple[float, float]] = {k: (1.0, 0.0) for k in keys}
        if self.intensity_match:
            gains = self._match_intensity(greys, positions, edges)

        _report(n + 1, "Rendering canvas ...")
        canvas = self._render(grid, positions, gains)

        _report(total, "Done.")
        return GridStitchResult(
            canvas=canvas,
            positions=positions,
            edges=edges,
            tile_shapes={k: grid[k].shape[:2] for k in keys},
        )

    # ------------------------------------------------------- input handling

    @staticmethod
    def _normalise_input(
        tiles: Mapping[GridKey, np.ndarray] | Sequence[Sequence[np.ndarray]],
    ) -> Dict[GridKey, np.ndarray]:
        if isinstance(tiles, Mapping):
            return {tuple(k): np.asarray(v) for k, v in tiles.items() if v is not None}
        grid: Dict[GridKey, np.ndarray] = {}
        for r, row in enumerate(tiles):
            for c, img in enumerate(row):
                if img is not None:
                    grid[(r, c)] = np.asarray(img)
        return grid

    @staticmethod
    def _to_float_grey(image: np.ndarray) -> np.ndarray:
        img = image
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        if hi > lo:
            img = (img - lo) / (hi - lo)
        return img

    # --------------------------------------------------- pairwise measuring

    def _measure_edges(self, greys: Dict[GridKey, np.ndarray], _report) -> List[EdgeMeasurement]:
        edges: List[EdgeMeasurement] = []
        keys = set(greys.keys())
        for i, (r, c) in enumerate(sorted(keys)):
            _report(i, f"Registering neighbours of tile ({r}, {c}) ...")
            if (r, c + 1) in keys:
                edges.append(self._measure_pair(greys, (r, c), (r, c + 1), horizontal=True))
            if (r + 1, c) in keys:
                edges.append(self._measure_pair(greys, (r, c), (r + 1, c), horizontal=False))
        return edges

    def _measure_pair(
        self,
        greys: Dict[GridKey, np.ndarray],
        ka: GridKey,
        kb: GridKey,
        horizontal: bool,
    ) -> EdgeMeasurement:
        a, b = greys[ka], greys[kb]
        ha, wa = a.shape
        hb, wb = b.shape

        if horizontal:
            nominal = (wa * (1.0 - self.overlap), 0.0)
            sw = int(round(wa * min(1.0, 1.5 * self.overlap)))
            sw = max(32, min(sw, wa, wb))
            strip_a = a[:, wa - sw:]
            strip_b = b[:, :sw]
            strip_a_origin = (wa - sw, 0)
        else:
            nominal = (0.0, ha * (1.0 - self.overlap))
            sh = int(round(ha * min(1.0, 1.5 * self.overlap)))
            sh = max(32, min(sh, ha, hb))
            strip_a = a[ha - sh:, :]
            strip_b = b[:sh, :]
            strip_a_origin = (0, ha - sh)

        h = min(strip_a.shape[0], strip_b.shape[0])
        w = min(strip_a.shape[1], strip_b.shape[1])
        strip_a = strip_a[:h, :w]
        strip_b = strip_b[:h, :w]

        offset, response = self._phase_correlate(strip_a, strip_b, strip_a_origin)

        tol_x = self.max_deviation * wa
        tol_y = self.max_deviation * ha
        ok = (
            offset is not None
            and response >= self.min_response
            and abs(offset[0] - nominal[0]) <= tol_x
            and abs(offset[1] - nominal[1]) <= tol_y
        )
        if not ok:
            logger.warning(
                "Edge %s->%s: measurement rejected (offset=%s, response=%.4f); "
                "using nominal grid offset.",
                ka,
                kb,
                offset,
                response,
            )
            return EdgeMeasurement(a=ka, b=kb, offset=nominal, response=0.0, used=False)
        return EdgeMeasurement(a=ka, b=kb, offset=offset, response=response, used=True)

    @staticmethod
    def _phase_correlate(
        strip_a: np.ndarray, strip_b: np.ndarray, strip_a_origin: Tuple[int, int]
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Return tile-B origin relative to tile-A origin, from overlap strips.

        ``cv2.phaseCorrelate(a, b)`` returns the shift ``s`` such that
        ``b(p) == a(p - s)``, i.e. b's origin sits at ``-s`` in a's frame.

        Strips are Hanning-windowed and zero-padded to twice their size
        before correlating; without the padding the FFT is circular and a
        large true shift aliases to a spurious wrapped-around peak.
        """
        h, w = strip_a.shape
        if h < 8 or w < 8:
            return None, 0.0
        window = cv2.createHanningWindow((w, h), cv2.CV_64F)
        pa = np.zeros((2 * h, 2 * w), dtype=np.float64)
        pb = np.zeros((2 * h, 2 * w), dtype=np.float64)
        pa[:h, :w] = (strip_a - strip_a.mean()) * window
        pb[:h, :w] = (strip_b - strip_b.mean()) * window
        try:
            (sx, sy), response = cv2.phaseCorrelate(pa, pb)
        except cv2.error:
            return None, 0.0
        if not (np.isfinite(sx) and np.isfinite(sy)):
            return None, 0.0
        ox = strip_a_origin[0] - sx
        oy = strip_a_origin[1] - sy
        return (float(ox), float(oy)), float(response)

    # ------------------------------------------------------- global solving

    def _solve_positions(
        self, greys: Dict[GridKey, np.ndarray], edges: List[EdgeMeasurement]
    ) -> Dict[GridKey, Tuple[float, float]]:
        """Weighted least squares for tile positions from pairwise offsets.

        Every edge contributes ``pos_b - pos_a = offset`` with weight from its
        correlation response; a weak prior pulls each tile towards its nominal
        grid position so the system stays well-conditioned even when whole
        edges fail or the grid is disconnected.
        """
        keys = sorted(greys.keys())
        index = {k: i for i, k in enumerate(keys)}
        n = len(keys)

        rows: List[np.ndarray] = []
        bx: List[float] = []
        by: List[float] = []

        for e in edges:
            w = max(e.response, 1e-3) if e.used else 0.05
            row = np.zeros(n)
            row[index[e.b]] = w
            row[index[e.a]] = -w
            rows.append(row)
            bx.append(w * e.offset[0])
            by.append(w * e.offset[1])

        prior_w = 1e-3
        ref_h, ref_w = next(iter(greys.values())).shape
        step_x = ref_w * (1.0 - self.overlap)
        step_y = ref_h * (1.0 - self.overlap)
        r0 = min(k[0] for k in keys)
        c0 = min(k[1] for k in keys)
        for k in keys:
            row = np.zeros(n)
            row[index[k]] = prior_w
            rows.append(row)
            bx.append(prior_w * (k[1] - c0) * step_x)
            by.append(prior_w * (k[0] - r0) * step_y)

        A = np.vstack(rows)
        xs, *_ = np.linalg.lstsq(A, np.array(bx), rcond=None)
        ys, *_ = np.linalg.lstsq(A, np.array(by), rcond=None)

        xs -= xs.min()
        ys -= ys.min()
        return {k: (float(xs[index[k]]), float(ys[index[k]])) for k in keys}

    # ---------------------------------------------------- intensity matching

    def _match_intensity(
        self,
        greys: Dict[GridKey, np.ndarray],
        positions: Dict[GridKey, Tuple[float, float]],
        edges: List[EdgeMeasurement],
    ) -> Dict[GridKey, Tuple[float, float]]:
        """Per-tile linear gain/offset from overlap statistics.

        Tiles are visited breadth-first from the anchor tile; each new tile is
        matched (mean/std) against the already-adjusted overlap pixels of its
        visited neighbours.
        """
        keys = sorted(greys.keys())
        gains: Dict[GridKey, Tuple[float, float]] = {keys[0]: (1.0, 0.0)}
        neighbours: Dict[GridKey, List[GridKey]] = {k: [] for k in keys}
        for e in edges:
            neighbours[e.a].append(e.b)
            neighbours[e.b].append(e.a)

        queue = [keys[0]]
        while queue:
            cur = queue.pop(0)
            for nb in neighbours[cur]:
                if nb in gains:
                    continue
                stats = self._overlap_stats(greys, positions, cur, nb)
                if stats is None:
                    gains[nb] = (1.0, 0.0)
                else:
                    mean_a, std_a, mean_b, std_b = stats
                    g_cur, o_cur = gains[cur]
                    mean_a = g_cur * mean_a + o_cur
                    std_a = g_cur * std_a
                    gain = std_a / std_b if std_b > 1e-6 else 1.0
                    gain = float(np.clip(gain, 0.5, 2.0))
                    offset = mean_a - gain * mean_b
                    gains[nb] = (gain, float(offset))
                queue.append(nb)

        for k in keys:
            gains.setdefault(k, (1.0, 0.0))
        return gains

    @staticmethod
    def _overlap_stats(
        greys: Dict[GridKey, np.ndarray],
        positions: Dict[GridKey, Tuple[float, float]],
        ka: GridKey,
        kb: GridKey,
    ) -> Optional[Tuple[float, float, float, float]]:
        a, b = greys[ka], greys[kb]
        ax, ay = positions[ka]
        bx, by = positions[kb]
        x0 = max(ax, bx)
        y0 = max(ay, by)
        x1 = min(ax + a.shape[1], bx + b.shape[1])
        y1 = min(ay + a.shape[0], by + b.shape[0])
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        pa = a[int(y0 - ay): int(y1 - ay), int(x0 - ax): int(x1 - ax)]
        pb = b[int(y0 - by): int(y1 - by), int(x0 - bx): int(x1 - bx)]
        h = min(pa.shape[0], pb.shape[0])
        w = min(pa.shape[1], pb.shape[1])
        if h < 4 or w < 4:
            return None
        pa, pb = pa[:h, :w], pb[:h, :w]
        return float(pa.mean()), float(pa.std()), float(pb.mean()), float(pb.std())

    # -------------------------------------------------------------- render

    def _render(
        self,
        grid: Dict[GridKey, np.ndarray],
        positions: Dict[GridKey, Tuple[float, float]],
        gains: Dict[GridKey, Tuple[float, float]],
    ) -> np.ndarray:
        keys = sorted(grid.keys())
        is_colour = any(grid[k].ndim == 3 for k in keys)

        x_max = max(positions[k][0] + grid[k].shape[1] for k in keys)
        y_max = max(positions[k][1] + grid[k].shape[0] for k in keys)
        cw = int(math.ceil(x_max)) + 1
        ch = int(math.ceil(y_max)) + 1

        acc = np.zeros((ch, cw, 3) if is_colour else (ch, cw), dtype=np.float64)
        weight = np.zeros((ch, cw), dtype=np.float64)

        for k in keys:
            tile = grid[k]
            if is_colour and tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            elif not is_colour and tile.ndim == 3:
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            tile_f = tile.astype(np.float64)

            gain, offset = gains.get(k, (1.0, 0.0))
            if gain != 1.0 or offset != 0.0:
                scale = 255.0 if grid[k].dtype == np.uint8 else 1.0
                tile_f = gain * tile_f + offset * scale

            h, w = tile_f.shape[:2]
            px, py = positions[k]

            if self.blend_mode == "feather":
                fmask = self._feather_mask(h, w)
            else:
                fmask = np.ones((h, w), dtype=np.float64)

            if self.subpixel:
                fx, fy = px - math.floor(px), py - math.floor(py)
                ix, iy = int(math.floor(px)), int(math.floor(py))
                if fx > 1e-3 or fy > 1e-3:
                    M = np.float64([[1, 0, fx], [0, 1, fy]])
                    tile_f = cv2.warpAffine(
                        tile_f, M, (w + 1, h + 1), flags=cv2.INTER_LINEAR
                    )
                    fmask = cv2.warpAffine(
                        fmask, M, (w + 1, h + 1), flags=cv2.INTER_LINEAR
                    )
                    h, w = tile_f.shape[:2]
            else:
                ix, iy = int(round(px)), int(round(py))

            y0, x0 = max(0, iy), max(0, ix)
            y1, x1 = min(ch, iy + h), min(cw, ix + w)
            if y1 <= y0 or x1 <= x0:
                continue
            ty0, tx0 = y0 - iy, x0 - ix
            sub = tile_f[ty0: ty0 + (y1 - y0), tx0: tx0 + (x1 - x0)]
            msub = fmask[ty0: ty0 + (y1 - y0), tx0: tx0 + (x1 - x0)]

            if self.blend_mode == "max":
                if is_colour:
                    acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], sub)
                else:
                    acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], sub)
                weight[y0:y1, x0:x1] = np.maximum(weight[y0:y1, x0:x1], msub)
            else:
                if is_colour:
                    acc[y0:y1, x0:x1] += sub * msub[..., None]
                else:
                    acc[y0:y1, x0:x1] += sub * msub
                weight[y0:y1, x0:x1] += msub

        valid = weight > 1e-9
        if self.blend_mode != "max":
            if is_colour:
                for ch_i in range(3):
                    acc[:, :, ch_i][valid] /= weight[valid]
            else:
                acc[valid] /= weight[valid]

        ref_dtype = grid[keys[0]].dtype
        if ref_dtype == np.uint8:
            return np.clip(acc, 0, 255).astype(np.uint8)
        if ref_dtype == np.uint16:
            return np.clip(acc, 0, 65535).astype(np.uint16)
        return acc.astype(np.float32)

    @staticmethod
    def _feather_mask(h: int, w: int) -> np.ndarray:
        """Weight mask that ramps down towards the tile borders."""
        mask = np.ones((h + 2, w + 2), dtype=np.uint8)
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)[1:-1, 1:-1]
        ramp = min(h, w) / 8.0
        return np.minimum(dist / ramp, 1.0).astype(np.float64) + 1e-4
