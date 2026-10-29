"""Sample drift: where a stitch's fitted drift puts the sample at a given time, and live drift trackers.

Pure numbers; no instrument. The Mapper's revisits use ``drift_at`` to put a mapped region
back under the probe; its live drift tracking (plan ``"drift": {"track": true}``, see
asyncroscopy/jobs/live_drift.py) uses a ``DriftTracker`` to predict, before each tile, how
far the image will sit from where it was planned, and ``tile_offset_m`` to measure it from
the overlap with a neighbouring tile. See docs/designs/concepts_brief.md (decision D1, item 8).

Every position here is the placement error of a tile: where its image really is minus
where it was planned, with the image shift the tracker applied taken out, relative to the
map's first tile. On an instrument that is stage error plus sample drift; the tracker
models it as a smooth function of the instrument's clock (``clock_s``).

A tracker is a small class registered by name, like the Mapper's instrument profiles and
the Tuner's strategies::

    from asyncroscopy.routines.drift import DriftTracker, register_tracker

    @register_tracker
    class Mine(DriftTracker):
        name = "mine"
        def predict(self, t_s): ...

chosen with the plan's ``"drift": {"tracker": "mine"}``; others install with the
``asyncroscopy.drift_trackers`` entry point. ``drift_bench`` scores one with no stack.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_WINDOW = 12  # observations the linear tracker fits (the most recent ones)
MIN_CONFIDENCE = 0.3  # normalised cross-correlation of the overlap below which a registration is no measurement


def drift_at(drift: dict | None, t_s: float) -> tuple[float, float]:
    """The stitch's fitted drift (x, y in metres) at ``t_s`` after the first tile; extrapolated from the last segment."""
    if not drift or len(drift.get("times_s", [])) < 2:
        return 0.0, 0.0
    times = np.asarray(drift["times_s"], dtype=np.float64)
    xy = np.asarray(drift["xy_m"], dtype=np.float64)
    if t_s <= times[-1]:
        return float(np.interp(t_s, times, xy[:, 0])), float(np.interp(t_s, times, xy[:, 1]))
    rate = (xy[-1] - xy[-2]) / max(times[-1] - times[-2], 1e-9)
    end = xy[-1] + rate * (t_s - times[-1])
    return float(end[0]), float(end[1])


# ----------------------------------------------------------------------
# Live trackers
# ----------------------------------------------------------------------
class DriftTracker:
    """Predicts a tile's placement error from the errors measured so far. Subclass, set ``name``, register it.

    ``observe(t_s, xy_m)`` gets each measured error (metres, relative to the first tile, at the
    instrument's clock ``t_s``); ``predict(t_s)`` returns the expected error then. ``state()``
    must return everything needed to continue after a restart (JSON-safe): the Mapper saves
    it with the job and passes it back as ``state``.
    """

    name = ""

    def __init__(self, settings: dict | None = None, state: dict | None = None):
        self.settings = dict(settings or {})
        self.observations: list[list[float]] = [list(map(float, o)) for o in (state or {}).get("observations", [])]

    def observe(self, t_s: float, xy_m) -> None:
        self.observations.append([float(t_s), float(xy_m[0]), float(xy_m[1])])

    def predict(self, t_s: float) -> tuple[float, float]:
        raise NotImplementedError

    def rate_m_per_s(self) -> tuple[float, float] | None:
        """The drift rate the tracker currently believes (x, y), or None if it has no rate."""
        return None

    def state(self) -> dict:
        return {"observations": self.observations}


class LinearTracker(DriftTracker):
    """Weighted least-squares line through the last ``window`` measurements: steady thermal drift plus stage noise.

    One measurement: that error, held. None: no error. The line is fitted on the clock, so a
    long stage move or a slow tile is extrapolated by its duration, not by a tile count.
    """

    name = "linear"

    def _recent(self) -> np.ndarray:
        window = int(self.settings.get("window", DEFAULT_WINDOW))
        kept = self.observations[-max(window, 2):]
        self.observations = kept  # the state stays bounded however long the map
        return np.asarray(kept, dtype=np.float64).reshape(-1, 3)

    def _fit(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        obs = self._recent()
        if len(obs) == 0:
            return None
        t = obs[:, 0]
        t0 = float(t.mean())
        if len(obs) < 2 or float(np.ptp(t)) <= 0:
            return obs[:, 1:].mean(axis=0), np.zeros(2), t0
        a = np.stack([np.ones_like(t), t - t0], axis=1)
        coef, *_ = np.linalg.lstsq(a, obs[:, 1:], rcond=None)
        return coef[0], coef[1], t0

    def predict(self, t_s: float) -> tuple[float, float]:
        fit = self._fit()
        if fit is None:
            return 0.0, 0.0
        offset, rate, t0 = fit
        xy = offset + rate * (float(t_s) - t0)
        return float(xy[0]), float(xy[1])

    def rate_m_per_s(self) -> tuple[float, float] | None:
        fit = self._fit()
        return None if fit is None else (float(fit[1][0]), float(fit[1][1]))


TRACKERS: dict[str, type[DriftTracker]] = {}
_entry_points_loaded = False


def register_tracker(cls: type[DriftTracker]) -> type[DriftTracker]:
    """Make ``cls`` available as the plan's ``"drift": {"tracker": cls.name}``; usable as a class decorator."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a name")
    TRACKERS[cls.name] = cls
    return cls


register_tracker(LinearTracker)


def trackers() -> dict[str, type[DriftTracker]]:
    """Every tracker: built in, registered, and installed with the ``asyncroscopy.drift_trackers`` entry point."""
    global _entry_points_loaded
    if not _entry_points_loaded:
        _entry_points_loaded = True
        from importlib.metadata import entry_points

        for point in entry_points(group="asyncroscopy.drift_trackers"):
            try:
                register_tracker(point.load())
            except Exception:  # a broken plug-in must not take the Mapper down
                pass
    return TRACKERS


def get_tracker(name: str) -> type[DriftTracker]:
    """The tracker class called ``name``; ValueError listing the known ones otherwise."""
    known = TRACKERS if name in TRACKERS else trackers()
    if name not in known:
        raise ValueError(f"unknown drift tracker {name!r}; registered: {sorted(known)}")
    return known[name]


# ----------------------------------------------------------------------
# Measuring: where a tile sits relative to an overlapping neighbour
# ----------------------------------------------------------------------
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def tile_offset_m(previous: np.ndarray, image: np.ndarray, pixel_m: float, expected_xy_m, bound_m: float, min_confidence: float = MIN_CONFIDENCE) -> tuple[float, float, float] | None:
    """Where ``image`` sits relative to ``previous`` (x, y in metres, and the overlap's NCC), searched within ``bound_m`` of ``expected_xy_m``.

    The same bounded, masked normalised cross-correlation the stitch uses (pyTEMlib's
    ``image_tools.mosaic.bounded_shift``), so the live tracker and the stitch measure
    tiles alike. Images are [row, column] with rows along +y and columns along +x.
    None when there is no confident peak inside the bound (a featureless overlap, or a
    drift bigger than the bound).
    """
    from pyTEMlib.image_tools.mosaic import bounded_shift

    a = np.asarray(previous, dtype=np.float64)
    b = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.std() == 0 or b.std() == 0:
        return None
    predicted = np.array([expected_xy_m[1], expected_xy_m[0]], dtype=np.float64) / pixel_m  # (row, col)
    rows, cols = a.shape
    height, width = rows - abs(predicted[0]), cols - abs(predicted[1])
    if height < 8 or width < 8:
        return None  # the tiles do not overlap enough to compare
    bound = min(float(bound_m) / pixel_m, 0.6 * min(height, width))
    shift = bounded_shift(a, b, predicted, max(bound, 2.0), 0.5 * height * width / a.size)
    if shift is None:
        return None
    dr, dc = int(round(shift[0])), int(round(shift[1]))
    r0, r1 = max(0, dr), min(rows, dr + rows)
    c0, c1 = max(0, dc), min(cols, dc + cols)
    if r1 - r0 < 8 or c1 - c0 < 8:
        return None
    confidence = _ncc(a[r0:r1, c0:c1], b[r0 - dr : r1 - dr, c0 - dc : c1 - dc])
    if confidence < min_confidence:
        return None
    return float(shift[1] * pixel_m), float(shift[0] * pixel_m), confidence


def bounded_correction(target_xy, current_xy, base_xy, max_step_m: float, max_total_m: float) -> tuple[list[float], bool]:
    """The image shift to apply: toward ``target_xy``, changing each axis by at most ``max_step_m`` from
    ``current_xy`` and staying within ``max_total_m`` of ``base_xy`` (the shift before the map). (shift, limited)."""
    target = np.asarray(target_xy, dtype=np.float64)
    current = np.asarray(current_xy, dtype=np.float64)
    base = np.asarray(base_xy, dtype=np.float64)
    step = np.clip(target - current, -max_step_m, max_step_m)
    shifted = base + np.clip(current + step - base, -max_total_m, max_total_m)
    return [float(v) for v in shifted], bool(np.any(np.abs(shifted - target) > 1e-15))


# ----------------------------------------------------------------------
# Bench: score a tracker with no stack (synthetic texture, steady drift, stage noise)
# ----------------------------------------------------------------------
def drift_bench(tracker: type[DriftTracker] | str = "linear", seeds=(0, 1, 2), rate_m_per_s: float = 5e-11, grid: int = 5,
                fov_m: float = 5e-8, imsize: int = 64, overlap: float = 0.25, tile_s: float = 12.0, stage_sigma_m: float = 1e-9,
                max_step_m: float = 2e-8, settings: dict | None = None) -> dict:
    """Run ``tracker`` over a serpentine ``grid`` x ``grid`` map of a random texture that drifts at ``rate_m_per_s``.

    Returns {"max_error_fov": worst tile placement error over the seeds (fraction of fov_m),
    "untracked_max_error_fov": the same without corrections, "seeds": per-seed values}.
    The images are cut from one smooth random field, so this measures the tracker and the
    registration, not a microscope; it is circular in the same way as the twin proof.
    """
    from scipy import ndimage

    cls = get_tracker(tracker) if isinstance(tracker, str) else tracker
    pixel = fov_m / imsize
    step_px = imsize * (1.0 - overlap)
    per_seed = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        size = int(imsize + step_px * (grid - 1) + 4 * imsize)
        field = ndimage.gaussian_filter(rng.normal(size=(size, size)), 1.5)
        origin = 2 * imsize  # canvas pixel of the first tile's corner (room for drift)

        def cut(x_m: float, y_m: float) -> np.ndarray:
            r, c = origin + y_m / pixel, origin + x_m / pixel
            window = field[int(r): int(r) + imsize + 2, int(c): int(c) + imsize + 2]
            return ndimage.shift(window, (-(r - int(r)), -(c - int(c))), order=1)[:imsize, :imsize]

        worst = {}
        for tracked in (True, False):
            trk = cls(settings)
            shift = np.zeros(2)
            done: dict[tuple[int, int], tuple] = {}
            errors = []
            first = None
            previous_key = None
            for k in range(grid * grid):
                row, j = divmod(k, grid)
                col = grid - 1 - j if row % 2 else j
                planned = np.array([col, row]) * step_px * pixel
                t = k * tile_s
                if tracked:
                    target = -np.asarray(trk.predict(t))
                    shift = np.asarray(bounded_correction(target, shift, [0, 0], max_step_m, 1.0)[0])
                error = direction * rate_m_per_s * t + rng.normal(0, stage_sigma_m, 2)
                first = error + shift if first is None else first
                errors.append(float(np.hypot(*(error + shift - first))))
                if not tracked:
                    continue
                image = cut(*(planned + error + shift))
                measured = np.zeros(2) if not done else None
                if done:
                    adjacent = previous_key is not None and abs(previous_key[0] - row) + abs(previous_key[1] - col) == 1
                    p_img, p_planned, p_shift, p_e = done[previous_key if adjacent else (row - 1, col)]
                    expected = planned + shift - p_planned - p_shift + np.asarray(trk.predict(t)) - p_e
                    found = tile_offset_m(p_img, image, pixel, expected, 4 * stage_sigma_m + 3 * pixel)
                    if found is not None:
                        measured = np.array(found[:2]) - (planned - p_planned) - (shift - p_shift) + p_e
                if measured is not None:
                    trk.observe(t, measured)
                done[(row, col)] = (image, planned, shift.copy(), measured if measured is not None else np.asarray(trk.predict(t)))
                previous_key = (row, col)
            worst[tracked] = max(errors) / fov_m
        per_seed.append({"seed": seed, "max_error_fov": worst[True], "untracked_max_error_fov": worst[False]})
    return {
        "tracker": cls.name,
        "max_error_fov": max(s["max_error_fov"] for s in per_seed),
        "untracked_max_error_fov": max(s["untracked_max_error_fov"] for s in per_seed),
        "seeds": per_seed,
    }
