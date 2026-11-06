"""A map's HDF5 run file: every tile, its reported stage position and any spectra, in one file per run.

The file is ``maps/map_<run>.h5``, named ``.h5.partial`` while tiles are being taken so DATA does not
register it half-written. It uses layout 2 (see asyncroscopy.analysis_server.map_store): ``tiles`` is one chunked
array [n, rows, cols] with per-tile arrays ``written``, ``stage_reported_m``, ``commanded_xy_m``,
``acquired_s``, ``row_col`` and ``source_keys``; further detectors go in ``channels/<DETECTOR>``;
``spectra`` [n, channels] and ``spectrum_keys`` [n] hold EDS point spectra, or ``spectrum_image`` a coarse
spectrum image per tile. Nothing in it depends on the vendor or on whether the instrument was a twin.
``acquired_s`` is the instrument's ``clock_s`` (simulated time on a twin), the clock the drift fit and
revisits use.

One ``RunFile`` keeps the file open for the whole run and flushes it after every tile, so a crash of the
device loses at most the tile being written, without reopening the file for each tile.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

RUN_LAYOUT = 2
RENAME_RETRY_DELAYS_S = (0.0, 0.2, 0.5, 1.0, 2.0)  # a finished run file may briefly be open elsewhere (Windows)
TRACE_LINES_KEPT = 2000  # job events copied into the run file; the rest are in the shared trace (see trace_file)
PARTIAL = ".partial"


def job_trace_events(trace: Path, run_id: str, offset: int) -> tuple[list[str], int]:
    """One job's own events (reserve, start, stop, approvals, failed steps) from the shared trace; (lines, dropped).

    Only the part written since ``offset`` (where the trace ended when the job was reserved) is read, line
    by line, so the cost is this job's, not the history's. Successful per-tile steps are left out (each
    tile's position and time are already in the run file), and at most TRACE_LINES_KEPT lines are kept.
    """
    events: list[str] = []
    dropped = 0
    marker = f'"run_id": "{run_id}"'
    if not trace.exists():
        return events, dropped
    with trace.open("rb") as handle:
        handle.seek(min(offset, trace.stat().st_size))
        for raw in handle:
            line = raw.decode("utf-8", errors="replace").strip()
            if marker not in line or ('.step"' in line and '"ok": true' in line):
                continue
            if len(events) < TRACE_LINES_KEPT:
                events.append(line)
            else:
                dropped += 1
    return events, dropped


class RunFile:
    """One run's file: opened once, written tile by tile, flushed after each, then closed and renamed ``.h5``."""

    def __init__(self, final: Path) -> None:
        self.final = Path(final)
        self.partial = self.final.with_name(self.final.name + PARTIAL)
        self._h5: h5py.File | None = None

    @property
    def path(self) -> Path:
        """Where the file is now: the final name once finished, ``.h5.partial`` before."""
        return self.final if self.final.exists() else self.partial

    def exists(self) -> bool:
        return self.final.exists() or self.partial.exists()

    @property
    def is_open(self) -> bool:
        return self._h5 is not None

    # -- writing ---------------------------------------------------------
    def open(self, attrs: dict, first_attrs: dict) -> None:
        """Open (or reopen, on a resume) the ``.partial`` file for writing and set its run attributes.

        ``first_attrs`` are written only when the file is new: a resumed run keeps the id, sample and time
        it was first written with.
        """
        if self._h5 is not None:
            return
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.partial, "a")
        if "id" not in self._h5.attrs:
            self._h5.attrs.update(first_attrs)
        self._h5.attrs.update({"run_layout": RUN_LAYOUT, **attrs})
        self._h5.flush()

    def write_tile(
        self, n: int, tile: dict, plan: dict, reported: list[float], acquired: float, image_key: str,
        images: dict, spectrum: dict | None, spectrum_key: str | None, arrays: dict | None = None,
    ) -> None:
        """Put one tile into the per-tile arrays (created at the first tile, sized for the whole grid of ``n``).

        ``arrays``: more per-tile values by dataset name (live drift tracking's drift_correction_m and
        drift_error_m), each a float or a list of floats; created NaN-filled at the first tile that has them.
        The file is flushed before returning, so a crash later loses nothing written here.
        """
        h5 = self._require_open()
        k = tile["index"]
        if "written" not in h5:
            h5.create_dataset("written", shape=(n,), dtype=bool)
            h5.create_dataset("stage_reported_m", shape=(n, 5), dtype=np.float64, fillvalue=np.nan)
            h5.create_dataset("commanded_xy_m", shape=(n, 2), dtype=np.float64, fillvalue=np.nan)
            h5.create_dataset("acquired_s", shape=(n,), dtype=np.float64, fillvalue=np.nan)
            h5.create_dataset("row_col", shape=(n, 2), dtype=np.int32, fillvalue=-1)
            h5.create_dataset("source_keys", shape=(n,), dtype=h5py.string_dtype())
        for m, (name, (array, attrs)) in enumerate(sorted(images.items())):
            detector = str(attrs.get("detector", name.rsplit("/", 1)[-1])).upper()
            target = "tiles" if m == 0 else f"channels/{detector}"
            if target not in h5:
                array = np.asarray(array)
                dset = h5.create_dataset(target, shape=(n, *array.shape), dtype=array.dtype, chunks=(1, *array.shape))
                dset.attrs.update({"detector": detector, "fov_m": float(plan["fov_m"]), "pixel_size_m": float(plan["fov_m"]) / int(plan["imsize"])})
            h5[target][k] = array
        h5["stage_reported_m"][k] = (list(reported) + [0.0] * 5)[:5]
        h5["commanded_xy_m"][k] = [tile["x"], tile["y"]]
        h5["acquired_s"][k] = acquired
        h5["row_col"][k] = [tile["row"], tile["col"]]
        h5["source_keys"][k] = image_key
        if spectrum is not None:
            array, attrs = next(iter(spectrum.values()))
            array = np.asarray(array)
            target = "spectrum_image" if spectrum_key == "eds_map" else "spectra"
            if target not in h5:
                dset = h5.create_dataset(target, shape=(n, *array.shape), dtype=array.dtype, chunks=(1, *array.shape))
                dset.attrs.update({key: (json.dumps(v) if isinstance(v, (list, dict)) else v) for key, v in attrs.items() if v is not None})
                h5.create_dataset("spectrum_keys", shape=(n,), dtype=h5py.string_dtype())
            h5[target][k] = array
            h5["spectrum_keys"][k] = spectrum_key if spectrum_key != "eds_map" else ""
        for name, value in (arrays or {}).items():
            value = np.asarray(value, dtype=np.float64)
            if name not in h5:
                h5.create_dataset(name, shape=(n, *value.shape), dtype=np.float64, fillvalue=np.nan)
            h5[name][k] = value
        h5["written"][k] = True
        h5.flush()

    def write_trace(self, events: list[str], trace_file: str, offset: int, dropped: int) -> None:
        """Replace the run's copy of its job's trace events (job_trace_events)."""
        if self._h5 is None:
            if not self.exists():
                return
            with h5py.File(self.path, "a") as h5:
                self._put_trace(h5, events, trace_file, offset, dropped)
            return
        self._put_trace(self._h5, events, trace_file, offset, dropped)
        self._h5.flush()

    @staticmethod
    def _put_trace(h5: h5py.File, events: list[str], trace_file: str, offset: int, dropped: int) -> None:
        if "trace" in h5:
            del h5["trace"]
        h5.create_dataset("trace", data=np.array(events, dtype=h5py.string_dtype()))
        h5["trace"].attrs.update({"trace_file": trace_file, "trace_offset": offset, "events_not_copied": dropped})

    def close(self) -> None:
        """Close the writer (an aborted or paused run reopens it on a resume)."""
        h5, self._h5 = self._h5, None
        if h5 is not None:
            h5.close()

    def finish(self) -> tuple[Path, str | None]:
        """Close the file and give it its final ``.h5`` name; (path, note). Never raises: the data is complete either way.

        On Windows a rename fails while anything has the file open, so it is retried for a few seconds and
        then copied instead.
        """
        self.close()
        if self.final.exists() or not self.partial.exists():
            return self.final, None
        last: OSError | None = None
        for delay in RENAME_RETRY_DELAYS_S:
            time.sleep(delay)
            try:
                self.partial.replace(self.final)
                return self.final, None
            except OSError as exc:
                last = exc
        try:
            shutil.copy2(self.partial, self.final)
        except OSError as exc:
            return self.partial, f"the run file could not be renamed ({last}) or copied ({exc}); it stays at {self.partial}"
        try:
            self.partial.unlink()
        except OSError:
            return self.final, f"the run file was in use, so it was copied to {self.final.name}; {self.partial.name} is left behind ({last})"
        return self.final, f"the run file was in use, so it was copied to {self.final.name} ({last})"

    # -- reading ---------------------------------------------------------
    def attr(self, name: str):
        """One file attribute, or None (also when there is no file yet)."""
        if self._h5 is not None:
            return self._h5.attrs.get(name)
        if not self.exists():
            return None
        with h5py.File(self.path, "r") as h5:
            return h5.attrs.get(name)

    def tile_image(self, index: int) -> np.ndarray | None:
        """The ``tiles`` image of tile ``index`` when it was written; None otherwise."""
        try:
            if self._h5 is not None:
                return self._tile(self._h5, index)
            if not self.exists():
                return None
            with h5py.File(self.path, "r") as h5:
                return self._tile(h5, index)
        except (OSError, KeyError, IndexError):
            return None

    @staticmethod
    def _tile(h5: h5py.File, index: int) -> np.ndarray | None:
        return np.asarray(h5["tiles"][index]) if "tiles" in h5 and bool(h5["written"][index]) else None

    def _require_open(self) -> h5py.File:
        if self._h5 is None:
            raise RuntimeError(f"run file {self.partial} is not open for writing")
        return self._h5
