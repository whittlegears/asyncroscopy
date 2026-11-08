"""The dose ledger: what every scanned frame this package saved put on the sample, kept per sample load.

``save_acquisition`` appends one line per exposure whose attributes say where it was and how it was taken:

* a scanned frame (``stage_reported_m`` or ``stage_commanded_m``, plus ``image_shift_m`` when recorded, ``fov_m``, ``pixel_size_m``,
  ``dwell_time_s``, ``beam_current_pa``): the rectangle it covered and its dose
  (``asyncroscopy.routines.dose.frame_dose``);
* a parked-beam spectrum (the stage position, ``fov_m``, ``beam_position`` as fractions of the field,
  ``real_time_s``, ``beam_current_pa``, and ``axes``): its electrons counted over a ``EDS_DOSE_SPREAD_M`` square
  around the spot, as the Mapper's dose budget counts them (``routines.dose.EDS_DOSE_SPREAD_M``).

Positions are where the instrument believes the beam was: the field centred on the recorded stage position,
rows along +y when the data records ``axes`` "rows=+y, columns=+x". A frame without ``axes`` is counted over
its whole field (part of a field cannot be placed), and a spot without ``axes`` is not counted. Scan rotation
and calibration errors are not known to the ledger, so a spot lands that far from where its electrons did.

Each line keeps the sample load and whether the beam current was measured (``beam_current_source``);
``dose_at`` adds up every exposure that covered a point and says how many used an unmeasured current.

What it does not count yet, and says so in every reply (``NOT_COUNTED``): exposures saved without those
attributes (camera frames, 4D data), and the beam left unblanked while the stage moves or the instrument
waits. The box is centred on the recorded stage position: an image shift (live drift tracking sets one)
is not recorded with a frame, so its dose lands that far off. A frame taken with no sample load recorded
is kept under ``sample_load_id`` null and never counted for a sample.

The ledger is one append-only JSON-lines file shared by every device server on the computer
(``ASYNCROSCOPY_DOSE_LEDGER``, default ``outputs/dose/ledger.jsonl``); writes take a file lock. Each line is
chained and keyed like the integrity manifest (``integrity.chain_append``), so a line edited, removed, inserted
or moved shows in ``startup_scripts/verify_integrity.py``. Lines written before the chain have no seal and are
reported as such; ``dose_at`` still counts them.

The file stays the record. ``dose_at`` answers from an index beside it (``<ledger>.index.sqlite``), one row per
exposure keyed by sample load, which it brings up to date first by reading only the lines added since it last looked.
So a query costs the load's own exposures, however long the file is. The index can be deleted at any time: the next
query rebuilds it from the file.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from asyncroscopy import settings
from asyncroscopy.data.integrity import LOCK_WAIT_S, chain_append
from asyncroscopy.core.fileio import file_lock
from asyncroscopy.routines.dose import EDS_DOSE_SPREAD_M, ELECTRON_CHARGE, frame_dose

LEDGER_ENV = "ASYNCROSCOPY_DOSE_LEDGER"
NOT_COUNTED = ("exposures saved without their position, current and time (camera frames, 4D data, and spectra that do not record "
               "their scan axes, such as the AutoScript microscope's), time the beam spent unblanked between exposures, and image "
               "shift and scan rotation (each exposure is placed where the instrument believes the beam was)")
AXES = "rows=+y, columns=+x"


INDEX_SUFFIX = ".index.sqlite"
INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS indexed (id INTEGER PRIMARY KEY CHECK (id = 0), position INTEGER NOT NULL);
INSERT OR IGNORE INTO indexed (id, position) VALUES (0, 0);
CREATE TABLE IF NOT EXISTS exposure (
    offset INTEGER PRIMARY KEY, load TEXT NOT NULL, x0 REAL NOT NULL, x1 REAL NOT NULL, y0 REAL NOT NULL, y1 REAL NOT NULL,
    dose REAL NOT NULL, unmeasured INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS exposure_by_load ON exposure (load);
"""


def ledger_path() -> Path:
    return settings.get().dose_ledger


def _number(value) -> float | None:
    try:
        number = float(json.loads(value) if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vector(value) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        numbers = [_number(v) for v in value]
        return numbers if all(n is not None for n in numbers) else None
    return None


def exposure_of(attrs: dict) -> dict | None:
    """The ledger entry for one saved frame or spectrum, or None when its attributes do not say where or how it was taken."""
    stage = _vector(attrs.get("stage_reported_m")) or _vector(attrs.get("stage_commanded_m"))
    fov, current = _number(attrs.get("fov_m")), _number(attrs.get("beam_current_pa"))
    if stage is None or fov is None or current is None or not (fov > 0 and current >= 0):
        return None
    shift = _vector(attrs.get("image_shift_m"))
    if shift is not None:  # the field's centre is the stage plus the image shift it was taken with
        stage = [stage[0] + shift[0], stage[1] + shift[1], *stage[2:]]
    source = {"current_source": str(attrs["beam_current_source"])} if attrs.get("beam_current_source") else {}
    placed = attrs.get("axes") == AXES
    spot, seconds = _vector(attrs.get("beam_position")), _number(attrs.get("real_time_s"))
    if spot is not None and seconds is not None and attrs.get("pixel_size_m") is None:
        if not (placed and len(spot) == 2 and 0 <= spot[0] <= 1 and 0 <= spot[1] <= 1 and seconds > 0):
            return None
        x, y = stage[0] + (spot[0] - 0.5) * fov, stage[1] + (spot[1] - 0.5) * fov
        half = EDS_DOSE_SPREAD_M / 2
        electrons = current * 1e-12 * seconds / ELECTRON_CHARGE
        return {"box_m": [x - half, x + half, y - half, y + half], "dose_e_per_a2": electrons / (EDS_DOSE_SPREAD_M / 1e-10) ** 2, "kind": "spot", **source}
    pixel, dwell = _number(attrs.get("pixel_size_m")), _number(attrs.get("dwell_time_s"))
    if None in (pixel, dwell) or not (pixel > 0 and dwell > 0):
        return None
    region = _vector(attrs.get("scan_region")) if placed else None
    region = region or [0.0, 0.0, 1.0, 1.0]
    if len(region) != 4 or not (region[2] > 0 and region[3] > 0):
        return None
    across, down, width, height = region  # [left, top, width, height] as fractions of the field; rows run along +y
    left, bottom = stage[0] - fov / 2, stage[1] - fov / 2
    return {
        "box_m": [left + across * fov, left + (across + width) * fov, bottom + down * fov, bottom + (down + height) * fov],
        "dose_e_per_a2": frame_dose(current, dwell, fov, max(1, round(fov / pixel))),
        "kind": "frame",
        **source,
    }


def record_frame(attrs: dict, sample_load_id: str | None, key: str, instrument_id: str) -> bool:
    """Append the frame ``attrs`` describe to the ledger; False (nothing written) when they do not describe one."""
    exposure = exposure_of(attrs)
    if exposure is None:
        return False
    record = {"at": time.time(), "sample_load_id": sample_load_id, "instrument_id": instrument_id, "key": key, **exposure}
    try:
        chain_append(ledger_path(), record)
    except (OSError, ValueError) as exc:  # an unreadable integrity key must not lose the dose: the line is kept unsealed, which verify reports
        print(f"[dose ledger] {key} written without a seal: {exc}")
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_name(path.name + ".lock"), "a+b") as lock, file_lock(lock, time.monotonic() + LOCK_WAIT_S):
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
    return True


def _row(raw: bytes, offset: int) -> tuple | None:
    """The index row of one ledger line; None for a line that is not an exposure (one cut short by a crash)."""
    try:
        entry = json.loads(raw)
        left, right, bottom, top = (float(v) for v in entry["box_m"])
    except (ValueError, KeyError, TypeError):
        return None
    load = entry.get("sample_load_id")
    return (offset, "" if load is None else str(load), left, right, bottom, top, float(entry.get("dose_e_per_a2") or 0.0), int(entry.get("current_source", "measured") != "measured"))


def _bring_up_to_date(db: sqlite3.Connection, path: Path) -> None:
    """Index the whole lines added since the last query (from the start, if the file was replaced by a shorter one)."""
    size = path.stat().st_size if path.exists() else 0
    if size == db.execute("SELECT position FROM indexed").fetchone()[0]:
        return
    db.execute("BEGIN IMMEDIATE")
    try:
        position = db.execute("SELECT position FROM indexed").fetchone()[0]  # another device server may have just done it
        if size < position:
            db.execute("DELETE FROM exposure")
            position = 0
        rows = []
        if size:
            with open(path, "rb") as handle:
                handle.seek(position)
                for raw in handle:
                    if not raw.endswith(b"\n"):
                        break  # a line still being written: the next query takes it
                    row = _row(raw, position)
                    if row is not None:
                        rows.append(row)
                    position += len(raw)
        db.executemany("INSERT OR IGNORE INTO exposure VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        db.execute("UPDATE indexed SET position = ?", (position,))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def _query(index: Path, path: Path, load: str, x_m: float | None, y_m: float | None) -> tuple[int, tuple[int, float, int]]:
    with closing(sqlite3.connect(index, timeout=LOCK_WAIT_S, isolation_level=None)) as db:
        db.executescript(INDEX_SCHEMA)
        _bring_up_to_date(db, path)
        exposures = db.execute("SELECT count(*) FROM exposure WHERE load = ?", (load,)).fetchone()[0]
        if x_m is None or y_m is None:
            return exposures, (0, 0.0, 0)
        here = db.execute(
            "SELECT count(*), coalesce(sum(dose), 0), coalesce(sum(unmeasured), 0) FROM exposure WHERE load = ? AND x0 <= ? AND ? <= x1 AND y0 <= ? AND ? <= y1",
            (load, x_m, x_m, y_m, y_m),
        ).fetchone()
    return exposures, here


def dose_at(sample_load_id: str | None, x_m: float | None = None, y_m: float | None = None) -> dict:
    """The dose the ledger holds for one sample load: exposures counted (frames and spots), and the total at (x_m, y_m) when given."""
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    index = path.with_name(path.name + INDEX_SUFFIX)
    load = "" if sample_load_id is None else str(sample_load_id)
    try:
        exposures, (at_point, dose, unmeasured) = _query(index, path, load, x_m, y_m)
    except sqlite3.DatabaseError:  # a damaged index: the file is the record, so it is built again
        index.unlink(missing_ok=True)
        exposures, (at_point, dose, unmeasured) = _query(index, path, load, x_m, y_m)
    reply = {"sample_load_id": sample_load_id, "exposures": exposures, "not_counted": NOT_COUNTED}
    if x_m is not None and y_m is not None:
        reply.update(x_m=x_m, y_m=y_m, exposures_here=at_point, dose_e_per_a2=float(dose))
        if unmeasured:
            reply["from_unmeasured_current"] = f"{unmeasured} of these exposures used a beam current that was not measured, so their dose is an estimate"
    return reply
