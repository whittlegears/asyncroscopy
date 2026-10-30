"""The dose ledger: what every scanned frame this package saved put on the sample, kept per sample load.

``save_acquisition`` appends one line per exposure whose attributes say where it was and how it was taken:

* a scanned frame (``stage_reported_m`` or ``stage_commanded_m``, ``fov_m``, ``pixel_size_m``,
  ``dwell_time_s``, ``beam_current_pa``): the rectangle it covered and its dose
  (``asyncroscopy.routines.dose.frame_dose``);
* a parked-beam spectrum (the stage position, ``fov_m``, ``beam_position`` as fractions of the field,
  ``real_time_s``, ``beam_current_pa``, and ``axes``): its electrons counted over a ``SPOT_SPREAD_M`` square
  around the spot, as the Mapper's dose budget counts them (its ``eds_dose_spread_m`` default).

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
(``ASYNCROSCOPY_DOSE_LEDGER``, default ``outputs/dose/ledger.jsonl``); writes take a file lock.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
from pathlib import Path

from asyncroscopy.routines.dose import ELECTRON_CHARGE, frame_dose

if os.name == "nt":
    import msvcrt
else:
    import fcntl

LEDGER_ENV = "ASYNCROSCOPY_DOSE_LEDGER"
DEFAULT_LEDGER = "outputs/dose/ledger.jsonl"
NOT_COUNTED = ("exposures saved without their position, current and time (camera frames, 4D data, and spectra that do not record "
               "their scan axes, such as the AutoScript microscope's), time the beam spent unblanked between exposures, and image "
               "shift and scan rotation (each exposure is placed where the instrument believes the beam was)")
AXES = "rows=+y, columns=+x"
SPOT_SPREAD_M = 1e-8
LOCK_WAIT_S = 5.0  # well inside a Tango client's timeout, so a stuck lock never makes a caller take a frame again
_lock = threading.Lock()


def ledger_path() -> Path:
    return Path(os.environ.get(LEDGER_ENV) or DEFAULT_LEDGER)


@contextlib.contextmanager
def _locked(path: Path):
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_name(path.name + ".lock"), "a+b") as lock:
            lock.seek(0)
            deadline = time.monotonic() + LOCK_WAIT_S
            while True:
                try:
                    if os.name == "nt":
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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
    source = {"current_source": str(attrs["beam_current_source"])} if attrs.get("beam_current_source") else {}
    placed = attrs.get("axes") == AXES
    spot, seconds = _vector(attrs.get("beam_position")), _number(attrs.get("real_time_s"))
    if spot is not None and seconds is not None and attrs.get("pixel_size_m") is None:
        if not (placed and len(spot) == 2 and 0 <= spot[0] <= 1 and 0 <= spot[1] <= 1 and seconds > 0):
            return None
        x, y = stage[0] + (spot[0] - 0.5) * fov, stage[1] + (spot[1] - 0.5) * fov
        half = SPOT_SPREAD_M / 2
        electrons = current * 1e-12 * seconds / ELECTRON_CHARGE
        return {"box_m": [x - half, x + half, y - half, y + half], "dose_e_per_a2": electrons / (SPOT_SPREAD_M / 1e-10) ** 2, "kind": "spot", **source}
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
    line = json.dumps({"at": time.time(), "sample_load_id": sample_load_id, "instrument_id": instrument_id, "key": key, **exposure})
    path = ledger_path()
    with _locked(path):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return True


def _entries(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:  # a line cut short by a crash
                continue
            if isinstance(entry, dict) and isinstance(entry.get("box_m"), list) and len(entry["box_m"]) == 4:
                yield entry


def dose_at(sample_load_id: str | None, x_m: float | None = None, y_m: float | None = None) -> dict:
    """The dose the ledger holds for one sample load: exposures counted (frames and spots), and the total at (x_m, y_m) when given."""
    exposures = at_point = unmeasured = 0
    dose = 0.0
    for entry in _entries(ledger_path()):
        if entry.get("sample_load_id") != sample_load_id:
            continue
        exposures += 1
        if x_m is None or y_m is None:
            continue
        left, right, bottom, top = entry["box_m"]
        if left <= x_m <= right and bottom <= y_m <= top:
            at_point += 1
            dose += float(entry.get("dose_e_per_a2") or 0.0)
            unmeasured += entry.get("current_source", "measured") != "measured"
    reply = {"sample_load_id": sample_load_id, "exposures": exposures, "not_counted": NOT_COUNTED}
    if x_m is not None and y_m is not None:
        reply.update(x_m=x_m, y_m=y_m, exposures_here=at_point, dose_e_per_a2=dose)
        if unmeasured:
            reply["from_unmeasured_current"] = f"{unmeasured} of these exposures used a beam current that was not measured, so their dose is an estimate"
    return reply
