"""File plumbing shared by every layer: a lock between processes, sequenced JSONL appends, atomic writes.

``SequencedAppender`` gives every line of a JSONL file the file's next ``seq`` and the Unix time it
was ``written``, under a lock shared by every process appending to that file, so a reader can
follow several files at once and merge them in order (``asyncroscopy.trace.stream``).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator

LOCK_TIMEOUT_S = 5.0  # longest one attempt waits for another process's append
LOCK_ATTEMPTS = 3  # attempts before an append gives up and raises
_LOCK_BYTE = 1 << 20  # Windows locks this byte of the .seq file, past its data, so its data stays readable


def seq_path(path: str | Path) -> Path:
    """The sidecar holding a log's last ``seq`` and ``written`` time: ``<name>.seq`` next to it.

    Every process appending to the log shares it, and it outlives a rotation of the log, so
    ``seq`` keeps rising when ``jobs.jsonl`` is renamed to ``jobs.jsonl.1`` and a new one begun.
    """
    path = Path(path)
    return path.with_name(path.name + ".seq")


@contextmanager
def file_lock(handle: IO[bytes], deadline: float) -> Iterator[None]:
    """An exclusive lock between processes on ``handle`` (opened "a+b"), waited for until ``deadline``.

    Raises TimeoutError when another process holds it past ``deadline``.
    """
    if os.name == "nt":
        import msvcrt

        def attempt() -> None:
            handle.seek(_LOCK_BYTE)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

        def release() -> None:
            handle.seek(_LOCK_BYTE)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        def attempt() -> None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def release() -> None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    while True:
        try:
            attempt()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("another process held the file's lock too long") from None
            time.sleep(0.002)
    try:
        yield
    finally:
        release()


def write_atomic(path: str | Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one step: a reader sees the old file or the new one, never half of it.

    The text goes to a temporary file in the same folder first, which then replaces ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _last_seq_in(path: Path) -> tuple[int, float]:
    """(highest seq, latest written) already in ``path``: how a log without a .seq sidecar is continued."""
    seq, written = 0, 0.0
    if not path.exists():
        return seq, written
    with path.open("rb") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict):
                if isinstance(row.get("seq"), int):
                    seq = max(seq, row["seq"])
                if isinstance(row.get("written"), (int, float)):
                    written = max(written, float(row["written"]))
    return seq, written


class SequencedAppender:
    """Appends JSON lines to one file, each with the file's next ``seq`` and its ``written`` time.

    ``seq`` rises by one per line of the file whichever thread or process wrote it: the sidecar
    (``seq_path``) is locked while a line is numbered and appended, so lines land in ``seq``
    order. ``written`` is the Unix time of the append, never earlier than the line before it;
    the merging reader orders the lines of several files by it. The file is opened for each
    line and closed again, so it can be renamed (rotated) between lines, on Windows too.

    A line is never written without its ``seq``: when the lock cannot be had after
    ``LOCK_ATTEMPTS`` tries the append raises TimeoutError and writes nothing.
    """

    def __init__(self, path: str | Path, lock_timeout_s: float = LOCK_TIMEOUT_S, attempts: int = LOCK_ATTEMPTS) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_s = lock_timeout_s
        self.attempts = max(1, int(attempts))
        self._lock = threading.Lock()

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        """Number ``row``, append it as one line and return it.

        Raises TimeoutError when another process holds the lock through every attempt, and
        OSError when the file cannot be written.
        """
        with self._lock:
            for attempt in range(self.attempts):
                try:
                    with seq_path(self.path).open("a+b") as counter, file_lock(counter, time.monotonic() + self.lock_timeout_s):
                        counter.seek(0)
                        stored = counter.read(100).decode("ascii", errors="replace").split()
                        try:
                            last_seq, last_written = int(stored[0]), float(stored[1])
                        except (IndexError, ValueError):  # a new sidecar (or a damaged one): continue from the log itself
                            last_seq, last_written = _last_seq_in(self.path)
                        row["seq"] = last_seq + 1
                        row["written"] = max(time.time(), last_written)
                        with self.path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row) + "\n")
                        counter.seek(0)
                        counter.truncate()
                        counter.write(f"{row['seq']} {row['written']!r}".encode("ascii"))
                    return row
                except TimeoutError:
                    if attempt + 1 == self.attempts:
                        raise TimeoutError(f"{self.path}: another process held its lock through {self.attempts} attempts; line not written") from None
        raise AssertionError("unreachable")  # pragma: no cover
