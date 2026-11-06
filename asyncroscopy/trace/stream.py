"""Follow every Instrument Trace file of a stack at once and get its events in one order.

Each process appends its own JSONL file (``outputs/trace/jobs.jsonl``, ``mcp_agent.jsonl``,
``approvals.jsonl``, pyTEMlib's ``analysis.jsonl``, ...). Every line a TraceWriter writes
carries the file's ``seq`` and the Unix time it was ``written`` (see
``asyncroscopy.trace.writer``). This module merges the files by ``written`` (``ts`` for lines
that have none), file name and ``seq``:

- ``read_trace(sources, run=...)``: everything written so far, in order (retrospective);
- ``follow_trace(sources, run=...)``: the same events as they are written (live);
- ``TraceStream``: both, plus a ``cursor()`` to carry on from later (the operator MCP
  endpoint's ``trace_since`` tool hands it to the GUI).

A live follower holds an event back until ``grace_s`` after it was written, so a line still
being appended by another process can never be overtaken; the order is then the one
``read_trace`` gives. Lines are never yielded twice and never missed when a file grows, is
cut short while being written, is truncated, or is rotated (renamed, e.g. to
``jobs.jsonl.1``, and begun again): each file is read once, from where the follower stopped,
and the follower finds a renamed file by its identity (inode, or first bytes) to read what it
had not yet read. Within one file, a line whose ``seq`` is not above the last one read is a
repeat and is skipped.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_SOURCES = ("outputs/trace",)
GRACE_S = 0.5  # a live event is held this long after it was written, so a slower writer's line cannot be overtaken
POLL_S = 0.2
HEAD_BYTES = 64  # a file's first bytes: its identity where the file system reports no inode
RETIRED_KEPT = 64
CURSOR_PREFIX = "tc1."
SIDECAR_SUFFIXES = (".seq", ".lock")


def _expand(sources: Iterable[str | os.PathLike]) -> list[Path]:
    """Files to follow: a directory gives its ``*.jsonl`` and their rotated copies (``*.jsonl.1``), a file
    itself (even before it exists) and its rotated copies, a glob what it matches now."""
    paths: list[Path] = []
    for source in sources:
        text = str(source)
        if any(ch in text for ch in "*?["):
            paths.extend(Path(p) for p in sorted(glob.glob(text)))
        elif Path(text).is_dir():
            paths.extend(sorted(Path(text).glob("*.jsonl")))
            paths.extend(sorted(Path(text).glob("*.jsonl.*")))
        else:
            paths.append(Path(text))
            paths.extend(sorted(Path(text).parent.glob(glob.escape(Path(text).name) + ".*")))
    paths = [p for p in paths if not p.name.endswith(SIDECAR_SUFFIXES)]
    unique: dict[str, Path] = {}
    for path in paths:
        unique.setdefault(str(path.resolve()), path.resolve())
    return list(unique.values())


def _logical(path: Path) -> str:
    """The trace file a (possibly rotated) file belongs to: ``jobs.jsonl`` for ``jobs.jsonl.1``."""
    name = path.name
    cut = name.find(".jsonl.")
    return name[: cut + len(".jsonl")] if cut >= 0 else name


def _stat(path: Path) -> os.stat_result | None:
    try:
        return os.stat(path)
    except OSError:
        return None


def _same_file(ino_a: int, head_a: bytes, ino_b: int, head_b: bytes) -> bool:
    """Whether two (inode, first bytes) identities name the same file; first bytes decide where there is no inode."""
    if ino_a and ino_b:
        return ino_a == ino_b
    n = min(len(head_a), len(head_b))
    return head_a[:n] == head_b[:n]


def _as_set(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    return {value} if isinstance(value, str) else set(value)


@dataclass
class _File:
    path: Path
    ino: int = 0
    head: bytes = b""
    read_off: int = 0  # bytes of the file (the one ino/head names) already parsed
    read_seq: int | None = None  # highest seq parsed: a line at or below it is a repeat
    last_t: float = float("-inf")  # ordering time of the last line parsed (kept non-decreasing per file)
    known: bool = False  # ino/head describe a file this follower has read
    committed: dict[str, Any] = field(default_factory=dict)  # where the last event handed out ended (the cursor)
    pending: deque = field(default_factory=deque)  # parsed, not yet handed out: (key, event, commit)


class TraceStream:
    """The Instrument Trace of a stack, merged from several JSONL files, filtered and in order.

    sources: files, directories (their ``*.jsonl``) or globs; re-expanded on every poll, so a
        file that appears later is followed from its first line. Default: ``outputs/trace``.
    run, caller, device, job: keep only matching events (each a value or a collection).
        ``run`` is the agent run (X-Asyncroscopy-Run); ``device`` matches the writer (a Tango
        device name), a dry run's ``device`` or the action's prefix (``Mapper`` matches
        ``Mapper_start`` and ``Mapper.step``); ``job`` matches ``job_id``, ``run_id`` (the
        reservation) or ``plan_id``.
    since: Unix time; only events written (or, without ``written``, started) at or after it.
    cursor: a ``cursor()`` from an earlier stream: carry on right after its last event.
    from_end: start at the current end of every file (only what is written from now on).
    grace_s: how long a live event is held after it was written (see the module docstring).
    """

    def __init__(
        self,
        sources: Iterable[str | os.PathLike] | str | os.PathLike | None = None,
        *,
        run: str | Iterable[str] | None = None,
        caller: str | Iterable[str] | None = None,
        device: str | Iterable[str] | None = None,
        job: str | Iterable[str] | None = None,
        since: float | None = None,
        cursor: str | None = None,
        from_end: bool = False,
        grace_s: float = GRACE_S,
    ) -> None:
        if sources is None:
            sources = DEFAULT_SOURCES
        elif isinstance(sources, (str, os.PathLike)):
            sources = [sources]
        self.sources = [str(s) for s in sources]
        self.run, self.caller, self.job = _as_set(run), _as_set(caller), _as_set(job)
        self.device = {d.lower() for d in _as_set(device)} if device is not None else None
        self.since = since
        self.grace_s = grace_s
        self._files: dict[str, _File] = {}
        self._retired: list[int] = []  # inodes of rotated files already read: never followed again under a new name
        if cursor:
            self._restore(cursor)
        elif from_end:
            for path in _expand(self.sources):
                st = _stat(path)
                if st is None:
                    continue
                f = _File(path, ino=st.st_ino, head=self._head(path), read_off=st.st_size, known=True)
                f.committed = self._commit(f, st.st_size, None, float("-inf"))
                self._files[str(path)] = f

    # ------------------------------------------------------------------ public
    def poll(self, final: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        """New matching events, in order. ``final``: hand out everything read (a finished trace), not
        only events older than ``grace_s``. ``limit``: at most this many; the rest come next time."""
        self._scan()
        return self._merge(final, limit)

    def follow(self, timeout: float | None = None, poll_s: float = POLL_S, stop: threading.Event | None = None) -> Iterator[dict[str, Any]]:
        """Yield matching events as they are written, until ``timeout`` seconds pass or ``stop`` (a threading.Event) is set."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            yield from self.poll()
            if stop is not None and stop.is_set():
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            wait = poll_s if deadline is None else max(0.0, min(poll_s, deadline - time.monotonic()))
            if stop is not None:
                stop.wait(wait)
            else:
                time.sleep(wait)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.follow()

    def cursor(self) -> str:
        """Where this stream stands: right after the last event it handed out (filtered-out ones included)."""
        state = {
            "files": {name: f.committed for name, f in self._files.items() if f.committed},
            "retired": self._retired[-RETIRED_KEPT:],
        }
        raw = json.dumps(state, separators=(",", ":")).encode("utf-8")
        return CURSOR_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")

    # ------------------------------------------------------------------ files
    @staticmethod
    def _head(path: Path) -> bytes:
        try:
            with open(path, "rb") as handle:
                return handle.read(HEAD_BYTES)
        except OSError:
            return b""

    @staticmethod
    def _commit(f: _File, offset: int, seq: int | None, t: float) -> dict[str, Any]:
        return {"ino": f.ino, "head": f.head.hex(), "off": offset, "seq": seq, "t": t if t != float("-inf") else None}

    def _restore(self, cursor: str) -> None:
        if not cursor.startswith(CURSOR_PREFIX):
            raise ValueError("not a trace cursor (expected one returned by an earlier call)")
        try:
            state = json.loads(base64.urlsafe_b64decode(cursor[len(CURSOR_PREFIX):].encode("ascii")))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"damaged trace cursor: {exc}") from exc
        self._retired = [int(i) for i in state.get("retired", [])]
        for name, c in state.get("files", {}).items():
            f = _File(Path(name), ino=int(c.get("ino") or 0), head=bytes.fromhex(c.get("head") or ""), read_off=int(c.get("off") or 0),
                      read_seq=c.get("seq"), last_t=c["t"] if c.get("t") is not None else float("-inf"), known=True)
            f.committed = dict(c)
            self._files[name] = f

    def _tracked_inos(self, but: _File | None = None) -> set[int]:
        """Inodes this follower reads (or has finished reading) under some name, other than ``but``'s."""
        return {f.ino for f in self._files.values() if f.ino and f is not but} | set(self._retired)

    def _scan(self) -> None:
        tracked = self._tracked_inos()
        for path in _expand(self.sources):
            if str(path) in self._files:
                continue
            st = _stat(path)
            if st is not None and st.st_ino and st.st_ino in tracked:
                continue  # a rotated file, renamed to a name the sources match: read under its first name
            self._files[str(path)] = _File(path)
        for f in self._files.values():
            self._scan_file(f)

    def _scan_file(self, f: _File) -> None:
        for _ in range(3):  # a rotation between stat and open is seen at open, and retried
            st = _stat(f.path)
            if f.known:
                current_head = self._head(f.path) if st is not None and not (st.st_ino and f.ino) else b""
                if st is None or not _same_file(f.ino, f.head, st.st_ino, current_head):
                    self._drain_rotated(f)
                    if st is None:
                        return
                    if st.st_ino and st.st_ino in self._tracked_inos(but=f):
                        # A file already read under another name was rotated onto this one (jobs.jsonl.1 -> .2,
                        # jobs.jsonl -> .1): its owner reads what is left of it; this name takes it as read.
                        f.ino, f.head, f.read_off, f.known = st.st_ino, self._head(f.path), st.st_size, True
                        if not f.pending:  # else the cursor moves on when the old file's last events are handed out
                            f.committed = self._commit(f, st.st_size, None, f.last_t)
                        return
                    continue
                if st.st_size < f.read_off:  # truncated in place: start again from its first line
                    f.read_off, f.head, f.read_seq = 0, b"", None
            if st is None:
                return
            if self._read(f, f.path, st.st_ino):
                return

    def _drain_rotated(self, f: _File) -> None:
        """The file was renamed (or removed): read what was left of it, then follow the new one from its start."""
        old = self._find_rotated(f)
        if old is not None:
            self._read(f, old, f.ino, rotated=True)
        if f.ino:
            self._retired.append(f.ino)
            del self._retired[:-RETIRED_KEPT]
        # The new file is read from its first line; its seq may restart if the rotation also moved the .seq sidecar.
        f.ino, f.head, f.read_off, f.known, f.read_seq = 0, b"", 0, False, None

    def _find_rotated(self, f: _File) -> Path | None:
        directory = f.path.parent
        try:
            candidates = [p for p in directory.iterdir() if p != f.path and p.name.startswith(f.path.stem) and not p.name.endswith(SIDECAR_SUFFIXES)]
        except OSError:
            return None
        for candidate in sorted(candidates):
            st = _stat(candidate)
            if st is None:
                continue
            if _same_file(f.ino, f.head, st.st_ino, self._head(candidate) if not (f.ino and st.st_ino) else b""):
                return candidate
        return None

    def _read(self, f: _File, path: Path, expect_ino: int, rotated: bool = False) -> bool:
        """Parse the complete lines of ``path`` past ``f.read_off``; False if ``path`` turned out to be another file."""
        try:
            with open(path, "rb") as handle:
                ino = os.fstat(handle.fileno()).st_ino
                if expect_ino and ino and ino != expect_ino:
                    return False
                head = handle.read(HEAD_BYTES)
                if f.known and not rotated and not _same_file(f.ino, f.head, ino, head):
                    return False
                handle.seek(f.read_off)
                data = handle.read()
        except OSError:
            return True  # gone or locked for now: the next poll tries again
        if not f.known:
            f.ino, f.known = ino, True
        if len(f.head) < len(head):
            f.head = head
        end = data.rfind(b"\n")
        if end < 0:
            return True  # nothing complete yet
        offset = f.read_off
        for raw in data[: end + 1].split(b"\n")[:-1]:
            offset += len(raw) + 1
            try:
                event = json.loads(raw)
            except ValueError:
                continue  # a damaged line (a crash mid-write): skipped, never retried
            if not isinstance(event, dict):
                continue
            seq = event.get("seq") if isinstance(event.get("seq"), int) and not isinstance(event.get("seq"), bool) else None
            if seq is not None and f.read_seq is not None and seq <= f.read_seq:
                continue  # already read (a copy-truncate or a re-read after rotation)
            if seq is not None:
                f.read_seq = seq
            t = event.get("written", event.get("ts"))
            t = float(t) if isinstance(t, (int, float)) else f.last_t
            f.last_t = t = max(t, f.last_t)
            key = (t, _logical(f.path), seq if seq is not None else offset)
            f.pending.append((key, event, self._commit(f, offset, f.read_seq, t)))
        f.read_off = offset
        return True

    # ------------------------------------------------------------------ order and filters
    def _merge(self, final: bool, limit: int | None) -> list[dict[str, Any]]:
        cutoff = float("inf") if final else time.time() - self.grace_s
        out: list[dict[str, Any]] = []
        while limit is None or len(out) < limit:
            best = None
            for f in self._files.values():
                if f.pending and (best is None or f.pending[0][0] < best.pending[0][0]):
                    best = f
            if best is None or best.pending[0][0][0] > cutoff:
                break
            key, event, commit = best.pending.popleft()
            best.committed = commit
            if self._matches(event, key[0]):
                out.append(self._normalized(event, best.path))
        return out

    @staticmethod
    def _normalized(event: dict[str, Any], path: Path) -> dict[str, Any]:
        """The event as written, plus the file it came from; pyTEMlib's ``tool`` also reads as ``action``."""
        out = dict(event)
        if "action" not in out and "tool" in out:
            out["action"] = out["tool"]
        out["file"] = _logical(path)
        return out

    def _matches(self, event: dict[str, Any], t: float) -> bool:
        if self.since is not None and t < self.since:
            return False
        if self.run is not None and event.get("run") not in self.run:
            return False
        if self.caller is not None and event.get("caller") not in self.caller:
            return False
        if self.job is not None and not ({event.get("job_id"), event.get("run_id"), event.get("plan_id")} & self.job):
            return False
        if self.device is not None:
            action = str(event.get("action") or event.get("tool") or "").lower()
            names = {str(event.get("writer") or "").lower(), str(event.get("device") or "").lower()}
            if not (names & self.device or any(action.startswith(d + "_") or action.startswith(d + ".") for d in self.device)):
                return False
        return True


def read_trace(sources: Iterable[str | os.PathLike] | str | os.PathLike | None = None, **filters: Any) -> list[dict[str, Any]]:
    """Every matching event written so far, in order (the retrospective reader). Filters as for TraceStream."""
    return TraceStream(sources, **filters).poll(final=True)


def follow_trace(sources: Iterable[str | os.PathLike] | str | os.PathLike | None = None, timeout: float | None = None, poll_s: float = POLL_S, stop: threading.Event | None = None, **filters: Any) -> Iterator[dict[str, Any]]:
    """Matching events as they are written, in the order read_trace gives (live). Filters as for TraceStream."""
    return TraceStream(sources, **filters).follow(timeout=timeout, poll_s=poll_s, stop=stop)


def trace_since(sources: Iterable[str | os.PathLike] | str | os.PathLike | None, cursor: str = "", limit: int = 200, from_end: bool = False, grace_s: float = GRACE_S, **filters: Any) -> dict[str, Any]:
    """One poll for a client that keeps only a cursor (the GUI): {"events": [...], "cursor": next, "more": bool}.

    An empty cursor starts at the first line (``from_end``: at the current end). Pass the returned
    cursor to the next call to get the events written since, with none twice and none missed.
    """
    stream = TraceStream(sources, cursor=cursor or None, from_end=from_end and not cursor, grace_s=grace_s, **filters)
    events = stream.poll(limit=limit)
    more = len(events) >= limit
    return {"events": events, "cursor": stream.cursor(), "more": more}
