"""Tamper evidence for files and chained logs (written; not yet called by any writer).

``seal(path)`` records a file's SHA-256 and size in an append-only manifest (``ASYNCROSCOPY_INTEGRITY_LOG``,
default ``outputs/integrity/manifest.jsonl``). Each manifest line carries an HMAC (a keyed SHA-256) over its own
fields and the HMAC of the line before it (``chain_append``), so:

* a sealed file whose bytes change no longer matches its hash, and ``check_file`` raises ``IntegrityError``;
* a line edited, deleted, inserted or moved breaks its HMAC or the chain after it (``verify_chain``);
* forging a line, or re-sealing a changed file, needs the key.

``verify`` checks the manifest's chain, every sealed file, and any other chained logs it is given.

Nothing calls ``seal`` yet: ``save_acquisition``, the dose ledger and the map writer are wired to it in phase P7
of ``docs/designs/multiscale_maps.md``. Until then this module protects nothing on its own.

The key comes from ``ASYNCROSCOPY_INTEGRITY_KEY`` (hex), or the file ``ASYNCROSCOPY_INTEGRITY_KEY_FILE``
(default ``outputs/integrity/key``, made from 32 random bytes the first time it is needed). Whoever can read
the key can forge seals, and whoever can write the manifest can cut off its end: on an instrument, run the
device servers under an account that alone can read the key and write ``outputs/integrity``, and copy the
chain ``head`` that ``verify`` reports somewhere else now and then (an e-mail, a lab notebook) so a cut-off
end is noticed too. No MCP tool reads or writes the key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

from asyncroscopy.mcp.tool_log import file_lock

LOG_ENV = "ASYNCROSCOPY_INTEGRITY_LOG"
DEFAULT_LOG = "outputs/integrity/manifest.jsonl"
KEY_ENV = "ASYNCROSCOPY_INTEGRITY_KEY"
KEY_FILE_ENV = "ASYNCROSCOPY_INTEGRITY_KEY_FILE"
DEFAULT_KEY_FILE = "outputs/integrity/key"
LOCK_WAIT_S = 5.0  # well inside a Tango client's timeout
CHUNK = 1 << 20
_guard = threading.Lock()
_index_cache: dict[str, tuple[tuple[int, float], dict[str, dict]]] = {}


class IntegrityError(RuntimeError):
    """A sealed file that is not what was written, or a manifest that has been altered."""


def manifest_path() -> Path:
    return Path(os.environ.get(LOG_ENV) or DEFAULT_LOG)


def _key() -> bytes:
    if os.environ.get(KEY_ENV):
        return bytes.fromhex(os.environ[KEY_ENV])
    path = Path(os.environ.get(KEY_FILE_ENV) or DEFAULT_KEY_FILE)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "x", encoding="ascii") as handle:  # only one process creates it
                handle.write(secrets.token_hex(32))
            os.chmod(path, 0o600)
        except FileExistsError:
            pass
    return bytes.fromhex(path.read_text(encoding="ascii").strip())


def _mac(key: bytes, fields: dict) -> str:
    return hmac.new(key, json.dumps(fields, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _last_line(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 65536))
        lines = handle.read().splitlines()
    return lines[-1].decode("utf-8", "replace") if lines else ""


def _ends_with_newline(path: Path) -> bool:
    """True for a missing or empty file too: nothing has to be closed off before the next line."""
    if not path.exists() or path.stat().st_size == 0:
        return True
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def _line_count(path: Path) -> int:
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def chain_append(log: Path, record: dict) -> dict:
    """Append ``record`` to the chained log ``log`` with its seq, the previous entry's mac and its own mac."""
    key = _key()
    log.parent.mkdir(parents=True, exist_ok=True)
    with _guard, open(log.with_name(log.name + ".lock"), "a+b") as lock, file_lock(lock, time.monotonic() + LOCK_WAIT_S):
        cut = False
        try:
            last = json.loads(_last_line(log) or "{}")
        except ValueError:  # a last line cut short by a crash: the chain reports it, the next entry follows it
            last, cut = {}, True
        seq = int(last.get("seq", 0)) + 1 if not cut else _line_count(log) + 1
        entry = {**record, "seq": seq, "prev": last.get("mac", "")}
        entry["mac"] = _mac(key, entry)
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(("\n" if not _ends_with_newline(log) else "") + json.dumps(entry) + "\n")
    return entry


def verify_chain(log: Path) -> dict:
    """{"entries", "problems": [...], "head"}: every line's mac and its link to the line before."""
    problems: list[str] = []
    entries, previous = 0, ""
    if not log.exists():
        return {"entries": 0, "problems": [], "head": ""}
    key = _key()
    with open(log, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                entry = json.loads(line)
            except ValueError:
                problems.append(f"{log.name} line {number} is not a record (cut short or edited)")
                previous = ""
                continue
            mac = entry.pop("mac", None) if isinstance(entry, dict) else None
            if not isinstance(entry, dict) or mac is None:
                problems.append(f"{log.name} line {number} has no seal")
                continue
            entries += 1
            if not hmac.compare_digest(mac, _mac(key, entry)):
                problems.append(f"{log.name} line {number} was changed after it was written (its seal does not match)")
            elif entry.get("prev", "") != previous:
                problems.append(f"{log.name} line {number} does not follow the line before it (a line was removed, inserted or moved)")
            previous = mac
    return {"entries": entries, "problems": problems, "head": previous}


def seal(path: str | Path, key: str = "", writer: str = "") -> dict:
    """Record ``path``'s SHA-256 and size in the manifest (a later seal of the same path supersedes it)."""
    resolved = Path(path).resolve()
    return chain_append(manifest_path(), {
        "path": str(resolved), "key": key or str(path), "sha256": sha256_of(resolved), "size": resolved.stat().st_size,
        "writer": writer[:120], "at": time.time(),
    })


def _index() -> dict[str, dict]:
    """The latest manifest entry per resolved path, re-read only when the manifest has changed."""
    log = manifest_path()
    if not log.exists():
        return {}
    stat = log.stat()
    stamp = (stat.st_size, stat.st_mtime)
    cached = _index_cache.get(str(log))
    if cached and cached[0] == stamp:
        return cached[1]
    index: dict[str, dict] = {}
    with open(log, encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                index[entry["path"]] = entry
    _index_cache[str(log)] = (stamp, index)
    return index


def check_file(path: str | Path) -> dict | None:
    """The file's manifest entry when it is sealed and unchanged; None when it was never sealed.
    Raises IntegrityError when it was sealed and its bytes are no longer what was written."""
    resolved = Path(path).resolve()
    entry = _index().get(str(resolved))
    if entry is None:
        return None
    if not resolved.exists():
        raise IntegrityError(f"{path} was sealed when written and is now missing")
    if resolved.stat().st_size != entry["size"] or sha256_of(resolved) != entry["sha256"]:
        raise IntegrityError(f"{path} has changed since it was written (its SHA-256 no longer matches the sealed one)")
    return entry


def verify(paths: list[str] | None = None, logs: tuple[Path, ...] = ()) -> dict:
    """Check the manifest's chain, every sealed file (or only ``paths``), and the other chained ``logs``.

    {"ok", "checked", "changed": [...], "missing": [...], "log_problems": [...], "head"}; "head" is the manifest's
    last mac: copy it elsewhere to notice a manifest whose end was cut off later."""
    chain = verify_chain(manifest_path())
    problems = list(chain["problems"])
    for log in logs:
        problems += verify_chain(log)["problems"]
    index = _index()
    wanted = index if paths is None else {str(Path(p).resolve()): index.get(str(Path(p).resolve())) for p in paths}
    changed, missing, unsealed = [], [], []
    for resolved, entry in wanted.items():
        if entry is None:
            unsealed.append(resolved)
            continue
        try:
            check_file(resolved)
        except IntegrityError as exc:
            (missing if "missing" in str(exc) else changed).append(entry.get("key") or resolved)
    report = {
        "ok": not (changed or missing or problems), "checked": len(wanted) - len(unsealed), "changed": changed, "missing": missing,
        "log_problems": problems, "head": chain["head"],
    }
    if unsealed:
        report["unsealed"] = unsealed
    return report
