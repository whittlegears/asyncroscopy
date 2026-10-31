"""Tamper evidence for files and chained logs.

``seal(path)`` records a file's SHA-256 and size in an append-only manifest (``ASYNCROSCOPY_INTEGRITY_LOG``,
default ``outputs/integrity/manifest.jsonl``). Each manifest line carries an HMAC (a keyed SHA-256) over its own
fields and the HMAC of the line before it (``chain_append``), so:

* a sealed file whose bytes change no longer matches its hash, and ``check_file`` raises ``IntegrityError``;
* a line edited, deleted, inserted or moved breaks its HMAC or the chain after it (``verify_chain``);
* forging a line, or re-sealing a changed file, needs the key.

``verify`` checks the manifest's chain, every sealed file, and any other chained logs it is given.

``save_acquisition`` seals every file it writes (``data/data_writer.py``), and the dose ledger is a chained log
(``data/dose_ledger.py``). The analysis server runs without the key: ``verify_integrity.py --seal`` seals its map
files (``seal``) and the lines its placement log holds so far (``seal_log``). Maps it writes to Tiled are not sealed. ``startup_scripts/verify_integrity.py``
runs ``verify`` and prints what changed, went missing or broke the chain.

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
import math
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


def sha256_of(path: Path, limit: int | None = None) -> str:
    """The SHA-256 of ``path``'s bytes, or of its first ``limit`` bytes."""
    digest = hashlib.sha256()
    left = math.inf if limit is None else limit
    with open(path, "rb") as handle:
        while left > 0:
            block = handle.read(int(min(CHUNK, left)))
            if not block:
                break
            digest.update(block)
            left -= len(block)
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


def _legacy_lines(log: Path) -> tuple[int, str]:
    """The number of lines before the first sealed one and the SHA-256 of their bytes (written before chaining)."""
    digest, count = hashlib.sha256(), 0
    with open(log, "rb") as handle:
        for raw in handle:
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and "mac" in parsed:
                break
            digest.update(raw.rstrip(b"\r\n") + b"\n")
            count += 1
    return count, digest.hexdigest()


def chain_append(log: Path, record: dict) -> dict:
    """Append ``record`` to the chained log ``log`` with its seq, the previous entry's mac and its own mac.

    The first sealed line of a log that already holds unsealed lines (written before it was chained) records
    how many there are and the SHA-256 of their bytes, so those lines cannot later be edited, added to or
    stripped of their seals unnoticed."""
    key = _key()
    log.parent.mkdir(parents=True, exist_ok=True)
    with _guard, open(log.with_name(log.name + ".lock"), "a+b") as lock, file_lock(lock, time.monotonic() + LOCK_WAIT_S):
        cut = False
        try:
            last = json.loads(_last_line(log) or "{}")
        except ValueError:  # a last line cut short by a crash: the chain reports it, the next entry follows it
            last, cut = {}, True
        if not isinstance(last, dict):
            last, cut = {}, True
        seq = int(last.get("seq", 0)) + 1 if not cut else _line_count(log) + 1
        entry = {**record, "seq": seq, "prev": last.get("mac", "")}
        if (cut or "mac" not in last) and log.exists() and log.stat().st_size:
            count, digest = _legacy_lines(log)
            if count and count == _line_count(log):
                entry.update(legacy_lines=count, legacy_sha256=digest)
        entry["mac"] = _mac(key, entry)
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(("\n" if not _ends_with_newline(log) else "") + json.dumps(entry) + "\n")
    return entry


def verify_chain(log: Path) -> dict:
    """{"entries", "problems": [...], "head"}: every line's mac and its link to the line before.

    Lines at the start of a log written before it was chained are counted as "unsealed_before_chain". They are
    accepted only when the first sealed line recorded that many lines with the same SHA-256, or when no line
    is sealed yet; an unsealed line after the first sealed one is a problem."""
    problems: list[str] = []
    entries, previous = 0, ""
    legacy, legacy_digest = 0, hashlib.sha256()
    if not log.exists():
        return {"entries": 0, "problems": [], "head": ""}
    key = _key()
    with open(log, "rb") as handle:
        for number, raw in enumerate(handle, 1):
            try:
                entry = json.loads(raw)
            except ValueError:
                entry = None
            mac = entry.pop("mac", None) if isinstance(entry, dict) else None
            if mac is None:
                if entries == 0:
                    legacy += 1
                    legacy_digest.update(raw.rstrip(b"\r\n") + b"\n")
                elif entry is None:
                    problems.append(f"{log.name} line {number} is not a record (cut short or edited)")
                    previous = ""
                else:
                    problems.append(f"{log.name} line {number} has no seal")
                continue
            if entries == 0 and legacy and (entry.get("legacy_lines") != legacy or entry.get("legacy_sha256") != legacy_digest.hexdigest()):
                problems.append(f"{log.name}: the {legacy} line(s) before line {number} are not the unsealed lines its first seal recorded (lines were edited, added or stripped of their seals)")
            entries += 1
            if not hmac.compare_digest(mac, _mac(key, entry)):
                problems.append(f"{log.name} line {number} was changed after it was written (its seal does not match)")
            elif entry.get("prev", "") != previous:
                problems.append(f"{log.name} line {number} does not follow the line before it (a line was removed, inserted or moved)")
            previous = mac
    report = {"entries": entries, "problems": problems, "head": previous}
    if legacy:
        report["unsealed_before_chain"] = legacy
    return report


def seal(path: str | Path, key: str = "", writer: str = "") -> dict:
    """Record ``path``'s SHA-256 and size in the manifest. Sealing an unchanged file again returns its entry;
    a file sealed before whose bytes have changed is refused (IntegrityError): a sealed result is never re-sealed
    as something else."""
    resolved = Path(path).resolve()
    digest, size = sha256_of(resolved), resolved.stat().st_size
    earlier = _index().get(str(resolved))
    if earlier is not None:
        if earlier.get("sha256") == digest and earlier.get("size") == size:
            return earlier
        raise IntegrityError(f"{path} was sealed when written and has changed since; it is not sealed again")
    return chain_append(manifest_path(), {
        "path": str(resolved), "key": key or str(path), "sha256": digest, "size": size,
        "writer": writer[:120], "at": time.time(),
    })


def seal_log(path: str | Path, key: str = "", writer: str = "") -> dict:
    """Seal the complete lines an append-only log written by another program holds now (the analysis server's
    placement log): their count, byte length and SHA-256. The log may keep growing; ``check_file`` then checks
    only that the sealed lines are unchanged. A log whose sealed lines have changed is refused (IntegrityError)."""
    resolved = Path(path).resolve()
    earlier = _index().get(str(resolved))
    if earlier is not None:
        check_file(resolved)
    with open(resolved, "rb") as handle:
        data = handle.read()
    size = data.rfind(b"\n") + 1
    if earlier is not None and earlier["size"] == size:
        return earlier
    return chain_append(manifest_path(), {
        "path": str(resolved), "key": key or str(path), "sha256": hashlib.sha256(data[:size]).hexdigest(), "size": size,
        "lines": data[:size].count(b"\n"),"writer": writer[:120], "at": time.time(),
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
    if "lines" in entry:
        if resolved.stat().st_size < entry["size"] or sha256_of(resolved, entry["size"]) != entry["sha256"]:
            raise IntegrityError(f"{path}: its first {entry['lines']} line(s) have changed since they were sealed")
        return entry
    if resolved.stat().st_size != entry["size"] or sha256_of(resolved) != entry["sha256"]:
        raise IntegrityError(f"{path} has changed since it was written (its SHA-256 no longer matches the sealed one)")
    return entry


def verify(paths: list[str] | None = None, logs: tuple[Path, ...] = ()) -> dict:
    """Check the manifest's chain, every sealed file (or only ``paths``), and the other chained ``logs``.

    {"ok", "checked", "changed": [...], "missing": [...], "log_problems": [...], "head"}; "head" is the manifest's
    last mac: copy it elsewhere to notice a manifest whose end was cut off later."""
    chain = verify_chain(manifest_path())
    problems = list(chain["problems"])
    before_chain, heads = {}, {}
    for log in logs:
        checked = verify_chain(log)
        problems += checked["problems"]
        heads[str(log)] = checked["head"]
        if checked.get("unsealed_before_chain"):
            before_chain[str(log)] = checked["unsealed_before_chain"]
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
    if heads:
        report["log_heads"] = heads
    if before_chain:
        report["unsealed_before_chain"] = before_chain
    return report
