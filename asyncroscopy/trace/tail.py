"""Follow the Instrument Trace live: ``python -m asyncroscopy.trace.tail --run <id>``.

Prints every event of every trace file (default: ``outputs/trace``) in order, then the new
ones as they are written, until Ctrl+C. See docs/Operation/trace.md.

Examples::

    python -m asyncroscopy.trace.tail                          # everything, then follow
    python -m asyncroscopy.trace.tail --run 3f9c... --json     # one agent run, as JSON lines
    python -m asyncroscopy.trace.tail --device Mapper --since 10m --no-follow
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime

from asyncroscopy.trace.stream import DEFAULT_SOURCES, TraceStream

UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_since(text: str | None) -> float | None:
    """'90s', '10m', '2h', '1d' ago; an ISO time; or Unix seconds."""
    if not text:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", text.strip())
    if match:
        return time.time() - float(match.group(1)) * UNITS[match.group(2)]
    try:
        return float(text)
    except ValueError:
        return datetime.fromisoformat(text).timestamp()


def line(event: dict) -> str:
    """One readable line: time, source, action, ok, then who and what."""
    when = datetime.fromtimestamp(float(event.get("written", event.get("ts", 0)))).strftime("%H:%M:%S.%f")[:-3]
    mark = "ok " if event.get("ok", True) else "ERR"
    parts = [when, f"{event.get('source', '?'):<9}", mark, str(event.get("action") or event.get("tool") or "?")]
    for key in ("caller", "run", "job_id", "step", "clock_s", "error"):
        value = event.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    if event.get("dry_run"):
        parts.append("dry_run")
    if event.get("simulated") is False:
        parts.append("HARDWARE")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m asyncroscopy.trace.tail", description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="*", help=f"Trace files, directories or globs (default: {' '.join(DEFAULT_SOURCES)}).")
    parser.add_argument("--run", action="append", help="Only this agent run (X-Asyncroscopy-Run); repeatable.")
    parser.add_argument("--caller", action="append", help="Only this caller; repeatable.")
    parser.add_argument("--device", action="append", help="Only this device: a Tango name or a tool prefix such as Mapper; repeatable.")
    parser.add_argument("--job", action="append", help="Only this job: job_id, reservation id or plan_id; repeatable.")
    parser.add_argument("--since", help="Start at this time: 10m, 2h (ago), an ISO time or Unix seconds.")
    parser.add_argument("--from-end", action="store_true", help="Skip what is already written; show only new events.")
    parser.add_argument("--no-follow", action="store_true", help="Print what is written so far and stop.")
    parser.add_argument("--json", action="store_true", help="Print each event as one JSON line.")
    parser.add_argument("--timeout", type=float, default=None, help="Stop following after this many seconds.")
    args = parser.parse_args(argv)

    stream = TraceStream(
        args.sources or None, run=args.run, caller=args.caller, device=args.device, job=args.job,
        since=parse_since(args.since), from_end=args.from_end,
    )
    show = (lambda e: json.dumps(e, default=str)) if args.json else line
    try:
        events = stream.poll(final=True) if args.no_follow else stream.follow(timeout=args.timeout)
        for event in events:
            print(show(event), flush=True)
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:  # e.g. piped into head
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
