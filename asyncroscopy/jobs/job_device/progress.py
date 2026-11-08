"""Which steps of a job are done: ranges, step counts, the tail of its step file."""

from __future__ import annotations

import bisect
import json
from collections import deque
from pathlib import Path

def step_count(job: dict) -> int:
    """How many steps the job has: a list from resolve_plan, or a step_count with step_at."""
    return int(job["step_count"]) if "step_count" in job else len(job["steps"])


def read_last_steps(path: Path, count: int) -> dict:
    """The last ``count`` step records of a job's .steps.jsonl, read from the end of the file.

    Start-up cost does not grow with the job's history: a job of a million steps reads only
    its tail. A line cut short by a crash is skipped.
    """
    if not path.exists():
        return {}
    newest: deque = deque(maxlen=count)
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = position = handle.tell()
        block = 1 << 16
        data = b""
        while position > 0 and data.count(b"\n") <= count:
            position = max(0, position - block)
            handle.seek(position)
            data = handle.read(end - position)
        lines = data.split(b"\n")
        if position > 0:
            lines = lines[1:]  # the first line may start before the block
    for line in lines[-(count + 1):]:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        newest.append((str(row["index"]), row["record"]))
    return dict(newest)


def done_count(job: dict) -> int:
    """Steps done, from the job's sorted [start, end) ranges."""
    return sum(end - start for start, end in job["done_ranges"])


def is_done(job: dict, index: int) -> bool:
    ranges = job["done_ranges"]
    k = bisect.bisect_right(ranges, [index, float("inf")]) - 1
    return k >= 0 and ranges[k][0] <= index < ranges[k][1]


def mark_done(job: dict, index: int) -> None:
    """Add ``index``, merging ranges, so a job of a million steps stores a few numbers, not a million."""
    ranges = job["done_ranges"]
    if is_done(job, index):
        return
    k = bisect.bisect_right(ranges, [index, float("inf")])
    joins_left = k > 0 and ranges[k - 1][1] == index
    joins_right = k < len(ranges) and ranges[k][0] == index + 1
    if joins_left and joins_right:
        ranges[k - 1][1] = ranges[k][1]
        del ranges[k]
    elif joins_left:
        ranges[k - 1][1] = index + 1
    elif joins_right:
        ranges[k][0] = index
    else:
        ranges.insert(k, [index, index + 1])


def done_ranges_from(job: dict) -> list[list[int]]:
    """Ranges from a saved job: new files store done_ranges, older ones a list of indices."""
    if "done_ranges" in job:
        return [[int(a), int(b)] for a, b in job["done_ranges"]]
    rebuilt = {"done_ranges": []}
    for index in job.get("done", []):
        mark_done(rebuilt, int(index))
    return rebuilt["done_ranges"]
