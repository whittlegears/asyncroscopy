"""The fields every dataset this package writes carries (decision 7.2 of the concepts brief).

* ``id``: permanent, issued once when the file is first written (a resumed map keeps it);
* ``instrument_id``: the Tango device that wrote it;
* ``kind``: what it is (``stem_image``, ``eds_spectrum``, ``map_run``, ...);
* ``sample_id`` and ``sample_load_id``: what the STAGE device's ``load_sample`` last recorded. A new
  load gets a new ``sample_load_id``, so a position read from an older dataset or note can be told
  apart from one taken on the sample in the holder now. Both are null when no STAGE device answers;
  ``sample_error`` then says why;
* ``code_version``: this package's version and, in a git checkout, its commit;
* ``created``: when it was written (Unix seconds).

Settings stay where they were (each dataset's own attributes, and a map's ``plan``), and a map run
file also records the agent ``run`` and the trace offset its events start at.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RECORD_KEYS = ("id", "instrument_id", "kind", "sample_id", "sample_load_id", "code_version", "created")


@lru_cache(maxsize=1)
def code_version() -> str:
    """``asyncroscopy <version>``, plus ``+g<commit>`` (and ``.dirty``) in a git checkout."""
    try:
        text = f"asyncroscopy {version('asyncroscopy')}"
    except PackageNotFoundError:
        text = "asyncroscopy (not installed)"
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return text
    return f"{text}+g{commit}{'.dirty' if dirty else ''}" if commit else text


def current_sample(stage) -> dict:
    """{"sample_id", "sample_load_id"} from a STAGE device proxy, or nulls with "sample_error" saying why."""
    if stage is None:
        return {"sample_id": None, "sample_load_id": None, "sample_error": "no STAGE device is configured for this instrument"}
    try:
        sample = json.loads(stage.get_sample())
    except Exception as exc:
        return {"sample_id": None, "sample_load_id": None, "sample_error": f"the STAGE device did not say which sample is loaded: {str(exc).splitlines()[0][:200]}"}
    found = {"sample_id": sample.get("sample_id"), "sample_load_id": sample.get("sample_load_id")}
    if sample.get("error"):
        found["sample_error"] = str(sample["error"])[:300]
    elif found["sample_load_id"] is None:
        found["sample_error"] = "no sample load is recorded: an operator calls the STAGE device's load_sample when a sample goes in"
    return found


def record_fields(instrument_id: str, kind: str, sample: dict) -> dict:
    """The record fields for one new dataset (``sample``: what current_sample gave); null values are left out
    of the file (HDF5 has no null)."""
    fields = {
        "id": uuid.uuid4().hex,
        "instrument_id": instrument_id,
        "kind": kind,
        **sample,
        "code_version": code_version(),
        "created": time.time(),
    }
    return {key: value for key, value in fields.items() if value is not None}


def stage_of(device):
    """The STAGE proxy a microscope device already holds (twin and hardware devices keep it with their detectors)."""
    proxies = getattr(device, "_detector_proxies", None)
    return proxies.get("stage") if isinstance(proxies, dict) else None
