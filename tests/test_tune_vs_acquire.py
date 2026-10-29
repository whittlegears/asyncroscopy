"""benchmarks/tune_vs_acquire.py --tiny: the tune-vs-acquire bench runs end to end and tuning pays off from a large error.

The bench starts its own in-process Tango context (twin, SCAN, Tuner), which PyTango allows
once per interpreter, so it runs in a child process. About 15-30 s.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_tune_vs_acquire(tmp_path):
    out = tmp_path / "tva.json"
    done = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "tune_vs_acquire.py"), "--tiny", "--out", str(out), "--work", str(tmp_path / "work")],
        cwd=ROOT, capture_output=True, text=True, timeout=240, check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]
    results = json.loads(out.read_text(encoding="utf-8"))
    assert out.with_suffix(".md").exists()
    rows = {row["budget"]: row for row in results["rows"]}
    assert set(rows) == {"none", "focus_only"}
    none, tuned = rows["none"], rows["focus_only"]
    # Tuning took time (fewer tiles fit in T) and lowered the focus error behind the map ...
    assert none["tune_s"] == 0 and tuned["tune_s"] > 0 and tuned["tune_state"] == "done"
    assert tuned["tiles"] < none["tiles"]
    assert tuned["median_focus_error_m"] < none["median_focus_error_m"]
    # ... and from a 250 nm defocus it more than paid for itself.
    assert tuned["information"] > none["information"] > 0
    assert results["summary"]["large_error"]["optimum"]["information"]["budget"] == "focus_only"
    # The evaluators read the run from its trace: the Tuner's events and the bench's acquisitions.
    assert tuned["events"] > none["events"]
    assert set(tuned["scores"]) == {"information", "resolution", "dose", "time"}
