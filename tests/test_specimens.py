"""Swappable specimens for the STEM twin (docs/Operation/specimens.md).

* The procedural specimen is bit-identical to the twin before specimens were
  plug-ins: images, spectra and ground truth from a fixed command sequence
  match tests/data/specimen_procedural_golden.npz.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tango

import asyncroscopy.instruments.electron_microscope.digital_twin as digital_twin_module

GOLDEN = Path(__file__).resolve().parent / "data" / "specimen_procedural_golden.npz"
# ASYNCROSCOPY_WRITE_GOLDEN=1 rewrites the golden file instead of checking it (only ever on purpose).
WRITE_GOLDEN = os.environ.get("ASYNCROSCOPY_WRITE_GOLDEN") == "1"
TRUTH_KEYS = ("sample_seed", "sample_bounds_m", "particles", "moves", "acquisitions", "simulated_time_s", "dose", "focus")


def _device(tango_ctx, name: str):
    tango.DeviceProxy(tango_ctx.get_device_access(name)).ping()
    return tango.Util.instance().get_device_by_name(name)


def _capture_saves(monkeypatch) -> list:
    """Keep what the twin would save (kind, detectors, arrays) instead of writing files."""
    saved: list = []

    def fake_save(device, data_server, kind, detectors, arrays, **kwargs):
        saved.append((kind, detectors, arrays))
        return f"key-{len(saved)}"

    monkeypatch.setattr(digital_twin_module, "save_acquisition", fake_save)
    return saved


def _golden_run(dev, saved: list) -> dict[str, np.ndarray]:
    """A fixed sequence touching every way the instrument reads the sample."""
    dev._move_stage([2e-8, -1e-8, 0.0, 0.0, 0.0])
    dev._set_fov(2e-7)
    dev._acquire_scanned_image(64, 2e-6, ["haadf", "bf"])
    dev._acquire_scanned_image(64, 2e-6, ["haadf"])  # past damage_critical_dose: particles fade
    particles = json.loads(dev.export_ground_truth())["particles"]
    nearest = min(particles, key=lambda p: np.hypot(p["center_m"][0] - 2e-8, p["center_m"][1] + 1e-8))
    dev._move_stage([*nearest["center_m"], 0.0, 0.0, 0.0])
    dev._set_fov(4e-9)  # 0.125 nm pixels: the atomic renderer
    dev._acquire_scanned_image(32, 2e-6, ["haadf"])
    dev._place_beam([0.5, 0.5])
    dev._acquire_spectrum("eds", 0.1)
    truth = json.loads(dev.export_ground_truth())
    images = [arrays for kind, _det, arrays in saved if kind == "stem_image"]
    spectra = [arrays for kind, _det, arrays in saved if kind == "spectrum"]
    return {
        "survey_haadf": np.asarray(images[0][0]),
        "survey_bf": np.asarray(images[0][1]),
        "survey2_haadf": np.asarray(images[1][0]),
        "atomic_haadf": np.asarray(images[2][0]),
        "spectrum": np.asarray(spectra[0]),
        "truth": np.array(json.dumps({k: truth[k] for k in TRUTH_KEYS}, sort_keys=True)),
    }


def test_procedural_specimen_is_bit_identical_to_the_golden_run(tango_ctx, monkeypatch):
    dev = _device(tango_ctx, "asyncroscopy/digitaltwin/specimen_golden")
    run = _golden_run(dev, _capture_saves(monkeypatch))
    if WRITE_GOLDEN:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, platform=np.array(sys.platform), numpy=np.array(np.__version__), **run)
        pytest.skip(f"wrote {GOLDEN}")
    golden = np.load(GOLDEN)
    assert json.loads(str(run["truth"])) == json.loads(str(golden["truth"]))  # particles, moves, drift, dose, focus
    # Survey damage is visible: the second frame of the same area is dimmer where particles are.
    assert run["survey2_haadf"].sum() != run["survey_haadf"].sum()
    same_build = str(golden["platform"]) == sys.platform and str(golden["numpy"]) == np.__version__
    for name in ("survey_haadf", "survey_bf", "survey2_haadf", "atomic_haadf", "spectrum"):
        if same_build:
            assert np.array_equal(run[name], golden[name]), f"{name} changed: the procedural specimen is no longer bit-identical"
        else:  # another platform's FFT and BLAS may round differently; the sample and the counts statistics still match
            assert np.allclose(run[name], golden[name], rtol=1e-3, atol=3.0), name
