"""Render-time budget for the twin: a mapping run must not be bottlenecked by simulation.

Marked slow; run with ``pytest -m slow``. Times are written to
benchmarks/results/twin_render.json so regressions are visible.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from asyncroscopy.instruments.electron_microscope.twin import (
    EventRng,
    SampleModel,
    SampleParams,
    ScanParams,
    pixel_sample_coordinates,
)
from asyncroscopy.instruments.electron_microscope.twin.optics import aberrations_nm
from asyncroscopy.instruments.electron_microscope.twin.render import (
    Renderer,
    RenderParams,
)

RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "twin_render.json"


@pytest.mark.slow
def test_render_time_budget():
    sample = SampleModel(
        SampleParams(size_xy_m=2e-6, particle_count=400, radius_mean_m=8e-9, radius_std_m=4e-9, aspect_ratio=0.6, elements=("Au", "Pt", "Fe", "Pd"), support_thickness_m=1e-8),
        EventRng(12345),
    )
    renderer = Renderer(sample, RenderParams())
    ab = aberrations_nm(None, 300e3, 0.0)
    biggest = sample.centers[int(np.argmax(sample.semi_axes[:, 0]))]
    timings = {}
    for name, fov, origin in [("survey_1um_512", 1e-6, (0.0, 0.0)), ("atomic_10nm_512", 1e-8, biggest)]:
        xs, ys = pixel_sample_coordinates(512, fov, (0, 0, 1, 1), 1e-6, ScanParams(), lambda t, o=origin: np.asarray(o), 0.0, EventRng(1).for_event("scan"))
        start = time.perf_counter()
        renderer.render(xs, ys, fov / 512, "HAADF", ab, 100.0)
        timings[name] = time.perf_counter() - start
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({k: round(v, 4) for k, v in timings.items()}, indent=2))
    assert timings["survey_1um_512"] < 1.0
    assert timings["atomic_10nm_512"] < 2.0
