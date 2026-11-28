"""The aberration notebooks' helper code runs, and uses the shared FFT score from the ANALYSIS device."""

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def _cell(notebook: str, marker: str) -> str:
    """Source of the one code cell that contains ``marker``."""
    cells = json.loads((NOTEBOOKS / notebook).read_text(encoding="utf-8"))["cells"]
    [source] = ["".join(c["source"]) for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"])]
    return source


def _run(namespace: dict, notebook: str, *markers: str) -> dict:
    for marker in markers:
        exec(compile(_cell(notebook, marker), f"{notebook}:{marker}", "exec"), namespace)
    return namespace


class FakeAnalysis:
    """Answers resolution() like the ANALYSIS device: Nyquist 2 A, resolution as given."""

    def __init__(self, resolution_angstrom):
        self.resolution_angstrom = resolution_angstrom

    def resolution(self, key):
        return json.dumps({"key": key, "nyquist_angstrom": 2.0, "resolution_angstrom": self.resolution_angstrom})


class FakeCorrector:
    def __init__(self):
        self.state = {"C1": [5e-9], "A1": [1e-9, 2e-9], "C3": [1e-6]}

    def get_aberrations_coeff_sim(self):
        return json.dumps(self.state)

    def set_aberrations_coeff_sim(self, text):
        self.state = json.loads(text)


class FakeNode(dict):
    def __init__(self, array=None, **children):
        super().__init__(children)
        self.array = array

    def read(self):
        return self.array


@pytest.mark.parametrize("notebook", ["12_Aberration_Hill_Climbing.ipynb", "12_Digital_twin_Aberration_Hill_Climbing.ipynb"])
def test_hill_climbing_scores_with_the_analysis_device(notebook):
    namespace = _run({"np": np, "json": json, "scan": SimpleNamespace(), "analysis": FakeAnalysis(4.0)},
                     notebook, "def acquire_and_measure")

    assert namespace["fft_sharpness"]("key-1") == pytest.approx(0.5)
    namespace["analysis"] = FakeAnalysis(None)
    assert namespace["fft_sharpness"]("key-1") == 0.0


def test_bayesian_optimization_seed_loop_runs_on_the_current_devices():
    image = np.random.default_rng(0).random((32, 32))
    corrector = FakeCorrector()
    namespace = {
        "np": np, "json": json, "plt": plt, "corrector": corrector, "analysis": FakeAnalysis(4.0),
        "microscope": SimpleNamespace(acquire_scanned_image=lambda detectors: "key-1"),
        "client": {"key-1": FakeNode(image=FakeNode(HAADF=FakeNode(image)))},
    }
    _run(namespace, "AberrationsBO.ipynb", "def read_haadf", "def acquire_stem_image_contrast_and_fft",
         "param_ranges = {", "full_grid = np.stack", "seed_indices =", "QUERY SEED POINTS")
    plt.close("all")

    seeds = namespace["seed_points"]
    assert namespace["seed_scores"].shape == (len(seeds), 2)
    assert namespace["seed_scores"][:, 1] == pytest.approx(0.5)  # FFT score from the ANALYSIS device
    c10, c12a, c12b = seeds[-1]
    assert corrector.state["C1"] == pytest.approx([c10 * 1e-9])
    assert corrector.state["A1"] == pytest.approx([c12a * 1e-9, c12b * 1e-9])
    assert corrector.state["C3"] == [1e-6]  # other aberrations untouched
