"""view_data's words for stored spectra must not change: the live benchmarks depend on them.

tests/data/view_spectra_golden.json was written by the code before the X-ray logic moved out of data/views.py.
Each case is a synthetic stored spectrum (seeded counts) viewed as text and as a PNG's summary.
"""

import json
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parent / "data" / "view_spectra_golden.json"
OFFSET, DISPERSION, CHANNELS = 0.0, 10.0, 2048
SEED = 20261108


class Node:
    """A stored array, as Tiled serves it: shape, dtype, metadata and read(slices)."""

    def __init__(self, array: np.ndarray, metadata: dict) -> None:
        self.array, self.metadata = array, metadata
        self.shape, self.dtype = array.shape, array.dtype

    def read(self, slices=None) -> np.ndarray:
        return self.array if slices is None else self.array[slices]


def energy() -> np.ndarray:
    return OFFSET + DISPERSION * np.arange(CHANNELS, dtype=np.float64)


def expected_counts(lines: list[tuple[float, float]], background: float, fall_ev: float = 4000.0) -> np.ndarray:
    x = energy()
    counts = background * np.exp(-x / fall_ev)
    for position, height in lines:
        fwhm = np.sqrt(np.maximum(130.0**2 + 2.5 * (position - 5895.0), 30.0**2))
        counts += height * np.exp(-0.5 * ((x - position) / (fwhm / 2.355)) ** 2)
    return counts


def poisson(expected: np.ndarray, salt: int) -> np.ndarray:
    return np.random.default_rng(SEED + salt).poisson(expected).astype(np.uint32)


def eds_attrs(**extra) -> dict:
    return {"detector": "eds", "energy_offset_ev": OFFSET, "energy_dispersion_ev": DISPERSION, "units": "counts", **extra}


def cases() -> dict[str, tuple[Node, dict]]:
    x = energy()
    rise = 60.0 * np.exp(-0.5 * ((x - 600.0) / 180.0) ** 2)  # the detector window's bend of the continuum
    gold = [(2120.0, 900.0), (9713.0, 240.0), (11442.0, 110.0)]
    iron = [(6404.0, 1200.0), (7058.0, 170.0), (705.0, 120.0)]
    copper = [(8046.0, 1800.0), (8046.0 - 1740.0, 90.0), (2 * 8046.0, 60.0), (930.0, 200.0)]
    platinum = [(2050.0, 700.0), (9442.0, 300.0), (11071.0, 120.0), (8268.0, 40.0)]
    cube_rng = np.random.default_rng(SEED + 40)
    cube = cube_rng.poisson(np.broadcast_to(expected_counts(iron, 2.0)[:256] / 2.0, (3, 4, 256))).astype(np.uint16)
    return {
        "eds_gold": (Node(poisson(expected_counts(gold, 40.0), 1), eds_attrs()), {}),
        "eds_iron": (Node(poisson(expected_counts(iron, 60.0), 2), eds_attrs(acquisition_type="spectrum")), {}),
        "eds_copper_escape_and_sum": (Node(poisson(expected_counts(copper, 30.0), 3), eds_attrs()), {}),
        "eds_platinum_minor_lines": (Node(poisson(expected_counts(platinum, 20.0), 4), eds_attrs()), {}),
        "eds_window_rise_only": (Node(poisson(expected_counts([], 25.0) + rise, 5), eds_attrs()), {}),
        "eds_carbon_on_the_rise": (Node(poisson(expected_counts([(277.0, 300.0)], 25.0) + rise, 6), eds_attrs()), {}),
        "eds_noise_only": (Node(poisson(expected_counts([], 0.4), 7), eds_attrs()), {}),
        "eds_no_counts": (Node(np.zeros(CHANNELS, dtype=np.uint32), eds_attrs()), {}),
        "eds_cropped": (Node(poisson(expected_counts(iron, 60.0), 8), eds_attrs()), {"region": [5000.0, 8000.0]}),
        "raman": (Node(poisson(expected_counts([(1332.0, 800.0), (520.0, 300.0)], 15.0), 9), {
            "axis_offset": 0.0, "axis_scale": 10.0, "axis_label": "Raman shift", "axis_units": "cm-1"}), {}),
        "eels": (Node(poisson(expected_counts([(284.0, 500.0), (532.0, 200.0)], 80.0), 10), {
            "detector": "eels", "start_energy": 0.0, "dispersion": 10.0, "energy_units": "eV"}), {}),
        "profile": (Node(poisson(expected_counts([(5000.0, 400.0)], 10.0), 11), {}), {}),
        "composition": (Node(np.array([0.61, 0.27, 0.12]), {"elements": ["Fe", "Cr", "Ni"]}), {}),
        "spectrum_image": (Node(cube, eds_attrs()), {}),
        "spectrum_image_position": (Node(cube, eds_attrs()), {"index": [1, 2]}),
    }


def outputs() -> dict[str, dict]:
    from asyncroscopy.data.views import ViewOptions, summarize, view

    out = {}
    for name, (node, options) in cases().items():
        entry = {"summarize": summarize(node, name)}
        for fmt in ("text", "png"):
            summary, png = view(node, name, fmt, None, ViewOptions(**options))
            assert (png is not None) == (fmt == "png")
            entry[fmt] = summary
        out[name] = entry
    return json.loads(json.dumps(out, default=str))


def test_view_output_on_stored_spectra_is_what_it_was_before_the_move():
    from asyncroscopy import settings

    produced = outputs()
    if settings.get().write_golden:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(produced, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(produced) == sorted(golden)
    for name in golden:
        assert produced[name] == golden[name], name
