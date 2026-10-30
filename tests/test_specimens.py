"""Swappable specimens for the STEM twin (docs/Operation/specimens.md).

* The procedural specimen is bit-identical to the twin before specimens were
  plug-ins: images, spectra and ground truth from a fixed command sequence
  match tests/data/specimen_procedural_golden.npz.
* Sources are registered by name; the ``specimen`` property is parsed the way
  run_servers passes it, and every shipped config's specimen builds.
* recorded: any image format, resampled at the instrument's pixel; ground truth
  unknown without labels. crystal: lattice columns at a/2, grains, EDS.
* Same controller, different samples: one Mapper plan on three twins that share
  the instrument and differ only in the specimen. Moves, drift, timing and dose
  are identical; the images and spectra are not.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import tango

import asyncroscopy.instruments.electron_microscope.digital_twin as digital_twin_module
from asyncroscopy.instruments.electron_microscope.twin import EventRng, SampleParams
from asyncroscopy.instruments.electron_microscope.twin.optics import aberrations_nm
from asyncroscopy.instruments.electron_microscope.twin.render import Renderer, RenderParams
from asyncroscopy.instruments.electron_microscope.twin.specimen import (
    SPECIMENS,
    Specimen,
    SpecimenContext,
    build_specimen,
    parse_specimen,
    register_specimen,
    specimen_sources,
)
from asyncroscopy.instruments.electron_microscope.twin.specimen_recorded import Labels, load_image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_IMAGE = ROOT / "configs" / "specimens" / "recorded_example_haadf.png"
EXAMPLE_LABELS = ROOT / "configs" / "specimens" / "recorded_example_labels.json"
EXAMPLE_PIXEL = 1.5625e-9
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


def _contains(new, old) -> bool:
    """Whether ``new`` holds everything ``old`` does, unchanged (dicts may have gained keys)."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(k in new and _contains(new[k], v) for k, v in old.items())
    if isinstance(old, list):
        return isinstance(new, list) and len(new) == len(old) and all(_contains(a, b) for a, b in zip(new, old))
    return new == old


def test_procedural_specimen_is_bit_identical_to_the_golden_run(tango_ctx, monkeypatch):
    dev = _device(tango_ctx, "asyncroscopy/digitaltwin/specimen_golden")
    run = _golden_run(dev, _capture_saves(monkeypatch))
    if WRITE_GOLDEN:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, platform=np.array(sys.platform), numpy=np.array(np.__version__), **run)
        pytest.skip(f"wrote {GOLDEN}")
    golden = np.load(GOLDEN)
    truth, recorded = json.loads(str(run["truth"])), json.loads(str(golden["truth"]))
    # Ground truth may gain fields (lattice orientations, each frame's scan distortion and detector gain); everything
    # the golden run recorded must still be there unchanged: particles, moves, drift, focus.
    dose, recorded_dose = truth.pop("dose"), recorded.pop("dose")
    assert _contains(truth, recorded)
    # The dose map keeps each frame's dose in cells sized to it and counts every electron (the old map dropped
    # a few at cells its grid lined up badly with), so its cell count differs and its peak by 1 part in 10^5.
    assert dose["max_dose_e_per_a2"] == pytest.approx(recorded_dose["max_dose_e_per_a2"], rel=1e-3)
    images = [a for a in truth["acquisitions"] if a["kind"] == "image"]
    assert images and all({"rotation_deg", "shear", "fov_scale_error"} <= set(a["scan"]) and set(a["detector"]) == {"gain", "offset"} for a in images)
    # Survey damage is visible: the second frame of the same area is dimmer where particles are.
    assert run["survey2_haadf"].sum() != run["survey_haadf"].sum()
    same_build = str(golden["platform"]) == sys.platform and str(golden["numpy"]) == np.__version__
    for name in ("survey_haadf", "survey_bf", "survey2_haadf", "atomic_haadf", "spectrum"):
        if same_build:
            assert np.array_equal(run[name], golden[name]), f"{name} changed: the procedural specimen is no longer bit-identical"
        else:  # another platform's FFT and BLAS may round differently; the sample and the counts statistics still match
            assert np.allclose(run[name], golden[name], rtol=1e-3, atol=3.0), name


# ----------------------------------------------------------------------
# Registry and selection
# ----------------------------------------------------------------------
def _context(**sample) -> SpecimenContext:
    return SpecimenContext(rng=EventRng(3), sample=SampleParams(**{"size_xy_m": 4e-7, "particle_count": 30, "radius_mean_m": 8e-9, **sample}), base_dir=ROOT)


def _render(specimen: Specimen, fov: float, n: int, centre=(0.0, 0.0), detector: str = "HAADF") -> np.ndarray:
    coords = (np.arange(n) + 0.5) * fov / n - fov / 2
    xs, ys = np.meshgrid(coords + centre[0], coords + centre[1])
    return Renderer(specimen, RenderParams()).render(xs, ys, fov / n, detector, aberrations_nm(None, 300e3, 0.0), 0.0)


def test_built_in_sources_are_registered_and_unknown_ones_are_named():
    assert {"procedural", "recorded", "crystal"} <= set(specimen_sources())
    with pytest.raises(ValueError, match=r"unknown specimen source 'nope'; registered: \[.*'crystal'.*'procedural'.*'recorded'"):
        build_specimen({"source": "nope"}, _context())
    with pytest.raises(ValueError, match=r"has no option\(s\) \['grains'\]"):  # a typo must not pass silently
        build_specimen({"source": "crystal", "grains": 4}, _context())


def test_a_plug_in_source_needs_only_the_interface():
    @register_specimen
    class Slab(Specimen):
        """A uniform carbon slab: the smallest useful source."""

        source = "test_slab"

        def __init__(self, thickness_m):
            self.thickness_m = thickness_m
            self.bounds = (-1e-7, 1e-7, -1e-7, 1e-7)

        @classmethod
        def from_options(cls, options, context):
            return cls(float(options.get("thickness_m", 2e-8)))

        def projected_weight(self, x_min, y_min, n, pixel_m, level, z_exponent, damage):
            return np.full((n, n), 6.0 ** z_exponent * 0.1 * self.thickness_m / 1e-10)

        def areal_density(self, x, y):
            return {"C": 0.1 * self.thickness_m / 1e-10}

    try:
        slab = build_specimen({"source": "test_slab", "thickness_m": 4e-8}, _context())
        image = _render(slab, 1e-7, 16)
        assert np.allclose(image, image.mean()) and image.mean() > 0
        assert slab.ground_truth() == {"known": False, "particles": []}
    finally:
        SPECIMENS.pop("test_slab", None)


def test_the_documented_example_source_works():
    """docs/Operation/specimens.md's 40-line source, run from the page itself."""
    page = (ROOT / "docs" / "Operation" / "specimens.md").read_text(encoding="utf-8")
    code = next(block for block in page.split("```python\n")[1:] if "class CrossGrating" in block).split("```")[0]
    assert len(code.strip().splitlines()) <= 50
    try:
        exec(compile(code, "specimens.md", "exec"), {"__name__": "specimens_doc"})
        grating = build_specimen({"source": "cross_grating", "pitch_m": 1e-7}, _context(size_xy_m=1e-6))
        image = _render(grating, 4e-7, 64)
        assert image.max() > 0.05 and image.min() < 0.01  # lines and holes
        assert np.allclose(image[:, :16], image[:, 16:32], atol=0.02)  # 16 pixels is one 100 nm pitch
        assert grating.areal_density(1e-9, 5e-8) == {"Au": pytest.approx(0.059 * 200)}
        assert grating.areal_density(5e-8, 5e-8) == {}
    finally:
        SPECIMENS.pop("cross_grating", None)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", {}),
        ("crystal", {"source": "crystal"}),
        ('{"source": "recorded", "pixel_size_m": 1e-9}', {"source": "recorded", "pixel_size_m": 1e-9}),
        (str({"source": "recorded", "path": "C:\\data\\x.png", "invert": True}), {"source": "recorded", "path": "C:\\data\\x.png", "invert": True}),
        ("{source: crystal, grain_count: 4}", {"source": "crystal", "grain_count": 4}),
    ],
)
def test_the_specimen_property_takes_json_yaml_python_or_a_name(text, expected):
    assert parse_specimen(text) == expected


def test_the_specimen_property_refuses_a_list():
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_specimen("[1, 2]")


def _launcher_config(path: Path):
    spec = importlib.util.spec_from_file_location("launcher_run_servers", ROOT / "startup_scripts" / "run_servers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["launcher_run_servers"] = module
    spec.loader.exec_module(module)
    return module.load_config(path)


@pytest.mark.parametrize("name, source", [("digital_twin_mapping.yaml", "procedural"), ("digital_twin_recorded.yaml", "recorded"), ("digital_twin_crystal.yaml", "crystal")])
def test_each_example_config_builds_its_specimen(name, source):
    """run_servers turns the YAML mapping into one string property; the twin parses it back and builds it."""
    properties = _launcher_config(ROOT / "configs" / name).instrument.properties
    options = parse_specimen(properties["specimen"][0])
    assert options["source"] == source
    size = float(properties.get("sample_size_xy", ["2e-6"])[0])
    specimen = build_specimen(options, _context(size_xy_m=size))
    assert specimen.source == source and specimen.bounds[1] > specimen.bounds[0]


def test_a_procedural_seed_option_changes_the_sample_and_nothing_else():
    a = build_specimen({"source": "procedural"}, _context())
    b = build_specimen({"source": "procedural", "seed": 99}, _context())
    again = build_specimen({"source": "procedural"}, _context())
    assert a.particle_table() == again.particle_table()
    assert a.particle_table() != b.particle_table()


# ----------------------------------------------------------------------
# recorded
# ----------------------------------------------------------------------
def test_recorded_images_load_from_every_format_with_their_pixel_size(tmp_path):
    image, pixel = load_image(EXAMPLE_IMAGE)
    assert image.shape == (256, 256) and pixel is None  # a PNG carries no pixel size: the option gives it
    np.save(tmp_path / "a.npy", image)
    np.savez(tmp_path / "a.npz", image=image, pixel_size_m=EXAMPLE_PIXEL)
    with h5py.File(tmp_path / "a.h5", "w") as h5:  # a Mapper run file: a stack of tiles, pixel size in attrs
        dset = h5.create_dataset("tiles", data=np.stack([image * 0, image]))
        dset.attrs["pixel_size_m"] = EXAMPLE_PIXEL
    built = {}
    for name, extra in (("a.npy", {"pixel_size_m": EXAMPLE_PIXEL}), ("a.npz", {}), ("a.h5", {"index": 1})):
        built[name] = build_specimen({"source": "recorded", "path": str(tmp_path / name), **extra}, _context())
        assert built[name].pixel_m == EXAMPLE_PIXEL
        assert np.array_equal(built[name].image, built["a.npy"].image)
    with pytest.raises(ValueError, match="pixel_size_m"):
        build_specimen({"source": "recorded", "path": str(EXAMPLE_IMAGE)}, _context())


def test_recorded_specimen_is_imaged_by_the_instrument_not_copied():
    specimen = build_specimen({"source": "recorded", "path": "configs/specimens/recorded_example_haadf.png", "pixel_size_m": EXAMPLE_PIXEL}, _context())
    assert specimen.bounds == pytest.approx((-2e-7, 2e-7, -2e-7, 2e-7))
    native = _render(specimen, 4e-7, 256)  # the instrument's view at the recorded pixel size
    assert np.corrcoef(native.ravel(), specimen.image.ravel())[0, 1] > 0.9
    # A zoomed-out view sees vacuum (plus support) beyond the recorded field; BF is the complement of HAADF.
    wide = _render(specimen, 8e-7, 64)
    assert wide[:8, :8].max() < 0.02 and wide[24:40, 24:40].mean() > 0.0
    bf = _render(specimen, 4e-7, 64, detector="BF")
    assert np.corrcoef(bf.ravel(), _render(specimen, 4e-7, 64).ravel())[0, 1] < -0.9
    # Without labels the twin does not know what it is looking at.
    assert specimen.ground_truth() == {"known": False, "particles": []}
    assert specimen.areal_density(0.0, 0.0) == {}


def test_recorded_labels_give_ground_truth_eds_and_damage():
    labelled = build_specimen(
        {"source": "recorded", "path": str(EXAMPLE_IMAGE), "pixel_size_m": EXAMPLE_PIXEL, "labels_path": str(EXAMPLE_LABELS)}, _context()
    )
    truth = labelled.ground_truth()
    particles = json.loads(EXAMPLE_LABELS.read_text(encoding="utf-8"))["particles"]
    assert truth["known"] is True and len(truth["particles"]) == len(particles) > 10
    big = max(particles, key=lambda p: p["semi_axes_m"][0])
    density = labelled.areal_density(*big["center_m"])
    assert set(density) == {big["element"]} and density[big["element"]] > 0
    x, y = big["center_m"]
    index = int(labelled.in_box(x, x, y, y)[0])
    pixel = 1e-9
    fresh = labelled.projected_weight(x - 2e-8, y - 2e-8, 40, pixel, "survey", 1.7, {})
    faded = labelled.projected_weight(x - 2e-8, y - 2e-8, 40, pixel, "survey", 1.7, {index: 0.5})
    assert faded[20, 20] == pytest.approx(0.5 * fresh[20, 20])
    composition = build_specimen({"source": "recorded", "path": str(EXAMPLE_IMAGE), "pixel_size_m": EXAMPLE_PIXEL, "composition": {"Au": 3, "Pd": 1}}, _context())
    bright = composition.areal_density(x, y)
    assert bright["Au"] == pytest.approx(3 * bright["Pd"]) and bright["Au"] > 0


# ----------------------------------------------------------------------
# crystal
# ----------------------------------------------------------------------
def test_crystal_shows_lattice_columns_at_atomic_resolution():
    crystal = build_specimen({"source": "crystal", "grain_count": 4}, _context())
    lattice = crystal.ground_truth()["lattice"]
    assert lattice["element"] == "Au" and lattice["zone_axis"] == [0, 0, 1]
    fov, n = 4e-9, 128
    image = _render(crystal, fov, n, centre=crystal.seeds[0])
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean())))
    peak = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    spacing = fov / np.hypot(peak[0] - n // 2, peak[1] - n // 2)
    # fcc [001]: a square lattice of columns a/2 apart (one FFT pixel is ~2.5 % at this field).
    assert spacing == pytest.approx(lattice["lattice_constant_m"] / 2, rel=0.05)


def test_crystal_grains_show_in_an_overview_and_in_eds():
    crystal = build_specimen({"source": "crystal", "grain_count": 8, "thickness_m": 6e-9}, _context())
    truth = crystal.ground_truth()
    assert len(truth["grains"]) == 8 and truth["particles"] == []
    survey = _render(crystal, 4e-7, 64)
    levels = {round(float(v), 4) for v in survey[8:-8:8, 8:-8:8].ravel()}
    assert len(levels) > 3  # grains differ in thickness and channelling
    g = truth["grains"][0]
    density = crystal.areal_density(*g["seed_m"])
    assert density == {"Au": pytest.approx(0.059 * g["thickness_m"] / 1e-10)}
    assert crystal.areal_density(3e-7, 0.0) == {}  # off the film
    other = build_specimen({"source": "crystal", "grain_count": 8, "seed": 5}, _context())
    assert other.ground_truth()["grains"] != truth["grains"]


# ----------------------------------------------------------------------
# Same controller, different samples
# ----------------------------------------------------------------------
CROSS_PLAN = {
    "box_m": [-1.2e-7, 1.2e-7, -1.2e-7, 1.2e-7],
    "fov_m": 1e-7,
    "overlap": 0.15,
    "imsize": 32,
    "dwell_s": 1e-6,
    "per_tile": ["eds_point"],
    "eds_exposure_s": 0.05,
    "stitch": False,
}
INSTRUMENT_TRUTH = ("moves", "acquisitions", "simulated_time_s", "dose", "focus", "sample_bounds_m")


def _call(proxy, verb, payload):
    return json.loads(getattr(proxy, verb)(json.dumps(payload)))


def _run_map(mapper) -> dict:
    reserved = _call(mapper, "reserve", {"plan": CROSS_PLAN})
    assert "reservation_id" in reserved, reserved
    started = _call(mapper, "start", {"reservation_id": reserved["reservation_id"]})
    deadline = time.time() + 120
    while _call(mapper, "job_status", {"job_id": started["job_id"]})["state"] not in {"done", "failed", "aborted"}:
        assert time.time() < deadline
        time.sleep(0.05)
    results = _call(mapper, "results", {"job_id": started["job_id"]})
    results["status"] = _call(mapper, "job_status", {"job_id": started["job_id"]})
    _call(mapper, "release", {"reservation_id": reserved["reservation_id"]})
    assert results["state"] == "done", json.dumps(results)[:3000]
    with h5py.File(results["result_keys"]["run_file"], "r") as h5:
        return {name: h5[name][()] for name in ("tiles", "spectra", "stage_reported_m", "commanded_xy_m", "acquired_s")}


def _material_under_beam(truth: dict) -> np.ndarray:
    """For each EDS point of the run, whether the specimen had anything under the beam (from its ground truth)."""
    points = [a["beam_true_m"] for a in truth["acquisitions"] if a["kind"] == "spectrum"]
    if truth["specimen"]["source"] == "crystal":  # a continuous film over the whole sample area
        x_min, x_max, y_min, y_max = truth["sample_bounds_m"]
        return np.array([x_min <= x <= x_max and y_min <= y <= y_max for x, y in points])
    labels = Labels(truth["particles"])
    return np.array([any(labels.thickness_at(i, x, y) > 0 for i in labels.in_box(x, x, y, y)) for x, y in points])


def test_the_same_mapper_plan_on_three_specimens(tango_ctx):
    runs, truths = {}, {}
    for source in ("procedural", "crystal", "recorded"):
        twin = tango.DeviceProxy(tango_ctx.get_device_access(f"asyncroscopy/digitaltwin/specimen_{source}"))
        mapper = tango.DeviceProxy(tango_ctx.get_device_access(f"asyncroscopy/mapper/specimen_{source}"))
        mapper.set_timeout_millis(60_000)
        twin.unblank_beam()
        runs[source] = _run_map(mapper)
        truths[source] = json.loads(twin.export_ground_truth())
        assert truths[source]["specimen"]["source"] == source
        assert json.loads(twin.get_parameters())["specimen_source"] == source

    reference = truths["procedural"]
    assert len(reference["moves"]) >= 9 and reference["dose"]["max_dose_e_per_a2"] > 0
    for source in ("crystal", "recorded"):
        # The instrument behaved identically: stage error, backlash, drift, timing, dose and focus.
        for key in INSTRUMENT_TRUTH:
            assert truths[source][key] == reference[key], f"{source}: {key} differs from the procedural run"
        for key in ("stage_reported_m", "commanded_xy_m", "acquired_s"):
            assert np.array_equal(runs[source][key], runs["procedural"][key]), key
    # ...while what it saw did not. A spectrum differs exactly where one of the specimens has material
    # under the beam; where neither has, both see the same support film with the same detector noise.
    material = {source: _material_under_beam(truths[source]) for source in runs}
    assert len(material["crystal"]) == len(runs["crystal"]["spectra"]) and material["crystal"].all()
    for source, run in runs.items():  # shown with pytest -s
        print(f"{source}: {len(truths[source]['moves'])} moves, sim time {truths[source]['simulated_time_s']:.3f} s, "
              f"max dose {truths[source]['dose']['max_dose_e_per_a2']:.4g} e/A^2, tile mean {run['tiles'].mean():.2f} counts, "
              f"spectra on material {int(material[source].sum())}/{len(material[source])}, spectrum counts {run['spectra'].sum():.0f}")
    for a, b in (("procedural", "crystal"), ("procedural", "recorded"), ("crystal", "recorded")):
        assert not np.array_equal(runs[a]["tiles"], runs[b]["tiles"]), (a, b)
        for k, (spectrum_a, spectrum_b) in enumerate(zip(runs[a]["spectra"], runs[b]["spectra"])):
            assert np.array_equal(spectrum_a, spectrum_b) == (not material[a][k] and not material[b][k]), (a, b, k)
    # Ground truth is what each source knows: particles, grains, or labels.
    assert len(truths["procedural"]["particles"]) > 0 and truths["procedural"]["specimen"]["known"] is True
    assert truths["crystal"]["particles"] == [] and len(truths["crystal"]["specimen"]["grains"]) == 12
    assert truths["recorded"]["specimen"]["labelled"] is True and len(truths["recorded"]["particles"]) > 0
