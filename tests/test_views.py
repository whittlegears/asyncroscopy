"""Data views: kinds, server-side slicing, PNG and text output, plug-in kinds."""

from __future__ import annotations

import io
import json
import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from asyncroscopy.data import views
from asyncroscopy.data.kinds import eds
from asyncroscopy.data.views import Axis, ViewOptions, line_kind, register_kind

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class Recorded:
    """An array node that records what was read and computes values per slice (no big array is built)."""

    def __init__(self, shape, metadata=None, dtype="float64", values=None):
        self.shape, self.dtype, self.metadata = tuple(shape), np.dtype(dtype), dict(metadata or {})
        self.values = values
        self.reads: list[tuple] = []

    def read(self, slices=None):
        slices = slices if slices is not None else tuple(slice(0, n) for n in self.shape)
        self.reads.append(slices)
        if self.values is not None:
            return np.asarray(self.values)[slices]
        axes = [np.arange(*s.indices(n)) if isinstance(s, slice) else np.array([s]) for s, n in zip(slices, self.shape)]
        grid = np.add.outer(axes[0], axes[1]) if len(axes) >= 2 else axes[0]
        out = np.broadcast_to(grid.reshape(grid.shape + (1,) * (len(axes) - 2)), tuple(len(a) for a in axes)).astype(self.dtype)
        return out.reshape([len(a) for a, s in zip(axes, slices) if isinstance(s, slice)])


class Container(dict):
    def __init__(self, items=(), metadata=None):
        super().__init__(items)
        self.metadata = dict(metadata or {})


class Table:
    structure_family = "table"
    metadata: dict = {}

    def read(self):
        return pd.DataFrame({"time_s": np.arange(500.0), "current_a": np.linspace(0, 1, 500)})


def gaussian_spectrum(centres_ev, dispersion=10.0, channels=2048):
    energy = np.arange(channels) * dispersion
    counts = 5.0 + sum(1000.0 * np.exp(-0.5 * ((energy - c) / 40.0) ** 2) for c in centres_ev)
    return counts


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == PNG_MAGIC
    return Image.open(io.BytesIO(data)).size


def kind_of(node, key="k.h5"):
    return views.detect(views.pick(node, key)).name


def test_kinds_are_detected_from_attributes_this_repository_writes():
    twin_spectrum = Recorded((2048,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0})
    eels = Recorded((1024,), {"start_energy": -20.0, "dispersion": 0.5, "energy_units": "eV"})
    composition = Recorded((3,), {"elements": json.dumps(["Au", "Pt", "Fe"])}, values=[0.5, 0.3, 0.2])
    image = Recorded((64, 64), {"pixel_size_m": 1e-9})
    stem4d = Recorded((8, 8, 32, 32))
    cube = Recorded((4, 4, 128), {"spectrum_image_axes": "rows=+y, columns=+x, channels"})
    labels = Recorded((16, 16), {"acquisition_type": "segmentation"}, dtype="int32")
    stitched = Container({"level_0": Recorded((2048, 2048)), "level_1": Recorded((1024, 1024)), "level_2": Recorded((512, 512))},
                         {"kind": "stitched_map", "canvas_shape": [2048, 2048], "pixel_size_m": 1e-9, "levels": 3, "pois": []})

    assert kind_of(twin_spectrum) == "spectrum"
    assert kind_of(eels) == "spectrum"
    assert kind_of(composition) == "composition"
    assert kind_of(Recorded((100,))) == "profile"
    assert kind_of(image) == "image"
    assert kind_of(labels) == "label_image"
    assert kind_of(stem4d) == "stack"
    assert kind_of(cube) == "spectrum_image"
    assert kind_of(Table()) == "table"
    assert views.pick(stitched, "map").path == "level_1"
    assert views.summarize(stitched, "map")["kind"] == "map"


def test_a_big_image_is_read_strided_under_the_byte_cap_and_drawn_small():
    big = Recorded((4096, 4096), {"pixel_size_m": 2e-10})
    summary, png = views.view(Container({"image": Container({"HAADF": big})}), "big.h5", "png")

    assert max(png_size(png)) <= views.PREVIEW_MAX_SIDE
    for slices in big.reads:
        rows, cols = (len(range(*s.indices(4096))) for s in slices)
        assert rows * cols * 8 <= views.MAX_READ_BYTES
        assert slices[0].step > 1
    assert summary["read"].startswith("read [0:4096:")


def test_a_hot_pixel_does_not_black_out_the_image():
    values = np.random.default_rng(0).normal(100.0, 10.0, (128, 128))
    values[5, 5] = 1e9
    _summary, png = views.view(Recorded((128, 128), values=values), "hot.h5", "png")
    assert np.asarray(Image.open(io.BytesIO(png))).mean() > 60


def test_spectrum_text_lists_peaks_with_energies_and_xray_candidates():
    node = Recorded((2048,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0, "detector": "eds"}, values=gaussian_spectrum([1486.0, 8040.0]))
    summary, png = views.view(node, "eds.h5", "text")
    text = summary["text"]

    assert png is None
    peaks = text.split("peaks (strongest first")[1].split("data (")[0].strip().splitlines()[2:]
    energies = sorted(float(row.split(",")[0]) for row in peaks)
    assert len(energies) == 2
    assert abs(energies[0] - 1486.0) <= 10.0 and abs(energies[1] - 8040.0) <= 10.0
    assert "Al" in text and "Cu" in text
    assert len(text) <= views.MAX_TEXT_CHARS


def test_a_low_peak_is_named_by_an_element_whose_main_lines_fit_not_by_minor_lines():
    # A carbon support's K peak at about 300 eV was named Ca L and Ag M, though no Ca K or Ag L peak was there.
    attrs = {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0, "detector": "eds"}
    carbon = views.view(Recorded((2048,), attrs, values=gaussian_spectrum([298.0])), "c.h5", "text")[0]["text"]
    assert "consistent with these peaks (the share of the peak signal each one explains): C 100%" in carbon
    assert "Ca " not in carbon and "Ag " not in carbon
    stranded = views.view(Recorded((2048,), attrs, values=gaussian_spectrum([2430.0])), "stranded.h5", "text")[0]["text"]
    assert "unidentified" in stranded and "consistent with these peaks" not in stranded
    calcium = views.view(Recorded((2048,), attrs, values=gaussian_spectrum([302.0, 3692.0])), "ca.h5", "text")[0]["text"]
    assert "no Ca K-L3 peak" not in calcium and "Ca " in calcium.split("explains): ")[1].splitlines()[0]
    iron = views.view(Recorded((2048,), attrs, values=gaussian_spectrum([6404.0, 7058.0])), "fe.h5", "text")[0]["text"]
    assert "Fe K-M3" in iron and "Fe K-M3 (" not in iron


def test_the_detector_windows_continuum_rise_is_not_read_as_oxygen_but_real_lines_are():
    # The window bends the continuum into a broad rise near 0.6 keV, which read as O or F (qwen runs 2026-11-03).
    from asyncroscopy.instruments.electron_microscope.twin.eds_model import EdsParams, point_spectrum

    energies, counts, _live = point_spectrum({"C": 10.0}, 100.0, 3.0, EdsParams(), np.random.default_rng(6))
    axis = Axis(float(energies[0]), float(energies[1] - energies[0]), "Energy", "eV")
    rows = eds.line_peaks(energies, counts, axis)
    named = set().union(*(eds.note_elements(row[4]) for row in rows))
    assert "C" in named and not named & {"O", "F"}
    assert any(row[4] == eds.WINDOW_RISE_NOTE for row in rows)

    # A spectrum whose only peak is the rise says no element was found (qwen once read the data rows as peaks).
    attrs = {"energy_offset_ev": float(energies[0]), "energy_dispersion_ev": float(energies[1] - energies[0]), "detector": "eds"}
    rise_only = counts.copy()
    rise_only[energies < 400.0] = np.random.default_rng(2).poisson(2.0, int((energies < 400.0).sum()))
    assert [row[4] for row in eds.line_peaks(energies, rise_only, axis)] == [eds.WINDOW_RISE_NOTE]
    text = views.view(Recorded(counts.shape, attrs, values=rise_only), "rise.h5", "text")[0]["text"]
    assert "no X-ray line of any element detected" in text and "consistent with these peaks" not in text

    oxygen = counts + np.random.default_rng(1).poisson(900.0 * np.exp(-0.5 * ((energies - 525.0) / (eds.eds_fwhm_ev(525.0) / 2.355)) ** 2) / 5.0)
    rows = eds.line_peaks(energies, oxygen, axis)
    assert "O" in set().union(*(eds.note_elements(row[4]) for row in rows))
    assert not any(row[4] == eds.WINDOW_RISE_NOTE for row in rows)


def test_spectrum_csv_is_binned_by_summing_and_capped():
    counts = np.ones(2048)
    node = Recorded((2048,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0}, values=counts)
    summary, _ = views.view(node, "flat.h5", "text", opts=ViewOptions(max_rows=64))
    rows = summary["text"].split("data (sums of 32 channels):")[1].strip().splitlines()[1:]
    assert len(rows) == 64
    assert all(float(row.split(",")[1]) == 32.0 for row in rows)


def test_regions_crop_images_in_pixels_and_spectra_in_axis_units():
    image = Recorded((256, 256))
    views.view(image, "img.h5", "text", opts=ViewOptions(region=[10, 50, 20, 40]))
    assert image.reads[-1] == (slice(10, 50, 1), slice(20, 40, 1))

    spectrum = Recorded((2048,), {"energy_offset_ev": 100.0, "energy_dispersion_ev": 10.0}, values=gaussian_spectrum([1486.0]))
    summary, png = views.view(spectrum, "eds.h5", "png", opts=ViewOptions(region=[1100.0, 2100.0]))
    assert spectrum.reads[-1] == (slice(100, 201, 1),)
    assert png[:8] == PNG_MAGIC and summary["read"] == "read [100:201]"

    with pytest.raises(ValueError, match="region"):
        views.view(image, "img.h5", "png", opts=ViewOptions(region=[1, 2]))


def test_a_stack_shows_one_frame_and_checks_the_index():
    stem4d = Recorded((8, 6, 32, 32))
    summary, png = views.view(stem4d, "4d.h5", "png")
    assert stem4d.reads[-1][:2] == (4, 3)
    assert png_size(png) == (32, 32)

    views.view(stem4d, "4d.h5", "text", opts=ViewOptions(index=[1, 2]))
    assert stem4d.reads[-1][:2] == (1, 2)
    with pytest.raises(ValueError, match="index"):
        views.view(stem4d, "4d.h5", "png", opts=ViewOptions(index=[9, 0]))


def test_a_container_lists_its_datasets_and_a_named_dataset_can_be_chosen():
    run = Container(
        {
            "tiles": Container({f"tile_{i:05d}": Recorded((128, 128), dtype="uint16") for i in range(12)}),
            "spectra": Container({f"spectrum_{i:05d}": Recorded((2048,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0}) for i in range(12)}),
        }
    )
    summary, _ = views.view(run, "map_run.h5", "text")
    assert "tiles/: 12 x (128, 128) uint16" in summary["text"]
    assert "spectra/: 12 x (2048,) float64" in summary["text"]

    chosen, _ = views.view(run, "map_run.h5", "text", dataset="spectra/spectrum_00003")
    assert chosen["kind"] == "spectrum" and chosen["dataset"] == "spectra/spectrum_00003"
    with pytest.raises(ValueError, match="no dataset"):
        views.view(run, "map_run.h5", "text", dataset="tiles/missing")


def test_composition_labels_are_decoded_and_drawn_as_bars():
    node = Recorded((3,), {"elements": '["Au", "Pt", "Fe"]'}, values=[0.5, 0.3, 0.2])
    summary, _ = views.view(node, "comp.h5", "text")
    assert "Au: 0.5" in summary["text"] and "Fe: 0.2" in summary["text"]
    assert views.view(node, "comp.h5", "png")[1][:8] == PNG_MAGIC
    malformed = Recorded((2,), {"elements": "not json"}, values=[1.0, 2.0])
    assert kind_of(malformed) == "profile"


def test_spectrum_image_and_table_views():
    cube = Recorded((6, 5, 256), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 20.0},
                    values=np.broadcast_to(gaussian_spectrum([1486.0], 20.0, 256), (6, 5, 256)).copy())
    summary, png = views.view(cube, "si.h5", "text")
    assert "6 x 5 positions x 256 channels" in summary["text"]
    assert views.view(cube, "si.h5", "png")[1][:8] == PNG_MAGIC

    table, _ = views.view(Table(), "log", "text", opts=ViewOptions(max_rows=10))
    lines = table["text"].splitlines()
    assert "columns: time_s (float64), current_a (float64)" in lines
    assert lines[-1] == "9,0.0180361" and len(lines) == 3 + 1 + 10


def test_a_plugged_in_kind_takes_precedence_over_profile():
    saved = list(views.KINDS)
    try:
        register_kind(line_kind(
            "diffractogram",
            matches=lambda ds: ds.ndim == 1 and "two_theta_start_deg" in ds.attrs,
            axis=lambda ds: Axis(ds.attrs["two_theta_start_deg"], ds.attrs["two_theta_step_deg"], "2theta", "deg"),
        ), before="profile")
        node = Recorded((500,), {"two_theta_start_deg": 10.0, "two_theta_step_deg": 0.1}, values=gaussian_spectrum([300.0], 1.0, 500))
        summary, _ = views.view(node, "xrd.h5", "text")
        assert summary["kind"] == "diffractogram"
        assert "2theta 10 to 59.9 deg" in summary["text"]
    finally:
        views.KINDS[:] = saved


def test_nodes_that_do_not_report_a_shape_are_read_once():
    class Bare:
        metadata = {"energy_offset_ev": 0.0, "energy_dispersion_ev": 5.0}

        def read(self):
            return np.arange(50.0)

    summary, _ = views.view(Bare(), "bare.h5", "text")
    assert summary["kind"] == "spectrum" and summary["shape"] == [50]


def test_bad_format_is_refused():
    with pytest.raises(ValueError, match="format"):
        views.view(Recorded((4, 4)), "x", "jpeg")


def test_noise_has_no_peaks_and_a_real_peak_on_noise_is_found():
    rng = np.random.default_rng(1)
    attrs = {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0, "detector": "eds"}
    energy = np.arange(2048) * 10.0
    eds_width = eds.eds_fwhm_ev(energy) / 10.0
    false_peaks = 0
    for seed in range(20):
        for mean in (2.0, 200.0):
            noise = np.random.default_rng(seed).poisson(mean, 2048).astype(np.float64)
            false_peaks += len(views.find_peaks(noise, eds_width)) + len(views.find_peaks(noise))
    assert false_peaks == 0
    summary, _ = views.view(Recorded((2048,), attrs, values=rng.poisson(2.0, 2048).astype(np.float64)), "noise.h5", "text")
    assert "no peaks detected above noise at this exposure" in summary["text"] and "longer exposure" in summary["text"]
    signal = rng.poisson(gaussian_spectrum([6404.0])).astype(np.float64)
    peaks = views.find_peaks(signal, eds_width)
    assert len(peaks) == 1 and abs(peaks[0].index * 10.0 - 6404.0) <= 20.0


def test_a_weak_peak_is_found_from_its_summed_counts():
    energy = np.arange(2048) * 10.0
    width = eds.eds_fwhm_ev(energy) / 10.0
    sigma_ch = float(eds.eds_fwhm_ev(8040.0)) / 2.355 / 10.0
    line = np.exp(-0.5 * ((np.arange(2048) - 804) / sigma_ch) ** 2) / (sigma_ch * np.sqrt(2 * np.pi))
    for net_counts, needed in ((80.0, 20), (60.0, 14)):  # on 2 counts per channel; one channel never stands 5 sigma out
        found = 0
        for seed in range(20):
            counts = np.random.default_rng(seed).poisson(2.0 + net_counts * line).astype(np.float64)
            found += any(abs(p.index - 804) <= 3 for p in views.find_peaks(counts, width))
        assert found >= needed, (net_counts, found)


def test_escape_and_sum_peaks_are_flagged_not_named_as_elements():
    spectrum = 5.0 + gaussian_spectrum([6404.0]) + 0.05 * gaussian_spectrum([6404.0 - 1740.0]) + 0.03 * gaussian_spectrum([12808.0])
    node = Recorded((2048,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0, "detector": "eds"}, values=spectrum)
    text = views.view(node, "fe.h5", "text")[0]["text"]
    assert re.search(r'^46\d\d,.*"possible Si escape peak of the 64\d\d eV peak"$', text, re.MULTILINE)
    assert re.search(r'^1\.28\de\+04,.*"possible sum peak \(2 x 64\d\d eV\)"$', text, re.MULTILINE)


def test_png_mode_text_carries_the_summary():
    node = Recorded((2048,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0, "detector": "eds"}, values=gaussian_spectrum([1486.0, 8040.0]))
    summary, png = views.view(node, "eds.h5", "png")
    assert png[:8] == PNG_MAGIC and "peaks (strongest first" in summary["text"] and "Al" in summary["text"]
    image = views.view(Recorded((64, 64), {"pixel_size_m": 1e-9}), "img.h5", "png")[0]["text"]
    assert "statistics of the pixels read" in image and "histogram" in image


def test_a_group_named_as_dataset_lists_its_members_and_shows_the_first():
    run = Container({"tiles": Container({f"tile_{i:05d}": Recorded((16, 16)) for i in range(5)}),
                     "spectra": Container({"spectrum_00000": Recorded((64,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0})})})
    summary, _ = views.view(run, "run.h5", "text", dataset="tiles")
    assert summary["dataset"] == "tiles/tile_00000"
    assert "tiles/ contains:" in summary["text"] and "tiles/tile_00004" in summary["text"]


def test_deep_groups_are_still_listed():
    deep = Container({"a": Container({"b": Container({"c": Container({"d": Container({"e": Container({"f": Recorded((4,))})})})})})})
    assert "a/b/c/d/: 1 groups (deeper levels not listed" in views.container_listing(deep)


def test_a_bare_index_is_accepted_and_a_bad_one_explained():
    stack = Recorded((5, 8, 8))
    views.view(stack, "tilt.h5", "png", opts=ViewOptions(index=3))
    assert stack.reads[-1][0] == 3
    with pytest.raises(ValueError, match="whole numbers"):
        views.view(stack, "tilt.h5", "png", opts=ViewOptions(index=["x"]))


def test_every_group_of_a_large_mapper_run_is_listed():
    detectors = Container({"HAADF": Container({f"tile_{i:05d}": Recorded((128, 128), dtype="uint16") for i in range(400)}),
                           "BF": Container({f"tile_{i:05d}": Recorded((128, 128), dtype="uint16") for i in range(400)})})
    run = Container(
        {
            "tiles": Container({f"tile_{i:05d}": Recorded((128, 128), dtype="uint16") for i in range(400)}),
            "channels": detectors,
            "spectra": Container({f"spectrum_{i:05d}": Recorded((2048,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0}) for i in range(400)}),
            "spectrum_image": Recorded((20, 20, 2048), {"spectrum_image_axes": "rows=+y, columns=+x, channels"}),
            "trace": Recorded((5,), dtype="<U40", values=np.array(["a", "b", "c", "d", "e"])),
        }
    )
    text = views.container_listing(run)
    for line in ("tiles/: 400 x (128, 128) uint16", "channels/: 2 groups", "channels/HAADF/: 400 x (128, 128) uint16",
                 "channels/BF/: 400 x (128, 128) uint16", "spectra/: 400 x (2048,) float64", "spectrum_image: (20, 20, 2048) float64",
                 "trace: (5,) <U40"):
        assert line in text, line
    assert views.pick(run, "run.h5").path == "tiles/tile_00000"


def test_text_bool_and_complex_data_have_text_views():
    strings = Recorded((3,), dtype="<U10", values=np.array(["start", "move", "stop"]))
    summary, _ = views.view(strings, "trace", "text")
    assert summary["kind"] == "text" and "move" in summary["text"]
    with pytest.raises(ValueError, match="use format text"):
        views.view(strings, "trace", "png")

    mask = Recorded((4, 4), dtype="bool", values=np.eye(4, dtype=bool))
    assert "true: 4 of 16" in views.view(mask, "mask", "text")[0]["text"]
    wave = Recorded((8, 8), dtype="complex128", values=np.full((8, 8), 3 + 4j))
    text = views.view(wave, "wave", "text")[0]["text"]
    assert "magnitude" in text and "max 5" in text
    assert views.view(wave, "wave", "png")[1][:8] == PNG_MAGIC


def test_regions_outside_the_data_or_not_numbers_are_refused_clearly():
    image = Recorded((64, 64))
    for region, message in (([100, 120, 0, 10], "outside the data"), ([10, 5, 0, 10], "must be after"), ([float("nan"), 5, 0, 10], "finite")):
        with pytest.raises(ValueError, match=message):
            views.view(image, "img", "png", opts=ViewOptions(region=region))
    spectrum = Recorded((100,), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0})
    with pytest.raises(ValueError, match="outside the data"):
        views.view(spectrum, "s", "text", opts=ViewOptions(region=[5000.0, 6000.0]))


def test_spectrum_image_uses_index_and_region_and_caps_channels():
    cube = Recorded((6, 5, 256), {"energy_offset_ev": 0.0, "energy_dispersion_ev": 20.0},
                    values=np.broadcast_to(gaussian_spectrum([1486.0], 20.0, 256), (6, 5, 256)).copy())
    at, _ = views.view(cube, "si", "text", opts=ViewOptions(index=[2, 3]))
    assert "spectrum at position [2, 3]" in at["text"] and cube.reads[-1] == (2, 3, slice(0, 256))
    views.view(cube, "si", "text", opts=ViewOptions(region=[1, 3, 0, 2]))
    assert cube.reads[-1][:2] == (slice(1, 3, 1), slice(0, 2, 1))
    with pytest.raises(ValueError, match="row, col"):
        views.view(cube, "si", "png", opts=ViewOptions(index=[9, 9]))

    deep = Recorded((2, 2, 10_000_000), {"spectrum_image_axes": "rows=+y, columns=+x, channels"})
    views.view(deep, "deep", "text")
    rows, cols, channels = deep.reads[-1]
    assert len(range(*rows.indices(2))) * len(range(*cols.indices(2))) * len(range(*channels.indices(10_000_000))) * 8 <= views.MAX_READ_BYTES


def test_float_colour_images_are_pictures_not_stacks():
    colour = Recorded((32, 48, 3), values=np.random.default_rng(2).random((32, 48, 3)))
    summary, png = views.view(colour, "rgb", "png")
    assert summary["kind"] == "image"
    assert Image.open(io.BytesIO(png)).mode == "RGB" and png_size(png) == (48, 32)


def test_partitioned_tables_read_only_the_first_partition():
    class Partitioned(Table):
        def read(self):
            raise AssertionError("whole table read")

        def read_partition(self, index):
            return super().read()

    summary, _ = views.view(Partitioned(), "log", "text", opts=ViewOptions(max_rows=5))
    assert "first partition" in summary["text"]


def test_register_kind_refuses_an_unknown_before():
    with pytest.raises(ValueError, match="no such kind"):
        register_kind(line_kind("x", lambda ds: False, lambda ds: None), before="profil")
    assert "x" not in [k.name for k in views.KINDS]


def test_data_taken_with_the_beam_blanked_says_so_instead_of_asking_for_a_longer_exposure():
    attrs = {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0, "detector": "eds"}
    summary, _ = views.view(Recorded((2048,), attrs, values=np.zeros(2048)), "dark.h5", "text")
    assert "0 counts" in summary["text"] and "unblank_beam" in summary["text"] and "longer exposure is needed" not in summary["text"]
    image = Recorded((64, 64), {"pixel_size_m": 1e-9, "beam_current_pa": 0.0}, values=np.random.default_rng(0).poisson(5, (64, 64)).astype(np.uint16))
    summary, _ = views.view(image, "dark_image.h5", "text")
    assert "beam blanked" in summary["text"] and summary["beam_blanked"]
    lit = Recorded((64, 64), {"pixel_size_m": 1e-9, "beam_current_pa": 50.0}, values=np.random.default_rng(0).poisson(5, (64, 64)).astype(np.uint16))
    seen = views.view(lit, "lit.h5", "text")[0]
    assert "beam blanked" not in seen["text"] and "beam_blanked" not in seen


def test_likely_elements_rank_the_elements_that_explain_the_most_peak_signal():
    rows = [(9702, 2127, 34.25, 0, "Au L3-M5, W L2-M4, Ta L3-N5"), (11500, 980, 33.96, 0, "Au L2-M4, Au L3-N5"),
            (2134, 173, 12.83, 0, "Pt M4-N6, Zr L2-M4, Au M5-N7"), (590, 105, 8.15, 0, "Cr L2-M4, Ba M5-N3")]
    assert eds.likely_elements(rows).startswith("Au ")
    assert eds.likely_elements([]) == "" and eds.likely_elements([(500, 10, 5.0, 0, "")]) == ""
    assert eds.likely_elements([(300, 30, 6.0, 0, "Ca L3-M1, Ag L3-M5")]) == "Ag 50%, Ca 50%"
    palladium = [(2838, 90, 9.0, 0, "Pd L3-M5, Pd L3-M4"), (2990, 40, 4.0, 0, "Pd L2-M4"), (3172, 30, 3.0, 0, "Cd L3-M5, Pd L2-N4"),
                 (3290, 20, 2.0, 0, "In L3-M5, Pd L1-M3")]
    assert eds.likely_elements(palladium) == "Pd 100%"
    assert eds.likely_elements(palladium + [(4651, 30, 5.0, 0, "La L3-M5")]) == "Pd 78%, La 22%"
    attrs = {"energy_offset_ev": 0.0, "energy_dispersion_ev": 10.0, "detector": "eds"}
    signal = np.random.default_rng(2).poisson(gaussian_spectrum([6404.0])).astype(np.float64)
    summary, _ = views.view(Recorded((2048,), attrs, values=signal), "fe.h5", "text")
    assert "elements most consistent with these peaks" in summary["text"]
