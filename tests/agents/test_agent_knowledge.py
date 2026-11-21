"""Tests for what an agent is told without asking a person.

They cover the energy axis of an EDS spectrum, tool descriptions filled in from the local source, and the skills
chosen for mapping and spectrum requests.
"""

from __future__ import annotations

import asyncio

import numpy as np

from asyncroscopy.data import views
from asyncroscopy.mcp import local_docs
from asyncroscopy.skills import list_skills, load_skill, select_skills

from tests.analysis.test_views import Recorded
from tests.agents.test_workflows import INPUTS, twin

# The attributes AutoScript writes into an EDS file, as text.
SUPER_X = {"Dispersion": "5", "OffsetEnergy": "-250", "detector": "eds", "acquisition_type": "spectrum"}


def super_x_spectrum(lines_ev: list[float]) -> np.ndarray:
    energy = -250.0 + 5.0 * np.arange(4096)
    counts = 2.0 + 4000.0 * np.exp(-0.5 * (energy / 12.0) ** 2)  # the zero-energy peak every spectrum has
    for line in lines_ev:
        counts = counts + 1500.0 * np.exp(-0.5 * ((energy - line) / 55.0) ** 2)
    return np.random.default_rng(3).poisson(counts).astype(np.uint32)


def test_an_eds_spectrum_saved_by_the_microscope_is_read_in_ev_with_its_elements():
    assert views.energy_axis(SUPER_X) == views.Axis(-250.0, 5.0, "energy", "eV")
    assert views.energy_axis({"Dispersion": "0", "OffsetEnergy": "-250"}) is None
    node = Recorded((4096,), SUPER_X, values=super_x_spectrum([1487.0, 4840.0, 7478.0, 8048.0]))
    text = views.view(node, "spectrum_eds.h5", "text")[0]["text"]
    assert "energy -250 to" in text and "channel 0 to" not in text
    rows = text.split("peaks (strongest first")[1].split("data (")[0].strip().splitlines()[2:]
    found = {float(row.split(",")[0]): row for row in rows}
    for line, element in ((1487.0, "Al K"), (4840.0, "Ce L"), (7478.0, "Ni K"), (8048.0, "Cu K")):
        nearest = min(found, key=lambda e: abs(e - line))
        assert abs(nearest - line) <= 15.0 and element in found[nearest], (line, found)
    # The zero energy peak is detector noise and must not be counted as an element.
    zero = found[min(found, key=abs)]
    assert "zero-energy noise peak" in zero and "not an element" in zero
    likely = text.split("explains): ")[1].splitlines()[0]
    assert {"Cu", "Ce"} <= {part.split()[0] for part in likely.split(", ")} and "the" not in likely.split()


def test_a_command_the_device_does_not_describe_is_described_from_this_installations_source():
    summary, argument = local_docs.describe("AutoScriptMicroscope", "place_beam")  # declared on a base class
    assert "fractions of the image from 0 to 1" in summary and "not metres" in summary
    assert argument.startswith(":param position:") and "[0.5, 0.5] is the centre" in argument
    assert "x/y/z in meters" in local_docs.describe("DigitalTwin", "move_stage")[0]
    assert "seconds" in local_docs.describe("EDS", "set_eds_exposure_time")[0]
    assert local_docs.describe("NoSuchDevice", "place_beam") == ("", "") == local_docs.describe("AutoScriptMicroscope", "no_such_command")


def test_a_map_is_asked_for_by_its_tile_count():
    from asyncroscopy.agents.workflows.runner import run_workflow

    instrument, analysis, state = twin()
    record = asyncio.run(run_workflow("map_grid", {"tiles_per_side": 3, "fov_m": 1.0e-7}, instrument=instrument, analysis=analysis, durable=False))
    assert record["state"] == "done" and state["taken"] == 9, record
    assert record["outputs"]["plan"]["rows"] == 3 and record["outputs"]["plan"]["cols"] == 3
    by_size = asyncio.run(run_workflow("map_grid", INPUTS, instrument=twin()[0], analysis=twin()[1], durable=False))
    assert by_size["outputs"]["plan"]["rows"] == 2  # without a tile count the size decides


CHAT = {"AutoScriptMicroscope_place_beam", "AutoScriptMicroscope_acquire_spectrum", "AutoScriptMicroscope_acquire_scanned_image",
        "AutoScriptMicroscope_get_fov", "view_data", "list_data", "find_tools", "call_tool", "run_workflow", "workflow_status",
        "answer_workflow", "stop_workflow", "resume_workflow", "map_info", "get_region"}


def test_requests_for_maps_and_spectra_select_the_matching_skills():
    available = {s.name for s in list_skills(CHAT) if s.available}
    assert {"map_with_workflow", "spectra_in_image"} <= available and "mapping" not in available  # no Mapper device there

    def chosen(request: str) -> list[str]:
        return [s.name for s in select_skills(request, CHAT)]

    assert chosen("acquire a 4 by 4 grid and create a map")[0] == "map_with_workflow"
    assert chosen("acquire another image and then take an eds spectra at the interface")[0] == "spectra_in_image"
    assert "spectra_in_image" in chosen("I want spectra from the different areas of the bright phase")
    mapping, spectra = load_skill("map_with_workflow"), load_skill("spectra_in_image")
    assert '"tiles_per_side": 4' in mapping and "Never loop over" in mapping
    assert "fractions of the current image" in spectra and "-250 + 5 * n" in spectra and "zero-energy noise peak" in spectra
    # The mapping skill needs the workflow tools, so it is not offered without them.
    assert "map_with_workflow" not in {s.name for s in list_skills(CHAT - {"run_workflow"}) if s.available}
