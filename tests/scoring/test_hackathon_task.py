"""The hackathon task's scorer, analysis and focus controllers (no twin needed: the twin run is benchmarks/hackathon itself)."""

import numpy as np
import pytest

from benchmarks.hackathon import analysis, baselines, task
from benchmarks.hackathon.score import score, true_size_ratio, truth_maps

TRUTH = {
    "marker_element": "Pt",
    "regions": [
        {"seed_m": [-2e-6, 0.0], "hole": False, "radius_factor": 1.5, "element_shares": {"Pt": 0.0, "Fe": 1.0}},
        {"seed_m": [2e-6, 2e-6], "hole": False, "radius_factor": 0.5, "element_shares": {"Pt": 1.0, "Fe": 0.0}},
        {"seed_m": [2e-6, -2e-6], "hole": True, "radius_factor": 1.0, "element_shares": {"Pt": 0.5, "Fe": 0.5}},
    ],
}


def perfect() -> task.Answer:
    share, hole = truth_maps(TRUTH)
    return task.Answer("Pt", true_size_ratio(TRUTH), share, hole)


class TestScore:
    def test_the_truth_scores_100(self):
        assert true_size_ratio(TRUTH) == pytest.approx(1 / 3)
        assert score(perfect(), TRUTH)["score"] == pytest.approx(100.0)

    def test_knowing_nothing_about_place_earns_no_map_points(self):
        share, hole = truth_maps(TRUTH)
        answer = task.Answer("Pt", 1.0, np.full_like(share, share[~hole].mean()), np.zeros_like(hole))
        result = score(answer, TRUTH)
        assert result["map_points"] == 0.0 and result["rule_points"] == 0.0 and result["marker_points"] == 30.0

    def test_the_wrong_marker_earns_no_rule_points(self):
        answer = perfect()
        answer.marker_element = "Fe"
        result = score(answer, TRUTH)
        assert result["marker_points"] == 0.0 and result["rule_points"] == 0.0 and result["map_points"] == pytest.approx(40.0)

    def test_a_missed_hole_costs_map_points(self):
        answer = perfect()
        answer.hole_map = np.zeros_like(answer.hole_map)
        assert score(answer, TRUTH)["map_points"] < 40.0

    def test_cells_cover_the_area(self):
        assert task.cell_of(-task.AREA_M / 2, -task.AREA_M / 2) == (0, 0)
        assert task.cell_of(task.AREA_M, task.AREA_M) == (task.GRID - 1, task.GRID - 1)
        assert task.cell_of(task.cell_centres()[3], task.cell_centres()[9]) == (9, 3)


class TestAnalysis:
    def image(self, with_film=True):
        rng = np.random.default_rng(0)
        pixels = rng.poisson(1.2 if with_film else 0.0, (256, 256)) + 5.0 + rng.normal(0, 0.5, (256, 256))
        rows, columns = np.mgrid[:256, :256]
        if with_film:
            for row, column, radius in ((60, 80, 6.0), (180, 150, 3.0)):
                pixels += 40.0 * np.sqrt(np.clip(1 - ((rows - row) ** 2 + (columns - column) ** 2) / radius**2, 0, None))
        return task.Image(pixels, 0.0, 0.0, task.FOV_M / task.IMSIZE)

    def test_particles_are_found_where_they_are_and_film_noise_is_not(self):
        found = sorted(analysis.find_particles(self.image()), key=lambda p: p.row)
        assert [(round(p.row), round(p.column)) for p in found] == [(60, 80), (180, 150)]
        assert found[0].radius_m > 1.5 * found[1].radius_m

    def test_a_hole_is_told_from_film(self):
        assert analysis.is_hole(self.image(with_film=False)) and not analysis.is_hole(self.image())

    def test_an_element_is_named_only_when_its_line_stands_out(self):
        energies = np.arange(2048) * 9.765625
        background = np.full(2048, 0.3)
        gold = background.copy()
        gold[np.abs(energies - analysis.LINE_EV["Au"]) < 60] += 40.0
        elements = ("Au", "Pt", "Fe", "Pd")
        assert analysis.identify(task.Spectrum(energies, gold), elements) == "Au"
        assert analysis.identify(task.Spectrum(energies, background), elements) is None


class FakeMicroscope:
    """A microscope whose needed defocus is a smooth surface, for the focus controllers."""

    def __init__(self):
        self.xy, self.set, self.autofocus_runs = (0.0, 0.0), 0.0, 0

    def needed(self) -> float:
        x, y = self.xy
        return 1e-7 * np.sin(x / 2e-6) + 0.05 * y

    def autofocus(self) -> float:
        self.autofocus_runs += 1
        self.set = self.needed()
        return self.set

    def set_defocus(self, defocus_m: float) -> None:
        self.set = defocus_m


def focus_errors(focus) -> tuple[np.ndarray, int]:
    microscope = FakeMicroscope()
    errors = []
    for x, y in baselines.serpentine():
        microscope.xy = (x, y)
        focus.at(microscope, x, y)
        errors.append(abs(microscope.set - microscope.needed()))
    return np.array(errors), microscope.autofocus_runs


class TestFocusControllers:
    def test_the_serpentine_visits_every_cell_once(self):
        cells = {task.cell_of(x, y) for x, y in baselines.serpentine()}
        assert len(cells) == task.GRID**2 and len(baselines.serpentine()) == task.GRID**2

    def test_focusing_once_drifts_out_of_focus(self):
        errors, runs = focus_errors(baselines.FocusOnce())
        assert runs == 1 and errors.max() > 1e-7

    def test_the_surface_controller_is_sharper_than_periodic_with_fewer_runs(self):
        periodic, periodic_runs = focus_errors(baselines.FocusPeriodically())
        surface, surface_runs = focus_errors(baselines.FocusFromSurface())
        assert surface_runs < periodic_runs and np.median(surface) < np.median(periodic)
