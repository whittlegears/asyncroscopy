"""Tests of the ML suggestions server and of the procedures that use it.

Every test has a known answer: a known optimum, a known Pareto set, or an image whose
measured value really does depend on local structure.
"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

pytest.importorskip("torch")
pytest.importorskip("gpytorch")
pytest.importorskip("botorch")
analysis_tools = pytest.importorskip("pyTEMlib.mcp_analysis_tools")

from fastmcp import Client  # noqa: E402

from asyncroscopy.mcp.ml_server import (MLConfig, MLServer, SuggestionError, suggest_locations,  # noqa: E402
                                        suggest_pareto_setting, suggest_setting)
from asyncroscopy.mcp.procedures import Limits, ProceduresConfig, ProceduresServer  # noqa: E402
from tests.test_procedures import Scope, analysis_server  # noqa: E402


def test_bayesian_optimisation_finds_a_two_parameter_optimum():
    target = np.array([3.0, -2.0])

    def score(point):
        return float(np.exp(-np.sum((np.asarray(point) - target) ** 2) / 8))

    points = [[0, 0], [2, 0], [0, 2], [-2, 0], [0, -2]]
    scores = [score(point) for point in points]
    for _ in range(12):
        suggestion = suggest_setting(points, scores, [[-10, 10], [-10, 10]])
        assert all(-10 <= value <= 10 for value in suggestion["next_point"])
        points.append(suggestion["next_point"])
        scores.append(score(suggestion["next_point"]))
    np.testing.assert_allclose(points[int(np.argmax(scores))], target, atol=0.3)


def test_minimisation_and_bounds():
    points = [[1.0], [4.0], [7.0]]
    suggestion = suggest_setting(points, [(p[0] - 5) ** 2 for p in points], [[0, 6]], maximize=False)
    assert 0 <= suggestion["next_point"][0] <= 6
    assert suggestion["best_measured_point"] == [4.0]


def test_pareto_set_and_suggestion():
    points = [[0.0], [1.0], [2.0], [3.0], [1.5]]
    objectives = [[0, 3], [1, 2], [2, 1], [3, 0], [0.5, 0.5]]
    result = suggest_pareto_setting(points, objectives, [[0, 3]])
    assert result["pareto_indices"] == [0, 1, 2, 3]
    assert 0 <= result["next_point"][0] <= 3


def test_bad_inputs_are_refused():
    with pytest.raises(SuggestionError):
        suggest_setting([[0.0]], [1.0], [[0, 1]])
    with pytest.raises(SuggestionError):
        suggest_setting([[0.0], [1.0]], [1.0, 2.0], [[1, 0]])
    with pytest.raises(SuggestionError):
        suggest_locations(np.zeros((32, 32)), [[0, 0, 1.0], [1, 1, 2.0]], window_size=8)


def _blob_image():
    rng = np.random.default_rng(0)
    centres = np.zeros((128, 128))
    for row, column in rng.integers(0, 128, (60, 2)):
        centres[row, column] = 1
    density = ndimage.gaussian_filter(centres, 6)
    image = ndimage.gaussian_filter(centres, 1.2) + 0.02 * rng.random((128, 128))
    return image, density, rng


def test_deep_kernel_learns_a_structure_dependent_value():
    image, density, rng = _blob_image()
    measurements = [[int(r), int(c), float(density[r, c])] for r, c in rng.integers(0, 128, (25, 2))]
    result = suggest_locations(image, measurements, window_size=16, step=4, count=5, acquisition="ei")
    rows, columns = np.meshgrid(np.arange(0, 128, 4), np.arange(0, 128, 4), indexing="ij")
    truth = density[rows, columns]
    assert np.corrcoef(result["mean_map"].ravel(), truth.ravel())[0, 1] > 0.75
    picked = [density[row, column] for row, column in result["next_positions"]]
    assert np.mean(picked) > np.percentile(truth, 75)
    chosen = np.asarray(result["next_positions"], dtype=float)
    spacing = np.hypot(*(chosen[:, None, :] - chosen[None, :, :]).transpose(2, 0, 1))
    assert spacing[np.triu_indices(len(chosen), 1)].min() >= 16
    measured = np.asarray([[row, column] for row, column, _ in measurements], dtype=float)
    assert np.hypot(*(chosen[:, None, :] - measured[None, :, :]).transpose(2, 0, 1)).min() >= 16


def test_ml_tools_over_mcp_read_a_file_and_write_maps(tmp_path):
    image, density, rng = _blob_image()
    np.save(tmp_path / "haadf.npy", image)
    server = MLServer(MLConfig(output_dir=str(tmp_path / "out")))
    measurements = [[int(r), int(c), float(density[r, c])] for r, c in rng.integers(0, 128, (12, 2))]

    async def call():
        async with Client(server.mcp) as client:
            names = {tool.name for tool in await client.list_tools()}
            result = await client.call_tool("dkl_suggest_locations", {
                "source": str(tmp_path / "haadf.npy"), "measurements": measurements, "count": 2, "epochs": 60})
            return names, json.loads(result.content[0].text)

    names, result = asyncio.run(call())
    assert names == {"bo_suggest_next_setting", "bo_suggest_pareto_setting", "dkl_suggest_locations", "ml_status"}
    assert len(result["next_positions"]) == 2
    assert np.load(result["mean_map_file"]).shape == (32, 32)


def _run_procedure(scope, tool, arguments, ml=True):
    ml_server = MLServer(MLConfig()).mcp if ml else None
    procedures = ProceduresServer(ProceduresConfig(limits=Limits()), scope.server, analysis_server(), ml_server)

    async def call():
        async with Client(procedures.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(call())
    text = result.content[0].text
    return result.is_error, (text if result.is_error else json.loads(text))


def test_bayesian_focus_through_the_procedures_server(tmp_path):
    scope = Scope(Path(tmp_path), focus_z=30e-9)
    failed, result = _run_procedure(scope, "focus_stage_z", {"step_m": 40e-9, "max_offset_m": 400e-9,
                                                             "optimizer": "bayesian", "max_iterations": 10})
    assert not failed, result
    assert result["best_z_m"] == pytest.approx(30e-9, abs=8e-9)
    assert all(abs(item["point"][0] - 120e-9) <= 400e-9 + 1e-15 for item in result["history"])
    assert any(item["trial"].startswith("bayesian") for item in result["history"])


def test_bayesian_aberration_tuning_restores_on_failure(tmp_path):
    scope = Scope(Path(tmp_path), aberration=(6e-9, -4e-9), fail_after=7)
    failed, message = _run_procedure(scope, "tune_aberration", {"name": "A1", "step_m": 2e-9, "max_offset_m": 15e-9,
                                                                "optimizer": "bayesian"})
    assert failed and "detector timed out" in message
    np.testing.assert_allclose(scope.aberration, [6e-9, -4e-9], atol=1e-18)


def test_bayesian_without_ml_server_is_refused_before_moving(tmp_path):
    scope = Scope(Path(tmp_path))
    failed, message = _run_procedure(scope, "focus_stage_z", {"optimizer": "bayesian"}, ml=False)
    assert failed and "ml_url" in message
    assert scope.stage[2] == pytest.approx(120e-9)


def test_unknown_optimizer_is_refused(tmp_path):
    scope = Scope(Path(tmp_path))
    failed, message = _run_procedure(scope, "focus_stage_z", {"optimizer": "genetic"}, ml=False)
    assert failed and "optimizer" in message and scope.moves == []
