"""Tests of the calibration procedures server.

The instrument is a small in-process FastMCP server whose images blur with
distance from a known focus (or a known aberration), written to real .npy files
so that the real pyTEMlib analysis tools read them. Every test checks that the
procedure finds the known answer, or that it leaves the instrument where it began.
"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

fastmcp = pytest.importorskip("fastmcp")
analysis_tools = pytest.importorskip("pyTEMlib.mcp_analysis_tools")

from fastmcp import Client, FastMCP  # noqa: E402

from asyncroscopy.mcp.procedures import Limits, ProceduresConfig, ProceduresServer  # noqa: E402


class Scope:
    """Instrument state plus the MCP server that exposes it."""

    def __init__(self, directory: Path, focus_z=0.0, aberration=(0.0, 0.0), fail_after=None, prefix="Scope"):
        self.directory = directory
        self.stage = [0.0, 0.0, 120e-9, 0.0, 0.0]
        self.defocus = 0.0
        self.focus_z = focus_z
        self.aberration = list(aberration)
        self.fail_after = fail_after
        self.acquisitions = 0
        self.moves = []
        self.fov = 96 * 0.5e-10
        texture = np.random.default_rng(0).random((96, 96))
        self.sample = ndimage.gaussian_filter(texture, 0.8)
        self.server = FastMCP("scope")
        tool = self.server.tool

        @tool(name=f"{prefix}_get_stage")
        def get_stage() -> list[float]:
            return list(self.stage)

        @tool(name=f"{prefix}_move_stage")
        def move_stage(position: list[float]) -> None:
            self.moves.append(list(position))
            self.stage = list(position)

        @tool(name=f"{prefix}_get_defocus")
        def get_defocus() -> float:
            return self.defocus

        @tool(name=f"{prefix}_set_defocus")
        def set_defocus(defocus: float) -> None:
            self.defocus = float(defocus)

        @tool(name=f"{prefix}_get_fov")
        def get_fov() -> float:
            return self.fov

        @tool(name="SCAN_get_imsize")
        def get_imsize() -> int:
            return 96

        @tool(name=f"{prefix}Corrector_correct_aberration")
        def correct_aberration(args: str) -> str:
            name, *values = args.split()
            self.aberration = [current - float(value) for current, value in zip(self.aberration, values)]
            return name

        @tool(name=f"{prefix}_acquire_scanned_image")
        def acquire_scanned_image(detector_list: list[str]) -> str:
            if self.fail_after is not None and self.acquisitions >= self.fail_after:
                raise RuntimeError("detector timed out")
            self.acquisitions += 1
            error_nm = np.hypot(abs(self.stage[2] - self.defocus - self.focus_z), np.hypot(*self.aberration)) * 1e9
            path = self.directory / f"image_{self.acquisitions}.npy"
            np.save(path, ndimage.gaussian_filter(self.sample, min(4.0, error_nm / 18.0) + 0.3))
            return str(path)


def analysis_server():
    server = FastMCP("analysis")
    analysis_tools.register(server, {})
    return server


def run(scope, tool, arguments, limits=None):
    procedures = ProceduresServer(ProceduresConfig(limits=limits or Limits()), scope.server, analysis_server())

    async def call():
        async with Client(procedures.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(call())
    text = result.content[0].text
    return result.is_error, (text if result.is_error else json.loads(text))


def test_focus_stage_z_finds_focus_and_stays_there(tmp_path):
    scope = Scope(tmp_path, focus_z=30e-9)
    failed, result = run(scope, "focus_stage_z", {"step_m": 40e-9, "max_offset_m": 400e-9, "metric": "edge_sharpness"})
    assert not failed, result
    assert result["best_z_m"] == pytest.approx(30e-9, abs=6e-9)
    assert scope.stage[2] == pytest.approx(result["best_z_m"])
    assert result["best_score"] > result["baseline_score"]
    assert all(item["data_key"].endswith(".npy") for item in result["history"])


def test_failure_mid_run_moves_the_stage_back(tmp_path):
    scope = Scope(tmp_path, fail_after=4)
    failed, message = run(scope, "focus_stage_z", {"step_m": 40e-9, "max_offset_m": 400e-9})
    assert failed
    assert "detector timed out" in message
    assert scope.stage == [0.0, 0.0, 120e-9, 0.0, 0.0]


def test_requests_beyond_limits_touch_nothing(tmp_path):
    scope = Scope(tmp_path)
    failed, message = run(scope, "focus_stage_z", {"max_offset_m": 5e-6})
    assert failed and "configured limit" in message
    failed, message = run(scope, "focus_stage_z", {"step_m": 1e-6, "max_offset_m": 1e-6})
    assert failed and "step_m" in message
    assert scope.moves == [] and scope.acquisitions == 0


def test_acquisition_limit_stops_and_restores(tmp_path):
    scope = Scope(tmp_path)
    failed, message = run(scope, "focus_stage_z", {"step_m": 40e-9, "max_offset_m": 400e-9},
                          limits=Limits(max_acquisitions=5))
    assert failed and "acquisition limit" in message
    assert scope.acquisitions == 5
    assert scope.stage[2] == pytest.approx(120e-9)


def test_sweep_defocus_finds_focus_between_steps(tmp_path):
    scope = Scope(tmp_path)
    scope.stage[2] = 0.0
    scope.defocus = 33e-9
    failed, result = run(scope, "sweep_defocus", {"half_range_m": 50e-9, "steps": 11, "metric": "edge_sharpness"})
    assert not failed, result
    assert result["fit_has_maximum_in_range"]
    assert scope.defocus == pytest.approx(0.0, abs=3e-9)


def test_sweep_defocus_without_apply_restores_start(tmp_path):
    scope = Scope(tmp_path)
    scope.defocus = 10e-9
    failed, result = run(scope, "sweep_defocus", {"half_range_m": 20e-9, "steps": 5, "apply_best": False})
    assert not failed, result
    assert scope.defocus == pytest.approx(10e-9)


def test_tune_two_fold_aberration(tmp_path):
    scope = Scope(tmp_path, aberration=(6e-9, -4e-9))
    scope.stage[2] = 0.0
    failed, result = run(scope, "tune_aberration", {"name": "A1", "step_m": 2e-9, "max_offset_m": 15e-9,
                                                    "metric": "edge_sharpness", "max_iterations": 12})
    assert not failed, result
    assert np.hypot(*scope.aberration) < 2e-9
    np.testing.assert_allclose(result["applied_correction_m"], [6e-9, -4e-9], atol=2e-9)


def test_tune_aberration_undoes_everything_on_failure(tmp_path):
    scope = Scope(tmp_path, aberration=(6e-9, -4e-9), fail_after=6)
    failed, message = run(scope, "tune_aberration", {"name": "A1", "step_m": 2e-9, "max_offset_m": 15e-9})
    assert failed
    np.testing.assert_allclose(scope.aberration, [6e-9, -4e-9], atol=1e-18)


def test_ambiguous_instrument_tools_are_reported(tmp_path):
    scope = Scope(tmp_path)

    @scope.server.tool(name="OtherScope_get_stage")
    def other_get_stage() -> list[float]:
        return [0.0] * 5

    failed, message = run(scope, "focus_stage_z", {})
    assert failed
    assert "tool_names.get_stage" in message


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "procedures.yaml"
    path.write_text("procedures:\n  instrument_url: http://x\n  limts: {}\n")
    with pytest.raises(ValueError, match="limts"):
        ProceduresConfig.from_yaml(path)


def test_flat_response_reports_no_improvement_and_restores(tmp_path):
    scope = Scope(tmp_path, focus_z=2e-6)
    failed, result = run(scope, "focus_stage_z", {"step_m": 40e-9, "max_offset_m": 400e-9})
    assert not failed, result
    assert result["improved"] is False
    assert "no improvement" in result["outcome"]
    assert scope.stage[2] == pytest.approx(120e-9)


def test_bracketed_peak_is_reported(tmp_path):
    scope = Scope(tmp_path, focus_z=30e-9)
    failed, result = run(scope, "focus_stage_z", {"step_m": 40e-9, "max_offset_m": 400e-9, "metric": "edge_sharpness"})
    assert result["improved"] and result["peak_bracketed"]


def test_tune_aberration_refuses_coarse_pixels(tmp_path):
    scope = Scope(tmp_path, aberration=(6e-9, -4e-9))
    scope.fov = 96 * 1.5e-10
    failed, message = run(scope, "tune_aberration", {"name": "A1"})
    assert failed and "coarser than" in message
    assert scope.acquisitions == 0 and scope.aberration == [6e-9, -4e-9]


def test_tune_aberration_refuses_contrast(tmp_path):
    scope = Scope(tmp_path)
    failed, message = run(scope, "tune_aberration", {"name": "A1", "metric": "contrast"})
    assert failed and "contrast" in message
    assert scope.acquisitions == 0
