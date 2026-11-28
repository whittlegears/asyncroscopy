"""End to end: twin acquisitions scored through the ANALYSIS Tango device."""

import json

import pytest
import tango


def _acquire(twin_proxy) -> str:
    return twin_proxy.acquire_scanned_image(["haadf"])


@pytest.fixture
def twin_at_origin(twin_proxy, scan_proxy):
    scan_proxy.imsize = 128
    scan_proxy.dwell_time = 1e-6
    twin_proxy.set_fov(100e-9)
    twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])
    yield twin_proxy
    twin_proxy.move_stage([0.0, 0.0, 0.0, 0.0, 0.0])


def test_targets_come_from_the_device_config(analysis_proxy):
    targets = json.loads(analysis_proxy.get_targets())

    assert targets["convergence_semi_angle_mrad"] == 25.0
    assert targets["expected_spacings_m"] == {"coarse": 2e-9}
    assert targets["target_resolution_m"] is None


def test_drift_between_two_twin_frames(analysis_proxy, twin_at_origin):
    first = _acquire(twin_at_origin)
    twin_at_origin.move_stage([5e-9, 0.0, 0.0, 0.0, 0.0])
    second = _acquire(twin_at_origin)

    result = json.loads(analysis_proxy.drift(json.dumps([first, second])))

    assert result["shift_m"] == pytest.approx(5e-9, rel=0.05)
    assert result["confidence"] > 0.5
    assert result["rate_m_per_s"] > 0


def test_drift_needs_two_keys(analysis_proxy):
    with pytest.raises(tango.DevFailed, match="two keys"):
        analysis_proxy.drift(json.dumps(["only-one.h5"]))


def test_resolution_of_a_twin_image(analysis_proxy, twin_at_origin):
    result = json.loads(analysis_proxy.resolution(_acquire(twin_at_origin)))

    # 100 nm over 128 px: pixel 7.8 A, Nyquist 15.6 A. Particles are about 3 nm
    # across, so their edges show detail well above Nyquist.
    assert result["nyquist_angstrom"] == pytest.approx(2 * 100e-9 / 128 * 1e10, rel=1e-3)
    assert result["resolution_angstrom"] > result["nyquist_angstrom"]
    # The particle sample has no lattice, so a 2 nm spacing is reported but not found.
    assert result["resolved_spacings"] == {"coarse": False}


def test_correction_scores_a_standard_tableau_and_refuses_a_fast_one(analysis_proxy, corrector_proxy):
    score = json.loads(analysis_proxy.correction(corrector_proxy.acquire_tableau("Standard 18")))

    assert score["aberration_free_angle_mrad"] > 0
    assert score["target_mrad"] == 25.0
    with pytest.raises(tango.DevFailed, match="Standard or Enhanced"):
        analysis_proxy.correction(corrector_proxy.acquire_tableau("Fast 9"))

def test_agent_tools_include_the_scores_but_not_the_twin_answers(tango_ctx, monkeypatch):
    """Run the MCP server's real tool discovery on the live test devices with the shipped blocklist."""
    from pathlib import Path

    import yaml

    from asyncroscopy.mcp.mcp_server import MCPServer

    monkeypatch.setattr("asyncroscopy.mcp.mcp_server.Database", lambda host, port: None)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs" / "mcp_dt.yaml").read_text())["mcp"]
    server = MCPServer(
        "test",
        "localhost",
        1234,
        data_device_address="asyncroscopy/data/default",
        blocked_functions=config["blocked_functions"],
        blocked_classes=config["blocked_classes"],
        verbose=False,
    )
    names = ["asyncroscopy/analysis/default", "asyncroscopy/digitaltwin/default", "asyncroscopy/corrector/default"]
    monkeypatch.setattr(server, "_list_all_devices", lambda: [tango_ctx.get_device_access(n) for n in names])

    tools = server._find_tools()

    assert {"drift", "resolution", "correction", "get_targets"} <= set(tools["ANALYSIS"])
    assert "get_ground_truth" not in tools["DigitalTwin"]
    assert "acquire_scanned_image" in tools["DigitalTwin"]
    assert "get_correction_limits" in tools["DigitalTwinCorrector"]
    assert not {"get_aberrations_coeff_sim", "set_aberrations_coeff_sim"} & set(tools["DigitalTwinCorrector"])
