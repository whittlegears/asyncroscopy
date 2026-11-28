"""The instrument profile and the trace logger that fills it."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from asyncroscopy.calibration import profile as instrument_profile
from asyncroscopy.calibration import trace_logger

ROOT = Path(__file__).resolve().parents[2]
SPECTRA = ROOT / "configs" / "profiles" / "spectra300.yaml"


def test_the_shipped_profile_names_only_real_twin_properties():
    from asyncroscopy.instruments.electron_microscope.digital_twin import DigitalTwin
    from asyncroscopy.instruments.electron_microscope.hardware.stage_digital_twin import DigitalTwinSTAGE

    properties = instrument_profile.twin_properties(instrument_profile.load(SPECTRA))
    assert properties["instrument"] and properties["stage"]
    assert all(hasattr(DigitalTwin, name) for name in properties["instrument"])
    assert all(hasattr(DigitalTwinSTAGE, name) for name in properties["stage"])


def test_values_draws_and_ranges():
    profile = instrument_profile.load(SPECTRA)
    entry = profile["parameters"]["stage_move_noise_std"]
    assert instrument_profile.twin_properties(profile)["stage"]["stage_move_noise_std"] == entry["value"]
    assert instrument_profile.twin_properties(profile, np.random.default_rng(1))["stage"]["stage_move_noise_std"] == entry["value"]
    across = [instrument_profile.twin_properties(profile, np.random.default_rng(seed), across=True)["stage"]["stage_move_noise_std"] for seed in range(20)]
    assert all(entry["range"][0] <= v <= entry["range"][1] for v in across) and len(set(across)) == 20


def test_record_stamps_the_source_and_keeps_the_rest(tmp_path):
    path = tmp_path / "scope.yaml"
    path.write_text(SPECTRA.read_text(encoding="utf-8"), encoding="utf-8")
    assert "stage_backlash_x" in instrument_profile.unmeasured(instrument_profile.load(path))
    instrument_profile.record(path, {"stage_backlash_x": 9e-8}, source="stage_repeatability", sigma={"stage_backlash_x": 1e-8})
    entry = instrument_profile.load(path)["parameters"]["stage_backlash_x"]
    assert (entry["value"], entry["sigma"], entry["source"], entry["device"]) == (9e-8, 1e-8, "stage_repeatability", "stage")
    assert entry["measured"] and entry["range"] == [2.0e-8, 1.0e-6]
    assert "stage_backlash_x" not in instrument_profile.unmeasured(instrument_profile.load(path))


def test_a_bad_profile_is_refused_by_name(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("parameters:\n  stage_speed: {value: fast}\n  fov_scale_error: {value: 0.1, device: lens}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stage_speed.*fov_scale_error"):
        instrument_profile.load(path)


def test_the_stack_config_extends_its_base_and_loads(tmp_path):
    import importlib.util
    import sys

    out = ROOT / "configs" / "local" / "_profile_test.yaml"
    config = instrument_profile.stack_config(instrument_profile.load(SPECTRA), ROOT / "configs" / "digital_twin_mapping.yaml", out)
    assert config["extends"] == "../digital_twin_mapping.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(config), encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("run_servers_for_profile_test", ROOT / "startup_scripts" / "run_servers.py")
        run_servers = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = run_servers  # its dataclasses look their module up by name
        spec.loader.exec_module(run_servers)
        merged = run_servers.read_config(out)
    finally:
        out.unlink()
    assert merged["instrument"]["properties"]["focus_drift_m"] == 3.0e-8
    assert merged["devices"]["stage"]["properties"]["stage_backlash_y"] == 2.5e-7
    assert merged["devices"]["stage"]["class_name"] == "DigitalTwinSTAGE"  # the base's device is kept


def test_the_logger_only_reads_and_its_summary_fills_a_profile(mapping_twin_proxy, tmp_path):
    before = list(mapping_twin_proxy.get_stage())
    log_path = tmp_path / "state.jsonl"
    trace_logger.log(mapping_twin_proxy, log_path, interval_s=0.0, readings=2)
    mapping_twin_proxy.set_defocus(4e-8)  # the operator refocuses between readings
    trace_logger.log(mapping_twin_proxy, log_path, interval_s=0.0, readings=1)
    assert list(mapping_twin_proxy.get_stage()) == before
    summary = trace_logger.summarise(log_path)
    assert summary["readings"]["defocus_m"]["n"] == 3 and summary["readings"]["stage[0]"]["swing"] == 0.0
    values, _sigmas = trace_logger.twin_parameters(summary)
    assert values["focus_drift_m"] == pytest.approx(2e-8) and values["probe_current_pa"] > 0
    profile = instrument_profile.record(tmp_path / "scope.yaml", values, source="trace_logger")
    assert profile["parameters"]["focus_drift_m"]["source"] == "trace_logger"
    mapping_twin_proxy.set_defocus(0.0)
