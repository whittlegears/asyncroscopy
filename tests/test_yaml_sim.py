"""Instruments described in YAML (asyncroscopy/instruments/yaml_sim): the format's checks, the simulator, and the Tango device."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from asyncroscopy.instruments.yaml_sim.spec import (
    Refused,
    SimInstrument,
    SpecError,
    load_spec,
)
from asyncroscopy.safety.policy import risk_of

HERE = Path(__file__).parent
EXAMPLE = HERE.parent / "asyncroscopy" / "instruments" / "yaml_sim" / "examples" / "xrd.yaml"


def write(tmp_path, text: str) -> Path:
    path = tmp_path / "instrument.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def xrd() -> SimInstrument:
    return SimInstrument(load_spec(EXAMPLE), sleep=lambda seconds: None)


@pytest.mark.parametrize("text, message", [
    ("settings: {power_w: {default: 1, maximum: 5}}", "unknown key(s) ['maximum']"),
    ("settings: {power_w: {default: 1, min: 5, max: 2}}", "min 5.0 is above max 2.0"),
    ("settings: {power_w: {default: 9, max: 5}}", "settings.power_w.default"),
    ("settings: {power_w: {min: 0}}", "needs a default"),
    ("settings: {Power: {default: 1}}", "not a valid name"),
    ("settings: {power_w: {default: 1}}\ncommands: {go: {doc: Go., sets: {power_w: \"voltage * 2\"}}}", "names 'voltage'"),
    ("settings: {power_w: {default: 1}}\ncommands: {go: {doc: Go., sets: {power_w: \"__import__('os')\"}}}", "calls something other than"),
    ("settings: {power_w: {default: 1}}\ncommands: {get_power_w: {doc: Read.}}", "defined twice"),
    ("settings: {power_w: {default: 1}}\ncommands: {get_now: {doc: Read., sets: {power_w: 0}}}", "must not change anything"),
    ("settings: {power_w: {default: 1}}\ncommands: {go: {sets: {power_w: 0}}}", "needs a doc"),
    ("settings: {power_w: {default: 1}}\nacquisitions: {scan: {doc: Scan., kind: image}}", "start with 'acquire'"),
    ("settings: {power_w: {default: 1}}\nacquisitions: {acquire_x: {doc: X., kind: movie}}", "kind: 'movie'"),
    ("settings: {power_w: {default: 1}}\nacquisitions: {acquire_x: {doc: X., kind: spectrum, axis: {start: 0, stop: 1, points: 1}}}", "axis.points"),
    ("settings: {power_w: {default: 1}}\ncommands: {go: {doc: Go., duration_s: 600}}", "duration_s"),
    ("{}", "nothing for an agent to call"),
    ("setting: {}", "unknown key(s) ['setting']"),
])
def test_a_bad_file_is_refused_at_load_naming_the_place(tmp_path, text, message):
    with pytest.raises(SpecError) as raised:
        load_spec(write(tmp_path, text))
    assert message in str(raised.value)


def test_the_example_loads_and_lists_its_commands():
    spec = load_spec(EXAMPLE)
    names = spec.command_names()
    assert {"get_tube_power_w", "set_start_deg", "xray_on", "xray_off", "get_status", "acquire_diffractogram"} <= set(names)
    assert "set_tube_power_w" not in names  # settable: false
    assert spec.settings["step_count"].type is int


def test_settings_are_range_checked_and_typed():
    sim = xrd()
    sim.set("step_count", "301")
    assert sim.get("step_count") == 301
    with pytest.raises(Refused, match=r"start_deg = 1 deg is outside \[5, 140\]"):
        sim.set("start_deg", 1)
    with pytest.raises(Refused, match="must be a int"):
        sim.set("step_count", "many")
    with pytest.raises(Refused, match="read-only"):
        sim.set("tube_power_w", 100)


def test_commands_check_requirements_set_settings_and_return_json():
    sim = xrd()
    with pytest.raises(Refused, match=r"acquire_diffractogram refused: it needs tube_power_w > 0"):
        sim.acquire("acquire_diffractogram")
    with pytest.raises(Refused, match="power_w = 5000 W is outside"):
        sim.run_command("xray_on", 5000)
    assert json.loads(sim.run_command("xray_on", 800)) == {"tube_power_w": 800.0, "state": "on"}
    status = json.loads(sim.run_command("get_status"))
    assert status["scan_deg"] == [20.0, 80.0] and status["seconds_per_scan"] == pytest.approx(1201 * 0.002)
    sim.run_command("xray_off")
    assert sim.get("tube_power_w") == 0


def test_a_refused_command_changes_nothing(tmp_path):
    sim = SimInstrument(load_spec(write(tmp_path, """
settings:
  a: {default: 1, max: 10}
  b: {default: 1, max: 10}
commands:
  both:
    doc: Set a and b.
    sets: {a: 5, b: 50}
""")))
    with pytest.raises(Refused, match="b = 50"):
        sim.run_command("both")
    assert sim.values == {"a": 1.0, "b": 1.0}


def test_a_scan_longer_than_the_simulator_allows_is_refused():
    sim = xrd()
    sim.run_command("xray_on", 800)
    sim.set("step_count", 20001)
    sim.set("count_time_s", 0.01)
    with pytest.raises(Refused, match="would take 200 s"):
        sim.acquire("acquire_diffractogram")


def test_a_diffractogram_has_its_axis_and_peaks_where_the_file_says():
    sim = xrd()
    sim.run_command("xray_on", 1600)
    kind, detector, data, attrs = sim.acquire("acquire_diffractogram")
    assert (kind, detector, data.shape) == ("diffractogram", "point_detector", (1201,))
    assert attrs["axis_offset"] == 20 and attrs["axis_scale"] == pytest.approx(0.05)
    assert (attrs["axis_label"], attrs["axis_units"]) == ("2-theta", "deg")
    two_theta = attrs["axis_offset"] + attrs["axis_scale"] * np.arange(data.size)
    assert abs(two_theta[np.argmax(data)] - 28.44) < 0.15  # the strongest line, Si 111


def test_an_image_and_a_value_acquisition():
    sim = xrd()
    _, _, image, attrs = sim.acquire("acquire_camera_image")
    assert image.shape == (240, 320) and attrs["pixel_size_m"] == pytest.approx(2e-5) and image.max() > 100
    _, _, data, values = sim.acquire("acquire_tube_readings")
    assert data is None and values == {"voltage_kv": 0.0, "current_ma": 0.0, "anode_temperature_c": 25.0}


def test_the_safety_policy_classes_the_generated_commands():
    assert risk_of("XRDTwin", "get_start_deg") == "low"
    assert risk_of("XRDTwin", "set_start_deg") == "high"
    assert risk_of("XRDTwin", "acquire_diffractogram") == "high"
    assert risk_of("XRDTwin", "xray_off") == "low"  # a safe stop is never held


def test_the_check_command_validates_a_file(tmp_path):
    good = subprocess.run([sys.executable, "-m", "asyncroscopy.instruments.yaml_sim.spec", str(EXAMPLE)], capture_output=True, text=True, timeout=60, check=False)
    assert good.returncode == 0 and "acquire_diffractogram" in good.stdout
    bad = subprocess.run([sys.executable, "-m", "asyncroscopy.instruments.yaml_sim.spec", str(write(tmp_path, "settings: {x: {min: 0}}"))], capture_output=True, text=True, timeout=60, check=False)
    assert bad.returncode != 0 and "needs a default" in bad.stderr


def test_the_example_runs_as_a_tango_device_behind_the_mcp_server(tmp_path):
    # A fresh interpreter: PyTango allows one in-process test context each, and this suite has one.
    done = subprocess.run([sys.executable, str(HERE / "yaml_sim_check.py"), str(tmp_path)], capture_output=True, text=True, timeout=180, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert {"XRDTwin_get_start_deg", "XRDTwin_set_start_deg", "XRDTwin_xray_off", "XRDTwin_acquire_diffractogram"} <= set(result["tools"])
    assert result["set_doc"].startswith("Set the first 2-theta angle of a scan, in deg, from 5 to 140.")
    assert "look at it with view_data" in result["acquire_doc"]
    assert result["risks"] == {
        "XRDTwin_get_start_deg": "low", "XRDTwin_set_start_deg": "high", "XRDTwin_xray_on": "high",
        "XRDTwin_xray_off": "low", "XRDTwin_acquire_diffractogram": "high", "XRDTwin_get_status": "low",
    }
    outcome = result["outcome"]
    assert "acquire_diffractogram refused: it needs tube_power_w > 0" in outcome["refused_without_tube"]
    assert "start_deg = 1 deg is outside [5, 140]" in outcome["out_of_range"]
    assert "count_time_s = 5 s is outside" in outcome["attribute_out_of_range"]
    assert outcome["spectrum"] == {"shape": [301], "axis_offset": 20.0, "axis_scale": 0.2, "axis_label": "2-theta", "axis_units": "deg"}
    assert outcome["tube_attribute"] == 1200.0
    assert "acquire_diffractogram" in outcome["after_init"]  # Init rebuilds the commands


def test_the_server_takes_its_class_name_from_the_launchers_database_entry(monkeypatch):
    import tango

    from asyncroscopy.instruments.yaml_sim import device

    class Servers:
        value_string = ["XRDTwin/instrument_instance"]

    class Database:
        def get_server_list(self, pattern):
            assert pattern == "*/instrument_instance"
            return Servers()

    monkeypatch.setattr(tango, "Database", Database)
    assert device._class_name_for("instrument_instance") == "XRDTwin"
    cls = device.device_class("XRDTwin")
    assert cls.__name__ == "XRDTwin" and issubclass(cls, device.YamlSimDevice) and device.device_class("XRDTwin") is cls

    class NoDatabase:
        def __init__(self):
            raise tango.DevFailed()

    monkeypatch.setattr(tango, "Database", NoDatabase)
    assert device._class_name_for("instrument_instance") == "YamlSimDevice"
