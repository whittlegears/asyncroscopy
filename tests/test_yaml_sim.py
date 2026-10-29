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
    job_estimate,
    job_targets,
    load_spec,
    peak_profile,
    plan_example,
    resolve_job_plan,
)
from asyncroscopy.safety.policy import risk_of

HERE = Path(__file__).parent
EXAMPLE = HERE.parent / "asyncroscopy" / "instruments" / "yaml_sim" / "examples" / "xrd.yaml"
RAMAN = EXAMPLE.with_name("raman.yaml")


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
    ("settings: {g: {default: a, type: str, choices: [b, c]}}", "must be one of ['b', 'c']"),
    ("settings: {g: {default: b, type: str, choices: []}}", "choices: a non-empty list"),
    ("settings: {x: {default: 0}}\ncommands: {go: {doc: Go., input: {name: a}, inputs: [{name: b}]}}", "not both"),
    ("settings: {x: {default: 0}}\ncommands: {go: {doc: Go., inputs: [{name: a}, {name: a}]}}", "used twice"),
    ("settings: {x: {default: 0}}\nacquisitions: {acquire_s: {doc: S., kind: spectrum, axis: {start: 0, stop: 1, points: 9}, peaks: [{position: 0.5, height: 1}]}}", "give fwhm"),
    ("settings: {x: {default: 0}}\nacquisitions: {acquire_s: {doc: S., kind: spectrum, axis: {start: 0, stop: 1, points: 9}, peaks: [{position: 0.5, height: 1, width: 0.1, shape: lorentzian}]}}", "takes fwhm"),
    ("settings: {x: {default: 0}}\nacquisitions: {acquire_s: {doc: S., kind: spectrum, axis: {start: 0, stop: 1, points: 9}, peaks: [{position: 0.5, height: 1, fwhm: 0.1, shape: cauchy}]}}", "not one of ['gaussian', 'lorentzian', 'voigt']"),
    ("settings: {x: {default: 0}}\nacquisitions: {acquire_s: {doc: S., kind: spectrum, axis: {start: 0, stop: 1, points: 9}, peaks: [{position: 0.5, height: 1, fwhm: 0.1, shape: voigt, eta: 2}]}}", "Lorentzian fraction"),
    ("settings: {x: {default: 0}}\nacquisitions: {acquire_v: {doc: V., kind: value, values: {a: x}}}\njobs: {scan: {doc: S., axes: [y], acquire: [acquire_v]}}", "['y'] are not settings"),
    ("settings: {x: {default: 0, settable: false}}\nacquisitions: {acquire_v: {doc: V., kind: value, values: {a: x}}}\njobs: {scan: {doc: S., axes: [x], acquire: [acquire_v]}}", "must be a settable number"),
    ("settings: {x: {default: 0}}\nacquisitions: {acquire_v: {doc: V., kind: value, values: {a: x}}}\njobs: {scan: {doc: S., axes: [x], acquire: []}}", "list the acquisitions"),
    ("settings: {x: {default: 0}}\nacquisitions: {acquire_v: {doc: V., kind: value, values: {a: x}}}\njobs: {scan: {doc: S., axes: [x], acquire: [acquire_v], on_stop: [halt]}}", "['halt'] are not commands"),
    ("settings: {x: {default: 0}}\ncommands: {abort: {doc: Stop.}}\nacquisitions: {acquire_v: {doc: V., kind: value, values: {a: x}}}\njobs: {scan: {doc: S., axes: [x], acquire: [acquire_v]}}", "job device's own commands"),
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
        value_string = ["DataBaseds/2", "XRDTwin/instrument_instance", "XRDTwin/other_instance"]

    class Database:
        def get_server_list(self, pattern):
            assert pattern == "*"  # PyTango's database matches no pattern that starts with a wildcard
            return Servers()

        def get_device_class_list(self, server):
            assert server == "XRDTwin/instrument_instance"
            return type("Listing", (), {"value_string": ["dserver/XRDTwin/instrument_instance", "DServer", "asyncroscopy/instrument/default", "XRDTwin"]})()

        def get_device_property(self, name, key):
            assert (name, key) == ("asyncroscopy/instrument/default", "yaml_path")
            return {"yaml_path": [str(path)]}

    path = EXAMPLE
    monkeypatch.setattr(tango, "Database", Database)
    assert device._registration("instrument_instance") == ("XRDTwin", False)
    cls = device.device_class("XRDTwin")
    assert cls.__name__ == "XRDTwin" and issubclass(cls, device.YamlSimDevice) and device.device_class("XRDTwin") is cls
    path = RAMAN  # a file with jobs: the job device, under the same class name
    assert device._registration("instrument_instance") == ("XRDTwin", True)
    jobs = device.device_class("XRDTwin", jobs=True)
    assert jobs.__name__ == "XRDTwin" and issubclass(jobs, device.YamlSimJobDevice) and jobs is not cls

    class NoDatabase:
        def __init__(self):
            raise tango.DevFailed()

    monkeypatch.setattr(tango, "Database", NoDatabase)
    assert device._registration("instrument_instance") == ("YamlSimDevice", False)


def raman() -> SimInstrument:
    return SimInstrument(load_spec(RAMAN), sleep=lambda seconds: None)


def test_peak_shapes_share_their_full_width_at_half_maximum():
    x = np.linspace(-10, 10, 20001)
    for eta in (0.0, 0.5, 1.0):
        profile = peak_profile(x, 0.0, 4.0, eta)
        above = x[profile >= 0.5]
        assert profile.max() == pytest.approx(1.0) and above[-1] - above[0] == pytest.approx(4.0, abs=0.01)
    # A Lorentzian keeps more in its tails than a Gaussian of the same width.
    assert peak_profile(np.array([6.0]), 0.0, 4.0, 1.0)[0] > 10 * peak_profile(np.array([6.0]), 0.0, 4.0, 0.0)[0]


def test_width_still_means_a_gaussian_sigma(tmp_path):
    spec = load_spec(write(tmp_path, """
settings: {x: {default: 0}}
acquisitions:
  acquire_s: {doc: S., kind: spectrum, noise: none, axis: {start: -10, stop: 10, points: 20001}, peaks: [{position: 0, height: 1, width: 1.0}]}
"""))
    _, _, data, attrs = SimInstrument(spec).acquire("acquire_s")
    x = attrs["axis_offset"] + attrs["axis_scale"] * np.arange(data.size)
    above = x[data >= 0.5]
    assert above[-1] - above[0] == pytest.approx(2.3548, abs=0.01)


def test_choices_inputs_and_units():
    sim = raman()
    with pytest.raises(Refused, match=r"grating must be one of \['600', '1800'\]"):
        sim.set("grating", "1200")
    sim.set("grating", "600")
    assert "one of 600, 1800" in sim.spec.settings["grating"].describe()
    assert json.loads(sim.run_command("move_xy", {"x_m": 1e-4, "y_m": -2e-4})) == {"stage_m": [1e-4, -2e-4]}
    assert json.loads(sim.run_command("move_xy", '{"x_m": 0, "y_m": 0}')) == {"stage_m": [0.0, 0.0]}
    with pytest.raises(Refused, match=r"missing \['y_m'\]"):
        sim.run_command("move_xy", {"x_m": 1e-4})
    with pytest.raises(Refused, match="x_m = 1 m is outside"):
        sim.run_command("move_xy", {"x_m": 1, "y_m": 0})
    sim.run_command("laser_on", 5)
    sim.run_command("move_xy", {"x_m": 1e-4, "y_m": 0})
    _, _, data, attrs = sim.acquire("acquire_raman_spectrum")
    shift = attrs["axis_offset"] + attrs["axis_scale"] * np.arange(data.size)
    assert attrs["units"] == "counts" and attrs["axis_units"] == "cm-1"
    assert abs(shift[np.argmax(data)] - 520.7) < 5  # Si
    assert data[np.argmin(abs(shift - 2690))] > 5 * attrs.get("background", 60)  # graphene 2D band on the flake


def test_a_single_call_is_capped_but_a_job_step_is_not():
    sim = raman()
    sim.run_command("laser_on", 5)
    sim.set("exposure_s", 30)
    with pytest.raises(Refused, match="Run it as part of the line_scan job"):
        sim.acquire("acquire_raman_spectrum")
    waited = []
    sim.acquire("acquire_raman_spectrum", limit=None, sleep=waited.append)
    assert waited == [30.0]


def test_a_job_plan_is_resolved_estimated_and_bounded():
    spec, sim = load_spec(RAMAN), raman()
    with pytest.raises(ValueError, match="needs laser_power_mw > 0"):
        resolve_job_plan(spec, plan_example(spec, spec.jobs["line_scan"]), sim.values)
    sim.run_command("laser_on", 5)
    grid = resolve_job_plan(spec, {"axes": {"stage_x_m": {"start": 0, "stop": 1e-4, "points": 3}, "stage_y_m": {"start": 0, "stop": 2e-4, "points": 2}}, "settings": {"exposure_s": 2}}, sim.values)
    assert grid.count == 6 and grid.point(0) == {"stage_x_m": 0.0, "stage_y_m": 0.0}
    assert grid.point(1) == {"stage_x_m": 0.0, "stage_y_m": 2e-4} and grid.point(5) == {"stage_x_m": 1e-4, "stage_y_m": 2e-4}
    estimate = job_estimate(spec, sim, grid, sim.values)
    assert estimate["points"] == 6 and estimate["per_point_s"] == pytest.approx(2.2)
    listed = resolve_job_plan(spec, {"points": [{"stage_x_m": 1e-4}, {"stage_x_m": 0.01, "stage_y_m": 0}]}, sim.values)
    violations = [t for t in job_targets(spec, listed) if not t[3] <= t[2] <= t[4]]
    assert violations == [(1, "stage_x_m", 0.01, -0.005, 0.005)]
    for plan, message in [
        ({"axes": {"focus_z_m": {"start": 0, "points": 2}}}, "not an axis of line_scan"),
        ({"axes": {"stage_x_m": {"start": 0, "stop": 1, "points": 0}}}, "at least 1"),
        ({"axes": {"stage_x_m": {"start": 0}}, "points": [{"stage_x_m": 0}]}, "not both"),
        ({"axes": {"stage_x_m": {"start": 0, "stop": 1e-3, "points": 1000}, "stage_y_m": {"start": 0, "stop": 1e-3, "points": 1000}}}, "at most 100000"),
        ({"axes": {"stage_x_m": {"start": 0}}, "settings": {"objective": "10x"}}, "cannot be set by line_scan"),
        ({"axes": {"stage_x_m": {"start": 0}}, "settings": {"grating": "1200"}}, "must be one of"),
        ({"axis": {}}, "unknown plan field"),
    ]:
        with pytest.raises(ValueError, match=message):
            resolve_job_plan(spec, plan, sim.values)


def test_the_raman_example_runs_jobs_as_a_tango_job_device(tmp_path):
    # A fresh interpreter: PyTango allows one in-process test context each, and this suite has one.
    done = subprocess.run([sys.executable, str(HERE / "yaml_sim_job_check.py"), str(tmp_path)], capture_output=True, text=True, timeout=240, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    outcome = json.loads(done.stdout.strip().splitlines()[-1])
    assert outcome["risks"] == {
        "RamanTwin_reserve": "low", "RamanTwin_start": "high", "RamanTwin_abort": "low",
        "RamanTwin_laser_off": "low", "RamanTwin_move_xy": "high", "RamanTwin_set_grating": "high",
    }
    assert '"x_m"' in outcome["move_doc"] and '"y_m"' in outcome["move_doc"]  # the MCP server offers both fields
    assert outcome["refused_without_laser"]["error"]["code"] == "plan_rejected"
    assert "needs laser_power_mw > 0" in outcome["refused_without_laser"]["error"]["message"]
    assert outcome["move_two_inputs"] == '{"stage_m": [0.0001, -0.0001]}'
    assert "missing ['y_m']" in outcome["move_missing"] and "must be one of" in outcome["grating_choice"]
    assert outcome["out_of_limits"]["error"]["violations"][0]["attribute"] == "stage_x_m"
    assert outcome["estimate"]["points"] == 3 and outcome["done_state"] == "done" and outcome["keys"] == 3
    assert outcome["last_point"] == {"point": {"stage_x_m": 0.0001}, "units": "counts", "axis_units": "cm-1"}
    assert "busy: job" in outcome["busy_set"] and "busy: job" in outcome["busy_acquire"]
    assert json.loads(outcome["status_while_running"])["laser_power_mw"] == 5.0  # reading still works
    assert outcome["aborted_state"] == "aborted" and outcome["laser_after_abort"] == 0.0  # on_stop switched the laser off
