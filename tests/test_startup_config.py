"""Startup scripts own their config rules: settings, device validation, disabled devices and shutdown."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from startup_scripts import run_llm, run_mcp, run_servers
from startup_scripts.config_schema import ConfigError, Setting, coerce, read_settings
from startup_scripts.lifecycle import install_shutdown_handlers, process_alive

CONFIGS = run_servers.PROJECT_DIR / "configs"


def server_yaml(tmp_path, devices, **top):
    raw = yaml.safe_load((CONFIGS / "DigitalTwin.yaml").read_text())
    raw["devices"] = devices
    raw.update(top)
    path = tmp_path / "servers.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def robot_module(tmp_path, monkeypatch):
    """A device module that exists only in this test, like one a user adds for new hardware."""
    package = tmp_path / "lab_devices"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "robot.py").write_text(textwrap.dedent("""
        class Helper:
            pass

        class PipettingRobot:
            pass

        if __name__ == "__main__":
            PipettingRobot.run_server()
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    return "lab_devices.robot"


# ---------------------------------------------------------------- schema


def test_coerce_converts_and_explains():
    assert coerce(Setting("p", "int"), "9094") == 9094
    assert coerce(Setting("p", "bool"), "no") is False
    assert coerce(Setting("p", "list"), "a, b,") == ["a", "b"]
    assert coerce(Setting("p", "int"), "") is None
    with pytest.raises(ValueError, match="whole number"):
        coerce(Setting("tango.port", "int"), "ninety")
    with pytest.raises(ValueError, match="whole number"):
        coerce(Setting("tango.port", "int"), True)
    with pytest.raises(ValueError, match="mapping"):
        coerce(Setting("p", "yaml", {}), ["not", "a", "mapping"])


def test_read_settings_reports_every_problem_at_once():
    settings = [Setting("a.b", "int", required=True), Setting("c", "bool"), Setting("d", default="x")]
    with pytest.raises(ConfigError) as excinfo:
        read_settings({"c": "maybe"}, settings)
    assert len(excinfo.value.problems) == 2
    assert read_settings({"a": {"b": "3"}}, settings) == {"a.b": 3, "c": None, "d": "x"}


# ---------------------------------------------------------------- run_servers devices


def test_every_shipped_server_config_is_checked(tmp_path):
    for path in sorted(CONFIGS.glob("*.yaml")):
        if not run_servers.is_config(yaml.safe_load(path.read_text())):
            continue
        if path.name == "DigitalTwin copy.yaml":
            with pytest.raises(ConfigError, match="flucam.*not found"):
                run_servers.load_config(path)
        else:
            assert run_servers.load_config(path).support_devices, path.name


def test_a_new_device_in_yaml_starts_without_code_changes(tmp_path, robot_module):
    config = run_servers.load_config(server_yaml(tmp_path, {
        "data": {"module_name": "asyncroscopy.data.data"},
        "robot": {"module_name": robot_module, "properties": {"speed": 3, "tips": ["a", "b"]}},
    }))
    robot = next(device for device in config.support_devices if device.key == "robot")
    assert robot.class_name == "PipettingRobot"  # the class passed to run_server(), not the first class
    assert robot.properties == {"speed": ["3"], "tips": ["a", "b"]}
    assert robot.command[-2:] == [robot_module, "robot_instance"]
    assert run_servers.instrument_properties(config)["robot_device_address"] == ["asyncroscopy/robot/default"]


def test_disabled_device_is_not_built_and_may_be_unfinished(tmp_path):
    config = run_servers.load_config(server_yaml(tmp_path, {
        "data": {"module_name": "asyncroscopy.data.data"},
        "scan": {"module_name": "asyncroscopy.instruments.electron_microscope.hardware.scan", "enabled": False},
        "draft": {"module_name": "not.written.yet", "enabled": False},
    }))
    assert [device.key for device in config.support_devices] == ["data"]
    assert set(config.disabled_devices) == {"scan", "draft"}
    assert "scan_device_address" not in run_servers.instrument_properties(config)
    assert [d.key for d in run_servers.build_devices(config)] == ["data", "instrument"]


@pytest.mark.parametrize("devices, message", [
    ({"cam": {"module_name": "asyncroscopy.missing"}}, "module 'asyncroscopy.missing' not found"),
    ({"cam": {"modul_name": "x"}}, "unknown key"),
    ({"cam": None}, "needs at least a module_name"),
    ({"cam": {"class_name": "CAMERA"}}, "missing module_name"),
    ({"bad name": {"module_name": "asyncroscopy.data.data"}}, "must start with a letter"),
    ({"instrument": {"module_name": "asyncroscopy.data.data"}}, "reserved"),
    ({"cam": {"module_name": "asyncroscopy.data.data", "class_name": "Nope"}}, "class 'Nope' is not defined"),
    ({"cam": {"module_name": "asyncroscopy.data.data", "enabled": "perhaps"}}, "true or false"),
    ({"cam": {"module_name": "asyncroscopy.data.data", "properties": [1]}}, "properties must be a mapping"),
])
def test_bad_device_entries_are_refused_before_anything_starts(tmp_path, devices, message):
    with pytest.raises(ConfigError, match=message):
        run_servers.load_config(server_yaml(tmp_path, devices))


def test_all_problems_are_listed_together(tmp_path):
    path = server_yaml(tmp_path, {"data": {"module_name": "asyncroscopy.data.data"},
                                  "a": {"module_name": "nope.a"}, "b": {"module_name": "nope.b"}})
    with pytest.raises(ConfigError) as excinfo:
        run_servers.load_config(path)
    assert len(excinfo.value.problems) == 2


def test_instrument_class_is_checked_against_its_file(tmp_path):
    path = server_yaml(tmp_path, {"data": {"module_name": "asyncroscopy.data.data"}}, instrument={"file": "asyncroscopy/instruments/electron_microscope/digital_twin.py", "class_name": "Typo"})
    with pytest.raises(ConfigError, match="class 'Typo' is not defined in digital_twin.py"):
        run_servers.load_config(path)
    path = server_yaml(tmp_path, {"data": {"module_name": "asyncroscopy.data.data"}}, instrument={"file": "asyncroscopy/instruments/electron_microscope/auto_script.py"})
    assert run_servers.load_config(path).instrument.class_name == "AutoScriptMicroscope"


def test_class_inference_ignores_a_variable_passed_to_run_server(tmp_path):
    path = server_yaml(tmp_path, {"data": {"module_name": "asyncroscopy.data.data"},
                                  "feedback": {"module_name": "asyncroscopy.instruments.scanning_probe_microscope.jupiter_api"}})
    with pytest.raises(ConfigError, match="feedback: cannot tell which class"):
        run_servers.load_config(path)


def test_external_package_runs_its_main(tmp_path, monkeypatch):
    package = tmp_path / "ext_lab"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text("class Robot:\n    pass\n\nRobot.run_server()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    config = run_servers.load_config(server_yaml(tmp_path, {
        "data": {"module_name": "asyncroscopy.data.data"}, "robot": {"module_name": "ext_lab"}}))
    assert [(d.key, d.class_name) for d in config.support_devices][-1] == ("robot", "Robot")


def test_instrument_file_outside_the_project_is_refused(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("class Outside:\n    pass\n")
    with pytest.raises(ConfigError, match="must be inside"):
        run_servers.load_config(server_yaml(tmp_path, {"data": {"module_name": "asyncroscopy.data.data"}},
                                            instrument={"file": str(outside)}))


def test_tiled_autostart_needs_the_data_device(tmp_path):
    devices = {"data": {"module_name": "asyncroscopy.data.data", "enabled": False}}
    with pytest.raises(ConfigError, match="tiled.autostart needs the data device"):
        run_servers.load_config(server_yaml(tmp_path, devices))
    tiled = yaml.safe_load((CONFIGS / "DigitalTwin.yaml").read_text())["tiled"] | {"autostart": False}
    assert run_servers.load_config(server_yaml(tmp_path, devices, tiled=tiled)).disabled_devices == ("data",)


def test_register_devices_removes_disabled_devices_from_a_kept_database(monkeypatch):
    calls = []

    class FakeDatabase:
        def get_db_host(self):
            return "localhost"

        def get_db_port(self):
            return 9094

        def add_device(self, info):
            calls.append(("add", info.name))

        def unexport_server(self, name):
            pass

        def put_device_property(self, name, props):
            calls.append(("put", name, *props))

        def delete_device(self, name):
            calls.append(("delete", name))

        def delete_device_property(self, name, props):
            calls.append(("delete_property", name, *props))

    monkeypatch.setattr(run_servers.tango, "Database", FakeDatabase)
    devices = [run_servers.DeviceConfig("data", "DATA", "asyncroscopy.data.data"),
               run_servers.DeviceConfig("instrument", "DigitalTwin", "x", start_after_dependencies=True)]
    run_servers.register_devices(devices, {}, disabled=["eds"])
    assert ("delete", "asyncroscopy/eds/default") in calls
    assert ("delete_property", "asyncroscopy/instrument/default", "eds_device_address") in calls
    assert ("add", "asyncroscopy/eds/default") not in calls


# ---------------------------------------------------------------- run_mcp / run_llm


def test_mcp_tool_log_and_validation(tmp_path):
    raw = yaml.safe_load((CONFIGS / "mcp_dt.yaml").read_text())
    raw["mcp"]["tool_log"] = "outputs/calls.jsonl"
    path = tmp_path / "mcp.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = run_mcp.load_config(path)
    assert run_mcp.build_command(config)[-2:] == ["--tool-log", "outputs/calls.jsonl"]
    raw["mcp"]["http_port"] = "eighty"
    del raw["mcp"]["name"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError) as excinfo:
        run_mcp.load_config(path)
    assert len(excinfo.value.problems) == 2


def test_llm_config_rejects_unknown_keys_readably(tmp_path):
    raw = yaml.safe_load((CONFIGS / "gemma-llm.yaml").read_text())
    assert run_llm.load_config(CONFIGS / "gemma-llm.yaml").agent.startup_agents[0]["name"] == "base"
    raw["agent"]["mcp_ulr"] = "typo"
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="unknown key.*mcp_ulr"):
        run_llm.load_config(path)


def test_llm_config_checks_agents_and_steps(tmp_path):
    raw = yaml.safe_load((CONFIGS / "gemma-llm.yaml").read_text())
    raw["agent"]["max_steps"] = 0
    raw["agent"]["startup_agents"] = [{"system_prompt": "no name"}]
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError) as excinfo:
        run_llm.load_config(path)
    assert len(excinfo.value.problems) == 2


def test_config_kinds_do_not_overlap():
    for path in CONFIGS.glob("*.yaml"):
        raw = yaml.safe_load(path.read_text())
        kinds = [m.__name__ for m in (run_servers, run_mcp, run_llm) if m.is_config(raw)]
        assert len(kinds) <= 1, (path.name, kinds)


# ---------------------------------------------------------------- shutdown


def test_process_alive():
    assert process_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not process_alive(proc.pid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_repeated_stop_signals_raise_once():
    previous = {name: signal.getsignal(getattr(signal, name)) for name in ("SIGTERM", "SIGHUP")}
    install_shutdown_handlers()
    try:
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)  # a forwarded duplicate must not interrupt cleanup
        time.sleep(0.1)
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_script_stops_when_its_parent_exits(tmp_path):
    """Like a GUI that is force-killed: the parent was reading the child's output, so printing breaks too."""
    marker = tmp_path / "stopped"
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(run_servers.PROJECT_DIR)!r})
        from startup_scripts.lifecycle import install_shutdown_handlers
        install_shutdown_handlers(int(sys.argv[1]))
        try:
            while True:
                print("still running", flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("shutting down", flush=True)  # must not raise BrokenPipeError
            open({str(marker)!r}, "w").write("ok")
    """))
    parent = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            import os, subprocess, sys, time
            child = subprocess.Popen([sys.executable, {str(child)!r}, str(os.getpid())], stdout=subprocess.PIPE)
            child.stdout.readline()
            print(child.pid, flush=True)
            os._exit(0)
        """)],
        capture_output=True, text=True, timeout=30,
    )
    child_pid = int(parent.stdout.strip())
    deadline = time.time() + 15
    while process_alive(child_pid) and time.time() < deadline:
        time.sleep(0.2)
    assert not process_alive(child_pid)
    assert marker.read_text() == "ok"


def test_cleanup_flag_exits_without_starting(tmp_path):
    result = subprocess.run(
        [sys.executable, str(run_servers.PROJECT_DIR / "startup_scripts" / "run_mcp.py"),
         "--yaml", str(CONFIGS / "mcp_dt.yaml"), "--cleanup"],
        capture_output=True, text=True, timeout=60, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "Stopped processes left by an earlier run" in result.stdout
