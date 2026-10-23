"""The startup GUIs are built from the startup scripts: no field, device or rule lives in the GUI."""

import os
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from startup_guis.qt_compat import QApplication  # noqa: E402
from startup_guis.script_gui import ScriptGui  # noqa: E402
from startup_guis.shared import ManagedCommand  # noqa: E402
from startup_scripts import run_llm, run_mcp, run_servers  # noqa: E402
from startup_scripts.config_schema import DeviceTable, Setting, get_path, read_settings  # noqa: E402

CONFIGS = run_servers.PROJECT_DIR / "configs"
SCRIPTS = (run_servers, run_mcp, run_llm)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(app, tmp_path):
    windows = []

    def make(script, path=None):
        window = ScriptGui(script, path)
        window.generated_path = tmp_path / "generated.yaml"
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.close()


def write(tmp_path, raw, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return path


def preset_paths(script):
    return {s.path for s in script.SETTINGS if s.presets}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_config_round_trips_unchanged(gui, script):
    """Loading a shipped config and generating with no edits keeps every key, including ones with no widget."""
    names = [p for p in sorted(CONFIGS.glob("*.yaml")) if script.is_config(yaml.safe_load(p.read_text()))]
    assert names
    window = gui(script, names[0])
    for path in names:
        window.load_config_from_path(path)
        for preset in preset_paths(script):  # undo the safe-default preset applied on open
            window.set_shown(preset, get_path(window.raw, preset))
        assert window.current_config() == yaml.safe_load(path.read_text()), path.name


def test_device_checkboxes_come_from_the_yaml(gui):
    window = gui(run_servers, CONFIGS / "Spectra300eels.yaml")
    assert set(window.device_checks) == set(yaml.safe_load((CONFIGS / "Spectra300eels.yaml").read_text())["devices"])
    assert window.device_checks["eels"].isChecked()


def test_server_gui_opens_on_the_digital_twin(gui):
    assert gui(run_servers).loaded_path.name == "DigitalTwin.yaml"


def test_unticked_device_is_never_started_and_reticking_restores_it(gui):
    window = gui(run_servers, CONFIGS / "DigitalTwin.yaml")
    window.device_checks["eds"].setChecked(False)
    window.device_checks["corrector"].setChecked(False)
    path, error = window.check_config(window.generated_path)
    assert error is None
    config = run_servers.load_config(path)
    assert {d.key for d in config.support_devices} == {"camera", "data", "scan", "stage"}
    assert set(config.disabled_devices) == {"eds", "corrector"}
    original = yaml.safe_load((CONFIGS / "DigitalTwin.yaml").read_text())["devices"]["corrector"]
    assert window.current_config()["devices"]["corrector"] == {**original, "enabled": False}
    window.device_checks["corrector"].setChecked(True)
    assert window.current_config()["devices"]["corrector"] == original


def test_new_yaml_with_custom_devices_works_in_the_gui(gui, tmp_path, monkeypatch):
    package = tmp_path / "gui_lab_devices"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "robot.py").write_text("class Robot:\n    pass\n\nif __name__ == '__main__':\n    Robot.run_server()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    raw = yaml.safe_load((CONFIGS / "DigitalTwin.yaml").read_text())
    raw["devices"] = {
        "data": {"module_name": "asyncroscopy.data.data"},
        "robot": {"module_name": "gui_lab_devices.robot", "properties": {"speed": 2}},
        "spare_robot": {"module_name": "gui_lab_devices.robot", "enabled": False},
    }
    window = gui(run_servers, write(tmp_path, raw))
    assert window.device_checks["robot"].isChecked() and not window.device_checks["spare_robot"].isChecked()
    window.device_checks["spare_robot"].setChecked(True)
    window.refresh_preview()
    assert window.validity.text() == "Config is valid."
    config = run_servers.load_config(window.check_config(window.generated_path)[0])
    assert [(d.key, d.class_name) for d in config.support_devices] == [
        ("data", "DATA"), ("robot", "Robot"), ("spare_robot", "Robot")]


def test_invalid_config_is_shown_and_never_started(gui, tmp_path):
    raw = yaml.safe_load((CONFIGS / "DigitalTwin.yaml").read_text())
    raw["devices"]["typo"] = {"module_name": "asyncroscopy.no_such_device"}
    window = gui(run_servers, write(tmp_path, raw))
    window.refresh_preview()
    assert "module 'asyncroscopy.no_such_device' not found" in window.validity.text()
    window.start()
    assert not window.command.running
    window.device_checks["typo"].setChecked(False)  # switching the broken device off makes it startable
    window.refresh_preview()
    assert window.validity.text() == "Config is valid."


def test_edits_write_only_what_changed(gui, tmp_path):
    raw = yaml.safe_load((CONFIGS / "mcp_dt.yaml").read_text())
    raw["mcp"]["tool_log"] = "outputs/calls.jsonl"
    raw["mcp"]["future_option"] = {"kept": True}
    window = gui(run_mcp, write(tmp_path, raw))
    window.set_shown("mcp.http_port", 8123)
    window.set_shown("mcp.include_only_functions", "list_devices, Stage.move")
    config = window.current_config()
    assert config["mcp"]["http_port"] == 8123
    assert config["mcp"]["include_only_functions"] == ["list_devices", "Stage.move"]
    assert config["mcp"]["tool_log"] == "outputs/calls.jsonl"
    assert config["mcp"]["future_option"] == {"kept": True}
    window.set_shown("mcp.http_port", "")
    window.refresh_preview()
    assert "missing required setting 'mcp.http_port'" in window.validity.text()


def test_mcp_gui_opens_on_the_digital_twin_host(gui, tmp_path):
    raw = yaml.safe_load((CONFIGS / "mcp_dt.yaml").read_text())
    raw["tango"]["host"] = "10.46.217.241"
    window = gui(run_mcp, write(tmp_path, raw))
    assert window.current_config()["tango"]["host"] == "localhost"


def test_a_setting_added_to_a_script_appears_without_gui_changes(gui, tmp_path):
    """The contract that keeps the GUI untouched: a script declares something new, the window shows it."""
    settings = [
        Setting("stack.port", "int", required=True, section="Stack"),
        Setting("stack.mode", choices=("fast", "safe"), default="safe", section="Stack"),
        Setting("stack.dry_run", "bool", False, section="Stack"),
        Setting("stack.tags", "list", [], section="Stack"),
    ]
    path = write(tmp_path, {"stack": {"port": 1}, "units": {"a": {"x": 1}, "b": {"enabled": False}}})

    def load_config(p):
        return read_settings(yaml.safe_load(Path(p).read_text()), settings)

    script = SimpleNamespace(
        TITLE="Toy stack", LAUNCHER="startup_scripts/run_toy.py", DEFAULT_CONFIG_PATH=path, SETTINGS=settings,
        DEVICES=DeviceTable(path="units", label="Units"), is_config=lambda raw: "stack" in raw, load_config=load_config,
    )
    window = gui(script)
    assert set(window.fields) == {s.path for s in settings}
    assert set(window.device_checks) == {"a", "b"} and not window.device_checks["b"].isChecked()
    window.set_shown("stack.mode", "fast")
    window.set_shown("stack.dry_run", True)
    window.device_checks["b"].setChecked(True)
    assert window.current_config() == {
        "stack": {"port": 1, "mode": "fast", "dry_run": True}, "units": {"a": {"x": 1}, "b": {"enabled": True}}}


def test_non_matching_yaml_is_not_loaded(gui):
    window = gui(run_mcp, CONFIGS / "mcp_dt.yaml")
    before = window.current_config()
    window.load_config_from_path(CONFIGS / "DigitalTwin.yaml")
    assert window.current_config() == before


def test_status_badge_follows_script_patterns(gui):
    window = gui(run_mcp, CONFIGS / "mcp_dt.yaml")
    window.enqueue_output("MCP ready: 12 tool(s) registered\n")
    assert window.badge.text() == "12 tools"
    window.enqueue_output("MCP ERROR: port 8000 already in use\n")
    assert window.badge.text() == "port in use" and window.badge_error


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_stop_lets_the_script_clean_up(app, tmp_path):
    """Stop interrupts the script like Ctrl+C instead of SIGKILLing it, so its own cleanup runs."""
    marker = tmp_path / "cleaned"
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(run_servers.PROJECT_DIR)!r})
        from startup_scripts.lifecycle import install_shutdown_handlers
        install_shutdown_handlers()
        print("ready", flush=True)
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            open({str(marker)!r}, "w").write("ok")
    """))
    lines, done = [], []
    command = ManagedCommand(lines.append, done.append)
    command.start([sys.executable, str(script)])
    deadline = time.time() + 10
    while not any("ready" in line for line in lines) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    command.stop(wait=True)
    assert marker.read_text() == "ok"
    assert not command.running
