"""The startup window is built from the YAML it opens and the script that runs it: no key, device or rule lives in the GUI."""

import ast
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
from startup_guis.script_gui import ScriptGui, form_fields  # noqa: E402
from startup_guis.shared import ManagedCommand  # noqa: E402
from startup_scripts import registry, run_llm, run_mcp, run_segmentation, run_servers  # noqa: E402
from startup_scripts.config_schema import Setting, read_settings  # noqa: E402

CONFIGS = run_servers.PROJECT_DIR / "configs"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(app, tmp_path, monkeypatch):
    monkeypatch.setattr("startup_guis.script_gui.GENERATED_CONFIG_DIR", tmp_path)  # keep outputs/ clean
    windows = []

    def make(path=None):
        window = ScriptGui(path)
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.close()


def write(tmp_path, raw, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return path


def devices(window):
    return window.entry_checks["devices"]


# ---------------------------------------------------------------- backend and UI stay apart


@pytest.mark.parametrize("path", sorted((run_servers.PROJECT_DIR / "startup_scripts").glob("*.py")), ids=lambda p: p.name)
def test_startup_scripts_never_import_the_gui(path):
    tree = ast.parse(path.read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not {name for name in imported if name.startswith(("startup_guis", "PyQt", "PySide"))}


def test_scripts_declare_only_what_their_config_means():
    assert set(Setting.__dataclass_fields__) == {"path", "kind", "default", "required", "choices"}
    for script in registry.scripts():
        assert not {"TITLE", "LAUNCHER", "STATUS_PATTERNS", "DEVICES", "GUI_DEFAULT_CONFIG_PATH"} & set(vars(script))


# ---------------------------------------------------------------- configs and devices come from the YAML


def test_every_config_in_the_folder_is_listed_with_its_script(gui):
    window = gui()
    listed = [window.config_combo.itemData(i) for i in range(window.config_combo.count())]
    on_disk = sorted(str(p) for p in [*CONFIGS.glob("*.yaml"), *CONFIGS.glob("*.yml")])
    assert sorted(listed) == on_disk
    assert {registry.script_name(s) for _, s in registry.config_files()} == {
        "run_servers", "run_mcp", "run_llm", "run_segmentation"}


def test_every_config_round_trips_unchanged(gui):
    """Loading each shipped config and generating with no edits keeps every key exactly."""
    window = gui()
    for path, _script in registry.config_files():
        window.load_config_from_path(path)
        assert window.current_config() == yaml.safe_load(path.read_text()), path.name


def test_opening_a_config_switches_to_its_script(gui):
    window = gui(CONFIGS / "mcp_dt.yaml")
    assert window.script is run_mcp and window.windowTitle() == "Asyncroscopy MCP Startup"
    window.load_config_from_path(CONFIGS / "Segmentation.yaml")
    assert window.script is run_segmentation and "compute_device" in window.fields
    window.load_config_from_path(CONFIGS / "gemma-llm.yaml")
    assert window.script is run_llm and window.launch_command(Path("x.yaml"))[4] == "startup_scripts/run_llm.py"


def test_window_opens_on_the_digital_twin(gui):
    assert gui().loaded_path.name == "DigitalTwin.yaml"


def test_device_checkboxes_come_from_the_yaml(gui):
    window = gui(CONFIGS / "Spectra300eels.yaml")
    assert set(devices(window)) == set(yaml.safe_load((CONFIGS / "Spectra300eels.yaml").read_text())["devices"])
    assert devices(window)["eels"].isChecked()


def test_every_yaml_key_gets_a_field_even_if_no_script_declares_it(gui, tmp_path):
    raw = yaml.safe_load((CONFIGS / "mcp_dt.yaml").read_text())
    raw["mcp"]["retry_seconds"] = 0.5
    raw["mcp"]["mode"] = "fast"
    window = gui(write(tmp_path, raw))
    assert {"mcp.retry_seconds", "mcp.mode", "tango.host"} <= set(window.fields)
    window.set_shown("mcp.retry_seconds", "1.5")
    assert window.current_config()["mcp"]["retry_seconds"] == 1.5  # a number stays a number


def test_unticked_device_is_never_started_and_reticking_restores_it(gui):
    window = gui(CONFIGS / "DigitalTwin.yaml")
    devices(window)["eds"].setChecked(False)
    devices(window)["corrector"].setChecked(False)
    path, error = window.check_config(window.generated_path)
    assert error is None
    config = run_servers.load_config(path)
    assert {d.key for d in config.support_devices} == {"analysis", "camera", "data", "scan", "stage"}
    assert set(config.disabled_devices) == {"eds", "corrector"}
    original = yaml.safe_load((CONFIGS / "DigitalTwin.yaml").read_text())["devices"]["corrector"]
    assert window.current_config()["devices"]["corrector"] == {**original, "enabled": False}
    devices(window)["corrector"].setChecked(True)
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
    window = gui(write(tmp_path, raw))
    assert devices(window)["robot"].isChecked() and not devices(window)["spare_robot"].isChecked()
    devices(window)["spare_robot"].setChecked(True)
    window.refresh_preview()
    assert window.validity.text() == "Config is valid."
    config = run_servers.load_config(window.check_config(window.generated_path)[0])
    assert [(d.key, d.class_name) for d in config.support_devices] == [
        ("data", "DATA"), ("robot", "Robot"), ("spare_robot", "Robot")]


def test_invalid_config_is_shown_and_never_started(gui, tmp_path):
    raw = yaml.safe_load((CONFIGS / "DigitalTwin.yaml").read_text())
    raw["devices"]["typo"] = {"module_name": "asyncroscopy.no_such_device"}
    window = gui(write(tmp_path, raw))
    window.refresh_preview()
    assert "module 'asyncroscopy.no_such_device' not found" in window.validity.text()
    window.start()
    assert not window.command.running
    devices(window)["typo"].setChecked(False)  # switching the broken device off makes it startable
    window.refresh_preview()
    assert window.validity.text() == "Config is valid."


def test_edits_write_only_what_changed(gui, tmp_path):
    raw = yaml.safe_load((CONFIGS / "mcp_dt.yaml").read_text())
    raw["mcp"]["tool_log"] = "outputs/calls.jsonl"
    raw["mcp"]["future_option"] = {"kept": True}
    window = gui(write(tmp_path, raw))
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


def test_string_booleans_show_as_the_script_reads_them(gui, tmp_path):
    raw = yaml.safe_load((CONFIGS / "DigitalTwin.yaml").read_text())
    raw["devices"]["eds"]["enabled"] = "False"
    raw["devices"]["camera"]["enabled"] = "0"
    raw["tiled"]["autostart"] = "false"
    window = gui(write(tmp_path, raw))
    assert not devices(window)["eds"].isChecked() and not devices(window)["camera"].isChecked()
    assert window.get_shown("tiled.autostart") is False
    devices(window)["camera"].setChecked(True)
    config = run_servers.load_config(window.check_config(window.generated_path)[0])
    assert "camera" in {d.key for d in config.support_devices} and "eds" in config.disabled_devices


def test_failed_config_write_is_reported_not_a_crash(gui, tmp_path):
    window = gui(CONFIGS / "DigitalTwin.yaml")
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    _, error = window.check_config(blocker / "cannot.yaml")
    assert error


def test_a_new_script_and_its_yaml_appear_without_gui_changes(gui, tmp_path, monkeypatch):
    """The contract that keeps the GUI untouched: a script declares its settings, the window shows its YAML."""
    settings = [
        Setting("stack.port", "int", required=True),
        Setting("stack.mode", choices=("fast", "safe"), default="safe"),
        Setting("stack.dry_run", "bool", False),
        Setting("stack.tags", "list", []),
        Setting("units", "entries", {}),
    ]
    path = write(tmp_path, {"stack": {"port": 1, "note": "kept"}, "units": {"a": {"x": 1}, "b": {"enabled": False}}})

    def load_config(p):
        return read_settings(yaml.safe_load(Path(p).read_text()), settings)

    toy = SimpleNamespace(
        __name__="startup_scripts.run_toy", __file__=str(tmp_path / "run_toy.py"), DEFAULT_CONFIG_PATH=path,
        SETTINGS=settings, is_config=lambda raw: "stack" in raw, load_config=load_config, main=lambda argv=None: 0,
    )
    monkeypatch.setattr(registry, "scripts", lambda: (toy,))
    window = gui(path)
    assert window.windowTitle() == "Asyncroscopy Toy Startup"
    assert set(window.fields) == {"stack.port", "stack.note", "stack.mode", "stack.dry_run", "stack.tags"}
    assert set(window.entry_checks["units"]) == {"a", "b"} and not window.entry_checks["units"]["b"].isChecked()
    window.set_shown("stack.mode", "fast")
    window.set_shown("stack.dry_run", True)
    window.entry_checks["units"]["b"].setChecked(True)
    assert window.current_config() == {
        "stack": {"port": 1, "note": "kept", "mode": "fast", "dry_run": True},
        "units": {"a": {"x": 1}, "b": {"enabled": True}}}


def test_form_follows_the_yaml_order_then_declared_keys():
    settings = [Setting("a.declared_only"), Setting("a.x", "int")]
    fields = form_fields(settings, {"b": 1, "a": {"y": [1, 2], "x": 3}, "group": {"one": {}, "two": None}})
    assert [(f.path, f.kind) for f in fields] == [
        ("b", "int"), ("a.y", "list"), ("a.x", "int"), ("group", "entries"), ("a.declared_only", "str")]


def test_start_waits_for_a_previous_stop_to_finish(app):
    import threading

    lines = []
    command = ManagedCommand(lines.append, lambda code: None)
    release = threading.Event()
    command._stopping = threading.Thread(target=release.wait)
    command._stopping.start()
    command.start([sys.executable, "-c", "pass"])
    release.set()
    assert command.process is None and "Still stopping" in lines[-1]


def test_yaml_no_script_runs_is_not_loaded(gui, tmp_path):
    window = gui(CONFIGS / "mcp_dt.yaml")
    before = window.current_config()
    window.load_config_from_path(write(tmp_path, {"unrelated": {"key": 1}}))
    assert window.current_config() == before and "not a config for any startup script" in window.output.toPlainText()


def test_status_badge_follows_the_hints_for_the_script(gui):
    window = gui(CONFIGS / "mcp_dt.yaml")
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


def test_config_write_is_all_or_nothing(tmp_path, monkeypatch):
    from startup_guis import shared

    path = shared.write_yaml(tmp_path / "run.yaml", {"tango": {"host": "localhost"}})
    monkeypatch.setattr(shared.os, "replace", lambda *a: (_ for _ in ()).throw(OSError(28, "No space left on device")))
    with pytest.raises(OSError):
        shared.write_yaml(path, {"tango": {"host": "elsewhere"}})
    assert yaml.safe_load(path.read_text()) == {"tango": {"host": "localhost"}}  # a reader never sees a partial file
    assert [p.name for p in tmp_path.iterdir()] == ["run.yaml"]
