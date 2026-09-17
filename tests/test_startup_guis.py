import json
import os
import signal
import sys
import time

import pytest

from asyncroscopy.utils import process_manager
from startup_guis import mcp_gui, server_gui, shared


def _values(**overrides):
    """Minimal server_config_from_values input; tests override only what they exercise."""
    values = {
        'instrument': {
            'class_name': 'DigitalTwin',
            'file': 'asyncroscopy/instruments/electron_microscope/digital_twin.py',
            'description': 'Digital twin',
        },
        'instrument_file': 'asyncroscopy/instruments/electron_microscope/digital_twin.py',
        'hardware_host': '',
        'hardware_port': '',
        'hardware_timeout_seconds': '120',
        'devices': {},
        'enabled_devices': {},
        'tango_host': 'localhost',
        'tango_port': '9094',
        'reset_database_file': False,
        'tiled_host': 'localhost',
        'tiled_port': '9091',
        'acquisition_dir': 'outputs/tiled_acquisitions',
        'tiled_autostart': True,
        'tiled_register_on_startup': False,
        'device_timeout_seconds': '120',
    }
    values.update(overrides)
    return values


def test_server_gui_builds_server_yaml():
    config = server_gui.server_config_from_values(
        {
            'instrument': {
                'class_name': 'AutoScriptMicroscope',
                'file': 'asyncroscopy/instruments/electron_microscope/auto_script.py',
                'description': 'Real microscope',
            },
            'instrument_file': 'asyncroscopy/instruments/electron_microscope/auto_script.py',
            'hardware_host': '10.0.0.1',
            'hardware_port': '9095',
            'hardware_timeout_seconds': '120',
            'devices': {
                'data': {'module_name': 'asyncroscopy.data.data'},
                'scan': {'module_name': 'asyncroscopy.instruments.electron_microscope.hardware.scan'},
            },
            'enabled_devices': {'data': True, 'scan': False},
            'tango_host': 'localhost',
            'tango_port': '9094',
            'reset_database_file': True,
            'tiled_host': 'localhost',
            'tiled_port': '9091',
            'acquisition_dir': 'outputs/tiled_acquisitions',
            'tiled_autostart': True,
            'tiled_register_on_startup': False,
            'device_timeout_seconds': '120',
        }
    )

    assert config['instrument']['file'] == 'asyncroscopy/instruments/electron_microscope/auto_script.py'
    assert config['instrument']['hardware_host'] == '10.0.0.1'
    assert config['instrument']['hardware_port'] == 9095
    assert config['instrument']['timeout_seconds'] == 120
    assert config['devices'] == {'data': {'module_name': 'asyncroscopy.data.data'}}
    assert config['tango'] == {'host': 'localhost', 'port': 9094, 'reset_database_file': True}
    assert config['tiled']['register_on_startup'] is False
    assert config['device_timeout_seconds'] == 120


def test_server_gui_keeps_class_name_and_properties_of_enabled_devices():
    """A device's full YAML spec survives the round trip, not just its module."""
    spec = {
        'class_name': 'AutoScriptSTAGE',
        'module_name': 'asyncroscopy.instruments.electron_microscope.hardware.stage_autoscript',
        'properties': {'hardware_host': '10.0.0.1', 'hardware_port': 9095},
    }
    config = server_gui.server_config_from_values(_values(devices={'stage': spec}, enabled_devices={'stage': True}))

    assert config['devices']['stage'] == spec


def test_server_gui_writes_no_device_the_config_did_not_declare():
    """Only declared, ticked devices reach the generated YAML."""
    config = server_gui.server_config_from_values(
        _values(
            devices={'data': {'module_name': 'asyncroscopy.data.data'}},
            enabled_devices={'data': True},
        )
    )

    assert list(config['devices']) == ['data']


def test_server_gui_treats_a_device_without_a_checkbox_as_disabled():
    """Explicit opt-in: a device with no checkbox state is left out, not assumed on."""
    config = server_gui.server_config_from_values(
        _values(
            devices={'data': {'module_name': 'asyncroscopy.data.data'}},
            enabled_devices={},
        )
    )

    assert config['devices'] == {}


def test_server_gui_does_not_mutate_the_loaded_device_config():
    """The GUI writes a copy, so editing the generated config cannot corrupt the source YAML."""
    spec = {'module_name': 'asyncroscopy.data.data', 'properties': {'port': 9091}}
    devices = {'data': spec}
    config = server_gui.server_config_from_values(_values(devices=devices, enabled_devices={'data': True}))

    config['devices']['data']['properties']['port'] = 1234

    assert spec['properties']['port'] == 9091


def test_server_gui_populates_devices_from_the_loaded_config():
    """Checkboxes come from the config's devices mapping, in its own order."""
    added: list[tuple[str, int, int]] = []

    class FakeGrid:
        def count(self):
            return 0

        def addWidget(self, widget, row, column):  # Qt API
            added.append((getattr(widget, 'label', ''), row, column))

    class FakeCheckBox:
        def __init__(self, label):
            self.label = label
            self.checked = False

        def setChecked(self, value):  # Qt API
            self.checked = value

        @property
        def stateChanged(self):  # Qt API
            return type('Signal', (), {'connect': lambda _self, _slot: None})()

    class FakeSection:
        form = FakeGrid()

    class FakeGui:
        populate_devices = server_gui.ServerGui.populate_devices

    gui = FakeGui()
    gui.devices_section = FakeSection()
    gui.device_checks = {}
    gui.device_config = {
        'corrector': {'module_name': 'asyncroscopy.instruments.electron_microscope.hardware.corrector'},
        'data': {'module_name': 'asyncroscopy.data.data'},
    }
    gui.refresh_yaml = lambda: None

    original_checkbox = server_gui.CheckBox
    server_gui.CheckBox = FakeCheckBox
    try:
        gui.populate_devices()
    finally:
        server_gui.CheckBox = original_checkbox

    assert list(gui.device_checks) == ['corrector', 'data']
    assert all(checkbox.checked for checkbox in gui.device_checks.values())
    assert added == [('corrector', 0, 0), ('data', 0, 1)]


def test_server_gui_omits_hardware_host_port_for_digital_twin_file():
    config = server_gui.server_config_from_values(
        {
            'instrument': {
                'class_name': 'AutoScriptMicroscope',
                'file': 'asyncroscopy/instruments/electron_microscope/auto_script.py',
                'description': 'Real microscope',
                'hardware_host': '10.0.0.1',
                'hardware_port': 9095,
            },
            'instrument_file': 'asyncroscopy/instruments/electron_microscope/digital_twin.py',
            'hardware_host': '10.0.0.1',
            'hardware_port': '9095',
            'hardware_timeout_seconds': '120',
            'devices': {'data': {'module_name': 'asyncroscopy.data.data'}},
            'enabled_devices': {'data': True},
            'tango_host': 'localhost',
            'tango_port': '9094',
            'reset_database_file': True,
            'tiled_host': 'localhost',
            'tiled_port': '9091',
            'acquisition_dir': 'outputs/tiled_acquisitions',
            'tiled_autostart': True,
            'tiled_register_on_startup': False,
            'device_timeout_seconds': '120',
        }
    )

    assert config['instrument']['class_name'] == 'DigitalTwin'
    assert 'hardware_host' not in config['instrument']
    assert 'hardware_port' not in config['instrument']


def test_server_gui_reads_and_writes_line_and_combo_inputs():
    class FakeLineEdit:
        def __init__(self):
            self.value = ''

        def text(self):
            return self.value

        def setText(self, value):
            self.value = value

    class FakeComboBox:
        def __init__(self):
            self.value = ''

        def currentText(self):
            return self.value

        def setCurrentText(self, value):
            self.value = value

    class FakeGui:
        input_text = server_gui.ServerGui.input_text
        set_input_text = server_gui.ServerGui.set_input_text

    gui = FakeGui()
    gui.inputs = {'line': FakeLineEdit(), 'combo': FakeComboBox()}

    gui.set_input_text('line', 'localhost')
    gui.set_input_text('combo', server_gui.PROJECT_DIR / 'outputs' / 'tiled_acquisitions')

    assert gui.input_text('line') == 'localhost'
    assert gui.input_text('combo') == 'outputs/tiled_acquisitions'


def test_mcp_gui_builds_mcp_yaml():
    config = mcp_gui.mcp_config_from_values(
        {
            'tango_host': '10.0.0.2',
            'tango_port': '9094',
            'name': 'Spectra300_MCP',
            'transport': 'streamable-http',
            'http_host': '0.0.0.0',
            'http_port': '8000',
            'data_device_address': 'asyncroscopy/data/default',
            'quiet': True,
            'blocked_classes': 'DataBase, DServer',
            'blocked_functions': '"*":\n  - Init\n  - Kill\n',
        }
    )

    assert config['tango'] == {'host': '10.0.0.2', 'port': 9094}
    assert config['mcp']['http_host'] == '0.0.0.0'
    assert config['mcp']['blocked_classes'] == ['DataBase', 'DServer']
    assert config['mcp']['blocked_functions'] == {'*': ['Init', 'Kill']}


@pytest.mark.skipif(os.name == 'nt', reason='POSIX process-group semantics')
def test_managed_command_stop_lets_the_launcher_shut_its_own_children_down(tmp_path):
    """Stop must reach device servers that live in their own sessions.

    ProcessManager gives every device server its own session, so signalling the
    launcher's process group never reaches them; only the launcher's SIGTERM
    handler stops them. This stands in a fake launcher that behaves the same way
    and asserts the grandchild dies with it.
    """
    marker = tmp_path / 'grandchild.pid'
    launcher = tmp_path / 'launcher.py'
    launcher.write_text(
        'import os, signal, subprocess, sys, time\n'
        'child = subprocess.Popen(\n'
        '    [sys.executable, "-c", "import time\\nwhile True: time.sleep(0.05)"],\n'
        '    start_new_session=True,\n'
        ')\n'
        f'open({str(marker)!r}, "w").write(str(child.pid))\n'
        'def shutdown(_signum, _frame):\n'
        '    child.terminate()\n'
        '    child.wait(timeout=5)\n'
        '    sys.exit(0)\n'
        'signal.signal(signal.SIGTERM, shutdown)\n'
        'print("ready", flush=True)\n'
        'while True: time.sleep(0.05)\n',
        encoding='utf-8',
    )

    command = shared.ManagedCommand(lambda _text: None, lambda _code: None)
    command.start([sys.executable, '-u', str(launcher)])
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.exists(), 'fake launcher never started its child'
    grandchild = int(marker.read_text())

    command.stop_and_wait(timeout=15)

    assert command.process.poll() is not None, 'launcher still running'
    # The launcher exited 0, so it ran its handler rather than being killed.
    assert command.process.returncode == 0
    with pytest.raises(ProcessLookupError):
        for _ in range(100):
            os.kill(grandchild, 0)
            time.sleep(0.05)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX process-group semantics')
def test_managed_command_reaps_device_servers_a_wedged_launcher_left_behind(tmp_path):
    """The force-kill path must still clear the stack.

    A launcher that ignores SIGTERM gets killed, so it never runs shutdown_all(),
    and its device servers sit in their own sessions where no group signal
    reaches them. The GUI has to reap them through the recorded PIDs, which is
    what ProcessManager.reap_all() does for the launcher too.
    """
    state_dir = tmp_path / '.processes'
    state_dir.mkdir()
    launcher = tmp_path / 'wedged.py'
    launcher.write_text(
        'import json, signal, subprocess, sys, time\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        'child = subprocess.Popen(\n'
        '    [sys.executable, "-c", "import time\\nwhile True: time.sleep(0.05)"],\n'
        '    start_new_session=True,\n'
        ')\n'
        f'open({str(state_dir / "run_servers.json")!r}, "w").write(json.dumps([child.pid]))\n'
        'print("ready", flush=True)\n'
        'while True: time.sleep(0.05)\n',
        encoding='utf-8',
    )

    command = shared.ManagedCommand(lambda _text: None, lambda _code: None)
    original_state_dir = process_manager.DEFAULT_STATE_DIR
    process_manager.DEFAULT_STATE_DIR = state_dir
    try:
        command.start([sys.executable, '-u', str(launcher)])
        state_file = state_dir / 'run_servers.json'
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not state_file.exists():
            time.sleep(0.05)
        assert state_file.exists(), 'fake launcher never recorded its child'
        orphan = int(json.loads(state_file.read_text())[0])

        command.stop_and_wait(timeout=2)
    finally:
        process_manager.DEFAULT_STATE_DIR = original_state_dir

    assert command.process.poll() is not None, 'wedged launcher survived'
    # SIGTERM was ignored, so it had to be killed -- and the device server it
    # spawned is only reachable through the recorded PID.
    assert command.process.returncode == -signal.SIGKILL
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(orphan, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError('orphaned device server survived the GUI stop')
