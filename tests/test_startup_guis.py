from startup_guis import mcp_gui, server_gui


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


def _server_values(devices: dict, enabled: dict) -> dict:
    return {
        'instrument': {
            'class_name': 'DigitalTwin',
            'file': 'asyncroscopy/instruments/electron_microscope/digital_twin.py',
        },
        'instrument_file': 'asyncroscopy/instruments/electron_microscope/digital_twin.py',
        'hardware_host': '',
        'hardware_port': '',
        'hardware_timeout_seconds': '120',
        'devices': devices,
        'enabled_devices': enabled,
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


def test_device_rows_come_from_the_yaml_in_file_order():
    config = {
        'devices': {
            'eels': {'module_name': 'asyncroscopy.instruments.electron_microscope.detectors.eels_gatan', 'properties': {'a': 1}},
            'camera': None,
            'data': {'module_name': 'asyncroscopy.data.data'},
        }
    }

    rows = server_gui.device_rows_from_config(config)

    assert [key for key, _ in rows] == ['eels', 'camera', 'data']
    assert dict(rows)['camera'] == {}
    assert dict(rows)['eels']['properties'] == {'a': 1}
    assert server_gui.device_rows_from_config({}) == []
    assert server_gui.device_rows_from_config({'devices': ['not', 'a', 'mapping']}) == []


def test_device_summary_names_class_and_module():
    assert server_gui.device_summary('stage', {'class_name': 'AutoScriptSTAGE', 'module_name': 'x.stage_autoscript'}) == 'AutoScriptSTAGE  (x.stage_autoscript)'
    assert server_gui.device_summary('camera', {'module_name': 'x.camera'}) == 'CAMERA  (x.camera)'
    assert server_gui.device_summary('camera', {}) == 'CAMERA  (module_name missing)'


def test_server_config_keeps_only_checked_yaml_devices():
    devices = {
        'eels': {'module_name': 'x.eels_gatan', 'properties': {'data_device_address': 'asyncroscopy/data/default'}},
        'flucam': {'module_name': 'x.flucam'},
        'data': {'module_name': 'asyncroscopy.data.data'},
        'stage': {'class_name': 'AutoScriptSTAGE', 'module_name': 'x.stage_autoscript'},
    }
    enabled = {'eels': True, 'flucam': False, 'data': True, 'stage': False}

    config = server_gui.server_config_from_values(_server_values(devices, enabled))

    assert list(config['devices']) == ['eels', 'data']
    assert config['devices']['eels'] == devices['eels']
    assert 'flucam' not in config['devices'] and 'stage' not in config['devices']


def test_server_config_starts_no_devices_when_everything_is_unchecked():
    devices = {'data': {'module_name': 'asyncroscopy.data.data'}}

    config = server_gui.server_config_from_values(_server_values(devices, {'data': False}))

    assert config['devices'] == {}
    assert config['instrument']['class_name'] == 'DigitalTwin'


def test_server_config_files_lists_only_server_style_configs(tmp_path):
    (tmp_path / 'twin.yaml').write_text('instrument: {file: a.py}\ndevices: {data: {module_name: m}}\n', encoding='utf-8')
    (tmp_path / 'notes.yml').write_text('instrument: {file: b.py}\ndevices: {}\n', encoding='utf-8')
    (tmp_path / 'mcp.yaml').write_text('tango: {host: localhost, port: 9094}\nmcp: {name: x}\n', encoding='utf-8')
    (tmp_path / 'llm.yaml').write_text('tango: {host: localhost}\nmcp_url: http://x\n', encoding='utf-8')
    (tmp_path / 'broken.yaml').write_text('instrument: [unclosed\n', encoding='utf-8')
    (tmp_path / 'README.md').write_text('not yaml', encoding='utf-8')

    assert server_gui.server_config_files(tmp_path) == ['notes.yml', 'twin.yaml']


def test_is_server_config_requires_instrument_and_devices():
    assert server_gui.is_server_config({'instrument': {'file': 'a.py'}, 'devices': {}})
    assert not server_gui.is_server_config({'instrument': 'not-a-mapping', 'devices': {}})
    assert not server_gui.is_server_config({'tango': {}, 'mcp': {}})
    assert not server_gui.is_server_config([])


def test_launcher_state_name_matches_process_manager_default():
    from startup_guis import shared

    assert shared.launcher_state_name(['uv', 'run', 'python', '-u', 'startup_scripts/run_servers.py', '--yaml', 'x.yaml']) == 'run_servers'
    assert shared.launcher_state_name(['uv', 'run', 'python', '-u', 'startup_scripts/run_mcp.py']) == 'run_mcp'
    assert shared.launcher_state_name(['uv', 'run', 'something']) is None
