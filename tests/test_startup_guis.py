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


def test_is_server_config_accepts_server_configs_and_rejects_others(tmp_path):
    assert server_gui.is_server_config(server_gui.CONFIG_DIR / 'DigitalTwin.yaml') is True
    assert server_gui.is_server_config(server_gui.CONFIG_DIR / 'mcp.yaml') is False
    assert server_gui.is_server_config(tmp_path / 'missing.yaml') is False
    not_yaml = tmp_path / 'broken.yaml'
    not_yaml.write_text('{unbalanced', encoding='utf-8')
    assert server_gui.is_server_config(not_yaml) is False


def test_server_gui_config_loads_with_run_servers_loader(tmp_path):
    """The GUI-generated YAML must satisfy run_servers' own loader: the script
    is the source of truth for the config contract, and Start validates with it."""
    from startup_scripts import run_servers

    config = server_gui.server_config_from_values(
        {
            'instrument': {
                'class_name': 'DigitalTwin',
                'file': 'asyncroscopy/instruments/electron_microscope/digital_twin.py',
                'description': 'Twin',
            },
            'instrument_file': 'asyncroscopy/instruments/electron_microscope/digital_twin.py',
            'hardware_host': '',
            'hardware_port': '',
            'hardware_timeout_seconds': '120',
            'devices': {
                'data': {'module_name': 'asyncroscopy.data.data'},
                'corrector': {
                    'class_name': 'TestCorrector',
                    'module_name': 'asyncroscopy.instruments.electron_microscope.hardware.TestCorrector',
                },
            },
            'enabled_devices': {'data': True, 'corrector': True},
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
    path = tmp_path / 'generated.yaml'
    import yaml as yaml_module

    path.write_text(yaml_module.safe_dump(config), encoding='utf-8')

    loaded = run_servers.load_config(path)
    assert loaded.instrument.class_name == 'DigitalTwin'
    assert sorted(device.key for device in loaded.support_devices) == ['corrector', 'data']
    assert loaded.tango_host == 'localhost'


def test_mcp_gui_config_loads_with_run_mcp_loader(tmp_path):
    """Same contract for the MCP GUI: its YAML must satisfy run_mcp.load_config."""
    from startup_scripts import run_mcp

    config = mcp_gui.mcp_config_from_values(
        {
            'tango_host': 'localhost',
            'tango_port': '9094',
            'name': 'Twin_MCP',
            'transport': 'streamable-http',
            'http_host': '127.0.0.1',
            'http_port': '8000',
            'data_device_address': 'asyncroscopy/data/default',
            'quiet': True,
            'blocked_classes': 'DataBase, DServer',
            'blocked_functions': '"*":\n  - Init\n',
        }
    )
    path = tmp_path / 'generated.yaml'
    import yaml as yaml_module

    path.write_text(yaml_module.safe_dump(config), encoding='utf-8')

    loaded = run_mcp.load_config(path)
    assert loaded.mcp.name == 'Twin_MCP'
    assert loaded.tango_host == 'localhost'
    assert loaded.mcp.blocked_functions == {'*': ['Init']}


def test_mcp_gui_warning_pattern_extracts_skipped_device_count():
    line = 'MCP WARNING: 2 device(s) failed tool discovery after 3 attempts - their commands are NOT registered as tools:'
    match = mcp_gui.MCP_WARNING_PATTERN.search(line)
    assert match is not None
    assert int(match.group(1)) == 2
    assert mcp_gui.MCP_WARNING_PATTERN.search('MCP ready: 58 tool(s) registered') is None
