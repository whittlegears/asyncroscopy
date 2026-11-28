from startup_guis import mcp_gui, server_gui
from startup_scripts import run_mcp, run_servers


def test_server_gui_builds_server_yaml():
    config = run_servers.server_config_from_values(
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
    config = run_servers.server_config_from_values(
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
    config = run_mcp.mcp_config_from_values(
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


def test_the_gui_offers_every_device_and_instrument_the_configs_define(tmp_path):
    (tmp_path / "first.yaml").write_text(
        "instrument: {file: asyncroscopy/instruments/electron_microscope/digital_twin.py}\n"
        "devices:\n  stage: {module_name: first.stage}\n  data: {module_name: asyncroscopy.data.data}\n"
    )
    (tmp_path / "other.yaml").write_text(
        "instrument: {file: asyncroscopy/instruments/scanning_probe_microscope/jupiter_api.py}\n"
        "devices:\n  stage: {module_name: other.stage}\n  new_detector: {module_name: lab.new_detector}\n"
    )
    (tmp_path / "broken.yaml").write_text("devices: [not, a, mapping")
    first = tmp_path / "first.yaml"

    devices = run_servers.known_devices(first)
    assert devices == {
        "stage": {"module_name": "first.stage"},
        "data": {"module_name": "asyncroscopy.data.data"},
        "new_detector": {"module_name": "lab.new_detector"},
    }
    assert run_servers.known_instrument_files(first) == [
        "asyncroscopy/instruments/electron_microscope/digital_twin.py",
        "asyncroscopy/instruments/scanning_probe_microscope/jupiter_api.py",
    ]


def test_the_shipped_configs_name_instrument_files_that_exist():
    for file in run_servers.known_instrument_files(run_servers.PROJECT_DIR / "configs" / "DigitalTwin.yaml"):
        assert (run_servers.PROJECT_DIR / file).is_file(), file


def test_the_guis_define_no_config_logic_of_their_own():
    for name in ("server_config_from_values", "class_name_from_file", "project_path_text", "uses_hardware_connection"):
        assert getattr(server_gui, name, None) in (None, getattr(run_servers, name))
    for name in ("mcp_config_from_values", "parse_int_safe"):
        assert getattr(mcp_gui, name, None) in (None, getattr(run_mcp, name))
    assert not hasattr(server_gui, "DEVICE_MODULES") and not hasattr(server_gui, "INSTRUMENT_FILES")
