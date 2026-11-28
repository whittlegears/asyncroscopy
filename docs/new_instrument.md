# Adding your instrument

You need three things: an instrument class, the support devices it uses, and a YAML config. The startup scripts, the startup GUI and the MCP server find all three by themselves. You do not edit them.

Read the [architecture](architecture.md) page first for the names used here.

## 1. Pick the layer to start from

| Your instrument | Start from | Example |
|---|---|---|
| A new vendor for an existing family | The family base | `JeolMicroscope(ElectronMicroscope)` |
| A variant of an existing class | That class | `DigitalTwinTilt(DigitalTwin)` |
| A new kind of instrument with several vendors to come | A new family base on `Instrument` | `SPMMicroscope(Instrument)`, then `JupiterMicroscope(SPMMicroscope)` |
| A one-off instrument | `Instrument` directly | `Opentrons(Instrument)` |

Put the file in `asyncroscopy/instruments/<kind>/`. The startup GUI lists every `.py` file in those folders that calls `run_server(`.

## 2. Write the instrument class

A subclass of `Instrument` must implement four methods:

| Method | What it does |
|---|---|
| `read_instrument_type()` | Returns the modality, for example `"STEM"` or `"SPM"`. |
| `_init_device_attributes()` | Sets the starting values of your own attributes. |
| `_connect()` | Opens the vendor connection and the proxies to the support devices. Sets the device state. |
| `_disconnect()` | Closes the vendor connection. |

`Instrument.init_device` calls `_init_device_attributes` and then `_connect`. The `Connect` command calls `_connect` again. If `_connect` sets `self._connect_error`, `Connect` raises `ConnectFailed` with that text. Set the state to `FAULT` when the connection fails, so the device does not look connected.

A family base adds the public commands. Keep each public `@command` thin: validate the input, read settings from the support devices, call a `_helper`, return the result. Declare each `_helper` with `@abstractmethod` if every vendor must supply it. Give it a default if only some vendors support it. `Instrument` uses a metaclass that combines Tango's `DeviceMeta` and `ABCMeta`, so abstract methods work on Tango devices.

End the file with a server entry point:

```python
if __name__ == "__main__":
    MyInstrument.run_server()
```

### Properties

Read deployment values from Tango device properties, not from constants:

- `hardware_host`, `hardware_port` and `hardware_timeout_seconds` come from `instrument.hardware_host`, `instrument.hardware_port` and `instrument.timeout_seconds` in the YAML. Use these names.
- `<key>_device_address` is set for each support device in the YAML. Declare a `device_property` with that name for each device the instrument opens a proxy to.
- `data_device_address` and `testing_mode_bool` are on `Instrument` already.
- Any other property goes under `instrument.properties` in the YAML.

### Testing mode

`testing_mode_bool` is true when the tests run the device. In that mode, do not connect to hardware, and return `nan` or `False` from reads that need hardware.

## 3. Add support devices

A support device is a plain `tango.server.Device` that holds the settings for one part of the instrument. `SCAN` holds `dwell_time`, `imsize`, `scan_region` and `output_format`. It does not talk to hardware. The instrument reads those attributes through a `DeviceProxy` before it acquires.

A support device can also talk to hardware. Use a base class for the settings and a vendor subclass for the hardware, for example `STAGE` and `AutoScriptSTAGE`. The vendor subclass reads its own `hardware_host` and `hardware_port` from `devices.<key>.properties` in the YAML.

Each support device is one process. The runner starts it with `python -m <module_name> <key>_instance`. If one module holds several device classes, pick the class from the instance name, as `jupiter_api.py` does:

```python
_DEVICE_CLASSES = {"instrument": JupiterMicroscope, "scan": SCAN_Jupiter, "stage": STAGE_Jupiter}

if __name__ == "__main__":
    import sys
    key = sys.argv[1].rsplit("_instance", 1)[0]
    _DEVICE_CLASSES[key].run_server()
```

## 4. Write the YAML config

Put it in `configs/`. The startup GUI lists every YAML there that a startup script can run.

```yaml
instrument:
  file: asyncroscopy/instruments/my_kind/my_instrument.py
  class_name: MyInstrument          # blank: the class the file passes to run_server()
  description: "My instrument"
  hardware_host: "${ASYNCROSCOPY_MY_INSTRUMENT_HOST}"   # omit for a simulation
  hardware_port: 9095
  timeout_seconds: 120              # becomes hardware_timeout_seconds
  properties:
    any_other_property: value

devices:
  data:  { module_name: asyncroscopy.data.data }
  scan:  { module_name: asyncroscopy.instruments.my_kind.hardware.scan }
  stage:
    class_name: MySTAGE             # blank: the key upper-cased (STAGE)
    module_name: asyncroscopy.instruments.my_kind.hardware.stage_mine
    properties:
      hardware_host: "${ASYNCROSCOPY_MY_INSTRUMENT_HOST}"
  camera:
    module_name: asyncroscopy.instruments.my_kind.detectors.camera
    enabled: false                  # skipped, and removed from the database

tango:
  host: localhost
  port: 9094
  reset_database_file: false

tiled:
  host: localhost
  port: 9091
  acquisition_dir: outputs/tiled_acquisitions
  autostart: true                   # needs the data device
  register_on_startup: false

device_timeout_seconds: 120         # wait for each device to answer ping()
```

Write a machine's address as `"${NAME}"`, never as the address itself. At startup the name is filled in from an environment variable of that name, or else from `configs/site.env`. Git ignores that file; your team shares it privately. Add the name to `configs/site.env.example` so others know to set it. A test fails if a tracked file contains a lab address.

Rules the runner checks:

- A device key starts with a letter and uses only letters, digits and `_`. `instrument` is reserved.
- A device entry may only have `module_name`, `class_name`, `properties`, `enabled` and `description`.
- `instrument.file` must be inside the project folder.
- `tiled.autostart: true` needs an enabled `data` device. Set `autostart: false` if your instrument saves no data through Tiled.

The runner reports every problem in the config at once, before it starts anything.

## 5. Add a digital twin

Write a second vendor class that simulates the hardware, and a second YAML that points to it. Leave out `hardware_host` and `hardware_port`. A twin lets you test the full stack, the MCP server and agents with no hardware. See [digital twins](digital_twin.md) for the electron microscope twins.

## 6. Test it

Run the twin config first:

```bash
uv run startup_scripts/run_servers.py --yaml configs/my-instrument-twin.yaml
```

Add tests under `tests/`. `tests/conftest.py` starts the support devices and the instrument in one `MultiDeviceTestContext`, so no Tango database is needed. Use it as the template.

```bash
uv run pytest tests/ -v
```

## 7. Expose it to agents

There is nothing to do. When the stack is up, the [MCP server](MCP/mcp_server.md) turns every command of every exported device into an MCP tool. Block the commands an agent must not call before you connect one.
