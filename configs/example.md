# Server config example

`startup_scripts/run_servers.py` reads YAML files of this shape. Each file starts one stack. The full explanation is in [docs/new_instrument.md](../docs/new_instrument.md).

```yaml
instrument:
  file: asyncroscopy/instruments/my_kind/my_instrument.py
  class_name: MyInstrument          # blank: the class the file passes to run_server()
  description: "My instrument"
  hardware_host: localhost          # omit for a simulation
  hardware_port: 9095
  timeout_seconds: 120              # becomes hardware_timeout_seconds

devices:                            # each key becomes asyncroscopy/<key>/default
  data:
    module_name: asyncroscopy.data.data
  scan:
    module_name: asyncroscopy.instruments.electron_microscope.hardware.scan

tango:
  host: localhost
  port: 9094
  reset_database_file: false

tiled:
  host: localhost
  port: 9091
  acquisition_dir: outputs/tiled_acquisitions
  autostart: true                   # needs the data device

device_timeout_seconds: 120         # wait for each device to answer ping()
```

Start it with:

```bash
uv run startup_scripts/run_servers.py --yaml configs/my-instrument.yaml
```
