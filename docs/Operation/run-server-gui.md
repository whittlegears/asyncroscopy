# Running Servers in a GUI

## How to Run the GUI

```bash
uv run startup_guis/server_gui.py
```

The MCP server has its own window:

```bash
uv run startup_guis/mcp_gui.py
```

Both windows use PyQt6 by default; set `ASYNCROSCOPY_QT_API=pyqt5` on older
Windows 10 machines where Qt6 cannot load.

## Picking a config

The dropdown in the title bar lists every file in `configs/` that
`startup_scripts/run_servers.py` can start, i.e. YAML files with both an
`instrument` and a `devices` section. MCP and LLM configs never appear there.
The window opens on `DigitalTwin.yaml` so a fresh launch never points at a real
instrument by accident. "Load config file" opens any other YAML; a file that is
not a server config is reported in the terminal pane and left unloaded.

## Devices come from the YAML

The **Devices** section is populated from the `devices:` section of the
selected config and rebuilt whenever another config is loaded. Every device the
file declares gets one checkbox, ticked by default; hovering a checkbox shows
the Tango class and Python module it would start.

Only ticked devices are written into the generated config that Start hands to
`run_servers.py` (`outputs/startup_configs/server_gui.yaml`). An unticked
device is therefore never registered in the Tango database and never started,
and the instrument is not given a `<device>_device_address` property for it.
The instrument itself is always started. The caption under the checkboxes
shows exactly which devices the next Start will launch.

## Start, Stop and closing the window

- **Start** writes the generated config and runs
  `uv run python -u startup_scripts/run_servers.py --yaml <generated config>`,
  streaming its output into the terminal pane.
- **Stop** shuts down everything that Start created: the launcher, the Tango
  database, every device server and the Tiled server. The launcher is first
  asked to unwind its own `ProcessManager` (so Tiled is stopped cleanly and the
  device servers are terminated in order); anything still alive after the grace
  period is force-killed, together with any PID the launcher's
  `.processes/run_servers.json` state file still lists.
- **Closing the window** (the close button, Ctrl+C or `kill` in the terminal
  that launched the GUI, or that terminal being closed) performs the same
  shutdown before the window disappears. The GUI also passes its own PID to
  the launcher, which stops itself the moment the GUI process is gone, so even
  a force-killed GUI does not leave servers behind. On Windows the launched
  tree is additionally placed in a kill-on-close job object.

## Verified operating systems

- Windows
- macOS
- Linux
