# Running Servers in a GUI

```bash
uv run startup_guis/server_gui.py   # Tango database, devices, Tiled (run_servers.py)
uv run startup_guis/mcp_gui.py      # MCP bridge (run_mcp.py)
uv run startup_guis/llm_gui.py      # agent MCP server (run_llm.py)
```

The server GUI opens on `configs/DigitalTwin.yaml` and the MCP GUI on the digital-twin
Tango host, so a fresh window never targets the real microscope.

## The GUI has no rules of its own

Each window is drawn from its startup script. The script declares its settings once
(`SETTINGS` in `run_servers.py`, `run_mcp.py`, `run_llm.py`), and the same list both
validates the YAML on the command line and builds the window. So:

- **New option:** add a `Setting(...)` to the script. It appears in the GUI.
- **New device:** add it to the YAML's `devices:`. It appears as a checkbox.
- **New config:** drop it in `configs/`. It appears in the dropdown if its script accepts it.

The GUI edits the loaded YAML in place and writes back only what you changed, so keys
without a widget (device `properties`, `tool_log`, anything added later) are kept.
Before **Start**, it runs the script's own `load_config()` on the result. The pane
under the preview shows the same error the script would print, and nothing starts
until the config is valid.

## Devices

Any device is accepted if it is properly defined:

```yaml
devices:
  robot:
    module_name: lab_devices.robot   # must resolve to a .py file
    class_name: PipettingRobot       # optional: defaults to the class passed to run_server()
    properties: {speed: 3}           # optional Tango device properties
    enabled: false                   # optional: keep it in the file but do not start it
```

- An unticked checkbox writes `enabled: false`. That device is not registered or
  started, and the instrument gets no address for it. With a kept database, an earlier
  registration is removed.
- A disabled device is only shape-checked, so an unfinished entry never blocks startup.
- Typos (`modul_name`), missing modules, a `class_name` the module does not define,
  and the reserved name `instrument` are refused before anything starts, with every
  problem listed together.

## Stop

**Stop** interrupts the script the way Ctrl+C does, so it stops its own device servers,
Tiled and Tango database in order. If the script has not exited after 30 s, the GUI
kills it and runs it again with `--cleanup` to stop anything left behind. Closing the
window stops the script the same way. If the GUI itself dies, the script notices
(`--parent-pid`) and shuts down within about a second.
