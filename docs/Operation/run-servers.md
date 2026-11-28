# Running the servers

`startup_scripts/run_servers.py` starts one stack: the Tango database, the support devices, the DATA-managed Tiled server and the instrument. Start the MCP server and the agent after it, in other terminals or on other computers.

## Quick start with no hardware

```bash
uv sync
uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml
```

Leave the terminal open while you work. Press Ctrl+C to stop every process the runner started, Tiled included.

Then, in a second terminal, start the MCP server for the twin:

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
```

For a real instrument, use its config, for example `configs/Spectra300.yaml`.

## Modes

- With `--yaml`, the runner starts headless from that file and asks nothing.
- Without `--yaml`, the runner asks for each value. Press Enter to accept the value in brackets. The defaults come from `configs/Spectra300.yaml`.

## Options

| Option | What it does |
|---|---|
| `--yaml PATH` | Config to start from. |
| `--debug` | Writes each server's output to `output_tango_devices_logs/<timestamp>/<key>.log`. |
| `--isolated DIR` | Runs a second stack beside an existing one. See [scaling](../scaling.md). |
| `--cleanup` | Only stops processes that an earlier run of this config left behind, then exits. |

## Startup stages

1. Clear old processes. The runner frees the Tango and Tiled ports and stops old device servers.
2. Start the Tango database, or wait for an existing one.
3. Register the devices and write the instrument properties.
4. Start the support devices, then Tiled, then the instrument.
5. Print a summary: `TANGO_HOST`, PIDs, ready times and the Tiled URI.

If a stage fails, the runner prints the last output of each server and stops what it started.

Set `tango.reset_database_file: true` to delete a stale `tango_database.db` before the database starts.

## The startup GUI

```bash
uv run startup_guis/server_gui.py
uv run startup_guis/mcp_gui.py
uv run startup_guis/llm_gui.py
```

All three open the same window (`startup_guis/script_gui.py`) on a different config. The server window opens on `DigitalTwin.yaml`, so a new window never targets real hardware. The config list shows every YAML in `configs/` that a startup script can run.

The window shows a form built from the YAML, a preview of the YAML it will run, and the terminal output. Start writes the current form to `outputs/startup_configs/<script>_gui.yaml` and runs the script on it. Stop stops the script and its processes. Save current config writes the form to a YAML you can reuse.

## Process handling

`ProcessManager` (`asyncroscopy/utils/process_manager.py`) starts and stops every child process for `run_servers.py`, `run_mcp.py` and `run_llm.py`. Use it if you write a new startup script:

```python
with ProcessManager() as manager:
    manager.start_process(key="scan", label="SCAN", command=[...], env=environment)
```

- It records the PIDs in a state file under `.processes/`. At the next start, it stops processes that a crashed run left behind.
- `stop_process` sends a graceful stop first. It kills the process if it does not stop within `timeout`.
- On exit from the `with` block, `shutdown_all()` stops all tracked processes.
- It handles the differences between Windows and POSIX.

## Manual start

The runner automates this flow:

```bash
TANGO_HOST=localhost:9094 uv run python -m tango.databaseds.database 2

export TANGO_HOST=localhost:9094
uv run python -m asyncroscopy.instruments.electron_microscope.hardware.scan scan_instance
uv run python -m asyncroscopy.instruments.electron_microscope.auto_script instrument_instance
```

A manual start needs the devices to be in the Tango database already. Let `run_servers.py` register them once. Then you can stop and start single servers by hand.
