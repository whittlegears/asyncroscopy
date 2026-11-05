# Running The Servers

`startup_scripts/run_servers.py` starts the Tango/device side of asyncroscopy. It
clears stale processes, starts the Tango database, registers devices, launches
device servers, starts the DATA-managed Tiled HTTP server, and starts the
selected instrument last.

MCP is started separately with `startup_scripts/run_mcp.py`; see
[mcp_server.md](../MCP/mcp_server.md).

## TL;DR

```bash
uv run startup_scripts/run_servers.py

uv run startup_scripts/run_servers.py --yaml configs/Spectra300.yaml
uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml
uv run startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml
```

GUI:

```bash
uv run python startup_guis/server_gui.py
```

- Press **Enter** at prompts to accept the value in brackets.
- Leave the terminal open while you work. Press **Ctrl+C** to stop the managed
  processes and the managed Tiled server.
- Start MCP in a second terminal (on a twin, or on another computer):

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

On the Spectra 300, follow [first_light_runbook.md](first_light_runbook.md): it
adds the read-only pre-flight (`startup_scripts/preflight_hardware.py`), the
operator endpoint and the analysis server, in that order.

## What It Starts

| Order | Device(s) | Tango name |
|-------|-----------|------------|
| 1 | support devices: every key under `devices:` in the YAML | `asyncroscopy/<key>/default`; for `configs/Spectra300.yaml`: `asyncroscopy/{approvals,aperture,camera,corrector,data,eds,mapper,tuner,scan,stage}/default` |
| 2 | Tiled HTTP server | started through the `data` device |
| 3 | instrument | `asyncroscopy/instrument/default` |

The instrument starts last because it depends on the support devices. "All
asyncroscopy servers are ready" means every device answered a ping, not that
the instrument reached the hardware: a device that failed to connect to
AutoScript is in FAULT. On hardware, run `startup_scripts/preflight_hardware.py`
next; it reads every device's State and Status. The runner
writes support-device addresses and hardware connection values into Tango
database properties before the instrument starts.

## Server GUI

`startup_guis/server_gui.py` is a small YAML launcher for the server stack. It
does not start servers directly. It formats the current selections into YAML,
writes that YAML to `outputs/startup_configs/server_gui.yaml`, and runs:

```bash
uv run python startup_scripts/run_servers.py --yaml outputs/startup_configs/server_gui.yaml
```

The GUI includes:

- YAML preview generated from the current selections.
- Terminal output from the running process.
- **Start** and **Stop** controls.
- **Save current config** to write a YAML file you can reuse later.

## Configs

Server startup configs live in [configs/](../../configs):

| File | For |
|------|-----|
| [configs/Spectra300.yaml](../../configs/Spectra300.yaml) | The real Spectra 300 stack. This is the default config. |
| [configs/ThinkPad-utkarsh-covalent-setup.yaml](../../configs/ThinkPad-utkarsh-covalent-setup.yaml) | A localhost-everywhere setup for local testing. |

Each server config has:

- `instrument:` for the selected instrument class, Python file, and hardware connection values.
- `devices:` for support device modules.
- `tango:` for the Tango database host, port, and optional database-file reset.
- `tiled:` for the DATA-managed Tiled HTTP server.
- `device_timeout_seconds:` for device readiness waits.

Device `class_name` defaults to the upper-cased key (`scan` becomes `SCAN`).
`instrument.hardware_host`, `instrument.hardware_port`, and
`instrument.timeout_seconds` become instrument device properties.

Set `tango.reset_database_file: true` to delete a stale local
`tango_database.db` / `Tango_database.db` before starting the Tango database.
The GUI exposes the same setting as **Delete tango_database.db before start**.

## MCP

MCP has its own config: [configs/mcp.yaml](../../configs/mcp.yaml).

Start the server stack first. Then, from the MCP machine:

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

If MCP runs on a different computer, edit `configs/mcp.yaml`:

- `tango.host` should point to the machine running the Tango database.
- `mcp.http_host` should be `127.0.0.1` for local-only clients. Every shipped
  MCP config defaults to `127.0.0.1`.
- On a hardware stack (`simulated: false`), a LAN address or `0.0.0.0` needs
  both `mcp.allow_remote_bind: true` and an `mcp.auth` section (bearer tokens,
  [remote_access.md](remote_access.md)). Without `allow_remote_bind` the endpoint
  binds `127.0.0.1` instead; with it but without `mcp.auth`, `run_mcp.py` refuses
  to start. On a simulated stack a remote bind is allowed without tokens, so use
  it only on a trusted network.
- Never open the operator endpoint (`configs/mcp_operator.yaml`) to other
  machines: it approves held calls and has no login.

## Prompts

Interactive mode is used only when `--yaml` is omitted. The defaults come from
`configs/Spectra300.yaml`.

| Prompt | What it controls |
|--------|------------------|
| Tango database host / port | `TANGO_HOST` for Tango clients and servers. |
| Tiled HTTP host / port | Where the DATA device starts Tiled. |
| Acquisition save path | Directory written and served by Tiled. |
| Start Tiled HTTP server | Whether DATA starts its managed Tiled server. |
| Clear old processes first | Frees stale Tango/Tiled ports and old device servers. |
| Start Tango database | Starts the DB or waits for an existing one. |
| Register devices | Adds device entries and instrument properties to the DB. |
| Device startup timeout seconds | How long to wait for each device to answer `ping()`. |
| Hardware host / port / timeout seconds | Connection values for the selected instrument. |

## Startup Stages

1. **Clearing old processes** stops whatever is LISTENING on the database and
   Tiled ports (clients connected to them are left alone), and stops leftover
   device servers this config starts: processes running exactly
   `-m <module> <key>_instance` for one of its devices, and only those whose
   `TANGO_HOST` matches this launch when the OS lets it be read. Other
   asyncroscopy stacks are not touched. It uses psutil, falling back to
   `Get-NetTCPConnection`/`Get-CimInstance` on Windows and `lsof`/`ps` elsewhere.
2. **Starting Tango database** starts or waits for the database server.
3. **Registering devices** writes device entries and instrument properties.
4. **Starting device servers** starts support devices, Tiled, and then the
   selected instrument.
5. **Startup summary** prints `TANGO_HOST`, PIDs, ready times, and the Tiled URI.

## Manual Fallback

The runner automates this database-mode flow. It starts each child with
`uv run --no-sync python -u ...` (uv located through the `UV` variable that
`uv run` sets, or `PATH`), or with the launcher's own Python when uv cannot be
found:

```bash
TANGO_HOST=localhost:9094 uv run python -m tango.databaseds.database 2

export TANGO_HOST=localhost:9094
uv run python -m asyncroscopy.instruments.electron_microscope.hardware.scan scan_instance
uv run python -m asyncroscopy.instruments.electron_microscope.auto_script instrument_instance
```

In PowerShell:

```powershell
$env:TANGO_HOST = "localhost:9094"
uv run python -m tango.databaseds.database 2

# in each further terminal
$env:TANGO_HOST = "localhost:9094"
uv run python -m asyncroscopy.instruments.electron_microscope.hardware.scan scan_instance
uv run python -m asyncroscopy.instruments.electron_microscope.auto_script instrument_instance
```

Manual device startup requires the devices to already be registered in Tango.
Let `run_servers.py` do registration once, then stop and relaunch individual
servers as needed.
