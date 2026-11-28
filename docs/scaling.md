# Scaling

Horizontal scaling means more instruments, more stacks and more computers. Vertical scaling means more capability inside one stack: more devices, more commands and heavier compute.

## Horizontal: more instruments

### One stack per instrument

Every stack uses the same device names (`asyncroscopy/<key>/default`). So each instrument gets its own Tango database. Give each stack its own Tango port and Tiled port, then start each one from its own config:

```bash
uv run startup_scripts/run_servers.py --yaml configs/microscope-a.yaml
uv run startup_scripts/run_servers.py --yaml configs/afm-b.yaml
```

If both stacks run on different computers, nothing else is needed. Each computer runs its own database on its own host.

### Two stacks on one computer

Add `--isolated DIR` to the second stack, and use different `tango.port` and `tiled.port` values in its YAML:

```bash
uv run startup_scripts/run_servers.py --yaml configs/twin-b.yaml --isolated outputs/stack-b
```

With `--isolated`, the runner keeps the Tango database file and the process state in `DIR`. It does not clear ports, kill other device servers or delete the shared database file. So the first stack keeps running.

### Split one stack across computers

The parts of a stack talk over the network, so each part can run where it fits best:

| Part | Where it usually runs | Setting that points to it |
|---|---|---|
| Tango database and devices | The instrument control PC | `tango.host`, `tango.port` |
| Vendor API server | The instrument | `instrument.hardware_host`, `hardware_port` |
| Tiled | Next to the saved files | `tiled.host`, `tiled.port` |
| MCP server | Any computer that reaches the Tango database | `tango.host` in the MCP YAML |
| Agent | A GPU computer | `agent.mcp_urls` |

Bind a server to `0.0.0.0` when clients on other computers must connect. Use `127.0.0.1` for local clients only.

### One agent, many instruments

Run one MCP server per stack. Each one reads its own Tango database. Then list all of them in the agent config:

```yaml
agent:
  mcp_urls:
    - "http://microscope-pc:8000/mcp"
    - "http://afm-pc:8000/mcp"
```

The agent takes the tools of every server in the list. The MCP server names a tool `<TangoClass>_<command>`. Two stacks that run the same instrument class therefore give the agent two tools with the same name. Use a different subclass or `include_only_functions` to keep the names apart.

### Startup scripts and the GUI

`startup_scripts/registry.py` finds every `startup_scripts/run_*.py` that has `DEFAULT_CONFIG_PATH`, `SETTINGS`, `is_config`, `load_config` and `main`. The contract is in `startup_scripts/config_schema.py`. The startup GUI (`startup_guis/script_gui.py`) builds its form from the YAML it opens and the `SETTINGS` of the script that runs it. A new config, device or script appears in the GUI with no GUI change. Labels in `startup_guis/ui_hints.py` are optional.

## Vertical: more capability per stack

### More devices

Add a support device and a `<key>_device_address` property on the instrument (see [adding your instrument](new_instrument.md)). Switch a device off with `enabled: false`. The runner then skips it and removes it and its address property from the database, so a stale entry from an earlier run does not stay.

### More commands

Add a `@command` to a device. The MCP server picks it up the next time it starts. For a family, put the public command on the family base and a `_helper` on each vendor, so every vendor gets the command.

### More vendors and twins

Each vendor and each twin is one more subclass and one more YAML. The digital twins show this: `DigitalTwinTilt`, `DigitalTwinParticles` and `DigitalTwinDiffraction` all extend `DigitalTwin` and change only what they simulate.

### Heavier compute as a device

Put slow or GPU work in its own Tango device, so it can run on another computer against the same stack. The segmentation device is the example. `startup_scripts/run_segmentation.py` starts only that device against a stack that already runs Tango, DATA and Tiled:

```bash
uv run --extra segment startup_scripts/run_segmentation.py --yaml configs/Segmentation.yaml
```

`configs/Segmentation.yaml` sets `compute_device: cuda`. The launcher fails at startup if CUDA is not available. It does not fall back to the CPU.

### Limits to know

- Each device is one process. A stack with many devices starts many processes. The runner waits for each one, up to `device_timeout_seconds`.
- Tango commands block by default. Set a long client timeout for slow acquisitions. The runner sets 120 s on the DATA proxy.
- Async acquisition is not implemented yet. See [upcoming changes](upcoming_changes.md).
