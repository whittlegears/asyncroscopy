---
name: digital-twin-operation
description: Operate the simulated DigitalTwin microscope safely - device names, tool naming, stage and beam commands.
version: 1.0.0
tags: [digital-twin, simulation, stage, beam, safety]
triggers: ["digital twin", "simulated microscope", "move the stage", "field of view"]
tools: ["list_devices", "DigitalTwin_*"]
---

# Operating the DigitalTwin

The DigitalTwin is a simulated STEM started with
`uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml`. Its
Tango devices are `asyncroscopy/instrument/default` (class `DigitalTwin`),
plus `camera`, `eds`, `scan`, `stage` and `data`.

## Tool naming

MCP tools are `<DeviceClass>_<command>`, so instrument commands are
`DigitalTwin_acquire_scanned_image`, `DigitalTwin_acquire_spectrum`,
`DigitalTwin_move_stage`, and so on. Call `list_devices` to see what is
exported; the two native tools `list_devices` and `get_data_from_key` are
always present.

## Common commands

| Task | Tool | Arguments |
|------|------|-----------|
| HAADF image | `DigitalTwin_acquire_scanned_image` | `{"detector_list": ["haadf"]}` |
| Camera image | `DigitalTwin_acquire_camera_image` | none |
| EDS spectrum | `DigitalTwin_acquire_spectrum` | `{"detector_name": "eds"}` |
| Read stage | `DigitalTwin_get_stage` | none, returns `[x, y, z, alpha, beta]` |
| Move stage | `DigitalTwin_move_stage` | float array `[x, y, z, alpha, beta]` (absolute) |
| Field of view | `DigitalTwin_set_fov` / `DigitalTwin_get_fov` | float |
| Beam position | `DigitalTwin_place_beam` | float array `[x, y]` in fractional image coordinates (0..1) |
| Blank/unblank | `DigitalTwin_blank_beam` / `DigitalTwin_unblank_beam` | none |
| Defocus | `DigitalTwin_set_defocus` / `DigitalTwin_get_defocus` | float |

Always read the current value (`get_stage`, `get_fov`) before changing it and
move in small steps; report the values you read back after a move.

## Safety

- `Init`, `Kill` and `RestartServer` are blocked on every device. Never try to
  work around that.
- Do not call the `LLM` device from inside the agent (it is blocked in the MCP
  config to avoid recursion).
- If a tool returns an error, report it verbatim and stop rather than retrying
  blindly.
