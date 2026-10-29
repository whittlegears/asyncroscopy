# Run your instrument beside the twin

At a hackathon, the STEM twin and your own instrument (simulated or real) often
run on the same machine. Each needs its own Tango database, its own ports, its
own Tiled folder and its own MCP endpoints. This page lists every one of them,
with the Raman example's ready configs as a working reference:

- `configs/yaml_sim_raman.yaml`: the device stack (Raman twin, DATA, Approvals).
- `configs/mcp_raman.yaml`: the agent endpoint.
- `configs/mcp_raman_operator.yaml`: the operator endpoint (approvals, for the GUI and benchmarks).

## What must differ from the twin

| What | Twin (`digital_twin_mapping.yaml`, `mcp_dt.yaml`, `mcp_operator.yaml`) | Raman example | Where it is set |
|---|---|---|---|
| Tango database port | 9094 | 9194 | `tango.port` in the stack config and in **both** MCP configs |
| Tango database file and process records | project root | `outputs/stacks/raman/` | `--isolated <dir>` when starting the stack |
| Saved approval mode (auto or approve) | `outputs/approvals/mode.json` | `outputs/stacks/raman/approvals/mode.json` | `mode_file` on the Approvals device, or `--isolated <dir>` (which puts it in that folder when the config sets none) |
| Tiled port | 9091 | 9191 | `tiled.port` |
| Acquisition folder (holds Tiled's catalog) | `outputs/tiled_acquisitions` | `outputs/raman_acquisitions` | `tiled.acquisition_dir`; two Tiled servers must not share one |
| Agent MCP endpoint | 8000 | 8100 | `mcp.http_port` in the agent MCP config |
| Operator MCP endpoint | 8001 | 8101 | `mcp.http_port` in the operator MCP config |
| Map analysis (pyTEMlib) | 8003 | none | `mcp.analysis_url`: empty, or a second pyTEMlib server on its own port pointed at your Tiled (`--tiled http://localhost:9191 --port 8103`) |
| A Mapper's own analysis server (a second **twin**, or any stack with a Mapper) | `http://127.0.0.1:8003/mcp` | no Mapper | `devices.mapper.properties.analysis_url` in the stack config, as well as `mcp.analysis_url`: left at 8003, the second stack's maps are stitched by the first stack's pyTEMlib, which reads the first stack's Tiled, with no error |
| Job state and traces | `outputs/jobs`, `outputs/trace/*.jsonl` | `outputs/stacks/raman/jobs`, `outputs/trace/raman_*.jsonl` | `state_directory` and `trace_path` instrument properties; `mcp.trace` |

Device names (`asyncroscopy/data/default` and so on) may stay the same: each
stack has its own Tango database, so they do not collide.

## Start it

```bash
uv run startup_scripts/run_servers.py --yaml configs/yaml_sim_raman.yaml --isolated outputs/stacks/raman
```

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp_raman.yaml
```

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp_raman_operator.yaml
```

Wait for "Leave this terminal open" from the first before starting the other two.

Without `--isolated`, the launcher refuses to start while the twin runs: both
would use the Tango database file in the project root. It also refuses, naming
the other stack, if anything else holds your config's ports. It never stops
another stack unless you add `--replace`. Restarting your own stack is fine:
leftovers of the same config (with the same `--isolated` folder) are cleared
as before.

## Point clients at it

| Client | Setting |
|---|---|
| Agents (`python -m asyncroscopy.agents.run`, LangGraph Studio) | `ASYNCROSCOPY_MCP_URL=http://127.0.0.1:8100/mcp`, and `ASYNCROSCOPY_ANALYSIS_URL` when the stack has its own pyTEMlib server (it defaults to 8003, the twin's) |
| `startup_scripts/check_stack.py`, `check_gui_contract.py` | `--agent http://127.0.0.1:8100/mcp --operator http://127.0.0.1:8101/mcp` (and `--tiled http://localhost:9191` for the GUI check; `--pytemlib` for map analysis) |
| Workflow benchmarks | declare the stack (its `config`, and `mcp`, `operator` and `tiled` addresses 8100, 8101, 9191) under `stacks:` in your workflow file; see `benchmarks/README.md` |
| Scripts that read data from Tiled directly | `ASYNCROSCOPY_TILED_URI=http://localhost:9191` |
| SciAgentGUI | add the two endpoints (8100 agent, 8101 operator) on its MCP Servers page; it reads the Tiled address from DATA |

The defaults (8000, 8001, 9091) point at the twin, so a client you forget to
set talks to the wrong instrument rather than failing.

## Stop it

Stop only your stack's processes: Ctrl+C in each terminal, or by PID. Starting
the stack in the background (`Start-Process ... -PassThru` in PowerShell) gives
you the PIDs; `taskkill /PID <pid> /T /F` stops a launcher and the device servers
it started. Avoid the "stop every asyncroscopy stack" command in the quickstart's
Troubleshooting while the twin should keep running.

## Check it

With the Raman stack up, the agent endpoint offers `RamanTwin_*` tools plus
`run_job`, `wait_for_job`, `list_data` and `view_data`. A first session:

1. `RamanTwin_laser_on` with `power_mw: 5`.
2. `RamanTwin_move_xy` with `x_m: 1e-4, y_m: 0` (the graphene flake is at x > 0).
3. `RamanTwin_acquire_raman_spectrum` returns a key; `view_data(key, format="text")` lists the Si band at 521 cm-1 and graphene's G (1582) and 2D (2690) bands.
4. `run_job(device="RamanTwin", plan={"axes": {"stage_x_m": {"start": -1e-4, "stop": 1e-4, "points": 21}}, "settings": {"exposure_s": 1}})`, then `wait_for_job` until it is done; its summary lists one spectrum key per point.
5. `RamanTwin_laser_off`.
