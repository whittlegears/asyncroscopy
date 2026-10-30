# Approval mode

asyncroscopy has one switch that decides whether agents need a person's approval to act on the instrument. It lives in the `Approvals` Tango device, so it covers every agent: the server-side LangGraph agents, LangGraph Studio, SciAgentGUI and any other MCP client.

| Mode | What happens to a high-risk agent call |
| :-- | :-- |
| `auto` (default) | A routine command (imaging, spectra, stage moves, placing the beam) or one an operator switched on runs, and the Instrument Trace records `approved_by: "auto"`, so the record says no person looked at it. Any other high-risk command is held exactly as in `approve` until an operator switches it on (below). |
| `approve` | It does not run. The MCP server files a request and tells the agent `APPROVAL REQUIRED ... request_id=...`. Once an operator approves, the agent repeats exactly the same call and it runs, once. |

Notebooks and scripts talk to Tango directly and are never held, in either mode.

## What counts as high risk

`asyncroscopy/safety/policy.py` decides, in this order (its docstring is the authority):

1. Your own patterns from `mcp.safety.risk` in the MCP config.
2. **Low:** safe-stop commands: a name with exactly one stop verb (`abort`, `stop`, `halt`, `pause`, `estop`, `retract`, `withdraw`, `disengage`, or `off` as the last word; `emergency` may qualify it) where every other word is something that verb may stop. Stop and retreat verbs take hardware: `beam`, `emission`, `lamp`, `light`, `laser`, `shutter`, `gun`, `filament`, `hv`, `tip`, `probe`, `stage`, `x`, `y`, `z`, `xy`, `xyz`, `motor(s)`, `indenter`, `head`, `objective`, `robot`, `xray`, `source`, `all`, `now` (`retract` and `withdraw` only `tip`, `probe`, `head`, `z`, `indenter`, `objective` or `needle`, named; bare only on a device that also has an `approach`, `engage` or `indent` command, since a syringe pump's `withdraw` aspirates; `disengage` only `tip`, `probe` or `head`, named); `abort`/`stop`/`halt`/`pause`/`estop` may also take a motion or job (`approach`, `move`, `motion`, `movement`, `scan`, `scanning`, `acquisition`, `acquire`, `job`, `run`, `travel`, `focus`, `autofocus`, `live`, `indent`, `imaging`, `exposure`, `protocol`). `off` needs one of these before it, so a bare `Off` is held; it switches off only a source of energy (`beam`, `emission`, `lamp`, `light`, `led`, `illumination`, `laser`, `gun`, `filament`, `hv`, `xray`, `source`), so `stage_off` is held. The pairs close + shutter, blank + beam and safe + state also count. This is an allow-list: anything else is held, so `brake_off`, `vacuum_off`, `retract_approach`, `stop_retract`, `close_valve` and `estop_ack` are held. `asyncroscopy/safety/policy.py` has the exact lists. `server.list_tool_risks()` (see `asyncroscopy.testing`) shows how each of your commands is classed, and the MCP server prints a `[safety]` line at start-up for any stop-like command it still holds; mark such a stop low with `mcp.safety.risk`, as `"Class.command"` or the tool name (`"OptiScope_park_stage": "low"`). It also prints a line for an override that matches no command. So `Mapper_abort`, `blank_beam` and `stop_scan` are never held; a bare `Off` is (on a stage it drops holding torque).
3. **Low:** `reserve`, `job_status`, `results`, `release` on job devices.
4. **Low:** Tango's `State` and `Status`; `get_*` and `list_*` (read-only by convention); `Approvals.request_status`; and, on a device with both `acquire_scanned_image` and `blank_beam` (an electron microscope) only, `set_fov` and `set_defocus` (`place_beam` is high: with the beam unblanked it parks the probe on one spot).
5. **High:** everything else. That includes `move_stage`, every `acquire*`, `unblank_beam`, starting a job (`*.start`) and DATA's `register_path` (with `view_data` it would read any file). A new or renamed command is held in `approve` mode rather than waved through.

Add your own patterns in front of these with `mcp.safety.risk` in the MCP config, for example `{"MyStage.jog": "high", "MyCamera.preview": "low"}`.

## What auto mode holds

The agent can see and call every command the agent endpoint offers. In `auto`, a high-risk call runs without asking only if:

* its command is routine on an electron microscope (a device with both `acquire_scanned_image` and `blank_beam`): `policy.AUTO_DEFAULT` lists `move_stage`, `acquire_scanned_image`, `acquire_scanned_data_advanced`, `acquire_spectrum`, `acquire_eels_spectrum`, `place_beam`, `place_beam_list`, `unblank_beam`, `auto_focus`, `set_stigmator`, `set_beam_tilt` and `set_image_shift`. On any other instrument (an AFM, a YAML twin, an indenter) no high-risk command is routine: switch on the ones your team trusts; or
* an operator switched it on: `Approvals.set_auto_allowed({"patterns": [...], "who": ...})` takes the whole list of tool names or globs (`"DigitalTwin_set_column_valves"`, `"*_set_screen"`), records the change in the trace and saves it in `mode_file` beside the mode. `[]` goes back to the defaults. It is operator-only. `get_auto_allowed` reads the list; SciAgentGUI's Settings, Agent permissions, shows one switch per held command, and its "Always allow" button on a held request adds that command.

Everything else waits for a person, including column valves, the screen, apertures, the corrector, spectrometer set-up, calibration, camera exposures and any command the policy has not seen before. A job's start (`<device>_start`, and so `run_job`) runs in auto when every high-risk command the reserved plan drives would: the Mapper and the Tuner record them in the resolved plan as `drives` ({device address: commands}); a caller's plan may not contain `drives`. A job device that does not say what it drives is held in auto unless its start is switched on. A Tuner run with the `corrector` strategy drives the corrector's `correct_aberration`, so it is held until that is switched on; `image_metric` and `optistem` runs start on their own.

Two tools on every endpoint let a model reach any of them without carrying every schema: `find_tools(query)` lists or searches all tools, each with `"when": "runs"` or `"held"`, and `call_tool(name, arguments)` runs one through the same gate (and, on an endpoint with `mcp.auth`, needs the token role of the tool it runs). `find_tools` reports `mode: "unknown"` when the Approvals device does not answer; high-risk calls are refused then. `endpoint_info.auto` gives each high-risk tool's default treatment (`routine`, `held` or `job`).

## Two endpoints

| Config | Port | Who connects | Offers |
| :-- | :-- | :-- | :-- |
| `configs/mcp_dt.yaml` (agent) | 8000 | agents | every device command except operator-only ones |
| `configs/mcp_operator.yaml` (operator) | 8001, 127.0.0.1 only | SciAgentGUI | also `Approvals.approve`, `deny`, `set_mode` and the twin's `export_ground_truth` |

**What approve mode does and does not hold.** It holds agents whose only route to the instrument is the agent MCP endpoint. It does not hold anything with direct Tango access (port 9094) or a shell on the stack's machine: such a program can call devices directly, or the operator endpoint (unauthenticated, local only). Run unattended agents where their only access is the agent endpoint.

The agent endpoint never offers operator-only commands (approve, deny, set_mode, the grant-consuming `check`, `request`, `list_grants`, the twin's ground truth, `Init`/`Kill`/`RestartServer` on any device, and running the data service: DATA `configure`, `start_tiled_server`, `stop_tiled_server`), whatever its config says. Without this an agent could approve its own requests, because caller identity is not yet authenticated (the `X-Asyncroscopy-Caller` header is recorded, not checked; ATHENA WG1 owns tokens). Calls on the operator endpoint are never held: they come from the person at the GUI.

## Map runs

A map is one high-risk call, `Mapper_start`, so `approve` mode asks once per run, not once per tile. The approval covers the reserved plan: its stage box, tile count and dose budget. The request the operator sees carries that plan and its estimate (from `get_reservation`), not just a reservation id. In both modes the `Mapper` refuses to move outside its plan box, `reserve` refuses a plan that would exceed its dose budget, and the beam is blanked whenever a job ends. `Mapper_abort` is always allowed.

## Switching the mode

From SciAgentGUI's safety settings, or from Python:

```python
import json, tango
approvals = tango.DeviceProxy("asyncroscopy/approvals/default")
approvals.set_mode(json.dumps({"mode": "approve", "who": "your name"}))
print(approvals.list_pending())
approvals.approve(json.dumps({"request_id": "...", "who": "your name"}))
```

A request held in auto mode says why in the agent's message (`... held in auto mode until an operator switches it on`).

Each pending request names the tool (`action`), its `arguments`, the `caller`, the agent `run` that made it (the agent's run id, or null for a call from outside an agent run) and a `reason` (for a job start, the reserved plan). An operator interface can use `run` to offer "approve for the rest of this run": it approves each later request with the same `run` as it arrives, and every approval is still single use and traced.

Unanswered requests expire after `request_timeout_s` (default 30 minutes); an approval must be used within `grant_ttl_s` (default 15 minutes).

The mode is saved to the Approvals device's `mode_file` (default `outputs/approvals/mode.json`) and restored when the device restarts, so a crash or `Init` does not drop `approve` back to `auto`. If that file exists but cannot be read, the device starts in `approve`.

If the `Approvals` device cannot be reached, high-risk agent calls are refused (the server cannot tell whether a person must approve) and low-risk calls still run.

An agent endpoint must name its Approvals device (`mcp.safety.approvals_device_address`); `run_mcp.py` refuses to start one that does not, unless its config says `mcp.safety.allow_ungated: true` (for a simulator). The hardware configs (`Spectra300.yaml`, `Spectra300eels.yaml`, `covalent-*.yaml`) start their Approvals device with `default_mode: approve`, so on real hardware every high-risk agent call waits for a person. It stays that way: `auto` is refused on a stack that is not simulated (below).

**Before an unattended agent run on a real instrument, switch to `approve`.** `auto` is the default so the digital twin and hackathon setups work with no extra steps.

**On a stack that is not simulated, `auto` is refused.** The run_servers config's top-level `simulated:` reaches the Approvals device. With `simulated: false` (the hardware configs, and any config that omits the key), `set_mode(auto)` fails with the reason and the device always starts in `approve`. The MCP endpoint of such a stack also binds 127.0.0.1 unless its config sets `allow_remote_bind: true`. To rehearse a hardware run with nothing moving, start the job as a dry run (`"dry_run": true` in the plan); its start is approved like any other. See [dry_run.md](dry_run.md).

## Emergency stop

`stop_all` (on the agent and operator endpoints; SciAgentGUI's **Stop everything** button calls it on the operator endpoint) stops the instrument without waiting for anyone:

1. It puts the approval mode in `approve` and voids every unused approval and pending request (`Approvals.revoke_grants`), so nothing starts again without a person.
2. On every device at once it calls, straight to the device and never through the approval gate: `blank_beam`, `close_shutter`, `laser_off`, `lamp_off`, `xray_off` and the like; `abort_all` on job devices (it aborts whatever is running without naming it); the retract of a probe instrument; and any stop your deployment marked low in `mcp.safety.risk` (for example an `EStop` command the policy does not recognise).
3. When all of that has returned, it switches the sources off once more, in case a job step that ended late switched one back on.

It never switches off high tension, a gun or a filament: blanking stops the beam, and those take hours to recover. A device that cannot be stopped (no blank offered, no `abort_all`, not answering within 20 s) is reported as a failed step, not skipped.

## Job leases

A job device with `lease_s` set aborts a running job (and puts the instrument in its safe state) when nobody has checked on it for that long: `job_status`, `wait_for_job` and `renew_lease` all renew it. Set it on hardware so a crashed agent or GUI cannot leave a job running unwatched. It is off (0) by default.

## The monitor endpoint

`configs/mcp_monitor.yaml` (port 8002) runs an endpoint with `endpoint_role: monitor`. It offers only pure reads (`get_*`, `list_*`, `State`, `Status`, `job_status`, `results`), the data tools and `trace_since`: nothing that acts, not even `stop_all`. Point a status panel or a question-answering agent at it so it can watch the instrument while the worker agent runs.
