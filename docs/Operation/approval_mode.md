# Approval mode

asyncroscopy has one switch that decides whether agents need a person's approval to act on the instrument. It lives in the `Approvals` Tango device, so it covers every agent: the server-side LangGraph agents, LangGraph Studio, SciAgentGUI and any other MCP client.

| Mode | What happens to a high-risk agent call |
| :-- | :-- |
| `auto` (default) | A routine command (imaging, spectra, stage moves, placing the beam) or one an operator switched on runs, and the Instrument Trace records `approved_by: "auto"`, so the record says no person looked at it. Any other high-risk command is held exactly as in `approve` until an operator switches it on (below). |
| `approve` | It does not run. The MCP server files a request and tells the agent `APPROVAL REQUIRED ... request_id=...`. Once an operator approves, the agent repeats exactly the same call and it runs, once. |

Notebooks and scripts talk to Tango directly and are never held, in either mode.

## What counts as high risk

Your own patterns from `mcp.safety.risk` in the MCP config come first. Then what the device declares for the command (`command(risk=...)` on its base class, or a YAML instrument's file; every device of the twin stack declares all its commands, and Tango's `State`, `Status` and `Init` are known by name). Only a command its device declares nothing for (a third-party device) is classed by the name rules in `asyncroscopy/safety/policy.py`, and the MCP server lists those at start-up. The name rules decide, in this order (the module's docstring is the authority):

1. Your own patterns from `mcp.safety.risk` in the MCP config.
2. **Low:** safe-stop commands: a name with exactly one stop verb (`abort`, `stop`, `halt`, `pause`, `estop`, `retract`, `withdraw`, `disengage`, or `off` as the last word; `emergency` may qualify it) where every other word is something that verb may stop. Stop and retreat verbs take hardware: `beam`, `emission`, `lamp`, `light`, `laser`, `shutter`, `gun`, `filament`, `hv`, `tip`, `probe`, `stage`, `x`, `y`, `z`, `xy`, `xyz`, `motor(s)`, `indenter`, `head`, `objective`, `robot`, `xray`, `source`, `all`, `now` (`retract` and `withdraw` only `tip`, `probe`, `head`, `z`, `indenter`, `objective` or `needle`, named; bare only on a device that also has an `approach`, `engage` or `indent` command, since a syringe pump's `withdraw` aspirates; `disengage` only `tip`, `probe` or `head`, named); `abort`/`stop`/`halt`/`pause`/`estop` may also take a motion or job (`approach`, `move`, `motion`, `movement`, `scan`, `scanning`, `acquisition`, `acquire`, `job`, `run`, `travel`, `focus`, `autofocus`, `live`, `indent`, `imaging`, `exposure`, `protocol`). `off` needs one of these before it, so a bare `Off` is held; it switches off only a source of energy (`beam`, `emission`, `lamp`, `light`, `led`, `illumination`, `laser`, `gun`, `filament`, `hv`, `xray`, `source`), so `stage_off` is held. The pairs close + shutter, blank + beam and safe + state also count. This is an allow-list: anything else is held, so `brake_off`, `vacuum_off`, `retract_approach`, `stop_retract`, `close_valve` and `estop_ack` are held. `asyncroscopy/safety/policy.py` has the exact lists. `server.list_tool_risks()` (see `asyncroscopy.testing`) shows how each of your commands is classed, and the MCP server prints a `[safety]` line at start-up for any stop-like command it still holds; mark such a stop low with `mcp.safety.risk`, as `"Class.command"` or the tool name (`"OptiScope_park_stage": "low"`). It also prints a line for an override that matches no command. So `Mapper_abort`, `blank_beam` and `stop_scan` are never held; a bare `Off` is (on a stage it drops holding torque).
3. **Low:** `reserve`, `job_status`, `results`, `release` on job devices.
4. **Low:** Tango's `State` and `Status`; `get_*` and `list_*` (read-only by convention); `Approvals.request_status`; and, on a device with both `acquire_scanned_image` and `blank_beam` (an electron microscope) only, `set_fov` and `set_defocus` (`place_beam` is high: with the beam unblanked it parks the probe on one spot).
5. **High:** everything else. That includes `move_stage`, every `acquire*`, `unblank_beam`, starting a job (`*.start`) and DATA's `register_path` (with `view_data` it would read any file). A new or renamed command is held in `approve` mode rather than waved through.

Add your own patterns in front of these with `mcp.safety.risk` in the MCP config, for example `{"MyStage.jog": "high", "MyCamera.preview": "low"}`.

## What auto mode holds

The agent can see and call every command the agent endpoint offers. In `auto`, a high-risk call runs without asking only if:

* its device declares it routine (`command(routine=True)`): the `ElectronMicroscope` base, and so every vendor backend and twin, declares `move_stage`, `acquire_scanned_image`, `acquire_scanned_data_advanced`, `acquire_spectrum`, `place_beam`, `place_beam_list`, `place_beam_at`, `unblank_beam`, `auto_focus`, `set_stigmator`, `set_beam_tilt`, `set_image_shift` and `aim_image_shift`. `mcp.safety.auto_routine` in the MCP config, when given, replaces every device's declarations (by device kind, `{"stem": [...]}`, or by tool, `{"tools": [...]}`). On any other instrument (an AFM, a YAML twin, an indenter) no high-risk command is routine: switch on the ones your team trusts; or
* an operator switched it on: `Approvals.set_auto_allowed({"patterns": [...], "who": ...})` takes the whole list of tool names or globs (`"DigitalTwin_set_column_valves"`, `"*_set_screen"`), records the change in the trace and saves it in `mode_file` beside the mode. `[]` goes back to the defaults. It is operator-only. `get_auto_allowed` reads the list; SciAgentGUI's Settings, Agent permissions, shows one switch per held command, and its "Always allow" button on a held request adds that command.

Everything else waits for a person, including column valves, the screen, apertures, the corrector, spectrometer set-up, calibration, camera exposures and any command the policy has not seen before. A job's start (`<device>_start`, and so `run_job`) runs in auto when every high-risk command the reserved plan drives would: the Mapper and the Tuner record them in the resolved plan as `drives` ({device address: commands}); a caller's plan may not contain `drives`. A job device that does not say what it drives is held in auto unless its start is switched on. A Tuner run with the `corrector` strategy drives the corrector's `correct`, so it is held until that is switched on; `image_metric` and `vendor_autofocus` runs start on their own.

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
2. On every device at once it calls, straight to the device and never through the approval gate: `blank_beam`, `close_shutter`, `laser_off`, `lamp_off`, `xray_off` and the like; `abort_all` on job devices (it aborts whatever is running without naming it); and the retract of a probe instrument. Only these named stops are called: a stop your deployment marks low in `mcp.safety.risk` still runs without approval when called, but `stop_all` does not call it.
3. When all of that has returned, it switches the sources off once more, in case a job step that ended late switched one back on.

It never switches off high tension, a gun or a filament: blanking stops the beam, and those take hours to recover. A device that cannot be stopped (no blank offered, no `abort_all`, not answering within 20 s) is reported as a failed step, not skipped.

## Job leases

A job device with `lease_s` set aborts a running job (and puts the instrument in its safe state) when nobody has checked on it for that long: `job_status`, `wait_for_job` and `renew_lease` all renew it, and `get_jobs` does not. Set it on hardware so a crashed agent or GUI cannot leave a job running unwatched. It is off (0) by default. In SciAgentGUI's jobs panel a person can choose to keep one job's lease: the panel then renews it while its window is visible, and says so under the job. A paused job's lease does not lapse.

## Supervising a job

* `<Device>_pause` (low, always allowed) stops a running job after its current step. The device makes the instrument safe (the Mapper blanks the beam and lifts a probe) and keeps holding the reservation, so nothing else can start. `job_status` and `get_jobs` show `paused`.
* `<Device>_start` with the paused job's `reservation_id` and `"continue": true` continues it from the next step, after checking the device's suspender conditions again. Only a person continues a paused job: the agent endpoint refuses `continue`, and a plain start or `run_job` resume of a paused job is refused. A continue is refused unless the job is still paused, so a stale Continue never resumes a job someone aborted meanwhile. A paused job holds the device until it is continued or aborted, or until it has been paused for the job device's `pause_limit_s` (0, the default, means no limit), when it is aborted and its error says `pause_lapsed`.
* `<Device>_abort` stops a paused job like a running one, and `stop_all` reaches it too.
* `<Device>_revise` (operator only) replaces the plan of a reservation that has not started. The new plan is checked exactly as at reserve and gets a new `plan_id`. An approval covers the plan it was asked for: after a revise, the agent's start is held again with the new plan, and a start checked against the old plan is refused by the device (`plan_changed`).
* `<Device>_get_jobs` lists every job holding the device or running, with its progress.
* A job start in auto mode is judged by the commands its plan `drives`. Attribute values a job sets (a scan size, a dwell time, a stage axis) are not in that list: the endpoints offer no attribute writes, so the gate has no rule for them, and the job device checks each value against the attribute's limits at reserve (`step_targets`).

SciAgentGUI shows these under the Stop button: Pause, Continue, Abort, and Edit plan for a reservation that has not started.

## The monitor endpoint

`configs/mcp_monitor.yaml` (port 8002) runs an endpoint with `endpoint_role: monitor`. It offers only pure reads (`get_*`, `list_*`, `State`, `Status`, `job_status`, `results`) and the data tools: nothing that acts, not even `stop_all`, and no command your deployment marks high. `trace_since` stays on the operator endpoint. Point a status panel or a question-answering agent at it so it can watch the instrument while the worker agent runs.

## Sample loads, record fields and the dose ledger

* **Loading a sample.** Call `<Stage>_load_sample` (operator only) with `{"sample_id": "...", "who": "..."}` each time a sample goes into the holder, even the same one again. It issues a new `sample_load_id`; `<Stage>_get_sample` reads it back, and it survives a restart.
* **Record fields.** Every file `save_acquisition` writes, and every map run file, records `id` (permanent), `instrument_id`, `kind`, `sample_id`, `sample_load_id`, `code_version` and `created` (see `asyncroscopy/data/record.py`). A resumed map keeps its first id and sample, and a map whose sample was reloaded since it started refuses to resume. An AutoScript image saved as TIFF keeps its own metadata and gets no record fields yet. A stitched map from the analysis server records its run as `parents` and copies the run's sample fields; notes made with `annotate` keep the sample fields of their dataset.
* **Stale positions.** `to_stage` returns the dataset's `sample_load_id`. Before moving to that position, compare it with `<Stage>_get_sample`: a different id means the sample was reloaded since, so the position is stale. The `find_feature_then_measure` skill does this.
* **The dose ledger.** Every scanned frame saved with its stage position, field of view, pixel size, dwell time and beam current adds a line to the dose ledger (`ASYNCROSCOPY_DOSE_LEDGER`, default `outputs/dose/ledger.jsonl`). `<Microscope>_get_dose` with `{"x_m": ..., "y_m": ...}` gives the dose the current sample load has had at that point. Spectra are counted too: a parked beam's electrons over a 10 nm square around the spot (`SPOT_SPREAD_M`, the Mapper's `eds_dose_spread_m` default). Exposures are placed where the instrument believes the beam was, so scan rotation and calibration errors move them. AutoScript images record their stage position, field of view, dwell and beam current, but not which way their rows run, so each is counted over its whole field; AutoScript spectra are not counted yet. A dose computed from a current that was not measured on the screen is reported as an estimate (`from_unmeasured_current`). It does not count camera frames, 4D data, time the beam spent unblanked between exposures, or image shift; every reply says so under `not_counted`. With no sample load recorded `get_dose` gives no dose and says why.
