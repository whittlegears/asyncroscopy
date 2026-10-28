# Approval mode

asyncroscopy has one switch that decides whether agents need a person's approval to act on the instrument. It lives in the `Approvals` Tango device, so it covers every agent: the server-side LangGraph agents, LangGraph Studio, SciAgentGUI and any other MCP client.

| Mode | What happens to a high-risk agent call |
| :-- | :-- |
| `auto` (default) | It runs. The Instrument Trace records `approved_by: "auto"`, so the record says no person looked at it. |
| `approve` | It does not run. The MCP server files a request and tells the agent `APPROVAL REQUIRED ... request_id=...`. Once an operator approves, the agent repeats exactly the same call and it runs, once. |

Notebooks and scripts talk to Tango directly and are never held, in either mode.

## What counts as high risk

`asyncroscopy/safety/policy.py` decides, in this order (its docstring is the authority):

1. Your own patterns from `mcp.safety.risk` in the MCP config.
2. **Low:** safe-stop commands: a stop word (`abort`, `stop`, `halt`, `retract`, `withdraw`, `off`, `pause`, `estop`, `emergency`) as the first or last word of the name, or the pairs close + shutter, blank + beam, safe + state. So `Mapper_abort`, `blank_beam`, `Off` and `stop_scan` are never held. A name that also undoes a safeguard (`disable`, `override`, `bypass`, `ignore`, `reset`, `clear`, `release`, `unblank`, `open`, `force`) or does more than stop (`interlock`, `safety`, `limit`, `lock`, `unlock`, `guard`, `feedback`, `move`, `approach`, `set`, `drop`, `resume`, `start`, `run`, `enable`, `and`, `then`) is not a safe stop, so `interlock_off` and `withdraw_and_move` are held.
3. **Low:** `reserve`, `job_status`, `results`, `release` on job devices.
4. **Low:** Tango's `State` and `Status`; `get_*` and `list_*` (read-only by convention); `Approvals.request_status`; and, on a device with `acquire_scanned_image` (a STEM) only, `place_beam`, `place_beam_list`, `set_fov` and `set_defocus`.
5. **High:** everything else. That includes `move_stage`, every `acquire*`, `unblank_beam`, starting a job (`*.start`) and DATA's `register_path` (with `get_data_from_key` it would read any file). A new or renamed command is held in `approve` mode rather than waved through.

Add your own patterns in front of these with `mcp.safety.risk` in the MCP config, for example `{"MyStage.jog": "high", "MyCamera.preview": "low"}`.

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

Unanswered requests expire after `request_timeout_s` (default 30 minutes); an approval must be used within `grant_ttl_s` (default 15 minutes).

The mode is saved to the Approvals device's `mode_file` (default `outputs/approvals/mode.json`) and restored when the device restarts, so a crash or `Init` does not drop `approve` back to `auto`. If that file exists but cannot be read, the device starts in `approve`.

If the `Approvals` device cannot be reached, high-risk agent calls are refused (the server cannot tell whether a person must approve) and low-risk calls still run.

**Before an unattended agent run on a real instrument, switch to `approve`.** `auto` is the default so the digital twin and hackathon setups work with no extra steps.
