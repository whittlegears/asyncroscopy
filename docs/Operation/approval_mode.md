# Approval mode

asyncroscopy has one switch that decides whether agents need a person's approval to act on the instrument. It lives in the `Approvals` Tango device, so it covers every agent: the server-side LangGraph agents, LangGraph Studio, SciAgentGUI and any other MCP client.

| Mode | What happens to a high-risk agent call |
| :-- | :-- |
| `auto` (default) | It runs. The Instrument Trace records `approved_by: "auto"`, so the record says no person looked at it. |
| `approve` | It does not run. The MCP server files a request and tells the agent `APPROVAL REQUIRED ... request_id=...`. Once an operator approves, the agent repeats exactly the same call and it runs, once. |

Notebooks and scripts talk to Tango directly and are never held, in either mode.

## What counts as high risk

`asyncroscopy/safety/policy.py` decides by `Class.command` pattern; the first match wins:

- **Low:** reading state (`get_*`, `list*`), job planning and control (`reserve`, `job_status`, `results`, `release`, `abort`), `blank_beam`, `place_beam`, `set_fov`, `set_defocus`, the `Approvals` and `DATA` devices.
- **High:** `move_stage`, every `acquire*`, `unblank_beam`, starting a job (`*.start`), column valves, screen, calibration and correction, autofocus.
- **Anything else is high.** A new or renamed command is held in `approve` mode rather than waved through.

Add your own patterns in front of these with `safety.risk` in the MCP config, for example `{"MyStage.jog": "high", "MyCamera.preview": "low"}`.

## Two endpoints

| Config | Port | Who connects | Offers |
| :-- | :-- | :-- | :-- |
| `configs/mcp_dt.yaml` (agent) | 8000 | agents | every device command except operator-only ones |
| `configs/mcp_operator.yaml` (operator) | 8001, 127.0.0.1 only | SciAgentGUI | also `Approvals.approve`, `deny`, `set_mode` and the twin's `export_ground_truth` |

**What approve mode does and does not hold.** It holds agents whose only route to the instrument is the agent MCP endpoint. It does not hold anything with direct Tango access (port 9094) or a shell on the stack's machine: such a program can call devices directly, or the operator endpoint (unauthenticated, local only). Run unattended agents where their only access is the agent endpoint.

The agent endpoint never offers operator-only commands (approve, deny, set_mode, the grant-consuming `check`, `request`, `list_grants`, and the twin's ground truth), whatever its config says. Without this an agent could approve its own requests, because caller identity is not yet authenticated (the `X-Asyncroscopy-Caller` header is recorded, not checked; ATHENA WG1 owns tokens). Calls on the operator endpoint are never held: they come from the person at the GUI.

## Map runs

A map is one high-risk call, `Mapper_start`, so `approve` mode asks once per run, not once per tile. The approval covers the reserved plan: its stage box, tile count and dose budget. In both modes the `Mapper` refuses to move outside its plan box, `reserve` refuses a plan that would exceed its dose budget, and the beam is blanked whenever a job ends. `Mapper_abort` is always allowed.

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

If the `Approvals` device cannot be reached, high-risk agent calls are refused (the server cannot tell whether a person must approve) and low-risk calls still run.

**Before an unattended agent run on a real instrument, switch to `approve`.** `auto` is the default so the digital twin and hackathon setups work with no extra steps.
