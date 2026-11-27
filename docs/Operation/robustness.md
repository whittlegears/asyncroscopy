# Robustness: health, recovery, and backend conformance

An autonomous session succeeds or fails on what happens when something breaks
mid-run. A corrector PC drops its link, a vendor server hangs, or a device
server process exits. Asyncroscopy now handles each of these the same way for
the agent, the operator, and the person at the terminal.

## What happens when a device fails

| Failure | Detected by | Recovery |
|---|---|---|
| Vendor link drops; the device goes to FAULT. Example: the corrector link. | The failed MCP call's error, `system_health`, the doctor | `recover_device` runs the device's own `reconnect`, then `Init`. |
| A command times out | The MCP error, which includes a **readback** of the state the command was changing, for example `get_stage` after `move_stage` | The agent checks `system_health` before retrying. |
| A device server hangs | `system_health` | Restart the server through its admin device. This needs **operator approval** because it restarts every device in that server. `recover_device` queues the restart as an approval-gated action. |
| A device server process exits or crashes | `system_health`, the doctor. Devices of a stopped server are still checked. | `run_servers.py` **restarts it automatically**, with backoff. After 5 exits in 10 minutes it stops and prints the server's last stderr lines. Disable with `--no-watchdog`. |

A failed device call now explains itself to the agent:

```text
PyDs_PythonError: ConnectionError: link to the corrector is down ...
Health: asyncroscopy/corrector/default is fault (Link to corrector lost ...)
The link behind asyncroscopy/corrector/default is down. Call recover_device('asyncroscopy/corrector/default').
```

The MCP tools are always available, with or without supervision:

- `system_health`: reachability, state, latency and host server for every device.
- `recover_device(device)`: runs the recovery ladder and re-checks after each
  step.

With supervision on, recovery runs in the shared action queue:
- Each step is recorded as a child action.
- The steps the agent may take alone are set by `mcp.supervision.recovery_auto`.
  The default is `reconnect` and `init`.
- A server restart waits for the operator.

## Doctor

The doctor turns `troubleshooting.md` into checks. It covers:

- the Tango database
- every defined device
- the Tiled server and save path
- the MCP supervision API

```bash
uv run python -m asyncroscopy.health                         # read-only checklist
uv run python -m asyncroscopy.health --fix                   # reconnect/Init faulted devices
uv run python -m asyncroscopy.health --fix --allow-restart   # also restart hung device servers
uv run python -m asyncroscopy.health --mcp-url http://localhost:8000 --json
```

It exits with code 0 when everything is OK, so it works in scripts and CI.

## Fault drill on the digital twin

`DigitalTwinCorrector.inject_link_drop` simulates the corrector link dropping.
It is blocked in the MCP configs, so agents can't call it.

```bash
python -c "import tango; tango.DeviceProxy('asyncroscopy/corrector/default').command_inout('inject_link_drop')"
uv run python -m asyncroscopy.health          # FAIL corrector: fault
uv run python -m asyncroscopy.health --fix    # OK corrector: recovered by reconnect
```

To drill a crash, kill the corrector server process with `kill -9`. The
launcher prints `WARN corrector exited (code -9); restarted`, and the next
corrector call succeeds.

## Backend conformance

Run this on every new vendor backend: first against its simulator, then on the
instrument.

```bash
uv run python -m asyncroscopy.conformance asyncroscopy/instrument/default             # read-only
uv run python -m asyncroscopy.conformance asyncroscopy/instrument/default --writes --acquire
```

It checks each operation:

- Every read either works with the documented shape and units, or fails with a
  clean "does not implement" error. A crash is a failure.
- Units are checked: a stage in nm or µm instead of metres, or tilts not in
  degrees, are caught.
- With `--writes`: set/get round trips write the current value back, plus a
  zero-length stage move.
- With `--acquire`: one acquisition.

It then lists **feature readiness**: which features this backend can run,
including imaging, the specimen atlas, focus, autofocus, beam blanking (needed
by STOP), spectroscopy and beam tilt.
