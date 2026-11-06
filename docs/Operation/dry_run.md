# Dry runs and the `simulated` flag

A dry run rehearses a job on the real stack without moving anything. A team uses it before a hardware session to check the plan, the approval, and the order and bounds of every stage move.

## Starting a dry run

Put `"dry_run": true` in the plan:

```python
run_job(device="Mapper", plan={"size_m": [6e-7, 6e-7], "fov_m": 1.6e-7, "dry_run": True})
```

`JobDevice` removes it from the plan before the device sees the plan. It is part of the plan, not the start request, for three reasons:

- it is fixed at `reserve`;
- the operator sees it (`get_reservation` starts with `"dry_run": true`);
- a resume keeps it.

A start request whose `dry_run` disagrees with the reservation is refused, so a rehearsal never turns into a real run. A dry-run `start` goes through the approval gate like a real start: in `approve` mode it waits for a person.

## Which job devices take a dry run

Only a device whose class sets `SUPPORTS_DRY_RUN = True`. Any other device refuses a dry-run plan at `reserve` with the error code `dry_run_unsupported`, naming itself, and reserves nothing. A dry run never half-works: a device that cannot rehearse says so before anything is held.

| Device | Dry run |
| :-- | :-- |
| Mapper | yes: moves are recorded and checked, acquisitions and their analysis skipped |
| IndentGrid (the worked example) | yes: moves and indents are recorded, no hardness is computed |
| Tuner | no: every measurement reads an image or a corrector reply |
| YAML instrument jobs | no: their steps drive the simulator directly, not through `proxy()`, so nothing would be recorded |

In a dry run every recorded command returns the placeholder `"dry_run:not_performed"`, never what the instrument would have answered. A step that parses a reply (`json.loads(indenter.indent(...))`) breaks on it. To support dry runs, write each step so it records the call and skips the analysis, as `IndentGrid.run_step` does:

```python
def run_step(self, job, index):
    step = job["steps"][index]
    indenter = self._indenter()                      # self.proxy(...): records in a dry run
    indenter.move_xy([step["x"], step["y"]])
    reply = indenter.indent(job["resolved_plan"]["load_n"])
    if job.get("dry_run"):                           # recorded, not made: nothing to analyse
        return {"index": index, "x_m": step["x"], "y_m": step["y"], "indent": "skipped (dry run)"}
    return {"index": index, "x_m": step["x"], "y_m": step["y"], **json.loads(reply)}
```

Then set `SUPPORTS_DRY_RUN = True` on the class, make `summary` cope with records that lack the analysed fields, and add a test that runs a dry-run job to `done` and shows the instrument did not move.

## What a dry run does

Inside the job's worker, `JobDevice.proxy()` returns a `RecordingProxy` (`asyncroscopy/jobs/recording_proxy.py`) in place of a `DeviceProxy`:

| Call | In a dry run |
| :-- | :-- |
| Attribute reads, the read-only DeviceProxy methods (`read_attribute`, `get_attribute_config`, ...), commands read-only by the safety policy (`get_*`, `list_*`, `State`, `Status`, unless `risk_overrides` mark one high), `job_status`, `results` | pass through |
| The job device's safe commands (`blank_beam` for a Mapper on an electron microscope, the YAML file's `safe` commands, `retract` for `IndentGrid`) | really run (they only make the instrument safer) and are recorded |
| An AFM's `retract_tip` (and any other profile's safe-state command) | recorded, **not sent**: a dry run leaves the tip where it is, since retracting would disturb a tip someone may be using |
| Every other command (`move_stage`, `unblank_beam`, `acquire_*`, `set_fov`, ...), every attribute write, and the DeviceProxy write methods (`write_attribute*`, `set_attribute_config`, `poll_attribute`, `write_pipe`, properties) | recorded in the trace, **not sent** |
| Any other DeviceProxy method | refused (`DryRunRefused`): an unlisted method is assumed to write |

Commands marked high in the MCP config's `mcp.safety.risk` are recorded even when their name says they only read: an SPM `get_image` that scans the tip is not run. The MCP server passes its high patterns with every reserve (`risk_overrides` in the request), and the job keeps them for its dry run, a resume included. A pattern may be written `Class.command` or as the tool name `Class_command`. A job reserved straight over Tango without them falls back to the naming convention alone.

Each recorded call is an Instrument Trace event marked `dry_run: true`, with `performed`, `device`, `arguments` and `check`. The job device checks each call with `check_dry_call`. The Mapper checks every `move_stage`: the image it is for must lie inside the plan box (`check.inside_box`). A failed check makes the event `ok: false` and is counted in `results`. Every other event of the job (reserve, steps, end) also carries `dry_run: true`.

Acquisitions are **skipped**, never faked. The Mapper takes no image or spectrum and writes no run file, and the sample is not exposed. Each step record and the summary say `acquisitions: "skipped (dry run): ..."`. Stitching, scoring and the revisits of found regions are skipped too, since there are no images. Revisits of given points still rehearse their moves. `results` adds `dry_run: {recorded, performed, failed_checks}`.

On the twin, a dry-run map leaves `export_ground_truth()` unchanged: it has no new moves, no new acquisitions and the same simulated time (`tests/test_dry_run_and_clock.py`).

## The `simulated` flag

Every run_servers config says at the top level whether the stack drives a simulator:

```yaml
simulated: true    # twins and YAML instruments
simulated: false   # Spectra300*.yaml, covalent-*.yaml, ThinkPad, Jupiter-local
```

A config without it is treated as real hardware, and the launcher prints a note saying so. The launcher gives every device a `simulated` property with this value (a device's own `simulated` property wins). Devices read it as `self.simulated`. Twin classes default to `true`; the `Instrument`, `JobDevice` and `Approvals` bases default to `false`.

When `simulated` is false:

- **Approvals refuses `auto`.** `set_mode({"mode": "auto"})` fails with the reason. The device also starts in `approve` whatever `default_mode` or its mode file say. `Approvals.get_simulated()` reports the flag.
- **The MCP endpoint binds 127.0.0.1.** If the MCP config asks for `0.0.0.0` (or `::`), the endpoint binds 127.0.0.1 and prints a `[safety]` line. To bind every interface, the MCP config must also set `mcp.allow_remote_bind: true`. The MCP server learns the flag from the stack's Approvals device (`get_simulated`) at start-up. If there is no Approvals device, or it cannot be read, the server counts the stack as not simulated.

## One time source

Every instrument device has a `clock_s` attribute. On a twin it is the simulated clock that drift, moves and frames run on (`asyncroscopy/sim/clock.py`); on hardware it is Unix time. The Mapper times each tile at the middle of its frame on `clock_s`. Those times are the run file's `acquired_s`, which pyTEMlib's stitch fits drift against. A revisit extrapolates the fitted drift to `clock_s` now. So drift fitting and revisits hold at any `time_scale`, including 0. Step trace events and revisit records carry `clock_s`.
