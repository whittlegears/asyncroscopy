# Adding a new instrument

This guide puts a new instrument under agent control. It works through a simulated nanoindenter that ships with the repository. The same steps apply to an SPM, a diffractometer or a pipetting robot. The finished example is:

- `asyncroscopy/instruments/nano_indenter/sim_indenter.py`, the instrument: one Tango device.
- `asyncroscopy/instruments/nano_indenter/indent_grid.py`, a long experiment on it: a job device.
- `configs/sim_indenter.yaml`, the launcher config.
- `tests/test_new_instrument_example.py`, which runs all of it in-process. If the test passes, this guide is accurate.

Run the test first to see the whole thing work:

```bash
uv run pytest tests/test_new_instrument_example.py -q
```

The STEM devices (`instruments/electron_microscope/`) subclass the `Instrument` base in `instruments/instrument.py`, which adds a DATA device link and a test mode. A new instrument doesn't need it; the example is a plain Tango `Device`. The older `SPMMicroscope` and `Opentrons` devices predate these conventions, so copy the indenter rather than them.

## 0. No Python: a simulated instrument from one YAML file

To try an agent workflow on your instrument before writing a driver, describe it in a YAML file. `asyncroscopy/instruments/yaml_sim/examples/xrd.yaml` is a powder X-ray diffractometer written this way, and `configs/yaml_sim_xrd.yaml` runs it:

```bash
uv run startup_scripts/run_servers.py --yaml configs/yaml_sim_xrd.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
```

The file has three sections:

- **`settings`**: each value becomes a Tango attribute and a `get_<name>` command, plus `set_<name>` unless `settable: false`. A setting has a `default`, and optionally `min`, `max`, `units`, `type` (`float`, `int`, `bool` or `str`) and `doc`. Out-of-range values are refused, with the bounds in the message.
- **`commands`**: each one is a tool. A command has a `doc` (the agent's description) and optionally one `input` (with its own bounds), `requires` (checks such as `"tube_power_w > 0"`), `sets` (settings to change), `returns` and `duration_s`. A refused command changes nothing.
- **`acquisitions`**: names start with `acquire`. Each produces simulated data, saves it through DATA, and returns its key, so agents look at it with `view_data`. `kind: spectrum` is a 1-D curve with an `axis` (start, stop, points, label, units), `peaks`, a `background` and `noise`. `kind: image` is a 2-D image with `shape`, `pixel_size_m` and random `spots`. `kind: value` returns named numbers as JSON.

Values in `sets`, `returns`, `requires`, `duration_s` and an acquisition's numbers may be expressions over the settings and the command's input: arithmetic, comparisons, `and`/`or`/`not`, `x if c else y`, and `abs`, `min`, `max`, `sqrt`, `exp`, `log`, `sin`, `cos`, `radians`, `degrees`. The full format is in the docstring of `asyncroscopy/instruments/yaml_sim/spec.py`. Unknown keys, impossible bounds and expressions naming unknown settings are refused when the file is loaded. Check a file before starting it:

```bash
uv run python -m asyncroscopy.instruments.yaml_sim.spec my_instrument.yaml
```

In the launcher config, `class_name` names the tools (`XRDTwin_xray_on`, ...), `module_name` is `asyncroscopy.instruments.yaml_sim.device`, and the `yaml_path` property points at your file. The safety policy classes the generated commands like hand-written ones: `get_*` are read-only, `set_*`, `acquire*` and other commands are held in approve mode, and a plainly named stop (`xray_off`, `stop_scan`) is always allowed. The naming rules in section 1 apply, and a `get_` or `list_` command that `sets` anything is refused at load. When the workflow works on the simulator, write the real device (section 1) with the same command names, so the agent's prompts and skills carry over.

## 1. The instrument device

An instrument is a Tango `Device` with one command per action. The MCP server turns every command into an agent tool. Declare commands with `asyncroscopy.utils.commands.command` instead of `tango.server.command`: it carries the docstring to the agent as the tool's description. Tango itself does not, and plain `tango.server.command` gives the agent ":return: (not documented)". Write each docstring for a reader who has only that line:

```python
from tango.server import Device, device_property

from asyncroscopy.utils.commands import command


class SimIndenter(Device):
    max_load_n = device_property(dtype=float, default_value=0.5, doc="Largest peak load the head allows, in newtons.")

    @command(dtype_out=[float])
    def get_position(self) -> list[float]:
        """Stage position [x, y] in metres."""

    @command(dtype_in=float, dtype_out=str, doc_in="Peak load in newtons.")
    def indent(self, load_n: float) -> str:
        """Make one indent at the current position. Returns JSON with hardness (GPa) and contact depth (m)."""


if __name__ == "__main__":
    SimIndenter.run_server()  # the launcher starts each device as `python -m <module> <instance>`
```

Conventions the rest of the stack relies on:

- **Units in names and SI values.** Use `load_n`, `pitch_m`, `travel_m`. Agents mix up units more than anything else.
- **`get_*` and `list_*` must not change anything.** The safety policy treats them as read-only, so they are never held for approval. A command that reads *and* acts (an SPM `get_image` that scans the tip, an XRD `get_counts` that opens the shutter) needs a different name. Check your `get_*` commands for side effects.
- **Name stop commands plainly.** A safe stop is always allowed: an agent never waits for approval to make the instrument safe. It is a name with exactly one stop verb (`abort`, `stop`, `halt`, `pause`, `estop`, `retract`, `withdraw`, `disengage`, or `off` as the last word; `emergency` may qualify it) where every other word is something that verb may stop. Stop and retreat verbs take hardware: `beam`, `emission`, `lamp`, `light`, `laser`, `shutter`, `gun`, `filament`, `hv`, `tip`, `probe`, `stage`, `x`, `y`, `z`, `xy`, `xyz`, `motor(s)`, `indenter`, `head`, `objective`, `robot`, `xray`, `source`, `all`, `now` (`retract` and `withdraw` only `tip`, `probe`, `head`, `z`, `indenter`, `objective` or nothing; `disengage` only `tip`, `probe` or `head`, named); `abort`/`stop`/`halt`/`pause`/`estop` may also take a motion or job (`approach`, `move`, `motion`, `movement`, `scan`, `scanning`, `acquisition`, `acquire`, `job`, `run`, `travel`, `focus`, `autofocus`, `live`, `indent`, `imaging`, `exposure`, `protocol`). `off` needs one of these before it, so a bare `Off` is held; it switches off only a source of energy (`beam`, `emission`, `lamp`, `light`, `led`, `illumination`, `laser`, `gun`, `filament`, `hv`, `xray`, `source`), so `stage_off` is held. The pairs close + shutter, blank + beam and safe + state also count. This is an allow-list: anything else is held, so `brake_off`, `vacuum_off`, `retract_approach`, `stop_retract`, `close_valve` and `estop_ack` are held. `asyncroscopy/safety/policy.py` has the exact lists. `server.list_tool_risks()` (see `asyncroscopy.testing`) shows how each of your commands is classed, and the MCP server prints a `[safety]` line at start-up for any stop-like command it still holds; mark such a stop low with `mcp.safety.risk`, as `"Class.command"` or the tool name (`"OptiScope_park_stage": "low"`). It also prints a line for an override that matches no command; `server.risk_warnings` holds the same lines for a test to check. `stop_scan`, `AbortScan`, `abort_approach`, `retract_tip` and `shutter_close` qualify; a bare `Off`, `disengage_motor` (they de-energise an axis) and `retract_shutter` (it opens the beam path) do not.
- **Return within a second; run anything longer as a job.** Tango runs one command per device at a time, so while a 6 s scan command is running, even a safe stop sent to that device waits for it (and the MCP server gives up after 30 s while the hardware keeps going). A command that takes longer than about a second should be a job on a `JobDevice` (section 2), or start its work in a thread, return at once, and report progress through a `get_*` command.
- **Refuse unsafe arguments in the device** (`raise ValueError(...)`), not in the agent. The limit then applies to notebooks, the GUI and every agent. Expose the limits (the example's `get_limits`) so a job device can check a whole plan before it starts.
- **Return JSON strings for structured results**, and Tiled keys for large data. Never return arrays to an agent. To make a key, write the data with `asyncroscopy.data.data_writer.save_acquisition`, passing the DATA device's proxy as `data_server`: it registers the file, and the return value is the key. `asyncroscopy/instruments/electron_microscope/mapper.py` does this for every tile. The MCP server attaches a PNG preview, read back through the DATA device, to any command whose name starts with `acquire` and whose result is a data key (a file name ending in `.h5`, `.hdf5`, `.hdf`, `.nxs`, `.tif` or `.tiff`; with `output_format=".tiff"` and several detectors the result is a JSON list of keys, which gets no preview). A preview never fails the command: for `acquire_camera_image`, `acquire_scanned_image` and `acquire_spectrum` the result says why there is none (for example, no reachable DATA device), for other commands it is just left out. The key of an `acquire*` result is recorded in the trace.
- **Write attributes that say how to read your data.** Agents look at stored data with `view_data` (a PNG or text, never raw arrays), which picks a view from the data's shape and attributes. A 1-D dataset with `axis_offset`, `axis_scale`, `axis_label` and `axis_units` (or `energy_offset_ev` and `energy_dispersion_ev`) is plotted against that axis with its peaks listed; a 2-D one with `pixel_size_m` is shown as an image with its field of view; 3-D and 4-D data show one frame (`index` picks it). The same PNG is attached to `acquire*` results (see above), and with it a short text summary (statistics, peaks), so an agent that cannot see images still learns what was taken. If your data needs its own drawing (a diffractogram in 2θ, a force curve), register a kind in a module and list it under `mcp.data_views` in your MCP YAML:

  ```python
  # asyncroscopy/instruments/x_ray_diffractometer/views.py
  from asyncroscopy.data.views import Axis, line_kind, register_kind

  register_kind(line_kind(
      "diffractogram",
      matches=lambda ds: ds.ndim == 1 and "two_theta_start_deg" in ds.attrs,
      axis=lambda ds: Axis(ds.attrs["two_theta_start_deg"], ds.attrs["two_theta_step_deg"], "2θ", "deg"),
  ), before="profile")
  ```

  ```yaml
  mcp:
    data_views: [asyncroscopy.instruments.x_ray_diffractometer.views]
  ```
- **Start `doc_in` with "JSON" for an object argument** (`doc_in=':param plan: JSON {...}'`). Only then does the MCP server accept an object from the model and normalise it, so a retried call matches the approval it was given.
- **Argument names come from the Python signature.** `command` writes `:param load_n:` into `doc_in` for you; with plain `tango.server.command` you must write it yourself, or the agent's argument is called `arg`.

## 2. A job device for long experiments

A grid of 400 indents takes hours; a single Tango command times out after seconds (3 s by default for a client, 30 s from the MCP server). Subclass `JobDevice` and write four hooks:

| Hook | What it does |
| :-- | :-- |
| `resolve_plan(plan)` | Check the plan; raise `ValueError` to refuse it. Return `{"steps": [...], "estimate": {...}, ...}`. Everything except `steps` is kept as `job["resolved_plan"]`. |
| `run_step(job, index)` | Do one step and return a JSON-safe record of it. A step that can take long (minutes) should check `self.aborting(job)` and return `None` when it is set; the job then stops, and a resume runs that step again. |
| `on_start(job)` | Optional. Before the first step (and on resume): open a shutter, power a source. |
| `on_stop(job, reason)` | Abort or failure: put the instrument in a safe state. |
| `on_finish(job)` | After the last step: make it safe again, and return the keys of anything saved. |

A job's `state` is `reserved`, then `running` (`aborting` while an abort waits for the step), and ends `done`, `aborted` or `failed`. Also set `JOB_KIND`, one line that starts the tool descriptions, and `PLAN_EXAMPLE`, a valid plan. The agent sees the example in the `reserve` tool's description, and it comes back with every refused plan. `results` returns the records of the last 50 steps (what `run_step` returned); override `summary(job)` to return something more useful, such as one map key.

Inside `run_step`, reach other devices with `self.proxy(address)`, not `tango.DeviceProxy(address)`: a bare proxy gives up after Tango's 3 s default, so a stage move or autofocus that takes longer fails the job while the motor is still moving. `proxy()` uses the `step_timeout_s` property (120 s). In-process tests do not show this timeout, so they pass either way.

`JobDevice` provides the rest:

- `reserve` (one reservation per instrument), `start` (returns at once), `job_status`, `abort` (stops within one step) and `results`.
- `get_reservation`: the plan an operator sees before approving a start.
- `release`.
- A state file per job, so a job interrupted by a restart can be started again and skips the steps already done.
- One Instrument Trace line per step.

## 3. What the safety policy makes of it

You don't have to configure anything for the default policy:

| Command | Risk | Why |
| :-- | :-- | :-- |
| `SimIndenter.get_position`, `get_limits` | low | `get_*` |
| `SimIndenter.retract` | low | stop word |
| `SimIndenter.move_xy`, `indent` | high | everything else is high |
| `IndentGrid.reserve`, `job_status`, `results`, `release` | low | job-envelope verbs on a job device |
| `IndentGrid.get_reservation`, `State`, `Status` | low | read-only |
| `IndentGrid.start` | high | one approval per grid in `approve` mode |
| `IndentGrid.abort` | low | stop word |
| `Init`, `Kill`, `RestartServer` on any device | not offered to agents | operator only |

`set_fov` and `set_defocus` are low only on a device with both `acquire_scanned_image` and `blank_beam` (an electron microscope); on yours they are high, since an optical `set_defocus` drives the objective. To change any of these, add patterns to `mcp.safety.risk` in the MCP config, for example `{"SimIndenter.move_xy": "low"}`. `asyncroscopy/safety/policy.py` is the authority; see also [approval_mode.md](../Operation/approval_mode.md).

## 4. Config and launch

`configs/sim_indenter.yaml` names the instrument file and the other devices. Each device gets the Tango name `asyncroscopy/<key>/default`, and the instrument is `asyncroscopy/instrument/default`. The `data` device and the `tiled` and `tango` sections are required: the launcher starts Tiled through the DATA device. The instrument gets a `<key>_device_address` property for every device; other devices only get the properties you list:

```yaml
instrument:
  class_name: SimIndenter
  file: asyncroscopy/instruments/nano_indenter/sim_indenter.py
devices:
  data:
    module_name: asyncroscopy.data.data
  indentgrid:
    class_name: IndentGrid
    module_name: asyncroscopy.instruments.nano_indenter.indent_grid
    properties:
      indenter_device_address: asyncroscopy/instrument/default
  approvals:
    class_name: Approvals
    module_name: asyncroscopy.safety.approvals
tiled:
  host: localhost
  port: 9091
  acquisition_dir: outputs/tiled_acquisitions
  autostart: true
  register_on_startup: false
tango:
  host: localhost
  port: 9094
  reset_database_file: true
```

A config has one `instrument`; a second instrument (a robot beside an XRD) goes under `devices`. For an instrument kept outside this repository, replace `file` with `module_name` (for example `mylab.optiscope`) and give `class_name`; put its folder on `PYTHONPATH`, which the launched servers inherit (a package installed with `uv pip install -e` is removed by the next `uv sync`; `uv add --editable path/to/mylab` keeps it, but edits `pyproject.toml`). Supporting devices under `devices` work the same way. Each device is named `asyncroscopy/<config key>/default`; two devices of one class get tools named after their config keys (`IndentGrid_grid1_reserve`, `IndentGrid_grid2_reserve`), so give a skill's `requires:` a glob such as `IndentGrid*_reserve`.

```bash
uv run startup_scripts/run_servers.py --yaml configs/sim_indenter.yaml
```

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
```

The agent endpoint now offers `SimIndenter_indent`, `IndentGrid_reserve` and the rest. (`mcp_dt.yaml` serves whatever devices are running, not only the twin.) To approve held calls or switch the mode from SciAgentGUI, also start the operator endpoint:

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp_operator.yaml
```

`startup_scripts/check_stack.py` checks the STEM twin stack (Mapper, stitching, EDS); it is not a check for other instruments. For yours, run your copy of `tests/test_new_instrument_example.py`.

To run a second stack beside the twin (or two teams on one machine), give it its own Tango database and ports: `uv run startup_scripts/run_servers.py --yaml configs/mylab.yaml --isolated .stack_mylab`, with different `tango.port` and `tiled.port` in that YAML.

## 5. Let the general agent use it

The general agent (`uv run python -m asyncroscopy.agents.run general "..."`, or LangGraph Studio) picks its tools in this order:

1. The globs in `ASYNCROSCOPY_AGENT_TOOLS`, if set.
2. Otherwise, when the endpoint has a STEM (a device with both `acquire_scanned_image` and `blank_beam`, or a `Mapper` job device), a short STEM set. An AFM that names its scan `acquire_scanned_image` is not a STEM.
3. Otherwise, as here, every tool on the instrument endpoint (not the STEM analysis server's).

It prints the tools it chose at start-up.

Small models do better with fewer tools, so list yours. In bash:

```bash
export ASYNCROSCOPY_AGENT_TOOLS="SimIndenter_*,IndentGrid_abort,run_job,wait_for_job,approval_status"
```

In PowerShell:

```powershell
$env:ASYNCROSCOPY_AGENT_TOOLS = "SimIndenter_*,IndentGrid_abort,run_job,wait_for_job,approval_status"
```

On an endpoint with several microscopes or Mappers, `ASYNCROSCOPY_MICROSCOPE` (for example `DigitalTwin`) and `ASYNCROSCOPY_MAPPER` (for example `Mapper_lab2`) choose the ones the `mapping` graph uses.

`ASYNCROSCOPY_AGENT_PROMPT` replaces the system prompt. Skills are not part of it: before every model call the agent adds the skills that match the request and what has happened so far (see `docs/Operation/agents.md`). Braces in the prompt (a JSON plan example) are left as they are.

`wait_for_job` waits up to 25 s per call, and one agent run allows about 20 tool calls, so the agent can follow a job of a few minutes. For an hours-long job, tell it in the prompt or skill to report the job id and stop; check on it later with `job_status`.

For a multi-step procedure, add a skill: a folder with a `SKILL.md`, under `asyncroscopy/skills/` or in a folder named by `ASYNCROSCOPY_SKILLS_PATH`. Copy `asyncroscopy/skills/mapping/SKILL.md` for the format. Set `requires:` to the tools the skill cannot work without (for example `[IndentGrid_reserve]`); a skill is offered only where those tools exist, so the indenter agent is never offered the STEM mapping skill.

## 6. Test it

Copy `tests/test_new_instrument_example.py`. It covers:

- running a grid, then checking the results and the trace;
- refusing a bad plan, or one beyond the instrument's limits, at `reserve`;
- a restart not leaving the device busy;
- the agent seeing each command's docstring;
- how the policy classifies each command.

No Tango database or ports are needed.

**Instrument kept in its own repository?** Use `asyncroscopy.testing`: `instrument_context(devices_info)` starts your devices in a Tango test context (only one per Python process, so open it once in a session-scoped pytest fixture and share it between tests), and `mcp_server_for(devices_info, approvals_device_address=...)` builds the MCP server the agent would talk to, so you can check the tool descriptions it reads (`server.list_tool_descriptions()`) and which calls approve mode holds. `tests/out_of_tree_check.py` is a complete example. `server.list_tool_risks()` shows what approve mode would hold without calling anything. `instrument_context(..., process=True)` runs the devices in a child process, where Tango's 3 s timeout applies to calls from your test, as it does on the real launcher. Calls between devices in that one child process are not timed out, so it cannot catch a job that uses `tango.DeviceProxy` instead of `self.proxy()`; search your job devices for `DeviceProxy(` instead.

**Instrument inside this repository?** Add your devices to the `devices_info` list in `tests/conftest.py`, next to `SimIndenter` and `IndentGrid`, and add a fixture that returns a proxy. All tests share one in-process `MultiDeviceTestContext`: PyTango can't start a second one in the same interpreter. Use `prop_path()` for any path property: the test context cuts property values at `:`, which breaks Windows drive letters.

## Digital twin first

Validate a workflow against a simulator before running it on hardware. Make the simulator a sibling class with the same commands, like `SimIndenter`; the agent, the job device and the tests then run unchanged against either one. For realistic motion, reuse the twin's instrument-neutral models in `asyncroscopy/instruments/electron_microscope/twin/`: `stage_model.py` (commanded vs true vs reported position, backlash, settling), `clock.py` (simulated time that can run faster than real time) and `rng.py` (seeded noise per event). They import without the electron-optics code. The stage model has five axes `[x, y, z, alpha, beta]`; for an XY or XYZ stage, pass zeros for the rest. `asyncroscopy/instruments/electron_microscope/digital_twin.py` is the full-scale example.
