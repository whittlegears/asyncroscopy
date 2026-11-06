# Adding a new instrument

This guide puts a new instrument under agent control. It works through a simulated nanoindenter that ships with the repository. The same steps apply to an SPM, a diffractometer or a pipetting robot. The finished example is:

- `asyncroscopy/instruments/nano_indenter/nano_indenter.py`, the instrument's base device: its commands and limits.
- `asyncroscopy/instruments/nano_indenter/sim_indenter.py`, a backend of it: simulated physics behind the same commands.
- `asyncroscopy/instruments/nano_indenter/indent_grid.py`, a long experiment on it: a job device.
- `configs/sim_indenter.yaml`, the launcher config.
- `tests/test_new_instrument_example.py`, which runs all of it in-process. If the test passes, this guide is accurate.

Run the test first to see the whole thing work:

```bash
uv run pytest tests/test_new_instrument_example.py -q
```

Every instrument follows the same shape: an abstract base device for the kind of instrument, a subclass of `Instrument` (`instruments/instrument.py`, which adds a DATA device link, a test mode and `clock_s`), and one subclass per backend. The base owns the public commands (the agent's tools) and the checks every backend must apply; a backend implements only `_x` methods. The STEM devices do this (`ElectronMicroscope`, with `AutoScriptMicroscope`, `JeolMicroscope` and the twins), and so does the indenter (`NanoIndenter`, with `SimIndenter`). The older `SPMMicroscope` and `Opentrons` devices predate these conventions, so copy the indenter rather than them.

## 0. No Python: a simulated instrument from one YAML file

To try an agent workflow on your instrument before writing a driver, describe it in a YAML file. Three examples ship in `asyncroscopy/instruments/yaml_sim/examples/`: `xrd.yaml`, a powder X-ray diffractometer; `raman.yaml`, a confocal Raman microscope with a line-scan job; and `afm.yaml`, an atomic force microscope whose stage cannot move while the tip is down and whose force map lifts the tip around every move (start from it for any probe instrument). `configs/yaml_sim_xrd.yaml` runs the first on the twin's ports (stop the twin first); `configs/yaml_sim_raman.yaml` runs the second beside the twin, on its own ports ([second_stack.md](second_stack.md)):

```bash
uv run startup_scripts/run_servers.py --yaml configs/yaml_sim_xrd.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
```

The file has four sections:

- **`settings`**: each value becomes a Tango attribute and a `get_<name>` command, plus `set_<name>` unless `settable: false`. A setting has a `default`, and optionally `min`, `max`, `units`, `type` (`float`, `int`, `bool` or `str`), `choices` (the only values allowed, e.g. `["600", "1800"]` for a grating), `requires` and `doc`. Out-of-range values and values not in `choices` are refused, with the bounds or choices in the message. **Put an interlock on the setting, not only on a command:** `move_xy` with `requires: ["not tip_engaged"]` does not stop `set_stage_x_m` or a job's axes, but `requires` on the `stage_x_m` setting stops every way of changing it.
- **`commands`**: each one is a tool. A command has a `doc` (the agent's description) and optionally one `input` or several `inputs` (each with its own bounds or choices; with several, the tool takes them by name, e.g. `move_xy` with `x_m` and `y_m`), `requires` (checks such as `"tube_power_w > 0"`), `sets` (settings to change), `returns`, `duration_s` and `role` (a canonical role such as `beam.unblank`, for controllers that work by role; see [action_space.md](../Operation/action_space.md)). A refused command changes nothing.
- **`acquisitions`**: names start with `acquire`. Each produces simulated data, saves it through DATA, and returns its key, so agents look at it with `view_data`. `kind: spectrum` is a 1-D curve with an `axis` (start, stop, points, label, units), `peaks`, a `background` and `noise`. Each peak has a `position`, a `height`, an `fwhm` (full width at half maximum, in axis units) and a `shape`: `gaussian` (the default), `lorentzian` (Raman and most optical bands) or `voigt` (a pseudo-Voigt mix; `eta` is its Lorentzian fraction). Older files that give `width` mean a Gaussian's sigma, and still load. `kind: image` is a 2-D image with `shape`, `pixel_size_m` and random `spots`. `kind: value` returns named numbers as JSON. `units` says what the values are (default `counts`), and an acquisition may take `inputs` like a command. `noise` is `poisson` (counts only), `gaussian` or `none`: data in other units (metres of height, nN of force) need `noise: gaussian` with `noise_sigma` in those units, and a file that leaves it out is refused, because counting noise on a 20 nm height buries it. Gaussian noise keeps negative values (an adhesion dip).
- **`jobs`** (optional): a run of acquisitions over many points, such as a line scan or a map. A job names the `axes` it steps over (settable numeric settings, e.g. the stage), the `settings` a plan may set for the whole run, the acquisitions to `acquire` at every point, a `settle_s` per point, `requires`, the commands to run at every point `before_move` (e.g. `retract_tip`) and `after_move` (e.g. `approach_tip`, before the acquisitions), and the commands to run `on_stop` (aborted or failed, e.g. `laser_off`) and `on_finish`. With jobs, the device also has the job envelope, so the agent runs one with `run_job(device="RamanTwin", plan={"axes": {"stage_x_m": {"start": -1e-4, "stop": 1e-4, "points": 21}}, "settings": {"exposure_s": 2}})` (or `"points": [...]` for a list of positions) and waits with `wait_for_job`. Each point's values are checked against the settings' limits before anything moves, the reply carries a time estimate, and abort stops the job within a fraction of a second.

Limits to know before you write one:

- One command or acquisition call may take at most 25 s, because the MCP server stops waiting at 30 s. A longer acquisition (a long exposure, many accumulations) is refused with a pointer to the job that runs it; inside a job there is no such limit.
- While a job runs, `set_*` and `acquire*` calls are refused (the job owns the instrument); `get_*` reads and plain commands such as `laser_off` still work, and switching the source off makes the job's next acquisition fail safely.
- A job runs acquisitions without arguments and steps over numeric settings only; a job has at most `max_points` points (default 100 000).
- The data are shaped, not physical: peak positions, widths and heights follow your expressions, with the noise you choose. An image's `shape` is fixed. Its `spots` are drawn afresh on every call (`count`), so going back to a feature does not find the same one, unless you fix them on the sample with `x_m` and `y_m` (where the image centre is, usually `stage_x_m` and `stage_y_m`) and `per_um2`: then every image of a place shows the same spots, and a map of overlapping images stitches (see [Map any instrument](#map-any-instrument)). Use it to rehearse a workflow and the agent's decisions, not to validate a measurement.
- Job limits in the MCP config (`mcp.job_limits`) know a job's duration and an electron dose only; a limit on, say, tip force goes in the setting's `max` instead, or, for maps, in the file's `map.budgets`.

Test your file in-process, without starting servers, the way `tests/yaml_sim_afm_check.py` does:

```python
from asyncroscopy.instruments.yaml_sim.device import device_class
from asyncroscopy.testing import instrument_context, mcp_server_for, prop_path

devices = [{"class": device_class("AFMTwin", jobs=True), "devices": [{"name": "asyncroscopy/instrument/default", "properties": {
    "yaml_path": prop_path("my_afm.yaml"), "output_directory": prop_path("out"), "state_directory": prop_path("out/jobs")}}]}]
with instrument_context(devices):
    server = mcp_server_for(devices)                     # the agent endpoint, in-process
    print(server.list_tool_risks())                      # what approve mode would hold
    ran = server.run_job("AFMTwin", {"axes": {"stage_x_m": {"start": 0, "stop": 2e-4, "points": 3}}})
    print(server.wait_for_job(ran["job_id"], max_wait_s=30)["state"])
```

`class_name` (here `AFMTwin`) names the tools; `jobs=True` adds the job envelope when the file has `jobs:`. For the stack itself, copy `configs/yaml_sim_raman.yaml` with its two MCP configs (`configs/mcp_raman.yaml`, `configs/mcp_raman_operator.yaml`): they run beside the twin on their own ports and need no analysis server.

Values in `sets`, `returns`, `requires`, `duration_s` and an acquisition's numbers may be expressions over the settings and the command's input: arithmetic, comparisons, `and`/`or`/`not`, `x if c else y`, and `abs`, `min`, `max`, `sqrt`, `exp`, `log`, `sin`, `cos`, `radians`, `degrees`. The full format is in the docstring of `asyncroscopy/instruments/yaml_sim/spec.py`. Unknown keys, impossible bounds and expressions naming unknown settings are refused when the file is loaded. Check a file before starting it:

```bash
uv run python -m asyncroscopy.instruments.yaml_sim.spec my_instrument.yaml
```

In the launcher config, `class_name` names the tools (`XRDTwin_xray_on`, ...), `module_name` is `asyncroscopy.instruments.yaml_sim.device`, and the `yaml_path` property points at your file. The safety policy classes the generated commands like hand-written ones: `get_*` are read-only, `set_*`, `acquire*` and other commands are held in approve mode, and a plainly named stop (`xray_off`, `stop_scan`) is always allowed. The naming rules in section 1 apply, and a `get_` or `list_` command that `sets` anything is refused at load. When the workflow works on the simulator, write the real device (section 1) with the same command names, so the agent's prompts and skills carry over.

## 1. The instrument device

An instrument is a Tango device with one command per action. The MCP server turns every command into an agent tool. Write it as a base and a backend. The base subclasses `Instrument`, declares the commands and checks their arguments, then calls an abstract `_x` method; the backend implements those methods. Declare commands with `asyncroscopy.utils.commands.command` instead of `tango.server.command`: it carries the docstring to the agent as the tool's description. Tango itself does not, and plain `tango.server.command` gives the agent ":return: (not documented)". Write each docstring for a reader who has only that line:

```python
# nano_indenter.py: the base
from abc import abstractmethod

from tango.server import device_property

from asyncroscopy.instruments.instrument import Instrument
from asyncroscopy.utils.commands import command


class NanoIndenter(Instrument):
    max_load_n = device_property(dtype=float, default_value=0.5, doc="Largest peak load the head allows, in newtons.")

    @command(dtype_in=float, dtype_out=str, doc_in=":param load_n: Peak load in newtons.")
    def indent(self, load_n: float) -> str:
        """Make one indent at the current position. Returns JSON with hardness (GPa) and contact depth (m)."""
        if not 0 < load_n <= float(self.max_load_n):
            raise ValueError(f"load must be in (0, {self.max_load_n}] N")  # every backend refuses the same loads
        return json.dumps(self._indent(load_n))

    @abstractmethod
    def _indent(self, load_n: float) -> dict:
        """One indent, to a load already checked."""


# sim_indenter.py: a backend
class SimIndenter(NanoIndenter):
    def _indent(self, load_n: float) -> dict:
        ...  # simulated physics here; a real backend calls the instrument's API


if __name__ == "__main__":
    SimIndenter.run_server()  # the launcher starts each device as `python -m <module> <instance>`
```

`Instrument` also asks a backend for `read_instrument_type`, `_init_device_attributes`, `_connect` and `_disconnect`, and gives every device `Connect` and `Disconnect` commands. The class name of the backend names the tools (`SimIndenter_indent`), so a second backend of the same base offers the agent the same commands under its own name.

Conventions the rest of the stack relies on:

- **Units in names and SI values.** Use `load_n`, `pitch_m`, `travel_m`. Agents mix up units more than anything else.
- **`get_*` and `list_*` must not change anything.** The safety policy treats them as read-only, so they are never held for approval. A command that reads *and* acts (an SPM `get_image` that scans the tip, an XRD `get_counts` that opens the shutter) needs a different name. Check your `get_*` commands for side effects. A YAML instrument whose `get_` or `list_` command's doc starts with an action (`Make one indent ...`, `Moves the stage ...`) loads with a `[yaml_sim] WARNING` that suggests a rename or an `mcp.safety.risk` override.
- **Name stop commands plainly.** A safe stop is always allowed: an agent never waits for approval to make the instrument safe. It is a name with exactly one stop verb (`abort`, `stop`, `halt`, `pause`, `estop`, `retract`, `withdraw`, `disengage`, or `off` as the last word; `emergency` may qualify it) where every other word is something that verb may stop. Stop and retreat verbs take hardware: `beam`, `emission`, `lamp`, `light`, `laser`, `shutter`, `gun`, `filament`, `hv`, `tip`, `probe`, `stage`, `x`, `y`, `z`, `xy`, `xyz`, `motor(s)`, `indenter`, `head`, `objective`, `robot`, `xray`, `source`, `all`, `now` (`retract` and `withdraw` only `tip`, `probe`, `head`, `z`, `indenter`, `objective` or `needle`, named; a bare `retract` or `withdraw` is low only on a device that also has an `approach`, `engage` or `indent` command, since a syringe pump's `withdraw` aspirates; `disengage` only `tip`, `probe` or `head`, named); `abort`/`stop`/`halt`/`pause`/`estop` may also take a motion or job (`approach`, `move`, `motion`, `movement`, `scan`, `scanning`, `acquisition`, `acquire`, `job`, `run`, `travel`, `focus`, `autofocus`, `live`, `indent`, `imaging`, `exposure`, `protocol`). `off` needs one of these before it, so a bare `Off` is held; it switches off only a source of energy (`beam`, `emission`, `lamp`, `light`, `led`, `illumination`, `laser`, `gun`, `filament`, `hv`, `xray`, `source`), so `stage_off` is held. The pairs close + shutter, blank + beam and safe + state also count. This is an allow-list: anything else is held, so `brake_off`, `vacuum_off`, `retract_approach`, `stop_retract`, `close_valve` and `estop_ack` are held. `asyncroscopy/safety/policy.py` has the exact lists. `server.list_tool_risks()` (see `asyncroscopy.testing`) shows how each of your commands is classed, and the MCP server prints a `[safety]` line at start-up for any stop-like command it still holds; mark such a stop low with `mcp.safety.risk`, as `"Class.command"` or the tool name (`"OptiScope_park_stage": "low"`). It also prints a line for an override that matches no command; `server.risk_warnings` holds the same lines for a test to check. `stop_scan`, `AbortScan`, `abort_approach`, `retract_tip` and `shutter_close` qualify; a bare `Off`, `disengage_motor` (they de-energise an axis) and `retract_shutter` (it opens the beam path) do not.
- **Return within a second; run anything longer as a job.** Tango runs one command per device at a time, so while a 6 s scan command is running, even a safe stop sent to that device waits for it (and the MCP server gives up after 30 s while the hardware keeps going). A command that takes longer than about a second should be a job on a `JobDevice` (section 2), or start its work in a thread, return at once, and report progress through a `get_*` command.
- **Refuse unsafe arguments in the device** (`raise ValueError(...)`), not in the agent. The limit then applies to notebooks, the GUI and every agent. Expose the limits (the example's `get_limits`) so a job device can check a whole plan before it starts.
- **Return JSON strings for structured results**, and Tiled keys for large data. Never return arrays to an agent. To make a key, write the data with `asyncroscopy.data.data_writer.save_acquisition`, passing the DATA device's proxy as `data_server`: it registers the file, and the return value is the key. `asyncroscopy/jobs/mapping/mapper.py` does this for every tile. The MCP server attaches a PNG preview, read back through the DATA device, to any command whose name starts with `acquire` and whose result is a data key (a file name ending in `.h5`, `.hdf5`, `.hdf`, `.nxs`, `.tif` or `.tiff`; with `output_format=".tiff"` and several detectors the result is a JSON list of keys, which gets no preview). A preview never fails the command: for `acquire_camera_image`, `acquire_scanned_image` and `acquire_spectrum` the result says why there is none (for example, no reachable DATA device), for other commands it is just left out. The key of an `acquire*` result is recorded in the trace.
- **Write attributes that say how to read your data.** Agents look at stored data with `view_data` (a PNG or text, never raw arrays), which picks a view from the data's shape and attributes. A 1-D dataset with `axis_offset`, `axis_scale`, `axis_label` and `axis_units` (or `energy_offset_ev` and `energy_dispersion_ev`) is plotted against that axis with its peaks listed; a 2-D one with `pixel_size_m` is shown as an image with its field of view; 3-D and 4-D data show one frame (`index` picks it). The same PNG is attached to `acquire*` results (see above), and with it a short text summary (statistics, peaks), so an agent that cannot see images still learns what was taken. If your data needs its own drawing (a diffractogram in 2θ, a force curve), register a kind in a module you write and list it under `mcp.data_views` in your MCP YAML. The repository ships no such module; this one is the whole file, here saved as `asyncroscopy/instruments/x_ray_diffractometer/views.py` (any module the MCP server's Python can import works):

  ```python
  # asyncroscopy/instruments/x_ray_diffractometer/views.py (you create it)
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

  The MCP server imports each listed module at start-up and prints `MCP ERROR: mcp.data_views lists ...` for one it cannot import. `tests/test_views.py` registers a kind the same way.
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
- A dry run, once you opt in: `"dry_run": true` in a plan rehearses the job without moving anything. `SUPPORTS_DRY_RUN` is False by default, and `reserve` then refuses a dry-run plan with code `dry_run_unsupported`. Set it True only when every step copes with one. In a dry run `self.proxy()` passes reads through, runs `blank_beam`, and records every other command and attribute write in the trace without sending it, returning a placeholder (`DRY_RUN_PLACEHOLDER`), so a step must not parse what a command returns. `IndentGrid` is the pattern to copy:

  ```python
  SUPPORTS_DRY_RUN = True  # run_step below copes with a dry run

  def run_step(self, job, index):
      step = job["steps"][index]
      indenter = self._indenter()
      indenter.move_xy([step["x"], step["y"]])
      reply = indenter.indent(job["resolved_plan"]["load_n"])
      if job.get("dry_run"):  # recorded, not made: nothing to parse
          return {"index": index, "x_m": step["x"], "y_m": step["y"], "indent": "skipped (dry run)"}
      return {"index": index, "x_m": step["x"], "y_m": step["y"], **json.loads(reply)}
  ```

  Override `check_dry_call(job, device, name, arguments)` to check recorded calls against the plan (the Mapper checks each `move_stage` against its box). YAML job devices and the Tuner refuse dry runs. See [dry_run.md](../Operation/dry_run.md) for the whole pattern.

For drift or other time-dependent fits, time steps with the instrument's `clock_s` attribute (every `Instrument` has one: Unix time by default; a simulator overrides `instrument_clock_s()` to report its simulated time), not `time.time()`.

## 3. What the safety policy makes of it

You don't have to configure anything for the default policy:

| Command | Risk | Why |
| :-- | :-- | :-- |
| `SimIndenter.get_position`, `get_limits` | low | `get_*` |
| `SimIndenter.retract` | low | a bare retreat verb on a device that indents (on a syringe pump, `withdraw` would be high) |
| `SimIndenter.move_xy`, `indent` | high | everything else is high |
| `IndentGrid.reserve`, `job_status`, `results`, `release` | low | job-envelope verbs on a job device |
| `IndentGrid.get_reservation`, `State`, `Status` | low | read-only |
| `IndentGrid.start` | high | one approval per grid in `approve` mode |
| `IndentGrid.abort` | low | stop word |
| `AFMTwin.retract_tip`, `withdraw_tip`, `disengage_tip` | low | stop words |
| `AFMTwin.approach_tip`, `move_xy`, `set_*`, `acquire_*` | high | they touch or move over the sample |
| `Init`, `Kill`, `RestartServer` on any device | not offered to agents | operator only |

`set_fov` and `set_defocus` are low only on a device with both `acquire_scanned_image` and `blank_beam` (an electron microscope); on yours they are high, since an optical `set_defocus` drives the objective. To change any of these, add patterns to `mcp.safety.risk` in the MCP config, for example `{"SimIndenter.move_xy": "low"}`. `asyncroscopy/safety/policy.py` is the authority; see also [approval_mode.md](../Operation/approval_mode.md).

## 4. Config and launch

`configs/sim_indenter.yaml` names the instrument file and the other devices. Each device gets the Tango name `asyncroscopy/<key>/default`, and the instrument is `asyncroscopy/instrument/default`. The `data` device and the `tiled` and `tango` sections are required: the launcher starts Tiled through the DATA device. The instrument gets a `<key>_device_address` property for every device; other devices only get the properties you list.

Every device also gets `simulated`, from the config's top-level `simulated:`. Set it to `true` for a simulator and `false` for real hardware; a config without it counts as hardware. On a stack that is not simulated, the Approvals device refuses `auto` mode, and the MCP endpoint binds 127.0.0.1 unless its config sets `allow_remote_bind: true` ([dry_run.md](../Operation/dry_run.md)).

```yaml
simulated: true
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

To run a second stack beside the twin (or two teams on one machine), give it its own Tango database, ports and acquisition folder, and start it with `--isolated outputs/stacks/mylab`; [second_stack.md](second_stack.md) is the whole recipe, with the Raman example's ready configs. Without `--isolated`, a launcher refuses to start while another stack runs, rather than stopping it (add `--replace` to stop it on purpose).

## 5. Let the general agent use it

The general agent (`uv run python -m asyncroscopy.agents.run general "..."`, or LangGraph Studio) picks its tools in this order:

1. The globs in `ASYNCROSCOPY_AGENT_TOOLS`, if set.
2. Otherwise, when the endpoint has a STEM (one device with both `acquire_scanned_image` and `blank_beam`), a short STEM set. An AFM that names its scan `acquire_scanned_image` is not a STEM, and neither is an endpoint whose only map device is a `Mapper`: the Mapper maps any instrument.
3. Otherwise, as here, every tool on the instrument endpoint but its `State` and `Status`. With a `Mapper` on the endpoint (an AFM or YAML twin mapped through the `yaml` profile), the generic job and map tools come too (`run_job`, `wait_for_job`, `assemble_map`, `score_map`, `list_pois`, `view_data`, ...), so the agent keeps the instrument's own `retract_tip`, `approach_tip`, `move_xy` and `abort` and can still stitch and score its maps.

The consequence: on a STEM endpoint your extra device's tools are not given to the model unless you list them in `ASYNCROSCOPY_AGENT_TOOLS` (step 1), and the team graph's operator gets the same curated set. On any other endpoint the model sees every command, so name and document them well, and narrow the list when a small model loses track. It prints the tools it chose at start-up.

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

Programs that are not language models read the endpoint's `action_space()` instead: each tool with its canonical role (`stage.move`, `acquire.image`, `probe.retract`, ...), its arguments with units and ranges, what it observes, and the job plans. A controller that works by role runs on your instrument when your command names follow the rules there (`move_stage`, `get_stage`, `acquire_image`, `retract_tip`); for names that do not, or for units and ranges, add an `action_hints` attribute (a YAML file's `role:` key does it for you). See [action_space.md](../Operation/action_space.md).

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

## Map any instrument

The `Mapper` job device (`asyncroscopy/jobs/mapping/mapper.py`; the name and the `Mapper_*` tools are kept for every instrument) tiles a stage box, stitches the tiles into a zoomable map on the pyTEMlib analysis server, scores it and revisits the best regions. Only an **instrument profile** knows the instrument. Everything else is shared: the plan (area, grid and overlap, `size_m` around the stage, revisit-only plans), the stage-limit and dry-run checks, `clock_s` stamping and resume, the run file, the budget check and the stitch, score and revisit phases. The analysis side needs nothing from the instrument beyond image tiles with a pixel size and a stage position each, so any profile's maps stitch. Choose the profile with the Mapper's `instrument_profile` device property:

| Profile | Instrument | Budgets (plan `budgets`) |
| :-- | :-- | :-- |
| `stem` (default) | scanned images, optional EDS per tile (`per_tile`) | `dose_e_per_a2` (or the plan's `dose_budget_e_per_a2`), `time_s` |
| `sem` | any microscope device offering `move_stage`, `get_stage`, `set_fov`, `acquire_scanned_image`, `blank_beam`, `unblank_beam`, plus a SCAN device; no EDS | `dose_e_per_a2`, `time_s` |
| `yaml` | a YAML instrument with an image acquisition and stage x/y settings | `time_s` and the file's `map.budgets` |

The `stem` and `sem` profiles also need a **SCAN device**: their `setup` writes the plan's `imsize`, `dwell_s` (as `dwell_time`) and a full-frame `scan_region` to it, not to the microscope. It is the Mapper's `scan_device_address` property (default `asyncroscopy/scan/default`), so an SEM stack needs a `scan:` device in its launcher config (`asyncroscopy/instruments/electron_microscope/hardware/scan.py` and the `scan:` entry in `configs/DigitalTwin.yaml` show one), or a device of your own with writable `imsize`, `dwell_time` and `scan_region` attributes. Reserve does not check it: without one the job fails when it starts, in `setup`.

Every profile can budget `time_s` (the estimated duration). A plan over any budget it gives is refused at `reserve`, before anything moves, with the cost and how to lower it; a budget the profile does not know is refused by name.

### A YAML instrument gets maps for free

A YAML device whose file has an image acquisition and stage settings offers `get_map_profile`, which tells a Mapper with `instrument_profile: yaml` everything it needs. With no `map:` section it is inferred: the first image acquisition, the settings `stage_x_m` and `stage_y_m` (or the first job's two axes), the setting in metres that the image's `pixel_size_m` uses as the field of view, and the first job's `before_move`, `after_move`, `on_stop` (run between tiles) and `settings`. A `map:` section says it explicitly and adds budgets; `examples/afm.yaml` has:

```yaml
map:
  image: acquire_topography
  field_of_view: scan_size_m       # a tile's width in metres; without one, fov_m is the image's fixed width
  stage: [stage_x_m, stage_y_m]
  before_move: [retract_tip]       # the tip is lifted for every stage move...
  after_move: [approach_tip]       # ...and approaches again before the tile
  safe: [retract_tip]              # between tiles, and whenever the map stops
  settings: [setpoint_v, scan_rate_hz]    # what a plan's "settings" may set
  budgets:
    tip_force_nn: {per_tile: setpoint_v * 20, combine: max, units: nN, doc: peak tip force}
```

`combine: max` counts the worst single image (a peak force); `sum` adds every tile and revisit image (wear, exposure). For images that stitch, fix the image's `spots` on the sample (`x_m: stage_x_m`, `y_m: stage_y_m`, `per_um2`), as the AFM example does. A Mapper for it, in a launcher config:

```yaml
  mapper:
    class_name: Mapper
    module_name: asyncroscopy.jobs.mapping.mapper
    properties:
      instrument_profile: yaml
      microscope_device_address: asyncroscopy/instrument/default   # the YAML device
      stage_device_address: ""            # the stage limits come from the YAML settings
      analysis_url: http://127.0.0.1:8003/mcp
```

Then `run_job(device="Mapper", plan={"size_m": [3e-6, 3e-6], "fov_m": 1e-6, "overlap": 0.25, "settings": {"setpoint_v": 0.4}, "budgets": {"tip_force_nn": 10}})`. `tests/test_map_profiles.py` maps the AFM example into a stitched height map this way, through an in-process analysis server, and checks the tip was retracted for every move.

### Writing a profile

For a Python instrument, subclass `InstrumentProfile` (`asyncroscopy/jobs/mapping/profile.py`), set `name`, and register it with `@register_profile`, or with the `asyncroscopy.map_profiles` entry point for a package of your own.

**Make the Mapper's process import it.** The launcher runs every device in its own process, which imports only that device's module, so a profile defined in your own file is never registered there and `reserve` answers `unknown instrument profile 'my_afm'`. Name the file or module in the stack YAML's top-level `plugins:` list (or the `ASYNCROSCOPY_PLUGINS` variable, entries separated by commas or `;` on Windows, `:` elsewhere). Every device process imports them before it looks a name up; the same holds for drift trackers (`@register_tracker`). A file path is relative to the repository; a module name must be importable (on `PYTHONPATH`, or installed with `uv add --editable`). The launcher checks each entry when it loads the config:

```python
# my_lab/afm_profile.py
from asyncroscopy.jobs.mapping.profile import InstrumentProfile, register_profile

@register_profile
class MyAfm(InstrumentProfile):
    name = "my_afm"
    ...
```

```yaml
# configs/my_afm.yaml (run_servers.py)
plugins:
  - my_lab/afm_profile.py        # or a module name: my_lab.afm_profile
devices:
  mapper:
    class_name: Mapper
    module_name: asyncroscopy.jobs.mapping.mapper
    properties:
      instrument_profile: my_afm
```

Without the launcher (a notebook, a test), set the variable before the device starts: `ASYNCROSCOPY_PLUGINS=my_lab/afm_profile.py` (PowerShell: `$env:ASYNCROSCOPY_PLUGINS = "my_lab/afm_profile.py"`). A plug-in that fails to import is printed on the device's console, and the `unknown instrument profile` error names it with its exception. For a package you install, an entry point needs no list at all:

```toml
# your package's pyproject.toml
[project.entry-points."asyncroscopy.map_profiles"]
my_afm = "my_lab.afm_profile:MyAfm"

[project.entry-points."asyncroscopy.drift_trackers"]
my_tracker = "my_lab.drift:MyTracker"
```

Scorers for `score_map` run on the pyTEMlib analysis server, not in a device process; they install only through its `pytemlib.scorers` entry point (see [agentic_mapping.md](../Operation/agentic_mapping.md)).

The Mapper calls:

| Method | When | Must |
| :-- | :-- | :-- |
| `resolve(p)` | at reserve, defaults merged | check and complete your own plan fields; `ValueError` refuses the plan |
| `current_xy()` | for `size_m` without `centre_m` | read the stage |
| `costs(p, grid, revisit, revisits, budgets)` | at reserve | return `({name: Cost}, extra estimate fields)` |
| `setup(job)` | before the first step, and on resume | apply the plan's settings; end safe |
| `check_start(job)` | at start and on resume, before anything is set | raise `RuntimeError` when something checked at reserve no longer holds |
| `before_move` / `move(job, x, y)` / `after_move` | around every move | `move` returns False if `self.ctx.aborting(job)` |
| `stage_reported(job)` | after every move | where the stage says it is |
| `acquire_tile(job, tile)` | every tile | return `{"image_key", "acquired"}` (optionally `spectrum_key`); end safe, even on an error |
| `acquire_revisit(job, revisit)` | every revisit | image at `revisit["fov_m"]`, restore, end safe |
| `make_safe(job)` | on finish, abort and failure | never raise for an instrument already safe |

Declare `plan_fields` (fields beyond the generic ones; anything else is refused by name), `plan_defaults`, `needs` (commands checked on the instrument at `reserve`) and `settings` (your own Mapper device properties, each a `Setting(type, default, doc)`, read with `self.setting(name)` and set under the Mapper's `properties:` like any other); override `stage_targets` and `move_target` so reserve checks your stage limits and a dry run checks your moves. A profile sees the Mapper only through `self.ctx`, a `MapContext`: `proxy(address)`, `instrument()`, `aborting(job)`, `clock_s()`, `wait_for_stage(job)`, `read_acquisition(key)` and `settings`. Reach the instrument only through `self.instrument()` or `self.ctx.proxy(...)`, so a dry run records instead of moving, and give every wait a deadline. `PinholeCamera` in `tests/test_map_profiles.py` is a complete profile in about 40 lines; `asyncroscopy/instruments/electron_microscope/map_profile.py` (STEM, SEM) and `asyncroscopy/instruments/yaml_sim/map_profile.py` are the built-in ones.

## Digital twin first

Validate a workflow against a simulator before running it on hardware. Make the simulator a backend of the same base, like `SimIndenter`; the agent, the job device and the tests then run unchanged against either one. For realistic motion, reuse the modality-free models in `asyncroscopy/sim/`: `stage_model.py` (commanded vs true vs reported position, backlash, settling), `clock.py` (simulated time that can run faster than real time), `rng.py` (seeded noise per event) and `dose.py` (where a beam has been). They import without any instrument code. The stage model has five axes `[x, y, z, alpha, beta]`; for an XY or XYZ stage, pass zeros for the rest. `asyncroscopy/instruments/electron_microscope/digital_twin.py` is the full-scale example.

**Check that the twin and the driver behave alike, not only that they share command names.** `tests/test_parity.py` runs the same checks on the electron-microscope twin and on `AutoScriptMicroscope` over a recorded fake AutoScript client (`tests/fake_autoscript.py`, which logs every call that would reach the instrument and refuses out-of-range values as the AutoScript server does): the same accepted arguments, the same refused values (refused before anything reaches the instrument), and the same return shapes and `get_parameters` keys. The value checks both backends share live in the `ElectronMicroscope` base, which applies them before a backend sees the value. A command one side lacks is listed and asserted (`TWIN_LACKS`), never skipped. For a new driver, write the fake from the vendor API the driver uses, then add the driver as a third backend.
