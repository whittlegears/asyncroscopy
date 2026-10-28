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
- **Give stop commands a stop word in their name.** A command whose name contains `abort`, `stop`, `halt`, `retract`, `withdraw`, `off`, `pause` or `estop`, or the pairs close + shutter or blank + beam, is always allowed: an agent never waits for approval to make the instrument safe. `Off`, `stop_scan`, `AbortScan` and `shutter_close` all qualify. A name that also contains `disable`, `override`, `bypass`, `reset`, `release`, `open` or `force` does not.
- **Refuse unsafe arguments in the device** (`raise ValueError(...)`), not in the agent. The limit then applies to notebooks, the GUI and every agent. Expose the limits (the example's `get_limits`) so a job device can check a whole plan before it starts.
- **Return JSON strings for structured results**, and Tiled keys for large data. Never return arrays to an agent.
- **Argument names come from the Python signature.** `command` writes `:param load_n:` into `doc_in` for you; with plain `tango.server.command` you must write it yourself, or the agent's argument is called `arg`.

## 2. A job device for long experiments

A grid of 400 indents takes hours; a single Tango command times out after seconds (3 s by default for a client, 30 s from the MCP server). Subclass `JobDevice` and write four hooks:

| Hook | What it does |
| :-- | :-- |
| `resolve_plan(plan)` | Check the plan; raise `ValueError` to refuse it. Return `{"steps": [...], "estimate": {...}, ...}`. Everything except `steps` is kept as `job["resolved_plan"]`. |
| `run_step(job, index)` | Do one step and return a JSON-safe record of it. |
| `on_stop(job, reason)` | Abort or failure: put the instrument in a safe state. |
| `on_finish(job)` | After the last step: make it safe again, and return the keys of anything saved. |

Also set `JOB_KIND`, one line that starts the tool descriptions, and `PLAN_EXAMPLE`, a valid plan. The agent sees the example in the `reserve` tool's description, and it comes back with every refused plan. Optionally override `summary(job)` to add what `results` returns.

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

To change any of these, add patterns to `safety.risk` in the MCP config, for example `{"SimIndenter.move_xy": "low"}`. `asyncroscopy/safety/policy.py` is the authority; see also [approval_mode.md](../Operation/approval_mode.md).

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

A config has one `instrument`; a second instrument (a robot beside an XRD) goes under `devices`.

```bash
uv run startup_scripts/run_servers.py --yaml configs/sim_indenter.yaml
```

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
```

The agent endpoint now offers `SimIndenter_indent`, `IndentGrid_reserve` and the rest. (The endpoint's name in `mcp_dt.yaml` says Spectra300; it serves whatever devices are running.)

## 5. Let the general agent use it

The general agent (`uv run python -m asyncroscopy.agents.run general "..."`, or LangGraph Studio) picks its tools in this order:

1. The globs in `ASYNCROSCOPY_AGENT_TOOLS`, if set.
2. Otherwise a short STEM set.
3. When the STEM set matches no instrument tool, as here, every tool on the endpoint, with a printed warning.

Small models do better with fewer tools, so list yours. In bash:

```bash
export ASYNCROSCOPY_AGENT_TOOLS="SimIndenter_*,IndentGrid_*,approval_status,list_skills,load_skill"
```

In PowerShell:

```powershell
$env:ASYNCROSCOPY_AGENT_TOOLS = "SimIndenter_*,IndentGrid_*,approval_status,list_skills,load_skill"
```

`ASYNCROSCOPY_AGENT_PROMPT` replaces the system prompt; `{skills}` in it is filled with the skill list.

For a multi-step procedure, add a skill: a folder with a `SKILL.md`, under `asyncroscopy/skills/` or in a folder named by `ASYNCROSCOPY_SKILLS_PATH`. Copy `asyncroscopy/skills/mapping/SKILL.md` for the format. Set `requires:` to the tools the skill cannot work without (for example `[IndentGrid_reserve]`); a skill is offered only where those tools exist, so the indenter agent is never offered the STEM mapping skill.

## 6. Test it

Copy `tests/test_new_instrument_example.py`. It covers:

- running a grid, then checking the results and the trace;
- refusing a bad plan, or one beyond the instrument's limits, at `reserve`;
- a restart not leaving the device busy;
- the agent seeing each command's docstring;
- how the policy classifies each command.

No Tango database or ports are needed.

Add your devices to the `devices_info` list in `tests/conftest.py`, next to `SimIndenter` and `IndentGrid`, and add a fixture that returns a proxy. All tests share one in-process `MultiDeviceTestContext`: PyTango can't start a second one in the same interpreter. Use `prop_path()` for any path property: the test context cuts property values at `:`, which breaks Windows drive letters.

## Digital twin first

Validate a workflow against a simulator before running it on hardware. Make the simulator a sibling class with the same commands, like `SimIndenter`; the agent, the job device and the tests then run unchanged against either one. For realistic motion, reuse the twin's instrument-neutral models: `twin/stage_model.py` (commanded vs true vs reported position, backlash, settling), `twin/clock.py` (simulated time that can run faster than real time) and `twin/rng.py` (seeded noise per event). They import without the electron-optics code. `asyncroscopy/instruments/electron_microscope/digital_twin.py` is the full-scale example.
