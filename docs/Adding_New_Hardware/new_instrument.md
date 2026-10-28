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

## 1. The instrument device

An instrument is a Tango `Device` with one command per action. The MCP server turns every command into an agent tool and uses its docstring as the tool's description, so write docstrings for a reader who has only that line:

```python
class SimIndenter(Device):
    max_load_n = device_property(dtype=float, default_value=0.5, doc="Largest peak load the head allows, in newtons.")

    @command(dtype_out=[float])
    def get_position(self) -> list[float]:
        """Stage position [x, y] in metres."""

    @command(dtype_in=float, dtype_out=str, doc_in=":param load_n: Peak load in newtons.")
    def indent(self, load_n: float) -> str:
        """Make one indent at the current position. Returns JSON with hardness (GPa) and contact depth (m)."""
```

Conventions the rest of the stack relies on:

- **Units in names and SI values.** Use `load_n`, `pitch_m`, `travel_m`. Agents mix up units more than anything else.
- **`get_*` and `list_*` must not change anything.** The safety policy treats them as read-only, so they are never held for approval. A command that reads *and* acts needs a different name.
- **Use these names for stop actions:** `abort`, `stop`, `retract`, `halt`, `close_shutter`, `safe_state`, `blank_beam`, `emergency_stop`. They are always allowed, so an agent never waits for approval to make the instrument safe.
- **Refuse unsafe arguments in the device** (`raise ValueError(...)`), not in the agent. The limit then applies to notebooks, the GUI and every agent.
- **Return JSON strings for structured results**, and Tiled keys for large data. Never return arrays to an agent.
- **`doc_in` starts with `:param <name>:`.** The MCP server uses that name as the tool's argument name.

## 2. A job device for long experiments

A Tango command times out after seconds; a grid of 400 indents takes hours. Subclass `JobDevice` and write four hooks:

| Hook | What it does |
| :-- | :-- |
| `resolve_plan(plan)` | Check the plan; raise `ValueError` to refuse it. Return `{"steps": [...], "estimate": {...}, ...}`. Everything except `steps` is kept as `job["resolved_plan"]`. |
| `run_step(job, index)` | Do one step and return a JSON-safe record of it. |
| `on_stop(job, reason)` | Abort or failure: put the instrument in a safe state. |
| `on_finish(job)` | After the last step: make it safe again, and return the keys of anything saved. |

Also set `JOB_KIND`, one line that starts the tool descriptions, and `PLAN_EXAMPLE`, a valid plan that comes back with every refused plan so an agent can copy its shape.

`JobDevice` provides the rest: `reserve` (one reservation per instrument), `start` (returns at once), `job_status`, `abort` (stops within one step), `results`, `release`, a state file per job so a restarted device resumes where it stopped, and one Instrument Trace line per step.

## 3. What the safety policy makes of it

You don't have to configure anything for the default policy:

| Command | Risk | Why |
| :-- | :-- | :-- |
| `SimIndenter.get_position` | low | `get_*` |
| `SimIndenter.retract` | low | stop verb |
| `SimIndenter.move_xy`, `indent` | high | everything else is high |
| `IndentGrid.reserve`, `job_status`, `results`, `release` | low | job-envelope verbs on a job device |
| `IndentGrid.start` | high | one approval per grid in `approve` mode |
| `IndentGrid.abort` | low | stop verb |

To change any of these, add patterns to `safety.risk` in the MCP config, for example `{"SimIndenter.move_xy": "low"}`. See [approval_mode.md](../Operation/approval_mode.md).

## 4. Config and launch

`configs/sim_indenter.yaml` names the instrument file and the other devices. Each device gets the Tango name `asyncroscopy/<key>/default`, and the instrument is `asyncroscopy/instrument/default`:

```yaml
instrument:
  class_name: SimIndenter
  file: asyncroscopy/instruments/nano_indenter/sim_indenter.py
devices:
  indentgrid:
    class_name: IndentGrid
    module_name: asyncroscopy.instruments.nano_indenter.indent_grid
    properties:
      indenter_device_address: asyncroscopy/instrument/default
  approvals:
    class_name: Approvals
    module_name: asyncroscopy.safety.approvals
```

```bash
uv run startup_scripts/run_servers.py --yaml configs/sim_indenter.yaml
```

```bash
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
```

The agent endpoint now offers `SimIndenter_indent`, `IndentGrid_reserve` and the rest.

## 5. Let the general agent use it

The general agent sees only the tools that match its patterns. The defaults cover microscopes, so list yours:

```bash
ASYNCROSCOPY_AGENT_TOOLS="SimIndenter_*,IndentGrid_*,approval_status,list_skills,load_skill"
```

For a multi-step procedure, add a skill: a folder under `asyncroscopy/skills/` with a `SKILL.md`. Copy `asyncroscopy/skills/mapping/SKILL.md` for the format. The agent loads a skill by name before it starts the task.

## 6. Test it

Copy `tests/test_new_instrument_example.py`. It runs a grid, checks the results and the trace, checks that a bad plan is refused at `reserve`, and checks how the policy classifies each command. No Tango database or ports are needed.

Add your devices to the `devices_info` list in `tests/conftest.py`, next to `SimIndenter` and `IndentGrid`, and add a fixture that returns a proxy. All tests share one in-process `MultiDeviceTestContext`: PyTango can't start a second one in the same interpreter. Use `prop_path()` for any path property: the test context cuts property values at `:`, which breaks Windows drive letters.

## Digital twin first

Validate a workflow against a simulator before running it on hardware. Make the simulator the same device class with a `simulate` property, or a sibling class with the same commands. The agent, the job device and the tests then run unchanged against either one. `asyncroscopy/instruments/electron_microscope/digital_twin.py` is the full-scale example.
