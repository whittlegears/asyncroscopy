# Action space and canonical roles

An agent program, an optimiser or a benchmark harness needs to know three things about an instrument before it can drive it: what it can control, what each command does, and what it can observe. Every asyncroscopy MCP endpoint answers all three in one document, its **action space**. Each tool in it also carries a **canonical role**. Two twins may name the same action differently, a STEM's `move_stage` and an AFM file's `move_xy`, but both have the role `stage.move`. So a controller written against roles runs on both without change.

## Reading it

The MCP server offers the native tool `action_space()` on the agent and the operator endpoint. It is low risk and read-only. Like `endpoint_info`, it is for programs, not models: the document lists every tool with its arguments and runs to tens of kilobytes, so the general agent never gives it to its model (`asyncroscopy/mcp/agent_tools.py`, `NOT_FOR_MODELS`).

From a shell:

```bash
uv run python -m asyncroscopy.mcp.action_space --url http://127.0.0.1:8000/mcp          # a table
uv run python -m asyncroscopy.mcp.action_space --url http://127.0.0.1:8000/mcp --json   # the document
```

In Python, `asyncroscopy.mcp.action_space.build(server)` makes the document for an `MCPServer` after `setup()`, and `fetch(url)` reads it from a running endpoint. The schema is `asyncroscopy/contracts/action_space.schema.json`.

## What it contains

| Section | Contents |
| :-- | :-- |
| `instruments` | Each device: its tool prefix, Tango class and address, and its kind flags: `stem` (scans images and blanks a beam), `probe` (approaches the sample with a probe) and `envelope` (runs jobs). |
| `actions` | Every tool. Each has its `role` and where the role came from (`role_source`: `annotation`, `rule` or `default`), a one-line `purpose` from the command's doc, its `risk` as the approval gate sees it, its `effect` (`act`, or `observe` for a pure read), and its `args`. An acquisition marked `primary` is the instrument's main observable. |
| `args` | Each argument has a name, a JSON type, and whether it is `required`. Where known, it also has `units`, `min` and `max`, `enum` (the allowed values), `axis` (the stage axis a scalar sets) and `elements` (what each element of an array is, for example `[x, y, z, alpha, beta]`). `returns` describes a tool's result the same way. |
| `state` | What can be observed: each read tool (`get_*`, `State`, `Status`), and each readable Tango attribute, with units and range. Attributes are listed for Tango clients; an MCP client reads the matching tool. |
| `jobs` | Each job device: what its job does, its plan's JSON Schema (`plan_schema`, `null` when the device gives none), a valid `plan_example`, and its envelope tools. Run a job with `run_job`. |
| `roles` | The role vocabulary below, so the document explains itself. |
| `warnings` | Anything that could not be read, such as an invalid role annotation. |

Units and ranges come from, in order of preference:

1. the device's own `action_hints` attribute (a YAML instrument has one, built from its file);
2. for an electron microscope, `limits.action_hints` in `asyncroscopy/instruments/electron_microscope/limits.py`, the same ranges `check_fov` and `check_defocus` enforce, at the device's `min_fov`, `max_fov` and `max_defocus` properties;
3. the tool's input schema, from the Tango command's types;
4. a unit named in the command's doc, such as "in metres" or "(pA)".

An excerpt for the STEM digital twin:

```json
{"tool": "DigitalTwin_move_stage", "device": "DigitalTwin", "command": "move_stage", "role": "stage.move", "role_source": "rule",
 "purpose": "move stage: [x, y, z, alpha, beta]; x, y, z in metres, alpha and beta in degrees.", "risk": "high", "effect": "act",
 "args": [{"name": "position", "type": "array", "items": "number", "required": true, "doc": "[x, y, z, alpha, beta]; ...",
           "elements": ["x", "y", "z", "alpha", "beta"], "units": ["m", "m", "m", "deg", "deg"]}]}
{"tool": "DigitalTwin_set_fov", "role": "optics.fov", "risk": "low", "effect": "act",
 "args": [{"name": "fov", "type": "number", "required": true, "units": "m", "min": 1e-09, "max": 0.0001}]}
```

The STEM's commands are declared with plain `tango.server.command`, so no doc reaches the MCP server. In that case the `purpose` is the command's name plus what its argument is. Declare commands with `asyncroscopy.utils.commands.command` to give a real purpose.

And for the AFM YAML twin (`asyncroscopy/instruments/yaml_sim/examples/afm.yaml`):

```json
{"tool": "AFMTwin_move_xy", "device": "AFMTwin", "command": "move_xy", "role": "stage.move", "role_source": "rule",
 "purpose": "Move the coarse stage to (x_m, y_m). Refused while the tip is engaged; retract it first.", "risk": "high", "effect": "act",
 "args": [{"name": "x_m", "type": "number", "required": true, "units": "m", "min": -0.002, "max": 0.002, "doc": "Stage x", "axis": "x"},
          {"name": "y_m", "type": "number", "required": true, "units": "m", "min": -0.002, "max": 0.002, "doc": "Stage y", "axis": "y"}]}
{"tool": "AFMTwin_acquire_topography", "role": "acquire.image", "role_source": "annotation", "primary": true,
 "returns": {"type": "data key", "kind": "image", "units": "m"}}
```

## The role vocabulary

The vocabulary is short on purpose. Add a role only when a controller must tell apart two tools that the existing roles lump together.

| Role | A tool that |
| :-- | :-- |
| `stage.move`, `stage.read` | moves the sample stage (all axes, or one), or reads its position |
| `beam.blank`, `beam.unblank` | stops, or lets through, the probe beam: electrons, a laser, X-rays |
| `beam.place` | parks the beam on one point of the field of view |
| `optics.focus`, `optics.fov` | sets the focus or defocus (or runs an autofocus), or the field of view |
| `acquire.image`, `acquire.spectrum`, `acquire.other` | acquires a 2-D image, a 1-D spectrum or curve, or anything else; returns a data key |
| `probe.approach`, `probe.retract` | brings a probe (tip, indenter) onto the sample, or lifts it off |
| `job.run`, `job.reserve`, `job.start`, `job.status`, `job.abort`, `job.results`, `job.release` | runs a whole job (`run_job`), or is one verb of the job envelope |
| `safe_stop` | stops or switches off something; always low risk |
| `setting.read`, `setting.set` | reads a setting or the status, or changes a setting |
| `data.read` | reads stored data by key (`view_data`, `list_data`) |
| `other` | does anything else |

The vocabulary and the rules are in `asyncroscopy/mcp/roles.py`.

## How a tool gets its role

1. **An annotation.** A device that has an `action_hints` attribute names its commands' roles there, as JSON: `{"<command>": {"role": "...", "args": [...], "returns": {...}, "primary": true}}`. A YAML instrument builds this attribute from its file. An unknown role in an annotation is reported in `warnings`, and the rules decide instead.
2. **The rules** in `roles.role_for`, which reuse the safety policy's word lists (`asyncroscopy/safety/policy.py`). On a job device, the envelope verbs map to `job.*`. A plainly named stop becomes `beam.blank` (`blank_beam`, `laser_off`, `xray_off`), `probe.retract` (`retract_tip`) or `safe_stop`. A read (`get_*`, `list_*`) becomes `stage.read` or `setting.read`. An `acquire*` command becomes an image or a spectrum by its words. `move`/`goto` with a stage word becomes `stage.move`, `approach`/`engage` becomes `probe.approach`, a focus word becomes `optics.focus`, and `fov` or `scan_size` becomes `optics.fov`. Any other `set_*` command becomes `setting.set`.
3. **Otherwise** the role is `other`.

### Annotating your commands

In a YAML instrument, a command or an acquisition may declare `role:`. The file is refused at load if the role is not in the vocabulary:

```yaml
commands:
  laser_on:
    doc: Switch the 532 nm laser on at the given power at the sample.
    input: {name: power_mw, min: 0.1, max: 50, units: mW}
    sets: {laser_power_mw: power_mw}
    role: beam.unblank   # the name alone tells a controller nothing
```

Most YAML commands need no annotation, because the file already says more than the name:

- An acquisition's role follows its `kind`.
- The stage settings of the `map:` section (given or inferred) give `set_stage_x_m` the role `stage.move` and `get_stage_x_m` the role `stage.read`, each with its `axis`.
- The map's `field_of_view` setting gives its `set_` command the role `optics.fov`.
- A command input that `sets` a stage setting carries that setting's axis. This is how `move_xy`'s `x_m` gets `axis: x`.
- The map's image acquisition is marked `primary`.

`raman.yaml` and `xrd.yaml` annotate `laser_on` and `xray_on`, which the rules would leave as `other`.

A Python device annotates its commands by adding a read-only string attribute named `action_hints` that returns the same JSON. Name the attribute with `asyncroscopy.mcp.roles.ACTION_HINTS`. When your command names already follow the rules (`move_stage`, `get_stage`, `acquire_image`, `retract_tip`), you only need the attribute for units, ranges, axes, or a role the rules miss.

A job device publishes its plan's schema by setting `PLAN_SCHEMA` to the name of a schema in `asyncroscopy/contracts`. The Mapper sets it to `"map_plan"`. The `plan_schema` attribute then serves the schema. A device that gives no schema still has its `plan_example`.

## A controller that knows only roles

`asyncroscopy/examples/role_controller.py` reads the action space and, using roles alone:

1. reads the stage (`stage.read`, one array with `elements`, or one value per `axis`);
2. lifts a probe if there is one (`probe.retract`), then moves the stage by a small offset (`stage.move`, filling the arguments from `elements` or `axis`);
3. acquires the main observable (the `primary` acquisition, else the first `acquire.image`, then `acquire.spectrum`). It brings the probe down (`probe.approach`) and lets the beam through (`beam.unblank`) only for the acquisition, and undoes both afterwards;
4. reads the result back through the `data.read` tool that takes a key (`view_data`).

On the STEM twin, these steps call `get_stage`, `move_stage`, `unblank_beam`, `acquire_scanned_image`, `blank_beam` and `view_data`. On the AFM twin, they call `get_stage_x_m`, `get_stage_y_m`, `retract_tip`, `move_xy`, `approach_tip`, `acquire_topography`, `retract_tip` and `view_data`. None of these names appears in the controller.

`tests/test_action_space.py` runs the controller unchanged on both twins, in-process, with no stack running. Against a running stack (it moves the stage and acquires for real):

```bash
uv run python -m asyncroscopy.examples.role_controller --url http://127.0.0.1:8000/mcp
```

Telling which tool does what is the part that varies between twins. A program can do it from the roles, as here. An agent can do it from the `purpose`, `args` and `state` when a twin's commands fit no role.
