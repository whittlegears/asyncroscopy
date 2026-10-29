# Agentic sample mapping

An agent maps a sample from one sentence, then stitches it into a zoomable map, marks where EDS spectra were taken, and proposes interesting regions. The design is in [docs/designs/agentic_sample_mapping.md](../designs/agentic_sample_mapping.md); this page is how to use it.

## The pieces

```text
 request ─► agent (LangGraph: mapping or general)
              │  MCP (agent endpoint :8000)            MCP (pyTEMlib :8003)
              ▼                                         ▼
           Mapper ── moves, acquires ──► microscope   assemble_map / score_map / add_poi
              │                                         ▲
              └── run file (tiles + spectra + trace) ─► Tiled ◄── map (pyramid + points)
```

| Piece | Where | What it does |
| :-- | :-- | :-- |
| `Mapper` | `asyncroscopy/instruments/electron_microscope/mapper.py` | One background job per map, for any instrument: serpentine tile grid inside a stage box, budgets checked at reserve, abort with the instrument made safe, resume after a restart, then stitch, score and revisit. |
| Instrument profiles | `asyncroscopy/jobs/map_profile.py` | What differs between instruments: `stem` (default; optional EDS at each tile, dose budget), `sem`, `yaml` (any YAML twin with an image, e.g. the AFM), or your own. See [Map any instrument](../Adding_New_Hardware/new_instrument.md#map-any-instrument). |
| Routines | `asyncroscopy/routines/grid.py`, `dose.py`, `drift.py` | The pure arithmetic: tile grids, electron dose, the stitch's drift at a time, live drift trackers. |
| Job envelope | `asyncroscopy/contracts/` | `reserve` → `start` → `job_status` → `results` → `release`, plus `abort`. Shared with pyTEMlib's jobs. `reserve` refuses steps outside the instrument's limits, and `suspenders` pause a job while a condition such as beam current is bad (see `asyncroscopy/contracts/README.md`). |
| Approval mode | `asyncroscopy/safety/` | `auto` or `approve`; see [approval_mode.md](approval_mode.md). |
| Map tools | pyTEMlib `mcp_server/map_tools.py` | Stitch by stage position refined by image matching; pyramid in Tiled; view regions; score with plug-in scorers; points of interest. |
| Graphs | `asyncroscopy/agents/graphs/` | `mapping`: the model only writes the plan. `general`: the model makes every call, using the `mapping` skill. |
| Skill | `asyncroscopy/skills/mapping/SKILL.md` | The procedure agents load before mapping. |

## Map plans

A plan is JSON checked against `asyncroscopy/contracts/map_plan.schema.json`. Lengths are metres.

```json
{"box_m": [-3e-7, 3e-7, -3e-7, 3e-7], "fov_m": 2e-7, "overlap": 0.15, "imsize": 512, "dwell_s": 1e-6,
 "detectors": ["haadf"], "per_tile": ["eds_point"], "eds_exposure_s": 0.5, "dose_budget_e_per_a2": 300}
```

`per_tile` is `[]` (images only), `["eds_point"]` (one spectrum at each tile centre) or `["eds_map"]` (a coarse spectrum image of every tile: much slower, only when asked for by name). `reserve` refuses a plan whose overlapping tiles, EDS spectra and revisit image together would exceed `dose_budget_e_per_a2` and returns a time and dose estimate.

The fields after `overlap` are the STEM profile's. Another instrument's Mapper (its `instrument_profile`) takes the generic fields (`box_m` or `size_m` and `centre_m`, `fov_m`, `overlap`, `order`, `stitch`, `find_regions`, `revisit`, `budgets`, `drift`, `label`) plus its own, and refuses a field it does not use by name. An AFM (the YAML twin's `afm.yaml`) maps with:

```json
{"size_m": [3e-6, 3e-6], "fov_m": 1e-6, "overlap": 0.25, "settings": {"setpoint_v": 0.4},
 "budgets": {"time_s": 900, "tip_force_nn": 10}}
```

### Budgets

`budgets` maps a name to a limit; `reserve` refuses the plan, before anything moves, when the worst case is over any of them, and the estimate's `budgets` says what each comes to. Every instrument has `time_s` (the estimated duration). The STEM and SEM profiles add `dose_e_per_a2` (the same as `dose_budget_e_per_a2`, below); a YAML instrument adds its file's `map.budgets`, such as the AFM's `tip_force_nn` (peak force, from the setpoint).

### Dose budget and beam current

- The estimate's `max_dose_e_per_a2` counts the frames where tiles overlap, a revisit image, and the EDS spectra. A spectrum's parked-beam dose is counted over a square of side `eds_dose_spread_m` (a Mapper property, 10 nm by default); an `eds_map` grid can put several spectra in one square. `eds_dose_e_per_a2` gives the EDS share.
- The current comes from the microscope's `get_parameters`: `beam_current_pa` with `beam_current_source` `measured` or `default` (then `beam_current_reason` says why). The twin reports its configured current as measured. The Spectra 300 (AutoScript) reports `default` until the screen current is measured: insert the screen, call `get_screen_current`, then retract it.
- On a stack that is not simulated, `reserve` refuses a plan with a dose budget when the current is `default`, or measured too long ago (`max_beam_current_age_s`). Only the Mapper's `simulated` property (run_servers' top-level `simulated:`) makes a stack simulated; what the microscope reports does not.
- `start` checks the current again, and so does every resume: the approval may come long after the reserve. A plan with a dose budget fails at start, before anything is set, when the current can no longer be used or is more than 5 % above the one the budget was checked with. Reserve again after measuring it.
- The agent endpoint's `mcp.job_limits.max_dose_e_per_a2` refuses, on a stack that is not simulated, any beam job whose estimate gives no dose or a current that is not `measured` (`run_job`, `<Device>_reserve` and every `<Device>_start` alike).
- `set_screen_current` writes the monochromator focus only for a current inside the calibrated range (`calibrate_screen_current`, screen inserted); any other value is refused before anything is written.

### Live drift tracking

The stitch fits drift after the map, and revisits extrapolate that fit. A plan can also correct drift while the tiles are taken, with `"drift": {"track": true}`. It is off by default.

```json
{"box_m": [-1.2e-7, 1.2e-7, -1.2e-7, 1.2e-7], "fov_m": 5e-8, "overlap": 0.25, "drift": {"track": true, "max_shift_m": 5e-9}}
```

- **How it corrects.** Before each tile, after the stage move, a tracker predicts how far the tile will sit from where the plan put it. It predicts for the middle of the frame, on the instrument's clock (`clock_s`, simulated time on the twin). The Mapper then sets the image shift (the beam shift in STEM) to cancel that error. The stage still goes only to the planned tile centres, so tracking never takes it outside the plan box.
- **How it measures.** After the tile, the Mapper registers it against an overlapping tile already taken: the previous tile in serpentine order, or else the tile in the row before. It uses the same bounded cross-correlation as the stitch (pyTEMlib `image_tools.mosaic.bounded_shift`). The result, with the applied shift taken out, is the tracker's next measurement. A featureless or ambiguous overlap gives no measurement.
- **Limits.** One correction changes the shift by at most `max_shift_m` per axis (default: a tenth of `fov_m`). All corrections together stay within `max_total_m` of the shift before the map. That default, and its maximum, is the instrument profile's limit (1 µm for STEM and SEM).
- **Refusals at reserve.** `reserve` refuses, by name, tracking on a profile that cannot shift the image (the YAML/AFM profile), tracking on a plan without overlap, an unknown tracker, and limits above the profile's.
- **Trace.** Every correction is written to the Instrument Trace as `Mapper.drift_correction`, with the shift, the step, the predicted error and whether a limit cut it. If the instrument refuses a shift, the event has ok false and the job fails.
- **When the tiles end.** The image shift goes back to its value before the map when the tiles are done, and whenever the job stops.
- **Where the results go.** The run file gets `drift_correction_m` and `drift_error_m` per tile. `results` has `summary.drift_tracking`: the number of corrections and measurements, the largest step and total, and the fitted rate. Revisits add the tracker's estimate to the stitch's residual drift fit.
- **Dry run.** A dry run measures nothing, but it records each `set_image_shift` and checks it against `max_total_m`.
- **Trackers.** Trackers live in `asyncroscopy/routines/drift.py`. The default, `linear`, fits a line over the last `window` measurements on the clock. A hackathon team can add one with `@register_tracker` (or the `asyncroscopy.drift_trackers` entry point), choose it with `"tracker"`, and score it with `drift_bench(MyTracker)`. The bench needs no stack.

`tests/test_live_drift.py` runs this on a twin drifting at 5e-11 m/s: 36 tiles of 50 nm with 25 % overlap, a 10 s move per tile, and 430 s of simulated time. Placement error is measured against the twin's ground truth, not a stitch:

- Without tracking, the worst tile is 0.39 of a field of view off.
- With tracking, the worst tile is 0.08 off. There are 36 traced corrections, 36 measurements, and a largest single correction of 3.5 nm.

**This proof is circular.** The tracker recovers the twin's own drift model: steady linear drift with Gaussian stage noise, on a sample with plenty of texture. It shows the loop and its limits work. It does not show accuracy on an instrument, where drift changes rate, settles after a move, and may fight Velox's own drift correction. Measure the drift series at first light (`calibration/drift_series.py`) before trusting it.

## From Python, without an agent

```python
import json, tango
mapper = tango.DeviceProxy("asyncroscopy/mapper/default")
reserved = json.loads(mapper.reserve(json.dumps({"plan": {"box_m": [-3e-7, 3e-7, -3e-7, 3e-7], "fov_m": 2e-7}})))
job = json.loads(mapper.start(json.dumps({"reservation_id": reserved["reservation_id"]})))
print(json.loads(mapper.job_status(json.dumps({"job_id": job["job_id"]}))))
```

Then stitch with the pyTEMlib server (`assemble_map`), or in Python with `pyTEMlib.image_tools.mosaic`. The run file is the same for every instrument (tiles with a pixel size and a stage position each, plus `instrument_profile` in its attributes), so the analysis server stitches an AFM's height map like a STEM's.

## Exploring a map

`score_map` ranks regions with a scorer and adds the best as points of interest with a reason and the scorer's version. The `baseline` scorer looks for contrast and sharp edges. Add your own (a trained model, rules, an LLM judge) by implementing `score(region, metadata) -> (value, reason)` and registering it under the `pytemlib.scorers` entry point; see `pyTEMlib/scoring/__init__.py`.

## Testing a workflow on the twin before the instrument

`configs/digital_twin_mapping.yaml` gives the twin stage error, backlash, drift, scan distortion, focus, noise, EDS spectra and beam damage at literature magnitudes. `export_ground_truth` on the twin (operator endpoint or Tango, never the agent endpoint) returns where every tile really was, so a workflow can be scored: `tests/test_map_end_to_end.py` checks that stitching recovers the true layout to under a pixel while stage positions alone are visibly off.

## Measuring an agent

```bash
uv run python benchmarks/run_tool_json.py --models qwen3:8b       # valid tool calls per model
uv run python benchmarks/run_mapping_trials.py --n 10 --model qwen3:8b   # seeded mapping tasks
```

Results are written to `benchmarks/results/`.
