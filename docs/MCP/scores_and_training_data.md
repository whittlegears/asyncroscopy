# Scores and training data

The agent can run the microscope, but on its own it cannot tell whether a result is good. The ANALYSIS device gives it measured scores with units, a target and a pass or fail. The training tools record what the agent does and turn good runs into training data.

## The ANALYSIS device

`asyncroscopy/analysis/analysis_device.py` is a Tango device. Its commands take data keys and return JSON. A key is a Tiled key, or an absolute path to a local `.h5` file. The MCP server makes each command an agent tool with no extra code. Add it to a server config like any other device:

```yaml
devices:
  analysis:
    class_name: ANALYSIS
    module_name: asyncroscopy.analysis.analysis_device
    properties:
      data_device_address: asyncroscopy/data/default
      accelerating_voltage_v: 200000
      convergence_semi_angle_mrad: 30
      max_drift_per_frame_px: 0.5
      expected_spacings_m: '{"Au(111)": 2.355e-10}'
```

The properties are the targets. The operator sets them, and the agent can read them with `get_targets` but cannot change them. A target left at 0 means no target. `accelerating_voltage_v` must be set before `correction` works.

| Command | Input | Returns |
| --- | --- | --- |
| `drift` | JSON list of two keys, oldest first | Shift in pixels (row, column), drift rate in m/s, drift during one frame in pixels, a confidence value, and pass or fail |
| `resolution` | One key | The finest spacing whose FFT spots stand above the noise, in Angstrom, and which expected spacings show |
| `correction` | The JSON that `acquire_tableau` returned, from a Standard or Enhanced tableau | The largest aperture angle where the aberration phase stays below pi/4, the mrad each correction would gain, and the best next correction |
| `get_targets` | None | The targets above |

Things to know before you rely on a score:

- `drift` compares frames with a windowed cross-correlation. On a crystal lattice it can lock onto the wrong lattice period; the confidence value then drops. Measure drift at a field of view with non-periodic features. The two frames must have the same pixel size and image size. A blank frame is refused.
- `resolution` measures what the image shows. On a crystal that depends on the zone axis, and an amorphous area shows no spots. The noise level comes from the outer ring of the spectrum, so the finest result possible is about 2.35 pixels, not the 2-pixel Nyquist limit.
- `correction` refuses a Fast tableau, which measures only up to second order. It computes chi in the notation the CEOS software reports.

All scores need the metadata the writer stores with each image: `pixel_size_m`, `frame_time_s` and `acquisition_start_utc`. Images from the real microscope get the pixel size from the AutoScript metadata (`BinaryResult/PixelSize`). Save as `.h5`: AutoScript writes its own TIFF files, which do not carry these names.

## Corrector limits

The corrector refuses a correction that is too large. Two properties set the limits, in the aberration's own unit (m, or rad for WD):

- `max_step`: the largest single correction per aberration, for example `'{"C1": 50e-9}'`.
- `max_total_since_tableau`: the largest sum of corrections between two successful tableaus. This stops many small steps from adding up past the step limit.

Limits apply to the vector length of a correction, not to x and y separately. Aberrations that are not listed are not limited, and a name that is not a CEOS aberration is refused at startup. With no limits set, nothing changes. The corrector also refuses values that are not finite numbers, and a direction (a nonzero y) for C1, C3 or C5. Choose the limit values with the operators of the corrector. `get_correction_limits` shows the limits and how much of each total is used.

## Training data

The agent server records every prompt and output, as Tiled does with images. Turn it on in the LLM config:

```yaml
agent:
  training_log_dir: outputs/training_data
```

The recorder appends to two JSON-lines files in that directory:

- `llm_calls.jsonl`: one record per model call. `messages` holds what the model got, with its reply as the last message, and `tools` holds the tool definitions it was offered. This is the chat format fine-tuning tools read.
- `runs.jsonl`: one record per request: its `id`, the prompt, every tool call with a short result, and the final answer.

Each record is written as soon as it exists, so nothing is lost if the server stops. A write error gives a warning and never stops a run. `outputs/` is in `.gitignore`; keep recorded sessions out of git.

### Grade and export runs

Grades go in their own file, `grades.jsonl`, so the raw records never change. The newest grade for a run wins. The run id is the `id` field of a line in `runs.jsonl`. Without `--passed`, the grade is a fail.

```bash
python -m asyncroscopy.training.dataset grade outputs/training_data <run id> --passed --score 0.9
```

```bash
python -m asyncroscopy.training.dataset export outputs/training_data train.jsonl --holdout holdout.txt
```

The export writes every model call of every run that passed. Later calls in a run repeat the earlier calls' messages, so steps early in a run appear more than once. Prompts in the holdout file (one per line) never go in, so they stay fair test prompts.

### Grade against the twin's ground truth

The base digital twin knows its true particle positions. `get_ground_truth` returns the particles in the current view in image pixels: `row` comes from the stage x axis and `col` from y. The particle and gold twins keep no particle positions and refuse the command. `asyncroscopy.analysis.grading.grade_detections` scores detections, as (row, column) points, against the ground truth. The SEGMENTATION device returns centroids as (x, y) = (column, row), so swap them first.

The MCP configs block `get_ground_truth` and the corrector's `get_aberrations_coeff_sim` and `set_aberrations_coeff_sim`, so the agent is graded on what it measures.
