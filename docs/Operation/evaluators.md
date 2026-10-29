# Evaluators: scoring operations and runs

`asyncroscopy.evaluate` scores what a run did. It reads the run's Instrument Trace and the data the trace points to, and returns `{objective: value}` with units and a short reason. It does not drive the instrument, and it does not know which agent, graph or person made the run.

Evaluators are kept apart from the twin and from the agents (the hackathon organisers' principle 3). Resolution, dose, information and time are different objectives, and they often disagree. Separate evaluators let a team look at each objective on its own, then combine them only when it decides to. Learning the reward itself from traces is left open, as a research problem, but the inputs are ready for it (see [Features for learning a reward](#features-for-learning-a-reward)).

```bash
uv run python -m asyncroscopy.evaluate --list
uv run python -m asyncroscopy.evaluate --run 669e81c5855d                                  # the table
uv run python -m asyncroscopy.evaluate --run 669e81c5855d --tiled-uri http://127.0.0.1:9091   # read the images from Tiled
uv run python -m asyncroscopy.evaluate --run 669e81c5855d --evaluators dose,time --json scores.json
uv run python -m asyncroscopy.evaluate --run 669e81c5855d --weights information.features_found=1,dose.over_threshold_fraction=-100
uv run python -m asyncroscopy.evaluate --run 669e81c5855d --features                       # the flat row
```

`--run` matches the agent run (`X-Asyncroscopy-Run`, the trace's `run`) first, then the `caller`, then a job or reservation id. `--trace-dir` defaults to `outputs/trace`. Data keys are read as local HDF5 files, or from Tiled when `--tiled-uri` is given. A key that cannot be read is listed under `unreadable_keys`, and the evaluators score what they can.

From Python:

```python
from asyncroscopy.evaluate import load_run, score_run, score_event, features

data = load_run("669e81c5855d", "outputs/trace", tiled_uri="http://127.0.0.1:9091")
score_run(data)                          # {"information": {...}, "resolution": {...}, "dose": {...}, "time": {...}}
score_event(data.events[3], data)        # one operation
features(data)                           # one flat row
```

## What is evaluated

Each evaluator works at two levels. `score_event(event, data)` scores one operation: an acquisition, a job step or a tool call. It returns `{}` when the event has nothing to say about its objectives. `score_run(data)` scores the whole sequence. `RunData` holds the events in trace order and reads images lazily through `asyncroscopy.data.data_reader.read_acquisition`. On a twin, it can also hold the ground truth.

| Evaluator | Objectives (better) | With the twin's ground truth | Without it |
| :-- | :-- | :-- | :-- |
| `information` | `features_found` (max), `features_in_focus` (max), `features_per_s` (max), `features_per_dose` (max), `recall` (max), `images` | Ground-truth particles that a detected feature lands on, each counted once per run however often it was imaged. `features_in_focus` counts only frames taken within 20 nm of focus. | Features with sharp edges according to pyTEMlib's `baseline` scorer (`pyTEMlib.scoring`), summed over the images. This is a proxy: a particle imaged twice counts twice. |
| `resolution` | `focus_error_m` (min), `in_focus_fraction` (max), `sharpness` (max) | The hidden `hypot(C1, |A1|)` behind each frame, if the bench recorded it (`focus_by_acquisition`) | The pyTEMlib `baseline` score (0-1) |
| `dose` | `max_dose_e_per_a2`, `mean_dose_e_per_a2`, `over_threshold_fraction`, `job_dose_e_per_a2` (all min), `exposed_area_m2` | It also reports the twin's own dose map maximum | Each frame's current × dwell / pixel area, taken from the data's attributes and laid on a grid at the frame's field. The dose that jobs report (the Tuner's `dose_e_per_a2`) is counted, but not placed. |
| `time` | `wall_s`, `clock_s`, `idle_s`, `idle_fraction` (min), `busy_s` | Not used | `clock_s` is the instrument clock span (simulated seconds on a twin). `busy_s` is the union of call and step durations. `idle_s` is wall time with nothing running: thinking, waiting and approvals. |

Per unit of time means per `clock_s` of the run, unless the caller gives `duration_s`. Per unit of dose means per e/Å² summed over frames. Options are constructor keyword arguments, for example `get_evaluator("dose", threshold_e_per_a2=500)` or `get_evaluator("information", in_focus_m=10e-9)`. If an evaluator raises, its result is `{"error": ...}`, and the other evaluators still run.

### What is twin-only, and what is circular

- **Only the twin has ground truth.** The particle positions and the hidden focus error are the twin's own model. Scoring a run "against the truth" therefore measures how well the run did on this twin. It does not measure how the run would do on a real specimen. If the twin is wrong about how blur hides a particle, the score is wrong in the same way, and nothing here can notice.
- **The detector is plain.** A feature is a local maximum of a smoothed, background-subtracted image that stands 5 noise standard deviations above the background. It counts as found if it lies within the particle's radius. The twin's particles are compact, bright blobs, and this detector suits them. A different specimen would need a different finder.
- **The pyTEMlib fallback is a proxy.** The `baseline` scorer measures edge sharpness and contrast. Its own documentation says it has been checked against synthetic maps, not against what a microscopist would choose.
- **`focus_by_acquisition` comes from the bench.** The twin records the true field centre of every acquisition, but not the focus error at that moment. `benchmarks/tune_vs_acquire.py` records the focus error itself. Without it, `resolution` falls back to the image score.

## Add an evaluator

An evaluator is one class. Register it with `@register_evaluator`, or from another package through the `asyncroscopy.evaluators` entry point:

```python
from asyncroscopy.evaluate import Evaluator, Score, register_evaluator

@register_evaluator
class Revisits(Evaluator):
    """How often the beam went back to a field it had already imaged: a damage risk and wasted time."""

    name = "revisits"
    description = "Frames whose field overlaps an earlier frame of the same run"
    objectives = {"revisits": ("frames", "min"), "revisit_fraction": ("fraction", "min")}
    option_names = ("overlap",)
    overlap = 0.5  # fraction of a field of view that counts as the same field

    def score_event(self, event, data):
        images = data.images(event)
        return {"frames": Score(len(images), "frames", "images this event produced")} if images else {}

    def score_run(self, data):
        seen, revisits, frames = [], 0, 0
        for _event, image in data.all_images():
            centre, fov = data.centre_m(image), image.fov_m
            if centre is None or fov is None:
                continue
            frames += 1
            near = any(abs(centre[0] - x) < self.overlap * fov[0] and abs(centre[1] - y) < self.overlap * fov[1] for x, y in seen)
            revisits += near
            seen.append(centre)
        return {
            "revisits": Score(revisits, "frames", f"{revisits} of {frames} frames overlap an earlier one by > {self.overlap:g} FOV"),
            "revisit_fraction": Score(revisits / frames if frames else None, "fraction", "revisits / frames"),
        }
```

```toml
# in another package's pyproject.toml
[project.entry-points."asyncroscopy.evaluators"]
revisits = "my_package.evaluators:Revisits"
```

Then run `python -m asyncroscopy.evaluate --run <id> --evaluators revisits,dose`. When the score changes meaning, bump `version`, so that scores from different versions are not mixed. `tests/test_evaluate.py` contains a shorter example that also checks a failing evaluator.

## Objectives that conflict, and a team's own reward

The objectives pull against each other:

- More frames find more particles, but they cost dose (`dose.over_threshold_fraction`) and time.
- More tuning improves `resolution`, but it takes time that `information` could have used (see below).
- Finding a particle and resolving it are different goals. A 60 nm defocus still lets the detector find most of the twin's particles (`features_found`), but none of those frames counts as `features_in_focus`.

A team writes its trade-off as weights:

```python
from asyncroscopy.evaluate import composite, register_evaluator

register_evaluator(composite("careful", {
    "information.features_found": 1.0,
    "dose.over_threshold_fraction": -200.0,   # each 1 % of the area over the dose budget costs 2 features
    "time.idle_s": -0.1,
}))
```

`reward = Σ weight × value`. If an objective could not be computed for a run (no images, no clock), it is left out of the sum and named in the reason, so it is never treated as zero by accident. With `strict=True`, the reward is null instead. On the command line, `--weights` builds the same thing.

## Features for learning a reward

`features(run)` returns one flat row per run, made only of numbers and strings:

- `run.*` columns are counted from the events alone: `events`, `tool_calls`, `job_steps`, `jobs`, `failed`, `approvals`, `dry_run_events`, `acquire_calls`, `distinct_actions`, `simulated`.
- `<evaluator>.<objective>` columns hold every evaluator's run score. `feature_units()` gives their units.

`features_table(runs)` makes the rows for many runs, and `write_table(rows, "runs.csv")` saves them. To learn a reward, add an outcome column per run, such as an operator's rating or a later analysis result. Then regress the outcome on the columns, or fit the weights of a `composite`. Nothing in this package learns. It only makes trace → features → score a clean table. `--features` on the command line prints one run's row.

The workflow benchmark (`benchmarks/run_workflows.py`) stores each row's evaluator scores under `scores`. By default it uses `time`, which reads no data. Use `--evaluators information,resolution,dose,time` to read the images from the stack's Tiled. `--score-columns` adds their medians to the Markdown table. The scores never change whether a row passes.

## How long to tune before acquiring

`benchmarks/tune_vs_acquire.py` is the organisers' principle 4 as a bench. It needs no LLM and no stack. It starts a Tango test context in its own process, with the twin, a SCAN device and the real `Tuner`.

1. **Starting column.** Each trial has a fixed instrument-time budget, **T = 120 s**. The column starts with a known hidden defocus C1 and astigmatism A1, and optionally a focus that drifts.
2. **Tuning.** The Tuner (`image_metric`, focus and stigmation) runs on a sacrificial field next to the map, with a search budget from `none` to `l`:

   | Budget | Tuning images | Tuning time (s) |
   | :-- | --: | --: |
   | `none` | 0 | 0 |
   | `focus_only` | 10 | 5 |
   | `xs` | 26 | 13 |
   | `s` | 38 | 20 |
   | `m` | 50 | 26 |
   | `l` | 95 | 49 |

3. **Acquisition.** The rest of T goes on a fixed map of 100 nm tiles (256 px, 4 µs dwell), tile after tile, until no further tile fits.
4. **Scoring.** The evaluators score the trial from its trace and its data. Each point is the mean of 2 repeats.

Each frame also costs 0.5 s of overhead, the figure the Tuner's own estimate uses, and each stage move costs 1 s. The full study takes about 3 minutes; `--tiny` takes about 20 s. Results are in `benchmarks/results/tune_vs_acquire.json` and `.md`.

The bench adds two things that the twin does not model:

- **Focus drift.** The twin drifts the stage, not the focus. `FocusDriftTwin` adds `rate × (clock − t0)` to C1.
- **Per-frame overhead.** The twin's clock counts only frame time and stage moves.

The results below came from this checkout (`benchmarks/results/tune_vs_acquire.md`). `found` is the number of unique ground-truth particles found (twin), and `in focus` is `features_in_focus`.

| Budget | Tuning (s) | Map tiles | Small error (C1 30 nm, \|A1\| 10 nm): found / in focus | Large error (C1 250 nm, \|A1\| 50 nm) | Large error, drift 0.3 nm/s | Large error, drift 2 nm/s |
| :-- | --: | --: | --: | --: | --: | --: |
| `none` | 0 | 67 | **1353** / 0 | 265 / 0 | 194 / 0 | 49 / 0 |
| `focus_only` | 5.2 | 64 | 1299 / 0 | **1259** / 0 | **1204** / 0 | 713 / 0 |
| `xs` | 13.4 | 59 | 1185 / 0 | 1181 / 0 | 1151 / 0 | 665 / 0 |
| `s` | 19.6 | 56 | 1133 / **1133** | 1133 / **1133** | 1133 / **603** | **766** / 0 |
| `m` | 25.8 | 52 | 1045 / 1045 | 1045 / 1045 | 1044 / 436 | 729 / 0 |
| `l` | 49.1 | 39 | 781 / 781 | 781 / 781 | 780 / 352 | 511 / 0 |

Optima are in bold. Read the table as follows:

- **The starting error decides whether tuning pays at all.** With C1 = 30 nm, the blur hides almost nothing from the detector, so the best way to count particles is not to tune: every tuning second is a lost tile. From 250 nm, a 5 s focus-only tune finds 4.8 times as many particles (265 → 1259).
- **The objective decides how much tuning pays.** `focus_only` leaves the 50 nm of astigmatism, so no frame is within 20 nm of focus. If resolved particles are the goal, the optimum is `s` (20 s, 16 % of T), which also corrects A1. Beyond the optimum, tuning only costs tiles: `l` spends 41 % of T and finds 31 % fewer particles than `s`.
- **Drift makes tuning go stale.** At 0.3 nm/s, the focus leaves the 20 nm window about 50 s after tuning, so `s` gets only 603 particles in focus instead of 1133. At 2 nm/s, the focus drifts 10 to 100 nm while the Tuner is still running, and no frame is in focus whatever the budget. One tune before the map is then the wrong plan; the map has to re-tune as it goes, which this bench does not try. For counting particles, the fast-drift optimum moves from `focus_only` to `s` (766 against 713), because a better start keeps the error under the detection limit for longer. That difference is close to the spread between the two repeats (683 and 743 for `focus_only`).
- **The twin is generous.** Every Tuner run succeeded, and even `xs` brought a 250 nm error down to 39 nm. On a real column, with noisier sharpness metrics, the small budgets would fail more often (`peak_at_edge`, `sharpness_fell`). They would cost their time and leave the focus where it was.

The policy helper `asyncroscopy.evaluate.policy` writes the stopping rule as pure functions. It does not drive the Tuner; an agent or a job can call it:

```python
from asyncroscopy.evaluate import keep_tuning, stale_after_s

stale = stale_after_s(depth_of_field_m=20e-9, focus_drift_m_per_s=2e-9, residual_m=5e-9)   # 7.5 s
decision = keep_tuning(rate_now_per_s=0.5, rate_tuned_per_s=12.0, step_s=13.0, remaining_s=110.0, stale_after=stale)
decision.keep_tuning, decision.reason
```

The rule is: tune one more step while `(rate_tuned − rate_now) × min(remaining − step, stale_after) / step > rate_now`. In words, keep tuning while the information that tuning step adds, per second spent tuning, is more than what a second of acquiring now would give. The bench feeds the rule the rates it measured and compares the rule's answer with what actually paid off (the last two columns of the tables). Those rates come from the same trials, so the comparison checks the rule's arithmetic and its staleness term, not its ability to predict. An agent would have to estimate the rates in advance, for example with the `information` evaluator on its last few frames and a model of how the rate falls with the focus error.
