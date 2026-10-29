# Autofocus and stigmation (the Tuner job)

The `Tuner` device (`asyncroscopy/instruments/electron_microscope/tuner.py`) runs autofocus and astigmatism correction as one background job with the same envelope as the Mapper: `reserve`, `start`, `job_status`, `abort`, `results`, `release`. An agent calls it in one go:

```json
run_job {"device": "Tuner", "plan": {"corrections": ["focus", "stigmation"], "strategy": "image_metric", "dose_budget_e_per_a2": 5000}}
```

The agent decides when to tune and the job runs the loop. A focus sweep is dozens of set, image and score calls. Sent through an LLM, each one takes seconds and, in approve mode, each one is held. The job is approved once, as a whole.

## Plan

| Field | Default | Meaning |
| :-- | :-- | :-- |
| `corrections` | `["focus"]` | `focus` (C1), `stigmation` (A1), or both |
| `strategy` | `optistem` | One of the registered strategies (see below) |
| `dose_budget_e_per_a2` | required | Most dose the tuning may put on the field |
| `max_steps` | 80 | Most measurements: images, vendor runs and corrector measurements |
| `focus_range_m`, `stig_range_m` | 3e-7, 1e-7 | No defocus or stigmator value leaves snapshot ± this |
| `max_step_m` | 1e-7 | Largest change per write; longer moves are ramped |
| `fov_m`, `imsize`, `dwell_s`, `detector` | 1e-7, 256, 2e-6, haadf | Imaging settings for the tuning images |
| `metric` | `normalized_variance` | Sharpness metric (`asyncroscopy/routines/focus.py`) |
| `coarse_points`, `fine_points`, `rounds`, `stig_passes` | 13, 9, 2, 1 | Search shape for `image_metric` |
| `corrector_iterations`, `corrector_gain`, `corrector_target_m`, `error_tolerance_m` | 5, 1.0, 2e-9, 1e-9 | Loop settings for `corrector` |

`reserve` returns an estimate: the number of frames, the worst-case `max_dose_e_per_a2`, and the beam current with its source. `results` returns `summary.tune`, which holds the sharpness before and after, the dose used, the snapshot, what was restored, and the last measurements.

## Strategies

- **`optistem`**, the default (decision D2). Runs the microscope's `auto_focus` once: OptiSTEM C1A1 through AutoScript on the Spectra 300, or the twin's stand-in. It corrects C1 and A1 together. Its dose counts as `autofocus_frames` frames, a device property to calibrate at first light.
- **`image_metric`**. A coarse sweep over the defocus range, then two fine sweeps re-centred on the estimate, each fitted with a parabola. The score near focus is flat, so the information is in the flanks. A sweep that is symmetric about the peak cancels the misfit of the parabola. Stigmation runs the same search along each stigmator component in turn. It needs `set_stigmator` and `get_stigmator` on the microscope; the twin has them, and AutoScript does not yet. A peak at the edge of the range fails the job instead of settling there.
- **`corrector`**. Measures C1 and A1 with the corrector (`measure_c1a1`), sends a correction bounded by `max_step_m` (`correct_aberration`), and measures again. A JSON-RPC reply that contains `error` counts as a failure. On a stack that is not simulated, this strategy is refused until `allow_corrector_writes` is set, which happens only after the first-light sign test.

On the twin, the column has a hidden focus error that `get_defocus` does not report. Set it with the twin's `focus_offset_m` and `astigmatism_offset_a_m`/`_b_m` properties, or with the `set_focus_error` method in tests. The twin's images blur with it, and `export_ground_truth()["focus"]` gives the residual C1 and A1 at the field, including the specimen height. The twin corrector's measurements are noisy by default. Its `correction_sign: -1` property makes a correction add instead of subtract.

## Safety

- **Snapshot and restore.** The defocus, stigmator and imaging settings are read before anything changes. Every corrector correction is recorded once it is sent. On an abort or a failure, the optics go back in bounded steps, and the corrector receives the negated total of what was sent. The imaging settings are always restored. The tuned optics stay only when the job succeeds.
- **Error growth.** A corrector correction that makes the measured error larger stops the job (`error_grew`), for example when the sign is wrong. The job also fails if the image is less sharp after tuning than before (`sharpness_fell`).
- **Bounds.** Every write stays within the plan's ranges and `max_step_m`, and every measurement counts against `max_steps`.
- **Beam.** The beam is blanked when the job starts. It is unblanked only for an image, a vendor run or a corrector measurement, and blanked again in `finally`.
- **Dose.** `reserve` refuses a plan whose worst case is over its budget, before the beam is touched. While the job runs, each exposure is counted before it is taken. An exposure that would pass the budget fails the job instead. On a stack that is not simulated, a plan that would rely on the default beam current is refused, as the Mapper does.
- **Resume.** A job is one step, so restarting a failed or aborted job begins again from a fresh snapshot.

## Adding a strategy

A strategy is one class. It never calls Tango; it tunes through the `TuneContext` it is given, which enforces every rule above.

```python
from asyncroscopy.instruments.electron_microscope.tuner import Strategy, register_strategy

@register_strategy
class HillClimb(Strategy):
    name = "hill_climb"
    corrections = frozenset({"focus"})

    def exposures(self, plan, frames_per):      # most frames of dose run() may use
        return 40

    def run(self, ctx):
        here, step = ctx.snapshot["defocus_m"], ctx.plan["focus_range_m"] / 3
        best = ctx.sharpness_at(defocus=here)
        ...                                     # ctx.sharpness_at / set_defocus / set_stigmator /
        ctx.set_defocus(here)                   # auto_focus / corrector_measure / corrector_correct
        return {"defocus_m": here}
```

A strategy can also declare `needs` (`"stigmator"`, `"autofocus"`, `"corrector"`), override `check(plan, capabilities)`, and override `measurements(plan, frames_per)` when one measurement is worth several frames. A package can install a strategy without editing asyncroscopy, through the `asyncroscopy.focus_strategies` entry point. `tests/test_tuner.py` registers a 40-line example the same way and runs it on the twin.

The searches and metrics in `asyncroscopy/routines/focus.py` (`search_1d`, `search_2d`, `sharpness`, `parabola_vertex`, `error_grew`) are pure functions, and a strategy can reuse them.
