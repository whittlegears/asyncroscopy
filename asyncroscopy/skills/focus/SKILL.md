---
name: focus
description: Autofocus and correct astigmatism (stigmation) with one Tuner job instead of stepping the defocus yourself.
version: 1.0.0
min_asyncroscopy: 0.1.0
envelope_schema: 1.0.0
keywords: [focus, autofocus, defocus, refocus, out of focus, blurry, blurred, astigmatism, stigmation, stigmate, stigmator, optistem, sharpen]
requires: [Tuner*_reserve]
acts: true
tools: [run_job, wait_for_job, Tuner_abort, approval_status]
---

# Focus and stigmation

Focusing is ONE background job on the Tuner. Never loop over `set_defocus` and `acquire_scanned_image` yourself: each call would be slow, held for approval, and nothing would put the focus back if it went wrong. The job bounds every step, counts the dose, blanks the beam between images, and restores the starting focus if it fails or is aborted.

## 1. Plan

| Request says | Plan |
| :-- | :-- |
| "focus", "autofocus", "it looks blurry" | `{"corrections": ["focus"], "dose_budget_e_per_a2": 3000}` (the default strategy is the microscope's own OptiSTEM) |
| "and fix astigmatism", "stigmate" | `"corrections": ["focus", "stigmation"]` |
| "using the images", "image-based" | `"strategy": "image_metric"` (dose about 1400 e/A^2 for focus, about 4000 with stigmation) |
| "with the corrector" | `"strategy": "corrector"` (refused on the instrument until the corrector's sign is confirmed) |
| a dose limit ("keep it under 500 e/A^2") | `"dose_budget_e_per_a2": 500` |

`dose_budget_e_per_a2` is required. A plan over it is refused before the beam is unblanked, with the dose it needs: tell the user the number and ask, do not raise it yourself.

## 2. Run and wait

1. `run_job` with `{"device": "Tuner", "plan": {...}}`. In approve mode the operator sees the plan first.
2. `wait_for_job` with the `job_id` until the state is `done`, `failed` or `aborted`.
3. Read `summary.tune`: `sharpness_before` and `sharpness_after`, `dose_e_per_a2`, and `defocus_after_m`.

## 3. When it does not finish

- `failed` with `peak_at_edge`: the focus is further off than the search range. Ask before retrying with a larger `"focus_range_m"` (default 3e-7).
- `failed` with `error_grew`: the corrector's correction has the wrong sign. The starting state was restored. Do not retry; tell the user.
- `failed` with `dose_budget` or `max_steps`: the strategy needed more than the plan allowed. Report it.
- In every case the starting focus was restored and the beam blanked; say so.
