---
name: stem-focus
description: Use when the operator asks to focus the microscope, says images look blurry or soft, or after a stage move or tilt that may have changed the sample height. Brings a STEM image into focus with the calibration procedures server.
requires_tools: [sweep_defocus, focus_stage_z, procedure_limits]
---
# Focusing a STEM image

Two procedures do the work. Each is one tool call that acquires images, scores them,
and leaves the instrument at the best setting. If anything fails part-way, each puts
the instrument back where it started.

- `sweep_defocus` changes the objective lens. It is fast and the right first choice
  when the image is only slightly out of focus.
- `focus_stage_z` moves the stage height. Use it when the sample itself is at the
  wrong height: after loading a sample, after a large stage move, or after tilting.

## Steps

1. Call `procedure_limits` once and keep the limits in mind. A request beyond them
   is refused, never clipped.
2. Choose the metric. Use `edge_sharpness` or `high_frequency_power` for focus.
   `contrast` can rise with some aberrations, so it is a poor focus signal on its own.
3. Make sure the pixels are small enough to show focus changes. If the field of view
   is large compared with the features of interest, ask the operator before changing
   magnification.
4. First try `sweep_defocus` with a modest `half_range_m` (20 nm) and `steps` 9 to 11.
5. Read the result:
   - `fit_has_maximum_in_range: true`: done. Report `chosen_defocus_m` and the score.
   - `fit_has_maximum_in_range: false`: the best point is near an edge of the range,
     and the setting applied is the best measured one. Run once more; the next sweep
     is centred on the new setting.
6. If the defocus needed is large, or the operator says the sample moved, use
   `focus_stage_z` instead. Start with `step_m` 20 to 50 nm.
7. Read `outcome`:
   - "optimum found inside the explored range": done.
   - "the optimum may lie further out": run once more from here.
   - "no improvement": every image scored the same. The sample may be far out of
     focus, or there may be no features in view. Do not keep retrying. Tell the
     operator and suggest a larger `step_m` or a feature-rich area.

## Rules

- Run at most two procedure calls in a row without reporting back to the operator.
- Report in plain words: what changed, by how much, and the score before and after.
- Never call the low-level stage or lens tools to "nudge" focus yourself. The
  procedures enforce the limits and the undo; individual calls do not.
