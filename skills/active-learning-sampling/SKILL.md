---
name: active-learning-sampling
description: Use when the operator wants a spectroscopic or other slow measurement (EELS, EDS, diffraction) mapped over an area but cannot afford to measure every position. Picks the next probe positions from the structure image with deep-kernel active learning on the ML server.
requires_tools: [dkl_suggest_locations]
---
# Active-learning sampling with a structure image

Slow measurements are worth spending where they teach the most. `dkl_suggest_locations`
learns how the measured value relates to the local structure in a fast image (HAADF),
then says where to measure next. It never moves the instrument; you place the beam and
measure with the instrument's own tools.

## Steps

1. Acquire one structure image of the area and keep its data key. Every position below
   is a (row, column) pixel of this image.
2. Agree with the operator what single number each measurement gives (for example an
   edge intensity ratio, a thickness, a peak position) and how many measurements the
   budget allows.
3. Measure a few positions spread over the image (at least 3, better 8 to 10). Convert
   each pixel to a beam position the instrument understands, and record
   [row, column, value].
4. Call `dkl_suggest_locations` with the structure key, `dataset_key` (for example
   `HAADF`) and the measurements. Choose `acquisition`:
   - `ei` to find where the value is highest;
   - `max_variance` to map the whole area as evenly as possible;
   - `ucb` for a mix (larger `beta` explores more).
5. Measure at the suggested positions, add them to the list, and call again. Stop at
   the budget, or when the returned `predicted_std` values are small compared with the
   spread of the measured values.
6. Report the measured values and the paths in `mean_map_file` and `std_map_file`
   (the predicted map and its uncertainty on the candidate grid).

## Rules

- The value must come from a real measurement at that position, never from the
  structure image itself.
- The model needs structure that relates to the value. If predictions do not improve as
  measurements are added, tell the operator rather than continuing.
- Positions are suggestions; check each one is inside the region the operator allows
  before moving the beam.
