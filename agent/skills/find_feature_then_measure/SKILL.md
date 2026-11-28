---
name: find_feature_then_measure
description: Find a feature (a particle, a grain, a bright spot) in an image, then take a spectrum or a closer image on it and report where it is.
version: 3.0.0
min_asyncroscopy: 0.1.0
keywords: [find a particle, find a feature, find a grain, on it, on the particle, at the particle, locate, spot, cluster]
requires: ["*_acquire_scanned_image", view_data]
acts: true
tools: [run_workflow, workflow_status, view_data, find_features, "*_get_fov", "*_unblank_beam", "*_blank_beam", "*_acquire_scanned_image", "*_acquire_spectrum", "*_place_beam_at", "*_move_stage", "*_get_stage", "*_set_fov"]
---

# Find a feature, then measure it

## A spectrum on a feature

This is the `spectrum_on_feature` workflow: `run_workflow` with `{"name": "spectrum_on_feature"}`, and
`workflow_status` with its `run_id` while it is `running`. It takes a 300 nm survey image, finds the most distinct
feature, parks the beam there, takes an EDS spectrum, blanks the beam, reads the spectrum and puts the field of view
back. Add `"inputs": {"fov_m": ...}` only when the request names a field of view.

When it is `done`, answer from its `outputs` without reading the spectrum again:
- `outputs.feature.x_m`, `outputs.feature.y_m`: where the feature is (metres); `outputs.survey.key`: the survey image.
- `outputs.spectrum.key`: the spectrum; `outputs.peaks.elements`: the elements most consistent with its peaks, with
  their shares; `outputs.peaks.peaks`: the peaks themselves.

If it `failed` at `feature`, the survey showed nothing distinct: run it once more with `"fov_m": 6e-7`, and if that
finds nothing either, say so. Any other failure: say where and why, in its words.

Only when `run_workflow` is not among your tools, do the same steps yourself: `<Microscope>_set_fov` 3e-7,
unblank, `<Microscope>_acquire_scanned_image` with HAADF, blank, `find_features` with `{"image": <key>, "n": 1}`,
unblank, `<Microscope>_place_beam_at` the feature's `[x_m, y_m]`, `<Microscope>_acquire_spectrum` with EDS, blank,
then `view_data` on the spectrum as text.

## A closer image of a feature

1. Image the area with `<Microscope>_acquire_scanned_image` at a field of view several times the feature (300 nm for
   particles), with the beam unblanked (`<Microscope>_unblank_beam` first, `<Microscope>_blank_beam` after).
2. `find_features` with `{"image": <key>, "n": 1}` gives the feature's stage position `x_m`, `y_m`.
3. `<Microscope>_get_stage` for z, alpha and beta, `<Microscope>_move_stage` to `[x_m, y_m, z, alpha, beta]`, a smaller
   field of view with `<Microscope>_set_fov`, then unblank, `<Microscope>_acquire_scanned_image`, blank.

Report the feature's position (metres), the key of the measurement and what it shows. Never measure a random spot.
