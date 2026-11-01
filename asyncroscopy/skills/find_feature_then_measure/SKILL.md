---
name: find_feature_then_measure
description: Find a feature (a particle, a grain, a bright spot) in an image, then take a spectrum or a closer image on it and report where it is.
version: 2.1.0
min_asyncroscopy: 0.1.0
keywords: [find a particle, find a feature, find a grain, on it, on the particle, at the particle, locate, spot, cluster]
requires: ["*_acquire_scanned_image", view_data]
acts: true
tools: [run_skill, wait_for_skill, view_data, find_features, "*_get_fov", "*_unblank_beam", "*_blank_beam", "*_acquire_scanned_image", "*_acquire_spectrum", "*_place_beam_at", "*_move_stage", "*_get_stage", "*_set_fov"]
run:
  inputs: {fov_m: 3.0e-7}
  answer: "Answer now from outputs: the elements are on the peaks line 'elements most consistent with these peaks' (name them with their shares), the feature's position is feature x_m, y_m."
  steps:
    - {tool: "*_set_fov", args: {fov: "{inputs.fov_m}"}}
    - {tool: "*_unblank_beam"}
    - {id: survey, tool: "*_acquire_scanned_image", args: {detector_list: [haadf]}}
    - {tool: "*_blank_beam"}
    - {id: feature, tool: find_features, args: {image: "{survey}", n: 1}}
    - {tool: "*_unblank_beam"}
    - {tool: "*_place_beam_at", args: {position: ["{feature.features.0.x_m}", "{feature.features.0.y_m}"]}}
    - {id: spectrum, tool: "*_acquire_spectrum", args: {detector_name: eds}}
    - {tool: "*_blank_beam", always: true}
    - {id: peaks, tool: view_data, args: {key: "{spectrum}", format: text}, cut_at: "data (sums"}
---

# Find a feature, then measure it

**A spectrum on a feature** is one call: `run_skill` with `{"skill": "find_feature_then_measure"}`, then
`wait_for_skill` with the `run_id` it gives. It takes a 300 nm survey image, finds its most distinct feature, parks
the beam there, takes an EDS spectrum, blanks the beam and reads the spectrum. Give `"inputs": {"fov_m": ...}` only
when the request names a field of view. Its reply's `outputs` hold the survey image (`survey`), the feature with its
stage position (`feature`), the spectrum key (`spectrum`) and its peaks and likely elements (`peaks`): answer from
them without reading the spectrum again. If it `stopped`, say why in its words; when it found no feature, run it once
more with `"fov_m": 6e-7`, and if that finds none either, say so.

**A closer image of a feature**:
1. Image the area with `<Microscope>_acquire_scanned_image` at a field of view several times the feature (300 nm for
   particles), with the beam unblanked (`<Microscope>_unblank_beam` first, `<Microscope>_blank_beam` after).
2. `find_features` with `{"image": <key>, "n": 1}` gives the feature's stage position `x_m`, `y_m`.
3. `<Microscope>_get_stage` for z, alpha and beta, `<Microscope>_move_stage` to `[x_m, y_m, z, alpha, beta]`, a smaller
   field of view with `<Microscope>_set_fov`, then unblank, `<Microscope>_acquire_scanned_image`, blank.

Report the feature's position (metres), the key of the measurement and what it shows. Never measure a random spot.
