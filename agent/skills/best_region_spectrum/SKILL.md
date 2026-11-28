---
name: best_region_spectrum
description: Map an area, find its most interesting regions, then take an EDS spectrum at the best region and read which elements it shows.
version: 2.0.0
enabled: true
min_asyncroscopy: 0.1.0
envelope_schema: 1.0.0
keywords: [most interesting, interesting regions, best region, best one, best spot]
requires: [Mapper*_reserve, run_workflow]
acts: true
tools: [run_job, wait_for_job, run_workflow, workflow_status]
---

# A spectrum at the best region of a new map

1. The map, scored: `run_job` with `{"device": "Mapper", "plan": {"size_m": [w, h], "find_regions": N}}` (N regions the
   user asks for, default 3; `size_m` is the side of the area in metres, centred on the stage). Wait for it with
   `wait_for_job` until `done`.
2. The spectrum is the `spectrum_at_best_region` workflow: `run_workflow` with
   `{"name": "spectrum_at_best_region", "inputs": {"map": <result_keys.map>, "top_k": N}}`, and `workflow_status` with
   its `run_id` while it is `running`. It lists the map's best regions, moves to the first, takes a survey image there,
   parks the beam on the most distinct feature (a region's centre can be bare support), takes the spectrum, blanks the
   beam, reads the peaks and puts the field of view back.

When it is `done`, answer from its `outputs`: each region in `outputs.regions.regions` (`x_m`, `y_m`, `score`,
`reason`, best first), the feature it measured (`outputs.feature.x_m`, `outputs.feature.y_m`), the spectrum's key
(`outputs.spectrum.key`) and its elements (`outputs.peaks.elements`). If it `failed`, say at which node and why; a
map with no regions was made without `find_regions`.
