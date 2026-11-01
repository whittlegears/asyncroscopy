---
name: best_region_spectrum
description: Map an area, find its most interesting regions, then take an EDS spectrum at the best region and read which elements it shows.
version: 1.0.0
min_asyncroscopy: 0.1.0
envelope_schema: 1.0.0
keywords: [most interesting, interesting regions, best region, best one, best spot]
requires: [Mapper*_reserve, list_pois, find_features, "*_acquire_spectrum", "*_place_beam_at", view_data]
acts: true
tools: [run_job, wait_for_job, run_skill, wait_for_skill]
run:
  inputs: {map: null, top_k: 3}
  answer: "Answer now from outputs: the regions are regions pois (x_m, y_m, score, reason), best first; the spectrum was taken on the feature (feature x_m, y_m) found at the first region, and its elements are on the peaks line 'elements most consistent with these peaks' (name them with their shares)."
  steps:
    - {id: regions, tool: list_pois, args: {map: "{inputs.map}", kind: candidate, limit: "{inputs.top_k}"}}
    - {id: stage, tool: "*_get_stage"}
    - {tool: "*_move_stage", args: {position: ["{regions.pois.0.x_m}", "{regions.pois.0.y_m}", "{stage.2}", "{stage.3}", "{stage.4}"]}}
    - {skill: find_feature_then_measure}
---

# A spectrum at the best region of a new map

1. The map, scored: `run_job` with `{"device": "Mapper", "plan": {"size_m": [w, h], "find_regions": N}}` (N regions the
   user asks for, default 3; see the mapping skill for the area). Wait for it with `wait_for_job` until `done`.
2. One call: `run_skill` with `{"skill": "best_region_spectrum", "inputs": {"map": <result_keys.map>, "top_k": N}}`, then
   `wait_for_skill` with its `run_id`. It lists the map's regions, moves to the best one, images it, parks the beam on
   the most distinct feature there (a region's centre can be bare support), takes the spectrum, blanks the beam and
   reads the peaks.

Answer from its `outputs`: each region's `x_m`, `y_m`, score and reason, and the feature at the best one with its
elements. If it `stopped`, say at which step and why; a map without regions was made without `find_regions`.
