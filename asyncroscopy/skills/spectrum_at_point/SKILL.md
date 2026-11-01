---
name: spectrum_at_point
description: Take an EDS spectrum at a stage position already known (a region of a map, a point of interest, coordinates given) and read which elements it shows.
version: 1.0.0
min_asyncroscopy: 0.1.0
matched: false
requires: ["*_acquire_spectrum", "*_place_beam_at", "*_move_stage", view_data]
acts: true
tools: [run_skill, wait_for_skill]
run:
  inputs: {x_m: null, y_m: null}
  answer: "Answer now from outputs: the elements are on the peaks line 'elements most consistent with these peaks' (name them with their shares), taken at inputs x_m, y_m."
  steps:
    - {id: stage, tool: "*_get_stage"}
    - {tool: "*_move_stage", args: {position: ["{inputs.x_m}", "{inputs.y_m}", "{stage.2}", "{stage.3}", "{stage.4}"]}}
    - {tool: "*_unblank_beam"}
    - {tool: "*_place_beam_at", args: {position: ["{inputs.x_m}", "{inputs.y_m}"]}}
    - {id: spectrum, tool: "*_acquire_spectrum", args: {detector_name: eds}}
    - {tool: "*_blank_beam", always: true}
    - {id: peaks, tool: view_data, args: {key: "{spectrum}", format: text}, cut_at: "data (sums"}
---

# A spectrum at a known position

One call: `run_skill` with `{"skill": "spectrum_at_point", "inputs": {"x_m": <x>, "y_m": <y>}}`, the position in metres
copied exactly from where it came from (a region's or point's `x_m`, `y_m`), then `wait_for_skill` with its `run_id`.
It moves the stage there (keeping z and tilts), parks the beam on the point, takes the spectrum, blanks the beam and
reads the peaks. Answer from its `outputs.peaks` without reading the spectrum again; if it `stopped`, say why.
