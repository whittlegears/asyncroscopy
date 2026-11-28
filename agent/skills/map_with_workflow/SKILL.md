---
name: map_with_workflow
description: Map an area as a grid of images and stitch it into one map with the map_grid workflow, on an instrument that has no Mapper device. One run_workflow call, never a loop of moves and images.
version: 1.0.0
min_asyncroscopy: 0.1.0
keywords: [map, mapping, grid, tiles, stitch, mosaic, montage, overview]
requires: [run_workflow]
unless: [Mapper*_reserve]   # with a Mapper, the mapping skill maps
new_map: true
acts: true
tools: [run_workflow, workflow_status, answer_workflow, stop_workflow, resume_workflow, map_info, get_region]
---

# Mapping with the map_grid workflow

A map is ONE `run_workflow` call. Never loop over `move_stage`, `set_image_shift` and `acquire_scanned_image` yourself, and never fall back to that when the workflow is refused or fails: tell the user what it said.

## 1. Inputs

Read the field of view first (`<Microscope>_get_fov`, metres) and pass it as `fov_m`, so the map is at the magnification the user is looking at. Left out, `fov_m` is 300 nm, which changes the magnification.

| Request says | `inputs` |
| :-- | :-- |
| "a 4 by 4 grid", "4 x 4 map", "N by N tiles" | `{"tiles_per_side": 4, "fov_m": <get_fov>}`. Do not work out a size. |
| "a 2 um square", "map 500 nm around here" | `{"size_m": 2e-6, "fov_m": <get_fov>}` |
| a field of view per tile ("300 nm tiles") | that value as `fov_m` |
| nothing about overlap | leave `overlap` out (0.3) |

Lengths are metres: 1 um = 1e-6, 1 nm = 1e-9. A value the user states goes in unchanged; an input outside its range is refused, not adjusted: tell the user the range.

Tiles are reached by image shift, so the stage does not move. That reaches 1 um from the stage position on each axis: the mapped square can be at most about 2 um plus one field of view across. For a larger area add `"move_by": "stage"`. A grid that does not fit is refused before anything moves; tell the user, do not shrink it yourself.

## 2. Run and wait

1. `run_workflow` with `{"name": "map_grid", "inputs": {...}}`. It returns a `run_id`; the workflow runs by itself.
2. While `state` is `running`, call `workflow_status` with `{"run_id": ..., "wait_s": 100}`. Each call waits up to 100 s, so a map takes a few calls. A tile takes about 25 s on the microscope: 9 tiles about 4 minutes, 16 tiles about 7.
3. `next` in every reply says what to do. `waiting`: the workflow asks the user a question; put it to them and call `answer_workflow` with their answer, never your own. `failed` or `stopped`: say why; `resume_workflow` continues at the next tile once the cause is fixed.

"A scan is already running" means the microscope's own software is scanning (its live view): ask the user to stop that scan, then `resume_workflow`.

## 3. Report

When `done`, report from `outputs`:

- `outputs.stitch.map`: the stitched map (a file path). This is "the map".
- `outputs.stitch.summary`: `tiles`, `matches_used` (overlaps that matched) and `unplaced_tiles` (tiles placed from the stage position only, because their overlaps did not match). Say how many tiles are unplaced; do not call the map seamless when there are some.
- The instrument is left as it was found: image shift and field of view put back, beam blanked.

To look at the map: `map_info` with `{"map": <that path>}` gives its `stage_box_m`; `get_region` with that map and a `box_m` returns a picture of that part (a small box gives full resolution). Show the user the whole map once before describing it.

Images already taken one by one cannot be turned into a map afterwards by this workflow: it takes its own tiles.
