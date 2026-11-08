---
name: mapping
description: Map a sample or an area as a grid of images (optionally with an EDS spectrum at each tile), stitch it into a zoomable map, find interesting regions and go back to them.
version: 1.6.3
min_asyncroscopy: 0.1.0
envelope_schema: 1.0.0
keywords: [mapping, overview, stitch, grid, tiles, survey, mosaic, montage, large area]
requires: [Mapper*_reserve]
acts: true
tools: [run_job, wait_for_job, Mapper_abort, approval_status, assemble_map, score_map, map_info, get_region, list_pois, add_poi, view_data, list_data]
---

# Mapping a sample

A map is ONE background job, not hundreds of moves. Never loop over `move_stage` and `acquire_scanned_image` yourself.

A whole request is: run the map job, wait for it, then stitch. "The map" means the stitched map key (`result_keys.map`, ending in `_map`), not the run file.

## 1. Plan

Turn the request into a plan. Lengths are metres (1 um = 1e-6, 1 nm = 1e-9).

| Request says | Plan |
| :-- | :-- |
| "the sample", "whole sample", "a sample completely" | `box_m` = sample bounds from `<Microscope>_get_parameters` (`world_bounds_angstrom` x 1e-10) |
| "a 500 nm square around here / the current position" | `"size_m": [5e-7, 5e-7]` (centred on the stage) |
| "a 500 nm square around a point (x, y)" | `"size_m": [5e-7, 5e-7], "centre_m": [x, y]` |
| nothing about EDS, spectra or composition | no `per_tile`: images only |
| "with EDS", "and a spectrum at each image" | `"per_tile": ["eds_point"]` (one spectrum at each tile centre) |
| "spectrum image", "EDS map" of every tile | `"per_tile": ["eds_map"]`. Only when asked by name: it multiplies time and dose. |
| nothing about size | `"fov_m": 2e-7`, `"overlap": 0.15` (a sparse sample stitches better with `"overlap": 0.25`) |
| "find the N most interesting regions" | `"find_regions": N` |
| "... and image them closer / at high magnification" | also `"revisit": {"fov_m": <a few times smaller than fov_m>}` |
| "at most N e/Å²", "a dose limit of N", "N electrons per square angstrom" | `"dose_budget_e_per_a2": N` if stated |
| "down to the angstrom" | `"fov_m": 5.12e-8` (512 px of 1 Å); not size or dose |

Another instrument (an AFM)? Follow the Mapper's example plan; settings go in `settings`, limits (`time_s`, ...) in `budgets`.

Every number the user states is a constraint: keep it in the plan. Never drop or relax one (dose, size, field of view) to make a plan pass. If a field is refused, fix its name or form from the message and keep the value; if the value itself is refused, tell the user the limit and ask. Say in your answer which constraints the plan carries.

If the request is unclear about the area, ask the user before starting.

## 2. Run the map

1. `run_job` with `{"device": "Mapper", "plan": {...}}`. It checks the plan and starts the job; the reply has `job_id` and `estimate`. An `error` means the plan was refused: fix what the message says and call `run_job` again. A plan whose estimate is over the endpoint's limits (duration or dose) is refused before it starts: tell the user the estimate and ask; do not shrink it on your own. Check the estimate against the request: if it is far larger than what was asked, call `Mapper_abort` and ask the user.
   - If it says **APPROVAL REQUIRED** with a `request_id`, the instrument is in approve mode. Call `approval_status` every few seconds. When it is `approved`, call `run_job` again with exactly the same arguments. If `denied` or `expired`, stop and tell the user.
   - An endpoint with several Mappers names them `Mapper_<key>`: use that name as `device`.
   - If the user says not to wait (they want the job id now, or will stop it), add `"wait": false`, tell them the `job_id` and stop there.
2. `wait_for_job` with `{"job_id": ...}`: it waits until the job is `done`, `failed` or `aborted`, however long that takes, and returns once (a short map may already be `done` in the `run_job` reply). If it says the job is still running after a long wait, tell the user it is running and its `job_id`. The job does everything else itself: it stitches the tiles into a map, puts every EDS spectrum on the map as a point of interest linked to its spectrum, scores the map when `find_regions` is set, and images each region when `revisit` is set. To stop early, call `Mapper_abort` (always allowed; the beam is blanked). To continue an aborted map later, call `run_job` with `{"device": "Mapper", "resume": <its reservation_id>}` and no plan: it goes on from the next tile.

## 3. Read what came back

When `done`, the reply's `summary` has:
- `map`: the stitched map's key. `stitching`: its quality (`median_residual_px`, `fallback_tiles`, fitted `drift`).
- `regions` (with `find_regions`): each with `x_m`, `y_m`, `score` and `reason`. `revisits` (with `revisit`): the high-magnification image of each region, already linked to its point of interest, and how many were `skipped` (too near the edge of the area). Report only the image keys it lists.
- `stitch_error`, if stitching failed: the tiles are safe in `result_keys.run`; say so and give the `assemble_map` call it names. Do not run the map again.

Before saying what the sample contains, look at the data: `view_data` with `{"key": <a spectrum key from summary.spectra>, "format": "text"}` lists its peaks with candidate elements. If no peaks were detected at this exposure, say so; do not conclude an element is absent. To look at the map, zoom in: `map_info` gives the map's stage box; `get_region` with a large box returns a coarse preview, a small box around a region gives full resolution. Never walk the tiles one by one. `list_pois` lists the points of interest; `list_data` finds data when you have lost a key.

## 4. More regions later

To score an existing map again (another scorer, more regions), call `score_map` with `{"map": ..., "top_k": N}` and wait with `wait_for_job`. To image regions of an existing map closer without mapping again, `run_job` a revisit-only plan: `{"revisit": {"fov_m": 3e-8, "map": <map key>, "top_k": N}}` (its best N scored regions) or `{"revisit": {"fov_m": 3e-8, "points": [[x_m, y_m], ...]}}` (given stage points). Budgets and stage limits apply.

## Report

Say what was mapped (area, tiles, EDS or not), where the map is (its key), and list interesting regions with their stage coordinates and reasons. Say if anything was refused, denied or aborted.
