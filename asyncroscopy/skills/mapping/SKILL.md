---
name: mapping
description: Map a sample or an area as a grid of images (optionally with an EDS spectrum at each tile), stitch it into a zoomable map, find interesting regions and go back to them.
version: 1.1.0
min_asyncroscopy: 0.1.0
envelope_schema: 1.0.0
requires: [Mapper*_reserve]
tools: [run_job, wait_for_job, Mapper_abort, approval_status, assemble_map, score_map, map_info, get_region, list_pois, add_poi, view_data, list_data]
---

# Mapping a sample

A map is ONE background job, not hundreds of moves. Never loop over `move_stage` and `acquire_scanned_image` yourself.

A whole request is: run the map job, wait for it, then stitch. "The map" means the stitched map key (`result_keys.map`, ending in `_map`), not the run file.

## 1. Plan

Turn the request into a plan. Lengths are metres (1 um = 1e-6, 1 nm = 1e-9).

| Request says | Plan |
| :-- | :-- |
| "the sample", "whole sample" | `box_m` = sample bounds from `<Microscope>_get_parameters` (`world_bounds_angstrom` x 1e-10) |
| "a 500 nm square around here / the current position" | `"size_m": [5e-7, 5e-7]` (the Mapper centres it on the stage; do not compute a box) |
| "a 500 nm square around a point (x, y)" | `"size_m": [5e-7, 5e-7], "centre_m": [x, y]` |
| nothing about EDS, spectra or composition | no `per_tile` at all: images only |
| "with EDS", "and a spectrum at each image" | `"per_tile": ["eds_point"]` (one spectrum at each tile centre) |
| "spectrum image", "EDS map" of every tile | `"per_tile": ["eds_map"]`. Only when asked by name: it multiplies time and dose. |
| nothing about size | `"fov_m": 2e-7`, `"overlap": 0.15` |

If the request is unclear about the area, ask the user before starting.

## 2. Run the map

1. `run_job` with `{"device": "Mapper", "plan": {...}}`. It checks the plan and starts the job; the reply has `job_id` and `estimate` (tiles, duration_s, max_dose_e_per_a2). An `error` means the plan was refused: fix what the message says and call `run_job` again. If the estimate is over an hour, abort and ask the user.
   - If it says **APPROVAL REQUIRED** with a `request_id`, the instrument is in approve mode. Call `approval_status` every few seconds. When it is `approved`, call `run_job` again with exactly the same arguments. If `denied` or `expired`, stop and tell the user.
   - An endpoint with several Mappers names them `Mapper_<key>`: use that name as `device`.
2. `wait_for_job` with `{"job_id": ...}` until `state` is `done`, `failed` or `aborted` (each call waits up to 25 s). When done, the reply has `result_keys.run` (the run in Tiled) and `summary.spectra` (where each EDS spectrum was taken). To stop early, call `Mapper_abort` (always allowed; the beam is blanked). **You are not done yet: go on to section 3 and stitch.**

## 3. Stitch and look

1. `assemble_map` with `{"run": result_keys.run}` returns a `job_id`. Wait with `wait_for_job` and that `job_id`; when done, its reply has `result_keys.map`.
2. `map_info` gives the map's stage box. `get_region` with a stage box returns a preview image; never ask for raw pixels.
3. For each EDS spectrum, `add_poi` at its `stage_xy_m` with `kind: "eds_point"` and `links: [its key]`.
4. Before reporting what the sample contains, look at the data: `view_data` with `{"key": <a spectrum's key from summary.spectra>, "format": "text"}` lists its peaks with candidate elements; `{"key": result_keys.run, "dataset": "tiles", "format": "text"}` describes the tiles. If a spectrum says no peaks were detected at this exposure, say so; do not conclude an element is absent. `list_data` finds the newest data when you have lost a key.

## 4. Find interesting regions and go back

Do this whenever the request asks for interesting, notable or unusual regions; listing the existing points of interest is not enough.

1. `score_map` with `{"map": ..., "top_k": 3}`; wait with `wait_for_job`; its `summary.added_pois` lists regions with `x_m`, `y_m`, `score` and `reason`.
2. To image one closer, run a small map around it (`"size_m"` of a few fields of view, `"centre_m": [x_m, y_m]`, smaller `fov_m`), or move there and acquire once.
3. Mark what you took with `add_poi` and `links`, so the map shows it.

## Report

Say what was mapped (area, tiles, EDS or not), where the map is (its key), and list interesting regions with their stage coordinates and reasons. Say if anything was refused, denied or aborted.
