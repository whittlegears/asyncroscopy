---
name: explore_regions
description: Find the most interesting regions of a map and image them at higher magnification, with each image linked to its region.
version: 1.3.0
min_asyncroscopy: 0.1.0
envelope_schema: 1.0.0
keywords: [interesting, regions, features, explore, zoom, closer look, higher magnification, revisit, go back]
requires: [score_map, Mapper*_reserve]
acts: true
tools: [run_job, wait_for_job, score_map, list_pois, add_poi, map_info, get_region]
---

# Explore a map

Do as many regions as the user asks for (default three).

## A new map: one job does it all

`run_job` with `{"device": "Mapper", "plan": {... the area ..., "find_regions": N, "revisit": {"fov_m": <a few times smaller than the map's fov_m>}}}`, then `wait_for_job` until `done`. The job stitches the map, scores it, images each region and links each image to its point of interest. Its `summary.regions` lists each region (`x_m`, `y_m`, `score`, `reason`) and `summary.revisits` each image.

## An existing map (a key ending in `_map`)

1. `score_map` with `{"map": <key>, "top_k": N}`, then `wait_for_job` with its `job_id`; the reply's `summary.added_pois` lists the regions with `x_m`, `y_m`, `score`, `reason`.
2. Image them: `run_job` with `{"device": "Mapper", "plan": {"revisit": {"fov_m": <a few times smaller than the map's tile fov_m>, "map": <key>, "top_k": N}}}`, then `wait_for_job`. The job images the map's N best scored regions and links each image to its point of interest. Its `summary.revisits` lists each image. Do not image a region as a one-tile map, and do not move the stage yourself.
3. To look at a region before or after, `get_region` with a box of about two fields of view around (`x_m`, `y_m`).

Report each region: coordinates, score, reason and the key of its image.
