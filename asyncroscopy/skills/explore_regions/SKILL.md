---
name: explore_regions
description: Find the most interesting regions of a map and image them at higher magnification, with each image linked to its region.
version: 1.2.0
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
2. Look before going back: `get_region` with a box of about two fields of view around (`x_m`, `y_m`).
3. Image each region the user asked for: `run_job` with `{"device": "Mapper", "plan": {"size_m": [fov, fov], "centre_m": [x_m, y_m], "fov_m": fov}}` where `fov` is the high-magnification field of view, then `wait_for_job`.
4. Link each image to its region: `add_poi` on the map at (`x_m`, `y_m`) with `kind: "revisit"`, a short `reason` and `links: [the new run key]`.

Report each region: coordinates, score, reason and the key of its image.
