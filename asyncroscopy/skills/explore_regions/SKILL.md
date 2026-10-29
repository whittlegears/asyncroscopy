---
name: explore_regions
description: After a map, rank interesting regions, go back to the best ones at higher magnification, and mark what was taken.
version: 1.1.0
min_asyncroscopy: 0.1.0
envelope_schema: 1.0.0
keywords: [interesting, regions, features, explore, zoom, closer look, higher magnification, revisit, go back]
requires: [score_map, Mapper*_reserve]
tools: [score_map, wait_for_job, list_pois, add_poi, map_info, get_region, run_job]
---

# Explore a map

You need a stitched map key (ends in `_map`). If there is none, map first (mapping skill).

1. `score_map` with `{"map": <key>, "top_k": 3}`, then `wait_for_job` with its `job_id` until `done`; the reply's `summary.added_pois` lists regions with `x_m`, `y_m`, `score`, `reason`.
2. Look at each before going back: `get_region` with a box of about two fields of view around (`x_m`, `y_m`). `map_info` gives the whole map's stage box; zoom out with a bigger box, in with a smaller one.
3. Go back to the best one or two only (each revisit costs time and dose): `run_job` with `{"device": "Mapper", "plan": {"size_m": [3 * fov, 3 * fov], "centre_m": [x_m, y_m], "fov_m": fov / 4}}` where `fov` is the survey's field of view, then `wait_for_job` until `done`.
4. Mark each revisit: `add_poi` on the survey map at (`x_m`, `y_m`) with `kind: "revisit"`, a short `reason` and `links: [the new run key]`.

Stop after the regions the user asked for (default three). Report each region: coordinates, score, reason and the key of the closer look.
