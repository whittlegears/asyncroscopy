---
name: multiscale_region
description: Zoom into one region of a map tier by tier, down to atomic columns, placing each finer image on the coarser one by matching it (never by the stage), and stop at the dose cap.
version: 1.0.0
min_asyncroscopy: 0.1.0
keywords: [zoom in, magnification steps, tiers, atomic resolution, atomic columns, multiscale, finer image, closer and closer]
requires: [place_image, "*_acquire_scanned_image", "*_set_image_shift"]
acts: true
tools: [place_image, list_placements, view_data, to_stage, "*_get_stage", "*_move_stage", "*_set_fov", "*_get_fov", "*_set_image_shift", "*_get_image_shift", "*_acquire_scanned_image", "*_get_sample"]
---

# Zoom into a region, tier by tier

Each step is about 16x finer (fewer, about 8x, when the next image is atomic). The stage cannot land a 150 nm or 20 nm
field where it was asked (its error is hundreds of nanometres), so a finer image is aimed by image shift and then
placed on the coarser image it was taken inside by `place_image`, which matches the two. Never report where a finer
image is from the stage alone.

1. Start from a coarse image or map key that shows the region (from the request, a map's point of interest, or a
   fresh `<Microscope>_acquire_scanned_image`). Note its field of view (`view_data` with `"format": "text"`).
2. Check the sample has not been reloaded (`<Stage>_get_sample`; a different `sample_load_id` from the image's means
   positions from it are stale: start again from a new image).
3. Pick the point to zoom into, in pixels of that image, and turn it into a position with `to_stage`.
4. Aim there: if it lies within the image shift's reach of the current beam position (`<Microscope>_get_image_shift`;
   about 1 um on most instruments), set `<Microscope>_set_image_shift` to it; otherwise `<Microscope>_move_stage`
   to it first (and use the stage's error as the search window in step 6).
5. `<Microscope>_set_fov` to the parent's field divided by 16 (by 8 when that would give pixels under 0.2 nm), then
   `<Microscope>_acquire_scanned_image`.
6. `place_image` with `{"image": <new key>, "parent": <coarser key>, "centre_xy_m": <the point you aimed at>,
   "window_m": ...}`: about 20 parent pixels when aimed by image shift, 400 nm when the stage moved.
   - `"status": "placed"`: its `centre_xy_m` is where the image really is. Use it, not the point you aimed at, for
     the next step.
   - `"status": "unplaced"`: say why (its `reason`). Do not go finer from an unplaced image: take another image a
     little away from that point, or stop and report.
7. Repeat from step 3 with the new image as the parent, until the request's resolution is reached.

Dose: each finer image puts far more dose on its small area (about 7,800 e/A^2 per frame at 20 pm pixels on a
typical setting). Stop at the request's or the instrument's dose cap; never re-image the coarser tiers after the finer
ones (the finer ones damage and contaminate them).

Report every image's key, its tier, its placement (`centre_xy_m`, `score`) or why it was unplaced, and the chain from
the coarsest image to the finest. `list_placements` with `{"parent": <key>}` shows what has been placed on an image.
