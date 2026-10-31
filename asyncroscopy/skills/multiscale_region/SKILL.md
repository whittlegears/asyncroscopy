---
name: multiscale_region
description: Zoom into one region of a map tier by tier, down to atomic columns, placing each finer image on the coarser one by matching it (never by the stage), and stop at the dose cap.
version: 1.1.0
min_asyncroscopy: 0.1.0
keywords: [zoom in, magnification steps, tiers, atomic resolution, atomic columns, multiscale, finer image, closer and closer]
requires: [place_image, "*_acquire_scanned_image", "*_aim_image_shift"]
acts: true
tools: [place_image, list_placements, view_data, to_stage, "*_get_stage", "*_move_stage", "*_set_fov", "*_get_fov", "*_aim_image_shift", "*_set_image_shift", "*_get_image_shift", "*_acquire_scanned_image", "*_get_sample"]
---

# Zoom into a region, tier by tier

Each step is a finer image inside the one before. The stage cannot land a 75 nm or 10 nm field where it was asked (its
error is hundreds of nanometres), so a finer image is aimed by image shift and then placed on the coarser image by
`place_image`, which matches the two. Do not move the stage, park the beam or take a new survey: start from the image
you were given.

1. The parent is the image you were given. `view_data` with `{"key": <parent>, "format": "text"}` gives its field of
   view (fov_m) and size in pixels.
2. Pick the point to zoom into, in the parent's pixels (a bright particle; the centre, row and col = size / 2, if you
   cannot tell), and turn it into metres with `to_stage` `{"dataset": <parent>, "row": r, "col": c}`: x_m, y_m.
3. Aim there by image shift: `<Microscope>_aim_image_shift` with `{"position": [x_m, y_m]}` (the stage does not
   move; it works out the shift itself). Refused beyond about 1 um from the stage: then stop and report.
4. `<Microscope>_set_fov` with the next field of view, then `<Microscope>_acquire_scanned_image`
   `{"detector_list": ["haadf"]}`: the new image's key.

   | parent fov_m | next fov_m |
   | :-- | :-- |
   | 6e-7 (600 nm) | 7.5e-8 |
   | 7.5e-8 | 9.4e-9 |
   | other | parent / 16, or parent / 8 when that gives pixels under 2e-10 m (fov / pixels); never under 1e-9 |

5. `place_image` with `{"image": <new key>, "parent": <parent>, "centre_xy_m": [x_m, y_m], "window_m": 2e-8}`
   (the point you aimed at; `window_m` 4e-7 instead if you moved the stage).
   - `"status": "placed"`: its `centre_xy_m` is where the image really is. It is the next parent.
   - `"status": "unplaced"`: say why (its `reason`). Do not go finer from it: aim a little away and take it again, or
     stop and report.
6. Repeat from step 3 with the placed image as the parent and its placement's `centre_xy_m` as x_m, y_m (do not use
   `to_stage` on an image taken with an image shift: images do not record the shift), until the fov is about 1e-8 m
   (atomic columns show at pixels under 1e-10 m). Then `<Microscope>_set_image_shift` `{"shift": [0, 0]}`.

Dose: each finer image puts far more dose on its small area (about 7,800 e/A^2 per frame at 20 pm pixels on a
typical setting). Stop at the request's or the instrument's dose cap; never re-image the coarser tiers after the finer
ones (the finer ones damage and contaminate them).

Report every image's key, its field of view, and its placement (`centre_xy_m`, `score`) or why it was unplaced, from
the coarsest image to the finest. `list_placements` with `{"parent": <key>}` shows what has been placed on an image.
