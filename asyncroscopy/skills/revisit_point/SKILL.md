---
name: revisit_point
description: Go back to a point already known by its stage position (a point of interest on a map, or coordinates the user gives) and image or measure it there.
version: 1.0.0
min_asyncroscopy: 0.1.0
keywords: [go back, return to, revisit, back to it, point of interest, go to the point, image it again]
requires: ["*_move_stage", "*_acquire_scanned_image"]
acts: true
tools: ["*_get_stage", "*_move_stage", "*_get_sample", "*_set_fov", "*_acquire_scanned_image", "*_acquire_spectrum", list_pois, view_data]
---

# Go back to a known point

A point given in stage coordinates needs no map pixels: do not call `to_pixel` or read the map. Move there and take the
data.

1. The position in metres. When the point is on a map, call `list_pois` with the map, take the point nearest the
   given position, and copy its `x_m` and `y_m` exactly: no arithmetic. Only when no map holds it, convert:
   metres = nm x 1e-9, so -23.3 nm is -2.33e-8 m (not -2.33e-7) and 155 nm is 1.55e-7 m. Check before moving: the
   metres times 1e9 must give back the nm you were given.
2. `<Microscope>_get_stage` for the current z, alpha and beta, then `<Microscope>_move_stage` with
   `{"position": [x_m, y_m, z, alpha, beta]}`, keeping those three as they were.
3. Take what was asked there: `<Microscope>_acquire_scanned_image` with `{"detector_list": ["haadf"]}` for a HAADF image
   (keep the current field of view unless the request gives one; `<Microscope>_set_fov` first if it does), or
   `<Microscope>_acquire_spectrum` for a spectrum.
4. Answer with the stage position you sent to `move_stage` (the numbers in that call, not the ones you meant) and the
   new data's key. If the sample was reloaded since the map
   (`<Stage>_get_sample` gives a different `sample_load_id` from the map's), say that the point may no longer be there.
