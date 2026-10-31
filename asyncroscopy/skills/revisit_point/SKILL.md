---
name: revisit_point
description: Go back to a point already known by its stage position (a point of interest on a map, or coordinates the user gives) and image or measure it there.
version: 2.0.0
min_asyncroscopy: 0.1.0
keywords: [go back, return to, revisit, back to it, point of interest, go to the point, image it again]
requires: [run_job]
acts: true
tools: [run_job, wait_for_job, "*_get_fov", "*_get_stage", "*_move_stage", "*_get_sample", "*_unblank_beam", "*_blank_beam", "*_acquire_spectrum", list_pois, view_data]
---

# Go back to a known point

A point given in stage coordinates needs no map pixels: do not call `to_pixel` or `to_stage`, and do not read the map.

1. The position in metres. When the point is on a map, call `list_pois` with the map, take the point nearest the
   given position, and copy its `x_m` and `y_m` exactly: no arithmetic. Only when no map holds it, convert:
   metres = nm x 1e-9, so -23.3 nm is -2.33e-8 m (not -2.33e-7) and 155 nm is 1.55e-7 m. Check: the metres times 1e9
   must give back the nm you were given.
2. For an image, one job does the rest (it moves the stage there, unblanks the beam only while imaging, takes the
   image and leaves the beam blanked): `run_job` with `{"device": "Mapper", "plan": {"revisit": {"fov_m": <the
   request's field of view, or <Microscope>_get_fov now>, "points": [[x_m, y_m]]}}}`, then `wait_for_job` until it is
   done. The image's key and where it was taken (`stage_xy_m`) are in its summary's `revisits`.
3. For a spectrum instead: `<Microscope>_get_stage` for z, alpha and beta, `<Microscope>_move_stage` with
   `{"position": [x_m, y_m, z, alpha, beta]}`, `<Microscope>_unblank_beam`, `<Microscope>_acquire_spectrum`, then
   `<Microscope>_blank_beam`.
4. Answer with the stage position the point was taken at (from the job's summary, or the numbers you sent to
   `move_stage`) and the new data's key. If the sample was reloaded since the map (`<Stage>_get_sample` gives a
   different `sample_load_id` from the map's), say that the point may no longer be there.
