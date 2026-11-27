---
name: revisit_point
description: Go back to a point already known by its stage position (a point of interest on a map, or coordinates the user gives) and image or measure it there.
version: 2.4.0
min_asyncroscopy: 0.1.0
keywords: [go back, return to, revisit, back to it, point of interest, go to the point, image it again]
requires: [run_job]
acts: true
tools: [run_job, wait_for_job, run_workflow, workflow_status, "*_get_fov", "*_get_sample", list_pois, view_data]
---

# Go back to a known point

A point given in stage coordinates needs no map pixels: do not call `to_pixel` or `to_stage`, and do not read the map.

1. The position in metres. When the point is on a map, call `list_pois` with the map, take the point nearest the
   given position, and copy its `x_m` and `y_m` exactly: no arithmetic. Only when no map holds it, convert the
   request's own numbers: metres = nanometres x 1e-9. Check: the metres times 1e9 must give back the nm you were given.
2. For an image, one job does the rest (it moves the stage there, unblanks the beam only while imaging, takes the
   image and leaves the beam blanked): `run_job` with `{"device": "Mapper", "plan": {"revisit": {"points": [[x_m, y_m]]}}}`
   (add `"fov_m"` in metres only when the request gives a field of view; without it the job keeps the current one),
   then `wait_for_job` until it is done. The image's key and where it was taken (`stage_xy_m`) are in its summary's `revisits`.
3. For a spectrum instead, the `spectrum_at_point` workflow: `run_workflow` with
   `{"name": "spectrum_at_point", "inputs": {"x_m": x_m, "y_m": y_m}}`, and `workflow_status` while it is `running`.
   It moves there (keeping z and tilts), takes the spectrum, blanks the beam and reads the peaks: the spectrum's key is
   `outputs.spectrum.key` and its elements `outputs.peaks.elements`.
4. Answer with the stage position the point was taken at (from the job's summary, or the numbers you sent to
   `move_stage`) and the new data's key. If the sample was reloaded since the map (`SAMPLE_get_sample` gives a
   different `sample_load_id` from the map's), say that the point may no longer be there.
