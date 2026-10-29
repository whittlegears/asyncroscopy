---
name: eds_survey
description: Survey composition with EDS, one spectrum per tile of a map or at chosen points, then read the spectra for elements.
version: 1.1.0
min_asyncroscopy: 0.1.0
envelope_schema: 1.0.0
keywords: [eds, edx, spectrum, spectra, composition, element, elements, chemistry, x-ray]
requires: [Mapper*_reserve, view_data]
acts: true
tools: [run_job, wait_for_job, view_data, list_data, add_poi]
---

# EDS survey

1. Map with a spectrum at each tile: `run_job` with `{"device": "Mapper", "plan": {...}}` and `"per_tile": ["eds_point"]` in the plan (see the mapping skill for the area). Use `"eds_exposure_s"` of 1 s or more when light elements or minor phases matter: at 0.1-0.2 s weak peaks are lost in noise. A full spectrum image (`eds_map`) only when asked by name.
2. `wait_for_job` with the `job_id` until `state` is `done`; its reply's `summary.spectra` lists each spectrum's key and stage position.
3. Read a few: `view_data` with `{"key": <spectrum key>, "format": "text"}`. Note the peaks and candidate elements; ignore lines marked escape or sum peak.
4. If the spectra say no peaks were detected at this exposure, say the exposure was too short, and offer a longer one at the most interesting tiles; do not report the sample as empty.
5. Mark spectra on the stitched map with `add_poi` (`kind: "eds_point"`, `links: [key]`) when there is a map.

Report the elements seen, where, and how confident (strong peaks vs. near the noise).
