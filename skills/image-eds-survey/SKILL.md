---
name: image-eds-survey
description: Survey a region - acquire a HAADF image, then an EDS spectrum, then summarise both from Tiled.
version: 1.0.0
tags: [workflow, survey, imaging, spectroscopy]
triggers: ["survey", "image then spectrum", "image and eds", "characterise the region"]
tools: ["*_acquire_scanned_image", "*_acquire_spectrum", "get_data_from_key"]
---

# Image then EDS survey

This is the manual version of the deterministic LangGraph workflow
`image_eds_survey` (`asyncroscopy.agent.graphs.workflows.image_eds_survey`).
Prefer the workflow when the user wants a reproducible run; follow these steps
when you need to adapt it (different detector, extra checks, several regions).

1. `<DeviceClass>_acquire_scanned_image` with `{"detector_list": ["haadf"]}`.
   Keep the returned key as `image_key`.
2. `<DeviceClass>_acquire_spectrum` with `{"detector_name": "eds"}`.
   Keep the returned key as `spectrum_key`.
3. `get_data_from_key` for `image_key` (`max_values` 16) and note the shape
   of the `image/HAADF` dataset.
4. `get_data_from_key` for `spectrum_key` (`max_values` 32). If
   `attrs.elements` is present, pair it with `preview` to form the composition.
5. Summarise in two or three sentences: both keys, the image shape, and the
   composition (or the spectrum length when no element labels exist). Include
   any tool error verbatim.

The workflow runs the same steps as fixed graph nodes and always ends in a
summary, even when a step fails; the LLM device exposes it through
`RunWorkflow('{"name": "image_eds_survey"}')`.
