---
name: multiscale_region
description: Zoom into one region of an image tier by tier, down to atomic columns, placing each finer image on the coarser one by matching (never by the stage), as one Mapper job.
version: 2.0.0
min_asyncroscopy: 0.1.0
keywords: [zoom in, magnification steps, tiers, atomic resolution, atomic columns, multiscale, finer image, closer and closer]
requires: [run_job, wait_for_job]
acts: true
tools: [run_job, wait_for_job, list_placements, view_data]
---

# Zoom into a region, tier by tier

The Mapper does the whole zoom as one job: at each tier it finds the most distinct feature on the image before,
aims the beam there by image shift (the stage does not move), takes the finer image, and places it on the image
before by matching. Do not take, aim or place the images yourself.

1. `run_job` with `{"device": "Mapper", "plan": {"zoom": {"parent": <the image you were given>, "fovs_m": <fovs>}}}`.

   | the image you were given | fovs_m |
   | :-- | :-- |
   | 600 nm (6e-7 m) | [7.5e-8, 9.4e-9] |
   | 2.4 um (2.4e-6 m) | [1.5e-7, 1.9e-8] |
   | any other, F metres | [F / 8, F / 64], leaving out any under 5e-9 |

   The parent is the image key exactly as given (`*.h5`). The last fov shows atomic columns: do not go finer.
2. `wait_for_job` with the job id until it is done.
3. Answer from the job's summary `zoom` list, one line per tier, coarsest first: the tier's `fov_m`, its `image`
   key, and its placement: `"placed"` with `centre_xy_m` and `score`, or `"unplaced"` with its `reason` word for
   word, or `stopped` with why. Say "placed" only for a tier whose placement status is "placed". An unplaced
   atomic-scale image inside a crystal is expected (it matches its parent one lattice spacing away as well as at the
   right place): say that, it is not a failure of the run.

Dose: the job refuses a plan over the request's dose budget before anything is exposed; pass it as
`"dose_budget_e_per_a2"` in the plan when the request gives one.
