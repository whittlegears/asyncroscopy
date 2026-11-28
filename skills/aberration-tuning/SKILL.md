---
name: aberration-tuning
description: Use when the operator asks to correct, tune or improve aberrations (A1 astigmatism, A2, B2 coma, C1) or wants sharper atomic-resolution images after focus is already good. Tunes one corrector aberration at a time with the calibration procedures server.
requires_tools: [tune_aberration, sweep_defocus, procedure_limits]
---
# Tuning corrector aberrations

`tune_aberration` changes one aberration at a time by small steps from its current
value. It keeps the change that gives the sharpest image, within the configured
limit. If anything fails, it undoes the whole change.

## Before starting

1. Focus first (see the stem-focus skill). Defocus dominates every other aberration.
2. Call `procedure_limits`. Only aberrations listed in `aberration_max_offset_m` can
   be tuned, and `aberration_max_pixel_size_m` is the coarsest pixel it will run at.
3. Pixels must be fine enough to resolve the probe. Pixel size = field of view /
   image size. If the procedure refuses with "pixels are ... coarser than", ask the
   operator before increasing the image size or reducing the field of view.
4. Use the default metric, `edge_sharpness`. `high_frequency_power` also works.
   `contrast` is refused: it is not monotonic in aberration size and can lead the
   search away from the optimum.

## Order

Tune in this order, one call each: `A1`, then `A2`, then `B2`. Use `step_m` of about
a tenth of the configured limit for that aberration. Round aberrations (`C1`, `C3`)
have one coefficient; the others have two (x, y), handled automatically.

## Reading the result

- `improved` and `peak_bracketed` both true: keep it, report the score gain.
- `improved` true but not bracketed: run the same aberration once more.
- `improved` false: nothing helped and nothing was changed. Move on.
- The best value can sit slightly off zero, typically tenths of a nanometre, when
  higher-order aberrations of the same symmetry are present (S3 balances A1, for
  example). Report `applied_correction_m` as found; do not force it to zero.

## Rules

- One aberration per call; at most two passes per aberration before reporting.
- Never send `correct_aberration` directly. The procedure owns the limit and the undo.
