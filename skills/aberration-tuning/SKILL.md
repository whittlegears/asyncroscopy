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
   be tuned.
3. Use `high_frequency_power` as the metric. On the gold digital twin, `contrast`
   rose as A1 got worse, so it led the search the wrong way.
4. The pixels must be fine enough to resolve the probe. On the gold twin, nothing
   responded at 128 px over 20 nm (1.6 Å per pixel), and the signal was clear at
   256 px over 10 nm (0.4 Å per pixel). If the current settings are coarser, ask the
   operator before changing field of view or image size.

## Order

Tune in this order, one call each: `A1`, then `A2`, then `B2`. Use `step_m` of about
a tenth of the configured limit for that aberration. Round aberrations (`C1`, `C3`)
have one coefficient; the others have two (x, y), handled automatically.

## Reading the result

- `improved` and `peak_bracketed` both true: keep it, report the score gain.
- `improved` true but not bracketed: run the same aberration once more.
- `improved` false: nothing helped and nothing was changed. Move on.
- The best setting need not be zero. With other aberrations present, a small A1 can
  partly offset them, and the procedure optimises image sharpness, not the number.
  Report `applied_correction_m` as found; do not "correct it to zero".

## Rules

- One aberration per call; at most two passes per aberration before reporting.
- Never send `correct_aberration` directly. The procedure owns the limit and the undo.
