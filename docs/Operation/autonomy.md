# Autonomous routines and sessions

`asyncroscopy.autonomy` turns the team's tuning notebooks into routines that
are safe to run unattended. It covers aberration hill-climbing (notebook 12)
and stage-z focus (notebook 14). An agent can then run a whole session from
one instruction.

## Routines

| Routine | What it does | Safety net |
|---|---|---|
| `focus` | Trust-region search over defocus or stage z. | See below. |
| `tune_aberration` | Image-based tuning of one aberration through the corrector. Uses the notebook-12 quadratic response surface. | See below. |
| `measure_and_correct` | Corrector workflow: measure a tableau, correct what is over threshold, measure again. | A correction that makes an aberration larger is undone, and that aberration is left alone. A correction larger than `max_correction_m` is refused. |
| `align_zone_axis` | Tilts onto the zone axis using the Laue offset of camera diffraction patterns: the vector from the direct beam to the centroid of the diffracted intensity. It probes each tilt axis by ± one step, takes Newton steps on the part of the offset those axes control, updates the response after each step, and keeps re-centring the region with image shift (the alpha-tilt notebook's tracking). | Tilt limit; verified or the start tilt is restored; a crystal that doesn't respond to tilt is reported instead of chased. |
| `find_regions` | Ranks the tiles of a survey image by structure (contrast × entropy). Spreads the picks out. | Read-only. |
| `run_session` | Health → region → aberrations → focus → zone axis (optional) → final image → atlas record. | Each stage reports its status. A required device that stays down stops the session before anything moves. |

`focus` and `tune_aberration` share the same safety net:
- **Hard limit:** the setting never moves more than `max_offset` from where it
  started.
- **Verified gain:** a gain is kept only if a fresh baseline and repeated
  measurements at the best point confirm it.
- **Known end state:** on error or abort, the start setting is restored.

All routines share these behaviours:
- They use only the standard device commands. A vendor backend that passes
  `python -m asyncroscopy.conformance` can run them.
- Every instrument operation goes through the supervision queue when
  supervision is on, so it shows up as a child action that can be paused,
  stepped, aborted and limited.
- If a device fails as unreachable, link-down or timeout, its automatic
  recovery steps run (reconnect, Init), the operation is retried once, and the
  recovery is recorded in the result.

## Agent tools

| Tool | Notes |
|---|---|
| `autonomy_focus(mode, step_m, max_offset_m, score)` | |
| `autonomy_correct_aberrations(thresholds_m, max_correction_m)` | |
| `autonomy_align_zone_axis(axes, step_deg, max_offset_deg)` | |
| `autonomy_run_session(plan)` | `plan` keys: `name`, `find_region`, `focus_mode`, `focus_step_m`, `focus_max_offset_m`, `correct`, `thresholds_m`, `polish`, `zone_axis`, `zone_axis_axes`, `score`, `save_site`. |

They are also recipe steps: `autonomy.focus`, `autonomy.correct_aberrations`,
`autonomy.align_zone_axis` and `autonomy.session`. A session can therefore be saved, rehearsed on the
twin, and then run on the microscope.

## Which metric to optimise

Measured on the tilt twin across ±90 nm defocus (64 px, shot noise):

| Metric | At true focus |
|---|---|
| `edge_sharpness` | Clean peak (0.69 vs 0.24 at 60 nm). This is the default. |
| `contrast` | Clean peak. |
| `high_frequency_power` | False minimum about 30 nm out. Shot noise keeps its share up in blurred images. |
| `entropy` | *Lowest*, not highest. |

## Results on the twins

- **Tilt twin:**
  - **Focus alone:** starting 45 nm out of focus, `focus` returns to within
    12 nm of true focus.
  - **Whole session:** focus improves, aberrations are corrected, and edge
    sharpness rises by more than 20 %.
- **Tilt twin, zone axis on abTEM multislice diffraction:**
  - **Method:** coarse scan to bracket the zone, bisection on the sign of the
    controllable Laue component, then landing from one side with re-landing
    on a measured miss.
  - **Live benchmark stack:** 5–7/8 seeds within 0.05°, with passes
    0.002–0.05°. The stage gives every tilt about ±0.045° of error, which is
    the floor. See `benchmark.md`.
  - **Why not a plain search:** a search on |Laue offset| stopped 0.11° away.
    The component that the other axis would fix makes the minimum shallow,
    which is why the routine works on the controllable part only.
- **Gold twin, live through MCP** (corrector detuned to C1 25 nm and A1 7 nm,
  with the corrector link dropped before the start):
  - The session recovered the link at the health stage.
  - It corrected C1 to 1.0 nm and A1 to 1.4 nm, and kept a small verified
    focus gain.
  - It finished in 14 s with no failed stages.
