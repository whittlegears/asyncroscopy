---
name: return-to-site
description: Save a spot on the specimen and come back to it later, verified by image registration rather than trusted stage coordinates, including after the specimen was unloaded and remounted. Use when asked to revisit, re-image, compare over time, or go back to a feature, particle or region seen earlier.
---

# Return to a site, verified

Stage coordinates are a starting guess, not an address. Backlash, drift and,
above all, a remount move a feature by more than a field of view. You are back
at a site only when `register_images` says `match`.

If the agent MCP server is available (`agent_mcp_server`), its `save_site`,
`calibrate_stage_mapping`, `return_to_site` and `list_sites` tools run exactly
this procedure as a deterministic LangGraph workflow
(`asyncroscopy/mcp/workflows/return_to_site.py`) and write the same notes
file. Prefer them. Follow the steps below by hand when only the instrument MCP
server is connected.

Tool names are prefixed with the instrument's Tango class, for example
`DigitalTwin_move_stage`. Below, `<I>` stands for that prefix. Everything a
site needs is in your own notes, a JSON file you keep (default
`sites.json` in the working directory).

## 1. Save a site

1. Acquire the image you want to come back to: `<I>_acquire_scanned_image`.
   It returns a DATA/Tiled key.
2. Read the stage with `<I>_get_stage`, the field of view with `<I>_get_fov`,
   and the rest of the settings with `<I>_get_parameters`.
3. Append to `sites.json`:
   `{"name", "key", "stage": [x, y, z, alpha, beta], "fov_m", "imsize", "settings", "landmark": false, "saved_at", "mount": "original"}`.
   Mark 2–3 distinctive, well-separated sites as `"landmark": true`. They are
   what you use to find everything again after a remount.

## 2. Calibrate the stage-to-image mapping (once per session and magnification)

Stage axes are usually rotated and scaled relative to the scan, so the image
offset does **not** equal the stage move.

1. At any textured spot, note the stage position P.
2. For the x axis, with s about 15% of the field of view:
   - Move to P and take image A0, then move to P + s in x and take image A1.
     Call `register_images(A0, A1, image_fov_m=fov)` and keep its `offset_m`
     as o+.
   - Do the same with −s to get o−.
   - Column 1 of `stage_to_image` is (o+ − o−) / (2s). Stepping both ways from
     a freshly taken start image cancels most of the stage's move noise. A
     single one-sided step was measured 20–25% off.
3. Repeat for the y axis to get column 2, then move back to P.
4. If any registration is not `match`, choose a spot with more texture and
   repeat.
5. Record `stage_to_image = [[c1x, c2x], [c1y, c2y]]` in `sites.json`.

## 3. Go back

1. Move to the saved `stage`, or to the remount-corrected position (section 4).
2. **Survey.** Set the field of view to 4× the site's and acquire (key S).
   Call:

   ```
   register_images(site.key, S, reference_fov_m=site.fov_m, image_fov_m=4*site.fov_m,
                   stage_to_image=..., max_rotation_deg=0 or 10 after a remount)
   ```

   - `no_match`: the site isn't in the survey. Survey a 3×3 grid around the
     guess before giving up.
   - `ambiguous`: stop and report the reasons (repetitive structure, or parts
     of the image disagreeing). Do not pick one.
3. **Centre.** Move the stage by `stage_correction_m` from the survey result.
4. **Verify at native settings.** Restore the site's field of view, acquire
   (key N), and call:

   ```
   register_images(site.key, N, image_fov_m=site.fov_m, stage_to_image=...)
   ```

   - `match` and |`offset_m`| ≤ 2 pixels: you are there. Report `offset_m`,
     `sigma_m` and `ncc`.
   - `match` but further off: apply `stage_correction_m` and verify again, at
     most 3 times.
   - Anything else: say plainly that the site was **not** confirmed, and why.
5. Log the visit in `sites.json` under the site: time, keys S and N, status,
   residual, uncertainty and the final stage position. Log failures too.

## 4. After the specimen was remounted

1. Set `"mount"` to a new id. All saved coordinates are now rough.
2. For each landmark, run section 3 with a wider survey (6× field of view,
   3×3 grid) and `max_rotation_deg=10`.
3. With two or more landmarks at `match`, fit the rotation and translation
   that maps old landmark stage positions to new ones (least squares, 2-D
   rigid). Predict every other site's position with it.
4. With fewer than two, the rotation is unknown. Use the translation from the
   one landmark, keep `max_rotation_deg=10`, and say so.

## Rules

- Never add a pixel offset to a stage position yourself. Use
  `stage_correction_m` from a calibrated `stage_to_image`.
- Never call a site found without a `match` at native settings. `ambiguous`
  and `no_match` are answers. Report them, don't retry them away.
- Approach the final position from the same direction each time if the
  stage has backlash.
- Keep the reference key. The original image is the record a teammate uses to
  check the visit later.
