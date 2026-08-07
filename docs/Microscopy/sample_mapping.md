# Sample mapping (zoomable montages)

`asyncroscopy.mapping` turns a grid of overlapping acquisitions - optionally
plus lower-magnification "zoom out" images - into a "google-maps" style
zoomable map of the sample, viewable by humans and navigable by agents.

## Two MCP servers

Stitching and export are deliberately **not** on the instrument MCP bridge
(`asyncroscopy.mcp.mcp_server`). That server auto-reflects every Tango
command and carries the one real safety boundary in the stack (the
`blocked_classes`/`blocked_functions` YAML) - it should not also carry
opencv-heavy image processing or depend on it staying responsive.

```
Instrument server (Tango-backed)          Mapping server (no Tango dependency)
  move_stage, acquire_scanned_image,        start_map, add_tiles, preview_map,
  set_fov, get_data_from_key,               register_overview, get_map_status,
  acquire_map_grid                          finalize_map, list_maps, discard_map
```

`asyncroscopy.mcp.mapping_mcp_server` (module `MappingServer`) can run on a
different, more powerful machine than the one driving the microscope - it
only needs numpy/opencv/pillow/h5py/tifffile/fastmcp, none of pytango. Start
it with `startup_scripts/run_mapping_mcp.py --yaml configs/mapping_mcp.yaml`
(default `http://127.0.0.1:8001/mcp`) and connect it in the GUI as a second
MCP server alongside the instrument bridge.

## Workflow — agent-driven loop

The agent orchestrates the loop itself, tile by tile or in small batches,
using tools it already has:

1. `start_map(name, overlap, pixel_size_nm)` on the mapping server → `map_id`.
2. Repeat: `move_stage` → `acquire_scanned_image` → resolve the returned DATA
   key to a path via `get_data_from_key` → `add_tiles(map_id, [{row, col,
   image_path, stage_xy_m}, ...])` on the mapping server. `add_tiles`
   accumulates across calls and re-registers the whole grid each time, so
   later batches correct and extend earlier ones - the response's
   `edges_measured`/`edges_total` and `mean_edge_residual_px` can be checked
   after every batch, not just at the end.
3. `preview_map(map_id)` any time for a small PNG of progress so far.
4. Optionally "zoom out": increase FOV, acquire one wider image, and call
   `register_overview(map_id, image_path)`. Unlike tile-to-tile registration
   (pure-translation phase correlation, appropriate because same-magnification
   neighbours cannot differ in scale), overview registration uses
   scale-invariant ORB feature matching plus a RANSAC similarity transform,
   since an overview differs from the fine mosaic in scale and possibly
   rotation. Returns `matched: false` instead of raising if there isn't
   enough shared texture to match confidently.
5. `finalize_map(map_id, name)` stitches everything and writes the bundle.

A convenience command still exists on the instrument for the "just acquire a
simple grid" case:

```python
spec = {"rows": 3, "cols": 3, "overlap": 0.15, "eds": False}
manifest = microscope.acquire_map_grid(json.dumps(spec))
```

It reads the current FOV, steps the stage over an N x M serpentine grid,
acquires per tile with current SCAN settings, optionally takes an EDS point
spectrum at each tile centre, and restores the stage - but it only drives
stage/camera, exactly like the granular tools; it does not stitch. Pass its
`tiles` list to the mapping server's `add_tiles` in one call, then
`finalize_map`. Spec keys: `rows`, `cols`, `overlap` (0-0.5), `detectors`
(default `["haadf"]`), `eds`, `settle_s`, `snake`, `flip_x`/`flip_y`,
`return_to_start`.

## The map bundle

```
maps/<name>/
    map.json          machine-readable manifest (format "sciagent-map/1.1")
    index.html        self-contained pan/zoom viewer - open in any browser
    preview.png       small overview
    tiles/{z}/{x}/{y}.png
    overviews/        registered zoom-out images, one per register_overview call
```

Zoom levels follow the web-map convention: `z = 0` is the whole sample in a
single tile, `z = max_zoom` is native resolution, each level doubles linear
resolution.

`map.json` is designed to be read by agents: it contains the level table, the
tile URL template, `pixel_size_nm`, the stage origin, every source tile's
canvas position, EDS annotations anchored to canvas pixels, and an
`agent_instructions` string that explains the coordinate conventions to a
model that encounters a bundle cold. `overviews` entries carry a 2x3
`matrix` mapping overview pixel coordinates onto the fine canvas's native
pixel coordinates (plus `scale`, `rotation_deg`, `offset_px`, `inliers`) so
an agent can place a lower-magnification image in the same coordinate system
without re-registering it. An agent can survey the sample from the `z=0`
tile and drill down only into regions of interest, which keeps token cost
independent of map size.

The stitcher can also be used offline, without either MCP server:

```python
from asyncroscopy.mapping.grid_stitcher import GridStitcher
from asyncroscopy.mapping.map_export import export_map
from asyncroscopy.mapping.overview import register_overview

result = GridStitcher(overlap=0.2).stitch({(r, c): img, ...})
export_map(result.canvas, "maps/demo", name="demo", pixel_size_nm=0.05)
```

Only :func:`asyncroscopy.mapping.plan_grid` and
:func:`asyncroscopy.mapping.parse_map_grid_result` are re-exported at package
level (no opencv dependency, safe to import from instrument-side code);
everything else - `GridStitcher`, `export_map`, `register_overview`,
`build_sample_map`, `load_acquisition_image` - must be imported from its
submodule.

Tests: `tests/test_mapping.py`.
