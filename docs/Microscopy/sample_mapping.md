# Sample mapping

A sample map is a grid of overlapping images stitched into one zoomable map,
with regions worth a closer look marked on it. Three pieces make one:

| Piece | Where | What it does |
|---|---|---|
| Grid scan | `ElectronMicroscope.acquire_map_grid` (instrument MCP bridge) | Measures which way the stage moves the image, steps the stage, saves every tile |
| Stitching and analysis | pyTEMlib analysis MCP server (`python -m pyTEMlib.analysis_server`) | Stitches, scores regions, keeps points of interest and notes |
| Workflow | `asyncroscopy.agent_workflows.sample_map` (a LangGraph graph) | Joins the two, pauses for a person, reports |

asyncroscopy has no stitcher and no mapping server of its own.

## Start the servers

```bash
uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml
```

```bash
uv run python -m pyTEMlib.analysis_server --tiled http://localhost:9091 --api-key secret --port 8003
```

```bash
uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml
```

The pyTEMlib server comes from the fork pinned in `pyproject.toml` until it
is merged into pycroscopy/pyTEMlib. `configs/gemma-llm.yaml` connects the
agents to the instrument bridge (port 8000) and to pyTEMlib (port 8003). The
agent server (port 8002) offers `list_workflows`, `start_workflow`,
`resume_workflow` and `workflow_status`.

## Run the workflow

```python
start_workflow("sample_map", '{"request": {"name": "grid1", "rows": 3, "cols": 3, "overlap": 0.25, "fov_m": 1e-8, "calibration_fov_m": 2e-8}}')
```

1. **Plan.** The request is checked, and the spec and time estimate are
   worked out. A bad request (for example overlap 0.9) fails here.
2. **Approve.** The run pauses with the plan, because the scan moves the
   stage. Answer with `resume_workflow(thread_id, '{"approved": true}')`, or
   `false`, or `{"changes": {...}}` to plan again.
3. **Acquire.** One `acquire_map_grid` call. The stage loop runs on the
   instrument, not in the model.
4. **Collect.** The tiles become a pyTEMlib map run (`collect_map_run`, or
   a run file written directly when the workflow is given `run_dir`).
5. **Assemble.** `assemble_map` stitches, and the workflow waits for the job.
6. **Score.** `score_map` ranks regions and adds the best as candidates.
7. **Review.** The run pauses with a varied shortlist: the best candidate,
   then each next one as far as possible from those already picked. Answer
   with `{"actor": "human:<name>", "decisions": [{"id": ..., "status":
   "accepted"|"rejected", "note": ...}], "more": false}`. Each decision is
   saved as a note on the map (`annotate`), with its author and history.
8. **Report.** The stitch quality, calibration, candidates, decisions and
   any errors.

A failure at any step ends in the report, with the instrument's or the
server's own message.

## Which way the stage moves the image

Before the grid, `acquire_map_grid` makes one small test move per stage axis
(20% of the field of view) and measures how far and which way the image moved.
That 2 x 2 matrix goes in the manifest as `calibration.px_per_m`, and the grid
is planned from it. Tile `(r, c + 1)` is then always to the right of tile
`(r, c)` in the image, whatever the scan rotation or axis convention. The
digital twin, for example, draws stage x down the image rows.

The scan stops before the grid, with the numbers, when:

- a test move shifts the image by more than 15% more or less than the pixel
  size predicts, or
- the two stage axes do not look roughly perpendicular in the image.

At atomic resolution a crystal lattice can fool the correlation by one lattice
period; then do the test moves at a lower magnification with
`calibration_fov_m` (this assumes the scan rotation is the same at both). To
skip the test moves, pass `stage_to_image_px_per_m` from an earlier manifest.
`calibrate: false` keeps the old assumption (stage x moves the view right),
adjusted with `flip_x` / `flip_y`. Each tile records `stage_xy_m` (read back
after the move) and `commanded_xy_m`.

pyTEMlib assumes image columns run along +x and rows along +y.
`asyncroscopy.mapping.run_file` converts every tile centre into that frame
using only the direction of the measured matrix; the scale comes from the
pixel size, which is better known than the length of a test move.

## Measured

`tests/test_agent_workflows.py` and `tests/test_mapping.py`:

| Case | Result |
|---|---|
| 3 x 3 synthetic grid, stage axes swapped, with the conversion | 20 / 20 matches used, median residual 0.014 px |
| The same grid, raw stage axes | 7 / 14 matches used, one tile placed from the stage alone |
| 3 x 3 twin grid (10 nm tiles, 256 px), pyTEMlib fits its own `scan_linear` | tiles up to 9 px from the true stage |
| The same, identity `scan_linear` | every tile within 0.9 px (39 pm) over 9 runs |

On 9 atomic-resolution tiles pyTEMlib fitted a 0.2-0.9 degree scan turn that
the twin does not have, from matches one lattice period off. The workflow
therefore passes the identity `scan_linear` for calibrated grids under 25
tiles (`assemble_arguments`); a request can pass `"scan_linear": "measure"`
or its own matrix, and `stage_sigma_m` for how well the stage repeats. Tiles of
128 px or less leave too few overlap pixels for reliable matching.
