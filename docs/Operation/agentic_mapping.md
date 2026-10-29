# Agentic sample mapping

An agent maps a sample from one sentence, then stitches it into a zoomable map, marks where EDS spectra were taken, and proposes interesting regions. The design is in [docs/designs/agentic_sample_mapping.md](../designs/agentic_sample_mapping.md); this page is how to use it.

## The pieces

```text
 request ─► agent (LangGraph: mapping or general)
              │  MCP (agent endpoint :8000)            MCP (pyTEMlib :8003)
              ▼                                         ▼
           Mapper ── moves, acquires ──► microscope   assemble_map / score_map / add_poi
              │                                         ▲
              └── run file (tiles + spectra + trace) ─► Tiled ◄── map (pyramid + points)
```

| Piece | Where | What it does |
| :-- | :-- | :-- |
| `Mapper` | `asyncroscopy/instruments/electron_microscope/mapper.py` | One background job per map: serpentine tile grid inside a stage box, optional EDS at each tile, dose budget, abort with beam blank, resume after a restart. |
| Job envelope | `asyncroscopy/contracts/` | `reserve` → `start` → `job_status` → `results` → `release`, plus `abort`. Shared with pyTEMlib's jobs. `reserve` refuses steps outside the instrument's limits, and `suspenders` pause a job while a condition such as beam current is bad (see `asyncroscopy/contracts/README.md`). |
| Approval mode | `asyncroscopy/safety/` | `auto` or `approve`; see [approval_mode.md](approval_mode.md). |
| Map tools | pyTEMlib `mcp_server/map_tools.py` | Stitch by stage position refined by image matching; pyramid in Tiled; view regions; score with plug-in scorers; points of interest. |
| Graphs | `asyncroscopy/agents/graphs/` | `mapping`: the model only writes the plan. `general`: the model makes every call, using the `mapping` skill. |
| Skill | `asyncroscopy/skills/mapping/SKILL.md` | The procedure agents load before mapping. |

## Map plans

A plan is JSON checked against `asyncroscopy/contracts/map_plan.schema.json`. Lengths are metres.

```json
{"box_m": [-3e-7, 3e-7, -3e-7, 3e-7], "fov_m": 2e-7, "overlap": 0.15, "imsize": 512, "dwell_s": 1e-6,
 "detectors": ["haadf"], "per_tile": ["eds_point"], "eds_exposure_s": 1.0, "dose_budget_e_per_a2": 100}
```

`per_tile` is `[]` (images only), `["eds_point"]` (one spectrum at each tile centre) or `["eds_map"]` (a coarse spectrum image of every tile: much slower, only when asked for by name). `reserve` refuses a plan whose overlapping tiles would exceed `dose_budget_e_per_a2` and returns a time and dose estimate.

## From Python, without an agent

```python
import json, tango
mapper = tango.DeviceProxy("asyncroscopy/mapper/default")
reserved = json.loads(mapper.reserve(json.dumps({"plan": {"box_m": [-3e-7, 3e-7, -3e-7, 3e-7], "fov_m": 2e-7}})))
job = json.loads(mapper.start(json.dumps({"reservation_id": reserved["reservation_id"]})))
print(json.loads(mapper.job_status(json.dumps({"job_id": job["job_id"]}))))
```

Then stitch with the pyTEMlib server (`assemble_map`), or in Python with `pyTEMlib.image_tools.mosaic`.

## Exploring a map

`score_map` ranks regions with a scorer and adds the best as points of interest with a reason and the scorer's version. The `baseline` scorer looks for contrast and sharp edges. Add your own (a trained model, rules, an LLM judge) by implementing `score(region, metadata) -> (value, reason)` and registering it under the `pytemlib.scorers` entry point; see `pyTEMlib/scoring/__init__.py`.

## Testing a workflow on the twin before the instrument

`configs/digital_twin_mapping.yaml` gives the twin stage error, backlash, drift, scan distortion, focus, noise, EDS spectra and beam damage at literature magnitudes. `export_ground_truth` on the twin (operator endpoint or Tango, never the agent endpoint) returns where every tile really was, so a workflow can be scored: `tests/test_map_end_to_end.py` checks that stitching recovers the true layout to under a pixel while stage positions alone are visibly off.

## Measuring an agent

```bash
uv run python benchmarks/run_tool_json.py --models qwen3:8b       # valid tool calls per model
uv run python benchmarks/run_mapping_trials.py --n 10 --model qwen3:8b   # seeded mapping tasks
```

Results are written to `benchmarks/results/`.
