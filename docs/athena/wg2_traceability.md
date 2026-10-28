# ATHENA WG2 traceability: what this work provides as evidence

Each row links a WG2 Year-1 item (Project Execution Plan v7a §2.1.4.2, the milestone table in §3.9, and the WG2 quarterly report of 2026-10-26) to the artifact here and to the check that exercises it. **All evidence is from the digital twin unless a row says otherwise.** Nothing here has run on the Spectra 300 yet; that needs the facility owner's readiness decision.

| WG2 item | Artifact | How it is checked | Status |
| :-- | :-- | :-- | :-- |
| Standardised typed action schemas with a common envelope (reserve, release, status, abort, results) | `asyncroscopy/contracts/envelope.schema.json`, `map_plan.schema.json`; `JobDevice` base class; `Mapper`; the same envelope in pyTEMlib's analysis jobs (byte-identical copy) | `tests/test_mapper.py`, `tests/test_contracts.py` (every reply validated; copies compared), pyTEMlib `tests/test_mcp_map_tools.py` | Implemented for STEM mapping (twin). Written so nanoindentation and SPM can reuse it; their schemas stay on WG2's own track. |
| Authentication hooks | Caller name (`X-Asyncroscopy-Caller`) recorded on every call; operator-only commands never offered on the agent endpoint | `tests/test_approval_mode.py::TestEndpointRoles` | Hook only. Identity is self-reported until WG1 tokens exist. |
| Trace logging on by default | Instrument Trace: `trace_event.schema.json`; MCP server, jobs and approvals write it; each map run file carries its own trace | `tests/test_approval_mode.py` (events validated), `tests/test_mapper.py` (one event per tile in the run file) | On by default in `configs/mcp_dt.yaml` and every job device. |
| Action-validation layer exercised on selected workflows | Plan validation (schema, box, dose budget), plan box enforced in the device, approval mode for high-risk calls (unknown commands are high risk), fail-closed when approvals are unreachable | `tests/test_mapper.py`, `tests/test_approval_mode.py`, `startup_scripts/check_stack.py` (live) | Twin-validated. |
| Retrospective and online Instrument Trace pipeline | Per-run JSONL traces registered with the map in Tiled; `read_tool_log` for retrospective analysis | Map run files contain the trace; check_stack reads runs back from Tiled | Retrospective: yes. Online streaming pipeline: not built. |
| STEM reward workflow (field-of-view selection by quality and feature visibility) | Scorer interface with versioned scores (`pyTEMlib.scoring`), baseline contrast/sharpness scorer, scores stored as points of interest with reasons | pyTEMlib `tests/test_scoring.py`; `score_map` in check_stack | Interface and baseline only. Not a learned or validated reward. |
| Capability registry (v0.3) | `docs/athena/registry_entry_stem_twin.yaml` | n/a (document) | Entry ready to merge. |
| Remote/agentic operation with human-in-the-loop | Approval mode switched from SciAgentGUI; held calls run once after approval; denials stop the agent | check_stack (hold → approve → run once → single use) | Twin-validated. |
| Digital twins (WBS 1.3.6) for rehearsing workflows | Seeded twin with stage error, backlash, drift, scan distortion, focus, noise, EDS spectra, dose; hidden ground truth; calibration scripts | `tests/test_twin_models.py`, `test_twin_device.py`, `test_calibration.py` (scripts recover configured errors), `test_map_end_to_end.py` (stitching within 1 px of truth) | Twin values partly unverified; to be calibrated on the instrument. |
| Local-model agent evaluation | `benchmarks/run_tool_json.py`, `benchmarks/run_mapping_trials.py`, results in `benchmarks/results/` | See results below | See below. |

## Agent results on this machine

Laptop with a 6 GB GTX 1660 Ti and 16 GB RAM, Ollama, twin stack as in `docs/Operation/quickstart.md`, 2026-10-27/28.

**Valid tool calls**: 20 instrument requests, scored on the right tool with schema-valid arguments (`benchmarks/results/tool_json.json`).

| Model | Valid | Time for 20 |
| :-- | --: | --: |
| qwen3:8b (default) | 95 % | 218 s |
| qwen3:4b | 100 % | 301 s |

**Mapping trials** (`benchmarks/run_mapping_trials.py`). A trial passes when:

- the map completes with a stitched map key;
- the stage box covers the requested area without being far larger;
- EDS is used exactly when asked;
- interesting regions are scored exactly when asked.

| Task | `mapping` graph (model plans, graph runs the workflow) | `general` agent (model makes every call, using the mapping skill) |
| :-- | --: | --: |
| Map a 600 nm square | 9/10 | 5/5 |
| ... with an EDS spectrum per tile | 10/10 | 2/5 |
| ... and find the 3 most interesting regions | 10/10 | 0/5 |
| Stage moves outside a plan box (all trials) | 0 | 0 |

The `general` column is the last of four rounds. Overall it went 0/15, 1/15, 6/14 (one model-server crash not counted), then 7/15. These changes moved it:

- The Mapper accepts a plan's size and does the box arithmetic (`size_m`).
- Every refused plan comes back with a valid example.
- `wait_for_job` replaces dozens of status polls.
- The prompt tells the agent to act instead of announcing.
- The skill is explicit about stitching and scoring.

What is left is an 8B model following a long procedure: it stops after collecting data, skips scoring, or adds EDS nobody asked for. Use the `mapping` graph when the workflow is known; the general agent is for open-ended requests. Raw rows, including the tools called and skills loaded per trial: `benchmarks/results/mapping_trials_*.json`.

**Live checks**, each run on the stack after every change: `startup_scripts/check_stack.py` passes 19/19 (map, stitch, score, POIs, approval hold/approve/single-use, a held map start showing its plan) and `startup_scripts/check_gui_contract.py` passes 12/12 (SciAgentGUI's approvals and map/spectrum reads).

## Not claimed

- No run on the physical instrument.
- No learned reward, no human-preference data.
- No authenticated identity, authorisation tokens or Kubernetes deployment (WG1).
- No nanoindentation or SPM schemas.
