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

Laptop with a 6 GB GTX 1660 Ti and 16 GB RAM, Ollama models, twin stack as in `docs/Operation/quickstart.md`.

Valid tool calls (20 instrument requests, correct tool and schema-valid arguments): see `benchmarks/results/tool_json.json`.

Seeded mapping trials (10 per task for the `mapping` graph, 5 for the `general` agent): see `benchmarks/results/mapping_trials.json`.

## Not claimed

- No run on the physical instrument.
- No learned reward, no human-preference data.
- No authenticated identity, authorisation tokens or Kubernetes deployment (WG1).
- No nanoindentation or SPM schemas.
