# Tool usage analysis (general agent, mapping trials)

Source: the `tool_names` of every `general` row in `benchmarks/results/mapping_trials_*.json`
(22 files, 217 rows with a tool list, qwen3:8b on the digital twin, up to commit 5149b39), and
`outputs/trace/mcp_agent.jsonl` (8,325 calls on the agent endpoint; 7,294 from the trials).

## Calls per tool (217 rows)

| Tool | Calls | Tool | Calls |
| :-- | --: | :-- | --: |
| wait_for_job | 726 | map_info | 112 |
| analysis_job_status | 301 | Mapper_release | 50 |
| analysis_job_results | 269 | list_devices | 38 |
| Mapper_reserve | 235 | list_scorers | 35 |
| Mapper_results | 227 | list_skills | 25 |
| assemble_map | 218 | list_pois | 13 |
| load_skill | 217 | DigitalTwin_get_stage / get_fov | 10 / 10 |
| Mapper_start | 217 | get_region | 9 |
| score_map | 147 | DigitalTwin_get_parameters | 8 |
| Mapper_job_status | 147 | Mapper_abort | 4 |

Never called in any row although offered: `Approvals_get_mode`, `add_poi` (outside the mapping graph),
`DigitalTwin_move_stage`, `*_acquire_*`, `*_set_fov`, `*_set_defocus`, `*_place_beam`, `*_blank_beam`,
`*_unblank_beam` (correctly: the mapping skill forbids them for maps). On the endpoint, no agent call ever
went to DATA's `configure`, `copy_and_register_remote_file`, `reconcile`, `register_path`,
`start_tiled_server`, `stop_tiled_server`, to Approvals' `check`, `request`, `request_status`,
`list_grants`, `list_pending`, or to `Mapper_get_reservation`.

## Calls that always come together

| Sequence | Count |
| :-- | :-- |
| First call is `load_skill` | 205 of 217 rows (the other 12 start with `list_skills`, then 9 of those call `load_skill`) |
| `Mapper_reserve` → `Mapper_start` next | 217 of 235 reserves; the other 18 are reserve → reserve after a refusal |
| A successful reserve that was started (trace) | 316 of 316; 0 successful reserves were left unstarted. The 103 refused reserves were `plan_rejected` 97, `bad_request` 5, `busy` 1 |
| `analysis_job_status` → `analysis_job_results` next | 269 of 301 (the rest are repeated status polls) |
| `assemble_map` → `wait_for_job` (the instrument's wait, wrong for an analysis job) | 166 of 218; → `analysis_job_status` 52 |
| `wait_for_job` → `Mapper_results` | 193; `Mapper_job_status` → itself 106 (polling that `wait_for_job` replaces) |
| `Mapper_results` → `Mapper_release` | 43 of 227 (a finished job does not block the next reservation, so release is not needed after `done`) |

## Calls per successful trial

| Task | Rows | Median | Min | Max |
| :-- | --: | --: | --: | --: |
| map_square | 72 | 10 | 8 | 22 |
| map_with_eds | 61 | 13 | 12 | 14 |
| map_and_explore | 64 | 17 | 13 | 25 |

## What changed

| Before | After | Why |
| :-- | :-- | :-- |
| `Mapper_reserve` + `Mapper_start` | `run_job(device, plan)` | always paired; the start still goes through the approval gate once, with the reserved plan shown to the operator; repeating the same `run_job` after approval starts that reservation; `approval_status` frees it when denied or expired |
| `wait_for_job` (instrument jobs) and `analysis_job_status` (analysis jobs) | `wait_for_job(job_id)` for both | one concept; the model already used `wait_for_job` for analysis jobs 166 times |
| `wait_for_job` → `Mapper_results` / `analysis_job_status` → `analysis_job_results` | results in the wait's reply when `done` | always followed |
| `load_skill` as the first call | skills matched to the request and added to the prompt before each model call (`agents.skill_middleware`); `load_skill` only for skills that do not fit | always made first |
| `Mapper_job_status`, `Mapper_release`, `Approvals_get_mode`, `list_skills` in the general agent's set | left out | covered by `wait_for_job`, not needed after `done`, never called, skills are in the prompt |
| 12 DATA / Approvals / Mapper infrastructure commands, and each device's Tango `State`/`Status`, on the agent endpoint | hidden there (`configs/mcp_dt.yaml` `blocked_functions`) | never called by an agent (nor `State`/`Status` by SciAgentGUI); the devices and the gate call the rest directly |

Tool counts:

- Agent endpoint (`configs/mcp_dt.yaml`, mapping twin): 76 → 53 by count of the devices' commands (12 infrastructure commands and 12 `State`/`Status` hidden, `run_job` added); `check_stack.py` prints the live count.
- General agent's curated set on that endpoint: 35 → 25 (plus `load_skill` when a skill does not fit in the prompt).

Fewest calls for each task (no retries, one wait per job):

| Task | Before | After |
| :-- | :-- | :-- |
| map_square | 8: load_skill, Mapper_reserve, Mapper_start, wait_for_job, Mapper_results, assemble_map, analysis_job_status, analysis_job_results | 4: run_job, wait_for_job, assemble_map, wait_for_job |
| map_with_eds | 9: as above + view_data on a spectrum | 5: as above + view_data |
| map_and_explore | 11: as map_square + score_map, analysis_job_status, analysis_job_results | 6: as map_square + score_map, wait_for_job |

The endpoint keeps `Mapper_reserve`, `Mapper_start`, `Mapper_job_status`, `Mapper_results`, `Mapper_release`,
`list_skills` and `load_skill`: the `mapping` graph, SciAgentGUI (which identifies servers by the skill tools)
and other clients use them.
