# find_tools on qwen3:8b (2026-10-30)

Can a small local model reach a tool that is not in its curated list, through `find_tools` and `call_tool`?
Twin stack (`configs/digital_twin_mapping.yaml`) in approve mode, general agent, qwen3:8b on Ollama
(`num_ctx` 8192, 6 GB GTX 1660 Ti), one run per question. Each question needs a read-only tool outside the
curated list: `TestStage_get_holder`, `DigitalTwin_get_dose`, `Mapper_get_jobs`.

## Tool schema size (measured on the agent endpoint)

| Selection | Tools | Characters | About tokens |
|---|---|---|---|
| Every tool | 79 | 37,918 | 9,500 (more than `num_ctx`) |
| Curated `agent_tools` | 22 | 10,771 | 2,700 |
| `find_tools`, `call_tool` and the job tools only | 5 | 3,182 | 800 |

## Runs

| Prompt variant | Holder question | Dose question | Jobs question |
|---|---|---|---|
| No rule about `find_tools` | Answered with the sample from `get_sample`; no tilt limits | Said it could not measure dose | Empty answer |
| Rule: search with `find_tools` before saying it cannot | Same as above | Tried `place_beam` (held for approval; denied by the operator) | Empty answer |
| Rule plus the names of every other tool | Empty answer | Empty answer after `get_stage` | Empty answer |

`find_tools` was never called. Naming every other tool made the prompt long enough that the model returned
nothing; a simple question ("What is the current stage position?") answered correctly between runs, so the
model itself was working.

## What this means

On this model and GPU the curated list is what the agent can use: a tool the agent needs must be in
`agent_tools` (or a loaded skill's `tools`). `find_tools`/`call_tool` stay for larger models and for
harnesses such as SciAgentGUI, which list every tool. The short search rule stays in the prompt (about 60
tokens); the list of names does not.
