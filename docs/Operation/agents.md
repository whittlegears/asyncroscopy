# Agents: graphs, a team, goals, skills and memory

asyncroscopy ships three LangGraph agents. All of them reach the instrument only through the MCP endpoints, so the safety rules (risk policy, approval mode, operator-only tools) apply to every call they make. They are built for a small local model (qwen3:8b through Ollama on a 6 GB GPU) and work the same with a larger one.

| Graph | Who decides the calls | Use it for |
| :-- | :-- | :-- |
| `mapping` | Code; the model only writes the map plan | Maps: the most reliable path for a known workflow |
| `general` | One model, with a curated tool set | Open-ended requests on one instrument |
| `team` | A planner, then an `operator` and an `analyst` agent, then a reporter | Requests that record data and then analyse it |

```bash
uv run python -m asyncroscopy.agents.run mapping "map a 600 nm square here with EDS"
uv run python -m asyncroscopy.agents.run general "what is in the latest spectrum?"
uv run python -m asyncroscopy.agents.run team "map a 600 nm square here and find the 3 most interesting regions"
```

The agent MCP server (`asyncroscopy/mcp/agent_mcp_server.py`) runs the same graphs through `query_agent(prompt, graph="team")`.

Long experiments are one job: `run_job` (plan checked, then started, through the approval gate) and `wait_for_job` (waits on instrument jobs and on the analysis server's stitching and scoring jobs, and returns their `result_keys` and `summary` when done). The envelope's separate reserve/start/status/results/release tools stay on the endpoint for scripts and the `mapping` graph, but agents are not given them.

## The team

A small model loses track with many tools and long instructions, so the `team` graph never gives one agent both.

1. **Planner** (one model call, JSON out): splits the request into steps for `operator` or `analyst`. An unusable reply falls back to a plan from the request's words (record, then analyse).
2. **Specialists**, one per step, each holding only its role's tools:
   - `operator`: the instrument: `run_job`, `*_abort`, stage, beam and acquisitions. On a STEM a curated set; on any other instrument (for example a YAML-defined twin) every device command.
   - `analyst`: stored data and maps: `list_data`, `view_data`, `assemble_map`, `score_map`, `map_info`, `get_region`, points of interest.
   - both: `wait_for_job` and `approval_status`. Operator-only tools (`Approvals_*`) are never given to an agent.
3. Each step sees what earlier steps produced as **facts from the servers' replies** (run keys, map keys, job states), not the previous agent's prose. A failed step is retried once with the error as feedback.
4. **Reporter**: writes the answer from those facts only.

## Skills, found automatically

Skills are short procedures in `asyncroscopy/skills/<name>/SKILL.md`. Agents never have to look for them: before every model call `SkillMiddleware` (`asyncroscopy/agents/skill_middleware.py`) decides, by fixed rules and without a model call, which skills apply and writes them into the prompt.

- From the request: the best one or two by `keywords`, name and description.
- As the task develops, once each:

| Skill | Added when |
| :-- | :-- |
| `recover_from_errors` | a tool call failed |
| `approval_etiquette` | a call was held (APPROVAL REQUIRED), or approval_status said denied or expired |
| `inspect_data` | a data key came back, or the request asks about data |
| `explore_regions` | a stitched map came back and the request asks for regions |

At most four skills per run, within about 7000 characters; a skill that does not fit is listed by its description and only then is `load_skill` shown to the model. A skill is given only to an agent holding at least one of its tools. Which skills were applied, and why, is kept in the run's state as `skills_applied` (per step in the team, per attempt in a goal run).

| Shipped skill | When |
| :-- | :-- |
| `mapping` | Map an area as tiles, stitch it, find regions |
| `eds_survey` | Composition: EDS at each tile, then read the spectra |
| `explore_regions` | After a map: rank regions, go back closer, mark them |
| `inspect_data` | Find and read images, spectra and maps; judge peaks and noise |
| `approval_etiquette` | A call is held, approved, denied or expired |
| `recover_from_errors` | A refused or failed call: fix it, do not repeat it |

## Goals

A goal is what must be true at the end, checked from facts rather than taken from the model's word.

```bash
uv run python -m asyncroscopy.agents.run team --goal "map a 600 nm square here with EDS and find 3 interesting regions" --max-minutes 30 --max-dose 200
uv run python -m asyncroscopy.agents.run team --goal-file goal.yaml
```

```yaml
# goal.yaml
statement: Map a 600 nm square around the current position with an EDS spectrum at each tile.
criteria:
  - {kind: map}                                  # a stitched map from this run opens
  - {kind: data, data_kind: spectrum, count: 9}  # spectra from this run open with view_data
  - {kind: no_unrequested}                       # no high-risk call beyond the job (allow: [...] to widen)
  - {kind: grounded_answer}                      # the answer names only keys the tools returned
budget: {max_seconds: 1800, max_tool_calls: 60, max_dose_e_per_a2: 200, max_attempts: 2}
```

A goal given as a sentence gets criteria from its words: a map, `N` interesting regions, spectra or an image; `no_unrequested` and `grounded_answer` are always added.

After each attempt a checker reads the tool calls and the servers' replies (including `run_job`'s job id and estimate and `wait_for_job`'s final `result_keys`), asks the live servers whether the keys exist (`map_info`, `list_pois`, `view_data`), and either accepts the result, retries with the unmet criteria as feedback, or stops and reports which criteria were not met and why. The exit code is 0 when the goal was reached and 1 otherwise. Through MCP: `query_agent(prompt, graph="team", goal=True)`, where the prompt is a sentence or the goal as JSON.

## Memory: Markdown the agent reads

The agent keeps notes as plain Markdown in `$ASYNCROSCOPY_AGENT_MEMORY` (default `outputs/agent_memory`):

| File | What | Written by |
| :-- | :-- | :-- |
| `INSTRUMENT.md` | What the instrument is: devices, jobs, conventions, limits, samples | Seeded from the tool list on first use; then you |
| `LESSONS.md` | One dated line per lesson ("- 2026-10-28: ...") | A goal run that was not met; you |
| `experiments/<date>_<run>.md` | Request, goal, steps, keys produced, outcome, issues | Every goal run |
| `skills/<name>/SKILL.md` | Skills learned from successful runs | Goal runs that were met (drafts) |

Before each model call the sections that share the most words with the request are added to the prompt, at most about 1500 characters: never the whole folder.

## Learned skills

After a goal run the checker found **met**, the run's successful calls are turned into a draft skill: the steps in order (repeated waits collapsed) with their arguments, data keys and job ids replaced by placeholders, `tools:` limited to the tools actually used, and provenance (runs, dates, model, success count) in its front matter. One short model call names it and writes its description and keywords. The same tool sequence succeeding again raises the success count of the existing draft instead of making a new one. A run that was not met adds a lesson to `LESSONS.md` instead.

Drafts are never given to an agent. Review and promote them:

```bash
uv run python -m asyncroscopy.agents.skills list
uv run python -m asyncroscopy.agents.skills show <name>
uv run python -m asyncroscopy.agents.skills promote <name> --check-endpoint
```

`ASYNCROSCOPY_AUTO_PROMOTE_AFTER=N` promotes a draft after N successes (default 0: never). Promotion re-checks the skill: not too long, no text about getting around approvals or safety, no operator-only tools, no hard-coded data keys, and (with `--check-endpoint`) only tools the agent endpoint offers. A promoted learned skill never replaces a shipped or lab skill of the same name, is given only to an agent offered every tool it lists, and only describes calls: every call still goes through the approval gate.

## For a hackathon team

1. Copy `docs/Operation/templates/INSTRUMENT.md` to `outputs/agent_memory/INSTRUMENT.md` (or start the stack and run one agent request: it seeds one from your tool list) and fill in your limits, units and samples, in the words your requests will use.
2. Write a skill for your instrument's main procedure: copy `docs/Operation/templates/SKILL.md` into `asyncroscopy/skills/<name>/` (or a folder in `ASYNCROSCOPY_SKILLS_PATH`), set `keywords`, `requires` and `tools`, and list the steps. Agents get it as soon as a request matches.
3. Run your workflow as a goal on the twin (`--goal`); look at the experiment page, the lesson or the learned draft it leaves, and promote drafts you agree with.

## Testing without a model

`tests/test_team_graph.py`, `tests/test_goals.py`, `tests/test_agents_skills.py` and `tests/test_agents_memory.py` run the graphs, skill triggers, memory and learning with a scripted model and fake MCP servers. Live trials on the twin: `benchmarks/run_mapping_trials.py`.
