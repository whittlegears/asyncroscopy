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

What `run_job` guarantees, so an agent cannot leave the instrument tied up:

- Every reply carries the plan's estimate (tiles, duration, dose), also when a held run is retried.
- A plan whose estimate is over the endpoint's `mcp.job_limits` (`max_duration_s`, `max_dose_e_per_a2`; see `configs/mcp_dt.yaml`) is refused before it starts, whether it comes through `run_job` or a direct `<device>_reserve`; the agent is told to ask the user. Only an operator raises the limits.
- A start that fails (an error reply, a Tango failure, an unreachable approval service) releases the reservation it made, and says so.
- A start held in approve mode keeps its reservation for the same caller and agent run. Agents send their run id with every call (`X-Asyncroscopy-Run`), because they open a new MCP session per call; two agents with one caller name stay apart by it. Asking again before the operator decides files no second request. A denied or expired request, or one nobody came back for within two hours, frees the reservation even if the agent never asks again.

- An aborted job can be continued: `run_job(device, resume=<reservation_id>)` (no plan) starts the same reservation again, through the approval gate, and it goes on from the next step. A resume that fails to start keeps the reservation, so it can be tried again.
- `run_job` waits up to 15 s, and an agent's own `run_job` call goes on waiting inside the call until the job ends. Pass `wait=false` when the user wants the job id now, or will stop the job: the call returns as soon as the job runs. An abort reply includes `resume_with`, the exact `run_job` call that continues the job (the agent graphs show it to the model as `to_resume`).

What the graphs do so a small model's context does not overflow (`asyncroscopy/agents/guards.py`), measured against `num_ctx`:

- One tool round per job: when the model calls `wait_for_job`, the call itself keeps waiting until the job ends (up to its estimated duration x1.5 + 2 min, else an hour, and never past the run's deadline) and returns one reply with `waited_s`. A job still running at the cap says so; the model is not asked to poll.
- Before every model call the messages are fitted to what the system prompt and the tool schemas leave: the system prompt, the first and the latest request and the latest tool reply (cut at 6000 characters) are kept; older tool replies become one-line digests, and the oldest finished rounds go first. The skills and memory notes share at most a quarter of the context.
- A call repeated with the same arguments and the same result: the second time the model is told so in the prompt; the third time the run stops with an honest answer saying what was repeated, instead of looping to the recursion limit.

`endpoint_info` (for agent programs, not given to models) reports each tool's risk as the approval gate sees it, the job devices and the analysis server. Agents read the analysis server's address from it, so `mcp.analysis_url` in the MCP config is the one place it is set (`ASYNCROSCOPY_ANALYSIS_URL` overrides it); the goal checker and the benchmarks read risks from it, so a deployment's `mcp.safety.risk` overrides count everywhere.

An endpoint with `mcp.auth` needs a bearer token: agents send `ASYNCROSCOPY_MCP_TOKEN` as `Authorization: Bearer` to the instrument and analysis endpoints, and the endpoint takes the caller from the token's subject instead of `ASYNCROSCOPY_CALLER`. See [Remote access](remote_access.md).

## The team

A small model loses track with many tools and long instructions, so the `team` graph never gives one agent both.

1. **Planner** (one model call, JSON out): splits the request into steps for `operator` or `analyst`. An unusable reply falls back to a plan from the request's words (record, then analyse).
2. **Specialists**, one per step, each holding only its role's tools:
   - `operator`: the instrument: `run_job`, `*_abort`, stage, beam and acquisitions. On a STEM a curated set; on any other instrument (for example a YAML-defined twin) every device command.
   - `analyst`: stored data and maps: `list_data`, `view_data`, `assemble_map`, `score_map`, `map_info`, `get_region`, points of interest.
   - both: `wait_for_job` and `approval_status`. Operator-only tools (`Approvals_*`) are never given to an agent.

   The split is data in the agent profile (`team:` in `asyncroscopy/agents/profiles/default.yaml`: `roles`, `shared`, `hidden`). Whether a tool is a device command, an acquisition or a job's start comes from the endpoint's `endpoint_info` (`tools`, read once per run), never from the tool's name.
3. Each step sees what earlier steps produced as **facts from the servers' replies** (run keys, map keys, job states), not the previous agent's prose. A failed step is retried once with the error as feedback.
4. **Reporter**: writes the answer from those facts only.

## Skills, found automatically

Skills are short procedures in `asyncroscopy/skills/<name>/SKILL.md`. Agents never have to look for them: before every model call `SkillMiddleware` (`asyncroscopy/agents/skill_middleware.py`) decides, by fixed rules and without a model call, which skills apply and writes them into the prompt.

- From the request: the best one or two by `keywords`, name and description; a request for a new map always gets `mapping` first. Negated words ("without EDS spectra") do not count.
- As the task develops, once each:

| Skill | Added when |
| :-- | :-- |
| `recover_from_errors` | a tool call failed |
| `approval_etiquette` | a call was held (APPROVAL REQUIRED), or approval_status said denied or expired |
| `inspect_data` | a data key came back, or the request asks about data |
| `explore_regions` | a stitched map came back and the request asks for regions |

A request counts as acting unless it is clearly read-only: it says not to act ("do not acquire anything"), or it is a question about data that already exist (a key, "the latest", "the stitched map") with no instruction to act anywhere in it ("show me a new image", "by taking a spectrum", "blank it" all act). Taking an acting request for read-only would keep the skills it needs away; the opposite is still held by the risk policy and the approval gate. A read-only request never gets a skill that acts on the instrument (`acts: true` in its front matter, or no `acts` at all: a skill that does not say it only reads counts as acting), whatever comes back during the run. Each turn of a conversation is judged by its own latest request, and `load_skill` is hidden only from a read-only request whose skills all fitted.

At most four skills per run, within about 7000 characters shared with the memory notes; a skill that does not fit is listed by its description and only then is `load_skill` shown to the model. A skill is given only to an agent holding at least one of its tools. Which skills were applied, and why, is kept in the run's state as `skills_applied` (per step in the team, per attempt in a goal run).

| Shipped skill | When |
| :-- | :-- |
| `mapping` | Map an area as tiles, stitch it, find regions |
| `eds_survey` | Composition: EDS at each tile, then read the spectra |
| `explore_regions` | After a map: rank regions, go back closer, mark them |
| `focus` | Autofocus and stigmation as one Tuner job, never a hand loop over `set_defocus` |
| `inspect_data` | Find and read images, spectra and maps; judge peaks and noise |
| `approval_etiquette` | A call is held, approved, denied or expired |
| `find_feature_then_measure` | Find a particle or feature in an image, then a spectrum or closer image on it |
| `recover_from_errors` | A refused or failed call: fix the form, never a value the user stated; after a timeout read the state first; never "reset" by unblanking or toggling |

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

A goal given as a sentence gets only the criteria its words make plain, because a criterion the request did not ask for makes the retry push work nobody wanted:

- a new map from "map a 600 nm square", "make an overview map", "do a 1 um map" (never from a map the request names: "on the map X" or a key), with `N` interesting regions when it asks for them;
- spectra or images from an acquisition verb on them ("take an EDS spectrum", "image them at high magnification"), and data of any kind from "acquire a diffractogram"; "without EDS" takes spectra away;
- `no_unrequested` with the categories of action the words ask for, read from tool names on any instrument: `job` (run_job and the job envelope), `acquire` (`acquire*` ...), `move` (`move*`, `set_stage*`), `settings` (`set_*`), `beam` (blanking, `*_on`/`*_off`). An image allows acquire, settings and beam; "go to" or "revisit" allows move; a map allows job. It is checked against the endpoint's own risk table, and reported as not checked when that is unavailable rather than guessed;
- `grounded_answer`, always.

When the words are unclear the goal has fewer criteria, not wrong ones. A goal file states exactly what you want and is the precise path; inference is a convenience. "Do not wait" (or `leave_running: true` in a goal file) means the job it starts is left running at the end.

Only what the run made counts: acquisitions, jobs it started and their results, and the stitching of a run it acquired. Data found by `list_data` or opened with `view_data`, a wait on someone else's job, and scoring or re-stitching an existing map never count.

After each attempt a checker reads the tool calls and the servers' replies (including `run_job`'s job id and estimate and `wait_for_job`'s final `result_keys`), asks the live servers whether the keys exist (`map_info`, `list_pois`, `view_data`), and either accepts the result, retries with the unmet criteria as feedback, or stops and reports which criteria were not met and why. A dose the request states ("at most 20 e/A^2") becomes the goal's dose budget, checked against the jobs' estimates. Every instrument call of a goal run is recorded as its reply arrives (the run's ledger), so an attempt that hits the time budget or the recursion limit, even in the middle of a team step, still knows the jobs it started: the retry is told the keys this goal made and any job still running (wait for it; do not start another), and when the goal ends, jobs it started that are still running are aborted (unless the request said not to wait) and the report says so. A `run_job` whose reply never arrived before the run stopped is not known; the job device's own busy rule still keeps a second job from starting. The exit code is 0 when the goal was reached and 1 otherwise. Through MCP: `query_agent(prompt, graph="team", goal=True)`, where the prompt is a sentence or the goal as JSON.

## Memory: Markdown the agent reads

The agent keeps notes as plain Markdown in `$ASYNCROSCOPY_AGENT_MEMORY` (default `outputs/agent_memory`, which git does not track; point it at a folder in your repository to keep the notes):

| File | What | Written by |
| :-- | :-- | :-- |
| `INSTRUMENT.md` | What the instrument is: devices, jobs, conventions, limits, samples | Seeded from the tool list on first use; then you |
| `LESSONS.md` | One dated line per lesson ("- 2026-10-28: ...") | A goal run that was not met; you |
| `experiments/<date>_<run>.md` | Request, goal, steps, keys produced, outcome, issues | Every goal run |
| `skills/<name>/SKILL.md` | Skills learned from successful runs | Goal runs that were met (drafts) |

Once per run the sections that share the most words with the request are added to the prompt, at most about 1500 characters: never the whole folder. Data keys, map keys and ids in them are replaced by placeholders, so an agent never takes an earlier run's key for this one's.

`$ASYNCROSCOPY_INSTRUMENT_NOTES` names a team's own instrument notes kept in its repository (for example `configs/instrument_notes/raman.md`); they are read as well, and no `INSTRUMENT.md` is seeded then.

Nothing an agent writes here is reviewed before another run reads it, so it is kept small and checked: a lesson is one short line with keys and ids taken out, refused if it advises getting around approvals or safety, never added twice; `LESSONS.md` keeps the newest 100 and `experiments/` the newest 200 pages.

## Learned skills

After a goal run the checker found **met**, the run's successful calls are turned into a draft skill: the steps in order (repeated waits collapsed) with their arguments, data keys and job ids replaced by placeholders, `tools:` limited to the tools actually used, and provenance (runs, dates, model, success count) in its front matter. One short model call names it and writes its description and keywords. The same tool sequence succeeding again raises the success count of the existing draft instead of making a new one. A run that was not met adds a lesson to `LESSONS.md` instead.

Drafts are never given to an agent. Check, review and promote skills with:

```bash
uv run python -m asyncroscopy.agents.skills list                                  # every skill: shipped, lab, learned (drafts too) and problems
uv run python -m asyncroscopy.agents.skills show <name>
uv run python -m asyncroscopy.agents.skills lint <name or path/to/SKILL.md>       # any skill, before you add it
uv run python -m asyncroscopy.agents.skills promote <name> --check-endpoint       # learned drafts only
```

A learned skill file that cannot be read is reported by `list` (exit code 1), not skipped.

`ASYNCROSCOPY_AUTO_PROMOTE_AFTER=N` promotes a draft after N successes (default 0: never). Promotion re-checks the skill: not too long, no text about getting around approvals or safety, no operator-only tools, no hard-coded data keys, and (with `--check-endpoint`) only tools the agent endpoint offers. A promoted learned skill never replaces a shipped or lab skill of the same name, is given only to an agent offered every tool it lists, and only describes calls: every call still goes through the approval gate.

## For a hackathon team

1. Copy `docs/Operation/templates/INSTRUMENT.md` into your repository (for example `configs/instrument_notes/<instrument>.md`) and set `ASYNCROSCOPY_INSTRUMENT_NOTES` to it, so the notes are versioned with your code (or start the stack and run one agent request: it seeds `outputs/agent_memory/INSTRUMENT.md` from your tool list). Fill in your limits, units and samples, in the words your requests will use.
2. Write a skill for your instrument's main procedure: copy `docs/Operation/templates/SKILL.md` into `asyncroscopy/skills/<name>/` (or a folder in `ASYNCROSCOPY_SKILLS_PATH`), set `keywords`, `requires`, `tools` and `acts`, list the steps, and check it with `python -m asyncroscopy.agents.skills lint <path>`. Agents get it as soon as a request matches.
3. Run your workflow as a goal on the twin (`--goal`); look at the experiment page, the lesson or the learned draft it leaves, and promote drafts you agree with.

## Testing without a model

`tests/test_team_graph.py`, `tests/test_goals.py`, `tests/test_agents_skills.py` and `tests/test_agents_memory.py` run the graphs, skill triggers, memory and learning with a scripted model and fake MCP servers. Live trials on the twin: `benchmarks/run_mapping_trials.py`.
