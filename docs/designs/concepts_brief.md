# Concept brief: a portable, hackathon-ready asyncroscopy stack

Status: brainstorm output (not a plan of record). Date: 2026-10-29.
Method: six independent reviewers (design, codebase details, zoom-out, comprehension, implementation readiness, structure) read the repository and reported in parallel. This brief merges and deduplicates their findings. Line numbers refer to the repository as of today.

## 1. Overview

The concept: a hackathon team clones asyncroscopy, has the digital twin running within minutes, builds a twin of its own instrument from a YAML file, and checks an agent workflow on that twin before anyone touches real hardware. The work should also produce evidence for the ATHENA WG2 deliverables.

The main finding is that **most of the features already exist**. The YAML twins, the job envelope, the approval gate, goal checking, workflow benchmarks and isolated second stacks all work. What is missing is:

- a short, cross-platform path to the first result;
- a pass/fail check that works for any instrument, not only STEM;
- proof, in CI, that the stack runs on computers other than the developer's Windows machine;
- a safe hand-over from twin to real hardware.

One reviewer also found a real safety gap. Today, the real-instrument MCP config runs with **no approval gate**.

## 2. Analysis

Findings are grouped by theme. The severity is the highest any reviewer gave the finding. The "Lenses" line names the reviewers who raised it.

### 2.1 Safety on the way to hardware

- **Blocker: the real-instrument config bypasses the approval gate.**
  - `configs/mcp.yaml` sets no `approvals_device_address` and binds `http_host: 0.0.0.0` (line 17).
  - `mcp_server.py:734` skips the gate whenever the address is `None`.
  - The approvals device defaults to `auto` (`approval_mode.md:59`), and nothing ties the mode to "simulated or real".
  - The result: a workflow validated on the twin in `auto` mode would run unheld on the Spectra 300, and any machine on the network could reach it.
  - Lenses: design, implementation readiness.
- **Major: there is no dry-run and no sim-to-real check.**
  - No mode checks and traces high-risk calls without sending them to the device.
  - No check confirms that a sim and its real driver share command names, units, bounds and risk classes. The docs only ask for this (`new_instrument.md:65,263`).
  - `configs/mcp.yaml:34-36` allows 4 h and 5000 e/A², which are twin-scale limits.
  - Lenses: design, implementation readiness.
- **Major: a client that is not pointed at the right stack silently drives the twin instead of failing** (`second_stack.md:61-62`). Lenses: design, implementation readiness.
- **Nit (Major on hardware): risk class comes from command names, not from a declared field.** YAML commands have no `risk` key (`spec.py:109`). Lens: design.

### 2.2 Running on other computers

- **Blocker: CI does not test what the quickstart promises.**
  - `.github/workflows/test.yaml:38` runs `uv sync --dev`, but `pyproject.toml` has no `[dependency-groups]`. `dev` and `agent` are extras (lines 68, 85).
  - Agent tests import `langchain_core` at module level (for example `tests/test_goals.py:8`).
  - The cache is keyed on `uv.lock`, which is gitignored.
  - The matrix covers Ubuntu and Windows only (lines 15-17). There is no macOS runner and no live-stack job.
  - Lenses: codebase details, implementation readiness, zoom-out.
- **Major: there is no lockfile.** Versions resolve fresh on every machine (`quickstart.md:45`), and pyTEMlib is pinned to a personal fork (`pyproject.toml:120`). Lenses: codebase details, design, implementation readiness.
- **Major: starting and stopping the stack is documented only for PowerShell** (`quickstart.md:60-89,176`, `second_stack.md:66-70`).
  - `asyncroscopy/utils/process_manager.py` already has the POSIX code path, but nothing exposes it as a command.
  - Job state paths are relative to the launch folder (`job_device.py:228-229`).
  - Lenses: all six.
- **Major: the core install is heavy.**
  - Every user installs vendor stubs, PyQt5 and PyQt6, jupyter and `tiled[all]` (`pyproject.toml:14-57`).
  - The quickstart installs Ollama and pulls 5 GB before the twin even starts (`quickstart.md:13,28-32`).
  - Lenses: zoom-out, comprehension.

### 2.3 Building your own instrument

- **Major: `check_stack.py` checks only the STEM stack** (`README.md:35`, `new_instrument.md:211`).
  - The building blocks for a general check already exist: `endpoint_info`, `list_tool_risks`, `run_job`/`wait_for_job`, `asyncroscopy.testing`, and the spec validator (`spec.py:910`).
  - Lenses: all six.
- **Major: STEM assumptions are built into the core.**
  - `PREVIEW_COMMANDS` is hard-coded (`mcp_server.py:61`).
  - Job limits accept only duration and electron dose (`mcp_server.py:228-230,915`).
  - There is a single `analysis_url`, and analysis tools are dropped on stacks that are not STEM (`general.py:95-96`).
  - Agents default to the `Mapper` device and STEM units (`goals.py:228,511`, `mapping.py:61`).
  - The neutral stage, clock and rng models live under `electron_microscope/twin/`.
  - On a large YAML instrument the tool count (2N+M) may overflow `num_ctx` 8192. That could make the 8B model's false claims worse.
  - Lenses: design, structural, codebase details.
- **Major: there is no single template to copy.**
  - `afm.yaml` is recommended but has no launcher config.
  - `yaml_sim_xrd.yaml` reuses the twin's ports.
  - Legacy instrument folders sit beside the template.
  - `configs/` holds 25+ flat files, including personal machine files.
  - Lenses: structural, comprehension, zoom-out, codebase details.
- **Major: a YAML twin has no lasting sample.** Spots are redrawn on every call (`spec.py:746-748`), so a team cannot rehearse "find a feature, then go back to it". Lens: design.

### 2.4 Validating agent workflows on the twin

- **Major: nothing checks the agent's claimed actions against the device log.**
  - `grounded_answer` (`goals.py:582-587`) checks only that the data keys in the answer exist.
  - The known miss where the model reported resetting defocus without doing it (`wg2_traceability.md:62`) would still pass.
  - Lenses: implementation readiness, codebase details.
- **Major: validation needs a live model and a live stack.**
  - There is no scripted-plan mode and no offline re-scoring of recorded runs (`run_workflows.py:824-833`).
  - No document defines what "validated" means.
  - Lenses: zoom-out, implementation readiness, comprehension.
- **Major: two agent implementations sit side by side.** `query_agent` with the default empty `graph` uses the older `AgentSwarm`, not the LangGraph graphs (`agent_mcp_server.py:105-126`). Lens: codebase details.

### 2.5 Docs and ATHENA evidence

- **Blocker: a newcomer gets no early win.**
  - The core terms (Tango, MCP, twin, Tiled, job envelope) are never defined.
  - No step shows the twin producing an image before the GUI, which has no install link.
  - Lens: comprehension.
- **Blocker: the published site hides the pages for journeys 2 and 3.** `docs/myst.yml` leaves out `agents.md`, `second_stack.md`, `large_maps.md` and `athena/*`. Lens: structural.
- **Major: the docs contradict each other.**
  - Some pages say there are two agent graphs; there are three, including `team` (`README.md:45`, `quickstart.md:155`).
  - Two pages describe `wait_for_job` timing differently (`new_instrument.md:241` vs `agents.md`).
  - Four MCP pages overlap, and two troubleshooting sources overlap.
  - Lenses: comprehension, structural, codebase details.
- **Major: the traceability doc does not map cleanly to deliverables.**
  - Rows have no deliverable IDs, and evidence is cited as commit hashes buried in prose.
  - The "Not claimed" list omits most of the known open items.
  - `related_work.md:45` links to a section that does not exist.
  - Lenses: structural, design, comprehension.
- **Major: the abTEM license statement contradicts the code.** `related_work.md` says GPL abTEM is "not imported", but `digital_twin_diffraction.py:295` and `digital_twin_tilt.py:507` import it. This needs a maintainer ruling. Lens: codebase details.

### 2.6 What already holds up (keep it)

- High-risk calls are refused when the approvals device cannot be reached.
- Limits are checked when a plan is reserved, and a plan whose limits cannot be read is refused.
- Expressions in YAML specs are checked against an allow-list.
- The job envelope is one shared, versioned contract.
- The goal checker works from facts reported by the server.

## 3. Goals vs Non-Goals

### Goals

1. On Windows, Linux and macOS, go from a fresh clone to a running twin with a passing check in under 10 minutes, with no Ollama, GPU, GUI or pyTEMlib.
2. Turn a team's YAML instrument into a running stack with a generic PASS/FAIL check by following one numbered recipe.
3. Validate a workflow on the twin with a pass/fail that does not need an LLM, and that catches claimed actions the agent did not take.
4. Hand over from twin to hardware with the approval gate, conservative limits and a dry-run enforced, not just recommended.
5. Back every ATHENA WG2 claim with a file-path evidence cell and an honest "Not claimed" list.

### Non-goals (state these in the docs)

- Maps above about 400,000 tiles, and stitching accuracy on the largest maps.
- Live drift correction during a map.
- A physical microscope run. Access to the instrument is controlled by the facility, so this is not a code task. The work here only prepares for it.
- Stitching, scoring and revisit for instruments other than STEM. The work here only defines the analysis-provider contract.
- Fixing the 8B model itself. Detect its false claims instead.
- Learned skills, auto-promotion, team-graph changes, the online trace pipeline, and authentication (authentication belongs to WG1).

## 4. Key Decisions and Staging

### Decisions to make first

- **D1: Is safety tied to `simulated: true|false` in the stack config?** Recommended: yes. A config that is not simulated defaults to `approve`, refuses `auto` unless the operator sets a traced override, and refuses to start without an approvals device.
- **D2: Commit `uv.lock`?** Recommended: yes, and use `uv sync --locked` in CI and in the docs.
- **D3: Where do STEM-specific parts live?** Recommended: in "instrument profiles" (tool sets, preview commands, unit hints, job device name, analysis providers) that each instrument package registers. The core stays neutral.
- **D4: What counts as "validated on the twin"?** Recommended: the goal file passes N of M runs, `no_unrequested` holds, the new `claims_match_trace` criterion holds, and the run is reproducible offline.
- **D5: abTEM.** Either run it in a separate process or correct the doc. This needs a maintainer ruling.

### Staging

- **Stage 1, before the hackathon:** steps 1 to 5 below. These are mostly small and medium launcher, CI and doc changes.
- **Stage 2, during or after the hackathon:** steps 6 and 7 (validation strength, instrument profiles).
- **Stage 3, before first light:** step 8.
- **Quick wins to do alongside Stage 1:**
  - Extend "Not claimed" with every known open item.
  - Fix the `related_work.md:45` link.
  - Give the XRD config its own ports.
  - Move personal configs out of `configs/`.

## 5. Top 8 next steps (ranked by value to hackathon users and ATHENA)

1. **Close the hardware approval bypass.**
   - Real-instrument configs must name a reachable approvals device that defaults to `approve`, and must bind to `127.0.0.1` by default.
   - Done when: a test starting `configs/mcp.yaml` with no approvals address fails at startup, and `set_mode(auto)` from the agent endpoint is refused on a config that is not simulated.
2. **Make CI prove "other computers".**
   - Fix the sync command to `uv sync --extra agent --extra ollama --locked` and commit `uv.lock`.
   - Add macOS.
   - Add a live job that starts the twin and runs `check_stack.py` with `time_scale: 0`.
   - Done when: CI is green on Ubuntu, macOS and Windows, `test_goals.py` and `test_team_graph.py` are reported as passed, and the live twin check passes.
3. **Add a light install and a one-command, cross-platform launcher.**
   - Move vendor, GUI and notebook dependencies into extras.
   - Add `startup_scripts/stack.py up|down|status`, built on `process_manager.py`, that waits until the stack is ready.
   - Ship a hackathon config with `time_scale: 0`.
   - Done when: on a fresh Linux machine and a fresh Windows machine, `uv sync` then `stack.py up` then `check_stack.py` passes within 10 minutes, with no Ollama installed.
4. **Add a generic stack check for any instrument.**
   - Add `check_stack.py --generic`. It reads `endpoint_info`, lists risks, runs each job's `PLAN_EXAMPLE` through `run_job` and `wait_for_job`, reads one key, and runs one hold, approve and run-once cycle.
   - Done when: it passes on the AFM, XRD and Raman YAML twins and fails, naming the field, on a deliberately broken YAML file.
5. **Write a hackathon onboarding path.** Put it on one page, in the order "first image", then "own instrument", then "agent", then "approval", then "goal". It should include:
   - a glossary;
   - a first-PNG snippet;
   - a numbered YAML recipe, plus a new `configs/yaml_sim_afm.yaml`;
   - a manual tool-call client;
   - bash equivalents for every command;
   - a fix for the stale "two graphs" and `wait_for_job` text;
   - every page added to the `myst.yml` table of contents.

   Done when: a newcomer on Linux reaches a PASS and a saved image using only that page, and a test confirms every docs page is either in the TOC or on an exclude list.
6. **Make twin validation stronger and independent of the LLM.**
   - Add a `claims_match_trace` criterion and a `final_state` criterion.
   - Add a scripted-plan mode (`--script calls.yaml`).
   - Add `run_workflows.py --rescore`.
   - Done when: the recorded `focus_series` miss is marked as failed with "claimed set_defocus(0); no such call", and CI re-scores the stored 31-workflow result with no model and no stack.
7. **Move STEM assumptions out of the core into instrument profiles.**
   - Profiles hold the preview commands, unit hints and the job device name.
   - Make `job_limits` generic, for example `max_force_n`.
   - Replace the single analysis URL with a list of analysis providers.
   - Done when: a grep of `mcp/`, `jobs/` and `agents/` for `Mapper|acquire_scanned_image|blank_beam` finds matches only in the profiles, and an AFM job over its force limit is refused at `run_job`.
8. **Prepare the hand-over to hardware.**
   - Add a sim-to-real conformance check (command names, arguments, units, bounds, risk class).
   - Add a dry-run endpoint mode that compares traces against the twin.
   - Write a first-light runbook with tiny limits.
   - Done when: the conformance check fails, naming the field, on a twin copy with a renamed argument; and a dry-run of the 600 nm mapping goal against `Spectra300.yaml` produces a trace whose differences from the twin trace are empty or explained, with the stage unmoved.
