# Sample Mapping Building Blocks for Asyncroscopy

* **Author:** whittlegears
* **Date:** 2026-10-27

## Overview

The idea is to add general building blocks to asyncroscopy so an agent can map a sample at low magnification. It would plan a grid of image tiles over a stage region and run that grid on the server, optionally taking an EDS spectrum at each tile. It would place the tiles by stage position, correct them with image matching, and save the result as a zoomable map file with stage coordinates and a layer of marked points. Agents and ML models could then query the map and drive the stage back to any point on it. A plug-in scorer would rank tiles by how worth a closer look they are. The pieces would be exposed as MCP tools, described in SKILL.md files, run from LangGraph graphs and shown in SciAgentGUI. All of it would be tested on the seeded digital twin with a local Qwen3 model. The same work should also produce evidence for the ATHENA WG2 Year-1 deliverables.

Six reviewers looked at the concept through different lenses: design, codebase details, zoom-out (scope), comprehension, implementation readiness and structure. They agree the core idea is sound. One server-side grid run in place of 1000+ per-tile tool calls is the right fit for a small local model. However, the concept as written rests on four assumptions that are false today:

- The digital twin can test registration.
- One Tango command can run a whole grid.
- Approval can be enforced on the server-side path.
- The asyncroscopy checkout on this machine is the one SciAgentGUI integrates with.

It also bundles four or five separate projects into one.

## Analysis

Findings are grouped by severity and deduplicated across reviewers. Each one names the lenses that raised it. File paths are absolute.

### Blocker

**B1. The digital twin cannot test the registration claim.** Lenses: design, codebase-details, implementation-readiness.
- `_move_stage` adds noise to the target and then stores that noisy value as the stage position. `_get_stage` returns that same value, so the reported position is always the true position (asyncroscopy\asyncroscopy\instruments\electron_microscope\digital_twin.py, lines 638-664). As a result:
  - Placing tiles by reported stage position will look perfect.
  - Image-matching refinement will have nothing to correct.
  - Scores against ground truth will look better than they should.
- The move noise comes from the unseeded global `np.random.normal` (line 652). Runs are therefore not repeatable even with a fixed `sample_seed`.
- The particle ground truth (`_particle_records_base`) is never exposed as a command. `get_viewport_metadata` is a plain method that returns only a particle count (lines 695-721).
- The default sample is 6 nm across and the default field of view is 20 nm (lines 44-48, 104). The whole sample fits inside one tile, so there is nothing to map.
- Rendering builds an atom-level potential for each tile (lines 479-505). Time and memory per tile at around 1 µm field of view have never been measured.

**B2. A whole grid run cannot be one Tango command.** Lenses: design, codebase-details, implementation-readiness, comprehension.
- The timeouts are 30 s per Tango command (asyncroscopy\asyncroscopy\mcp\mcp_server.py, line 59), 60 s per SciAgentGUI MCP request (comment at lines 54-58) and 12 s for the twin's proxy (digital_twin.py, line 146).
- A run of 1000 tiles takes minutes to hours. The client would time out while the server keeps moving the stage.
- safety-gate.md says a long-running call cannot be cancelled once approved (sciagentgui\docs\guide\safety-gate.md, line 110).
- The run has to be a background job: start returns a job id, followed by status, abort and results calls. That set is the same as the WG2 common envelope (reserve, release, status, abort, results). All four lenses said to build the envelope once, here, as the contract for long-running jobs.

**B3. "One approval at the start, and approval to leave the region" has nothing to enforce it.** Lenses: all six.
- The server-side Query path runs tools with no approval step (asyncroscopy\asyncroscopy\mcp\agent_mcp_server.py, lines 104-112; safety-gate.md, lines 73-80; sciagentgui\docs\reference\feature-status.md, lines 33-42).
- The MCP server listens on 0.0.0.0 with no authentication (asyncroscopy\configs\mcp_dt.yaml, line 17).
- The GUI's risk classifier works from words in the tool name, and it treats unknown names as low risk (safety-gate.md, line 107). A tool named `Mapper_RunGrid` contains no word such as "stage", "scan" or "acquire". It would therefore run automatically even in approval mode, though it moves the stage 1000 times.
- The concept does not say where the region check runs, who receives a request to leave the region, or what happens while nobody answers.

**B4. The asyncroscopy checkout here is not the one SciAgentGUI integrates with.** Lenses: codebase-details, structural, design, comprehension, implementation-readiness.
- SciAgentGUI's bridge calls `Complete` on a Tango LLM device (sciagentgui\scripts\llm_bridge.py, lines 64, 99, 170).
- The local `startup_scripts\run_llm.py` starts a FastMCP server instead (`asyncroscopy.mcp.agent_mcp_server`, line 88). That server is not a Tango device.
- The integration skill and CLAUDE.md name `C:\Users\ldunn21\...\Leviasyncroscopy` as the source of truth (sciagentgui\.claude\skills\asyncroscopy-integration\SKILL.md, line 8). They describe `mcp/llm.py`, `mcp/backends/`, `SyncSkills` and a skill store, and none of these exist here.
- The "shared skill registry", the gated GUI path and the backend switching all depend on a fork that is not in this workspace.

**B5. Nothing can run on this machine yet, and the configured model does not fit.** Lenses: zoom-out, implementation-readiness, design, comprehension.
- asyncroscopy requires Python 3.12 or newer (asyncroscopy\pyproject.toml, line 6). Python, uv and Rust are not installed.
- `DEFAULT_MODEL = "gemma4:31b"` appears in asyncroscopy\asyncroscopy\mcp\agent.py (line 30), configs\gemma-llm.yaml (line 8) and startup_scripts\run_llm.py (line 34). That model will not fit in 6 GB of VRAM.
- `ChatOllama` is created without `num_ctx` (agent.py, lines 121-125), so Ollama's small default context will quietly cut off the tool descriptions.
- `include_only_functions: []` (mcp_dt.yaml, line 32) exposes every Tango command as a tool.
- The paper's 189-trial benchmark ran on an RTX 5090 and cannot be repeated at that scale here.

**B6. The concept is four or five projects at once.** Lenses: zoom-out, design, structural.
- It combines:
  - (A) the mapping workflow
  - (B) agent plumbing: the skill registry, the LangGraph restructure and LangGraph Studio
  - (C) a map viewer in Tauri
  - (D) the WG2 contract layer: the envelope, authentication, trace and the capability registry
  - (E) STEM reward research
- The WG2 directive is "real workflow, then minimum useful interface, then validate on the instrument, then generalize" (WG2 Workdoc.docx.txt, line 35). The concept starts by generalizing, before a single workflow runs.

### Major

**M1. Trace logging exists but is off by default and too thin.** Lenses: all six.
- `tool_log: ""` turns it off (mcp_dt.yaml, line 34; mcp_server.py, lines 100, 111, 765).
- Each row records the tool, its arguments, the duration, ok/error and a 500-character result preview (asyncroscopy\asyncroscopy\mcp\tool_log.py, lines 12, 36-45).
- Rows carry no run id, caller, approval id, tile index, stage position or model/prompt record.
- Once one grid-run call replaces 1000 calls, the log will show a single row with nothing inside it.
- Agent-side tracing is `print` statements (agent.py, lines 419-429).
- The GUI audit log never sees server-side calls.
- This log is the cheapest starting point for the Instrument Trace, but it needs per-tile events and IDs that link them together.

**M2. Rebuilding stitching, pyramid files and the viewer when libraries already exist.** Lenses: design, zoom-out, codebase-details (contrarian/YAGNI).
- Both stitcher copies segment atoms first (STEM-ML-Functions\stem_stitcher\stitcher.py, lines 157-161). They also:
  - chain each tile only to the one before it, so the jump at the end of each row has no overlap and errors build up (lines 169-173)
  - fit a full homography rather than a shift (line 305)
  - allocate float64 canvases up to 32768², about 17 GB (lines 354-369)

  The fix is a different method, not a patch.
- Off-the-shelf alternatives:
  - **Registration:** ASHLAR, or `skimage.registration.phase_cross_correlation` seeded by stage position. SciAgentGUI already has phase correlation in Rust.
  - **Map file:** ome-zarr-py `write_multiscale` with scale/translation transforms in stage metres.
  - **Serving regions:** the Tiled server asyncroscopy already uses (mcp_server.py, lines 23, 228-235).
  - **Viewer:** napari (with napari-ome-zarr) on the desktop, or Viv in React.
  - **Grid run:** Bluesky `grid_scan` with ophyd-async Tango support gives pause, resume and abort, plus structured event documents. At least copy its document layout for the trace.
- DZI has no place for coordinate metadata. NGFF has no standard points-of-interest layer, so points need a sidecar file in stage coordinates.

**M3. Memory and disk do not fit a 1000-tile map on 16 GB.** Lenses: design, implementation-readiness, codebase-details.
- `pyramid.py` keeps every level in RAM and saves loose PNGs (STEM-ML-Functions\stem_stitcher\pyramid.py, lines 57, 103-111, 185-200).
- 1000 tiles of 1024² float32 is about 4 GB raw, and the mosaic is about 4 GB more. Ollama, Tango, Tiled and the twin also need memory on the same machine.
- Tiles must be written to disk in chunks as they arrive, with a disk-space check before the run starts.

**M4. Tiles are saved without stage position or field of view.** Lens: codebase-details.
- The twin calls `save_acquisition` with no file attributes (digital_twin.py, line 532). Dataset attributes are only the acquisition type and detector (asyncroscopy\asyncroscopy\data\data_writer.py, line 104).
- Placing tiles by position needs those values in every tile.

**M5. MCP exposes only commands, not attributes.** Lens: codebase-details.
- `_find_tools` only goes through `command_list_query()` (mcp_server.py, line 640).
- These settings are attributes, so an agent cannot set them over MCP:
  - scan dwell time, image size and scan region (hardware\scan.py, lines 18-48)
  - EDS exposure time (detectors\eds.py, line 26)
  - stage x and y (hardware\stage.py, lines 34-70)
- Grid-planning commands must take these as arguments, or new commands are needed.

**M6. ATHENA WG2 priorities do not match a STEM-first envelope.** Lenses: zoom-out, structural, design.
- Typed action schemas start with nanoindentation and then SPM (Project Execution Plan, lines 1318-1322 and 2570-2572).
- The common envelope already exists as nanoindentation schema v0.2. Capability registry v0.3 exists and has another owner. Authentication is a WG1 deliverable (quarterly report, lines 17-19 and 125).
- What this work can honestly claim is the "STEM reward baseline": matched settings and image sequences with quality descriptors, due December 2026 (quarterly report, line 161).
- A twin run counts as pre-validation, not as validation on the instrument.

**M7. The LangGraph restructure and Studio are premature.** Lenses: design, zoom-out, codebase-details, implementation-readiness.
- The graph is built and compiled inside `_run_swarm` on every query (agent.py, lines 438-550).
- A failed import calls `sys.exit(1)` when the module loads (lines 24-27), which would kill Studio's loader.
- In the multi-agent path, workers do not get the recursion limit (lines 476-478).
- With the grid running on the server, the agent makes about five calls. A single ReAct-style agent is enough, and a supervisor swarm on qwen3:4b/8b adds more chances to route the task wrongly.

**M8. Skills and tool names will break as things change.** Lenses: structural, design, implementation-readiness, comprehension.
- Tool names are generated as `{dev_class}_{command}` (mcp_server.py, line 512), and SciAgentGUI's rules forbid hard-coding tool names.
- Two skill registries already exist: the sciagent-skills marketplace and the server skill store, which is in the missing fork.
- skills.md and CLAUDE.md disagree about how skills reach the asyncroscopy provider.
- "Skill registry" and WG2's "capability registry" sound alike but mean different things.

**M9. The stitcher has no installable home.** Lens: structural.
- STEM-ML-Functions has only a requirements.txt, so asyncroscopy cannot depend on it.
- The SciAgentGUI copy adds zoom-hierarchy functions, but that folder is a read-only vendored reference according to SciAgentGUI's CLAUDE.md.

**M10. Stopping, watching and recovering a run are not defined.** Lenses: comprehension, implementation-readiness.
- The concept does not say how a user stops the run at tile 400 or what partial result they keep.
- It does not say what happens when a server restarts, or whether a resumed run skips finished tiles.
- `_query_lock` (agent_mcp_server.py, line 74) runs queries one at a time, so a long run blocks every other query.

**M11. Key terms are undefined.** Lens: comprehension.
- Undefined terms: "interestingness", "intent-level call", "point of interest", "leaving the planned region", "common envelope", "Instrument Trace", "capability registry" and "skill registry".
- A small LLM tends to invent a meaning for a vague word such as "interestingness".

**M12. Windows setup is steep and not written down.** Lenses: comprehension, implementation-readiness.
- A new developer needs Python 3.12, uv, a Tango database on Windows, Rust and the MSVC build tools for Tauri, and langgraph-cli.
- The README assumes bash.

**M13. The scorer plug-in framework and the reward workflow are the same thing.** Lenses: zoom-out, design.
- Per-tile sharpness, drift and feature visibility serve as both the baseline interestingness score and the STEM reward descriptors.
- A plug-in framework with only one plug-in is speculative.

### Nit

- **N1.** Pick one map format. Three candidates leave the viewer, the region query and the metadata undecided. Lenses: comprehension, design, implementation-readiness.
- **N2.** `configs\DigitalTwin copy.yaml` is config sprawl. Lenses: structural, implementation-readiness.
- **N3.** Notebook 13 is a real-hardware demo with one move and three images. It is not a mapping tool, and its printed output is stale. Lens: codebase-details.
- **N4.** `pose_seed` relies on Python `hash()` of float tuples (digital_twin.py, line 511). Note it as an assumption. Lens: implementation-readiness.
- **N5.** Clicking a point in the GUI to drive the stage is a stage move and must go through the gate. Lens: comprehension.
- **N6.** Map queries for a first version need only list tiles in a box, list points and go to a point. Lenses: zoom-out, design.
- **N7.** Worth reusing: the MCP wrapper that turns Tango commands into tools, its repeated-failure guard (mcp_server.py, lines 605-614), `_fallback_transform` with a position hint (stitcher.py, lines 322-333), and `tifffile`, which is already a dependency. Lens: codebase-details.

## Goals vs. Non-Goals

* **In-Scope Goals:**
  - Fix the twin before building anything on it:
    - separate commanded, reported and true stage positions
    - seed the move noise from `sample_seed`
    - add a ground-truth export that agents cannot reach
    - add a mapping-scale sample config (about 1-2 µm with hundreds of particles)
    - measure the time each tile takes
  - One server-side map job: plan, reserve, run in the background, status, abort, results, release.
    - It is driven by a frozen plan: stage bounding box, tile count, dose budget and per-tile actions (HAADF, with EDS optional).
    - The Tango device enforces the plan, and the tool name or a declared risk level triggers the GUI gate.
  - Tile placement by stage position, refined by phase correlation, falling back to stage-only placement when matching confidence is low.
  - One map format: OME-Zarr written chunk by chunk with stage-coordinate transforms, plus a sidecar file of points in stage coordinates.
  - A small set of map tools: list tiles in a box, list points, add a point, go to a point.
  - One scoring function, `score(tile, metadata) -> (value 0-1, reason)`, built from sharpness, drift and feature visibility, with the version logged.
  - Trace logging on by default, with run id, plan id, tile index, caller, approval id and one event per tile.
  - One ReAct-style agent on qwen3 (4b or 8b), with a set `num_ctx` and an allow-list of about six tools.
  - A WG2 evidence package:
    - the per-tile trace as the STEM reward baseline
    - one entry in capability registry v0.3
    - call shapes aligned with the v0.2 envelope
    - a clear statement that this is twin evidence, not instrument evidence
* **Deferred to Stage 5 (in scope, may start in parallel once Stage 2 runs):**
  - LangGraph Studio, `langgraph.json` and the importable-graph restructure, living in asyncroscopy.
  - The skill registry, owned by asyncroscopy. The first entry is one mapping SKILL.md shipped with Stage 4.
  - A map view in SciAgentGUI that reads from Tiled. Use napari as the check viewer until then.
* **Out-of-Scope Non-Goals:**
  - Any instrument logic, agent loop, safety enforcement or analysis inside SciAgentGUI.
  - DZI and pyramidal TIFF output.
  - A scorer plug-in framework, trained ML scorers and a fine-tuned LLM judge.
  - Building authentication (stub the hook; it belongs to WG1) or building a capability registry (add an entry).
  - Nanoindentation and SPM schema work. It belongs on WG2's own track.
  - A live Instrument Trace pipeline. A JSONL record written during the run is enough for now.
  - Repairing either existing stitcher for low-magnification work.
  - A large benchmark on this GPU, gemma4:31b, and real-hardware autonomy.

## Key Decisions & Staging for Next Steps

Decisions made by the author after the first review (2026-10-27):

- **asyncroscopy (whittlegears/asyncroscopy, the local checkout) is the base.** All infrastructure lives there: device servers, the MCP server, the agent loop and graphs, approval and region enforcement, the job envelope, traces and skills. This settles B4. SciAgentGUI's integration notes, which point at Leviasyncroscopy, must be rewritten against this repo. Features that exist only in that fork (`Complete`, `mcp/backends/`, `SyncSkills`) are ported only if a stage needs them.
- **SciAgentGUI becomes a thin client.** It views data and calls asyncroscopy commands. It does not run its own agent loop or hold the safety rules. Its approval screen becomes a front end for an approval queue that asyncroscopy owns, so that a script, the GUI and an agent all hit the same check. This makes decision 2 below mandatory, not optional.
- **Analysis happens in a separate pyTEMlib MCP server.** Acquired data is analysed there, not in the GUI's Rust code and not in the acquisition devices.
- **Tiled stores all data.** The map, its tiles, its points file and the per-tile trace are all registered in Tiled. The OME-Zarr choice therefore has to be checked against what Tiled can serve (multiscale Zarr, or one array per pyramid level).
- **LangGraph Studio and skills follow mapping.** They may be built at the same time once the map job runs, but they are not allowed to block Stages 0-4.

Questions these decisions open:

- The exploration loop now crosses three services: score in pyTEMlib, move the stage in asyncroscopy, read pixels from Tiled. Something has to tie those together. The candidates are the agent, a LangGraph graph in asyncroscopy, or a key contract that pyTEMlib reads directly from Tiled.
- Tile placement needs stage coordinates during acquisition, but the scorer is analysis. Decide which side the stitcher and the scorer each live on.
- Is there a pyTEMlib MCP server today? No. pyTEMlib is the open-source library `pycroscopy/pyTEMlib` (MIT licence, maintained by Gerd Duscher, last pushed 2026-10-16). SciAgentGUI ported parts of it to Rust (`src-tauri\src\analysis\`: eds, eels, fft, image, atoms, peak_fit, probe, optimize) and calls the rest through `scripts\pytemlib_backend.py`. The author wants most of that moved back into pyTEMlib. That has these consequences:
  - **Ownership.** pyTEMlib belongs to another maintainer. The MCP server and any returned features either go upstream as pull requests that he accepts, or live in a separate package that depends on pyTEMlib. A private fork would add a fourth copy of drifting code; the stitcher already has two.
  - **Moving Rust back to Python is a port, not a copy.** Any fixes made in the Rust versions must be found by diffing against the pyTEMlib originals. Each returned function needs a test that runs the Rust and Python versions on the same pyTEMlib example data and compares outputs. `docs\internal\PYTEMLIB_REVIEW.md` already recorded verified behaviour on that data, which gives the baseline.
  - **Dependency weight.** pyTEMlib pulls in xraylib, simpleitk, ase, spglib, scikit-learn and more. It needs its own Python environment check on Windows in Stage 0.
- **Decided (2026-10-27): pyTEMlib work goes upstream.** The maintainer has agreed to accept it once it is complete. Development happens in a fork, `whittlegears/pyTEMlib` (not yet created as of this date), and is offered upstream as one or more pull requests only when built out and tested. Keep the fork rebased on `pycroscopy/pyTEMlib` `main` so the final pull requests stay reviewable; keep the MCP server in its own subpackage so it can be accepted separately from the returned analysis code.
- Recommended split for mapping: asyncroscopy acquires tiles and writes each tile to Tiled with its reported stage position and field of view. The pyTEMlib MCP server reads tiles from Tiled by key, does registration, assembly, the pyramid and scoring, and writes the map back to Tiled. Only keys, scores and stage coordinates pass through the agent. pyTEMlib's `image_tools` already contains registration, so the mosaic code would sit next to it.

Decisions needed before any code is written:

1. **Which asyncroscopy is canonical?** Choose between the local checkout and "Leviasyncroscopy", and decide whether the backends and skill-store work is merged in or dropped. Until then, nothing in SciAgentGUI's integration contract is a dependency.
2. **Where is safety enforced?** In the Tango map device and an MCP-layer policy hook, not in the GUI. Define:
   - the approval token: who issues it, what it covers (bounding box plus margin, tile and dose budget, expiry) and where it is checked
   - what happens to a move outside the region (refuse and queue it for the GUI)
   - the default while waiting (pause with the beam blanked)
3. **Job contract.** Make the map run asynchronous, using the five-verb envelope with a generic shape that nanoindentation and SPM can reuse later.
4. **Map format and home.** Use OME-Zarr plus a sidecar points file. Consolidate the stitcher into `asyncroscopy/mapping/`, and reduce STEM-ML-Functions to experimental scorers that import the asyncroscopy interface. Dependencies should run STEM-ML-Functions -> asyncroscopy <- SciAgentGUI, with SciAgentGUI reaching asyncroscopy only over MCP/HTTP.
5. **Model.** Run a short model test to choose the qwen3 tag, `num_ctx`, the tool allow-list and a token budget. Pass bar: at least 90% valid tool-call JSON over a fixed set of prompts on this GPU.

Staging:

- **Stage 0 (setup):**
  - Install Python 3.12, uv and Rust.
  - Pass `uv sync --extra agent --extra ollama` and the existing tests.
  - Write a Windows quickstart.
  - Point the configs at qwen3, turn trace logging on and delete the duplicate config.
  - Configs gain only optional keys with defaults, and a test checks that existing YAML files still load.
- **Stage 1 (twin):**
  - Make the pose and seeding fixes and add the ground-truth export.
  - Write stage position and field of view into each tile's metadata.
  - Measure render time at mapping scale.
- **Stage 2 (map job):**
  - Planner, background run, abort and resume, a per-tile manifest and an enforced region.
  - Acceptance:
    - tile centres match a closed-form result
    - no centre falls outside the box
    - abort stops within one tile
    - resume skips finished tiles
    - a move outside the region is refused at the device and logged with the approval id
- **Stage 3 (registration and map file):**
  - Acceptance with stage noise at 5% of the field of view:
    - median placement error under 1 px against the true position
    - with no noise, refinement is no worse than stage-only placement
  - Other acceptance:
    - peak RAM under 4 GB for a 2 GB map
    - pixel-to-stage conversion round-trips within 1 px
    - the file opens in napari
- **Stage 4 (agent run):**
  - On the chosen qwen3 tag, at least M of N seeded trials complete "map region, report three particles":
    - in 10 or fewer tool calls
    - with no moves outside the region
  - Record tokens and wall time for each trial.
  - Keep to about 10-20 trials on this machine.
- **Stage 5 (WG2 package and follow-ons):**
  - Write a WG2 traceability table: deliverable, artifact, milestone.
  - Only then consider the SciAgentGUI map view, the LangGraph Studio packaging and the shared skill registry.

The follow-on design doc should contain these sections:

1. Problem and goals
2. The canonical-repo decision
3. Data flow
4. Interfaces: the envelope, the map commands, the map tools and the scorer
5. Data format: the map file and the points schema
6. Safety and autonomy
7. Skill versioning, with minimum asyncroscopy and schema versions in the front matter
8. Migration and deprecation
9. WG2 traceability
10. Validation metrics
11. Alternatives considered: ASHLAR, Bluesky, DZI and pyramidal TIFF
12. Risks
