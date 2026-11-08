# Aligning asyncroscopy, pyTEMlib and SciAgentGUI with upstream

* **Date:** 2026-11-06
* **Status:** built on `refactor/upstream-alignment` (asyncroscopy), `refactor/sidpy-tools` (pyTEMlib fork) and `claude_dev` (SciAgentGUI); not pushed.

Still open:

- asyncroscopy points pyTEMlib at the local `../pyTEMlib-sidpy` until that branch is pushed and pinned by rev.

## Why

The fork is 209 commits and about 47k lines ahead of pycroscopy/asyncroscopy `main` (f12a44f). The pyTEMlib fork is 54 commits ahead of pycroscopy/pyTEMlib. Much of that work was built for one digital twin, one STEM, and one small local model (qwen3:8b). An audit of all three repositories on 2026-11-06 found the same pattern everywhere:

- Code sits in the wrong layer. A Tiled map store and job queue live inside an analysis library. A modality-free Mapper lives under `electron_microscope/`. Data, safety and job code imports from `mcp/`.
- Instrument facts are guessed instead of declared. Three separate word-parsers decide from a tool's name whether it is risky or what it does. The GUI guesses again in Rust and in TypeScript.
- Model-specific workarounds sit in core code, such as prompt text inside tool replies and regex claim checks.
- Base classes are missing where several implementations exist (scanning-beam instruments, twins, nano-indenter, model providers).

This document is the target. Each section lists the upstream convention, the target layout and the work items.

## The upstream conventions we align to

**asyncroscopy** (`docs/dev_guide.md`):

- Every instrument part is a Tango device. `instruments/instrument.py` holds the abstract `Instrument`. Each modality has a base device, for example `ElectronMicroscope(Instrument)`, and vendor or twin subclasses implement its `_private` methods, for example `AutoScriptMicroscope`, `JEOL` and `DigitalTwin`.
- Sub-devices live in `hardware/` and `detectors/` as a base plus vendor files: `stage.py` with `stage_autoscript.py`, and `corrector.py` with `corrector_ceos.py` and `corrector_digital_twin.py`.
- The microscope orchestrates through `DeviceProxy` and does not own detector logic.
- The MCP server generates tools from Tango commands. Clients discover capabilities and do not hard-code them.
- Typed public interfaces, explicit contracts, simplicity over cleverness, minimal core dependencies with heavy ones in extras, Tango exceptions for errors, and no hidden global state.

**pyTEMlib**:

- A library of `*_tools` subpackages: `image_tools`, `eels_tools`, `eds_tools`, `diffraction_tools`, `crystal_tools` and `graph_tools`.
- Module files follow the subpackage's naming, for example `image_*.py`.
- Public functions take and return `sidpy.Dataset`. Calibration lives in `sidpy.Dimension`s. Experiment information lives in `dataset.metadata['experiment']`.
- numpy-style docstrings, exports in the subpackage's `__init__`/`__all__`, and tests in `tests/test_<subpackage>.py`.
- `pyTEMlib/mcp_server.py` is a thin, stateless MCP wrapper.

**SciAgentGUI** takes the same ideas:

- A base contract with concrete implementations.
- Capabilities discovered from the server, never hard-coded.
- Typed contracts at the IPC and MCP boundaries.
- Adding an instrument, detector, panel or model provider must not mean editing core files.

## Dependency rule (asyncroscopy)

```
core  <-  contracts / trace / data / safety / sim  <-  instruments / jobs  <-  mcp  <-  agents
```

Nothing points the other way. A test imports every package with only core dependencies installed, to enforce the rule.

## asyncroscopy target layout

```
asyncroscopy/
  settings.py                 one resolved Settings (outputs_root and every ASYNCROSCOPY_* variable), no import-time env reads
  registry.py                 one Registry(group, builtins) for map profiles, tuning strategies, drift trackers, specimens
  core/                       dependency-free plumbing
    fileio.py                 file_lock, SequencedAppender (with rotation)        <- mcp/tool_log.py
    net.py                    loopback/bind helpers                               <- mcp/binding.py
    names.py                  tool name <-> (device, command), headers, envelope verbs, job states, key patterns
  contracts/                  JSON schemas + validate()
  trace/                      writer.py (TraceWriter, ToolCallLog), clock, stream, tail
  safety/                     policy (declared risk), decide (one approval decision), approvals device, estop, limits
  sim/                        modality-free simulation primitives: clock, rng, stage_model, dose_map
  data/                       DATA device, Tiled client, record, integrity, dose ledger, views (image/line/stack/table)
  instruments/
    instrument.py             Instrument base
    electron_microscope/      ElectronMicroscope base, auto_script, jeol, twins, hardware/, detectors/, map_profile.py
    scanning_probe_microscope/  ... + map_profile.py
    nano_indenter/            NanoIndenter base + simulated subclass
    yaml_sim/
  jobs/
    job_device.py             the job envelope base (reserve/start/status/abort/results)
    recording_proxy.py
    mapping/                  mapper.py, plan.py, run_file.py, pipeline.py, profile.py (InstrumentProfile + MapContext), live_drift.py
    tuning/                   tuner.py + strategies
  analysis_server/            the analysis MCP server (map store, map tools, placements, notes, jobs), moved out of pyTEMlib;
                              calls pyTEMlib's sidpy-native image_tools
  mcp/
    mcp_server.py             upstream shape: discovery, command wrapping, an interceptor list, main
    descriptors.py            ToolDescriptor{device, command, action_role, risk, acts, operator_only, native}
    action_roles.py, action_space.py, endpoint_auth.py
    components/               gate, jobs, data_tools, discovery, skills_tools, trace_tools, estop_tool; each register(server, cfg)
    agent.py, agent_mcp_server.py, segment.py   (upstream entry points)
  agents/                     [agent] extra: graphs, middleware (opt-in), profiles/ (model-specific data), guidance
  skills/                     SKILL.md files + cached registry
```

### Work items: foundations and MCP layer

1. Create `core/`, `settings.py` and `trace/writer.py`. Move `file_lock`/`SequencedAppender`/`TraceWriter`/`binding` there and fix every inverted import.
2. Add `ToolDescriptor`, built once at discovery. `action_space`, the gate, auth, `find_tools`, `endpoint_info` and goals all read it. Devices declare `risk`, `operator_only` and stop commands. The word parser becomes an opt-in startup lint.
3. Add `safety.decide()`, the one approval decision used by the gate, `find_tools`, `endpoint_info` and auth.
4. Split `MCPServer` into components registered by config. `_create_wrapper` runs a list of interceptors (`before_call`/`after_call`/`on_error`) instead of one closure.
5. Tool replies return structured fields (`state`, `next_actions`, `resume_with`, `warnings`). An agent-side `guidance.py` turns them into prose for small models.
6. Model-specific data (tool subsets, word lists, prompt budgets) moves to `agents/profiles/*.yaml`. Middleware is a configured list. There is no default model in library code.
7. Approvals: atomic mode-file writes, and persisted requests and grants. Trace and ledger files rotate by size.
8. pyproject: declare `jsonschema` and `httpx`. Put the pyTEMlib pin in an `analysis` extra, `psutil` in `launcher`, and dev tools in a dev group.

### Work items: device layer

1. Add `registry.py` and replace the four copies of the registry/entry-point pattern.
2. Move `mapper.py`, `map_profile.py` and `live_drift.py` to `jobs/mapping/`, and split the Mapper into plan/run_file/pipeline. Profiles get a typed `MapContext` instead of reaching into Mapper private members. Each modality registers its own profile, and `jobs/` imports nothing from `instruments/`.
3. Move `tuner.py` to `jobs/tuning/`. Rename the strategy `optistem` to `vendor_autofocus`. `CORRECTOR` gets typed `measure`/`correct` commands, so the Tuner never parses CEOS JSON-RPC.
4. Put limits (`min_fov`, `max_fov`, `max_defocus`), beam current, stigmator, `stage_moving` and the command lock on the `ElectronMicroscope` base (since then, what a scanning-beam instrument shares moved to `ScanningBeamInstrument`, beside which `ScanningElectronMicroscope` is the SEM's contract). Vendors implement only `_x` methods. Remove the import-time monkey-patching in `auto_script.py`.
5. Twins: add a thin `ElectronMicroscopeTwin` base, a `hardware/stage_digital_twin.py` that owns the twin stage model, and samples as `Specimen` plug-ins in place of hard-coded twin classes. Shared simulation primitives move to `asyncroscopy/sim/`.
6. Add the missing bases: `NanoIndenter` (and fix the "worked example"), and `YamlInstrument(Instrument)` with a separate `YamlJob(JobDevice)`. The job drives the instrument through a step protocol (`hold`, `release_hold`, `start_step`, `step_status`, `stop_step`) that agents do not see, so steps of up to an hour stay stoppable within 0.2 s while no Tango call is long.
7. Scalability: cache one proxy per address, split job state into a plan written once and a small progress record, keep one open HDF5 writer per run, and append ground-truth records to a file instead of memory.
8. One `ELECTRON_CHARGE`. One beam-current default. Role addresses are injected by the launcher, not repeated in every config.

## pyTEMlib target layout

```
pyTEMlib/
  image_tools/
    image_mosaic.py           Tile.from_dataset, register_tiles, assemble, stitch_mosaic(list[Dataset]) -> Dataset
    image_registration.py     + register_to_parent(child: Dataset, parent: Dataset, ...)
    image_scoring.py          Scorer protocol, BaselineScorer, score_windows(dataset, ...), select_regions(...)
    image_atoms.py            + find_blobs(dataset, ...)
    image_utilities.py        + pixel_to_stage / stage_to_pixel(dataset)
    __init__.py               all of the above in __all__
  mcp_server.py               upstream's file, unchanged
tests/
  conftest.py                 synthetic tile/scene generators
  test_image_mosaic.py, test_image_registration.py, test_image_scoring.py
```

Everything that stores, schedules or serves goes to `asyncroscopy/analysis_server/`: `map_store`, `map_tools`, `jobs`, `frames`, `placement_tools`, `annotation_store`, `trace`, the envelope schema copy, and the twin-specific parts of scoring evaluation.

PRs to offer upstream, in order:

1. The `low_loss_tools` import-time fit fix.
2. The bug fixes from `feature/rust-equivalence` (`casnati_x_section`, `get_fwhm`, `get_resolution_functions`, `zl_func`, `atom_refine`), each with a unit test.
3. `image_mosaic` with `register_to_parent`.
4. `image_scoring`.

The Rust-equivalence test suite and its fixtures belong in SciAgentGUI.

## SciAgentGUI target architecture

The GUI stops guessing. asyncroscopy already serves `endpoint_info` (role, job devices, risks, auto, analysis URL) and `action_space` (for every tool: device, command, canonical action role, risk, effect, arguments with units). SciAgentGUI reads these once in Rust and builds `Capabilities` from them. Routing, risk, panel visibility and operator calls all derive from `Capabilities`.

```
src-tauri/src/
  core/        instrument.rs, registry.rs (InstrumentRegistry keyed by InstrumentId), capabilities.rs, endpoint.rs (Endpoint trait)
  mcp/         client.rs
  instruments/ asyncroscopy.rs (operator verbs by role), builtin.rs (workbench tools)
  gate/        policy.rs (ServerReportedRisk, KeywordGuess fallback), queue.rs, audit.rs
  models/      provider.rs (ModelProvider trait + caps), openai.rs, bridge.rs
  data/        tiled.rs (one bytes endpoint), media.rs
  analysis/    eels, eds, atoms, image, ...
  platform/    db, secrets, logging, updater, ...
src/
  core/        ipc.ts (one invoke, no mocks), contracts/ (generated types), instruments.ts (one store)
  agent/       ChatSession (per-chat backend), harnesses
  instruments/ operator.ts, panels registered by capability
  data/        one Tiled client
  views/       map/, spectrum/, image/
  panels/      eels/, eds/, ... (StemWorkbench split)
  pages/, shell/
```

What adding things costs in this layout:

- **New instrument or detector:** no GUI code.
- **New panel:** one file plus `registerPanel({requires: caps => ...})`.
- **New model provider:** one `impl ModelProvider`.

Work items, in order:

1. Delete the mocks and dead weight: the browser mock table, the TS MCP client, the vendored `pystemllmgui/`, mirrored configs and scripts, and hard-coded capability badges.
2. Fix rule-1 breaches, such as the hard-coded 200 kV quantification.
3. `Capabilities` from `endpoint_info`/`action_space` in Rust, replacing every tool-name guess in Rust and TS.
4. One defaults table owned by Rust.
5. `InstrumentRegistry` keyed by id, with one supervisor poll per instrument.
6. `ModelProvider` trait and a per-chat `AgentBackend`.
7. Split the large components and collapse to one Tiled client.
8. Rewrite `docs/architecture/overview.md` and cut CLAUDE.md down to rules, commands and layout.
