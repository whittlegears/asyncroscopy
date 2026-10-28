# Agentic Sample Mapping and Instrument-Agent Infrastructure for Asyncroscopy

*   **Author:** whittlegears
*   **Status:** Implemented (Stages 0-4 and parts of 5); see "Implementation notes" at the end
*   **Date:** 2026-10-27

--------------------------------------------------------------------------------

## Overview

This design adds sample mapping to asyncroscopy, along with the agent infrastructure around it. When it is done, an agent can be told in plain language to "map this sample", "map this area and take an EDS spectrum at each tile", or "find interesting regions and go back to them". The agent can do this unattended on the digital twin, and later on the Spectra 300.

The work spans four codebases, each with one job:

| Codebase | Job |
| :-- | :-- |
| asyncroscopy (`whittlegears/asyncroscopy`) | All instrument and agent infrastructure: devices, the map job, the approval service, the MCP server, LangGraph graphs, skills, traces |
| pyTEMlib (fork `whittlegears/pyTEMlib`, offered upstream to `pycroscopy/pyTEMlib`) | All analysis: tile registration, map assembly, scoring, plus the analysis code moved back from SciAgentGUI. Exposed as an MCP server. |
| Tiled | Storage for every acquired tile, map, point of interest and trace |
| SciAgentGUI (`whittlegears/sciagentgui`) | A thin client: views data from Tiled and calls asyncroscopy commands. It holds no agent loop and no safety rules. |

All of it is developed and checked on the seeded digital twin, using a local Qwen3 model through Ollama on a 6 GB GPU. The same runs produce evidence for the ATHENA WG2 Year-1 deliverables.

## The Problem

Today an agent connected to asyncroscopy can take one image or one spectrum at a time. It cannot carry out a multi-hour experiment that covers a whole sample. Five gaps stand in the way:

1.  **No long-running operations.** Every action is a single Tango command with a 30-second timeout (`asyncroscopy/mcp/mcp_server.py`, `COMMAND_TIMEOUT_MILLIS`). A map of hundreds of tiles cannot run as one call. Asking the model to issue one call per move and acquisition would mean more than 1,000 calls. A 4B-parameter local model loses track long before that.
2.  **No enforced boundary on what the agent may do.** The server-side agent path runs tools with no approval step, and the MCP server has no authentication. SciAgentGUI's approval queue only sees calls made by its own agent loop, and it judges risk from words in the tool name. "Approve once, then stay in the region" therefore has nowhere to live.
3.  **No map as data.** Two diverged copies of a stitcher exist (`STEM-ML-Functions/stem_stitcher` and `sciagentgui/pystemllmgui/pycrosGUI/stem_stitcher`). Both assume atomic-resolution tiles, keep the pyramid in memory and save loose PNGs. Neither records stage coordinates or points of interest. Without those, nothing found on the map can be driven back to.
4.  **The twin cannot test mapping.**
    *   The twin reports the noisy stage position as the real one (`digital_twin.py`, `_move_stage`), so stitching always looks perfect.
    *   Its stage noise is unseeded, so runs cannot be repeated.
    *   It does not export ground truth.
    *   Its default sample (6 nm) fits inside one 20 nm field of view.
5.  **The agent layer cannot be inspected or extended.**
    *   The LangGraph supervisor is built at runtime inside `asyncroscopy/mcp/agent.py`, so LangGraph Studio cannot load it.
    *   Skills exist only inside SciAgentGUI.
    *   SciAgentGUI's integration notes describe a different fork ("Leviasyncroscopy") from the one this work builds on.

The proposed solution is a set of composable mapping building blocks rather than one fixed workflow:
*   **A map job** in asyncroscopy. It runs in the background, with start, status, abort and results, and the device itself enforces the region.
*   **Analysis tools** in a pyTEMlib MCP server, which read tiles from Tiled by key.
*   **A map stored in Tiled** with stage coordinates and a points-of-interest layer.
*   **Map-query and scoring tools** so agents and ML models can explore the map.
*   **A skill** that teaches the agent how to combine these blocks.
*   **LangGraph graphs** structured so LangGraph Studio can run and debug them.

## Goals

Each goal states how success is checked. "Twin" means the asyncroscopy digital twin with a fixed seed.

*   **G1. Working development environment on Windows.**
    *   Python 3.12, uv and Rust are installed.
    *   `uv sync --extra agent --extra ollama` succeeds.
    *   The existing asyncroscopy test suite passes.
    *   A local Tiled server runs.
    *   A Windows quickstart document lets a second person reach the same state.
*   **G2. The twin presents the problems a real STEM presents.**
    *   Stage error, backlash, drift, scan distortion, focus, noise, dose damage and EDS spectra are all simulated, with magnitudes taken from the literature. Each is to be calibrated later against Spectra 300 measurements.
    *   Every effect is seeded, so the same seed and commands give the same run.
    *   A simulated clock gives runs a realistic duration, and can be set to zero for tests.
    *   Ground truth (particles, true stage pose, drift) can be exported through a path agents cannot reach.
    *   The mapping config uses a 2 µm sample with 400 particles.
    *   Acceptance:
        *   A survey tile (512², pixel size > 2 nm) renders in under 1 s on this laptop, and an atomic-resolution tile in under 2 s.
        *   Changing defocus by 20 nm measurably lowers image sharpness.
        *   Tiles of the same region taken at different times differ only by noise, drift and dose effects, not by brightness scaling.
        *   A twin EDS spectrum fits in eXspy with the correct elements.
        *   All existing twin configs still load and pass their tests.
*   **G3. A map job that runs in the background and stays inside its region.**
    *   The job uses the common envelope: reserve, start, status, abort, results, release.
    *   It runs a frozen plan: stage bounding box, tile size and overlap, tile count, dose budget, and per-tile actions (HAADF, optional EDS).
    *   Acceptance:
        *   Tile centres match the planned grid.
        *   No tile centre falls outside the box.
        *   Abort stops within one tile, with the beam blanked.
        *   Resume skips finished tiles.
        *   A stage move outside the region is refused by the device and logged with the approval id.
*   **G4. One approval mode, owned by asyncroscopy.**
    *   asyncroscopy has a runtime setting with two values:
        *   `auto` (the default): every call runs, and the trace records that no person approved it.
        *   `approve`: high-risk agent calls are held until a person allows or denies them.
    *   The GUI switches the mode, and each switch is recorded in the trace.
    *   Notebooks and scripts behave the same in both modes.
    *   In `approve` mode, a map run needs one approval covering its plan.
    *   Regardless of mode, a map job never moves outside its own planned box. Hitting the box or the dose budget stops the job with the beam blanked.
    *   SciAgentGUI shows the queue and sends the operator's decision back. It makes no decisions itself.
*   **G5. A stitched, zoomable map in Tiled.**
    *   Each tile is stored in Tiled with its reported stage position, field of view and acquisition settings.
    *   The pyTEMlib MCP server reads the tiles by key. It places them by stage position, refines placement by image matching, and falls back to stage-only placement when matching confidence is low. It then writes a multi-resolution map plus a points-of-interest layer back to Tiled.
    *   Acceptance on the twin:
        *   With stage noise at 5% of the field of view, the median placement error is under 1 px against ground truth.
        *   With no noise, refinement is no worse than stage-only placement.
        *   Peak memory stays under 4 GB for a 2 GB map.
        *   Pixel-to-stage conversion round-trips within 1 px.
        *   The map opens in napari.
*   **G6. Agents and ML models can explore the map.**
    *   Map tools:
        *   list tiles in a box
        *   get a region at a zoom level
        *   list, add and go to points of interest
    *   A scorer interface takes a map region and its metadata. It returns a score from 0 to 1, a reason and the scorer's version.
    *   One baseline scorer (contrast and sharpness) ships, so the explore-score-revisit loop runs end to end.
    *   Trained models, rules and LLM judges can be added later without changing the interface.
*   **G7. A local agent completes mapping tasks.**
    *   A short model test picks the Qwen3 tag, context size and tool allow-list. To pass, it must produce at least 90% valid tool-call JSON on a fixed prompt set on this GPU.
    *   On the task "map this region and report three particles", the agent completes at least M of N seeded twin trials. Each trial must take 10 or fewer tool calls and make zero moves outside the region.
    *   Tokens and wall time are recorded for every trial.
*   **G8. Every run leaves a trace by default.**
    *   Each run writes one event per action. The event holds the run id, plan id, tile index, caller, approval id, settings, and the result's Tiled key.
    *   The trace is stored in Tiled and is enough to rebuild the run.
*   **G9. LangGraph graphs that LangGraph Studio can run.**
    *   The mapping workflow and the general agent are each written as an importable graph module, listed in `langgraph.json`.
    *   `langgraph dev` loads them, and the mapping graph runs against the twin from Studio.
    *   Graph tool calls still pass through the G4 approval service.
*   **G10. Skills served by asyncroscopy.**
    *   Skills are stored in asyncroscopy and offered to any agent, including the GUI, through MCP tools that list and load them.
    *   A skill is loaded only when needed, so its text is never pasted into every prompt.
    *   A mapping skill teaches how to combine the G3–G6 blocks for "map a sample", "map an area" and "map with EDS at each tile".
    *   Skills are written by people. Skills written by agents are a future improvement.
*   **G11. SciAgentGUI as a thin client.**
    *   The GUI shows the map from Tiled with pan, zoom and the points-of-interest layer.
    *   It shows run status and the approval queue, and calls asyncroscopy commands.
    *   Its integration notes describe this repository, not Leviasyncroscopy.
    *   Its built-in agent loop and its Rust-side safety rules are retired.
*   **G12. Analysis moved back into pyTEMlib.**
    *   The analysis that SciAgentGUI ported to Rust (EDS, EELS, FFT, image, atom finding, peak fitting, probe, optimisation) is returned to the pyTEMlib fork.
    *   For each returned function, a test runs the Rust and Python versions on pyTEMlib's example data and checks that the outputs agree.
    *   The MCP server sits in its own subpackage. The fork is kept rebased on upstream `main`, so the pull requests are ready to offer to `pycroscopy/pyTEMlib`.
*   **G13. WG2 evidence.**
    *   A traceability table links each ATHENA WG2 deliverable to the artifact that supports it:
        *   envelope → typed action schema
        *   trace → Instrument Trace
        *   baseline scorer and per-tile trace → STEM reward workflow
        *   capability entry → registry v0.3
    *   The table states that the evidence comes from the twin, not the instrument.

## Non-Goals

*   **Autonomous runs on the real Spectra 300.** This design gets workflows ready for the instrument. Running them there needs the facility owner's readiness approval, which is out of scope.
*   **Trained ML scorers, a fine-tuned LLM judge, or a library of scoring rules.** Only the interface and one baseline scorer are built.
*   **Building authentication, authorisation tokens or the capability registry.** Those belong to WG1 and NICS. This work leaves a hook where identity will be checked, and adds one registry entry.
*   **Action schemas for nanoindentation or SPM.** The envelope is written so they can reuse it, but they stay on WG2's own track.
*   **A live, streaming Instrument Trace pipeline.** The per-run trace stored in Tiled is the deliverable.
*   **Map formats other than the one chosen for Tiled.** That rules out DZI and pyramidal TIFF exports.
*   **Large benchmarks or large models on this machine.** That covers repeating the paper's 189-trial benchmark and running `gemma4:31b`. Trials stay at about 10–20 per task.
*   **Merging into upstream pyTEMlib.** The pull requests are made ready. Whether and when they are accepted is the maintainer's decision.
*   **Repairing the two existing stitcher copies.** They are replaced, not fixed.

## Open Questions

*   **Tiled and multi-resolution maps**: Can Tiled serve a multi-resolution OME-Zarr map as one item? If not, the fallback is one array per zoom level plus an index. In that case viewers must understand our layout.
*   **Who approves when no one is watching**: The plan is to pause with the beam blanked. Open questions remain: how long a paused run may wait, whether it times out to an abort, and who is notified.
*   **What "EDS at each tile" means by default**: The default proposed here is one point spectrum at each tile centre. A full spectrum image multiplies time and dose and should require an explicit request.
*   **Trial thresholds**: Values of M and N for G7.
*   **Where the explore loop is orchestrated**: The loop spans pyTEMlib (score), asyncroscopy (move, acquire) and Tiled (pixels). It could be driven by the agent turn by turn or by a LangGraph graph in asyncroscopy. Either way, only keys, scores and coordinates should pass through the model.
*   **Leviasyncroscopy features**: Decide which, if any, of `Complete`, `mcp/backends/` and the skill store are ported, and which are dropped.
*   **Relation to `whittlegears/sciagent-skills`**: Decide whether that public registry becomes the published source for asyncroscopy's skills, or is retired.

## Critical User Journeys (CUJs)

### CUJ 1: Map a sample from one sentence

*   **Given**: The twin, Tiled, the asyncroscopy MCP server and the pyTEMlib MCP server are running from one config. The agent runs on local Qwen3.
*   **When**: The operator types "map the sample".
*   **Then**:
    *   The agent loads the mapping skill and proposes a plan (box, tile count, time and dose estimate).
    *   The operator approves the plan once. The run then completes unattended.
    *   A map appears in Tiled and can be panned and zoomed in the GUI.
    *   The trace records every tile.

### CUJ 2: Map an area with EDS at each tile

*   **Given**: The same setup as CUJ 1.
*   **When**: The operator types "map a 500 nm square around the centre and take an EDS spectrum with each image".
*   **Then**:
    *   The plan shows one point spectrum per tile, with a time estimate.
    *   After approval, each spectrum is stored in Tiled and appears on the map as a point of interest at its stage position.
    *   Clicking a point in the GUI shows its spectrum.

### CUJ 3: Find interesting areas and go back

*   **Given**: A finished map in Tiled, and a scorer available on the pyTEMlib server.
*   **When**: The operator asks "find the three most interesting regions and image them at high magnification".
*   **Then**:
    *   The scorer ranks map regions and adds the top three as points of interest, each with a reason and the scorer's version.
    *   The agent drives the stage to each point and acquires a high-magnification image, inside the approved region.
    *   The new images are linked to their points of interest.

### CUJ 4: Switching to approval mode for a riskier session

*   **Given**: asyncroscopy is in the default `auto` mode, and a notebook user and an agent are both connected.
*   **When**: The operator switches the mode to `approve` in the GUI, and the agent then asks to move the stage outside its approved map box.
*   **Then**:
    *   The MCP server does not run the move. It returns an approval request to the agent, and the GUI shows that request.
    *   Approval runs the move, and denial returns a refusal to the agent. Both outcomes and the mode change are in the trace.
    *   The notebook user's own `move_stage` calls are unaffected.

### CUJ 5: Debug the mapping graph in LangGraph Studio

*   **Given**: A developer has the twin stack running.
*   **When**: The developer runs `langgraph dev` and opens the mapping graph in Studio.
*   **Then**:
    *   The developer can step through nodes, inspect state, edit a prompt and rerun against the twin.
    *   Tool calls made from Studio still pass through the approval service.

### CUJ 6: Add a new scorer

*   **Given**: A team member has a trained model that flags grain boundaries.
*   **When**: The team member wraps it in the scorer interface and registers it on the pyTEMlib server.
*   **Then**:
    *   CUJ 3 can use it by name with no change to asyncroscopy or the GUI.
    *   Its scores are logged with its version, so it can be compared with the baseline.

## Technical Plan (Architecture)

### Components and data flow

```text
                      +-------------------------------------------------+
  Operator  --------> |  SciAgentGUI (thin client)                      |
                      |  chat view | map view | run status | approvals   |
                      +------+-------------------+----------------------+
                             | MCP (HTTP)        | Tiled HTTP (read pixels, POIs)
                             v                   v
  +------------------------------------------+   +-----------------------------+
  | asyncroscopy                              |   | Tiled                       |
  |                                           |   |  maps/<run_id>/tiles/...    |
  |  Agent server (LangGraph graphs, skills)  |   |  maps/<run_id>/spectra/...  |
  |        | MCP client                       |   |  maps/<run_id>/mosaic/...   |
  |        v                                  |   |  maps/<run_id>/poi          |
  |  MCP server + policy hook  ----------+    |   |  maps/<run_id>/trace        |
  |        | Tango                       |    |   +-------------^---------------+
  |        v                             v    |                 |
  |  Mapper (job envelope) ----> Approvals    |                 | read tiles by key,
  |        |                     (grants)     |                 | write mosaic + POIs
  |        v                         ^        |                 |
  |  ElectronMicroscope / DigitalTwin         |   +-------------+---------------+
  |    move_stage, acquire_*                  |   | pyTEMlib MCP server (fork)  |
  |        |                                  |   |  register, assemble,        |
  |        v                                  |   |  score, map queries,        |
  |  DATA device ---- register ---------------+-->|  returned analysis tools     |
  +------------------------------------------+   +-----------------------------+
```

There is one rule for what crosses the model's context. Only Tiled keys, stage coordinates, scores, short reasons and job ids pass through the agent. Pixels go from Tiled to pyTEMlib and from Tiled to the GUI directly.

### 1. Digital twin as a test bed (G2)

The twin exists to rehearse and test agents and analysis. It does not have to reproduce exact specimen physics. It does have to make agents and the stitcher face the same problems the Spectra 300 gives them, at a cost of about a second per tile on this laptop.

#### What is wrong with the twin today

Line numbers refer to `digital_twin.py`.

*   **Positions and randomness.**
    *   The reported stage position equals the true one (L663).
    *   Stage noise is unseeded (L652), and so are several draws in `digital_twin_particles.py`.
    *   Shot noise is seeded only by pose (L511-512), so two acquisitions at the same position are identical. That breaks frame averaging and repeat tests.
*   **Focus and aberrations do nothing.**
    *   `set_defocus` only stores a number (L630-636).
    *   The probe always uses fixed Spectra 300 target aberrations (L497-503) and ignores the corrector twin.
    *   An autofocus or tuning agent therefore gets no signal.
*   **The image is not physical.**
    *   The projected potential is scaled to its own maximum in each tile (L262-264), so brightness depends on tile content and stitched maps show seams.
    *   Contrast is linear in Z instead of about Z^1.7, with no thickness term.
    *   Every requested detector returns the same image (L529-531).
    *   `scan_region` is accepted but ignored.
    *   `get_fov` fails, because `_get_fov` is not implemented (`electron_microscope.py` L457).
*   **No time passes.** Dwell only scales counts. There is no frame time, stage-move time, settling, drift or dose history.
*   **EDS is not a spectrum.** It returns composition fractions (L553-604). The saved array has no energy axis, and off-particle positions return arbitrary values.
*   **Rendering cannot scale to a survey sample.** It loops over atoms in Python and copies all atoms through ASE on every pose change. A 2 µm sample with hundreds of nanoparticles is about 10^7 atoms, an estimated minutes per tile.
*   **Unconfirmed: a possible x/y transpose.** The potential is indexed `[x, y]`. If downstream code reads axis 0 as rows, images are transposed relative to stage and beam coordinates, which would corrupt both stitching and EDS targeting. A test settles this in Stage 1.
*   **Stale documentation.** `docs/digital_twin.md` describes `sample_extent_scale` and `regenerate_sample`, which do not exist.

`digital_twin_tilt.py` already has some of the needed patterns:
*   a per-event seeded generator (`_rng(event, index)`)
*   separate commanded and actual stage positions
*   focus drift with an autofocus residual

The new twin reuses these rather than writing them again. `polycrystalline_gold_digital_twin.py` already maps corrector-twin coefficients to pyTEMlib aberration names.

#### The twin model

**State and randomness.**
*   Every random draw comes from `_rng(event, index)`, seeded from `sample_seed`, the event name and an event counter.
*   The same seed and the same sequence of commands therefore give the same run.
*   Repeated acquisitions still get fresh noise, because the counter advances.

**Clock.**
*   The twin keeps a simulated clock. It advances on:
    *   every stage move: overhead plus distance divided by speed
    *   every frame: pixels × dwell + lines × flyback
    *   every spectrum: live time plus dead time
*   `time_scale` sets how much real time passes: 1 means real waiting, and 0 means none, so tests run fast.
*   Drift, settling and dose history all depend on the clock.

**Stage.** There are three positions per axis:
*   **commanded**
*   **true:** commanded, plus a fixed bias, a Gaussian error, and a backlash shortfall whenever an axis reverses direction
*   **reported:** commanded plus a small reading error

Two more effects:
*   For a short time after each move, settling drift decays exponentially.
*   Images are always rendered from the true position, plus the drift offset at the moment each line is scanned.

**Sample.**
*   The sample is a seeded table of particles, not atoms. Each particle has:
    *   centre, semi-axes and orientation
    *   element
    *   a ground-truth id
*   A support film of chosen thickness sits under them, and an optional height plane makes focus change across the map.
*   A spatial index finds the particles inside a field of view.
*   A 2 µm sample with 400 particles takes a few kilobytes.

**Rendering by level of detail.** The level is chosen from the pixel size:

| Level | Pixel size | Image source | Expected cost (512²) |
| :-- | :-- | :-- | :-- |
| L0 survey | ≥ 0.2 nm | projected thickness of each particle (ellipsoid) × Z^1.7, plus support-film background | ~10 ms (estimate) |
| L2 atomic | < 0.2 nm | atoms built with ASE only for particles in view; vectorised projection | under 1 s for 2–3 particles (estimate) |

Both levels use plain vectorised NumPy first. Compiled rasterising (numba) and caching are added only if the Stage 1 render test misses its time target.

**Optics.**
*   The probe comes from pyTEMlib's `get_probe`, using the current defocus (C1), astigmatism and coma from the corrector twin. It is cached for each aberration state.
*   At survey levels, a Gaussian blur stands in for the probe, widening with |defocus| and the convergence angle.
*   Autofocus and tuning agents therefore see real changes in sharpness.

**Detector.**
*   Signal is in absolute counts: beam current × dwell × collection fraction × the sample term.
*   On top of that come Poisson noise, gain, offset and read noise.
*   Output is raw counts in `uint16`, with no per-tile normalisation, so tiles agree in brightness.
*   HAADF, BF and other detectors each get their own contrast rule; BF is the inverse of mass-thickness.

**Scan distortions.**
*   Per-line jitter.
*   Compression at the left edge from flyback.
*   Scan rotation and shear.
*   Field-of-view scale error.
*   Drift applied line by line during the frame.
*   All are applied as one coordinate remap.

**EDS.**
*   A point spectrum has an energy axis of 2,048 channels from 0 to 20 keV, made from:
    *   a Kramers continuum
    *   characteristic lines for the elements under the beam, weighted by projected mass-thickness, with detector resolution of about 130 eV at Mn Kα and widening with energy
    *   Si escape peaks and sum peaks
    *   dead time
    *   Poisson noise
*   Real time and live time are both reported.
*   The beam position used is the true one, including drift during the dwell.
*   The X-ray line energies come from the line tables that pyTEMlib already ships, so no GPL library is pulled in.
*   A spectrum image is built per pixel from element maps on a coarse grid.

**Dose and damage.**
*   A dose map in stage coordinates records where the beam has been.
*   A carbon contamination layer builds up with dose and saturates.
*   Particles beyond a critical dose dim.
*   Re-imaging the same spot therefore has a visible cost, which dose-aware agents and the STEM reward work need.

**Ground truth.**
*   `export_ground_truth` returns the particle table, plus the true stage pose and drift offset of every acquisition.
*   The MCP configs block it, so agents never see it. Test scripts call it through Tango directly.

**Fixes.**
*   `_get_fov` is implemented.
*   `scan_region` is honoured.
*   The documentation is corrected.

#### Calibration and defaults

`configs/digital_twin_mapping.yaml` turns all of this on with the defaults below. The existing twin configs keep every new effect at zero, so their behaviour stays the same apart from the bug fixes.

| Group | Parameter | Default | Source |
| :-- | :-- | :-- | :-- |
| Sample | `sample_size_xy_m`, `sample_particle_count` | 2e-6, 400 | chosen |
| Sample | `particle_radius_mean_m` / `_std_m` | 8e-9 / 4e-9 | chosen |
| Sample | `haadf_z_exponent` | 1.7 | literature: 1.6–2 |
| Stage | `stage_bias_m`, `stage_noise_std_m` | [100, −150] nm, 150 nm | Fiedler 2023: about ±350 nm typical on a JEOL GrandARM |
| Stage | `stage_backlash_m` | [150, 250] nm | **unverified**, to calibrate |
| Stage | `stage_reported_noise_std_m` | 10 nm | **unverified** |
| Stage | `stage_move_overhead_s`, `stage_speed_m_s` | 1.0, 5e-6 | **unverified** |
| Stage | `settle_drift_m_s`, `settle_tau_s` | 2 nm/s, 30 s | **unverified** |
| Drift | `drift_rate_m_per_min` | 0.5 nm | Thermo Fisher spec ≤0.5 nm/min |
| Optics | voltage, convergence angle, beam current | 300 kV, 30 mrad, 50 pA | Spectra 300 datasheet; current chosen |
| Detector | `haadf_collection_fraction`, `detector_read_noise` | 0.02, 0.5 counts | **unverified** |
| Scan | `flyback_s`, `line_jitter_std_px`, `scan_shear`, `fov_scale_error` | 500 µs, 0.3, 0.005, 0.01 | literature magnitudes; FOV error **unverified** |
| EDS | `eds_fwhm_mnka_ev`, `eds_solid_angle_sr`, `eds_dead_time_tau_s` | 130 eV, 0.7 sr, 1 µs | detector specs; solid angle **unverified** |
| Damage | `contamination_saturation`, `damage_critical_dose_e_per_a2` | 0.3, 1e5 | arbitrary until calibrated |
| Time | `time_scale` | 1.0 (0 in tests) | — |

Values marked unverified are placeholders. Each will be replaced by a measurement from the Spectra 300:
*   a repeated-move test, for stage error and backlash
*   a drift series on a fixed feature
*   a dose series on a sacrificial area

The calibration script and its results are part of Stage 1.

#### Reuse and licences

*   pyTEMlib (MIT) supplies the probe and the X-ray line tables.
*   ASE is used at L2 only.
*   abTEM and Prismatic (both GPL-3.0) are not runtime dependencies. They may be used offline to check the Z^1.7 and thickness approximation against multislice.
*   eXspy/HyperSpy (GPL-3.0) is used only in tests, to fit twin spectra and confirm they behave like real ones.
*   pycroscopy's DTMicroscope, STEMGym (with its dose-budget scoring) and instamatic's stage handling are references for design ideas, not dependencies.

### 2. The job envelope and the Mapper device (G3)

One generic contract governs any long-running instrument job. It is written once as a JSON Schema (`contracts/envelope.schema.json`) so pyTEMlib and, later, nanoindentation and SPM can follow it without importing asyncroscopy. It has six verbs:

| Verb | Input | Output |
| :-- | :-- | :-- |
| `reserve` | plan JSON | reservation id, plan id, resolved plan (tile list, time and dose estimate), required approval scope |
| `start` | reservation id, grant id | job id |
| `status` | job id | state (`queued`, `running`, `paused_awaiting_approval`, `aborting`, `done`, `failed`, `aborted`), progress, last error, pending approval id |
| `abort` | job id | final state |
| `results` | job id | Tiled keys of the run container, finished tile count |
| `release` | reservation id | confirmation; frees the instrument for other callers |

A base Tango device class, `JobDevice`, implements the envelope:
*   It keeps job state in memory and also in a small per-run state file, so `status` survives a device restart and `resume` can skip finished work.
*   Work runs in a worker thread, so every Tango command returns in well under the 30-second timeout.

`Mapper` is the first `JobDevice`. Its plan holds:
*   the region (a stage box)
*   tile field of view and overlap
*   detectors
*   per-tile actions: `haadf`, plus optionally `eds_point` (one spectrum at the tile centre) or `eds_map` (a full spectrum image, which must be requested by name)
*   dwell and resolution
*   a dose budget
*   the order of travel (serpentine, to limit backlash)

For each tile, the worker does five things:
1.  Checks that the job is not aborted and that the grant still covers the move.
2.  Moves the stage.
3.  Acquires the requested data through the microscope device.
4.  Has the DATA device register the result in Tiled under the run container, with the commanded and reported stage position, field of view, pixel size and settings attached.
5.  Writes one trace event.

On abort, on a tile outside the plan box, or when the dose budget runs out, the worker blanks the beam before it stops.

### 3. Approval mode (G4)

This moves SciAgentGUI's existing auto/approval setting into asyncroscopy, so it covers every agent and not only the GUI's own loop.

`Approvals` is a Tango device. It holds:
*   **The mode.** Either `auto` or `approve`. The default comes from the YAML (`safety.default_mode`, defaulting to `auto`). `set_mode` changes it at runtime, and the GUI calls `set_mode`. Every change is written to the trace with who made it and when.
*   **A risk map.** Each `Class.command` has a risk level from the config. Anything that moves hardware, changes the beam, or starts a job is high. Unknown commands default to high. In `auto` mode that changes nothing, because everything runs. In `approve` mode it means a new, unclassified tool is held rather than waved through.
*   **Grants.** A grant has these fields:
    *   id
    *   scope: an allowed stage box with margin, tile and dose budget, allowed command classes, and expiry
    *   who granted it
    *   the run it belongs to
*   **The queue of pending requests.** The GUI lists them, and approves or denies them.

The mode is applied at the MCP server, which is the only way agents reach the instrument. A policy hook wraps every generated tool:
*   **In `auto` mode,** the call runs and its trace event records `approved_by: "auto"`.
*   **In `approve` mode,** a high-risk call without a covering grant does not run. The hook files a request and returns `{"status": "approval_required", "request_id": ...}` to the agent. The agent then waits on the request, or the mapping graph interrupts.

A caller-identity field is read from the MCP request and stored with each event. Checking that identity is a stub until WG1 provides tokens.

Notebooks and scripts call Tango directly and are treated as the operator. The mode does not affect them, so existing notebooks run unchanged.

One limit applies in both modes, because it is part of the plan rather than an approval: the `Mapper` worker refuses any tile outside its resolved plan box, and stops with the beam blanked when the dose budget is used up. That check lives in the `Mapper` device, not in the microscope device, so direct `move_stage` calls are not affected.

In `approve` mode, a run waiting on an unanswered request stays paused with the beam blanked. How long it may wait is an open question.

### 4. Traces (G8)

The existing `ToolCallLog` (`asyncroscopy/mcp/tool_log.py`) grows into a trace writer. Each event holds:
*   time
*   run id, plan id and tile index
*   caller and grant id
*   the command and its arguments
*   duration and outcome
*   the result's Tiled key

Events are appended to a local JSONL file while the run is going, which keeps working if Tiled is briefly down. The file is registered into Tiled as `maps/<run_id>/trace` every N tiles and at the end of the run. Tracing is on by default in the new configs.

### 5. The map in Tiled (G5)

Each run gets one Tiled container:

| Key | Contents |
| :-- | :-- |
| `maps/<run_id>/plan` | the resolved plan and grant id (metadata) |
| `maps/<run_id>/tiles/<i>` | one array per tile. Metadata: commanded, reported stage xyz; FOV; pixel size; detector; time. |
| `maps/<run_id>/spectra/<i>` | one spectrum per tile when `eds_point` is on, with its stage position |
| `maps/<run_id>/mosaic` | the stitched map at several resolutions, plus a pixel-to-stage transform |
| `maps/<run_id>/poi` | table of points of interest: id, stage x/y (metres), kind, source (agent, scorer name, operator), score, scorer version, reason, linked Tiled keys |
| `maps/<run_id>/trace` | the trace table |

Stage metres are the one master coordinate frame. Pixel coordinates are always derived from them through the stored transform, never stored on their own.

The mosaic is stored as multi-resolution OME-Zarr if Tiled can serve it as one item. If not, it is stored as one array per level under `mosaic/level_<n>` plus an index. This is settled in Stage 3 by a spike against the installed Tiled version.

### 6. pyTEMlib MCP server (G5, G6, G12)

This is a new subpackage, `pyTEMlib/mcp_server/`, in the `whittlegears/pyTEMlib` fork. It is a FastMCP server that reads from and writes to Tiled with the Tiled client. It never imports asyncroscopy.

Registration and assembly:
*   Initial placement comes from each tile's reported stage position.
*   Overlapping neighbour pairs are matched by phase correlation.
*   A global least-squares solve then combines the pairwise offsets with the stage positions as a weighted prior.
*   Pairs with low correlation confidence are dropped, so a tile with no good matches keeps its stage position.
*   Assembly writes the full-resolution canvas in chunks, then builds each lower level by 2× downsampling, so memory stays bounded.
*   The code sits next to pyTEMlib's existing registration in `image_tools`.

Assembly can take minutes, so it is a job under the same envelope (`reserve` / `start` / `status` / `abort` / `results` / `release`). The verbs match the shared JSON Schema.

Map tools:

| Tool | Job |
| :-- | :-- |
| `list_tiles(run_key, box)` | tiles that overlap a stage box |
| `get_region(map_key, box, level)` | a downsampled region returned as a small preview. Full arrays stay in Tiled and are passed by key. |
| `list_pois(map_key, filter)` | points of interest with scores and reasons |
| `add_poi(map_key, x, y, kind, reason, links)` | adds a point; refuses coordinates outside the map |
| `score_regions(map_key, scorer, level, top_k)` | runs a scorer over the map's regions and adds the top results as points of interest |
| `list_scorers()` | the names, versions and descriptions of installed scorers |

The scorer interface is `score(region_array, region_metadata) -> (value in [0, 1], reason, scorer_version)`. Scorers register through a Python entry point, so a new one installs as a package and needs no server change. The baseline scorer combines local contrast and sharpness.

The analysis returned from SciAgentGUI (G12) lives in pyTEMlib's existing subpackages, not in `mcp_server/`. It is exposed as MCP tools through thin wrappers in `mcp_server/`.

### 7. Agent layer: graphs and skills (G7, G9, G10)

A new package, `asyncroscopy/agents/`, holds code-defined LangGraph graphs:
*   **`mapping`**: an explicit graph. Its nodes, in order: plan → request approval → wait (a LangGraph interrupt, so Studio and the GUI both show it) → start job → poll status → assemble → optionally score → report.
*   **`general`**: a ReAct agent with skill loading.

Both graphs connect to the asyncroscopy and pyTEMlib MCP servers with `langchain-mcp-adapters`. A `langgraph.json` at the repository root lists them for `langgraph dev`.

The interrupt only shows the approval request. The decision is still enforced by the Approvals device, so a graph run from Studio cannot skip it.

The existing `AgentSwarm` and `agent_mcp_server.py` stay as the serving layer. Their `query_agent` runs a named graph, so the GUI's chat path keeps working.

Model settings come from config:
*   Ollama model tag
*   `num_ctx`, which is always set because the Ollama default silently cuts tool descriptions
*   the tool allow-list per graph
*   the step limit

Skills are stored as `asyncroscopy/skills/<name>/SKILL.md`. The front matter holds:
*   name and description
*   version
*   the minimum asyncroscopy version and envelope schema version
*   the tools the skill needs

The asyncroscopy MCP server adds two native tools, `list_skills` and `load_skill(name)`. An agent sees only names and descriptions until it loads a skill, which keeps small-model prompts short. Skills are added by people through the repository. Agents cannot write skills in this design.

### 8. SciAgentGUI as a thin client (G11)

The GUI keeps:
*   chat, which calls the agent server's `query_agent` and streams its transcript
*   the Instruments page (MCP connections)
*   the data views

It adds:
*   **A map view.** It fetches tiles from Tiled's HTTP array endpoints at the level that matches the zoom. It draws the points-of-interest layer from `maps/<run_id>/poi` and links each point to its spectrum or image.
*   **A run-status panel.** It polls `status`.
*   **An approvals panel.** It lists pending requests from the Approvals device and sends approve or deny decisions.

It drops:
*   the local agent loop
*   the risk classifier based on tool names
*   the Rust safety queue
*   the in-process analysis, which moves to the pyTEMlib server

The integration skill and docs are rewritten against this repository.

### 9. Configuration and compatibility

New YAML sections are all optional, and their defaults keep current behaviour:

| Section | Holds |
| :-- | :-- |
| `safety` | default mode (`auto`), risk map |
| `mapping` | Mapper device on or off, defaults |
| `trace` | path, Tiled flush interval |
| `twin.errors` | the error model |
| `agent.model` | tag, `num_ctx` |

A test loads every YAML file in `configs/` to prove nothing breaks. Agent configs move from `gemma4:31b` to the Qwen3 tag chosen in Stage 4, and the 31B configs are kept for the 5090 workstation. The duplicate config `DigitalTwin copy.yaml` is removed.

### 10. Stage order

| Stage | Delivers | Goals |
| :-- | :-- | :-- |
| 0 | environment, local Tiled, existing tests passing, Windows quickstart | G1 |
| 1 | twin error model, seeding, ground truth, mapping config | G2 |
| 2 | envelope schema, `JobDevice`, `Mapper` (with plan-box and dose limits), `Approvals` (mode, grants, queue), MCP policy hook, trace writer | G3, G4, G8 |
| 3 | pyTEMlib fork, MCP server, registration and assembly, map in Tiled, map tools, baseline scorer | G5, G6 |
| 4 | model test, mapping graph and skill, agent trials | G7, G10 (mapping skill) |
| 5 | Studio packaging, general graph, skills tools, GUI thin client, pyTEMlib analysis returned, WG2 table | G9–G13 |

Stage 5 items may start once Stage 2 runs. They must not change the Stage 2 contracts without updating the schema version.

## Alternatives Considered

### Alternative A: One fixed mapping script

*   **What it is**: A single deterministic `map_sample()` routine with parameters, called by a notebook or an agent.
*   **Pros**: The fastest route to a first map, and easy to test.
*   **Cons**:
    *   Every variation (map an area, add EDS, revisit points, map at two magnifications) needs a new script or a new parameter.
    *   The agent can only pick parameters, never combine steps.
*   **Reason for Rejection**:
    *   The goal is a base that other experiments reuse.
    *   Building blocks plus a skill keep the deterministic part (the tile loop) in code while leaving the combining to the agent.
    *   The `mapping` graph gives a fixed path when one is wanted.

### Alternative B: The agent drives every tile itself

*   **What it is**: The agent calls `move_stage`, `acquire_scanned_image` and `acquire_spectrum` once per tile.
*   **Pros**:
    *   No new device is needed.
    *   It uses the most flexible form of agent control.
*   **Cons**:
    *   A 400-tile map with EDS is over 1,200 calls.
    *   On `qwen3:4b` with a few thousand tokens of context, the run loses track or runs out of context long before it finishes.
    *   Each call also costs a model turn, which is seconds on this GPU.
*   **Reason for Rejection**: It is not workable on the local model, and it is slow on any model. The agent works at the level of plans and job status instead.

### Alternative C: One Tango command that runs the whole grid

*   **What it is**: A `map_region(plan)` command that returns when the map is done.
*   **Pros**: Simple to call.
*   **Cons**:
    *   Tango commands time out after 30 s, and the MCP client after 60 s.
    *   The client would time out while the stage keeps moving.
    *   There would be no way to abort, see progress or resume.
*   **Reason for Rejection**: Long jobs need start, status and abort as separate calls. That is the job envelope, which WG2 also needs.

### Alternative D: Approval checked inside the microscope device for every client

*   **What it is**: The guard runs in `ElectronMicroscope.move_stage` and the acquisition commands, so notebooks and scripts are held too.
*   **Pros**: There is no path around it.
*   **Cons**:
    *   Existing notebooks would change behaviour when the mode is switched.
    *   A synchronous Tango call cannot wait for a person for more than 30 s.
*   **Reason for Rejection**:
    *   The decision is that notebooks stay unchanged and that approval is a mode for agents.
    *   The mode is applied at the MCP server.
    *   Only the map job's own plan box and dose budget are enforced in a device.

### Alternative E: Keep the safety gate in SciAgentGUI

*   **What it is**: Leave the Rust approval queue in the GUI as the only gate.
*   **Pros**: It already exists and is tested.
*   **Cons**:
    *   It only sees calls from the GUI's own agent loop.
    *   It cannot see LangGraph graphs, Studio, server-side agents or other MCP clients.
    *   It treats unknown tools as low risk.
*   **Reason for Rejection**: Agents will run in asyncroscopy, so the mode has to live there. The GUI becomes the place a person sees the queue and switches the mode.

### Alternative F: Bluesky as the orchestration layer

*   **What it is**: Wrap the asyncroscopy devices as ophyd devices and run the map as a Bluesky plan through the RunEngine. Data goes to Tiled through Bluesky's event model.
*   **Pros**:
    *   It is a widely used standard with built-in provenance.
    *   It writes to Tiled natively.
    *   It matches the ATHENA "Bluesky reference integration" deliverable owned by WG1.
*   **Cons**:
    *   It adds a second control layer on top of Tango, and a second process model.
    *   Agents would have to drive the RunEngine, for which there is no MCP integration today.
    *   Pausing and approving would have to be designed twice.
*   **Reason for Rejection (for now)**:
    *   This design keeps one control layer.
    *   It is kept compatible with Bluesky:
        *   The envelope's verbs map onto RunEngine pause, resume and abort.
        *   The trace holds the information Bluesky documents need.
        *   Data lands in Tiled.
    *   A later Bluesky plan can wrap `Mapper`. That is listed under Future Improvements, as the bridge to WG1's deliverable.

### Alternative G: Keep or repair the existing stitchers

*   **What it is**: Fix `STEM-ML-Functions/stem_stitcher`, which finds atoms and then matches tiles with ORB features and RANSAC.
*   **Pros**:
    *   The code already exists.
    *   It works on atomic-resolution tiles.
*   **Cons**:
    *   Survey tiles show no atomic columns, so its first step has nothing to find.
    *   It ignores stage positions.
    *   It holds the whole map in memory.
    *   Two diverged copies already exist.
*   **Reason for Rejection**:
    *   Stage-first placement with phase-correlation refinement and a global solve suits survey maps and uses the information already recorded.
    *   The atom-based matching stays available as a refinement option for atomic-resolution maps, inside pyTEMlib.

### Alternative H: Use ASHLAR for registration

*   **What it is**: ASHLAR, the whole-slide stitcher from fluorescence microscopy. It also uses stage positions, phase correlation and a global solve.
*   **Pros**: A tested implementation of the same method.
*   **Cons**:
    *   It reads microscopy file formats through Bio-Formats (Java), not Tiled arrays.
    *   It is tuned for fluorescence channels.
    *   It would add a heavy dependency to the code going upstream into pyTEMlib.
*   **Reason for Rejection**: The method is small enough to implement next to pyTEMlib's existing registration. ASHLAR is used as a reference and, if convenient, as a cross-check in tests.

### Alternative I: Other map formats (Deep Zoom, pyramidal TIFF, loose PNGs)

*   **What they are**:
    *   Deep Zoom (DZI): a folder of image pyramids that OpenSeadragon reads.
    *   Pyramidal TIFF: one large file.
    *   Loose PNGs: what `pyramid.py` writes today.
*   **Pros**: DZI opens directly in browser viewers, and TIFF is one portable file.
*   **Cons**:
    *   None of them carry the stage-coordinate transform or the points-of-interest layer.
    *   None fit the "Tiled stores everything" decision.
    *   PNG and DZI are 8-bit, which loses raw counts.
*   **Reason for Rejection**: The map lives in Tiled with stage coordinates. An export to DZI can be added later if a browser-only viewer is needed.

### Alternative J: Multislice simulation per tile (abTEM or Prismatic)

*   **What it is**: Render every twin tile with full multislice physics.
*   **Pros**: Physically accurate contrast.
*   **Cons**:
    *   Seconds to minutes per tile even on a GPU, and impractical for 400 survey tiles on a 6 GB card.
    *   Both are GPL-3.0.
*   **Reason for Rejection**: The twin's purpose is rehearsal, not specimen physics. Multislice is used offline only, to check the cheaper approximation.

### Alternative K: Adopt pycroscopy's DTMicroscope as the twin

*   **What it is**: Replace the asyncroscopy twin with the Kalinin group's digital-twin server.
*   **Pros**: It is shared with a related group.
*   **Cons**:
    *   It has a different command interface.
    *   asyncroscopy's twin is a drop-in Tango device behind the same YAML as the real instrument. That is what lets a workflow move from twin to microscope unchanged.
*   **Reason for Rejection**: Swapping would break the rehearse-then-deploy property. DTMicroscope is a source of ideas only.

### Alternative L: pyTEMlib imports asyncroscopy's envelope code

*   **What it is**: The pyTEMlib MCP server depends on asyncroscopy to reuse `JobDevice` and the trace writer.
*   **Pros**: No duplicated code.
*   **Cons**: It would make the upstream pyTEMlib pull requests depend on asyncroscopy and Tango. The maintainer would have to accept that.
*   **Reason for Rejection**:
    *   A shared JSON Schema plus a conformance test in each repository gives the same guarantee without the dependency.

### Alternative M: A larger model or a cloud API

*   **What it is**: Run `gemma4:31b` or a hosted model, rather than Qwen3 on the 6 GB GPU.
*   **Pros**: Better tool use and longer context.
*   **Cons**:
    *   It does not fit this machine.
    *   WG2 reports that LLM API access is unresolved.
    *   Instrument PCs are often air-gapped.
*   **Reason for Rejection**:
    *   Designing for the small local model forces the coarse tools and short prompts that also make larger models cheaper and more reliable.
    *   Model choice stays a config setting, so the 5090 workstation or an API can be used where available.

### Alternative N: Separate design documents per stage

*   **What it is**: Design Stages 0–4 here, and Stage 5 (Studio, skills, GUI, pyTEMlib return) in later documents.
*   **Pros**: Smaller documents, and later designs can learn from Stage 2–4 results.
*   **Cons**: Contracts would be settled without Studio, skills and the thin GUI in view.
*   **Reason for Rejection**:
    *   The author chose one document.
    *   The approval mode reaching Studio runs is the clearest case where designing everything together matters.

--------------------------------------------------------------------------------

### Additional Resources

*   ATHENA Project Execution Plan v7a:
    *   WG1 and WG2 Year-1 deliverables (§2.1.4.1–2.1.4.2)
    *   WBS 1.3.5–1.3.7
    *   the milestone table (§3.9)
    *   Annex readiness milestones
*   ATHENA WG2 Quarterly Progress Report, revised 2026-10-26, and the WG2 working document:
    *   nanoindentation action schema v0.2
    *   common envelope
    *   capability registry v0.3
    *   STEM reward workflow
*   *Asyncroscopy: Vendor-Neutral, Agent-Compatible Control for Electron Microscopy*, paper outline and Chapter 3: the six design requirements and the twin's purpose.
*   `asyncroscopy/docs/designs/sample_mapping_building_blocks.md`: the RFC and multi-lens review this design follows.
*   `sciagentgui/docs/guide/safety-gate.md`: the auto/approval model that this design moves into asyncroscopy.
*   `sciagentgui/docs/internal/PYTEMLIB_REVIEW.md`: a pyTEMlib inventory and verified behaviour on its example data.
*   Twin references:
    *   Fiedler et al. 2023, stage error ([arXiv 2212.08683](https://arxiv.org/abs/2212.08683))
    *   Ophus et al. 2016, scan distortion ([arXiv 1507.00320](https://arxiv.org/abs/1507.00320))
    *   flyback distortion ([arXiv 2107.02066](https://arxiv.org/abs/2107.02066))
    *   STEMGym ([arXiv 2606.29592](https://arxiv.org/abs/2606.29592))
    *   AEcroscopy ([arXiv 2312.10281](https://arxiv.org/abs/2312.10281))
    *   pycroscopy DTMicroscope ([GitHub](https://github.com/pycroscopy/DTMicroscope))
    *   instamatic stage handling ([docs](https://instamatic.readthedocs.io/en/latest/api/components_stage/))
    *   Thermo Fisher Spectra 300 datasheet

### Future Improvements

*   **A Bluesky plan around `Mapper`**: Wrap the job envelope as a Bluesky plan, with documents that go to Tiled. This connects to WG1's Bluesky reference-integration deliverable.
*   **Segmentation as the first real scorer**:
    *   Wrap the existing SAM2 segmentation device (`asyncroscopy/mcp/segment.py`) or the `STEM-ML-Functions` segmenter as a `pytemlib.scorers` plug-in.
    *   Example scores: particle count, size or boundary density per region.
*   **Trained scorers and an LLM judge**: Each is registered through the same entry point and compared with the baseline using the logged scorer versions and the STEM reward criteria.
*   **Skills written by agents**: An agent captures a procedure as a skill. The skill is saved disabled until a person enables it on the operator endpoint. This restores what SciAgentGUI's skill store does today.
*   **`whittlegears/sciagent-skills`**: Decide whether it is archived or published as a mirror of asyncroscopy's `skills/`.
*   **Deep Zoom export**: Export a map from Tiled to DZI for viewers that work only in a browser.
*   **Instrument evidence**:
    *   Run the calibration scripts on the Spectra 300 and replace the unverified twin defaults.
    *   Repeat CUJ 1–3 on the instrument once the facility owner approves its readiness.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Defaults chosen for the open questions

Building cannot wait on these, so each has a default. Any of them can be changed before work starts.

| Question | Default used in this plan |
| :-- | :-- |
| Tiled and multi-resolution maps | Stage 3 opens with a one-day test against the installed Tiled version. If Tiled serves an OME-Zarr multiscale group as one node, use it. Otherwise write `mosaic/level_<n>` arrays plus `mosaic/index` metadata (level shapes, downsample factors, pixel-to-stage transform). Every reader and writer goes through one module, so the choice is made in one place. |
| An unanswered approval | The run pauses with the beam blanked. A request expires after `safety.request_timeout_s` (default 1800 s), and an expired request aborts the run. The expiry is traced. |
| EDS at each tile | `eds_point`, one spectrum at the tile centre. A full spectrum image (`eds_map`) only when the plan names it. |
| Trial thresholds | N = 10 seeded trials per task. Pass at M = 8. |
| Explore loop | The agent drives it with coarse tools: `score_regions` → `list_pois` → `Mapper` job over a small plan around each point. No separate explore graph is built. Pixels never pass through the model. |
| Leviasyncroscopy features | Not ported. `complete` already exists in `agent_mcp_server.py`. Agent backends are dropped. The skill store is replaced by asyncroscopy skills. |
| `whittlegears/sciagent-skills` | asyncroscopy's `skills/` folder is the source. Whether the public registry is archived or turned into a mirror is decided after Stage 5. |
| Operator-only commands | Two MCP endpoints run from config. The **agent** endpoint (port 8000) blocks `Approvals.approve`, `Approvals.deny`, `Approvals.set_mode` and `DigitalTwin.export_ground_truth`. The **operator** endpoint (port 8001, bound to 127.0.0.1) exposes them, and only the GUI connects to it. Without this, an agent could approve its own requests, because caller identity is a stub. |

### Target Files to Create/Modify

Paths are relative to each repository root:
*   `asyncroscopy/` → `whittlegears/asyncroscopy`
*   `pyTEMlib/` → the `whittlegears/pyTEMlib` fork
*   `sciagentgui/` → `whittlegears/sciagentgui`

#### Stage 0: environment

##### 1. `asyncroscopy/docs/Operation/windows_quickstart.md` (Create)

*   **Type / Scope**: Documentation.
*   **Proposed Changes**: Step-by-step setup on Windows:
    *   install uv (which installs Python 3.12), Rust and Ollama
    *   pull the Qwen3 models
    *   run `uv sync --extra agent --extra ollama --extra mapping`
    *   start the twin stack with local Tiled
    *   run the tests
    *   known Windows problems: long paths, the PyQt5 wheel, Tango database port 9094 in use

    It records what Stage 0 actually ran, so a second person can repeat it.

##### 2. `asyncroscopy/pyproject.toml` (Modify)

*   **Type / Scope**: Build configuration.
*   **Proposed Changes**: Add optional dependency groups:
    *   `mapping`: `scipy`, `zarr`, `jsonschema`. `numba` is added only if Stage 1 needs it.
    *   `analysis`: `pyTEMlib[mcp]` from the fork's branch, until upstream releases it
    *   `dev`: `exspy` (tests only, GPL), `langgraph-cli[inmem]`

    Base dependencies are unchanged.

##### 3. `asyncroscopy/configs/DigitalTwin copy.yaml` (Delete)

*   **Type / Scope**: Configuration.
*   **Proposed Changes**: Remove the duplicate config, which nothing references.

##### 4. `asyncroscopy/configs/qwen-llm.yaml` (Create)

*   **Type / Scope**: Configuration.
*   **Proposed Changes**: The agent config for this machine:
    *   model `qwen3:8b` until Stage 4 picks a tag
    *   `num_ctx` set
    *   `max_steps`
    *   MCP URL for the agent endpoint and the pyTEMlib server

    `gemma-llm.yaml` stays for the 5090 workstation.

##### 5. `asyncroscopy/tests/test_configs_load.py` (Create)

*   **Type / Scope**: Unit test.
*   **Proposed Changes**: Parses every YAML file in `configs/` with the launcher's config loader. This proves the new optional sections did not break existing files.

#### Stage 1: digital twin

##### 6. `asyncroscopy/asyncroscopy/instruments/electron_microscope/twin/__init__.py` (Create)

*   **Type / Scope**: Python package.
*   **Proposed Changes**: Holds the twin's models, so that `digital_twin.py` stays a thin Tango device. Exports the classes below.

##### 7. `.../twin/rng.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `EventRng(seed)` with `.for_event(name, index) -> numpy.random.Generator`. The seed is derived from the base seed, the name and the index. It is moved here from `digital_twin_tilt.py`'s `_rng`, so all twins share one helper.

##### 8. `.../twin/clock.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `SimClock(time_scale)`:
    *   `now()` returns simulated seconds.
    *   `advance(seconds)` moves simulated time forward and sleeps `seconds × time_scale` in real time.

##### 9. `.../twin/stage_model.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `StageModel(params, rng, clock)`:
    *   Keeps the commanded, true and reported 5-axis positions.
    *   `move(target)` applies the bias, Gaussian error and per-axis backlash on direction reversal, advances the clock by the move time, and starts settling drift.
    *   `true_position_at(t)` includes settling drift and global drift.
    *   `history()` returns every move for ground truth.

##### 10. `.../twin/sample_model.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   `SampleModel(params, rng)` builds the seeded particle table: id, centre, semi-axes, orientation, element, weight. It also builds a support film, an optional height plane, and a grid index.
    *   `particles_in(box)` finds the particles inside a box.
    *   `ground_truth()` returns the table.

##### 11. `.../twin/render.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `render_tile(sample, box_true, shape, level, detector) -> float array of expected signal`, at two levels of detail:
    *   **L0**: analytic projected thickness × Z^n, computed with vectorised NumPy over the particles in view.
    *   **L2**: atoms built with ASE for the particles in view, and projected without Python loops.

    The level is picked from the pixel size, using the `lod_atomic_px_m` threshold (default 0.2 nm). numba and an LRU cache of built particles are added only if `test_twin_render_perf.py` fails its time limits.

##### 12. `.../twin/optics.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   `probe_for(aberrations, shape, fov)` calls pyTEMlib's `get_probe`, cached by aberration state.
    *   `survey_blur_sigma(defocus, alpha)` gives the blur width used at L0.
    *   `aberrations_from_corrector(proxy)` reuses the coefficient mapping from `polycrystalline_gold_digital_twin.py`.

##### 13. `.../twin/detector.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `count(signal, current_pA, dwell_s, collection_fraction, gain, offset, read_noise, rng) -> uint16`. Raw counts, with no normalisation.

##### 14. `.../twin/scan_distortion.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `sample_coordinates(shape, fov, params, drift_fn, rng)` returns, for each pixel, the true sample coordinate. It includes per-line drift, line jitter, flyback compression, rotation, shear and FOV scale error. Rendering samples the signal at those coordinates.

##### 15. `.../twin/eds_model.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   `point_spectrum(sample, xy_true, live_time, params, rng) -> (energy_axis, counts, real_time, live_time)`. It builds the Kramers continuum, Gaussian lines with resolution that widens with energy, escape and sum peaks, and paralyzable dead time. Line energies come from pyTEMlib's tables.
    *   `spectrum_image(...)` builds a spectrum image on a coarse grid.

##### 16. `.../twin/dose.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: `DoseMap(resolution)` records dose in stage coordinates and returns:
    *   a contamination background, which saturates
    *   a damage factor for each particle

    This is the last item built in Stage 1. Stage 2 does not wait for it. Until it lands, the dose map only records dose and changes nothing in the image.

##### 17. `asyncroscopy/asyncroscopy/instruments/electron_microscope/digital_twin.py` (Modify)

*   **Type / Scope**: Tango device.
*   **Proposed Changes**:
    *   **Swap the internals.** Replace the inline sample, render and noise code with the `twin/` models.
    *   **Add properties.** One device property per parameter in the Technical Plan table, each defaulting to today's behaviour (all effects off).
    *   **Fix the stage and acquisition commands.**
        *   `get_stage` returns the reported position.
        *   `_get_fov` is implemented.
        *   `scan_region` is honoured.
        *   Each detector has its own contrast.
        *   Defocus and corrector aberrations drive the probe.
        *   `acquire_spectrum` returns a real spectrum with an energy axis.
    *   **Add `export_ground_truth`.** A new command returning JSON with the particles, every acquisition's true pose and drift offset, and the dose map summary.
    *   **Fix axis orientation.** One convention is fixed: axis 0 is y (rows), axis 1 is x. It is checked by a test.

##### 18. `asyncroscopy/asyncroscopy/instruments/electron_microscope/digital_twin_tilt.py` (Modify)

*   **Type / Scope**: Tango device.
*   **Proposed Changes**: Replace the local `_rng` with `twin.rng.EventRng`. Behaviour is unchanged, and `test_digital_twin_tilt.py` must still pass.

##### 19. `asyncroscopy/asyncroscopy/instruments/electron_microscope/digital_twin_particles.py` (Modify)

*   **Type / Scope**: Tango device.
*   **Proposed Changes**: Replace the global `np.random` calls with `EventRng`, so runs repeat for a given seed.

##### 20. `asyncroscopy/asyncroscopy/data/data_writer.py` (Modify)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   Spectra are saved with their energy axis, real time and live time.
    *   Images are saved with the commanded and reported stage position, FOV, pixel size and detector as attributes.
    *   An optional `container` argument places the file under `maps/<run_id>/...`.

##### 21. `asyncroscopy/configs/digital_twin_mapping.yaml` (Create)

*   **Type / Scope**: Configuration.
*   **Proposed Changes**:
    *   A twin stack with every effect on and the defaults from the Technical Plan table.
    *   `time_scale: 1.0`.
    *   The `Mapper` and `Approvals` devices (Stage 2).
    *   Local Tiled.

##### 22. `asyncroscopy/asyncroscopy/calibration/` (Create: `__init__.py`, `stage_repeatability.py`, `drift_series.py`, `dose_series.py`, `fit_twin_params.py`)

*   **Type / Scope**: Python scripts that run against any instrument config.
*   **Proposed Changes**:
    *   `stage_repeatability.py` makes repeated moves between points, approaching from both directions, and measures image shifts by cross-correlation.
    *   `drift_series.py` takes a time series on one feature.
    *   `dose_series.py` repeats acquisitions on a sacrificial area.
    *   `fit_twin_params.py` turns the results into a YAML block of twin parameters.

    These are run on the Spectra 300 by the facility team to replace the unverified defaults.

##### 23. `asyncroscopy/docs/digital_twin.md` (Modify)

*   **Type / Scope**: Documentation.
*   **Proposed Changes**:
    *   Remove `sample_extent_scale` and `regenerate_sample`, which do not exist.
    *   Document the models, parameters, levels of detail, ground truth and calibration scripts.

##### 24. `asyncroscopy/tests/test_twin_models.py` (Create)

*   **Type / Scope**: Unit tests.
*   **Proposed Changes**:
    *   **Determinism.** The same seed and command list give identical output, and a repeated acquisition differs only by noise.
    *   **Stage model.** Backlash appears only on a reversal, and reported minus commanded stays within the reading noise.
    *   **Clock.** With `time_scale` 0 there is no real sleep.
    *   **Detector.** Brightness is consistent between overlapping tiles: the mean ratio in the overlap is within 2% after noise.
    *   **Axes.** A particle placed at +x appears in the right half of the image.

##### 25. `asyncroscopy/tests/test_twin_render_perf.py` (Create)

*   **Type / Scope**: Performance test, marked `slow`.
*   **Proposed Changes**:
    *   An L0 512² tile renders in under 1 s and an L2 tile in under 2 s on the reference machine. The times are recorded to `benchmarks/results/twin_render.json`.
    *   A 20 nm defocus lowers a gradient-based sharpness score by at least 20%.

##### 26. `asyncroscopy/tests/test_twin_eds.py` (Create)

*   **Type / Scope**: Unit test (dev dependency exspy).
*   **Proposed Changes**: A twin spectrum from an Au particle fits in eXspy with Au lines found. Peak width at Mn Kα is within 5% of the configured value.

##### 27. `asyncroscopy/tests/test_digital_twin.py` (Modify)

*   **Type / Scope**: Unit test.
*   **Proposed Changes**: Update expectations that relied on reported equalling true, and on normalised images. Add `get_fov`, `scan_region` and `export_ground_truth` cases.

#### Stage 2: jobs, approval mode, trace

##### 28. `asyncroscopy/contracts/envelope.schema.json` (Create)

*   **Type / Scope**: JSON Schema.
*   **Proposed Changes**: Request and response shapes for `reserve`, `start`, `status`, `abort`, `results` and `release`. Covers the job states and error objects, plus `schema_version`. It is the shared contract for asyncroscopy, pyTEMlib and, later, other instruments.

##### 29. `asyncroscopy/contracts/trace_event.schema.json` (Create)

*   **Type / Scope**: JSON Schema.
*   **Proposed Changes**: One trace event:
    *   time (wall and simulated)
    *   run, plan, job and tile ids
    *   caller and grant id
    *   `approved_by`
    *   mode
    *   command, arguments, duration, outcome and result key

##### 30. `asyncroscopy/contracts/map_plan.schema.json` (Create)

*   **Type / Scope**: JSON Schema.
*   **Proposed Changes**: The Mapper plan:
    *   box
    *   FOV and overlap
    *   resolution and dwell
    *   detectors
    *   per-tile actions (`haadf`, `eds_point`, `eds_map`)
    *   dose budget
    *   travel order

##### 31. `asyncroscopy/contracts/README.md` (Create)

*   **Type / Scope**: Documentation.
*   **Proposed Changes**: What each schema is for, the versioning rule (a breaking change raises the major version), and how other repositories copy and test against the schemas.

##### 32. `asyncroscopy/asyncroscopy/jobs/__init__.py`, `job_device.py`, `state_store.py` (Create)

*   **Type / Scope**: Python package; Tango base class.
*   **Proposed Changes**:
    *   `JobDevice(Device)` implements the six envelope commands. Each takes and returns a JSON string, validated against the schema.
    *   It holds one worker thread per job. Subclasses implement `resolve_plan(plan) -> resolved` and `run_step(job, i) -> step_result`.
    *   `state_store` writes job state to `outputs/jobs/<job_id>.json` after every step, so `status` and resume survive a device restart.

##### 33. `asyncroscopy/asyncroscopy/instruments/electron_microscope/mapper.py` (Create)

*   **Type / Scope**: Tango device.
*   **Proposed Changes**:
    *   `Mapper(JobDevice)`.
    *   `resolve_plan` builds a serpentine tile list inside the box from FOV and overlap. It estimates time from stage, frame and spectrum timing, and dose from current, dwell and area. It rejects plans that exceed the dose budget.
    *   `run_step`, for each tile, in order:
        *   check abort
        *   check the tile is in the plan box
        *   `move_stage`
        *   acquire
        *   for `eds_point`, `place_beam` at the centre and `acquire_spectrum`
        *   register to Tiled via DATA under `maps/<run_id>/`
        *   emit a trace event
    *   On abort, a box violation or an exhausted budget, it runs `blank_beam` before stopping.
    *   It holds proxies to the microscope and DATA devices, named in config.

##### 34. `asyncroscopy/asyncroscopy/safety/__init__.py`, `approvals.py` (Create)

*   **Type / Scope**: Tango device.
*   **Proposed Changes**: `Approvals(Device)` holds the mode, the grants and the pending requests.
    *   **Commands:**
        *   `get_mode`, `set_mode(mode, who)`
        *   `request(json) -> request_id`
        *   `list_pending`
        *   `approve(request_id, who, scope_json) -> grant_id`, `deny(request_id, who, reason)`
        *   `check(command, args_json, grant_id) -> allowed | approval_required`
        *   `list_grants`
    *   **Expiry.** Requests expire after `request_timeout_s`.
    *   **Tracing.** Every change is written as a trace event.

##### 35. `asyncroscopy/asyncroscopy/safety/policy.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   Loads the risk map from config. Matching is by `Class.command` glob, with a default of `high`.
    *   `decide(mode, risk, grant_ok) -> run | hold`.
    *   Used by the MCP server hook.

##### 36. `asyncroscopy/asyncroscopy/mcp/mcp_server.py` (Modify)

*   **Type / Scope**: FastMCP server.
*   **Proposed Changes**:
    *   **Policy hook.** In `_create_wrapper`, before `_run` executes, call the policy:
        *   In `auto` mode, run, and trace with `approved_by: auto`.
        *   In `approve` mode, run high-risk calls only with a covering grant. Otherwise file a request and return `{"status": "approval_required", "request_id": ...}`.
    *   **Caller identity.** Read it from an MCP request header (`X-Asyncroscopy-Caller`), defaulting to `anonymous`.
    *   **Endpoint role.** A new config key, `endpoint_role: agent | operator`, selects which built-in block list applies (see Defaults).
    *   **Trace writer.** Replace the `ToolCallLog` use with the trace writer.

##### 37. `asyncroscopy/asyncroscopy/mcp/tool_log.py` (Modify)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   Add `TraceWriter`, which writes events to a local JSONL file in the `trace_event.schema.json` format.
    *   Every N events and at run end, it asks DATA to register the file as `maps/<run_id>/trace`.
    *   `ToolCallLog` and `read_tool_log` stay for existing callers.

##### 38. `asyncroscopy/asyncroscopy/data/data.py` (Modify)

*   **Type / Scope**: Tango device.
*   **Proposed Changes**: Add `register_in_container(json)`, which registers a file under a given Tiled container path with metadata. `Mapper`, the trace writer and pyTEMlib (through Tiled directly) all use the same layout.

##### 39. `asyncroscopy/configs/mcp_dt.yaml` (Modify) and `asyncroscopy/configs/mcp_operator.yaml` (Create)

*   **Type / Scope**: Configuration.
*   **Proposed Changes**:
    *   **`mcp_dt.yaml`** gets:
        *   `endpoint_role: agent`
        *   `safety.default_mode: auto`, a risk map and `request_timeout_s`
        *   `trace.enabled: true`
        *   blocks on `export_ground_truth` and the operator commands
    *   **`mcp_operator.yaml`** runs on 127.0.0.1:8001 with `endpoint_role: operator`, for the GUI.

##### 40. `asyncroscopy/startup_scripts/run_mcp.py` (Modify)

*   **Type / Scope**: Launcher.
*   **Proposed Changes**: Pass the new `safety`, `trace` and `endpoint_role` keys to `MCPServer`. Unknown keys keep raising, as today.

##### 41. `asyncroscopy/tests/test_job_device.py`, `test_mapper.py`, `test_approvals.py`, `test_mcp_policy.py`, `test_trace.py`, `test_contracts.py` (Create)

*   **Type / Scope**: Unit and integration tests on the twin.
*   **Proposed Changes**:
    *   **Envelope:** every response validates against the schema.
    *   **Mapper:**
        *   Tile centres match the closed-form grid.
        *   No centre falls outside the box.
        *   Abort stops within one tile, with the beam blanked.
        *   Restart then resume skips finished tiles.
        *   The dose budget stops the run.
    *   **Approvals:** mode switching, expiry, and that the operator endpoint is the only place `approve` exists.
    *   **Policy:**
        *   `auto` runs everything.
        *   `approve` holds unknown and high-risk tools.
        *   Notebook-style direct Tango calls are unaffected.
    *   **Trace:** one event per tile, carrying the grant id.

#### Stage 3: pyTEMlib analysis server and map

##### 42. `pyTEMlib/pyTEMlib/mcp_server/__init__.py`, `__main__.py`, `server.py` (Create)

*   **Type / Scope**: FastMCP server package in the fork.
*   **Proposed Changes**:
    *   `python -m pyTEMlib.mcp_server --tiled <uri> --host --port 8003` starts the server.
    *   It registers the map tools, the scoring tools and (Stage 5) the analysis tools.
    *   Its only dependencies beyond pyTEMlib are `fastmcp` and `tiled[client]`, both in a `mcp` extra.

##### 43. `pyTEMlib/pyTEMlib/mcp_server/tiled_io.py` (Create)

*   **Type / Scope**: Python module.
*   **Proposed Changes**: The only place that knows the `maps/<run_id>/` layout and the mosaic storage choice:
    *   `read_tiles(run_key)` returns arrays with their metadata.
    *   `write_mosaic(run_key, levels, transform)` writes the pyramid.
    *   `read_region(map_key, box, level)` reads part of it.
    *   `read_pois` and `append_poi` handle the points table.

##### 44. `pyTEMlib/pyTEMlib/mcp_server/jobs.py` and `pyTEMlib/pyTEMlib/mcp_server/schemas/envelope.schema.json` (Create)

*   **Type / Scope**: Python module, plus a copy of the schema.
*   **Proposed Changes**:
    *   An in-process job runner that follows the envelope, for assembly and scoring.
    *   Responses are validated against the copied schema.

##### 45. `pyTEMlib/pyTEMlib/image_tools/mosaic.py` (Create)

*   **Type / Scope**: Analysis module, next to the existing registration code.
*   **Proposed Changes**:
    *   `place_by_stage(tiles)` gives the initial placement from stage positions.
    *   `pairwise_offsets(tiles, pairs)` uses phase correlation and returns an offset and a confidence for each pair.
    *   `solve_positions(stage_xy, offsets, weights)` is a weighted least-squares solve. It drops pairs below the confidence threshold, and isolated tiles keep their stage position.
    *   `assemble(tiles, positions, writer)` writes the canvas in chunks with linear blending in the overlaps, then builds the pyramid by 2× downsampling.
    *   `pixel_to_stage` and `stage_to_pixel` convert between frames.
    *   It never imports asyncroscopy.

##### 46. `pyTEMlib/pyTEMlib/scoring/__init__.py`, `baseline.py` (Create)

*   **Type / Scope**: Analysis module.
*   **Proposed Changes**:
    *   A `Scorer` protocol (`name`, `version`, `score(region, metadata) -> (value, reason)`).
    *   Discovery through the `pytemlib.scorers` entry point.
    *   `baseline`: a contrast and gradient-sharpness score, normalised to the range 0–1.

##### 47. `pyTEMlib/pyTEMlib/mcp_server/map_tools.py` (Create)

*   **Type / Scope**: MCP tool definitions.
*   **Proposed Changes**:
    *   `assemble_map` (job)
    *   `list_tiles`
    *   `get_region`: a preview of at most 512 px, plus its Tiled key
    *   `list_pois`
    *   `add_poi`
    *   `score_regions` (job), which writes its top-k results as POIs carrying the scorer name and version
    *   `list_scorers`

##### 48. `pyTEMlib/pyproject.toml` (or `setup.py`, whichever the fork uses) (Modify)

*   **Type / Scope**: Build configuration.
*   **Proposed Changes**: Add the `mcp` extra and the `pytemlib.scorers` entry point for the baseline scorer.

##### 49. `pyTEMlib/tests/test_mosaic.py`, `test_scoring.py`, `test_mcp_server.py`, `test_envelope_conformance.py` (Create)

*   **Type / Scope**: Unit tests.
*   **Proposed Changes**:
    *   **Mosaic, on synthetic tiles cut from a known image with injected offsets:**
        *   With offsets at 5% of the FOV, the median error is under 1 px.
        *   With no offsets, the result is no worse than stage-only placement.
        *   Peak memory for a 2 GB canvas stays under 4 GB.
        *   Pixel-to-stage conversion round-trips within 1 px.
    *   **Scorer:** the baseline gives deterministic results.
    *   **Server:** run against a temporary Tiled server.
    *   **Envelope:** job responses validate against the schema.

##### 50. `asyncroscopy/tests/test_map_end_to_end.py` (Create)

*   **Type / Scope**: Integration test, marked `slow`.
*   **Proposed Changes**:
    *   Steps:
        1.  Start the mapping twin, with `time_scale` 0.
        2.  Run a 6×6 `Mapper` job.
        3.  Run `assemble_map` on the pyTEMlib server.
        4.  Compare tile positions with `export_ground_truth`.
    *   Acceptance criteria are the G5 thresholds.
    *   Also checks that the mosaic opens with `napari` reader code in headless mode.

##### 51. `asyncroscopy/tests/test_contract_copies.py` (Create)

*   **Type / Scope**: Unit test, run when the `analysis` extra is installed.
*   **Proposed Changes**: Checks that pyTEMlib's packaged copy of `envelope.schema.json` matches `contracts/envelope.schema.json` byte for byte. This catches drift between the two copies.

##### 52. `asyncroscopy/docs/designs/tiled_multiscale_spike.md` (Create)

*   **Type / Scope**: Documentation.
*   **Proposed Changes**: Records the Stage 3 test result: which mosaic storage was chosen, and why.

#### Stage 4: agent runs

##### 53. `asyncroscopy/asyncroscopy/agents/__init__.py`, `model.py`, `tools.py` (Create)

*   **Type / Scope**: Python package.
*   **Proposed Changes**:
    *   `model.py` builds the chat model from config (Ollama tag, `num_ctx`, temperature), or an OpenAI-compatible URL for the 5090 or an API.
    *   `tools.py` loads MCP tools from the asyncroscopy agent endpoint and the pyTEMlib server with `langchain-mcp-adapters`. It applies a per-graph allow-list and sets the `X-Asyncroscopy-Caller` header.

##### 54. `asyncroscopy/asyncroscopy/agents/graphs/__init__.py`, `mapping.py` (Create)

*   **Type / Scope**: LangGraph graph module.
*   **Proposed Changes**:
    *   `build_mapping_graph(config) -> CompiledGraph`.
    *   It runs a fixed sequence of nodes:
        1.  plan
        2.  reserve
        3.  approval
        4.  start
        5.  poll
        6.  assemble
        7.  score (optional)
        8.  report
    *   **Approval node.** It checks the mode. When `approve` is on, it uses a LangGraph `interrupt` while the request is pending.
    *   **State.** The state holds only ids, keys, coordinates and short text.
    *   **Export.** A module-level `graph` object is exported for `langgraph.json`.

##### 55. `asyncroscopy/asyncroscopy/mcp/agent.py` (Modify)

*   **Type / Scope**: Python module.
*   **Proposed Changes**:
    *   Set `num_ctx` on `ChatOllama`.
    *   Read the default model from config instead of the hard-coded `gemma4:31b`.
    *   `AgentSwarm.query(prompt, graph=None)` runs a named graph from `asyncroscopy.agents.graphs` when one is given. Otherwise it keeps today's supervisor behaviour.

##### 56. `asyncroscopy/asyncroscopy/mcp/agent_mcp_server.py` and `asyncroscopy/startup_scripts/run_llm.py` (Modify)

*   **Type / Scope**: FastMCP server and launcher.
*   **Proposed Changes**:
    *   `query_agent` gains an optional `graph` argument.
    *   The default model comes from YAML.

##### 57. `asyncroscopy/asyncroscopy/skills/mapping/SKILL.md` (Create)

*   **Type / Scope**: Agent skill.
*   **Proposed Changes**:
    *   **Front matter:** name, description, version, `min_asyncroscopy`, `envelope_schema`, required tools.
    *   **Body:**
        *   How to turn requests into a plan: whole sample, an area, EDS at each tile, revisit points.
        *   The defaults (`eds_point`) and when to ask the operator.
        *   How to read job status and approval responses.
        *   How to use the map and scoring tools without asking for pixels.

##### 58. `asyncroscopy/benchmarks/tool_json/prompts.jsonl`, `run_tool_json.py`, `run_mapping_trials.py`, `results/README.md` (Create)

*   **Type / Scope**: Benchmark scripts and data.
*   **Proposed Changes**:
    *   **`run_tool_json.py`** runs a fixed set of prompts against each candidate Qwen3 tag and `num_ctx`. It reports the valid tool-call JSON rate, which must be at least 90%.
    *   **`run_mapping_trials.py`** runs N = 10 seeded twin trials of the CUJ 1–3 tasks. For each it records:
        *   pass or fail
        *   tool calls
        *   tokens
        *   wall time
        *   any out-of-box move
    *   Results go to `results/` as JSON plus a summary table.

##### 59. `asyncroscopy/tests/test_mapping_graph.py` (Create)

*   **Type / Scope**: Unit test.
*   **Proposed Changes**: Runs the mapping graph with a scripted fake model against the twin. Checks the order of node visits and the interrupt in `approve` mode, and that no state field holds an array.

#### Stage 5: Studio, skills, thin GUI, pyTEMlib return, WG2

##### 60. `asyncroscopy/langgraph.json` (Create)

*   **Type / Scope**: LangGraph configuration.
*   **Proposed Changes**:
    *   Lists `mapping` and `general` as `asyncroscopy.agents.graphs.<name>:graph`.
    *   Includes the env file and dependencies, so `langgraph dev` loads both graphs.

##### 61. `asyncroscopy/asyncroscopy/agents/graphs/general.py` (Create)

*   **Type / Scope**: LangGraph graph module.
*   **Proposed Changes**: A ReAct agent with the allow-listed tools, plus `list_skills` and `load_skill`. Same approval handling as `mapping`.

##### 62. `asyncroscopy/asyncroscopy/skills/registry.py` and `asyncroscopy/asyncroscopy/skills/README.md` (Create)

*   **Type / Scope**: Python module and documentation.
*   **Proposed Changes**:
    *   **Discovery.** Finds `skills/*/SKILL.md` and parses front matter.
    *   **Version check.** Refuses skills whose minimum versions are not met.
    *   **README.** Explains the format and how to add a skill by pull request.

##### 63. `asyncroscopy/asyncroscopy/mcp/mcp_server.py` (Modify, second change)

*   **Type / Scope**: FastMCP server.
*   **Proposed Changes**: Add the native tools `list_skills` and `load_skill(name)` on both endpoints.

##### 64. `sciagentgui/src-tauri/src/harness.rs`, `llm_bridge.rs`, `model_provider.rs`, `src/lib/harness/` (Modify; remove the local agent loop)

*   **Type / Scope**: Rust backend and TypeScript.
*   **Proposed Changes**:
    *   Chat sends the prompt to asyncroscopy's `query_agent` over MCP and shows the returned transcript.
    *   The local tool-calling loop is removed. `model_provider.rs` keeps only what is needed to show which model the server uses.

##### 65. `sciagentgui/src-tauri/src/safety.rs`, `src/components/SafetyGate.tsx`, `src/components/SafetySettings.tsx` (Modify)

*   **Type / Scope**: Rust backend and React.
*   **Proposed Changes**:
    *   The Rust approval queue and the name-based classifier are replaced by a client of the operator endpoint (`list_pending`, `approve`, `deny`, `get_mode`, `set_mode`).
    *   `SafetySettings` switches the asyncroscopy mode, and `SafetyGate` becomes the approvals panel.

##### 66. `sciagentgui/src/components/pystem/MapView.tsx`, `src/components/RunStatus.tsx`, `src/lib/tiledClient.ts` (Create)

*   **Type / Scope**: React components and TypeScript client.
*   **Proposed Changes**:
    *   **`tiledClient.ts`** fetches array slices and metadata from Tiled's HTTP API.
    *   **`MapView`** draws the mosaic level that matches the zoom, panning by tiles, using OpenSeadragon with a custom tile source. It also:
        *   overlays POIs from `maps/<run_id>/poi`
        *   opens the linked spectrum or image on click
        *   shows stage coordinates under the cursor
    *   **`RunStatus`** polls `Mapper.status`.

##### 67. `sciagentgui/src-tauri/src/skills.rs`, `skill_frontmatter.rs`, `src/components/SkillsManager.tsx`, `src/lib/skillSync.ts` (Modify)

*   **Type / Scope**: Rust and React.
*   **Proposed Changes**:
    *   Skills are listed and read from asyncroscopy over MCP. The Skills page becomes read-only.
    *   The local skill store, agent skill writing and usage recording are removed. Agent skill writing returns when the future improvement lands in asyncroscopy.

##### 68. `sciagentgui/src-tauri/src/analysis/`, `analysis_tools.rs`, `pytemlib.rs`, `scripts/pytemlib_backend.py` (Delete, last step of Stage 5)

*   **Type / Scope**: Rust and Python.
*   **Proposed Changes**:
    *   Removed only after item 70's equivalence tests pass for every function.
    *   The workbench views then call the pyTEMlib MCP server.

##### 69. `sciagentgui/src-tauri/tests/export_analysis_fixtures.rs` (Create)

*   **Type / Scope**: Rust test that writes fixtures.
*   **Proposed Changes**: Runs each Rust analysis function on pyTEMlib's example data and writes inputs and outputs to JSON. These are the reference for the port back.

##### 70. `pyTEMlib/pyTEMlib/{eds_tools,eels_tools,image_tools,probe_tools}/...` and `pyTEMlib/tests/test_rust_equivalence.py` (Modify / Create)

*   **Type / Scope**: Analysis modules and tests.
*   **Proposed Changes**:
    *   Any behaviour the Rust versions added or fixed is ported into the matching pyTEMlib functions.
    *   The test loads the item 69 fixtures and checks each output within a stated tolerance per function.
    *   `pyTEMlib/mcp_server/analysis_tools.py` (Create) exposes them as MCP tools.

##### 71. `sciagentgui/.claude/skills/asyncroscopy-integration/SKILL.md`, `CLAUDE.md`, `docs/guide/safety-gate.md`, `docs/guide/skills.md`, `docs/architecture/mcp-integration.md`, `docs/reference/feature-status.md`, `docs/getting-started/connect-an-instrument.md` (Modify)

*   **Type / Scope**: Documentation.
*   **Proposed Changes**:
    *   Point at `whittlegears/asyncroscopy`.
    *   Describe the two endpoints, the asyncroscopy mode and the thin-client role.
    *   Remove Leviasyncroscopy references.

##### 72. `asyncroscopy/docs/athena/wg2_traceability.md` and `asyncroscopy/docs/athena/registry_entry_stem_twin.yaml` (Create)

*   **Type / Scope**: Documentation.
*   **Proposed Changes**:
    *   **Traceability table.** Each WG2 Year-1 item → artifact → test or run that shows it → twin-only or instrument status.
    *   **Registry entry.** A capability-registry entry in the v0.3 format, for Charles Cao to merge.

##### 73. `asyncroscopy/README.md` and `asyncroscopy/docs/myst.yml` (Modify)

*   **Type / Scope**: Documentation.
*   **Proposed Changes**: Add the mapping, approval-mode, skills and Studio sections and the quickstart link, and register the new pages.

### Verification Plan

*   **Compilation/Build**:
    *   asyncroscopy: `uv sync --extra agent --extra ollama --extra mapping --extra analysis --extra dev`
    *   sciagentgui: `npm run build`, `cargo build --manifest-path src-tauri/Cargo.toml`
    *   pyTEMlib: `pip install -e .[mcp]`
*   **Unit Tests**:
    *   asyncroscopy: `uv run pytest -m "not slow"`
    *   pyTEMlib: `pytest`
    *   sciagentgui: `cargo test --manifest-path src-tauri/Cargo.toml`, `npm test`
*   **Static Analysis & Linting**: `uv run ruff check` (asyncroscopy, pyTEMlib), `npm run lint`, `cargo clippy`.
*   **Test Coverage Requirement**: At least 85% line coverage on the new packages (`twin/`, `jobs/`, `safety/`, `agents/`, `skills/`, `pyTEMlib/mcp_server/`, `image_tools/mosaic.py`, `scoring/`). Existing modules must not drop in coverage.
*   **E2E/Integration**:
    *   `uv run pytest -m slow` covers `test_twin_render_perf`, `test_map_end_to_end` and the mapping graph against the real twin.
    *   `uv run python benchmarks/run_tool_json.py` must report at least 90% valid JSON for the chosen tag.
    *   `uv run python benchmarks/run_mapping_trials.py --n 10` must pass at least 8 of 10 trials per task, with zero out-of-box moves.
    *   `uv run langgraph dev` must load both graphs, and the mapping graph must complete one twin run from Studio.
    *   Manual GUI check: CUJ 1, 2 and 4 in SciAgentGUI against the twin stack. Screenshots are kept with the WG2 traceability table.

--------------------------------------------------------------------------------

*Drafted with EGM*

--------------------------------------------------------------------------------

## Implementation notes (2026-10-28)

What was built differs from the plan above in these places. Each change came from running the system, not from preference.

| Plan | Built | Why |
| :-- | :-- | :-- |
| Schemas in `contracts/` at the repository root | `asyncroscopy/contracts/` | Ships with the package, so an installed copy can validate. |
| Envelope verb `status` on Tango | `job_status` | Tango command names are case-insensitive and every device already has `Status`. |
| `DigitalTwin` rewritten in place | Original renderer moved to `AtomisticDigitalTwin` (`digital_twin_base.py`); new `DigitalTwin` builds on it | Four specialised twins depended on the old internals. |
| Three twin levels of detail (L0, L1, L2) and numba | Two levels, vectorised NumPy | Measured: survey tile ~30 ms, first atomic tile ~0.7 s, so L1 and numba were not needed. |
| Region guard inside the microscope device for every client | Approval mode at the MCP server; plan box and dose budget in `Mapper` | Author's decision: notebooks are never held. |
| `fit_twin_params.py` | Each calibration script prints its YAML block | Simpler; `tests/test_calibration.py` shows the stage script recovers the twin's configured errors. |
| `DATA.register_in_container` | `Mapper` writes one run file under the DATA save path and registers it | Tiled serves only files under that path; one file per run is easier to move and share. |
| Mosaic stored as OME-Zarr or per-level arrays (Stage 3 spike) | Tiled container `<run>_map` with `level_N` arrays and transform + points in its metadata | Tiled client writes arrays and metadata directly; no OME-Zarr reader needed. |
| Phase correlation of the predicted overlap strip | Masked normalised cross-correlation of whole tiles, outlier rejection in the solve | The strip method was biased and failed when the stage error approached the overlap width (found by the accuracy test). |
| `langchain-mcp-adapters` | `asyncroscopy/agents/mcp_tools.py` on fastmcp's client | The adapter pins an `mcp` version fastmcp 4 cannot use; the existing agent swarm could not import. |
| pyTEMlib analysis job tools `job_status`, `job_results` | `analysis_job_status`, `analysis_job_results` | Every local model tested confused them with `Mapper_job_status`. |
| JSON-string arguments for envelope and approval commands | Also accept the object, or its fields directly | Local models pass objects far more reliably than JSON strings. |
| Model chooses the whole plan | Plan repaired in code: units, swapped corners, tiny boxes, EDS keywords, and stated sizes ("a 600 nm square") | qwen3:8b planned the whole sample for "a 600 nm square around the current position". |
| Agent-written skills | Not built (as decided in the zoom-out review) | Future improvement. |
| pyTEMlib upstream package | `pyproject.toml` takes pyTEMlib from `whittlegears/pyTEMlib@feature/mcp-mapping` | The stack's launcher re-syncs the environment; the fork must be a declared source until upstream merges it. |

Found and fixed along the way, outside the plan: the test fixtures passed Windows paths as Tango properties, which the test database cut at the drive colon, so every test run wrote HDF5 files into the working tree; the launcher block-buffered its output when not writing to a terminal, so "ready" never appeared in logs or to programs reading it; pyTEMlib ran a demo fit and printed a non-ASCII character on every import, which crashed Windows consoles.
