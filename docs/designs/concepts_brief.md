# Concept brief: closing WG2 Year-1 on the twin, and STEM automation for first light

Status: brainstorm output, not a plan of record. Date: 2026-10-29. This brief replaces the earlier "portable, hackathon-ready stack" brief of the same date.

Method: six independent reviewers read the repository and reported in parallel: design, codebase details, zoom-out, comprehension, implementation readiness and structure. This brief merges and deduplicates what they found. Line numbers refer to today's tree. Paths without a prefix are under `C:\Users\leafe\Documents\Github\asyncroscopy`. Paths that start with `pyTEMlib/` are under `C:\Users\leafe\Documents\Github\pyTEMlib-mcp`.

## 1. Overview

The concept has two halves.

The first half finishes the three open WG2 Year-1 items that can be shown on the digital twin:

- an online Instrument Trace pipeline;
- a field-of-view reward that has actually been evaluated;
- parity checks between the twin and AutoScript, with a dry-run mode.

The second half automates steps the operator does by hand today: autofocus and astigmatism correction, drift tracking during a map, and acquisition that respects a dose budget. All of this has to be easy for a hackathon team to extend, and it has to turn first light on the Spectra 300 into a short, scripted calibration.

**This reverses two non-goals from the previous brief.** That brief ruled out live drift correction and the online trace pipeline. This brief brings both back into scope, but with limits. Live drift correction is built last and is off by default. The online trace is a small tailer over the existing files, not a new piece of infrastructure.

The reviewers agreed on the main finding. **First light is not held back by missing automation. It is held back by an AutoScript path that has never run and that has bugs no test catches.** Several of the features already exist in part:

- Dose budgets are checked when a plan is reserved.
- The twin's corrector changes how sharp the image is.
- The stitch fits drift after the map.
- The twin can export its hidden ground truth.

The real gaps are a dry-run seam, a recorded AutoScript fake to test parity against, a beam current that is measured rather than assumed, and a single time source.

## 2. Analysis

Each finding carries the highest severity any reviewer gave it.

### 2.1 Parity, dry-run and the AutoScript path (WG2: twin-to-instrument parity)

- **Blocker: AutoScript bugs that the tests cannot see.** The tests use `SimpleNamespace` fakes (`tests/test_auto_script_microscope.py:406-414`), so none of these fail today. In `asyncroscopy/instruments/electron_microscope/auto_script.py`:
  - The TEM branch of `_auto_focus` (585-586) has a missing dot (`autoscript_tem_microscope_client.structuresRun...`) and uses `self.microscope` where it should use `self._microscope`.
  - `_get_stage` is defined twice (515 and 568). The second definition silently wins, so `register_stage` (253-260) never reads the hardware. The dead first version also swaps x and y.
  - The mode string is spelled two ways: `'Stem'` at 551/581 and `'stem'` at 442.
  - The diffraction-shift setter checks `projector_mode` (605), but the getter checks `optical_mode` (611).
  - `_set_image_shift` and `_set_beam_tilt` catch and log their errors (597, 635), so a failed write looks like a success.
- **Blocker: `configs/Spectra300.yaml` has no Mapper device.** In addition, the AutoScript microscope's proxy map has no corrector entry (`auto_script.py:191-197`), although the twin wires one up (`digital_twin_base.py:141`). As configured, mapping, drift tracking and the reward have no way to reach the hardware.
- **Major: a check of command names alone would pass trivially.** Both classes inherit every command from `electron_microscope.py`. The real differences are in behaviour:
  - The twin lacks `_auto_focus`, the screen-current commands, image shift, beam tilt and diffraction shift. Calls to any of these fall through to `NotImplementedError` (`electron_microscope.py:381-480`).
  - The twin checks defocus, field of view, scan region and stage limits (`digital_twin.py:316-362`). AutoScript checks none of them.
  - The `get_parameters` keys differ between the two.
  - The defocus attribute is documented as micrometres but its unit is metres (`auto_script.py:117-120`).
- **Major: there is no dry-run seam where it is needed.**
  - `testing_mode` turns hardware calls into no-ops that return `nan`. That is not a dry-run.
  - The Mapper drives the microscope through `JobDevice.proxy` (`jobs/job_device.py:806-811`). That path sits below the MCP approval gate, so a dry-run added at the MCP layer would never see a stage move.
- **Major: an item from the previous brief is still partly open.**
  - Closed: `configs/mcp.yaml:37` names an approvals device, `Spectra300.yaml:18` defaults to `approve`, and `startup_scripts/run_mcp.py:54-62` refuses to start without approvals.
  - Still open: `mcp.yaml:17` binds `0.0.0.0`. There is no `simulated:` flag, so nothing stops `set_mode(auto)` on hardware.

### 2.2 Dose-aware acquisition

- **Blocker: the dose budget on hardware is fiction.**
  - `AutoScriptMicroscope._get_parameters` (`auto_script.py:535-566`) never reports `beam_current_pa`. The Mapper therefore always falls back to `default_beam_current_pa = 50` (`mapper.py:324,334-342`).
  - `_set_screen_current` fits a degree-11 polynomial to 15 points, then takes the largest real root and writes it to the monochromator focus with no range check (`auto_script.py:486-508`). That is an unbounded write to the instrument.
- **Major: EDS dose is reported but not counted against the budget** (`mapper.py:427-429`).
- What already works: the plan dose budget is checked at reserve (`mapper.py:391-403,468-471`), and the twin keeps a `DoseMap` whose damage and contamination feed back into the image (`twin/dose.py`, `digital_twin.py:409,444`).

### 2.3 Autofocus and astigmatism correction

- **Blocker: on the twin, "accurate" proves nothing.** `DigitalTwinCorrector._measure_c1a1` returns the hidden values, and its noise defaults to 0 (`hardware/corrector_digital_twin.py:33,61-65,85-89`). A measure-then-correct loop therefore converges by construction.
- **Blocker: the mapping twin has no `_auto_focus` and no image shift.** Only the tilt twin implements them (`digital_twin_tilt.py:775,811`).
- **Major: "through the corrector" covers two mechanisms that could fight each other.**
  - On the Spectra, AutoScript autofocus runs OptiSTEM C1A1 (`auto_script.py:579-583`), which is not the CEOS corrector.
  - CEOS `correct_aberration` takes a free-text string whose sign and units are "backend-native" (`hardware/corrector.py:55-61,101`).
  - The twin subtracts the requested value (`corrector_digital_twin.py:99-102`). Nothing confirms that CEOS behaves the same way. With the wrong sign, a correction doubles the aberration instead of removing it.
- **Major: CEOS failures can look like success.**
  - A JSON-RPC reply that contains `"error"` is returned as a success (`corrector_ceos.py:81-84`).
  - `socket_timeout` defaults to 120000 s (`corrector_ceos.py:29-33`).
  - The corrector does not appear in `safety/policy.py`. Its calls are held only because unknown commands count as high risk.
- **Major: astigmatism is only partly testable on the twin.**
  - At survey pixel sizes the blur is one round Gaussian whose size depends on `hypot(defocus, |A1|)` (`twin/optics.py:48-54`, `twin/render.py:165`). C1 and A1 cannot be told apart, the direction of A1 is invisible, and over-focus looks the same as under-focus.
  - Focus has two sources, the corrector's C1 and the microscope's `_defocus` (`optics.py:44`). `measure_c1a1` reports only the corrector's value.

### 2.4 Drift tracking during maps

- **Blocker for any drift work: the twin and the Mapper use different clocks.**
  - Twin drift advances on `SimClock` (`twin/clock.py:21-27`).
  - The Mapper timestamps tiles with `time.time()` and extrapolates drift in wall time (`mapper.py:706,931`).
  - With `time_scale: 0`, the hackathon setting, the fitted drift rates are wrong by an arbitrary factor.
- **Major: drift is only handled after the map.** The stitch fits drift (`mapper.py:826-846`), and `drift_at` (197-207) is used only for revisits (929-931). Corrections smaller than about 100 nm need image or beam shift, but `_revisit` moves the stage. On AutoScript, image shift swallows its errors (`auto_script.py:597`).
- **Major: drift and dose logic are already built into Mapper** (`frame_dose` at `mapper.py:191`, `drift_at` at 197, in a file of about 1,100 lines). A new tracker built next to this code would mean two drift models.

### 2.5 Evaluated reward (WG2: STEM field-of-view reward)

- **Major: there is ground truth but no evaluation harness.**
  - The only scorer is `BaselineScorer` (`pyTEMlib/scoring/baseline.py:17-36`).
  - Ground truth comes from `sample_model.ground_truth()` (`twin/sample_model.py:164-174`) through `export_ground_truth` (`digital_twin.py:605-620`).
  - No metric, threshold or labelled set exists.
- **Major: measuring the reward against particle positions measures detection, not "quality and feature visibility".** The twin draws particles bright, and the baseline scorer finds bright things. Report the result as a detection proxy. The "Not claimed" list already says there is no human-preference data (`docs/athena/wg2_traceability.md:69`).
- A plug-in point already exists: the `pytemlib.scorers` entry point (`pyTEMlib/scoring/__init__.py:37-49`).

### 2.6 Online Instrument Trace (WG2: online trace pipeline)

- **Major: there are three trace files and two schemas.**
  - Each process writes its own JSONL file: the MCP server, the approvals device, and the jobs device (`outputs/trace/jobs.jsonl`, `job_device.py:229`).
  - pyTEMlib's `ToolTrace` (`pyTEMlib/mcp_server/trace.py:24`) uses a different set of fields.
  - Events carry only a wall-clock `ts`. There is no `seq`, no `run_id` shared across sources, no sim time and no `simulated` or `dry_run` flag (`asyncroscopy/mcp/tool_log.py:74-90`; `contracts/trace_event.schema.json:13` has a closed `source` enum).
  - Direct Tango or notebook calls skip the trace entirely.
- **Nit, keep it small: "online" can be a follow-mode tailer** over the existing files plus one SSE or MCP endpoint, about 50 lines. A broker is not justified for one instrument.

### 2.7 Hackathon use and documentation

- **Major: there is no plug-in point for focus or drift strategies.** Use the same pattern as scorers: an entry point, an example of about 40 lines, and a bench, `asyncroscopy.testing.focus_bench(my_focuser, seeds=...)`, that injects known errors into the twin and scores how well they are recovered without a live stack or an LLM.
- **Major: automation loops must not go through the LLM.** qwen3:8b takes about 11 s per call. A focus sweep plus a stig grid would take about 7 minutes and overflow an 8k context. In approve mode, every one of those calls would also be held. The agent decides when to tune; a job runs the loop.
- **Major: no rule says where new code should go, and the traceability doc cannot carry one evidence cell per item.**
  - `electron_microscope/` already mixes drivers, five twin variants and a job.
  - Rows 11 and 12 of `wg2_traceability.md` each mix two statuses.
  - `docs/myst.yml` does not publish `athena/` or `designs/`.
- **Major: the key terms are undefined.** Section 3 gives working definitions.

## 3. Goals vs Non-Goals

### Working definitions

| Term | Meaning in this brief |
| :-- | :-- |
| Parity | Twin and AutoScript agree on each command's name, argument types, units, bounds, risk class, return shape, and which values they reject. Checked against a recorded fake AutoScript client. |
| Dry-run | Write calls are validated, gated, traced and checked against their bounds, but not sent. Read calls are answered. The stage does not move. |
| Online trace | A subscriber receives every event of a run, in order, within 2 s, from every source. |
| Evaluated reward | On N seeded twin samples, the scorer's top-k fields of view are compared with ground truth, using a stated metric and threshold, a random baseline, and the scorer version. Labelled as a detection proxy. |
| Dose-aware | A plan is refused or adapted before any area would exceed its budget, using a measured beam current, with EDS counted. |
| Drift tracking | Tile placement error stays bounded while the map runs, without relying on the stitch. |
| Robust | Four measurable claims: time to build an extension, accuracy against ground truth, map yield, and the share of steps that need no operator. "Optimal" cannot be tested on the twin and is not claimed. |

### Goals

1. Every hardware call that first light needs is covered by a behavioural parity test. A dry-run of the first-light sequence leaves the instrument untouched.
2. Dose budgets use a measured current that carries its provenance, and they include EDS.
3. Autofocus and stig run as bounded, abortable jobs. Strategies can be swapped (vendor OptiSTEM, image metric, corrector). Each step has a limit, the state is snapshotted and restored, and a correction aborts if the error grows.
4. The reward is evaluated on twin ground truth, and the twin is degraded with defocus and contamination so that the score has to respond to image quality.
5. Trace events stream online from every source, with a sequence number, sim time, run ID and a `simulated`/`dry_run` flag.
6. A hackathon team can add a scorer, focuser or drift tracker and get a number from a bench, with no LLM and no live stack.
7. Each WG2 row says whether its evidence shows "logic proven (twin)" or "performance proven (instrument)".

### Non-goals

- Proving accuracy or optimal results on the twin. The twin proves control logic and safety. Performance waits for first light.
- A learned reward, or human-preference labels before first light. Operator picks at first light become the first labels.
- A custom autostigmation algorithm that relies on the direction of A1. The twin cannot show the direction at survey pixel sizes.
- A message broker, Kafka, or live Tiled streaming for the trace.
- Running tuning loops through the LLM.
- WG1 authentication, and schemas for nanoindentation or SPM.

## 4. Key Decisions and Staging

### Decisions to make first

- **D1: Where does the code live?** Recommended:
  - pure, backend-agnostic functions in `asyncroscopy/routines/` (`focus.py`, `drift.py`, `dose.py`, with `frame_dose` and `drift_at` moved out of `mapper.py`);
  - multi-step work that moves the stage or exposes the sample goes in a JobDevice, as a new `Tuner` next to `Mapper`;
  - compute-only work goes in pyTEMlib (`scoring/evaluate.py`);
  - calibration goes in the existing `asyncroscopy/calibration/`.
- **D2: Who owns C1 and A1 on the Spectra?** Recommended: OptiSTEM is the default focus strategy. Writes to the CEOS corrector happen only after first light has settled the sign and units.
- **D3: What does dry-run allow?** Recommended: all reads, no writes, and blanking is the only exception.
- **D4: One time source.** Recommended: devices report `clock_s`, and the Mapper and the trace use it.
- **D5: A `simulated: true|false` stack flag.** Recommended, carried over from the previous brief. When the stack is not simulated, it binds to 127.0.0.1, refuses `auto`, and refuses a plan that has a dose budget but uses the default current.
- **D6: The reward metric and its threshold** (precision@k with k = 5 is suggested). Agree it with WG2 before any results are run.

### Staging

- **Stage 1, before the hackathon:** items 1 to 4. They make the twin trustworthy and make first light possible.
- **Stage 2, hackathon-facing:** items 5 to 7, plus `docs/hackathon/plug_in_points.md`.
- **Stage 3, after first light:** item 8, then tuning with the measured values.
- **Quick wins:**
  - Split traceability rows 11 and 12.
  - Give each WG2 row an ID, an artifact, a check command and an evidence-file cell.
  - Add `athena/` and `designs/` to `myst.yml`.
  - Add "optimal results unproven until first light" to "Not claimed".

### Ranked plan

Every "done when" test runs on the twin or on the recorded AutoScript fake.

1. **AutoScript fixes, a recorded fake client and a behavioural parity suite.**
   - Fix the bugs listed in 2.1.
   - Add the Mapper and the corrector address to `Spectra300.yaml`.
   - Done when: `tests/test_parity.py`, run against the twin and against `AutoScriptMicroscope` on the fake, fails on today's code (the duplicate `_get_stage`, the broken TEM autofocus, the swallowed shift errors) and passes after the fixes. It asserts the same accepted arguments, the same rejected out-of-range values and the same return shapes.
2. **Dry-run through a `RecordingProxy` at `JobDevice.proxy`, plus the `simulated` flag (D5).**
   - Done when: a dry-run 3x3 map leaves `export_ground_truth().moves` empty, and its trace lists 9 `move_stage` calls, each checked against the plan box and marked `dry_run`. A second test confirms that `set_mode(auto)` is refused on a stack that is not simulated.
3. **Measured beam current and a complete dose budget.**
   - Add `beam_current_pa` and `beam_current_source` to `get_parameters`.
   - Count EDS dwell in the budget.
   - Range-check the screen-current root.
   - Done when: `tests/test_dose_aware.py` shows the twin `DoseMap` total within ±5 % of the plan estimate with EDS, that an over-budget plan is refused before any unblank, that the fake client refuses an out-of-range root, and that a non-simulated plan using the default current is refused.
4. **One time source.**
   - Done when: with `time_scale: 0` and drift switched on, the fitted drift rate is within 10 % of ground truth, and a revisit lands within 0.1 field of view.
5. **Evaluated field-of-view reward** (`pyTEMlib/scoring/evaluate.py`, which takes any registered scorer by name).
   - Done when: `tests/test_reward_eval.py` shows baseline precision@5 at or above the D6 threshold on 3 seeds, a random scorer below it, and a lower score under added defocus and contamination. The result is stored with `scorer_version` and re-scored in CI.
6. **`Tuner` job for autofocus and stig.**
   - Add the twin's `_auto_focus` and image shift.
   - Set corrector noise above 0.
   - Strategies are vendor, image metric and corrector.
   - Each step is limited, the state is snapshotted and restored, and the beam is blanked in try/finally.
   - Done when: in 10 of 10 seeds starting at ±200 nm defocus with a height gradient, the residual C1 is below 5 nm within the declared dose, and a twin with the correction sign inverted aborts and restores the snapshot. A `focus_bench` gives a score to a 40-line example focuser.
7. **Online trace.**
   - Add `seq`, `run_id`, `clock_s`, `host` and `simulated`/`dry_run` (schema 1.1.0).
   - Add a follow-mode tailer that merges the per-process files, and one SSE or MCP endpoint.
   - Align pyTEMlib's `ToolTrace` with the schema.
   - Done when: `tests/test_trace_stream.py` attaches a subscriber during a 4-tile map and receives exactly `read_tool_log(run)`, in order, with no duplicates, each event within 2 s.
8. **Live drift tracking as a Mapper hook** (off by default, built on `routines/drift.py`).
   - Correct with image shift, with a limit on each correction.
   - Done when: `tests/test_live_drift.py` at 5e-11 m/s keeps the placement error below 0.2 field of view without stitching, traces every correction, and never takes the stage outside the plan box.

## 5. First-light checklist (Spectra 300, through AutoScript)

Before the session, items 1 to 3 must pass. Approvals must be in `approve` mode, the endpoint bound to 127.0.0.1, and the limits small. Every step is traced with `simulated=false`, and an operator signs it off. If any step aborts, blank the beam. `asyncroscopy/calibration/first_light.py --step N` runs the steps.

1. **Read-only connect.** Read the parameters, stage, field of view, defocus and corrector info. Abort if the mode or voltage is not what was expected.
2. **Dry-run the 3x3 map** against the live connection. The diff against the twin trace must be empty or explained, and the stage must not move.
3. **Blank and unblank round trip,** confirmed by reading the state back.
4. **Beam current from the screen.** Abort if it is above the approved pA limit, or if the screen does not retract.
5. **Stage repeatability and backlash** (`calibration/stage_repeatability.py`). Abort if the error is more than 3 times the twin's sigma.
6. **Drift series** (`calibration/drift_series.py`).
7. **OptiSTEM C1A1, three times.** Record how well it converges.
8. **Corrector sign test.** Measure C1/A1, make one tiny write, then measure again. Abort if |A1| grows.
9. **Dose series** on a sacrificial area (`calibration/dose_series.py`).
10. **One 3x3 map in approve mode.** Record the operator's region picks as the first reward labels.

Write the fitted values to a versioned twin calibration file. To roll back, go back to the previous file.

## 6. Strongest counter-arguments

- **Proof on the twin is circular.** The corrector reports the truth, the drift tracker recovers the twin's own drift model, and the scorer finds the bright particles the twin draws. All of this shows the logic works. None of it shows accuracy or "optimal results".
- **Thermo may already provide most of this.** OptiSTEM autofocus is already called, and Velox may offer drift-corrected frame integration. A home-built corrector loop or drift tracker could be wasted work, or could fight the vendor's tools.
- **Each new automated step widens the safety surface.** `safety/policy.py:57` already treats autofocus as motion. Corrector writes, image-shift loops and current changes all need limits, undo, and an approval story.
- **Six deliverables plus "optimal" has no natural stopping point.** Without the definitions in Section 3, this work never ends.

## 7. What you may not be seeing

- First light will fail on AutoScript call correctness and an assumed 50 pA beam current long before it needs any autonomy.
- The hackathon setting (`time_scale: 0`) quietly breaks drift fitting and revisits, because the Mapper and the twin run on different clocks.
- On the Spectra, two systems control C1 and A1 (OptiSTEM and CEOS), and the sign of a correction is unverified. A wrong sign doubles the aberration.
- The Mapper's own microscope calls go below the MCP gate. A dry-run or approval added only at the MCP layer would miss stage moves.
- The twin's survey images cannot show the direction of astigmatism, so stig is only partly testable before first light.
