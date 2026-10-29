# Related work: STEM digital twins, instrument agents and control frameworks

A survey made on 2026-10-28 to decide what asyncroscopy should borrow. Every repository below was checked on GitHub or GitLab (license, last activity) on that date. Licenses matter here: asyncroscopy is BSD/MIT-style, so GPL code may be run as a separate process, used as a design reference, or used offline to generate data that we ship, but not imported or ported.

## Digital twins and simulators

What our twin (`asyncroscopy/instruments/electron_microscope/twin/`) already models: a stage with bias, backlash and settling; drift; scan distortion; dose with contamination and damage; a probe built from aberrations by pyTEMlib; Poisson and read noise; EDS with a Kramers continuum, escape and sum peaks and dead time. What it lacks: a Ronchigram and aberrations that drift; diffraction, tilt and 4D-STEM; EELS; absorption and thickness in EDS; a detailed camera model.

| Name | License | What it simulates | Use for us |
|---|---|---|---|
| [directelectron/de-twin](https://github.com/directelectron/de-twin) | MIT | Whole TEM: C1–C5 aberrations, corrector drift and tuning, crystals that respond to tilt, SAED/CBED, 4D-STEM, camera dark/gain/defects/saturation, a noise-free "truth" output | Closest analogue; design reference, code may be ported |
| [pycroscopy/DTMicroscope](https://github.com/pycroscopy/DTMicroscope) | MIT | STEM twin replaying real datasets; Ronchigram via pyTEMlib; AFM/STM twins | Our own team's twin (pycroscopy); the first place to share or reuse models |
| [sea-ecosystem/rayTEM](https://github.com/sea-ecosystem/rayTEM), ORNL CNMS ([arXiv 2607.29411](https://arxiv.org/abs/2607.29411), "A versatile generalized digital twin for Electron Microscopy") | **no license** (repo); paper CC BY 4.0 | Whole-column physics used on ORNL's Nion: first-order ray-transfer matrices through every lens (Larmor rotation, quadrupoles, apertures), lens current to strength, convergence angle / camera length / rotation from lens settings and an inverse fit for a target mode, beam envelope and aperture-limited current, Krivanek C_{n,m} aberrations; talks to Nion AS2 over HTTP, no MCP; no damage, drift, scan distortion, detector or EDS model | Complements ours (column vs. sample and scan). Ideas only, reimplemented from the paper, until the authors add a license; its calibrated instrument models are private |
| [pycroscopy/pyTEMlib](https://github.com/pycroscopy/pyTEMlib) | MIT | Probe, Ronchigram, EELS edges and cross-sections, X-ray tables | Already a dependency |
| [h-walk/PySlice](https://github.com/h-walk/PySlice) | MIT | Multislice from MD trajectories, 4D-STEM, vibrational EELS | Possible dependency for realistic patches |
| [abTEM/abTEM](https://github.com/abTEM/abTEM), [prism-em/prismatic](https://github.com/prism-em/prismatic), [MuSTEM](https://github.com/HamishGBrown/MuSTEM), [Dr. Probe](https://github.com/ju-bar/drprobe_clt) | GPL-3.0 | Multislice / PRISM image and diffraction simulation | Offline generation of data we ship; not imported |
| [nion-software/nionswift-usim](https://github.com/nion-software/nionswift-usim), [TemGym](https://github.com/TemGym/TemGym), [hyperspy/exspy](https://github.com/hyperspy/exspy) | GPL-3.0 | STEM simulator for Nion Swift; column ray tracing; EDS/EELS models | Ideas only |
| [usnistgov/DTSA-II](https://github.com/usnistgov/DTSA-II) | no SPDX license detected | Physically based EDS (absorption, fluorescence, detector efficiency) | Equations only, after confirming the license |

## Agents that drive instruments

| Name | License | What it does | Use for us |
|---|---|---|---|
| [M3RG-IITD/AILA](https://github.com/M3RG-IITD/AILA) (Nat. Commun. 2025) | MIT | LangGraph agents on an AFM; AFMBench, 100 tasks scored 1/0.5/0; names "sleepwalking" (unrequested actions) | Benchmark format and the sleepwalking metric |
| TEM Agent, LBNL ([arXiv 2511.08819](https://arxiv.org/abs/2511.08819)) | no code | LLM drives TEM subsystems, data and HPC through MCP tools | Closest architecture to ours |
| Hierarchical SPM automation ([arXiv 2609.04015](https://arxiv.org/abs/2609.04015)) | no code | Agent sets the goal, deterministic algorithms pick points and stop when evidence stops changing | Pattern for an exploration skill |
| [ziatdinovmax/gpax](https://github.com/ziatdinovmax/gpax), [pycroscopy/atomai](https://github.com/pycroscopy/atomai) | MIT | Gaussian-process / Bayesian-optimisation active learning; deep learning for microscopy images | Candidate dependencies for exploration and scoring |
| [caidish/instrMCP](https://github.com/caidish/instrMCP) | MIT | MCP over QCoDeS: read-only safe mode, scoped allow, audit log | Scoped approval grants |
| VISION, BNL ([arXiv 2412.18161](https://arxiv.org/abs/2412.18161)) | paper | LLM modules drive Bluesky at a beamline | Reference |
| [Coscientist](https://github.com/gomesgroup/coscientist), [ChemCrow](https://github.com/ur-whitelab/chemcrow-public) | none declared / MIT | Tool-using chemistry agents with error-and-retry loops | Verify-and-retry pattern |

## Instrument-control and data frameworks

| Name | License | Relevant feature | Use for us |
|---|---|---|---|
| [bluesky/bluesky](https://github.com/bluesky/bluesky) | BSD-3 | Plans, pause/resume, suspenders, `check_limits` | Suspender and dry-run patterns |
| [bluesky/ophyd-async](https://github.com/bluesky/ophyd-async) | BSD-3 | Tango backend, simulated devices | Future path to run Bluesky plans on our devices |
| [bluesky/event-model](https://github.com/bluesky/event-model), [bluesky/tiled](https://github.com/bluesky/tiled) | BSD-3 | Standard run documents; data service (already used) | Provenance documents (roadmap) |
| [pyvisa/pyvisa-sim](https://github.com/pyvisa/pyvisa-sim), [microsoft/Qcodes](https://github.com/microsoft/Qcodes) | MIT | Simulated instruments defined in YAML | YAML-defined twin devices |
| [pymeasure/pymeasure](https://github.com/pymeasure/pymeasure), [PyLabRobot](https://github.com/PyLabRobot/pylabrobot), [caproto](https://github.com/caproto/caproto) | MIT / BSD-3 | Declarative drivers and protocol tests; swappable simulator backends; few-line simulated IOCs | Patterns for new-instrument docs |
| Sardana / Taurus ([GitLab](https://gitlab.com/sardana-org/sardana)), Qudi, Nion instrumentation-kit | LGPL-3.0 / GPL-3.0 | Tango macro and GUI stack; lab frameworks | References only |

## What we took, and what is on the roadmap

See the "Adopted from related work" section of `docs/athena/wg2_traceability.md` for what was built and tested. Not done yet, in rough order of value: a Bayesian-optimisation exploration skill (gpax); EELS and a Ronchigram with drifting aberrations in the twin (pyTEMlib, de-twin design); camera defects and a noise-free truth output (de-twin); event-model run documents into Tiled; crystal tilt, diffraction and 4D-STEM in the twin (patterns generated offline with a GPL simulator and shipped as data).
