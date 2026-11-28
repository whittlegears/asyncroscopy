# Agent skills for the calibration procedures

Each folder holds one `SKILL.md` in the agentskills layout SciAgentGUI reads (`name`,
`description`, `requires_tools` front matter). They teach an agent when and how to
use the tools of `asyncroscopy/mcp/procedures.py` (see `configs/procedures.yaml`)
and the pyTEMlib analysis MCP server.

| Skill | Uses | Checked on |
|---|---|---|
| `stem-focus` | `sweep_defocus`, `focus_stage_z` | tilt digital twin |
| `aberration-tuning` | `tune_aberration` | polycrystalline gold digital twin |
| `active-learning-sampling` | `dkl_suggest_locations` on the ML server | unit tests on images whose value depends on local structure (predicted map correlates 0.87 with the truth from 10 measurements); live only for reading a twin acquisition by data key, as no twin offers a spectroscopic value per position |
| `drift-tracked-tilt-series` | instrument tools + `measure_image_shift` | tilt digital twin: direction calibration within 1 %, residual drift at most half a pixel over 30 degrees |

The advice on metrics and pixel size comes from those twin runs. On the gold twin,
after fixing its noise scaling, tuning A1 from (6.0, -4.0) nm with `edge_sharpness`
ended at (-0.31, +0.08) nm; at 1.56 A pixels A1 barely changed the image, which is why
`tune_aberration` refuses pixels coarser than the configured limit. Check these on real
hardware before relying on them.
