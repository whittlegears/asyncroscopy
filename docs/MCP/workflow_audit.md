# Workflow audit: which procedures are LangGraph graphs

Status on branch `feat/sample-mapping`, 2026-11-28. This page lists every
workflow in the repository, whether it runs as a LangGraph graph, and what is
wrong with the graph code that exists.

## How the agent layer works

- `asyncroscopy/mcp/agent.py` builds the agents.
  - One agent: a single `langchain.agents.create_agent` executor.
  - Two or more agents: a hand-built `StateGraph`. A supervisor node picks the
    next worker and gets an edge back from each worker. It asks for structured
    output limited to the agent names, and parses free text when the model cannot.
- `asyncroscopy/mcp/agent_mcp_server.py` serves the agents over MCP:
  `query_agent`, `spawn_agent`, `list_agents`, `list_agent_tools`, `complete`
  and `set_max_steps`, plus `list_workflows`, `start_workflow`,
  `resume_workflow` and `workflow_status` for the graphs in
  `asyncroscopy/agent_workflows` (listed in `WORKFLOW_MODULES`).
- Tools come from the instrument MCP bridge through `MultiServerMCPClient`.
  They are named `{Class}_{command}`, and only Tango commands are exposed, not
  attributes. `blocked_functions` in `configs/mcp*.yaml` hides commands, and
  `needs_approval` holds them for an operator.
- LangGraph and LangChain are an optional extra (`uv sync --extra agent`). They
  are pinned in `uv.lock`: langgraph 1.2.9, langchain 1.3.12.

## Inventory

| Workflow | Where it lives | Graph today | Should be a graph |
|---|---|---|---|
| Free-form request (supervisor + workers) | `mcp/agent.py` | yes | yes |
| Single-agent request | `mcp/agent.py` | yes (`create_agent`) | yes |
| `complete` (one model step) | `mcp/agent.py` | no | no: the caller drives the loop |
| Sample map: acquire, stitch, score, review | `agent_workflows/sample_map.py` + pyTEMlib analysis server | **yes** (this branch) | yes. Served by the agent server as `start_workflow("sample_map", ...)`; pauses before the stage moves and for review |
| CEOS tableau correction loop | `agent_workflows/correction.py` (from notebook 01) | **yes** | yes. Pauses before every correction; steps fit the corrector's `max_step` and `max_total_since_tableau`. The corrector has no readback, so a restore undoes the applied corrections and re-measures |
| Aberration hill climb (C1/A1) | `agent_workflows/aberration_climb.py` + `climb.py` (from notebooks 12) | **yes** | yes. One approval before the climb; undoes every correction on failure. Open: a C1 climb can exceed `max_total_since_tableau`, and then its undo is refused until a new tableau |
| Stage-Z hill climb | `agent_workflows/stage_z_focus.py` + `climb.py` (from notebook 14) | **yes** | yes. Moves z with `move_stage` (the notebook set the `stage.z` attribute, which MCP cannot reach); returns to the start on failure |
| MAPED beam-tilt series | `agent_workflows/maped.py` (from notebook 15) | **yes** | yes. One scan per graph step, so a checkpointer resumes without repeating scans; restores the tilt. quantEM analysis is an optional hook |
| Alpha-tilt diffraction map with tracking | `agent_workflows/tilt_series.py` (from notebook 16) | **yes** | yes. Tracks with `analysis.drift.correlation_shift`; restores stage, image shift and field of view on every path |
| Drift and speed metric | notebook 07, `analysis.drift` | partly (drift is a tool) | optional. It sets scan attributes MCP cannot reach |
| Segmentation | `mcp/segment.py`, notebook 06 | no | no: two tool calls |
| Autofocus | `auto_focus` | no | no: one tool |
| AberrationsBO, gold-twin sweeps | notebooks | no | twin-only: they use setters the MCP configs block |
| Compressed sensing, BlobNet, particle statistics | notebooks 10, 20, 09 | no | no: they should become tools, not graphs |

## Problems in the graph code, and what this branch did

| # | Problem | Status |
|---|---|---|
| 1 | A failed tool call (a Tango `DevFailed` arrives as a `ToolException`) ended the whole query. The model never saw the error. | **Fixed.** Middleware returns the error to the model with no retry, so an instrument command cannot run twice. Tested with the real stack in `tests/test_agent_graph_real.py`. |
| 2 | Worker agents had no step limit (the `create_agent` default is about 10,000 steps); `max_steps` bounded only the supervisor. | **Fixed:** workers get `recursion_limit=max_steps`. |
| 3 | Worker replies went back to the supervisor as user messages. | **Fixed:** they are now the worker's own `AIMessage`. |
| 4 | The test stubs replaced LangGraph in every test, so no real graph was ever compiled. | **Fixed:** the stubs load only when the real stack is missing (`ASYNCROSCOPY_STUB_LLM_STACK=1` forces them). All 56 agent tests pass both ways. |
| 5 | No checkpointer and no approval step: `move_stage` or `correct_aberration` runs with no person in the loop, and a run cannot be resumed. | **Fixed.** Workflows use a checkpointer and `interrupt()`. The instrument bridge holds commands listed in `mcp.needs_approval` until an operator approves them on a separate operator endpoint (`mcp.role: operator`, same `mcp.approvals_db`); agents cannot approve their own calls. On for the twin (`configs/mcp_dt.yaml` + `configs/mcp_dt_operator.yaml`), commented out in `configs/mcp.yaml` until tried on the microscope. `asyncroscopy/mcp/approvals.py`, `tests/test_approvals.py`. |
| 6 | The supervisor routes by parsing free text and silently falls back on bad output. | **Fixed.** Routing asks for structured output limited to the valid agent names, and falls back to parsing text, with the fallback written to the transcript, when the model cannot. Not yet tried with the local Gemma model. |
| 7 | Each worker sees only its task, not earlier results (such as an image key from another worker). | Open. |
| 8 | The graph and every executor are rebuilt on every query. | Open: cache them per agent roster. |
| 9 | When the step limit is hit, `query` returns only the error text and drops the partial transcript. | **Fixed:** the transcript comes back with the error. |

## Pattern for turning a notebook into a graph

- One module per procedure. Each takes tools (the same MCP tools the agents
  get, so `blocked_functions` still applies) and returns a compiled
  `StateGraph`.
- Keep the loop deterministic: the model picks parameters, plain nodes run the
  steps. Never one model call per tile, focus step or tilt.
- Pause with `interrupt()` before anything that moves the instrument, and
  compile with a checkpointer so the run can be resumed.
- Pass large data (manifests, images) between tools inside a node, never
  through the model.
- Test offline: fake tools as async functions over a synthetic response, and
  resume through `Command(resume=...)`.

`asyncroscopy/agent_workflows/` follows this pattern: `tools.ToolBox` finds a
tool by command name whatever its instrument prefix, `runner.WorkflowRunner`
keeps paused runs so an MCP client can resume them, and `sample_map.py` is the
first graph. `tests/test_agent_workflows.py` runs it offline against the real
pyTEMlib server.

The long-running branch already has graphs built this way under
`asyncroscopy/agents/graphs/` (`mapping`, `general`, `team`, `goal`), and
`langgraph.json` registers them.


## Still open

- **The pyTEMlib scan fit.** On small atomic-resolution grids, pyTEMlib fits a
  scan turn that isn't there. The sample-map workflow works around it by passing
  the identity `scan_linear` for calibrated grids under 25 tiles.
  - A fix is on a local branch of the pyTEMlib fork (`fix/scan-linear-small-grids`,
    commits `05a7694`, `4fd2ad3` and `e647d93`, worktree `../pyTEMlib-scan-linear`),
    not pushed. Small lattice maps no longer get the false turn, and
    `estimate_scan_linear` no longer changes its caller's mask of matched tiles.
  - One existing pyTEMlib test, drift recovery on sparse texture, still misses
    its limit by 0.04 px (1.53 px against 1.49). All of that gap comes from the
    lattice guard refusing 10 pairs (1.46 px with the guard off); the scan-fit
    changes do not move it. Sparse texture can show several near-equal side
    peaks, and the mirror-peak and peak-count tests tried so far either missed
    lattice matches in small windows or did not separate the two. Keeping the
    suspect matches for placement (and only out of the scan fit) passes the
    drift test (1.46 px) but misplaces lattice tiles by 11-42 px, so on a
    lattice they must be refused. Finishing it needs a periodicity test that
    tells sparse texture from a lattice, or a decision on how much drift
    accuracy to trade for lattice safety.
- **Not run on a microscope or with Gemma.** Everything here is tested on the
  digital twins and offline only.
  - Structured routing has not been tried with the local Gemma model; Ollama is
    not installed on this machine.
  - The approvals and the workflows have not run against the real instrument.
- **Recursion limit.** `WorkflowRunner` has a limit of 100 graph steps. The
  workflows cap their rounds to stay under it (correction 15, climbs 60, tilt
  series 90).

## Closed in the last round

- **Corrector total limit.** The aberration climb checks
  `max_total_since_tableau` before every move: it moves only when the move and
  the way back to the start both fit. Otherwise it stops normally, and goes back
  to the start if the best point is out of reach. Test:
  `test_c1_climb_stays_inside_the_total_limit_since_the_tableau`.
- **Settings that are attributes.** `read_attribute` and `write_attribute` on
  the bridge reach Tango attributes listed under `mcp.attributes`, such as scan
  and camera settings. A write can be held for approval as `write_<attribute>`.
  On for the twin; commented out in `configs/mcp.yaml`. Test:
  `tests/test_mcp_attributes.py`.
- **The Tiled path.** The sample-map workflow runs against a real in-process
  Tiled server: tiles go into Tiled, `collect_map_run` gathers them, and
  pyTEMlib stitches from Tiled. Test:
  `test_full_run_through_tiled_uses_collect_map_run`.
