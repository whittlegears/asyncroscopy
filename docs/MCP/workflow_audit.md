# Workflow audit: which procedures are LangGraph graphs

Status on branch `feat/sample-mapping`, 2026-11-28. This page lists every
workflow in the repository, whether it runs as a LangGraph graph, and what is
wrong with the graph code that exists.

## How the agent layer works

- `asyncroscopy/mcp/agent.py` builds the agents.
  - One agent: a single `langchain.agents.create_agent` executor.
  - Two or more agents: a hand-built `StateGraph`. A supervisor node picks the
    next worker and gets an edge back from each worker. It decides by parsing
    free-text JSON.
- `asyncroscopy/mcp/agent_mcp_server.py` serves the agents over MCP:
  `query_agent`, `spawn_agent`, `list_agents`, `list_agent_tools`, `complete`
  and `set_max_steps`.
- Tools come from the instrument MCP bridge through `MultiServerMCPClient`.
  They are named `{Class}_{command}`, and only Tango commands are exposed, not
  attributes. `blocked_functions` in `configs/mcp*.yaml` hides unsafe ones.
- LangGraph and LangChain are an optional extra (`uv sync --extra agent`). They
  are pinned in `uv.lock`: langgraph 1.2.9, langchain 1.3.12.

## Inventory

| Workflow | Where it lives | Graph today | Should be a graph |
|---|---|---|---|
| Free-form request (supervisor + workers) | `mcp/agent.py` | yes | yes |
| Single-agent request | `mcp/agent.py` | yes (`create_agent`) | yes |
| `complete` (one model step) | `mcp/agent.py` | no | no: the caller drives the loop |
| Sample map: acquire, stitch, find, review | `acquire_map_grid`; graph on the long-running branch | not on `main` | yes (see below) |
| CEOS tableau correction loop | notebook 01 | no | yes, first: needs an approval before each correction |
| Aberration hill climb (C1/A1) | notebooks 12 | no | yes: loop in a plain node |
| Stage-Z hill climb | notebook 14 | no | yes. It sets `stage.z`, an attribute that MCP cannot reach |
| MAPED beam-tilt series | notebook 15 | no | yes: long run, needs a checkpointer to resume and must restore the tilt on failure |
| Alpha-tilt diffraction map with tracking | notebook 16 | no | yes: tilt moves need approval |
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
| 5 | No checkpointer and no approval step: `move_stage` or `correct_aberration` runs with no person in the loop, and a run cannot be resumed. | Open. The long-running branch moves approvals to the server (`needs_approval`, operator endpoint). Port that rather than adding a second, client-side approval layer. |
| 6 | The supervisor routes by parsing free text and silently falls back on bad output. | Open: use `with_structured_output` with a `Literal` of the agent names. |
| 7 | Each worker sees only its task, not earlier results (such as an image key from another worker). | Open. |
| 8 | The graph and every executor are rebuilt on every query. | Open: cache them per agent roster. |
| 9 | When the step limit is hit, `query` returns only the error text and drops the partial transcript. | Open. |

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

The long-running branch already has graphs built this way under
`asyncroscopy/agents/graphs/` (`mapping`, `general`, `team`, `goal`), and
`langgraph.json` registers them.
