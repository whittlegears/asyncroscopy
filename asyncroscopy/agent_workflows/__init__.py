"""Instrument procedures written as LangGraph graphs.

Each module builds one graph from a set of tools (the same MCP tools the agent
swarm gets, so the server-side ``blocked_functions`` boundary still applies).
The rules that every graph here follows:

- Deterministic loops are plain nodes. The model never runs once per tile,
  per focus step or per tilt; it picks parameters and reads results.
- Anything that moves the instrument waits for an operator first, through
  ``interrupt()``. A graph needs a checkpointer to pause and resume.
- Large data (manifests, images) moves between tools inside a node and never
  through the model's context.
- Every graph is testable offline with fake tools (see tests/test_agent_workflows.py).

``tools.ToolBox`` finds tools by the command name whatever instrument class
prefixes it, and normalises MCP results to Python values.
``runner.WorkflowRunner`` keeps graphs and their paused threads so a client
(or the agent MCP server) can start a run and resume it after an approval.
"""

# Workflow modules the agent server offers, by module name. Each has build_graph,
# WORKFLOW (name, what, inputs) and missing_tools.
WORKFLOW_MODULES = ("sample_map", "correction", "tilt_series", "maped")
