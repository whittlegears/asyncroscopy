# Deterministic workflows

Workflows are LangGraph `StateGraph`s whose nodes and edges are fixed in Python.
They are the right tool when a procedure must be reproducible and reviewable:
the model (if any) never decides which acquisition happens next.

## The example: `image_eds_survey`

`asyncroscopy/agent/graphs/workflows/image_eds_survey.py`

```
START -> acquire_image -> acquire_spectrum -> read_back -> summarize -> END
                 \\                 \\
                  +-- (on error) ---+---------> summarize
```

- `acquire_image` calls `*_acquire_scanned_image` with `detector_list=[state["detector"]]`.
- `acquire_spectrum` calls `*_acquire_spectrum` with `detector_name=state["spectrum_detector"]`.
- `read_back` calls `get_data_from_key` for both keys.
- `summarize` builds a plain-text report from the state (`deterministic_summary`)
  and, when a model is given, asks it to rephrase without changing numbers.

Every node catches its own exceptions and appends to `state["errors"]`; the
conditional edges route straight to `summarize`, so a run always ends with a
report.

```python
from asyncroscopy.agent.graphs.workflows.image_eds_survey import build_image_eds_survey_graph
from asyncroscopy.agent.tools import load_mcp_tools

tools = await load_mcp_tools("http://127.0.0.1:8000/mcp")
graph = build_image_eds_survey_graph(tools, model=None)     # fully deterministic
result = await graph.ainvoke({"detector": "haadf", "spectrum_detector": "eds"})
print(result["summary"])
```

The LLM Tango device runs the same graph with
`RunWorkflow('{"name": "image_eds_survey", "input": {"detector": "haadf"}}')`.

## Writing a new workflow

1. Create `asyncroscopy/agent/graphs/workflows/<name>.py` with a state
   `TypedDict` (add it to `asyncroscopy/agent/state.py` if it is reusable) and a
   `build_<name>_graph(tools, model=None, **options)` factory that returns
   `builder.compile()`.
2. Resolve tools once, up front, with `find_tool(tools, "<glob>")` so a missing
   tool fails at build time, not mid-acquisition.
3. Call tools with `invoke_tool_strict(tool, args)` inside nodes: it raises on
   tool errors instead of returning an error string (which is what the ReAct
   agent prefers). Use `tool_result_text` / `tool_result_dict` on the result.
4. Record failures in the state and route to a terminal reporting node.
5. Register it: `register_workflow("<name>", build_<name>_graph)` at the bottom
   of the module and import the module in
   `asyncroscopy/agent/graphs/workflows/__init__.py::_ensure_builtin_workflows`.
6. Add a Studio factory in `asyncroscopy/agent/studio.py` and an entry in
   `langgraph.json`.
7. Test it with the fake tools in `tests/_fakes.py` or an in-memory `FastMCP`
   server (see `tests/test_agent_graphs.py`).

Keep any model call optional and confined to reporting; if the model must make
a decision, that is a ReAct or supervisor job, not a workflow.
