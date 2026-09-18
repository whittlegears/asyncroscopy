# LangGraph Studio

[LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)
is a browser UI for visualising graphs, running them with custom inputs, and
inspecting or replaying the state after every node. It talks to a local
LangGraph server started by `langgraph dev`.

## Start it

```bash
uv sync --extra agent --extra studio        # once
cp .env.example .env                        # provider, model, MCP URL, API keys
uv run langgraph dev                        # opens Studio in the browser
```

The server reads `langgraph.json` at the repository root:

```json
{
  "dependencies": ["."],
  "graphs": {
    "react_agent": "./asyncroscopy/agent/studio.py:make_react_graph",
    "supervisor": "./asyncroscopy/agent/studio.py:make_supervisor_graph",
    "image_eds_survey": "./asyncroscopy/agent/studio.py:make_image_eds_survey_graph"
  },
  "env": ".env"
}
```

Studio needs a LangSmith account to open the UI (the local server itself does
not send data anywhere unless `LANGSMITH_API_KEY` is set). Use
`uv run langgraph dev --no-browser` and the REST API on `http://127.0.0.1:2024`
when you only need the server.

## Configuring a run

Every graph factory reads `config["configurable"]`, so in Studio's
*Configuration* panel you can override, per run:

| key | meaning |
|-----|---------|
| `provider` | `ollama`, `openai`, `anthropic`, `tango`, ... |
| `model` | model name or Tango device name |
| `mcp_url` | MCP endpoint (default `http://127.0.0.1:8000/mcp`) |
| `skills_dirs` | comma-separated skills folders |
| `temperature`, `max_steps`, `api_key_env`, `base_url` | as in `AgentSettings` |

Values not given fall back to the `ASYNCROSCOPY_AGENT_*` variables from `.env`.

## Without the microscope

If the MCP server is unreachable the factories still build the graph: the ReAct
and supervisor graphs get no MCP tools (skills tools remain), and the survey
workflow gets placeholder tools that fail with "MCP server not connected". This
lets you inspect graph structure and iterate on nodes before a beam time.

## Creating and changing nodes

Studio visualises and runs graphs; nodes themselves are Python. To add a node
or a workflow:

1. Write it under `asyncroscopy/agent/graphs/` (see [Workflows](workflows.md)).
2. Expose a factory in `asyncroscopy/agent/studio.py` and list it in
   `langgraph.json`.
3. `langgraph dev` reloads on file changes; refresh Studio to see the new graph.

## Blocking calls

The dev server flags blocking I/O inside async nodes. The factories run
settings, skills and model construction in worker threads, `TangoChatModel`
calls Tango through `asyncio.to_thread`, and the MCP tools are async. If you add
a node that calls Tango or reads files directly, wrap it the same way or start
the server with `langgraph dev --allow-blocking`.
