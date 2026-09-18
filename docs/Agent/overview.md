# AI agent overview

`asyncroscopy.agent` is the LangGraph layer that sits between a language model
and the MCP server. The same graphs run in three places:

```{mermaid}
flowchart LR
    NB[Notebook<br/>11_Test_AI_Agent.ipynb] --> AG
    ST[LangGraph Studio<br/>langgraph dev] --> AG
    DEV[LLM Tango device<br/>asyncroscopy/llm/default] --> AG
    AG[asyncroscopy.agent<br/>graphs + skills + models] -->|fastmcp.Client| MCP[MCP server<br/>run_mcp.py]
    MCP -->|Tango| INST[Instrument devices<br/>DigitalTwin / Spectra300]
    MCP -->|Tiled| DATA[(Saved acquisitions)]
    AG -.->|Ollama / API key / TangoChatModel| LLM[(Chat model)]
```

## Two kinds of behaviour

| | LangGraph workflows | Skills-aware ReAct agent |
|---|---|---|
| Where | `asyncroscopy/agent/graphs/workflows/` | `asyncroscopy/agent/graphs/react.py` |
| Control flow | fixed nodes and edges written in Python | the model decides which tool to call next |
| Model's role | optional, e.g. rephrase a summary | plans, calls tools, reads results |
| Context | the graph state | `skills/<name>/SKILL.md` loaded on demand |
| Use for | reproducible procedures (surveys, calibrations, scans) | open-ended requests, troubleshooting, one-off tasks |

The supervisor swarm (`graphs/supervisor.py`) combines the two: a routing node
delegates sub-tasks to worker ReAct agents that each see a subset of the tools.

## Package layout

```
asyncroscopy/agent/
├── config.py       AgentSettings, Agent, .env loading (stdlib only)
├── models.py       build_chat_model(): Ollama, API-key providers, TangoChatModel
├── tools.py        load_mcp_tools() over fastmcp.Client, glob filtering, result helpers
├── streaming.py    stream_agent(): token/tool-call printer
├── studio.py       graph factories referenced by langgraph.json
├── graphs/
│   ├── react.py                 skills-aware single agent (langchain create_agent)
│   ├── supervisor.py            supervisor/worker swarm
│   └── workflows/image_eds_survey.py   deterministic example
└── skills/         SKILL.md parser, SkillRegistry (SQLite FTS5), skill tools
```

Heavy dependencies are optional extras:

```bash
uv sync --extra agent            # langchain, langgraph, openai + anthropic clients
uv sync --extra agent --extra ollama   # + langchain-ollama
uv sync --extra agent --extra studio   # + langgraph-cli for `langgraph dev`
```

`langchain-mcp-adapters` is intentionally not used: it pins `mcp<2` while the
MCP server runs on fastmcp 4 (`mcp>=2`). `asyncroscopy.agent.tools` wraps MCP
tools with `fastmcp.Client` instead, which also allows in-memory servers in tests.

## Quick start

```bash
uv run startup_scripts/run_servers.py --yaml configs/DigitalTwin.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml
cp .env.example .env
uv run jupyter lab notebooks/11_Test_AI_Agent.ipynb
```

Then pick a model in the notebook (see [Model providers](model_providers.md)).

## Tests

```bash
uv run pytest tests/test_agent_skills.py                     # always runs
uv run pytest tests/test_agent_graphs.py tests/test_agent_models.py   # need --extra agent
uv run pytest tests/test_llm_device.py
```
