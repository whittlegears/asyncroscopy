# Asyncroscopy Agent MCP Server

The agent MCP server exposes the LLM agent swarm as MCP tools. It is how clients drive the LLM.

`asyncroscopy/agents/graphs/swarm.py` holds the swarm, beside the other agent graphs, and
`asyncroscopy/mcp/agent_mcp_server.py` wraps it in FastMCP. The server imports the agent layer only
when it starts, so the `mcp` package itself never depends on `agents`.
The default model comes from the agent profile (`asyncroscopy/agents/profiles/capable.yaml`, gemma4:31b).
`query_agent`'s `graph` argument is an enum of the graphs (`""` for the swarm, `mapping`, `general`,
`team`), so clients can list them from the tool's input schema.

## Start It

Start the device stack and the instrument MCP server first:

```bash
uv run startup_scripts/run_servers.py --yaml configs/Spectra300.yaml
uv run startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

Then start the agent in another terminal or on the agent computer:

```bash
uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml
```

The default endpoint is:

```text
http://127.0.0.1:8002/mcp
```

To prompt the swarm from a terminal instead of serving it over MCP:

```bash
uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml --interactive
```

If the agent runs on another computer, set `agent.mcp_urls` to the instrument MCP machine and `agent.http_host` to the agent machine's bind address. Use `0.0.0.0` when clients on other machines need to connect.

## YAML Contract

```yaml
agent:
  name: Asyncroscopy_Agent_MCP
  transport: streamable-http
  http_host: 0.0.0.0
  http_port: 8002
  model: "gemma4:31b"
  model_provider: "ollama"
  max_steps: 10
  use_init_chat_model: false
  mcp_urls:
    - "http://127.0.0.1:8000/mcp"
  startup_agents:
    - name: "base"
      system_prompt: "You are a helpful assistant."
      tools: ["list_devices"]
```

Workflows (`asyncroscopy/workflows`, for example the `map_grid` map) run in the agent's own process, and only there; `workflows: false` (under `agent:`, or `--no-workflows`) switches them off: `list_workflows`, `run_workflow`, `workflow_status`, `stop_workflow` and the rest are offered by this server and to the swarm's agents (give an agent `tools: ["*workflow*"]`). A map then needs three processes: the instrument's MCP endpoint, this agent server, and the pyTEMlib MCP server that stitches (`startup_scripts/run_pytemlib.py`). Workflows drive the first of `mcp_urls`. `workflows_hardware: true` also runs them on a stack that is not simulated; it needs `http_host: 127.0.0.1`, because this server has no login.

A person reviews, approves and rejects workflows through a second, separate endpoint of this module, the workflow
operator endpoint (`agent.endpoint_role: operator`, `configs/workflow_operator.yaml`, port 8004). It has no swarm and
no model, and the agent server above never offers its tools; [Workflow approval](workflow_approval.md) is its contract.

`mcp_urls` is the only link to the instrument stack. `use_init_chat_model: true` initializes through LangChain's `init_chat_model` using `model` and `model_provider` instead of a local Ollama model.

## Tools

| Tool | Signature | Purpose |
| --- | --- | --- |
| `query_agent` | `(prompt: str, include_transcript: bool, graph: str, goal: bool, model: str) -> str` | Run the swarm (or the named graph) and return its response. `model` runs it on that model, a name under the server's provider (Ollama by default), instead of the server's own; SciAgentGUI sends the Ollama model chosen there. |
| `spawn_agent` | `(config: str) -> str` | Add a worker; returns its name, or `""` on failure. |
| `list_agents` | `() -> list[str]` | Names of the agents in the swarm. |
| `list_agent_tools` | `() -> str` | JSON list of the inherited MCP tool names. |
| `complete` | `(request_json: str) -> str` | OpenAI-compatible single step. |
| `set_max_steps` | `(max_steps: int) -> int` | LangGraph recursion limit per query. |

`spawn_agent` takes `name`, `system_prompt`, and optionally `tools` (glob patterns matched against inherited tool names, e.g. `["*spectrum"]`), `model`, and `description`.

`query_agent` returns the whole run by default: supervisor routing, each tool call and its result, and the generated text, followed by `FINAL ANSWER:`. Pass `include_transcript=False` to get just the answer. A server whose own model failed to load at startup still connects its tools, so a request that names a model runs.