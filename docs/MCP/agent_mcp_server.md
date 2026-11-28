# Agent MCP server

The agent MCP server serves an LLM agent swarm as MCP tools. The swarm takes its own tools from one or more [MCP servers](mcp_server.md). `asyncroscopy/mcp/agent.py` holds the swarm, and `asyncroscopy/mcp/agent_mcp_server.py` wraps it in FastMCP.

## Start it

Start the stack and the MCP server first. Then:

```bash
uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml
```

The default endpoint is `http://127.0.0.1:8002/mcp`. To prompt the swarm in the terminal instead of serving it:

```bash
uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml --interactive
```

## Config

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

- `mcp_urls` is the only link to the instruments. List one URL per stack to give one agent several instruments. See [scaling](../scaling.md).
- `use_init_chat_model: true` starts the model through LangChain's `init_chat_model` with `model` and `model_provider`, instead of a local Ollama model.
- `tools` takes glob patterns against the inherited tool names, for example `["*spectrum"]`.
- If the agent runs on another computer, point `mcp_urls` at the MCP computer. Set `http_host` to `0.0.0.0` when clients on other computers must connect.

If the model does not start, the server still serves MCP, and `query_agent` reports the error.

## Tools

| Tool | Signature | What it does |
| --- | --- | --- |
| `query_agent` | `(prompt: str, include_transcript: bool) -> str` | Runs the swarm and returns its response. |
| `spawn_agent` | `(config: str) -> str` | Adds a worker. Returns its name, or `""` if it fails. |
| `list_agents` | `() -> list[str]` | Names of the agents in the swarm. |
| `list_agent_tools` | `() -> str` | JSON list of the inherited MCP tool names. |
| `complete` | `(request_json: str) -> str` | One OpenAI-compatible step. |
| `set_max_steps` | `(max_steps: int) -> int` | Sets the LangGraph recursion limit per query. |

`spawn_agent` takes `name` and `system_prompt`, and optionally `tools`, `model` and `description`.

By default, `query_agent` returns the whole run: the supervisor routing, each tool call with its result, and the text, then `FINAL ANSWER:`. Pass `include_transcript=False` to get only the answer.

Optional dependencies:

```bash
uv sync --extra agent                       # LangChain and OpenAI
uv sync --extra agent --extra ollama        # local models through Ollama
```
