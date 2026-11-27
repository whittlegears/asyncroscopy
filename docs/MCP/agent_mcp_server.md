# Asyncroscopy Agent MCP Server

The agent MCP server exposes the LLM agent swarm as MCP tools. It is how clients drive the LLM.

`asyncroscopy/mcp/agent.py` holds the swarm and `asyncroscopy/mcp/agent_mcp_server.py` wraps it in FastMCP.

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

`mcp_urls` is the only link to the instrument stack. `use_init_chat_model: true` initializes through LangChain's `init_chat_model` using `model` and `model_provider` instead of a local Ollama model.

## Tools

| Tool | Signature | Purpose |
| --- | --- | --- |
| `query_agent` | `(prompt: str, include_transcript: bool) -> str` | Run the swarm and return its response. |
| `spawn_agent` | `(config: str) -> str` | Add a worker; returns its name, or `""` on failure. |
| `list_agents` | `() -> list[str]` | Names of the agents in the swarm. |
| `list_agent_tools` | `() -> str` | JSON list of the inherited MCP tool names. |
| `complete` | `(request_json: str) -> str` | OpenAI-compatible single step. |
| `set_max_steps` | `(max_steps: int) -> int` | LangGraph recursion limit per query. |

`spawn_agent` takes `name`, `system_prompt`, and optionally `tools` (glob patterns matched against inherited tool names, e.g. `["*spectrum"]`), `model`, and `description`.

`query_agent` returns the whole run by default: supervisor routing, each tool call and its result, and the generated text, followed by `FINAL ANSWER:`. Pass `include_transcript=False` to get just the answer.
## Deterministic workflows: return to a saved site

Besides `query_agent`, the agent MCP server exposes a LangGraph workflow that
runs without the LLM deciding each step:

| Tool | What it does |
|---|---|
| `save_site(name, landmark, notes)` | Images the current spot and saves the image key, stage position and field of view. |
| `calibrate_stage_mapping()` | Measures how stage moves shift the image at the current field of view. Needed once per magnification. |
| `return_to_site(name, max_rotation_deg)` | Runs go to position → survey → centre → verify (up to 3 corrections) → record. |
| `list_sites()` | Lists the saved sites with the outcome of each one's last visit. |

The workflow (`asyncroscopy/mcp/workflows/return_to_site.py`) needs no model.
It connects to the instrument MCP server(s) in `--mcp-urls-json` with the
`fastmcp` client and uses only the instrument tools and `register_images`.
It therefore works even when no LLM is running.

`return_to_site` reports `status`:
- `match` only when the image at the original magnification matches within
  2 pixels.
- `ambiguous`, `no_match`, `not_converged` or `not_calibrated` otherwise.
  These are recorded and reported, never retried.

Sites and visits live in a JSON notes file (`sites_path`, default
`outputs/sites.json`). It is the same file `skills/return-to-site/SKILL.md`
uses, so an agent following the skill by hand and the workflow share their
memory.
