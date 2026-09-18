# LLM Tango device

`asyncroscopy/mcp/llm.py` wraps the agent layer in a Tango device
(`asyncroscopy/llm/default`) so other computers on the Tango network, and
external clients such as SciAgentGUI, can use the model without installing
LangChain.

## Start

```bash
uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml            # Ollama
uv run startup_scripts/run_llm.py --yaml configs/openai-llm.yaml           # API key
uv run startup_scripts/run_llm.py --yaml configs/gemma-llm.yaml --interactive
```

`run_llm.py` registers the device, writes its properties from the YAML, loads
`.env`, and launches `python -m asyncroscopy.mcp.llm` under the process manager.

## Configuration (YAML -> device properties)

| key | property | meaning |
|-----|----------|---------|
| `tango.host`, `tango.port` | | Tango database |
| `mcp_url` | `mcp_url` | MCP server to load tools from at start-up |
| `model_provider` | `model_provider` | `ollama` or an `init_chat_model` provider |
| `model_name` | `model_name` (and legacy `ollama_model`) | model name |
| `api_key_env` | `api_key_env` | name of the environment variable holding the key |
| `base_url` | `base_url` | Ollama host or OpenAI-compatible endpoint |
| `skills_dirs` | `skills_dirs` | folders with `SKILL.md` files |
| `temperature`, `use_init_chat_model` | same | model options |
| `startup_agents` | `startup_agents` | worker agents (JSON per entry) |

An `api_key` key in the YAML is rejected; keys stay in the environment.

## Commands and attributes

| name | type | purpose |
|------|------|---------|
| `Query(prompt) -> str` | command | run the worker agent(s); one agent = skills-aware ReAct, several = supervisor swarm |
| `Complete(json) -> json` | command | OpenAI-style single-step completion (`{"messages": [...], "tools": [...]}` -> `{"message": {...}}`); no tools are executed |
| `ConnectMCP(json) -> bool` | command | `{"url": ..., "transport": "streamable_http"}`; adds the server's tools |
| `SpawnAgent(json) -> bool` | command | `{"name", "system_prompt", "tools": [globs], "description"}` |
| `RunWorkflow(json) -> json` | command | `{"name": "image_eds_survey", "input": {...}, "use_model": true}`; returns the final state |
| `ReloadSkills() -> int` | command | re-scan the skills folders |
| `max_steps` | attribute (rw) | LangGraph recursion limit for `Query` |
| `agents` | attribute | worker names |
| `tools` | attribute | JSON list of tool names (used by SciAgentGUI's `/health`) |
| `skills` | attribute | JSON list of `{name, description, version}` |

`Complete` is what `asyncroscopy.agent.models.TangoChatModel` calls, so a
notebook can run graphs locally while the device does the inference.
