# Model providers

`asyncroscopy.agent.models.build_chat_model(settings)` returns a LangChain chat
model from an `AgentSettings` object. Three kinds of provider are supported.

## Ollama (local model)

```python
from asyncroscopy.agent.config import AgentSettings
from asyncroscopy.agent.models import build_chat_model

model = build_chat_model(AgentSettings(provider="ollama", model="gemma4:31b"))
```

Requires `uv sync --extra agent --extra ollama` and an Ollama install with a
model that supports tool calling (`gemma4`, `qwen3`, `llama3.1`, ...). Set
`base_url` when Ollama runs on another machine. The LLM Tango device starts
`ollama serve` automatically if it is not running.

## API-key providers (OpenAI, Anthropic, Google, ...)

```python
model = build_chat_model(AgentSettings(provider="openai", model="gpt-4o-mini"))
model = build_chat_model(AgentSettings(provider="anthropic", model="claude-sonnet-4-5"))
```

Any provider accepted by `langchain.chat_models.init_chat_model` works. The key
is read from the environment variable named by `api_key_env` (default:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, ...). A `.env` file at
the repository root is loaded automatically when `python-dotenv` is installed:

```bash
cp .env.example .env
echo 'OPENAI_API_KEY=sk-...' >> .env
```

Keys are never accepted in YAML configs: `run_llm.py` refuses an `api_key`
field and `AgentSettings.from_mapping` raises on it. Configure the LLM device
with `api_key_env: OPENAI_API_KEY` instead (see `configs/openai-llm.yaml`).

## The LLM Tango device (`TangoChatModel`)

```python
from asyncroscopy.agent.models import TangoChatModel

model = TangoChatModel(device_name="asyncroscopy/llm/default", timeout_ms=300_000)
```

Each model call becomes one `Complete` command on the device, so a notebook or
Studio on your laptop can run graphs while inference happens on the LLM
computer. Limitations: no token streaming (a whole turn arrives at once) and no
structured output. The device itself refuses `provider="tango"` to avoid
recursion.

## Settings sources

`AgentSettings` can be built from:

- keyword arguments (notebooks),
- `AgentSettings.from_env()` reading `ASYNCROSCOPY_AGENT_PROVIDER`,
  `ASYNCROSCOPY_AGENT_MODEL`, `ASYNCROSCOPY_AGENT_MCP_URL`,
  `ASYNCROSCOPY_AGENT_SKILLS_DIRS`, ... (used by LangGraph Studio),
- `AgentSettings.from_mapping(yaml_dict)` which also accepts the legacy
  `model_provider` / `model_name` keys of the LLM device configs.
