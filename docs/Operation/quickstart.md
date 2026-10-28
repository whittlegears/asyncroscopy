# Quick start: from clone to an agent mapping the digital twin

This takes a fresh machine to a running digital twin, both MCP endpoints, the pyTEMlib analysis server and a local-model agent that maps a sample. It was written while doing exactly that on Windows 11; macOS and Linux differ only where noted.

Nothing here needs a microscope, vendor software or a GPU. A GPU makes the local model faster.

## 1. Install the tools

| Tool | Windows | macOS / Linux |
| :-- | :-- | :-- |
| uv (installs Python 3.12 for you) | `winget install astral-sh.uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Git | `winget install Git.Git` | usually present |
| Ollama (local models) | `winget install Ollama.Ollama` | see ollama.com |

Then pull a model. `qwen3:8b` is the tested default (5 GB; runs partly on CPU on a 6 GB GPU); `qwen3:4b` (2.5 GB) works on smaller machines:

```bash
ollama pull qwen3:8b
```

## 2. Get the code and its dependencies

```bash
git clone https://github.com/pycroscopy/asyncroscopy.git
cd asyncroscopy
uv sync --extra agent --extra ollama
uv run pytest -q
```

The test suite starts real Tango devices in-process and takes under a minute. All tests should pass; a few are skipped on purpose.

**Windows: if `uv sync` stops with "Missing expected target directory for Python minor version link"**, uv's Python link is broken. Point uv at the interpreter directly:

```bash
uv sync --python "%APPDATA%\uv\python\cpython-3.12.<n>-windows-x86_64-none\python.exe" --extra agent --extra ollama
```

## 3. Start the stack

Use four terminals (or run each in the background), from the repository root:

```bash
# 1. Twin + data + Tiled + Mapper + Approvals. Wait for "Leave this terminal open".
uv run startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml

# 2. Agent MCP endpoint (port 8000)
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml

# 3. Operator MCP endpoint for the GUI (port 8001, this machine only)
uv run startup_scripts/run_mcp.py --yaml configs/mcp_operator.yaml

# 4. pyTEMlib analysis server: stitching, scoring, points of interest (port 8003)
uv run python -m pyTEMlib.mcp_server --tiled http://localhost:9091 --api-key secret --port 8003
```

`configs/digital_twin_mapping.yaml` simulates a real instrument's timing (`time_scale: 1.0`): stage moves take seconds. Set `time_scale: 0` in that file for fast runs. `configs/DigitalTwin.yaml` is an ideal instrument with a tiny sample, useful for unit-style experiments.

The Tiled API key `secret` is the default the DATA device starts Tiled with; set `ASYNCROSCOPY_TILED_API_KEY` to change it (and pass the same value to the pyTEMlib server).

## 4. Check that everything talks to everything

```bash
uv run startup_scripts/check_stack.py --pytemlib http://127.0.0.1:8003/mcp
```

It runs a small map, stitches it, scores it, marks a point, and exercises the approval mode. Every line should say PASS. A FAIL line says what failed and why.

## 5. Map with an agent

```bash
# The fixed mapping workflow: the model writes the plan, code runs it.
uv run python -m asyncroscopy.agents.run mapping "Map a 600 nm square around the current position and take an EDS spectrum with each image"

# The general agent: the model chooses every tool call, loading the mapping skill itself.
uv run python -m asyncroscopy.agents.run general "Map a 600 nm square around the current position and find the 3 most interesting regions"
```

Choose the model with `ASYNCROSCOPY_MODEL` (default `qwen3:8b`; `qwen3:4b` for machines with less memory). All settings are listed in `asyncroscopy/agents/__init__.py`.

## 6. LangGraph Studio

```bash
uv sync --extra agent --extra ollama --extra dev
cp .env.example .env
uv run langgraph dev
```

Studio opens in the browser with two graphs: `mapping` and `general`. Tool calls made from Studio go through the same MCP endpoints and approval mode as everything else.

## 7. Approval mode

The instrument starts in `auto` mode: agent calls run and are recorded. Switch to `approve` before an unattended run on real hardware; see [approval_mode.md](approval_mode.md).

## Troubleshooting

| Symptom | Cause and fix |
| :-- | :-- |
| `ModuleNotFoundError: No module named 'asyncroscopy'` running a script | Run it with `uv run` from the repository root; scripts in `startup_scripts/` and `benchmarks/` add the root to the path themselves. |
| A launcher seems stuck on `WAIT ...` while its output goes to a file | Older launchers buffered output when not writing to a terminal; update. The servers were probably ready. |
| pyTEMlib server: `Tiled ... refused to store the map` | Start it with `--api-key` set to the Tiled key (default `secret`). |
| `ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'` | An old `langchain-mcp-adapters` is installed; asyncroscopy no longer needs it. `uv sync` again. |
| Port already in use | A previous stack is still running. Stop it with Ctrl+C in its terminal; the launchers also clean up their own stale processes on the next start. |
| The agent picks the wrong tool or loops | Use `qwen3:8b` or larger, keep `ASYNCROSCOPY_NUM_CTX` at 8192 or more, and prefer the `mapping` graph for maps. `benchmarks/run_tool_json.py` measures a model on this machine. |
