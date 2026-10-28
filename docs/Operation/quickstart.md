# Quick start: from clone to an agent mapping the digital twin

This takes a fresh machine to a running digital twin, both MCP endpoints, the pyTEMlib analysis server and a local-model agent that maps a sample. It was written while doing exactly that on Windows 11; macOS and Linux differ only where noted.

Nothing here needs a microscope, vendor software or a GPU. A GPU makes the local model faster.

## 1. Install the tools

| Tool | Windows | macOS / Linux |
| :-- | :-- | :-- |
| uv (installs Python 3.12 for you) | `winget install astral-sh.uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Git | `winget install Git.Git` | usually present |
| Ollama (local models) | `winget install Ollama.Ollama` | see ollama.com |

**Open a new terminal after installing uv**: installers add uv to `PATH`, and a terminal that was already open does not see the change. Check with `uv --version`.

**Windows: `uv` still not found?** winget sometimes installs uv without a command shim on `PATH`. Call it by its full path; in PowerShell:

```powershell
$uv = (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe", "$env:USERPROFILE\.local\bin\uv.exe" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
& $uv --version
```

(The second path is where the official PowerShell installer puts uv. `where.exe uv` finds it anywhere on `PATH`.)

and write `& $uv` wherever this guide says `uv` (for example `& $uv sync --extra agent --extra ollama`). The launchers find uv on their own when you start them this way: `uv run` tells its child processes where uv is.

Then pull a model. `qwen3:8b` is the tested default (5 GB; runs partly on CPU on a 6 GB GPU); `qwen3:4b` (2.5 GB) works on smaller machines:

```bash
ollama pull qwen3:8b
```

## 2. Get the code and its dependencies

```bash
git clone --depth 1 https://github.com/pycroscopy/asyncroscopy.git   # Windows: add a short target, e.g. C:\src\asyncroscopy
cd asyncroscopy
uv sync --extra agent --extra ollama
uv run pytest -q -m "not slow"
```

`--depth 1` skips the project history (about 250 MB); leave it out if you want the history. On Windows, keep the clone's path short (under about 60 characters, such as `C:\src\asyncroscopy`): some installed DLLs have long names, and a deeply nested folder pushes them past Windows' path limit (see Troubleshooting).

`uv sync` installs asyncroscopy itself into `.venv` (editable), so `import asyncroscopy` works from `notebooks/` or any other folder, not only the repository root. This repository does not commit a `uv.lock`, so `uv sync` resolves dependency versions fresh on each machine; `pyproject.toml` caps the fast-moving dependencies (sidpy, SciFiReaders, tiled, fastmcp, numpy, LangChain/LangGraph) below their next breaking release so a fresh clone still gets a tested combination. pyTEMlib, whose map/analysis MCP server is not in a PyPI release yet, comes from a fork and is pinned to a fixed commit in `pyproject.toml`.

The test suite starts real Tango devices in-process. Without the slow tests, pytest prints its summary after about 40 s on a laptop; the devices then shut down and keep printing INFO lines ("Connected to detector proxy", "Restarting managed Tiled server") for up to a minute more. That is normal; let it finish. `uv run pytest -q` runs everything, including the two timing tests that `-m "not slow"` deselects. All tests should pass; five are skipped on purpose (two need the optional `diffraction` extra, three need a running Tango database). The DeprecationWarnings in the summary come from dependencies and are harmless. Add `-rs` to see why each was skipped. Running the tests leaves the working tree unchanged (set `ASYNCROSCOPY_WRITE_BENCHMARKS=1` to update `benchmarks/results/twin_render.json` on purpose).

**Windows: if `uv sync` stops with "Missing expected target directory for Python minor version link"**, uv's link to its Python install is broken. Point uv at the interpreter file directly (PowerShell); this is the fix that worked in testing:

```powershell
$py = (Get-ChildItem "$env:APPDATA\uv\python\cpython-3.12.*-windows-x86_64-none\python.exe" | Select-Object -First 1).FullName
& $uv sync --python $py --extra agent --extra ollama   # $uv from section 1; or plain uv if it is on PATH
```

If no file matches, install that Python first with `uv python install 3.12`. (`uv python dir` prints the folder uv keeps its Pythons in, if yours is elsewhere.) `uv python install 3.12 --reinstall` followed by `uv sync --python 3.12 ...` can also repair the link, but it fails when a different version's link is the broken one.

## 3. Start the stack

Use four terminals, from the repository root. Each MCP server prints a large FastMCP banner when it starts; that is normal. To run them in the background in PowerShell instead (use `$uv` from section 1 if `uv` is not on `PATH`):

```powershell
if (-not $uv) { $uv = (Get-Command uv).Source }   # keeps the $uv from section 1 if you set it there
$servers = Start-Process $uv 'run','startup_scripts/run_servers.py','--yaml','configs/digital_twin_mapping.yaml' -RedirectStandardOutput servers.log -RedirectStandardError servers.err -WindowStyle Hidden -PassThru
while (-not ((Test-Path servers.log) -and (Select-String -Path servers.log -Pattern 'Leave this terminal open' -Quiet))) { if ($servers.HasExited) { Get-Content servers.log, servers.err -Tail 20; break }; Start-Sleep 1 }   # about a minute
$agent = $operator = $analysis = $null
if (-not $servers.HasExited) {   # if the servers stopped, read the lines above and fix that first
    $agent = Start-Process $uv 'run','startup_scripts/run_mcp.py','--yaml','configs/mcp_dt.yaml' -RedirectStandardOutput mcp_dt.log -RedirectStandardError mcp_dt.err -WindowStyle Hidden -PassThru
    $operator = Start-Process $uv 'run','startup_scripts/run_mcp.py','--yaml','configs/mcp_operator.yaml' -RedirectStandardOutput mcp_op.log -RedirectStandardError mcp_op.err -WindowStyle Hidden -PassThru
    $analysis = Start-Process $uv 'run','python','-m','pyTEMlib.mcp_server','--tiled','http://localhost:9091','--api-key','secret','--port','8003' -RedirectStandardOutput pytem.log -RedirectStandardError pytem.err -WindowStyle Hidden -PassThru
}
```

Each `.log` file gets the server's request lines and each `.err` file its banner, warnings and tool errors; read them with `Get-Content -Encoding utf8 mcp_dt.err -Wait` (without `-Encoding utf8`, Windows PowerShell 5.1 shows the banner as garbled characters). Stop everything later with `foreach ($p in $servers, $agent, $operator, $analysis) { if ($p) { taskkill /PID $p.Id /T /F } }`, which also stops the child servers.

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

The environment was synced in step 2, so after that the launchers start their child servers with `uv run --no-sync` and never reinstall packages while the stack is coming up. If you change `pyproject.toml` or the extras, run `uv sync ...` once, in one terminal, before starting the stack again: several `uv run` commands syncing the same environment at the same time can trip over each other.

Both MCP endpoints listen on `127.0.0.1` only. They have no authentication, and the agent endpoint runs in `auto` approval mode, so anyone who can reach it can drive the instrument. To use the agent endpoint from another machine, do it deliberately: set `mcp.http_host` in `configs/mcp_dt.yaml` to this machine's address on a network you trust (or `0.0.0.0` for every interface), switch approval mode to `approve` (see section 7), and point the other machine's `ASYNCROSCOPY_MCP_URL` at `http://<this machine>:8000/mcp`. Keep the operator endpoint (`mcp_operator.yaml`) on `127.0.0.1`: it can approve held calls.

The Tiled API key `secret` is the default the DATA device starts Tiled with; set `ASYNCROSCOPY_TILED_API_KEY` to change it (and pass the same value to the pyTEMlib server).

## 4. Check that everything talks to everything

```bash
uv run startup_scripts/check_stack.py --pytemlib http://127.0.0.1:8003/mcp
```

It runs a small map, stitches it, scores it, marks a point, and exercises the approval mode. Every line should say PASS. A FAIL line says what failed and why.

If you use SciAgentGUI, this checks that the stack answers the way the GUI expects (approvals, maps and spectra from Tiled):

```bash
uv run startup_scripts/check_gui_contract.py
```

Both checks briefly switch approval mode to `approve`, so don't run them while an agent is working. While they run, the agent endpoint's terminal prints `Error calling tool ...` for each call they deliberately hold for approval; that is expected. Right after `check_stack.py`, `check_gui_contract.py` prints `SKIP spectrum check`: the check map has no EDS points. It is expected; after an EDS map (section 5) the spectrum check runs.

Ports: Tango database 9094, Tiled 9091, MCP endpoints 8000 and 8001, pyTEMlib 8003.

## 5. Map with an agent

```bash
# The fixed mapping workflow: the model writes the plan, code runs it.
uv run python -m asyncroscopy.agents.run mapping "Map a 600 nm square around the current position and take an EDS spectrum with each image"

# The general agent: the model chooses every tool call, loading the mapping skill itself.
uv run python -m asyncroscopy.agents.run general "Map a 600 nm square around the current position and find the 3 most interesting regions"
```

Choose the model with `ASYNCROSCOPY_MODEL` (default `qwen3:8b`; `qwen3:4b` for machines with less memory). All settings are listed in `asyncroscopy/agents/__init__.py`.

The command prints a progress line as each step finishes (`[mapping   40s] wait: done`), then the final answer. Expect about two minutes for the mapping request: the twin's mapping config runs in simulated real time (`time_scale: 1`), so 25 tiles with EDS take as long as they would on the instrument. On a 6 GB laptop GPU, with qwen3:8b, the `mapping` graph completed 29 of 30 trial runs, then 9 of 9 on the final code; the `general` agent, which makes every call itself, completed 30 of 30 on the final code. See `docs/athena/wg2_traceability.md`.

## 6. LangGraph Studio

```bash
uv sync --extra agent --extra ollama --extra dev   # Windows: add --python $py if section 2 needed it
cp .env.example .env
uv run langgraph dev
```

Studio opens in the browser with two graphs: `mapping` and `general`. Tool calls made from Studio go through the same MCP endpoints and approval mode as everything else.

## 7. Approval mode

The instrument starts in `auto` mode: agent calls run and are recorded. Switch to `approve` before an unattended run on real hardware; see [approval_mode.md](approval_mode.md).

## Troubleshooting

| Symptom | Cause and fix |
| :-- | :-- |
| `ModuleNotFoundError: No module named 'asyncroscopy'` | The environment predates asyncroscopy being installed into it, or you are using a different Python. Run `uv sync --extra agent --extra ollama` again, and start Python, Jupyter or scripts with `uv run` (or the `.venv` interpreter). |
| `FileNotFoundError` naming `uv` when a launcher starts its servers | uv is not on `PATH`. Open a new terminal, or call uv by its full path as in section 1; the launchers then reuse that path. |
| Windows: `DLL load failed ... The filename or extension is too long`, or `The 'dask-ml' library is required` although it is installed | The clone is in a deep folder, so DLLs inside `.venv` exceed Windows' path limit (the dask-ml message hides the real import error). Clone to a short path such as `C:\src\asyncroscopy`, or keep the clone and set `UV_PROJECT_ENVIRONMENT=C:\venvs\asyncroscopy` before `uv sync`. |
| The agent stops with `llama-server process has terminated ... 0xc0000409` or `CUDA error` | Ollama's GPU runner crashed loading the model (not asyncroscopy); it happens most on the first load on a small GPU. Run the command again; if it keeps failing, restart Ollama or use a smaller model (`ASYNCROSCOPY_MODEL=qwen3:4b`). The agent command exits with status 1 when a map did not complete. |
| `uv sync` fails to fetch pyTEMlib | It is installed from GitHub (git must be installed and github.com reachable). |
| A launcher seems stuck on `WAIT ...` while its output goes to a file | Older launchers buffered output when not writing to a terminal; update. The servers were probably ready. |
| pyTEMlib server: `Tiled ... refused to store the map` | Start it with `--api-key` set to the Tiled key (default `secret`). |
| `ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'` | An old `langchain-mcp-adapters` is installed; asyncroscopy no longer needs it. `uv sync` again. |
| `check_stack.py` fails at "a microscope with move_stage is exported", or the agent endpoint lists far fewer tools than usual | The MCP servers were started before `run_servers.py` printed "Leave this terminal open"; they offer only the devices that existed when they started. Restart the two `run_mcp.py` servers. |
| Port already in use | A previous stack is still running. Stop it with Ctrl+C in its terminal. On the next start `run_servers.py` also stops whatever is listening on its Tango and Tiled ports and leftover device servers from the same config; it leaves other asyncroscopy stacks alone. |
| The agent picks the wrong tool or loops | Use `qwen3:8b` or larger, keep `ASYNCROSCOPY_NUM_CTX` at 8192 or more, and prefer the `mapping` graph for maps. `benchmarks/run_tool_json.py` measures a model on this machine. |
