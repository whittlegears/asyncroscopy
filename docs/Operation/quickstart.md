# Quick start: from clone to an agent mapping the digital twin

This takes a fresh machine to a running digital twin, both MCP endpoints, the analysis server and a local-model agent that maps a sample. It was written while doing exactly that on Windows 11 and checked from a fresh clone on Ubuntu (WSL2); macOS and Linux differ only where noted.

Nothing here needs a microscope, vendor software or a GPU. A GPU makes the local model faster.

## 1. Install the tools

| Tool | Windows | macOS / Linux |
| :-- | :-- | :-- |
| uv (installs Python 3.12 for you) | `winget install astral-sh.uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Git | `winget install Git.Git` | usually present |
| Ollama (local models) | `winget install Ollama.Ollama` | `curl -fsSL https://ollama.com/install.sh \| sh` (Linux); the app from ollama.com (macOS) |

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
uv sync --extra agent --extra ollama   # Windows: fails with "Missing expected target directory"? see the fix below
uv run pytest -q -m "not slow"
```

`--depth 1` skips the project history (about 250 MB); leave it out if you want the history. On Windows, keep the clone's path short (under about 60 characters, such as `C:\src\asyncroscopy`): some installed DLLs have long names, and a deeply nested folder pushes them past Windows' path limit (see Troubleshooting).

`uv sync` installs asyncroscopy itself into `.venv` (editable), so `import asyncroscopy` works from `notebooks/` or any other folder, not only the repository root. This repository does not commit a `uv.lock`, so `uv sync` resolves dependency versions fresh on each machine; `pyproject.toml` caps the fast-moving dependencies (sidpy, SciFiReaders, tiled, fastmcp, numpy, LangChain/LangGraph) below their next breaking release so a fresh clone still gets a tested combination. pyTEMlib, whose map/analysis MCP server is not in a PyPI release yet, comes from a fork and is pinned to a fixed commit in `pyproject.toml`.

The first `uv sync` takes 5 to 10 minutes: it resolves about 400 packages (there is no lock file) and downloads about 4 GB into `.venv` and uv's cache. Later syncs take seconds.

The test suite starts real Tango devices in-process. Without the slow tests, pytest prints its summary after about 5 to 7 minutes on a laptop (about 1,200 tests) (the first run is the slowest and prints nothing for about the first minute while it imports everything); the devices then shut down and keep printing INFO lines ("Connected to detector proxy", "Restarting managed Tiled server") for up to a minute more. That is normal; let it finish. `uv run pytest -q` runs everything, including the two timing tests that `-m "not slow"` deselects. All tests should pass; five are skipped on purpose (two need the optional `diffraction` extra; three use Tango's database-mode device discovery, which cannot run in the same Python process as the in-process test devices). The DeprecationWarnings in the summary come from dependencies and are harmless. Add `-rs` to see why each was skipped. Running the tests leaves the working tree unchanged (set `ASYNCROSCOPY_WRITE_BENCHMARKS=1` to update `benchmarks/results/twin_render.json` on purpose).

**Windows: if `uv sync` stops with "Missing expected target directory for Python minor version link"**, uv's link to its Python install is broken. Point uv at the interpreter file directly (PowerShell); this is the fix that worked in testing:

```powershell
$py = (Get-ChildItem "$env:APPDATA\uv\python\cpython-3.12.*-windows-x86_64-none\python.exe" | Sort-Object { [version]($_.Directory.Name -replace '^cpython-([\d.]+)-.*', '$1') } -Descending | Select-Object -First 1).FullName
& $uv sync --python $py --extra agent --extra ollama   # $uv from section 1; or plain uv if it is on PATH
```

Later `uv run` commands do not need `--python`. If no file matches, install that Python first with `uv python install 3.12`. (`uv python dir` prints the folder uv keeps its Pythons in, if yours is elsewhere.) `uv python install 3.12 --reinstall` followed by `uv sync --python 3.12 ...` can also repair the link, but it fails when a different version's link is the broken one.

## 3. Start the stack

Use four terminals, from the repository root. Each MCP server prints a large FastMCP banner when it starts; that is normal. To run them in the background in PowerShell instead (use `$uv` from section 1 if `uv` is not on `PATH`):

```powershell
if (-not $uv) { $uv = (Get-Command uv -ErrorAction SilentlyContinue).Source }   # keeps the $uv from section 1 if you set it there
if (-not $uv) {
    'uv not found: set $uv as in section 1, then paste this again.'
} elseif (Get-NetTCPConnection -State Listen -LocalPort 8000,8001,8003,9091,9094 -ErrorAction SilentlyContinue) {
    'A stack is already running (ports 8000-9094 in use). Stop it first (below), or run a second one beside it on other ports (docs/Adding_New_Hardware/second_stack.md, run_servers.py --isolated).'
} else {
    $servers = $null; $ready = $false; $deadline = (Get-Date).AddMinutes(3)
    $servers = Start-Process $uv 'run','startup_scripts/run_servers.py','--yaml','configs/digital_twin_mapping.yaml' -RedirectStandardOutput servers.log -RedirectStandardError servers.err -WindowStyle Hidden -PassThru
    while (-not $ready) {
        $ready = (Test-Path servers.log) -and (Select-String -Path servers.log -Pattern 'Leave this terminal open' -Quiet)
        if (-not $ready -and (-not $servers -or $servers.HasExited -or (Get-Date) -gt $deadline)) { 'The servers did not get ready (stop them as below):'; Get-Content servers.log, servers.err -Tail 20 -ErrorAction SilentlyContinue; break }
        if (-not $ready) { Start-Sleep 1 }
    }   # about a minute
    $agent = $operator = $analysis = $null
    if ($ready) {   # if the servers stopped, read the lines above and fix that first
        $agent = Start-Process $uv 'run','startup_scripts/run_mcp.py','--yaml','configs/mcp_dt.yaml' -RedirectStandardOutput mcp_dt.log -RedirectStandardError mcp_dt.err -WindowStyle Hidden -PassThru
        $operator = Start-Process $uv 'run','startup_scripts/run_mcp.py','--yaml','configs/mcp_operator.yaml' -RedirectStandardOutput mcp_op.log -RedirectStandardError mcp_op.err -WindowStyle Hidden -PassThru
        $analysis = Start-Process $uv 'run','python','-m','asyncroscopy.analysis_server','--tiled','http://localhost:9091','--api-key','secret','--port','8003' -RedirectStandardOutput analysis.log -RedirectStandardError analysis.err -WindowStyle Hidden -PassThru
        Start-Sleep 20
        foreach ($s in @(@($agent, 'mcp_dt.err'), @($operator, 'mcp_op.err'), @($analysis, 'analysis.err'))) {
            if (-not $s[0] -or $s[0].HasExited) { "$($s[1]) says the server stopped (or did not start):"; Get-Content -Encoding utf8 $s[1] -Tail 10 -ErrorAction SilentlyContinue }
        }
    }
}
```

On Linux or macOS, the same in the background from bash (each server in its own process group, so one `kill` stops it with its children):

```bash
setsid uv run startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml > servers.log 2> servers.err & echo $! > servers.pid
until grep -q 'Leave this terminal open' servers.log; do sleep 1; done          # about a minute
setsid uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml > mcp_dt.log 2> mcp_dt.err & echo $! > mcp_dt.pid
setsid uv run startup_scripts/run_mcp.py --yaml configs/mcp_operator.yaml > mcp_op.log 2> mcp_op.err & echo $! > mcp_op.pid
setsid uv run python -m asyncroscopy.analysis_server --tiled http://localhost:9091 --api-key secret --port 8003 > analysis.log 2> analysis.err & echo $! > analysis.pid
```

Stop them with `for f in servers mcp_dt mcp_op analysis; do kill -TERM -- -$(cat $f.pid); done`, and check with `ss -ltn | grep -E ':(8000|8001|8003|9091|9094) '` (no output means all stopped). A launcher that died without stopping its servers leaves them running; `pkill -f 'startup_scripts/run_(servers|mcp)\.py|pyTEMlib\.mcp_server|-m asyncroscopy\.'` stops every asyncroscopy stack on the machine.

Each `.log` file gets the server's request lines and each `.err` file its banner, warnings and tool errors; read them with `Get-Content -Encoding utf8 mcp_dt.err -Wait` (without `-Encoding utf8`, Windows PowerShell 5.1 shows the banner as garbled characters). Stop everything later with `foreach ($p in $servers, $agent, $operator, $analysis) { if ($p) { taskkill /PID $p.Id /T /F 2>$null | Out-Null } }`, which also stops the child servers (in the same PowerShell window; from another one, see Troubleshooting); check with `Get-NetTCPConnection -State Listen -LocalPort 8000,8001,8003,9091,9094 -ErrorAction SilentlyContinue` (no output means all stopped). In the background `servers.log` still ends with "Leave this terminal open ... Ctrl+C": there it only marks that the servers are ready.

```bash
# 1. Twin + data + Tiled + Mapper + Approvals. Wait for "Leave this terminal open".
uv run startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml

# 2. Agent MCP endpoint (port 8000)
uv run startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml

# 3. Operator MCP endpoint for the GUI (port 8001, this machine only)
uv run startup_scripts/run_mcp.py --yaml configs/mcp_operator.yaml

# 4. analysis server: stitching, scoring, points of interest (port 8003)
uv run python -m asyncroscopy.analysis_server --tiled http://localhost:9091 --api-key secret --port 8003
```

`configs/digital_twin_mapping.yaml` simulates a real instrument's timing (`time_scale: 1.0`): stage moves take seconds. Set `time_scale: 0` in that file for fast runs. `configs/DigitalTwin.yaml` is an ideal instrument with a tiny sample, useful for unit-style experiments.

The environment was synced in step 2, so after that the launchers start their child servers with `uv run --no-sync` and never reinstall packages while the stack is coming up. If you change `pyproject.toml` or the extras, run `uv sync ...` once, in one terminal, before starting the stack again: several `uv run` commands syncing the same environment at the same time can trip over each other.

Both MCP endpoints listen on `127.0.0.1` only. They have no authentication, and the agent endpoint runs in `auto` approval mode, so anyone who can reach it can drive the instrument. To use the agent endpoint from another machine, do it deliberately: set `mcp.http_host` in `configs/mcp_dt.yaml` to this machine's address on a network you trust (or `0.0.0.0` for every interface), switch approval mode to `approve` (see section 7), and point the other machine's `ASYNCROSCOPY_MCP_URL` at `http://<this machine>:8000/mcp`. Keep the operator endpoint (`mcp_operator.yaml`) on `127.0.0.1`: it can approve held calls. For bearer tokens (WG1's token service, or development tokens to test with), TLS and a reverse proxy, see [Remote access](remote_access.md).

The Tiled API key `secret` is the default the DATA device starts Tiled with; set `ASYNCROSCOPY_TILED_API_KEY` to change it (and pass the same value to the analysis server).

## 4. Check that everything talks to everything

(In PowerShell without `uv` on `PATH`, write `& $uv` for `uv` here and in section 5, as in section 1.)

```bash
uv run startup_scripts/check_stack.py --analysis http://127.0.0.1:8003/mcp
```

It runs a small map, stitches it, scores it, marks a point, and exercises the approval mode. Every line should say PASS. A FAIL line says what failed and why.

To see what an agent will be able to do on this stack (every tool, its risk, whether auto mode runs it or holds it for you, and which tools a small model is given up front):

```bash
uv run startup_scripts/show_agent_view.py
```

A command your own device adds shows up as high risk and held: auto mode waits for a person until you switch it on (SciAgentGUI: Settings, Agent permissions) or mark a read-only command low with `mcp.safety.risk`. See [approval_mode.md](approval_mode.md).

If you use SciAgentGUI, this checks that the stack answers the way the GUI expects (approvals, maps and spectra from Tiled):

```bash
uv run startup_scripts/check_gui_contract.py
```

The checks use whatever stack is on these ports, so two people testing on one machine share it. Both checks briefly switch approval mode to `approve`, so don't run them while an agent is working. While they run, the agent endpoint's terminal (in the background: `mcp_dt.err`) prints `Error calling tool ...` for each call they deliberately hold for approval; that is expected.

Ports: Tango database 9094, Tiled 9091, MCP endpoints 8000 and 8001, analysis server 8003.

## 5. Map with an agent

```bash
# The fixed mapping workflow: the model writes the plan, code runs it.
uv run python -m asyncroscopy.agents.run mapping "Map a 600 nm square around the current position and take an EDS spectrum with each image"

# The general agent: the model chooses every tool call, loading the mapping skill itself.
uv run python -m asyncroscopy.agents.run general "Map a 600 nm square around the current position and find the 3 most interesting regions"
```

Choose the model with `ASYNCROSCOPY_MODEL` (default `qwen3:8b`; `qwen3:4b` for machines with less memory). Ollama on another machine (a GPU box, or Windows from WSL without mirrored networking): set `OLLAMA_HOST=http://<host>:11434`. All settings are listed in `asyncroscopy/agents/__init__.py`.

The command prints a progress line as each step finishes (`[mapping   40s] wait: done`), then the final answer. Expect about two minutes for the mapping request: the twin's mapping config runs in simulated real time (`time_scale: 1`), so 25 tiles with EDS take as long as they would on the instrument. On a 6 GB laptop GPU, with qwen3:8b, the `mapping` graph completed 29 of 30 trial runs, then 9 of 9 on the final code; the `general` agent, which makes every call itself, completed 30 of 30, then 15 of 15 on the final code. See `docs/athena/wg2_traceability.md`.

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
| Windows: `DLL load failed ... The filename or extension is too long`, or `The 'dask-ml' library is required` although it is installed | The clone is in a deep folder, so DLLs inside `.venv` exceed Windows' path limit (the dask-ml message hides the real import error). Clone again to a short path such as `C:\src\asyncroscopy`. Moving only the environment (`UV_PROJECT_ENVIRONMENT`) is not enough: the tests' temporary files (`.pytest_tmp`) and Tiled's acquisitions (`outputs/tiled_acquisitions`) are written inside the clone too, and fail past the limit (Tiled then answers 500; `outputs/logs/tiled/*.log` names the path). Turning on Windows long paths (`LongPathsEnabled`, needs an administrator) also works. |
| The agent stops with `llama-server process has terminated ... 0xc0000409` or `CUDA error` | Ollama's GPU runner crashed loading the model (not asyncroscopy); it happens most on the first load on a small GPU. Run the command again; if it keeps failing, restart Ollama or use a smaller model (`ASYNCROSCOPY_MODEL=qwen3:4b`). The agent command exits with status 1 when a map did not complete. |
| `uv sync` fails to fetch pyTEMlib | It is installed from GitHub (git must be installed and github.com reachable). |
| A launcher seems stuck on `WAIT ...` while its output goes to a file | Older launchers buffered output when not writing to a terminal; update. The servers were probably ready. |
| Analysis server: `Tiled ... refused to store the map` | Start it with `--api-key` set to the Tiled key (default `secret`). |
| `ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'` | An old `langchain-mcp-adapters` is installed; asyncroscopy no longer needs it. `uv sync` again. |
| `check_stack.py` fails at "a microscope with move_stage is exported", or the agent endpoint lists far fewer tools than usual | The MCP servers were started before `run_servers.py` printed "Leave this terminal open"; they offer only the devices that existed when they started. Restart the two `run_mcp.py` servers. |
| The recipe prints `mcp_op.err says the server stopped (or did not start)` (or another server's) | Read that `.err` file: it names the error. If it is empty, the process was stopped from outside before it printed anything (seen once while other test scripts were killing processes on the same machine). Start just that server again with its `Start-Process` line from the recipe. |
| `check_stack.py` says an endpoint is not reachable | The stack is not up yet: start it (section 3) and wait until both MCP logs say "MCP ready". |
| Started in the background and something is still running after stopping, "already running", or you need to stop it from another window | Free ports alone do not mean everything stopped: the launcher and its device servers hold no port. Stop every asyncroscopy stack on this machine with `Get-CimInstance Win32_Process \| Where-Object { $_.CommandLine -match 'startup_scripts[/\\]run_(servers\|mcp)\.py\|-m (pyTEMlib\.mcp_server\|asyncroscopy\.)' } \| ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>$null \| Out-Null }`, which also catches the device servers (`python -m asyncroscopy...`) of a launcher that died, then check the ports again; anything still holding one (a leftover Tango database or Tiled) stops with `Get-NetTCPConnection -State Listen -LocalPort 8000,8001,8003,9091,9094 -ErrorAction SilentlyContinue \| Select-Object -Unique OwningProcess \| ForEach-Object { taskkill /PID $_.OwningProcess /T /F 2>$null \| Out-Null }`. |
| Port already in use | A previous stack is still running. Stop it with Ctrl+C in its terminal. On the next start `run_servers.py` also stops leftovers of the same config (whatever an earlier run of it left on its Tango and Tiled ports, and its device servers). It does not stop another stack: if another config's stack holds the ports, or both would share the project-root Tango database file (neither started with `--isolated`), it refuses, names that stack, and says how to run beside it ([second_stack.md](../Adding_New_Hardware/second_stack.md)); `--replace` stops it on purpose. |
| A map plan is refused with `the limits of ... could not be read` | The Mapper checks every tile against the STAGE device's x/y limits and refuses a plan it cannot check. The STAGE device (`asyncroscopy/stage/default`) is not running or not answering: restart the servers (section 3). A job device for hardware without readable limits can set its `require_limits` property to false. |
| The agent picks the wrong tool or loops | Use `qwen3:8b` or larger, keep `ASYNCROSCOPY_NUM_CTX` at 8192 or more, and prefer the `mapping` graph for maps. `benchmarks/run_tool_json.py` measures a model on this machine. |
