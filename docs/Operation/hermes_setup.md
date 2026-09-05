# Hermes Agent Setup

The LLM Tango device supports a `hermes` agent backend that delegates queries to a
[Hermes Agent](https://github.com/NousResearch/hermes-agent) gateway over its
OpenAI-compatible HTTP API. The backend expects the gateway at
`http://127.0.0.1:8642` (the device property `hermes_url` defaults to this), and
`SetBackend('hermes')` verifies the gateway is reachable before switching — so
Hermes must be installed and its gateway running first.

This page covers installing Hermes on Windows **for all users, onto the D:
drive** (the layout used on the shared microscope PC), then wiring it to
asyncroscopy.

## How the installer picks its location

The Windows installer places *everything* — the `hermes-agent` checkout, the
Python venv, managed `uv`, PortableGit, Node.js, and Hermes config/state — under
a single root directory, `HERMES_HOME` (default `%LOCALAPPDATA%\hermes`, which
is per-user on C:). Setting `HERMES_HOME` before running the installer relocates
the whole install; nothing else needs to move.

The installer only persists `HERMES_HOME` and PATH for the account that ran it,
so making the install visible to every account is a separate (one-time,
elevated) step below.

## 1. Install to D:\hermes

In PowerShell (run as the account that will administer the install):

```powershell
$env:HERMES_HOME = 'D:\hermes'
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

The installer handles uv, Python, Node.js, and a portable Git — no admin
rights or pre-installed tooling required. When it finishes, `hermes setup`
runs to configure a model provider (skip with `-SkipSetup` if you want to do
that later).

## 2. Expose the install to all users

In an **elevated** PowerShell (machine-scope environment variables and ACL
changes require admin):

```powershell
# Every account resolves its Hermes home to the shared install
[Environment]::SetEnvironmentVariable('HERMES_HOME', 'D:\hermes', 'Machine')

# Mirror the PATH entries the installer added for the installing user,
# at machine scope: the hermes launcher, managed Node, and portable Git
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
foreach ($dir in 'D:\hermes\hermes-agent\bin', 'D:\hermes\node', 'D:\hermes\git\cmd') {
    if (($machinePath -split ';') -notcontains $dir) { $machinePath = "$dir;$machinePath" }
}
[Environment]::SetEnvironmentVariable('Path', $machinePath, 'Machine')

# Let all local users run Hermes and write its shared state
# (S-1-5-32-545 is BUILTIN\Users, locale-independent)
icacls D:\hermes /grant '*S-1-5-32-545:(OI)(CI)M' /T
```

Open a **new** terminal afterwards — environment changes don't reach shells
that are already running.

:::{note}
`HERMES_HOME` holds config and state as well as code: with a shared home,
every account shares one `.env` (provider API keys), one session history, and
one agent memory. On a lab instrument PC that is usually what you want; if a
user needs isolated state, they can set a per-user `HERMES_HOME` (user-scope
variables override machine-scope ones) — but that triggers a fresh install
under the new location.
:::

## 3. Enable the OpenAI-compatible API server

The asyncroscopy `hermes` backend talks to Hermes through its API server,
which is off by default. Add to `D:\hermes\.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=pick-a-key
```

Then start the gateway (any user, one instance per machine — it binds port
8642):

```bash
hermes gateway
```

You should see `[API Server] API server listening on http://127.0.0.1:8642`.
Verify from PowerShell:

```bash
curl.exe http://127.0.0.1:8642/v1/models
```

## 4. Point asyncroscopy at it

`hermes_url` already defaults to `http://127.0.0.1:8642`, so an LLM device
started from any config can switch at runtime with `SetBackend('hermes')` (or
by selecting the hermes harness in SciAgentGUI). To start on the hermes
backend directly, use the provided config:

```bash
uv run startup_scripts/run_llm.py --yaml configs/hermes-llm.yaml
```

If you set `API_SERVER_KEY` above, add the matching key to the LLM YAML
config:

```yaml
hermes_api_key: pick-a-key
```

If the switch is refused, the reason is in the device's `status` attribute —
the two common ones are the gateway not running (step 3) and a key mismatch.

## The two backends side by side

One LLM Tango device hosts both backends; only one is live at a time and
`SetBackend('langgraph' | 'hermes')` switches between them at runtime. Which
one is active is always visible on the device:

- `backend` attribute — `langgraph` or `hermes`.
- `status` attribute — `Agent backend: <name>` when healthy, or the full
  initialization/switch failure reason.
- `backend_capabilities` attribute — JSON flags, e.g.
  `{"complete": true, "connect_mcp": true, "skills": true}`. Clients (like
  SciAgentGUI's builtin harness) should check `complete` before relying on the
  `Complete` command and fall back to `Query`.
- `last_trace` attribute — JSON list of the most recent `Query`'s tool calls
  and results (langgraph only; always `[]` on hermes, whose agent loop runs
  out-of-process in the gateway).

| | `langgraph` | `hermes` |
|---|---|---|
| Config file | `configs/langgraph-llm.yaml` (gitignored, carries local keys) or `configs/gemma-llm.yaml` | `configs/hermes-llm.yaml` |
| Agent loop | In-process LangGraph supervisor swarm | Server-side in the Hermes gateway |
| Model | Ollama (`gemma4:31b` by default, port 11434) or a cloud provider | Whatever the gateway serves (`hermes_model`, default `hermes-agent`) |
| Tools | Inherited from the MCP bridge (`mcp_url`, port 8000) via `ConnectMCP` | Hermes's own server-side tools; `ConnectMCP` registers the server in the gateway's `config.yaml` (hot-reloaded), named `asyncroscopy-<port>` |
| `Complete` (raw model turn) | Supported | Refused with an error payload |
| Skills | Exposed as `find_skills`/`load_skill` tools inside the swarm | Top-matching skills are injected into each query as a system message, and their usage is logged |
| Skill round-trip | Reflection step writes proposals to the store's `_proposals/` | Store skills are mirrored to `<HERMES_HOME>/skills/asyncroscopy/` on device start and every `SyncSkills`; skills the Hermes agent authors elsewhere under `<HERMES_HOME>/skills/` are proposed back into the store after each query (first scan is a silent baseline) |
| Extra port | Ollama `11434` | Hermes gateway `8642` |

There are no port collisions between the stacks — Tango DB (`9094`) and the
MCP bridge (`8000`) are shared, Ollama (`11434`) belongs to langgraph, and the
Hermes gateway (`8642`) belongs to hermes — so the Ollama server and the
Hermes gateway can both stay running while you switch back and forth.

`configs/hermes-llm.yaml` still sets `mcp_url` and the Ollama model fields:
the hermes backend ignores them, but a runtime `SetBackend('langgraph')` uses
them to bring the swarm up with the microscope tools.

The Hermes install root for the skill round-trip and MCP registration comes
from the `hermes_home` device property, falling back to `HERMES_HOME`, then
`~/.hermes`.

## Multiple instruments at once

Each MCP bridge is one `ConnectMCP` call, so the gateway can hold the real
instrument and the digital twin side by side. Run a second MCP server on
another port (`mcp.http_port` in the MCP YAML, e.g. `8001` for the twin),
then connect both — from the GUI or directly:

```python
llm.ConnectMCP('{"url": "http://127.0.0.1:8000/mcp"}')   # real instrument
llm.ConnectMCP('{"url": "http://127.0.0.1:8001/mcp"}')   # digital twin
```

On hermes each URL becomes its own `mcp_servers` entry (`asyncroscopy-8000`,
`asyncroscopy-8001`) with distinct tool prefixes, so someone can drive the
real microscope while the twin runs untouched on its own port.

### When the gateway is unreachable

Nothing wedges:

- Starting from `hermes-llm.yaml` with the gateway down puts the device in
  `FAULT`, with the reason in `status`. `Query`/`Complete`/`ConnectMCP` then
  return a clear "No agent backend is initialized" error instead of hanging,
  and `SetBackend` (either name) recovers the device once the problem is
  fixed.
- `SetBackend('hermes')` with the gateway down builds and verifies the new
  backend *before* replacing the old one, so it returns `False`, writes the
  reason to `status`, and leaves the current backend running.
- A gateway that dies mid-query surfaces as an error naming the gateway URL;
  gateway HTTP errors (e.g. a key mismatch) surface with the gateway's own
  error message.

## Updating

`hermes update` updates the shared install in place. Run it from any account
(step 2's ACL grant makes the tree writable by all local users); close the
Hermes desktop app and stop `hermes gateway` first so no files in the tree are
locked.
