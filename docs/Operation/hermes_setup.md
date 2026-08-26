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

## Updating

`hermes update` updates the shared install in place. Run it from any account
(step 2's ACL grant makes the tree writable by all local users); close the
Hermes desktop app and stop `hermes gateway` first so no files in the tree are
locked.
