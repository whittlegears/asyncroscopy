# Spectra 300 first light: runbook

This is the step-by-step for the first sessions of asyncroscopy on the Thermo Fisher Spectra 300. It is written
for two readers: the operator at the microscope, and a Claude instance helping with the install on the microscope
PC. Follow it in order. Where it says "stop", stop and ask the operator.

Read [first_light_review.md](first_light_review.md) first. Its section B lists what is not fixed yet. Until it is,
**no agent drives this microscope.** The first session only connects and reads.

Addresses used below (from `configs/Spectra300.yaml`):

| What | Address | Who may reach it |
| :-- | :-- | :-- |
| AutoScript server | 10.46.217.241:9095 (confirm, see step 3) | this PC |
| Tango database | listens on every interface, port 9094 | this PC only (firewall, step 4) |
| Tiled (data) | 127.0.0.1:9091 | this PC only |
| pyTEMlib analysis server | 127.0.0.1:8003 | this PC only (no login) |
| Agent MCP endpoint (`configs/mcp.yaml`) | 127.0.0.1:8000 | this PC only, unless tokens are set up |
| Operator MCP endpoint (`configs/mcp_operator.yaml`) | 127.0.0.1:8001 | this PC only, always |
| Read-only monitor endpoint (`configs/mcp_monitor.yaml`) | 127.0.0.1:8002 | this PC only |
| CEOS corrector server | 10.46.217.241:9092 in the config (confirm, see step 3) | this PC |

## Rules for the Claude on the microscope PC

* You are a setup helper. Run only the commands in this runbook. Never run a command that moves the stage,
  unblanks the beam, acquires, changes an attribute, or switches the approval mode.
* Do not connect to the agent or operator MCP endpoints yourself, and never approve a held call. Only the operator
  approves.
* Never print, log or commit the Tiled API key. Use the placeholders below; the operator types the real key.
* The agent that later drives the microscope must **not** have a shell on this PC. The operator endpoint trusts any
  process on this PC: an agent with a shell could approve its own held calls. Give a driving agent only its MCP
  connection.
* The agent endpoint stays on 127.0.0.1. Opening it to other machines needs both `mcp.auth` (bearer tokens,
  [remote_access.md](remote_access.md)) and `mcp.allow_remote_bind: true`; without them `run_mcp.py` binds
  127.0.0.1 or refuses to start. The operator endpoint is never opened to other machines.

## Before you go (on the development PC, with internet)

The microscope PC may have no internet. Everything it needs travels in one transfer folder
(`C:\Users\leafe\Documents\Github\transfer` on the development PC):

| File | What it is |
| :-- | :-- |
| `asyncroscopy.bundle` | the git repository, branch `feature/agentic-mapping` |
| `uv.lock` | the exact versions of every package (not committed: `.gitignore` excludes it on purpose, so developers resolve fresh) |
| `uv.exe` | the uv tool itself |
| `uv-cache\` | every package, already downloaded (only for an offline install) |
| `uv-python\` | Python 3.12 for uv (only if the microscope PC has no Python 3.12) |

1. Commit the work you are taking, then rebuild the bundle so it holds that commit:

   ```powershell
   cd C:\Users\leafe\Documents\Github\asyncroscopy
   git bundle create ..\transfer\asyncroscopy.bundle feature/agentic-mapping
   git bundle verify ..\transfer\asyncroscopy.bundle
   ```

2. Make a fresh lock and check it matches `pyproject.toml`. The pyTEMlib fork (`whittlegears/pyTEMlib`, branch
   `feature/mcp-mapping`) is pinned by commit in `pyproject.toml` (`[tool.uv.sources] pytemlib = ... rev = ...`).
   A lock made before the pin moved installs the old commit without a word, because `--frozen` does not compare the
   two. So lock again, then check:

   ```powershell
   uv lock
   uv lock --check
   Select-String -Path uv.lock -Pattern 'whittlegears/pyTEMlib\?rev=' | Select-Object -First 1
   Select-String -Path pyproject.toml -Pattern 'pyTEMlib", rev'
   ```

   The two commit hashes must be the same. Copy `uv.lock` into the transfer folder.

3. For an offline install, fill a package cache from a clean clone of the bundle, on a Windows x64 machine (the
   microscope PC's platform):

   ```powershell
   $T = "C:\Users\leafe\Documents\Github\transfer"
   git clone --branch feature/agentic-mapping $T\asyncroscopy.bundle $env:TEMP\asyncroscopy-cachefill
   Copy-Item $T\uv.lock $env:TEMP\asyncroscopy-cachefill\uv.lock
   $env:UV_CACHE_DIR = "$T\uv-cache"
   $env:UV_PYTHON_INSTALL_DIR = "$T\uv-python"
   uv python install 3.12
   cd $env:TEMP\asyncroscopy-cachefill
   uv sync --frozen
   ```

   Copy `uv.exe` (from `Get-Command uv`) into the transfer folder too. Then carry the whole folder to the
   microscope PC.

## 1. Install on the microscope PC

Below, the transfer folder is at `C:\transfer` and the repository goes to `C:\asyncroscopy`. Change the paths if
the operator wants them elsewhere. Use PowerShell.

```powershell
Set-Alias uv C:\transfer\uv.exe        # only if uv is not already on PATH
git clone --branch feature/agentic-mapping C:\transfer\asyncroscopy.bundle C:\asyncroscopy
Copy-Item C:\transfer\uv.lock C:\asyncroscopy\uv.lock
cd C:\asyncroscopy
git log -1 --oneline                    # the commit you bundled
```

With internet access (PyPI and GitHub reachable):

```powershell
uv sync --frozen
```

Without internet, use the shipped cache and Python:

```powershell
$env:UV_CACHE_DIR = "C:\transfer\uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "C:\transfer\uv-python"
uv sync --frozen --offline
```

`--frozen` installs exactly what `uv.lock` says and never re-resolves. If it reports a missing package, the cache
is incomplete: stop and refill it on the development PC (step 3 above). Do not drop `--frozen`. Without a lock,
uv resolves fresh versions that were never tested.

From now on, always start programs with `uv run --no-sync`. A plain `uv run` re-syncs the environment on every
start, which needs the network and can change versions.

Check the install:

```powershell
uv run --no-sync python -c "import asyncroscopy, tango, tiled, fastmcp; print('imports ok')"
uv run --no-sync python -c "import importlib.metadata as m; print(m.version('autoscript-tem-microscope-client'))"
uv run --no-sync python -c "import importlib.metadata as m, json; print(json.loads(m.distribution('pytemlib').read_text('direct_url.json'))['vcs_info']['commit_id'])"
```

Expect `imports ok`, `1.18.0`, and the pyTEMlib commit pinned in `pyproject.toml`.

## 2. The Tiled API key

Tiled runs with a single API key. The default, `secret`, is public knowledge. Pick a lab key and use the **same**
key in every process that touches Tiled: `run_servers.py` (which starts Tiled with it), both MCP endpoints, the
monitor endpoint, and the pyTEMlib analysis server (its `--api-key`).

The operator makes a key once:

```powershell
uv run --no-sync python -c "import secrets; print(secrets.token_urlsafe(24))"
```

and sets it for their Windows account, so every new terminal has it:

```powershell
[Environment]::SetEnvironmentVariable("ASYNCROSCOPY_TILED_API_KEY", "<lab key>", "User")
```

Close and reopen every terminal after this. Check without printing it:

```powershell
if ($env:ASYNCROSCOPY_TILED_API_KEY -and $env:ASYNCROSCOPY_TILED_API_KEY -ne "secret") { "key set" } else { "KEY MISSING" }
```

If SciAgentGUI runs on this PC, enter the same key in its Tiled settings.

Why Tiled is on 127.0.0.1: everything that reads it runs on this PC. The device servers and the MCP endpoints get
the address from the DATA device, and the analysis server is started with `--tiled http://127.0.0.1:9091`.
Tiled runs with `--public`, so on a LAN address anyone on the facility network could read the data. To serve other
machines later: set `tiled.host: 0.0.0.0` (this PC still reaches it on 127.0.0.1), set
`ASYNCROSCOPY_TILED_PUBLIC_URI=http://10.46.217.241:9091` before `run_servers.py`, and open port 9091 only to
those machines.

## 3. Config checks (with the operator)

Open `configs/Spectra300.yaml` and confirm each item with the operator or the facility. Do not guess.

1. `simulated: false` at the top.
2. **AutoScript port and version.** The config uses port 9095 for `instrument`, `aperture` and `stage`.
   AutoScript's own default is 7521. Ask the facility which port this server listens on, and which AutoScript
   version it runs. The client in this install is 1.18 (`stubs/AutoScript_v_1.18`, set in `pyproject.toml`
   under `[tool.uv.sources]`). A server on another major.minor version needs the matching client wheels. The
   pre-flight (step 6) reads the server's version and fails on a mismatch.
3. **CEOS.** `devices.corrector.properties.ceos_host` and `ceos_port` (10.46.217.241:9092 in the config). CEOS
   usually runs on the corrector PC, not this one. Get its real address. `socket_timeout: 30` bounds a stalled
   reply.
4. **Stage holder limits.** No holder is configured, so only the generic limits apply (alpha ±35°, beta ±20°,
   z ±0.5 mm). Before anything moves the stage, add the fitted holder's limits from its specification sheet to
   `devices.stage.properties`:

   ```yaml
   holders: '{"<holder name>": {"z_m": [<min>, <max>], "alpha_deg": [<min>, <max>], "beta_deg": [<min>, <max>]}}'
   holder: "<holder name>"
   # The combined z/tilt check (review B1), if this build's STAGE device has it: beyond high_tilt_deg,
   # |z| must stay within high_tilt_max_abs_z_m. Its defaults are placeholders, not this holder's clearance.
   high_tilt_deg: <deg>
   high_tilt_max_abs_z_m: <m>
   ```

   The values here are placeholders. Use only numbers from the holder's documentation.
5. `tango.bind_host: 0.0.0.0` and `tiled.host: 127.0.0.1` (both explained in the file's comments).
6. `devices.mapper.properties.analysis_url: http://127.0.0.1:8003/mcp`.
7. `configs/mcp.yaml` and `configs/mcp_operator.yaml` both have `http_host: 127.0.0.1`.

## 4. Windows Firewall

The Tango database now listens on every interface (so the endpoints on this PC reach it as `localhost`). **Tango
has no access control:** anything that reaches it can move the stage, unblank the beam, approve held calls, or
restart the approval device in another mode. Each device server also listens on a random port. Nothing in this
setup needs another machine to reach this PC on these ports. Connections on this PC (loopback) are not filtered
by the firewall.

1. The first time `run_servers.py` starts, Windows asks whether Python may accept connections from networks. The
   operator clicks **Cancel** (or leaves both boxes unticked). That keeps every device server unreachable from
   other machines. This dialog needs an administrator.
2. Block the fixed ports from other machines. In an administrator PowerShell:

   ```powershell
   New-NetFirewallRule -DisplayName "asyncroscopy: Tango DB not from network" -Direction Inbound -Protocol TCP -LocalPort 9094 -Action Block
   New-NetFirewallRule -DisplayName "asyncroscopy: Tiled and MCP not from network" -Direction Inbound -Protocol TCP -LocalPort 9091,8000-8003 -Action Block
   ```

3. If a named machine must reach Tango later (a remote monitor, say), replace the 9094 rule with an allow rule
   scoped to that machine's address (`-RemoteAddress <its IP>`), and also allow it Python's random device ports.
   This is a facility decision. Note that Windows Firewall applies a block rule before any allow rule.

Ask the facility's IT whether a network firewall in front of the PC already does this.

## 5. Start the stack

One PowerShell terminal per program, all in `C:\asyncroscopy`, all with `ASYNCROSCOPY_TILED_API_KEY` set
(step 2). Start them in this order and wait for each to say it is ready.

**Terminal 1: the device servers.**

```powershell
uv run --no-sync startup_scripts/run_servers.py --yaml configs/Spectra300.yaml
```

It prints an `OK` line per stage, then a summary with each device's PID and ready time, the Tiled URI
(`http://127.0.0.1:9091`), then `All asyncroscopy servers are ready.` and
`Leave this terminal open while you use the servers. Press Ctrl+C to stop them.`

"Ready" means every device answered a ping. It does **not** mean the instrument reached AutoScript: a device that
failed to connect is in FAULT and still counts as ready. Step 6 checks that.

**Terminal 2: the pyTEMlib analysis server.**

```powershell
uv run --no-sync python -m pyTEMlib.mcp_server --tiled http://127.0.0.1:9091 --api-key $env:ASYNCROSCOPY_TILED_API_KEY --port 8003
```

It prints `pyTEMlib MCP server at http://127.0.0.1:8003/mcp (Tiled: http://127.0.0.1:9091; ...)`. Leave
`--host` at its default, 127.0.0.1: the analysis server has no login.

**Terminal 3: the read-only pre-flight.** See step 6.

**Terminal 4: the operator endpoint** (SciAgentGUI uses it to approve held calls and to stop everything).

```powershell
uv run --no-sync startup_scripts/run_mcp.py --yaml configs/mcp_operator.yaml
```

It prints `Starting MCP server Operator_MCP`, its config, `tango: localhost:9094` and
`http: http://127.0.0.1:8001/mcp`, and then `MCP ready: <n> tool(s) registered`. Start it only after Terminal 1
is ready: an endpoint offers only the devices that answered when it started.

**Terminal 5, first session: the read-only monitor endpoint**, instead of the agent endpoint:

```powershell
uv run --no-sync startup_scripts/run_mcp.py --yaml configs/mcp_monitor.yaml
```

It offers only pure reads (`get_*`, `list_*`, `State`, `Status`), the data tools and the trace, on
127.0.0.1:8002. It prints `MCP ready: <n> tool(s) registered`.

**Later sessions: the agent endpoint**, once section B of the review is fixed:

```powershell
uv run --no-sync startup_scripts/run_mcp.py --yaml configs/mcp.yaml
```

It prints `http: http://127.0.0.1:8000/mcp` and `MCP ready: ...`. The approval mode is `approve`: every
high-risk call waits for the operator. Auto mode is refused on this stack.

## 6. Pre-flight (read-only)

Run it after Terminals 1 and 2 are ready, and again after the endpoints start:

```powershell
uv run --no-sync startup_scripts/preflight_hardware.py --yaml configs/Spectra300.yaml
```

It never moves, blanks or unblanks, acquires, writes an attribute or switches the approval mode. It refuses to
send any Tango command that the risk policy does not count as a pure read. It does open its own short AutoScript
connection to read the server's version; add `--no-autoscript-server` to skip that.

It prints one line per check: `PASS`, `FAIL`, `WARN` or `SKIP`, then `PRE-FLIGHT PASSED` (exit code 0) or
`PRE-FLIGHT FAILED: <n> check(s): ...` (exit code 1). It checks:

* the YAML says `simulated: false`, and `ASYNCROSCOPY_TILED_API_KEY` is set and not `secret`;
* the Tango database answers at 10.46.217.241:9094 and at localhost:9094;
* every configured device answers and is not in FAULT (its Status is printed when it is not ON);
* the AutoScript client version in this install, that the AutoScript port accepts a connection, and the
  server's version against the client's;
* the microscope reads: stage position, optical mode, high tension, column valves, beam blanked, and the beam
  current if the screen current was measured;
* the Approvals device is in `approve` and says the stack is not simulated;
* CEOS accepts a TCP connection (nothing is sent);
* Tiled answers at 127.0.0.1:9091 and accepts the API key;
* the analysis server at the Mapper's `analysis_url` answers MCP `initialize` and `list_tools`;
* each MCP endpoint in `configs/mcp.yaml` and `configs/mcp_operator.yaml` answers, or `SKIP` when it is not running
  yet (add `--mcp-yaml configs/mcp_monitor.yaml` to check the monitor too).

Expected warnings on the first run: `beam current ... not measured` (step 8 measures it).

What a FAIL usually means:

| FAIL | Likely cause |
| :-- | :-- |
| `tango database at localhost:9094` | `tango.bind_host: 0.0.0.0` missing, or an old database still running |
| `device asyncroscopy/instrument/default` FAULT, "Not connected to AutoScript" | wrong AutoScript port or host, AutoScript server not running, or a version mismatch |
| `autoscript server version` | the server is not 1.18: install the matching client wheels |
| `ceos at ...` | wrong CEOS address, or CEOS not running on the corrector PC |
| `tiled accepts the API key` | `run_servers.py` was started in a terminal with a different key: stop it, fix the key, start again |
| `analysis server` | Terminal 2 not running, or started with `--host` other than 127.0.0.1 |
| `approvals mode` | someone switched to auto earlier; it must be approve |

Do not continue past a FAIL. Fix it and run the pre-flight again.

## 7. First session: read only

Goal: confirm the stack sees the microscope as the operator sees it. Nothing moves and the beam is not touched.

1. The pre-flight passes.
2. The operator compares the pre-flight's stage position, optical mode, high tension and beam-blanked state with
   the microscope's own user interface. Units: x, y, z in micrometres in the pre-flight line, tilts in degrees.
3. Read the full state through Tango, read-only:

   ```powershell
   $env:TANGO_HOST = "127.0.0.1:9094"
   uv run --no-sync python -c "import tango, json; d = tango.DeviceProxy('asyncroscopy/instrument/default'); print(d.get_stage()); print(json.dumps(json.loads(d.get_parameters()), indent=1))"
   ```

4. If an assistant reads the instrument through MCP in this session, connect it to the monitor endpoint
   (127.0.0.1:8002), not the agent endpoint. `list_devices` shows the tool names; use only `get_*` and `list_*`.
5. Record the AutoScript server version, the AutoScript port, the CEOS address, the holder in use, and anything
   that disagreed with the microscope's own display. These close items A4 and A7 of the review.

## 8. Calibration (operator-run, after the holder limits are set)

These run as the operator, straight through Tango. They bypass the approval gate, so only the operator starts
them, at the microscope, watching. Do them in this order. Each prints a YAML block to paste into the digital
twin's `instrument.properties`. All take `--device` (default `asyncroscopy/instrument/default`), `--scan`
(`asyncroscopy/scan/default`), `--fov` in metres (default 2e-7), `--imsize` in pixels (default 256) and `--dwell`
in seconds (default 2e-6). They set the SCAN device's size, dwell and region, and the field of view, before
they start.

Set the Tango address in the terminal first:

```powershell
$env:TANGO_HOST = "127.0.0.1:9094"
```

1. **Screen current (beam current).** Needed before any dose budget, and by `dose_series`. The operator inserts
   the fluorescent screen and unblanks from the microscope's own user interface, then:

   ```powershell
   uv run --no-sync python -c "import tango; print(tango.DeviceProxy('asyncroscopy/instrument/default').get_screen_current(), 'pA')"
   ```

   It counts as measured only with the screen inserted. The operator then blanks and retracts the screen.
   Run the pre-flight: `beam current` now passes. Changing spot size, apertures, mode, high tension or the
   monochromator makes it stale. Do **not** run `calibrate_screen_current` or `set_screen_current`: they sweep the
   monochromator focus over a range copied from the simulator (review section C).

2. **Stage repeatability.** Moves the stage ±`--excursion` (default 2e-6 m) in x and y around the current position
   and back, `--repeats` times (default 8), imaging each time. Tilts stay where they are; set them to 0 first.
   Choose a `--fov` several times larger than the landing error you expect, on an area with clear features.

   ```powershell
   uv run --no-sync python -m asyncroscopy.calibration.stage_repeatability --fov 2e-6 --excursion 2e-6 --repeats 8
   ```

3. **Drift.** Moves the stage once by 1 µm in x, then takes `--frames` images (default 30) every `--interval`
   seconds (default 20), about 10 minutes.

   ```powershell
   uv run --no-sync python -m asyncroscopy.calibration.drift_series --frames 30 --interval 20
   ```

4. **Dose.** On a sacrificial area only. Images the same field `--frames` times (default 40). It reads the beam
   current from step 1; without it, it assumes 50 pA and the dose numbers are wrong.

   ```powershell
   uv run --no-sync python -m asyncroscopy.calibration.dose_series --frames 40
   ```

The scripts do not unblank the beam, and they do not blank it after their acquisitions (review B2). The operator
unblanks from the microscope's user interface before each script (an image taken blanked is only noise) and
blanks again as soon as it ends.

## Never run on the microscope

* `startup_scripts/check_stack.py`: digital twin only. It approves its own stage move to z = 0 and both tilts 0,
  switches the approval mode and runs two maps.
* `startup_scripts/check_gui_contract.py`: switches the approval mode and makes held agent calls.
* Anything in `benchmarks/`, and `startup_scripts/run_llm.py` or any agent against `configs/mcp.yaml`, until
  section B of the review is fixed.
* Anything that moves the stage before the holder limits are set (step 3.4): `move_stage`, `Mapper` jobs, the
  calibration scripts, `register_stage`.
* `calibrate_screen_current`, `set_screen_current`, `Disconnect`, 4D-STEM acquisition (no limits yet, review B3),
  and the CEOS write commands.
* `uv sync` without `--frozen`, or `uv run` without `--no-sync`.
* `run_servers.py` with `tango.reset_database_file: true`.

## Stop and emergency stop

**The microscope's own controls come first.** To stop in an emergency, use the microscope's user interface or
hardware: blank the beam, close the column valves, stop the stage there. asyncroscopy's stop is a second line.

* **Stop everything (asyncroscopy).** SciAgentGUI's **Stop everything** button, or the `stop_all` tool on the
  operator endpoint. It switches the approval mode to approve, voids unused approvals, blanks the beam and aborts
  running jobs on every device at once. Today it does **not** stop the stage, and it waits behind a command
  already running on the microscope device for up to 20 s (review B4). That is why the microscope's own controls
  come first.
* **Stop a job.** The job's `<Mapper or Tuner>_abort` tool, or `stop_all`.
* **Stop the software.** Press Ctrl+C in each terminal, the MCP endpoints and the analysis server first, then
  Terminal 1. `run_servers.py` stops every device server and Tiled. Stopping the software does not blank the beam
  or stop the stage: blank first.
* **After a crash.** Run `run_servers.py` again; it clears its own leftover processes. The approval mode survives
  restarts in `outputs/approvals/mode.json`, and this stack always starts in approve.
