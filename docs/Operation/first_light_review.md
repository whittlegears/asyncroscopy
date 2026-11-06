# Spectra 300 first-light readiness review

Date: 2026-11-03. Code at asyncroscopy `6743870`. Config: `configs/Spectra300.yaml`, `configs/mcp.yaml`, `configs/mcp_operator.yaml`.

This is a read-only review. Nothing here has run on the real microscope. API names and units were checked against the AutoScript 1.18 client wheels in `stubs/AutoScript_v_1.18`, not against a live instrument. Items marked *verified* were re-read in the code after the reviewers reported them.

Verdict: **not ready for a session where an agent drives the microscope.** A supervised, read-only first session (connect, read stage and state) is possible once items A1 and A2 are fixed.

## A. Fix before the first power-on session

| # | Problem | Where | What happens on the instrument |
|---|---------|-------|-------------------------------|
| A1 | The Tango database listens only on 10.46.217.241, but the operator and monitor endpoints look for it on `localhost`. *verified* | `Spectra300.yaml` has no `tango.bind_host`; `run_servers.py:994`; `mcp_operator.yaml:14`, `mcp_monitor.yaml:9` | The operator endpoint cannot connect. It is the only place a person can approve a held call, and hardware starts in approve mode, so no high-risk call can ever run. Fix: set `tango.bind_host: 0.0.0.0`, or point both endpoints at 10.46.217.241. |
| A2 | The startup script prints "All asyncroscopy servers are ready" even when the instrument, stage or aperture failed to connect to AutoScript (FAULT state). If the AutoScript import itself fails, the stage only warns and stays ON with cached values. *verified* | `run_servers.py:690` (ping only); `stage_autoscript.py:57-59` | The operator believes the stack is live when it is not. Fix: a read-only hardware pre-flight that reads every device's State/Status, the AutoScript version, CEOS reachability, Tiled health and the analysis server, and moves nothing. |
| A3 | `check_stack.py` has no guard against hardware. It approves its own `move_stage` to `[1e-8, -1e-8, 0, 0, 0]` (z and both tilts to 0) and runs two maps. The README offers it as "check the whole stack". *verified* | `check_stack.py:170-187`; `README.md:27-28` | Run on the microscope, it drives the stage. Fix: refuse to run when the stack is not simulated. |
| A4 | The CEOS corrector's timeout default is 120000 seconds (about 33 hours), and the config does not set the CEOS host. *verified* | `corrector_ceos.py:18-31` | A stalled CEOS reply freezes the corrector device for a day and a half. The default host is the microscope PC, but CEOS usually runs on the corrector PC. |
| A5 | Address mismatches for Tiled and the analysis server. Tiled binds only the LAN address, while the documented pyTEMlib command uses `localhost:9091`. The Mapper's `analysis_url` is 10.46.217.241:8003, but pyTEMlib binds 127.0.0.1 by default. | `Spectra300.yaml:42,90`; `README.md:25`; pyTEMlib `__main__.py:15` | Stitching and scoring fail at the end of every map. Fix: use 127.0.0.1 for the Mapper's `analysis_url` (the analysis server has no auth and should stay local). |
| A6 | The install needs internet access. pyTEMlib comes from a GitHub URL, `uv.lock` is not shipped, and plain `uv run` re-syncs on every start. | `pyproject.toml:136`; `.gitignore:105` | Fails on a firewalled microscope PC. Where it works, it installs untested versions. Fix: ship a lock file or a wheelhouse, and start with `--no-sync`. |
| A7 | The AutoScript client comes from the repo's 1.18 wheels. Nothing checks that they match the server version, and the config uses port 9095 where AutoScript's default is 7521. | `pyproject.toml:127-130`; `Spectra300.yaml` | A version mismatch or wrong port shows up only as FAULT (see A2). Confirm the server version and port with the facility before the session. |

## B. Fix before an agent drives the microscope

| # | Problem | Where | What happens on the instrument |
|---|---------|-------|-------------------------------|
| B1 | Stage moves can hit the pole piece. Position and tilt move together through `absolute_move`; the client's `absolute_move_safe` sets tilt to zero first and is not used. No holder is configured, so the generic limits apply (alpha ±35°, beta ±20°, z ±0.5 mm), and z and tilt are never checked together. *verified* | `stage_autoscript.py:109`; `stage.py:18,50-56` | An approved move to high z with high tilt is accepted. Fix: use the safe move, configure the fitted holder's limits, and add a combined z/tilt check. |
| B2 | The beam is not blanked after direct acquisitions or beam placement. It stays unblanked during the file save, parked on one spot. An approved `unblank_beam` stays on indefinitely. *verified* | `auto_script.py:306-311, 438-455`; Mapper/Tuner unblank around `acquire_scanned_image` (`map_profiles.py:197`, `tuner.py:386`) | Uncounted dose on one spot of the sample after every frame, and during every Mapper tile save. Fix: blank in `finally` after each acquisition, before the save. |
| B3 | 4D-STEM acquisition has no limits. There is no `check_scan`, so a scan of up to 8192×8192 is accepted. It uses the STEM dwell time (default 1 µs) as the camera exposure, and has no abort or dose check. With the defaults that is 262,144 camera frames with the beam parked. *verified* | `auto_script.py:406-428` | A run that cannot be stopped, which also blocks the e-stop (B4). Fix: check the scan, set a separate camera exposure with a minimum, check for abort, or block the command until first light. |
| B4 | The e-stop only blanks the beam. It waits behind any command already running on the microscope device and gives up after 20 s. Nothing stops the stage. *verified* | `mcp_server.py:58,1064-1147`; `map_profiles.py:174` | E-stop during a long acquisition: the beam stays on and the stage keeps moving. Fix: a stop path that does not queue behind a long command, and a stage stop. |
| B5 | Direct Tango access bypasses every gate. Approvals and job limits exist only in the MCP layer, and the Tango database is on the LAN without access control. *verified* | `approvals.py:14-15` | Any machine on the network can move the stage, unblank the beam, approve requests, or change the Approvals `simulated` property and restart it into auto mode. Fix: Tango access control, or firewall 9094 to the microscope PC. |
| B6 | The operator endpoint approves everything and has no login. Any process on that machine counts as the operator. *verified* | `mcp_server.py:853`; `mcp_operator.yaml` | A script or an agent with shell access on that PC can approve its own held calls. Keep it on 127.0.0.1, and do not give the agent a shell on the microscope PC. |
| B7 | Corrector writes reach the agent directly. `correct_aberration`, `measure_c1a1` and `acquire_tableau` skip `allow_corrector_writes`, the step bounds and the sign check. The CEOS client also never checks the JSON-RPC `error` field, so a refused correction reads as success. | `corrector.py:43-61`; `corrector_ceos.py` | One approval of a raw string changes the aberration corrector. Fix: add them to `blocked_functions` in `mcp.yaml` until the sign test, and check the error field. |
| B8 | No hardware timeouts. `timeout_seconds: 120` is read and never used. The microscope device gives up on the stage after 12 s. *verified* | `auto_script.py:76,222` | A long tilt reports failure while the stage is still moving, and the agent plans from a wrong position. AutoScript calls can block forever. |
| B9 | Limits are enforced only at reserve. Nothing re-checks time or dose while a job runs. A map without a dose budget does not re-check the beam current at start or resume. Leases and pause limits default to 0 and are not set. | `job_device.py:910-946`; `mapper.py:395`; `mcp_server.py:1292,1322` | A "4 hour" map on the real stage can run much longer, and keeps running after the agent crashes. A current change between reserve and a late approval goes unnoticed. |

## C. Correctness problems that fail loudly or mislabel

- Beam tilt is documented as radians. AutoScript's `beam_tilt` is unitless, −1 to +1 of full scale, with no software limit. Diffraction shift is documented as radians but written to `image_shift` (metres).
- `optics.acceleration_voltage` is deprecated and removed in AutoScript 1.21 (`auto_script.py:595,663`). On 1.21 or later, the beam current never counts as measured, so every dose budget is refused.
- Silent no-ops:
  - an unknown column-valve state only prints (`auto_script.py:475`);
  - an unknown screen position does nothing (`auto_script.py:516`);
  - a nonzero beta on a single-tilt holder is dropped after passing the limit check (`stage_autoscript.py:102-107`);
  - `DualAxis` and `Polara` holders are treated as having no beta.
- `set_screen_current` silently runs a calibration that sweeps the monochromator focus from 10 to 150 V, a range copied from the fake, and fits a degree-11 polynomial through 15 points.
- Aperture position writes have no limits. The default mechanism is whichever one `get_available()` lists first.
- No reconnect after an AutoScript server restart. The device stays ON while every call fails.
- Reading `manufacturer` errors, because `_manufacturer` is never set on `AutoScriptMicroscope`. *verified*
- EDS is hard-coded to Super-X, 5 eV dispersion and 3 µs shaping.
- Camera images and spectra are saved without frame attributes, so they never reach the dose ledger.
- Tuner: OptiSTEM, which the config names as the strategy, is refused in two places: the vendor dose is not calibrated, and AutoScript has no stigmator. In practice only image-metric focus works.
- `Microscope_Disconnect` is exposed. After it, even the e-stop's blank fails.

## D. Network and documentation

- Tiled runs with `--public` and the API key `secret` on the LAN address. Anyone on the facility network can read the data, and anyone with the well-known key can write it. The same key must be set for run_servers, both MCP endpoints and pyTEMlib; no doc says so.
- The analysis server has no login. Its `--api-key` is only the Tiled key.
- `gemma-llm.yaml` and `qwen-llm.yaml` bind the agent server to `0.0.0.0:8002`, with no login, on the same port as the monitor endpoint.
- Windows Firewall prompts (admin needed) the first time 9094 and 9091 bind on the LAN. Device servers use random ports, so remote clients need wide rules.
- Stale docs:
  - README calls `mcp.yaml` "not hardware";
  - `run-servers.md` says to set `http_host: 0.0.0.0` with no auth, which the code now refuses, and lists only some devices;
  - `deployment_notes.md` predates `endpoint_auth`.
- The MCP endpoints discover devices once at start. Started before the device servers are ready, they are missing tools. If Tango is down, `stack_simulated` treats the stack as hardware, which is the safe direction.

## E. What already holds

- With `simulated: false`, approvals refuse auto mode, and the gate fails closed (`approvals.py:95,175`; `mcp_server.py:878`).
- Agent jobs with no measured beam current are refused, not run on an assumed current.
- A failed AutoScript connect sets FAULT, and calls raise NotConnected. Camera and EDS do not fall back to simulated data.
- `reset_database_file: false`: startup deletes nothing outside `outputs/` and the Tango log folder.
- Member names and units in `tests/fake_autoscript.py` match the 1.18 client (stage in m/rad, screen states, monochromator focus in V, current in A). Its gaps:
  - camera acquisition accepts any exposure;
  - stage moves are instant;
  - the blanker starts blanked, which hides B2;
  - hardware ranges are placeholders.

## Suggested order

1. A1–A7 so a supervised, read-only session can start and its state can be trusted.
2. B1–B4, B7 and B8 in code, with a test for each against the fake (and the fake fixed where it hides the problem).
3. B5, B6 and D in the deployment: firewall, access control, keys, bind addresses. These are facility decisions.
4. B9 and C as follow-ups once the first session has measured the real numbers.
