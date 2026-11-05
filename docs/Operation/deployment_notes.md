# Deployment notes for hosting the stack (ATHENA WG1)

This page is for whoever packages the digital twin (or an instrument stack) as
containers on a cluster such as WG1's Kubernetes/OpenShift. It lists every
process the stack runs, what it needs and how to check it. No container image or
manifest ships with this repository yet, and none has been run; this page records
what such a deployment has to set, so writing one is mechanical.

## The processes

| # | Process | Command | Listens on | Stateful? |
| :-- | :-- | :-- | :-- | :-- |
| 1 | Device stack: Tango database, every device server, Tiled (started by the DATA device), job devices | `python startup_scripts/run_servers.py --yaml configs/digital_twin_mapping.yaml --isolated outputs/stacks/twin --log-to-stdout` | Tango database `tango.port` (9094); Tiled `tiled.port` (9091); each device server on a port omniORB picks at random | Yes: Tiled catalog and acquisitions, approval mode, job state, traces |
| 2 | Agent MCP endpoint | `python startup_scripts/run_mcp.py --yaml configs/mcp_dt.yaml` | `mcp.http_port` (8000) | Writes its trace file only |
| 3 | Operator MCP endpoint (approvals, mode switch) | `python startup_scripts/run_mcp.py --yaml configs/mcp_operator.yaml` | `mcp.http_port` (8001) | Writes its trace file only |
| 4 | pyTEMlib analysis server (stitching, scoring) | `python -m pyTEMlib.mcp_server --tiled http://<tiled>:9091 --host 0.0.0.0 --port 8003` | 8003 | No: results go to Tiled |

The launchers start their children with `uv run --no-sync python` when uv is on
`PATH` (or `UV` names it), and with the launcher's own Python otherwise, so an
image with a plain virtual environment and no uv works.

**Startup order.** Start 1 and wait until it prints `Leave this terminal open`
(every device answers, Tiled is up). Only then start 2 and 3: an MCP endpoint
offers the devices that existed when it started. 4 can start any time; it reaches
Tiled when a job asks.

**Stopping.** Every launcher stops its children on SIGTERM. `run_servers.py` exits
1 when startup fails or when any of its servers exits after startup, and prints
that server's last output; `run_mcp.py` exits with its server's code (1 if that
was 0) when the server stops on its own. A supervisor's restart policy therefore
sees real failures.

## Settings

Everything a process needs is in its YAML file or its command line; the
environment variables below override or add to that.

| Setting | Where | Default | Container value |
| :-- | :-- | :-- | :-- |
| Tango database address clients use | `tango.host`, `tango.port` in the stack YAML **and** both MCP YAMLs | `localhost:9094` | the database's service name when the MCP endpoints run in another pod; `localhost` when they share a pod |
| Interface the Tango database binds | `tango.bind_host` in the stack YAML | `tango.host` | `0.0.0.0` when `tango.host` is a service name (that name's address is not on the pod, so the database cannot bind it) |
| Tiled bind address and port | `tiled.host`, `tiled.port` | `localhost:9091` | `localhost` if only the same pod reads Tiled; `0.0.0.0` otherwise |
| Tiled address given to clients | `ASYNCROSCOPY_TILED_PUBLIC_URI` (env, process 1) | the bind address (a wildcard reads as `127.0.0.1`) | `http://<tiled service>:9091` when Tiled binds `0.0.0.0` and other pods read it |
| Tiled API key | `ASYNCROSCOPY_TILED_API_KEY` (env, processes 1 to 3); `--api-key` or the same variable (process 4) | `secret` | from a Secret; the same value for all four |
| Acquisitions and Tiled catalog | `tiled.acquisition_dir` | `outputs/tiled_acquisitions` | on the persistent volume |
| Tiled server logs | `ASYNCROSCOPY_LOG_DIR` (env) | `outputs/logs` | anywhere writable |
| Approval mode file, job state, traces | device properties `mode_file`, `state_directory`, `trace_path`; `mcp.trace` | under `outputs/` | on the persistent volume (the approval mode must survive a restart) |
| Tango database file and launcher process records | `--isolated DIR` | project root and `./.processes` | `outputs/stacks/<name>` (recreated at every start: `reset_database_file: true`) |
| MCP bind address and port | `mcp.http_host`, `mcp.http_port` | `127.0.0.1` | `0.0.0.0` (allowed on a `simulated: true` stack; a hardware stack also needs `mcp.allow_remote_bind: true` and `mcp.auth`, or `run_mcp.py` refuses to start) |
| MCP bearer tokens | `mcp.auth` (`jwks_url` or `public_key_file`, `issuer`, `audience`) in the MCP YAML; clients send `ASYNCROSCOPY_MCP_TOKEN` | none: no token check | the token service's JWKS URL, issuer and audience ([remote_access.md](remote_access.md)) |
| Analysis server the Mapper and agent endpoint use | `analysis_url` on the Mapper device and in `mcp_dt.yaml` | `http://127.0.0.1:8003/mcp` | the analysis service's address |
| Simulated or hardware | top-level `simulated:` in the stack YAML | false when absent | `true` for every twin |

Relative paths resolve against the project directory (the servers' working
directory), so in an image the project directory, or at least its `outputs/`,
must be writable by the container's user. For OpenShift's arbitrary UID, make the
project directory group-writable by group 0 and mount the persistent volume at
`<project>/outputs`. No process writes to the home directory except uv's cache;
set `UV_CACHE_DIR` to a writable path if uv is in the image.

## Health checks

| Process | Check | Expected |
| :-- | :-- | :-- |
| 1 (Tiled) | `GET http://<tiled>:9091/healthz` | 200 |
| 1 (Tango) | `python -c "import tango; tango.DeviceProxy('tango://<host>:9094/asyncroscopy/approvals/default').ping()"` | exit 0 |
| 2, 3, 4 | `POST /mcp` with an MCP `initialize` request (below) | 200 |
| All, end to end | `python startup_scripts/check_stack.py --agent <url> --operator <url> --pytemlib <url>` | every line PASS; it moves the twin's stage and briefly switches approval mode, so it is a smoke test, not a probe |

A plain `GET /mcp` returns 400 (no MCP session), so an HTTP probe that only does
GET cannot check an MCP endpoint; use a TCP probe or this request:

```bash
curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}'
```

## One pod or several

The configs assume one host: the MCP YAMLs name the Tango database at
`localhost`, the Mapper reaches pyTEMlib at `127.0.0.1:8003`, and Tiled binds
`localhost`. Running all four processes as containers of **one pod** (they share
`localhost`) needs only the bind addresses changed to `0.0.0.0` for the
endpoints a Service exposes.

Splitting them across pods needs, in addition, `tango.host` set to the database's
service name with `tango.bind_host: 0.0.0.0`, Tiled on `0.0.0.0` with
`ASYNCROSCOPY_TILED_PUBLIC_URI`, and the analysis URLs set to the analysis
service. Keep the Tango database and the device servers in one pod
(`run_servers.py`): device servers register their own address and a random port
with the database, and clients connect to that address directly, not through a
Service. A pod's IP works for that on a flat pod network; a NetworkPolicy must
then allow those ports. (omniORB's `ORBendPointPublish` environment variable can
set the published address; not tried here.)

## Who provides what

| WG1 (the cluster) | The instrument owner (this repository and its configs) |
| :-- | :-- |
| The cluster, namespaces, storage classes and persistent volumes | A stack YAML per instrument (`simulated: true` for twins) with the Approvals device, and its two MCP YAMLs |
| Token minting and verification; the endpoint proxy with TLS in front of the agent endpoint; Routes or Ingress | Which endpoint is public: only the agent endpoint. The operator endpoint stays cluster-internal (it can approve held calls) |
| Secrets storage (the Tiled API key) | The image contents: Python 3.12, `uv sync` with the extras the stack needs (no `agent` or `ollama` extras for the servers), the pinned pyTEMlib |
| Monitoring, log collection, scheduling, the user portal | The process list, settings and health checks on this page |

The MCP server verifies bearer tokens itself when its YAML has an `mcp.auth`
section (`asyncroscopy/mcp/endpoint_auth.py`, [remote_access.md](remote_access.md)):
signature, expiry, issuer and audience, with the token's scopes deciding which
tools it sees. Without `mcp.auth`, an agent endpoint reachable through a Route
has no authentication of its own: anyone who reaches it can drive the stack. A
hardware stack refuses that combination (a remote bind without `mcp.auth`); a
twin allows it, so set `mcp.auth` or keep it behind WG1's endpoint proxy. Hardware
stacks start in approval mode `approve` and refuse `auto`
([approval_mode.md](approval_mode.md)).
