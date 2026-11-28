# Permissions and the endpoint contract

Asyncroscopy decides, on the server, whether a tool call runs at once, waits for a person, or never runs. It
does so for every client of the agent endpoint alike: an app, `query_agent`, a workflow, a script. The monitor
endpoint, which offers only reads, is gated the same way with the same settings, so setting `reads` to `approve`
holds reads there too. A person
releases a held call on the operator endpoint. An app only shows the held call and the settings.

## Deviations from the 2026-11-28 contract

- **Stops and the contract tools are never held.** `endpoint_info`, `action_space`, `find_tools`, `call_tool`
  (it is checked on the tool it runs), `stop_all`, and every command that declares `stop` or declares a stop role
  (`beam.blank`, `probe.retract`, `safe_stop`), always run. A stop role the name rules inferred is not enough, so a
  command such as `gun_off` with no declaration is permissioned like any other. Their permission is `allow` with
  `source: "fixed"` and `settable: false`. Without this, setting the `reads` group to `approve` would hold
  `endpoint_info`, and no client could even read the endpoint; holding an emergency stop for a person is unsafe.
- **The permissions file has a `groups` and a `tools` section** under each `stack_id`:
  `{stack_id: {"groups": {...}, "tools": {name: permission}}}`.
- **Held calls live beside the permissions file**, in `<name>.state.json`, with the agent endpoint's tool
  list. The agent and operator endpoints are separate processes, so the operator endpoint reads both from there.
- **The default path** is `$ASYNCROSCOPY_PERMISSIONS_FILE`, else `outputs/permissions/permissions.json`, when
  `mcp.permissions_file` is not set (the same pattern as the dose ledger), so tests and a second stack can keep
  their own. A relative `mcp.permissions_file` is read from the directory of the config that names it, not from
  the directory a server was started in, so the shipped configs name `../outputs/permissions/permissions.json`
  and every endpoint of a stack finds the same file wherever it was started.
- **A workflow resumes when asked.** A run waiting on a held call stays `waiting` until `resume_workflow` is
  called after the approval; nothing polls the operator endpoint.
- **Init, Kill and RestartServer stay hidden** on every endpoint, as before; `Disconnect` is listed on the
  agent endpoint with permission `never` and refused. The operator endpoint lists none of the four.
- **A job reservation's `get_reservation` stays `allow`** on `Mapper` and `Tuner`: it only reads a reservation.
- **A TIFF acquisition has no `primary_array`**: the file is the one array.
- **The agent MCP server reports `role: "agent_server"`** in its `endpoint_info` (it had no `endpoint_info`
  before).

## Permission values

| Value | Meaning |
| :-- | :-- |
| `allow` | runs at once |
| `approve` | held until a person approves that exact call (same tool, same arguments, same caller) |
| `never` | refused; cannot be set |

Tools are in two groups: `reads` (`acts` is false) and `acts` (`acts` is true). Each group has a default,
`allow` until a person changes it.

A tool's effective permission, first match wins (`asyncroscopy.safety.permissions.permission_and_source`):

| Order | Rule | `source` |
| :-- | :-- | :-- |
| 1 | `never` for the fixed hard-block set: Tango `Init`, `Kill`, `RestartServer`, and `Disconnect` on every class (`policy.hard_blocked`); `allow` for the tools that are never held (above) | `fixed` |
| 2 | the per-tool setting a person saved | `tool` |
| 3 | `approve` when `mcp.needs_approval` lists it | `config` |
| 4 | the group default | `group` |

Commands a device reports `unsupported` stay hidden from the agent endpoint, as before. Physical-safety checks
are not permissions and are unchanged: stage limits, `max_diffraction_shift`, the camera series limits,
`job_limits` and the dose checks, `allow_corrector_writes`, the turn guard, the Host/Origin guard, loopback
binding and token auth.

## Configuration

In the MCP YAML (`configs/mcp_dt.yaml`, `configs/mcp.yaml`):

```yaml
mcp:
  needs_approval:            # {Tango class or "*": [command, "Class.command" or glob]}
    DATA: [configure, copy_and_register_remote_file, reconcile, register_path, start_tiled_server, stop_tiled_server]
    CEOSCorrector: [correct, correct_aberration, measure_aberrations, measure_c1a1, acquire_tableau]
    DigitalTwinCorrector: [correct, correct_aberration, measure_aberrations, measure_c1a1, acquire_tableau]
  permissions_file: ../outputs/permissions/permissions.json   # relative to this config's directory
  approval_ttl_s: 600        # how long a held call waits, and how long an approval waits for its call
  urls:                      # the stack's other endpoints, reported in endpoint_info; leave out what does not run
    operator: http://127.0.0.1:8001/mcp
    agent_server: http://127.0.0.1:8002/mcp
    workflow_operator: http://127.0.0.1:8004/mcp
    tiled: http://localhost:9091
    analysis: http://127.0.0.1:8003/mcp
```

A native tool of the agent endpoint (`run_job`, `list_data`, ...) is named by its tool name under `"*"`
(`"*": [run_job]`). The agent endpoint refuses to start when an entry names nothing: no command of its class
(offered to agents or not) and, under `"*"`, no native tool either. An entry for a class the stack does not run is
not checked, because one config names the twin's and the hardware's classes alike; `tests/agents/test_agent_configs.py`
checks the corrector entries against the classes themselves.

### Corrector writes

Every corrector command that writes or drives the corrector (`correct`, `correct_aberration`,
`measure_aberrations`, `measure_c1a1`, `acquire_tableau`, all declared on `CORRECTOR` and inherited by
`CEOSCorrector` and `DigitalTwinCorrector`) is in `needs_approval`. The Tuner's `corrector` strategy writes the
corrector itself, through Tango, inside a job, so the endpoint holds the job instead: a job device reports, in
its `reserve` command's `action_hints`, `approval_if: {plan field: {"values": [...], "default": ...}}`
(`JobDevice.approval_plans`). The Tuner reports `{"strategy": {"values": [every registered strategy that needs
the corrector], "default": "vendor_autofocus"}}`. A `run_job` or `<device>_reserve` call whose plan matches is held
as if `needs_approval` listed the tool, so a person approves the job as submitted, with its plan, and a person's
own setting for that tool still comes first. `endpoint_info` reports `run_job` as `allow` because most plans run
at once; a client handles a held reply from any tool. Resuming an aborted job (`run_job(resume=...)`) is not held
again: its plan was approved when it was submitted. `allow_corrector_writes` is unchanged.

`needs_approval` replaced `blocked_functions`, and `urls.analysis` replaced `analysis_url`; `run_mcp.py`
refuses a config that still has either. Keep `permissions_file` outside every directory Tiled serves, and give
the agent and operator endpoints of one stack the same file (an operator endpoint started with
`mcp.operator_port` gets the agent endpoint's). The file is keyed by `stack_id`, so stacks can share it. The
agent MCP server (`run_llm.py`) does not read the file: its `query_agent` and workflows reach the instrument only
through the agent endpoint, whose gate holds their calls.

The agent MCP server (`run_llm.py`) takes the same `urls` under `agent:`.

## What the agent endpoint reports

`endpoint_info`:

- `tools[name].permission`: `allow`, `approve` or `never`, beside `acts`, `risk` and `role`;
- `permission_groups`: `{"reads": ..., "acts": ...}`;
- `urls`: `{"operator", "agent_server", "workflow_operator", "tiled", "analysis"}`, each a URL from the config or null;
- `command_timeout_s`: the endpoint's own limit on one Tango command (30), which a client's call timeout follows;
- `device_kinds`: `{device prefix: [kind, ...]}`, kinds from `stem`, `probe`, `envelope` (runs jobs), `map` (a Mapper), `stage` (moves a stage);
- `stack_fingerprint` (every endpoint role): which software, device code and device configuration serve this
  endpoint, all read when `endpoint_info` is called (`asyncroscopy/mcp/fingerprint.py`):
  - `software`: `{"package": "asyncroscopy", "version", "commit", "uncommitted_changes"}`. `version` is the installed
    package's; `commit` is `git rev-parse HEAD` of the checkout the endpoint runs from, and `uncommitted_changes` whether
    tracked files differ from it. Both are null outside a git checkout.
  - `devices`: `{device prefix: {"class", "module", "module_sha256"}}`. `class` is the Tango class. `module` is the
    device's `module_name` Tango property, which `run_servers.py` writes with the module it starts the device with
    (`python -m <module>`); `module_sha256` is the SHA-256 of that module's file as this endpoint's Python finds it.
    Both are null for a device started another way. Only the module that defines the class is hashed: code it imports is
    covered by `software`.
  - `config_digest`: the SHA-256 of every Tango property of every device (`{prefix: {name: [values]}}`, sorted), as the
    endpoint reads them from the Tango database. A property left in a kept database counts, because the device reads it too.
  - `unreadable_properties`: the devices whose properties could not be read (a test context has no database).
  - `simulated`: `{"value", "established_by"}`. The value is `simulated` above; `established_by` says it is declared by
    the devices' `simulated` property, not proved.
  - `digest`: the SHA-256 of all of the above. Two runs are on the same stack when their digests are equal; workflow
    rehearsals are matched to a stack by it.

Every listed MCP tool carries `ToolAnnotations` with `readOnlyHint = not acts` and `destructiveHint = acts`.

The agent MCP server (`role: "agent_server"`) reports `urls` and `services.workflows`: every workflow tool it
offers under its own key: `list`, `get`, `run`, `status`, `stop`, `resume`, `answer`, `list_runs`,
`list_blocks`, `propose`. It reports only what it offers, so there is no `skills` service.

## A held call

A call of an `approve` tool on the agent endpoint does not run. The reply has `isError: true`, text saying it
waits for a person, and:

```json
{"status": "approval_required", "request_id": "<uuid>", "tool": "<name>", "arguments": {},
 "arguments_sha256": "<hex of the canonical JSON>", "acts": true, "risk": "high",
 "expires_at": "2026-11-28T12:10:00Z", "requested_by": "<caller or null>"}
```

The arguments are first validated against the tool's signature, as FastMCP will validate them: defaults are
filled in and each value takes its declared type, so `{"fov": 1}` and `{"fov": 1.0}` for a float are one call. A
held call's `arguments` are these. They are then compared as canonical JSON (sorted keys, no spaces), so the same
call is the same request however it is sent, also through `call_tool`. A call of an `approve` tool whose arguments
do not validate is neither held nor run: it gives `{"status": "invalid_arguments", "tool"}` with `isError: true`. A request also belongs to its caller (`requested_by`: the token's
subject with `mcp.auth`, else the `X-Asyncroscopy-Caller` header): another caller's identical call makes its own
request and cannot use this one's approval. A caller whose token may not run the tool at all is refused for that
before the gate, so it never makes a request.

- Sending the same call again while it waits gives the same `request_id`; no second request is made.
- After a person approves it, the same call runs **once**: the approval is used up, and the next identical
  call is held again. An approval not used within `approval_ttl_s` expires.
- After a denial, the same call gives `{"status": "denied", "request_id", "reason"}` with `isError: true`
  until the denial expires.
- A `never` tool gives `{"status": "never", "tool"}` with `isError: true`.
- A call of a hardware run that a workflow sign-off pre-approved (`preapprove_run`) runs without a request when the
  run's rehearsal made the same call: the same tool and the same validated arguments, sent with the run's id
  (`X-Asyncroscopy-Run`) and its secret key (`X-Asyncroscopy-Run-Key`), and no more times than the rehearsal made it
  (each pre-approved call has a `count`, used up one pass at a time). Its trace event carries `approved_by`
  (`workflow_signoff:<workflow>@<hash>`) and `approver`. Any other call of that run is held as above, and the held
  reply and its `list_pending_approvals` entry add `signoff: {"approved_by", "approver", "run_id", "why_held"}`.
  Such a request keeps the run id and the digest of the run's key: a person's approval of it is used only by a call
  of that run with that key (a new grant for the resumed run moves it to the new key), so another client's identical
  call cannot use it. A pre-approval lives in `<name>.state.json` beside the held calls, so the gate honours it
  although the operator endpoint wrote it. It expires `approval_ttl_s` after the run's last call with its key, at most
  12 hours (`max_run_preapproval_s`) after it was granted. [Twin to instrument](twin_to_instrument.md) is its contract.

`query_agent` treats a held call as a pause: the agent is told not to call it again, and the reply starts
with "Waiting for a person to approve:" and each held tool, its arguments and its request id. A workflow stops
as `waiting`, with the held call in `approvals`; once a person approves it, `resume_workflow(run_id)` sends
it again and the run goes on.

LangGraph runs the waiting node again from its first line. The calls that node completed before it stopped are
not sent again: the run keeps a call journal (`runs/<run_id>.journal.json` beside the run record, so it survives a
restart), keyed by node, the call's place in the node, the tool and its canonical arguments, and a call that matches
gets its recorded reply. The held call is the first one sent again, and it uses its approval; an earlier approval
is never asked for again. When something ran after the node stopped (its own cleanup, the runner's `always` tools
such as blanking the beam, or its restores), it may have undone what the recorded calls did, so the journal is
dropped and the node runs again from its first line. The journal is removed when the run ends or is stopped.

## The operator endpoint

These tools are in the operator endpoint's `action_space` (device null), beside `stop_all`:

| Tool | Does |
| :-- | :-- |
| `list_pending_approvals()` | `[{request_id, tool, arguments, acts, risk, requested_at, expires_at, requested_by}]`, oldest first |
| `approve_call(request_id)` | approves one held call |
| `deny_call(request_id, reason)` | denies one held call |
| `get_permissions()` | `{"groups": {...}, "tools": {name: {"permission", "source", "acts", "risk", "description", "settable"}}}` |
| `set_permission(tool, permission)` | `allow`, `approve`, or `default` to clear the setting; `never` is refused, and so is a fixed tool |
| `set_group_permission(group, permission)` | `reads` or `acts`, to `allow` or `approve` |
| `preapprove_run(run_id, run_key_sha256, calls, approved_by, approver, workflow, hash, rehearsal_run_id)` | for a workflow server, not a model: pre-approves, for one hardware run, the calls a signed-off workflow's rehearsal made, each `{"tool", "arguments", "count"}` ([Twin to instrument](twin_to_instrument.md)); needs the `preapprove` role |
| `end_run_preapproval(run_id)` | for a workflow server: removes that run's pre-approval when it ends or is stopped |

`source` is `fixed`, `tool`, `config` or `group` (the table above), so a client can show "Default" when it is
`config` or `group`.

The operator endpoint is the person's console, and it is not gated by design: nothing on it is held, and it is
where held calls are released. Who may reach it decides who may approve:

- **Without `mcp.auth`** any process on the computer can reach the loopback-bound operator endpoint and approve,
  just as any local Tango client can drive the devices directly. Run agents only where that is acceptable.
- **With `mcp.auth`** the six person's tools above need the token role `approve` (`descriptors.person_only_natives`,
  read by `endpoint_auth.required_role`), also through `call_tool`. `preapprove_run`, `end_run_preapproval` and a
  twin's `export_ground_truth` need the role `preapprove` instead (scope `instrument:preapprove`), which grants
  nothing else, not even `read`: it is the workflow server's `ASYNCROSCOPY_OPERATOR_TOKEN`, so the process that hosts
  `query_agent` holds no token that approves a held call. Reading the pending list needs `approve` too: it is part
  of the console, and an agent learns what it waits for from its own held reply. Agent tokens must never hold the
  `approve` scope; an agent's scopes are `read` and `operate`.

## Acquisition files

Every acquisition the DATA writer saves records, in the file's and each dataset's attributes,
`acquisition_type`: the action role of the command that acquired it (`acquire.image`, `acquire.spectrum`,
`acquire.other`, ...), as `asyncroscopy.utils.commands.command(role=...)` declares it or as a YAML instrument
gives it. An `.h5` file also records `primary_array`, the dataset shown by default. The old free-form value
(`stem_image`, `spectrum`, ...) stays the record field `kind`. A pyTEMlib stitched map's metadata reports
`poi_kinds: [{"kind", "label"}]`, the kinds of point kept in its spatial cells.

The MCP endpoint adds a preview to the reply of every command whose role is `acquire.*`.
