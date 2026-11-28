# Twin to instrument

An agent writes a workflow, the workflow is rehearsed on a digital twin, a person signs it off, and it runs on the
real instrument the twin is paired with. This page is the contract an app builds against. The design is in
SciAgentGUI's `docs/internal/TWIN-TO-INSTRUMENT-PLAN.md`. [Workflow approval](workflow_approval.md) and
[Permissions](permissions.md) still hold; this page adds to them.

What the sign-off means: **it covers the calls exactly as rehearsed, for one run.** When an approved workflow runs on
a real instrument, every instrument call its passing rehearsal made counts as approved for that one hardware run. A
call of the same tool with the same canonical, validated arguments runs without a hold. Any other call is held for a
person as usual. A `never` tool stays refused. Physical range checks (stage limits, job limits, dose checks, the turn
guard) are not permissions and still apply to every call.

## Changes after review (2026-11-28)

These shapes changed after an adversarial review. An app built against the earlier text must change:

1. **`approve_workflow` takes `rehearsal_run_id: str` (required on the workflow operator endpoint).** Send
   `review_workflow`'s `rehearsal.run_id`, from a review with the same `instrument_url`. It is refused (`ok: false`,
   `error` says why) unless that run is still the rehearsal the review would show: the latest passing rehearsal of
   this exact code that counts for `instrument_url`. A newer passing rehearsal, or a changed instrument, means
   review again. The approval stores that rehearsal in `approval.rehearsed`: `run_id`, `at`, `inputs`, `stack_id`,
   `stack_fingerprint`, `calls_made` (with the mapping below) and `actions`. Later rehearsals and the library keeping
   only the last 10 rehearsals no longer change what was signed off.
2. **Inputs must match.** A hardware `run_workflow` of an approved workflow whose approval carries a rehearsal is
   refused (`state: "refused"`, before anything is opened) unless its inputs, with the workflow's defaults filled in,
   equal `approval.rehearsed.inputs` (canonical JSON). Send `review.rehearsal.inputs`; it is the same object as
   `approval.rehearsed.inputs`. Other inputs need a new rehearsal and a new sign-off.
3. **The call mapping in `review_workflow`.** The twin's and the instrument's tools have different names
   (`DigitalTwin_move_stage`, `AutoScriptMicroscope_move_stage`). A rehearsed call is matched to an instrument tool by
   what the two endpoints report in `endpoint_info`: the same `role` and `command` on a device with the same
   `device_kinds` (for a native tool, the same name). Each `rehearsal.calls_made[]` entry with `server: "instrument"`
   gains two keys, for the endpoint at `instrument_url`:
   - `instrument_tool`: the instrument tool it matches, or `null`;
   - `held_because`: `null` when it matches, else a sentence starting `"no match: will be held"` (no tool matches,
     two or more match, or the instrument cannot be reached).
   The sign-off covers exactly the calls with an `instrument_tool`, under that name, with the rehearsed arguments.
   At run time the mapping is worked out again; a call whose tool is not the one the approval recorded is held.
4. **Counts are used up.** A rehearsed call made `count` times is pre-approved `count` times: each call that passes
   uses one. `preapprove_run`'s `calls[]` items carry `count` (default 1). A call beyond its count is held, with
   `why_held: "the rehearsal made this call fewer times"`.
5. **The token role `preapprove`.** `preapprove_run`, `end_run_preapproval` and a twin's `export_ground_truth` need
   the new role `preapprove` (scope `instrument:preapprove`), and no other role grants them. Give
   `ASYNCROSCOPY_OPERATOR_TOKEN` only that scope, never `instrument:approve`. With `ASYNCROSCOPY_MCP_TOKEN` set and
   `ASYNCROSCOPY_OPERATOR_TOKEN` unset, the workflow server never falls back to the agents' token: the hardware run
   starts with `preapproval: {"granted": false, "why": "ASYNCROSCOPY_OPERATOR_TOKEN is not set ..."}` and every
   `approve` call is held.
6. **A held call of a signed-off run is bound to that run.** A person's `approve_call` on it is used only by a call
   carrying the same run id and that run's key (`resume_workflow` re-binds it to the resumed run's new key).
7. **A pre-approval is kept alive while the run drives.** Each call of the run with its key moves `expires_at` to
   `approval_ttl_s` from now, never beyond 12 hours after the grant (`max_run_preapproval_s`).
8. **`review_workflow`'s `rehearsal`, and `approve_workflow`'s choice,** pick the passing entry when a stopped and
   resumed rehearsal left two entries with one `run_id`.
9. **The run key** is sent only to the instrument endpoint, never to the analysis server.

## Who does what

| Process | Role here |
| :-- | :-- |
| Agent MCP server (`run_llm.py --yaml configs/gemma-llm.yaml`, `:8002`) | Runs workflows: the rehearsal on the twin and the run on the instrument. It acts on the instrument endpoints it was configured with, and on no other. |
| Workflow operator endpoint (`run_llm.py --yaml configs/workflow_operator.yaml`, `:8004`) | `review_workflow`, `approve_workflow`, `reject_workflow`, for a person. It reads the same library and run records. |
| Each stack's agent endpoint (`run_mcp.py`, the twin's and the instrument's) | The gate: holds, passes or refuses each call. |
| Each stack's operator endpoint (`mcp_operator.yaml`) | `export_ground_truth` on the twin; `preapprove_run` and `end_run_preapproval` on the instrument; `list_pending_approvals`, `approve_call`, `deny_call`. |
| The app | Owns the twin↔instrument pairing, starts the rehearsal and the run, shows the sign-off card, compares the results. |

## Configuration

The agent MCP server and the workflow operator endpoint both take the list of instrument endpoints they act on:

```yaml
agent:
  mcp_urls:
    - "http://127.0.0.1:8000/mcp"          # the default instrument endpoint (here the twin)
  workflow_instruments:                     # other instrument endpoints, by URL
    - "http://scope-a.lab:8000/mcp"         # the real instrument paired with it
  workflows_hardware: true                  # the agent server only: allow runs on a stack that is not simulated
```

`--workflow-instruments-json '["http://scope-a.lab:8000/mcp"]'` is the command-line form. The allowed list is the
first of `mcp_urls` followed by `workflow_instruments`, without repeats. Both servers report it as
`endpoint_info.workflow_instruments`, the default first:

```json
{"role": "agent_server", "urls": {...}, "services": {"workflows": {...}},
 "workflow_instruments": ["http://127.0.0.1:8000/mcp", "http://scope-a.lab:8000/mcp"]}
```

Each stack must report its operator endpoint in its own `endpoint_info.urls.operator` (`mcp.urls.operator` in its
MCP config). The workflow server calls it with the bearer token in `ASYNCROSCOPY_OPERATOR_TOKEN`, never the agents'
`ASYNCROSCOPY_MCP_TOKEN`. With `mcp.auth` all three operator tools it uses (`export_ground_truth`, `preapprove_run`,
`end_run_preapproval`) need the token role `preapprove` (scope `instrument:preapprove`), and that token should carry
that scope alone: `preapprove` grants nothing else, so it cannot approve a held call. With `ASYNCROSCOPY_MCP_TOKEN`
set and `ASYNCROSCOPY_OPERATOR_TOKEN` unset, the operator endpoint is not called (no fallback to the agents' token).
Without `mcp.auth` the operator endpoint binds to loopback and needs no token.

## New arguments

All are optional. Left out, a tool behaves as before, on the first instrument endpoint.

| Tool (server) | Argument | Meaning |
| :-- | :-- | :-- |
| `run_workflow` (agent server) | `instrument_url: str \| null` | Run on this instrument endpoint. It must be one of `workflow_instruments`; any other value is refused before anything is opened: `{"state": "refused", "workflow": name, "problems": ["<url> is not an instrument endpoint this workflow server acts on; it acts on [...] (agent.mcp_urls and agent.workflow_instruments in its config)"]}`. A resumed run (`resume_workflow`, `answer_workflow`) stays on the endpoint it started on. |
| `review_workflow` (workflow operator) | `instrument_url: str \| null` | Review for approval on this instrument endpoint: its `stack_fingerprint` is what a rehearsal must match. Refused (`ok: false`, the same sentence as `error`) for a URL outside the list. |
| `approve_workflow` (workflow operator) | `instrument_url: str \| null` | The same as for `review_workflow`. Pass the same value as to the review. `instrument` still names the `stack_id` the approval is limited to. |
| `propose_workflow` (agent server) | `context: str \| null` | What the conversation established, for example a summary of the chat. It is stored with the draft as sent, shown in the review as `context_as_reported_by_agent`, and never checked or run. |

The agent is never handed a URL it may use freely: the server accepts only the configured ones, whoever sends them.

## The rehearsal results

When a run on a simulated stack ends (`done`, `failed` or `stopped`), the rehearsal record in the library
(`rehearsals.json`) keeps:

- `calls_made`: every distinct call that succeeded, first made first:
  `[{"server": "instrument" | "analysis", "tool": str, "arguments": {...}, "count": int}]`. `arguments` is the
  canonical JSON the workflow sent (keys sorted). These, with `server: "instrument"`, are what a sign-off covers.
- `results`:

```json
{
  "data_keys": ["stem_image_HAADF_0001.h5", "stem_image_HAADF_0002.h5", "map_3f2a_map_v1"],
  "map": "map_3f2a_map_v1",
  "instrument_calls": 14,
  "distinct_instrument_calls": 12,
  "wall_s": 6.21,
  "instrument_s": 41.7,
  "instrument_s_from": "the twin's simulated clock",
  "dose": {"total_electrons": 3.1e9, "max_dose_e_per_a2": 812.0, "from": "the twin's dose map"},
  "twin_score": {
    "label": "twin score: measured against the digital twin's own ground truth, not a real specimen",
    "version": "1.0.0",
    "known": true,
    "acquisitions": 4,
    "images": 4,
    "particles_total": 120,
    "particles_imaged": 9,
    "features_reported": 1,
    "features_on_particles": 1,
    "feature_precision": 1.0,
    "stage_error_m": {"median": 3.2e-9, "max": 6.0e-9, "images": 4}
  }
}
```

- `data_keys`: every data key the run's replies named (an acquisition's first line; `key`, `image_key`,
  `spectrum_key`, `result_key` and `keys` in a JSON reply; the values of an analysis job's `result_keys`), and the
  stitched map. Read them from the twin's Tiled (`endpoint_info.urls.tiled` of the twin) for thumbnails.
- `map`: the key a `stitch_map` node produced (its `outputs.<node>.map`), or `null`.
- `instrument_calls`, `distinct_instrument_calls`: how many instrument calls succeeded, and how many different ones.
- `wall_s`: the time the run was driven, in seconds. Time spent waiting for a person is not counted.
- `instrument_s`: the twin's simulated clock from the start of the run to its end, or `null` when the twin's truth
  could not be read. `instrument_s_from` says where it came from.
- `dose`: from the twin's dose map. `total_electrons` is what the run put down; `max_dose_e_per_a2` is the highest
  dose on any spot of the specimen after the run, from every exposure so far. `null` when the twin keeps no dose map
  or its truth could not be read.
- `twin_score` (`asyncroscopy.sim.truth_score`, `version` changes when a number changes meaning). Always show
  `label` with it. It is measured from the twin's `export_ground_truth` read before the run started and after it ended:
  - `acquisitions`, `images`: what the twin recorded during the run, and how many of those were images;
  - `particles_total`, `particles_imaged`: the twin's particles, and those inside the true field of one of the run's images;
  - `features_reported`, `features_on_particles`, `feature_precision`: the features the run's `find_feature` nodes
    reported, how many land on a particle (within its radius, or two pixels of the run's images), and their ratio
    (`null` when none were reported);
  - `stage_error_m`: how far the stage really was from where it said it was, at each image (`null` without images).
    It is what a map stitched from the stage readings alone would misplace a tile by. The stitched map's own
    placements are not scored here.
  - When the truth could not be read, `twin_score` is `{"label", "version", "known": false, "why": "<reason>"}` and
    `instrument_s` and `dose` are `null`. `known` is also false when the twin knows neither particles nor images.

The truth score is computed by the workflow server from the twin's operator-only `export_ground_truth`. It is kept
only in the library's rehearsal record and returned only by `review_workflow`. No tool an agent is given returns
it: the run's own record, and so `run_workflow` and `workflow_status`, carry `results` without `instrument_s`,
`dose` and `twin_score`.

The run record's `results` (every run, rehearsal or not) is:

```json
{"data_keys": [...], "map": "map_3f2a_map_v1" | null, "instrument_calls": 14, "distinct_instrument_calls": 12, "wall_s": 6.21}
```

## The `review_workflow` reply

`review_workflow(name, instrument_url)` returns everything the sign-off card needs in one call. It is the reply
documented in [Workflow approval](workflow_approval.md), with these keys added:

```json
{
  "ok": true,
  "instrument_url": "http://scope-a.lab:8000/mcp",
  "name": "tiles_only",
  "status": "draft",
  "source": "the whole Python module",
  "hash": "02686b7b...",
  "context_as_reported_by_agent": {"text": "what the conversation established", "this_code": true},
  "last_approved": {"hash": "9c1e...", "by": "Sam Reed", "at": 1795817500.0},
  "diff_from_last_approved": "--- last approved\n+++ now\n@@ -12,7 +12,7 @@\n...",
  "stack_fingerprint": {"software": {"package": "asyncroscopy", "version": "0.1.0", "commit": "597be15...", "uncommitted_changes": false},
                        "devices": {...}, "config_digest": "...", "unreadable_properties": [],
                        "simulated": {"value": false, "established_by": "..."}, "digest": "830832d7..."},
  "rehearsal": {
    "run_id": "a9c880147930",
    "at": 1795845311.35,
    "inputs": {"size_m": 5e-07},
    "stack_id": "tango://twin-a:9094",
    "stack_fingerprint": {"software": {...}, "simulated": {"value": true, "established_by": "..."}, "digest": "ff322ffd..."},
    "results": {"...": "the rehearsal results above"},
    "calls_made": [{"server": "instrument", "tool": "Twin_move_stage", "arguments": {"position": [1e-06, 2e-06, 0.0, 0.0, 0.0]}, "count": 1}]
  },
  "rehearsals": [{"run_id": "...", "results": {"...": "as above"}, "...": "the keys documented in workflow_approval.md"}],
  "can_approve": true,
  "why_not": null
}
```

- `instrument_url`: the endpoint the review is for (the first one when the request left it out).
- `context_as_reported_by_agent`: the `context` sent with `propose_workflow`, or `null`. Like
  `request_as_reported_by_agent`, it is the agent's (or the app's) account, not the person's own record.
  `this_code` is false when the draft was replaced after it was sent.
- `last_approved`: the code as a person last approved it, which revoking and rejecting leave in place; `null` when it
  was never approved. `diff_from_last_approved` is a unified diff from that code (`--- last approved`, `+++ now`) to
  `source`: `""` when they are the same, `null` when there is no last approval.
- `stack_fingerprint`: the fingerprint of the endpoint at `instrument_url` now (`null` when it cannot be reached).
- `rehearsal`: the latest rehearsal of this exact code that passed on a stack that counts for `instrument_url`:
  the same fingerprint digest for a twin, or a simulated stack with the same `software` for a real instrument. It is
  `null` when there is none, and then `can_approve` is false. It is the rehearsal `approve_workflow` records in
  `approval.rehearsed`, so its `calls_made` are the calls the approval will pre-approve. `stack_fingerprint` here is
  the full fingerprint the rehearsal ran on.
- `rehearsals[]` items gain `results`.

## The run on the instrument

After `approve_workflow`, the app sends `run_workflow(name, inputs, instrument_url=<the paired instrument>)`. The
agent server, for an `approved` workflow on a stack that is not simulated:

1. finds the rehearsal the approval was given on (`approval.rehearsed.run_id`);
2. makes a run id and a secret run key, and calls the instrument's operator endpoint
   `preapprove_run(run_id, run_key_sha256, calls, approved_by, approver, workflow, hash, rehearsal_run_id)` with the
   rehearsal's instrument calls;
3. runs the workflow. Every call carries `X-Asyncroscopy-Run: <run_id>` (as every workflow call always has) and
   `X-Asyncroscopy-Run-Key: <run_key>`. The key appears in no reply, so a caller who knows the run id cannot use the
   pre-approvals;
4. when the run ends (`done`, `failed`, `stopped`, or an error), calls `end_run_preapproval(run_id)`. A run that is
   `waiting` for a person keeps its pre-approval until it expires. `resume_workflow` and `answer_workflow` grant it
   again, with a new key and a new expiry, for the same run id. `stop_workflow` on a waiting run removes it.

The gate on the instrument's agent endpoint, for a call of an `approve` tool:

- a `never` tool is refused first, as always;
- the arguments are validated (defaults filled in, values in their declared types) and the rehearsal's arguments for
  that tool are validated the same way; when the call's run id has a live pre-approval, the run key matches, and the
  call equals one of the rehearsal's, it runs. Its Instrument Trace event carries
  `approved_by: "workflow_signoff:<workflow>@<hash>"` and `approver: "<the approver's name>"`;
- otherwise it is held as usual, and the held reply and the pending entry carry `signoff` (below).

A pre-approval expires after the instrument's `mcp.approval_ttl_s` (600 s in the shipped configs), when the run ends,
or when it is stopped. A second run of the same workflow gets its own run id and key, and never uses the first run's
pre-approvals. Grants and removals are written to the operator endpoint's trace as actions `preapprove_run` and
`end_run_preapproval`, with `run`, `approved_by`, `approver`, `workflow`, `workflow_hash`, `calls` and `expires_at`.

A workflow approved on the command line without a rehearsal, a stack without an operator endpoint, or an operator
endpoint that refuses, gives no pre-approval: the run still starts and every `approve` call is held as usual. The
record's `preapproval` says why.

### `preapproval` in the run record

`run_workflow`, `workflow_status`, `resume_workflow` and `stop_workflow` replies carry it for a hardware run of an
approved workflow:

```json
{"granted": true,
 "approved_by": "workflow_signoff:two_doses@02686b7b...",
 "approver": "Sam Reed",
 "rehearsal_run_id": "a9c880147930",
 "calls": 2,
 "granted_at": "2026-11-28T05:55:11Z",
 "expires_at": "2026-11-28T06:05:11Z",
 "rehearsal_inputs": {"where": "{\"x_m\":0.0,\"y_m\":0.0}"}}
```

or `{"granted": false, "why": "<sentence>"}`. `calls` is how many calls are pre-approved. `rehearsal_inputs` are the
inputs the rehearsal ran with: other inputs usually mean other arguments, and those calls are held. The reply of
every run that started also has `instrument_url`, the endpoint it acts on.

### A held call of a signed-off run

The held reply from the instrument (and so a waiting run's `approvals[]` entry) adds `signoff`:

```json
{"status": "approval_required", "request_id": "9635538c-...", "tool": "SAMPLE_get_dose",
 "arguments": {"request": "{\"x_m\":1e-07,\"y_m\":0.0}", "x_m": null, "y_m": null},
 "arguments_sha256": "76258b56...", "acts": false, "risk": "low", "expires_at": "2026-11-28T06:05:13Z",
 "requested_by": "agent-workflow",
 "signoff": {"approved_by": "workflow_signoff:two_doses@02686b7b...", "approver": "Sam Reed", "run_id": "a937b83254e7",
             "why_held": "the rehearsal called this tool with other arguments"}}
```

`why_held` is `"the rehearsal called this tool with other arguments"` or `"the rehearsal never called this tool"`.
`list_pending_approvals` on the instrument's operator endpoint shows the same `signoff` on that entry; entries of
calls outside any signed-off run have no `signoff` key. A person approves it with `approve_call` as any other, and
`resume_workflow(run_id)` sends it again. The approval is used once, as always.

## Data-dependent arguments

A call whose arguments come from live data will differ on the instrument and be held. Examples: a stage position
from `find_feature`, a point of interest from `best_region`, a tile position after a drift correction, or anything
computed from an image. The rehearsal on the twin found the twin's particle, and the instrument's particle is
elsewhere, so the `move_stage` that follows is a different call. This is intended: the sign-off covers the calls a
person saw rehearsed, and a position the twin could not have shown is put to a person. Calls with fixed or input-only
arguments (a field of view, a dwell time, a grid computed from the same inputs and the same start position) match. A
grid that starts from the stage position read on the instrument matches only when the stage reads the same position
as on the twin.

## End to end

1. **Propose.** The agent, in chat: `propose_workflow(user_request, source, context)` on the agent server. Reply
   `{"ok": true, "name": "two_doses", "status": "draft", "hash": "...", ...}`.
2. **Rehearse on the twin.** The app: `run_workflow(name, inputs, instrument_url=<twin agent endpoint>)`. A draft
   runs on a simulated stack only. The reply ends in `state: "done"`; call `workflow_status(run_id, wait_s)` until it
   does. The rehearsal record is written when it ends.
3. **Review.** The app, on the workflow operator endpoint: `review_workflow(name, instrument_url=<paired instrument>)`.
   Show `rehearsal.results` (thumbnails of `data_keys` and `map` from the twin's Tiled, `instrument_s`, `dose`,
   `twin_score` with its `label`), `rehearsal.calls_made`, `source` or `diff_from_last_approved`,
   `request_as_reported_by_agent` and `context_as_reported_by_agent`. `can_approve` and `why_not` say whether
   Approve will work.
4. **Approve.** The person presses Approve: `approve_workflow(name, approver, expected_hash=<review.hash>,
   instrument_url=<paired instrument>)`. Refused unless this exact code passed a rehearsal on a twin with the
   instrument's `software`.
5. **Run on the instrument.** The app: `run_workflow(name, <same inputs>, instrument_url=<paired instrument>)`. The
   reply's `preapproval.granted` is true. The rehearsed calls run; a different call stops the run as `waiting` with
   `approvals[].signoff`. The person approves it on the instrument's operator endpoint (`approve_call`) and the app
   calls `resume_workflow(run_id)`, or `stop_workflow(run_id)`.
6. **Compare.** The hardware run's `results` (from `workflow_status`) sit next to `rehearsal.results` from the
   review: the same `data_keys` shape, `map`, `instrument_calls`, `distinct_instrument_calls` and `wall_s`. The
   instrument has no `twin_score`, `dose` or `instrument_s`; the twin-to-hardware gap is the difference in what both
   report.

## Tests

`tests/endpoints/test_signoff.py` drives this contract over real MCP endpoints on the session's Tango devices:
a rehearsed call runs without a hold, a different argument is held, an unrehearsed tool is held, a `never` tool is
refused, a pre-approval expires, a second run (or the run id without its key) gets nothing, and the whole sequence
from `propose_workflow` to a held changed call on a second endpoint that is not simulated.
`tests/scoring/test_twin_score.py` checks the twin score, and `tests/agents/test_workflows.py` the review's
results, calls and diff and the instrument allow-list.
