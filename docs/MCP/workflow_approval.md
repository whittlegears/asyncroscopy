# Workflow approval

A workflow that an agent writes (`propose_workflow`) is a **draft**. A draft runs on a digital twin only. It runs on
the real instrument only after a person approves it. A person approves in one of two places:

- the command line: `python -m asyncroscopy.workflows approve NAME`;
- the **workflow operator endpoint**, described on this page. An app such as SciAgentGUI calls it for a person.

Both write the same records in the same library (`outputs/workflows/library`: `approvals.json`,
`rejections.json`, `approved/NAME.py`). The endpoint adds two checks the command line does not make:

- an approval is refused unless this exact code passed a rehearsal on the digital twin the endpoint is connected
  to: the run reached `done` (no block failed), every cleanup call of its blocks succeeded, and the rehearsal's
  `stack_fingerprint` digest is the instrument endpoint's now (`docs/MCP/permissions.md`). A rehearsal on another
  stack, or on this one before its software, device code or properties changed, does not count, and the refusal
  names the stacks it did pass on. When the endpoint is a real instrument (its fingerprint is not simulated), a
  twin and the instrument can never share a fingerprint, so a rehearsal counts when it passed on a simulated stack
  whose `software` (package version and commit) is the instrument's. The command line shows the rehearsals and warns when there is none, but does not
  refuse;
- `approve_workflow` and `reject_workflow` take the `expected_hash` that `review_workflow` returned for the code the
  person read, and refuse when the workflow is not that code any more. An agent can call `propose_workflow` again
  with the same name while the person is reading; without this check the person would approve code they never saw.

## Who can call it

No agent can call these tools.

- The agent MCP server (`configs/gemma-llm.yaml`, port 8002) does not offer `review_workflow`,
  `approve_workflow` or `reject_workflow`. The swarm's agents do not get them either, even with the
  `*workflow*` tool glob: they are not in `WorkflowServer.tools()` or `AgentMCPServer.workflow_tools()`.
- The tools are offered only by the workflow operator endpoint. This is a separate process of the same module, with
  `--endpoint-role operator`. It has no swarm and no model, and it says that it is an operator endpoint
  (`endpoint_info.role` and `action_space.endpoint.role` are `"operator"`). This is the same pattern as the
  instrument's operator endpoint (`configs/mcp_operator.yaml`, port 8001).
- The endpoint has no login, so it binds to loopback only. `run_llm.py` and `agent_mcp_server` refuse any other
  host. Anyone who can reach the port can approve, in the same way that anyone who can run the command on this host
  can approve.
- The login in each record is the account the endpoint runs under (`getpass.getuser()` in that process). If the
  library has `approvers.json`, only those logins may approve or reject, as on the command line. `approver` is the
  name the person gives, and is kept as `by`.

## Start it

Start it on the GUI's machine, from the same directory as the agent MCP server, so that both read the same library
and the same run records (`outputs/workflows`):

```bash
uv run startup_scripts/run_llm.py --yaml configs/workflow_operator.yaml    # http://127.0.0.1:8004/mcp
```

The config is the agent config with `agent.endpoint_role: operator`. The model and agent keys are ignored.

## How an app reaches it

1. Connect to `http://127.0.0.1:8004/mcp`.
2. Call `endpoint_info`. `role == "operator"` and `serves == "workflow_approvals"` identify this endpoint. Find it
   by these fields, not by its port or name.
3. Call `action_space`. It is in the instrument endpoints' shape, with `instruments`, `state` and `jobs` empty. Its
   `actions` list the three tools. Each has `device: null` and `command: null`:

   | tool | role | risk | effect |
   | :-- | :-- | :-- | :-- |
   | `review_workflow` | `setting.read` | `low` | `observe` |
   | `approve_workflow` | `other` | `high` | `act` |
   | `reject_workflow` | `other` | `high` | `act` |

4. Never give these tools to a model. Call them only when a person presses a button. The list of workflows to show
   comes from `list_workflows` on the agent MCP server, where a draft has `status: "draft"`. A rejected draft also
   has `rejected` (the reason). A draft that changed after its approval has a `note`.

Every tool returns a JSON object with `ok`. When `ok` is false, `error` is a sentence to show to the person, and
nothing was changed.

## Tools

### `review_workflow(name: str, instrument_url: str | null = null)`

This tool only reads. `instrument_url` is the instrument endpoint the approval would be for, one of the endpoints
the server acts on (`endpoint_info.workflow_instruments`; the first when left out); a rehearsal must match its
`stack_fingerprint`. The reply below is extended for the sign-off card with `instrument_url`,
`context_as_reported_by_agent`, `last_approved`, `diff_from_last_approved`, `stack_fingerprint`, `rehearsal` (the
latest passing rehearsal with its results and every call it made) and `results` on each rehearsal:
[Twin to instrument](twin_to_instrument.md) gives their exact shapes. The reply:

```json
{
  "ok": true,
  "name": "tiles_only",
  "status": "draft",
  "summary": "Take every tile of a square grid around the current position by moving the stage, then return; no stitch.",
  "inputs": {"size_m": {"type": "number", "default": 6e-07, "min": 1e-07, "max": 5e-06, "doc": "Side of the square, metres."}},
  "budget": {},
  "graph": {"nodes": {"plan": {"block": "plan_grid", "with": {"size_m": "{inputs.size_m}", "move_by": "stage"}}},
            "edges": [["START", "plan"], ["home", "END"]]},
  "source": "the whole Python module",
  "hash": "the hash of the code and every block it uses",
  "acts": true,
  "tools": ["the instrument tools its blocks call"],
  "blocks": ["plan_grid", "..."],
  "problems": [],
  "request_as_reported_by_agent": {"text": "what the agent says the user asked", "at": 1795817468.2, "this_code": true},
  "approval": null,
  "approval_expired": false,
  "changed_since_approval": false,
  "rejection": null,
  "rehearsals": [
    {"run_id": "3f2a...", "state": "done", "at": 1795817470.1, "calls": 14, "inputs": {"size_m": 5e-07},
     "error": null, "problems": [], "this_code": true, "stack_id": "tango://localhost:9094", "stack_fingerprint": "9c1e...",
     "passed": true, "this_stack": true, "outputs": {"...": "shortened as in workflow_status"}}
  ],
  "rehearsed": true,
  "can_approve": true,
  "why_not": null
}
```

- `status` is `shipped`, `approved` or `draft`. An approved workflow whose code changed is a `draft` again.
- `request_as_reported_by_agent` is the `user_request` the proposing agent sent with `propose_workflow`. It is the
  agent's account of what the user asked, not a record the person made: show it as such, never as the person's own
  words. It is `null` when none was kept; `this_code` is false when the draft was replaced after it.
- `approval` is the approval record (see `approve_workflow`) or `null`. `changed_since_approval` is true when
  there is an approval and the code or one of its blocks has changed since.
- `rejection` is `{"hash", "reason", "by", "login", "at", "via", "this_code"}` or `null`.
- `rehearsals` lists the runs on a digital twin, oldest first, at most 10. `this_code` marks the runs of the
  code as it is now. `outputs` is read from the run's record, and is `null` when that record is gone. `problems`
  are the problems that the run recorded. `stack_id` and `stack_fingerprint` (its digest) name the stack it ran on;
  `this_stack` is true when that is the instrument endpoint's fingerprint now. `passed` is true when it reached `done`
  and every cleanup call succeeded.
- `hash` identifies this exact code (the module and every block it uses). Pass it back as `expected_hash`.
- `rehearsed` is true when a run of this code passed on this stack's fingerprint.
- `approval_expired` is true when the approval's `expires` has passed: the workflow no longer runs on an instrument
  and may be approved again.
- `can_approve` and `why_not` say whether `approve_workflow` would accept it now; `approve_workflow` decides from
  this same reply, so the two never disagree. The reasons are: it is shipped, it is already approved and the
  approval has not expired, it fails its checks, or it never passed a rehearsal on this stack's fingerprint.
- `problems` are the workflow's own checks. They do not check it against the live endpoint's tools; the runner
  does that before every run.
- Times (`at`) are Unix seconds.

### `approve_workflow(name: str, approver: str, expected_hash: str, instrument: str | null = null, days: float | null = null, instrument_url: str | null = null)`

This tool approves the workflow the person reviewed. `expected_hash` is the `hash` from the `review_workflow` reply
they read; the approval record keeps it as `hash`. `approver` is the person's name. `instrument` is the `stack_id` that
the approval is for; `null` or `"any"` means every instrument. `days` is how long the approval holds; `null` means
until it is revoked. `instrument_url` is as for `review_workflow`; pass the value the review used.

The tool refuses (`ok: false`) when:

- there is no such workflow;
- `review_workflow` would report `can_approve: false` (shipped, approved and not expired, failing its checks, or
  never rehearsed), with its `why_not` as the error;
- the workflow's hash is not `expected_hash` (it changed since it was reviewed: review it again);
- `approver` is empty;
- `days` is not above 0;
- the login is not in `approvers.json`;
- the workflow fails its checks;
- this exact code never ran to the end on a twin.

The reply when it succeeds:

```json
{"ok": true, "status": "approved",
 "approval": {"name": "tiles_only", "hash": "...", "by": "Sam Reed", "login": "sam", "at": 1795817500.0,
              "acts": true, "tools": ["..."], "instrument": "tango://scope-a:9094", "expires": 1798409500.0,
              "rehearsed": {"run_id": "3f2a...", "at": 1795817470.1, "calls": 14, "inputs": {"size_m": 5e-07}},
              "request": "what the proposing agent said the user asked", "via": "Workflow_Operator_MCP"}}
```

`request` is the proposing agent's account of the user's request (`request_as_reported_by_agent` in the review),
kept with the approval; it is not the person's own record. `rehearsed` is the last run of this code on a twin that
reached `done` on a stack that counts for `instrument_url` (the review's `rehearsal`), with the inputs it ran with. Its
calls are the ones a hardware run of the workflow gets pre-approved ([Twin to instrument](twin_to_instrument.md)). `via` is the endpoint's name, or `"cli"` when the approval came from the command line. An approval replaces any
earlier rejection.

### `reject_workflow(name: str, reason: str, expected_hash: str)`

This tool rejects the workflow the person reviewed (`expected_hash`, as for `approve_workflow`), with a reason. If
the workflow was approved, it loses its approval. Its
text stays as a draft, and the draft still runs on a twin. `list_workflows` shows the reason as `rejected`, so the
agent that wrote it can read it. The tool refuses when there is no such workflow, when it is shipped, when its hash
is not `expected_hash`, when `reason` is empty, or when the login is not in `approvers.json`.

```json
{"ok": true, "rejection": {"name": "tiles_only", "status": "draft", "hash": "...", "reason": "...", "by": null,
                           "login": "sam", "at": 1795817600.0, "via": "Workflow_Operator_MCP"}}
```

## Tests

`tests/agents/test_workflows.py::test_a_person_reviews_approves_and_rejects_through_the_operator_endpoint_only`
drives this contract over MCP, including a draft re-proposed between the review and the approval (refused). `tests/agents/test_agent.py` checks that the agent endpoint and the swarm never get
these tools. `tests/startup/test_run_servers.py` checks the loopback rule.
