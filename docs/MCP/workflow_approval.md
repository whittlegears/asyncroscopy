# Workflow approval

A workflow that an agent writes (`propose_workflow`) is a **draft**. A draft runs on a digital twin only. It runs on
the real instrument only after a person approves it. A person approves in one of two places:

- the command line: `python -m asyncroscopy.workflows approve NAME`;
- the **workflow operator endpoint**, described on this page. An app such as SciAgentGUI calls it for a person.

Both write the same records in the same library (`outputs/workflows/library`: `approvals.json`,
`rejections.json`, `approved/NAME.py`). The one difference is that the endpoint refuses an approval unless this
exact code ran to the end (state `done`) on a digital twin at least once. The command line shows the rehearsals and
warns when there is none, but does not refuse.

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

### `review_workflow(name: str)`

This tool only reads. The reply:

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
  "hash": "sha256 of the code and every block it uses",
  "acts": true,
  "tools": ["the instrument tools its blocks call"],
  "blocks": ["plan_grid", "..."],
  "problems": [],
  "request": {"text": "the user's own words", "at": 1795817468.2, "this_code": true},
  "approval": null,
  "changed_since_approval": false,
  "rejection": null,
  "rehearsals": [
    {"run_id": "3f2a...", "state": "done", "at": 1795817470.1, "calls": 14, "inputs": {"size_m": 5e-07},
     "error": null, "problems": [], "this_code": true, "outputs": {"...": "shortened as in workflow_status"}}
  ],
  "rehearsed": true,
  "can_approve": true,
  "why_not": null
}
```

- `status` is `shipped`, `approved` or `draft`. An approved workflow whose code changed is a `draft` again.
- `request` is `null` when nobody's request was kept. `this_code` is false when the draft was replaced after the
  request.
- `approval` is the approval record (see `approve_workflow`) or `null`. `changed_since_approval` is true when
  there is an approval and the code or one of its blocks has changed since.
- `rejection` is `{"hash", "reason", "by", "login", "at", "via", "this_code"}` or `null`.
- `rehearsals` lists the runs on a digital twin, oldest first, at most 10. `this_code` marks the runs of the
  code as it is now. `outputs` is read from the run's record, and is `null` when that record is gone. `problems`
  are the problems that the run recorded.
- `rehearsed` is true when a run of this code reached `done`.
- `can_approve` and `why_not` say whether `approve_workflow` would accept it now. The reasons are: it is shipped,
  it is already approved, it fails its checks, or it was never rehearsed to the end.
- `problems` are the workflow's own checks. They do not check it against the live endpoint's tools; the runner
  does that before every run.
- Times (`at`) are Unix seconds.

### `approve_workflow(name: str, approver: str, instrument: str | null = null, days: float | null = null)`

This tool approves the workflow as it is now. `approver` is the person's name. `instrument` is the `stack_id` that
the approval is for; `null` or `"any"` means every instrument. `days` is how long the approval holds; `null` means
until it is revoked.

The tool refuses (`ok: false`) when:

- there is no such workflow;
- the workflow is shipped;
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
              "request": "the user's own words", "via": "Workflow_Operator_MCP"}}
```

`via` is the endpoint's name, or `"cli"` when the approval came from the command line. An approval replaces any
earlier rejection.

### `reject_workflow(name: str, reason: str)`

This tool rejects the workflow as it is now, with a reason. If the workflow was approved, it loses its approval. Its
text stays as a draft, and the draft still runs on a twin. `list_workflows` shows the reason as `rejected`, so the
agent that wrote it can read it. The tool refuses when there is no such workflow, when it is shipped, when `reason`
is empty, or when the login is not in `approvers.json`.

```json
{"ok": true, "rejection": {"name": "tiles_only", "status": "draft", "hash": "...", "reason": "...", "by": null,
                           "login": "sam", "at": 1795817600.0, "via": "Workflow_Operator_MCP"}}
```

## Tests

`tests/agents/test_workflows.py::test_a_person_reviews_approves_and_rejects_through_the_operator_endpoint_only`
drives this contract over MCP. `tests/agents/test_agent.py` checks that the agent endpoint and the swarm never get
these tools. `tests/startup/test_run_servers.py` checks the loopback rule.
