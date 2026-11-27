# SciAgentGUI integration: queue panel and specimen atlas

SciAgentGUI already talks to the MCP bridge at `http://<host>:<port>/mcp`. The
same server, with `mcp.supervision.enabled: true`, serves the **operator API**
at `/supervision/`. That API is everything a queue panel and an atlas panel
need. The built-in page at `/supervision/ui` is a working reference
implementation of both (about 200 lines in
`asyncroscopy/supervision/operator_ui.html`).

The agent uses MCP tools only. Approve, resume, take over, STOP and policy
changes exist **only** on this HTTP API, so keep them in the GUI's operator
controls and never expose them as agent tools.

## Authentication

If `mcp.supervision.operator_token` is set, send
`Authorization: Bearer <token>` on every request. Image URLs also accept
`?token=<token>`. CORS is open, so a desktop or web shell can call it
directly.

## Queue panel

**Poll** `GET /supervision/state?limit=200` about once a second, or use
`GET /supervision/events?since=<seq>` to fetch only what changed.
`state.last_event` is the latest sequence number.

```jsonc
{
  "state": {"paused": false, "controller": "shared",   // "operator" after take-over
            "current": "a1f2-7", "queued": 2, "awaiting_approval": [], "last_event": 418},
  "actions": [                                           // oldest first; children follow their parent
    {"id": "a1f2-7", "tool": "atlas.return_to_site", "origin": "operator",   // agent | operator | recipe
     "label": "return to 'grain 3' and acquire", "site_id": "grain-3-ab12cd",
     "status": "running",           // queued | awaiting_approval | running | done | error | skipped |
                                    // denied | aborted | rejected | refused
     "parent_id": null, "children": ["a1f2-8", "a1f2-9"],
     "requires_approval": false, "approval_reason": "", "decided_by": null,
     "created": 1795759179.5, "started": 1795759179.6, "finished": null,
     "args": {}, "result": null, "error": null, "run_id": null, "note": null}
  ]
}
```

**What to show for each action:**
- **Who requested it:** `origin` (agent, operator or recipe).
- **Which site it targets:** `site_id`.
- **Status:** `status`. Show `error` or `approval_reason` underneath.
- **Steps:** indent the child steps (rows with a `parent_id`) under their
  parent.

**Controls** (all `POST`; each returns the new `state`):

| Button | Endpoint | Effect |
|---|---|---|
| Pause / Resume / Step | `/supervision/pause`, `/resume`, `/step` | Take effect between actions, including between the steps of a running composite action. |
| Abort current | `/supervision/abort` | Ends the running composite action at its next step. |
| Skip (per row) | `/supervision/actions/<id>/skip` | Only for `queued` and `awaiting_approval` rows. |
| Approve / Deny (per row) | `/supervision/actions/<id>/approve`, `/deny` | For `awaiting_approval` rows. |
| Take over / Release | `/supervision/take-over` `{"note": "..."}`, `/supervision/release` | While the operator holds control, agent requests are recorded as `rejected` and the history is kept. |
| **STOP** | `/supervision/stop` | Runs the instrument's stop commands immediately, bypassing the queue, then pauses the queue. Keep it visually separate from the queue controls. |
| Run a tool as the operator | `/supervision/actions` `{"tool": "instrument.move_stage", "args": {"arg": [...]}}` | Queued with origin `operator`. |

To **save a good run as a recipe**, let the user tick finished top-level rows,
then call `POST /supervision/recipes/from-actions`
`{"name": "...", "action_ids": [...]}`.

Other recipe endpoints:
- `GET /supervision/recipes` lists them. Each entry carries
  `validated_on_twin`.
- `POST /supervision/recipes/<name>/check` returns the missing tools or
  detectors, and whether a twin rehearsal is required.
- `POST /supervision/recipes/<name>/run` runs one.
- `PUT /supervision/recipes/<name>` edits one. An edit is a new version, so it
  must be rehearsed on the twin again.

## Atlas panel: click a site → return and acquire → compare

1. `GET /supervision/atlas` returns the specimen, the mount (with the fitted
   rotation after a remount), whether the stage mapping is calibrated, and
   `sites[]`. Each site has `id`, `name`, `landmark`, `predicted_stage`,
   `visits` and `last_visit` (`status` and `residual_nm`).
2. Thumbnail: `GET /supervision/atlas/sites/<id>/reference.png`.
3. Click → `POST /supervision/atlas/sites/<id>/return` `{}`. It returns
   `{"queued": <action>}`. Watch that action and its child steps in the queue
   panel.
4. When it finishes, `GET /supervision/atlas/sites/<id>` lists `visits[]`.
   Take the last one, then fetch the detail with
   `GET /supervision/atlas/sites/<id>/visits/<vid>`. It includes:
   - `status`: `match`, `ambiguous`, `no_match` or `not_converged`
   - `reasons[]`
   - `residual_nm`, `sigma_nm`, `ncc`, `ambiguity` (second-best peak relative
     to the best), `rotation_deg`
   - `requested_by`, `action_id`
5. Compare: show `reference.png` next to
   `GET /supervision/atlas/sites/<id>/visits/<vid>.png`. Only `match` means
   "this is the same place". For any other status, show the `reasons` in
   warning colours.

Other atlas actions: **Save current position**
(`POST /supervision/atlas/sites` `{"name", "landmark"}`), **Calibrate
mapping** (`/atlas/calibrate`), **Specimen remounted** (`/atlas/remount`), and
**Reacquire landmarks** (`/atlas/landmarks/reacquire`).

## Health strip (optional)

Agents already have `system_health` and `recover_device` as MCP tools. For a
status strip, poll the agent-side `system_health` tool. Or run
`python -m asyncroscopy.health --json`, which returns a checklist with fixes
and exits 0 when everything is OK.
