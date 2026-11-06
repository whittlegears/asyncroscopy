# The Instrument Trace, live

Every action on the instrument is written to the Instrument Trace: each MCP tool call, each job step, each approval decision. You can read a finished run back, or follow a run while it happens, from Python, from a terminal or from the GUI.

## What is traced

Each process appends JSON lines to its own file under `outputs/trace/`:

| File | Written by | One line per |
| :-- | :-- | :-- |
| `mcp_agent.jsonl` | the agent MCP endpoint (`mcp.trace` in `configs/mcp_dt.yaml`) | tool call, held call (`approval_required`), and release made by `run_job` |
| `mcp_operator.jsonl` | the operator (GUI) endpoint | tool call |
| `jobs.jsonl` | every job device (`trace_path` property) | reserve, step, failure, suspend and resume, the end state (`Mapper.done`), and, in a dry run, each recorded call |
| `approvals.jsonl` | the Approvals device | request, approve, deny, expire, mode change |
| `analysis.jsonl` | the analysis server | analysis tool call (its lines use `tool` for the action) |

The trace has gaps. A notebook or script that calls a Tango device directly, without the MCP server or a job, is not traced.

## Fields

The schema is `asyncroscopy/contracts/trace_event.schema.json`. Every line has `schema_version`, `ts`, `source`, `action` and `ok`. Since schema 1.1.0, lines also carry these fields:

| Field | Meaning |
| :-- | :-- |
| `seq` | The line's number in its file. It rises by one per line, whichever process wrote the line. Writers share a lock on the sidecar `<file>.seq`, so lines land in `seq` order, and `seq` keeps rising when the file is rotated. |
| `written` | The Unix time the line was appended. It never goes back within a file. Readers merge files by it. |
| `ts` | The Unix time the action started. A tool call's `ts` is earlier than its `written` by the call's `duration_s`. |
| `clock_s`, `clock` | The instrument's clock: simulated seconds on a twin, Unix time on hardware. `clock: "instrument"` means the value came from the instrument's `clock_s` attribute. For a job step, that is the time the step finished. For any other event, it is the time the event was written. `clock: "wall"` means the writer has no instrument clock or could not read it, so `clock_s` is this computer's time. A job device reads its `microscope_device_address`. An MCP server reads `mcp.clock_device`. |
| `run` | The agent run, from the `X-Asyncroscopy-Run` header that agents send with every call. The MCP server passes it on to the job it starts and to the approval request it files, so their events carry it too. It is absent on calls made without an agent (GUI, script). `run_id` is different: it is a job's reservation id. |
| `simulated` | Whether the stack drives a simulator: run_servers' top-level `simulated:`. Job devices and the Approvals device have it as a property. The MCP server asks its Approvals device for it. It is absent where unknown. |
| `dry_run` | On job events: `true` in a dry run, `false` in a real run. |
| `writer` | Who wrote the line: a Tango device name (`asyncroscopy/mapper/default`) or `mcp:<name>:<role>`. |
| `caller` | The `X-Asyncroscopy-Caller` header. On an approval decision, it is the operator's name. |

Lines written before 1.1.0 lack these fields. They still validate and are still read. They are ordered by `ts`.

## Following a run live

**Command line.** In a hackathon terminal, run:

```console
python -m asyncroscopy.trace.tail --run <run id>
```

It prints every event of that run written so far, in order, then each new event as it is written, until you press Ctrl+C. It reads `outputs/trace` by default. Name other files or directories as arguments. The options are:

- `--caller`, `--device` and `--job` filter the events, and each can be repeated. `--device` takes a Tango name or a tool prefix: `Mapper` matches `Mapper_start` and `Mapper.step`. `--job` takes a `job_id`, a reservation id or a `plan_id`.
- `--since 10m` starts at a time. It also takes an ISO time or Unix seconds.
- `--from-end` shows only new events.
- `--no-follow` prints what is there and stops.
- `--json` prints raw lines.

**Python.**

```python
from asyncroscopy.trace import read_trace, follow_trace, TraceStream

events = read_trace("outputs/trace", run=run_id)          # a finished run, in order
for event in follow_trace("outputs/trace", run=run_id):   # the same, live
    print(event["action"], event["clock_s"])

stream = TraceStream("outputs/trace", device="Mapper", from_end=True)
new = stream.poll()          # events since the last poll
cursor = stream.cursor()     # carry on later: TraceStream(..., cursor=cursor)
```

**GUI (operator endpoint).** The operator MCP endpoint has the `trace_since(cursor, run, caller, device, job, limit, from_end)` tool. It returns `{"events": [...], "cursor": "...", "more": bool}`. Each call passes the cursor back to get the next events. Only the operator endpoint offers this tool, not the agent endpoint, because the trace holds every caller's arguments. By default it reads every `*.jsonl` file next to the endpoint's own trace. `mcp.trace_follow` in `configs/mcp_operator.yaml` names other sources.

### How the reader keeps order

The reader merges the files by `written`, then file name, then `seq`.

A live follower holds each event for 0.5 s after it was written (`grace_s`). In that time, a line that another process is still appending cannot be overtaken. A follower therefore receives a run's events in the same order that `read_trace` returns them, within about 0.7 s of each event being written.

Lines are never delivered twice and never missed in these cases:

- **The file grows.** A line still being written (no newline yet) waits for its newline.
- **The file is truncated.** The reader starts again at the top of the file.
- **The file is rotated.** The file is renamed, say `jobs.jsonl` to `jobs.jsonl.1`, and a new one is started. The follower finds the renamed file by its identity, reads what it had not yet read, and then reads the new file.

Name rotated files `<name>.jsonl.<n>`. Rotate the `.seq` sidecar with its file, or leave it in place.
On Windows a rename fails (`PermissionError`, WinError 32) while a writer or the follower has the file open for its brief append or read; a rotation script there should retry the rename for a moment.

## Privacy

The trace records each call's **arguments in full**, and a preview of its result, up to 500 characters. This covers plans, prompts passed as arguments, file names and operator names, from every caller, and the files keep growing.

- Do not pass secrets as tool arguments.
- Treat `outputs/trace/` as you would lab notebook data when you share a stack or a log.
- Rotate or delete old files as your facility's data policy requires.

For this reason, the agent endpoint cannot read the trace.
