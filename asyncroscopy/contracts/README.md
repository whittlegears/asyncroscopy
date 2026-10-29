# Contracts

Shared shapes that let different instruments, analysis servers and agents work together without importing each other's code.

| Schema | Used by | What it fixes |
| :-- | :-- | :-- |
| `envelope.schema.json` | `asyncroscopy.jobs.JobDevice` (e.g. `Mapper`), the pyTEMlib MCP server's jobs | The six verbs of a long-running job: `reserve`, `start`, `status`, `abort`, `results`, `release`. Each takes and returns one JSON object. |
| `map_plan.schema.json` | `Mapper.reserve` (STEM only) | What a STEM map run acquires: stage box, field of view, overlap, detectors (HAADF by default), per-tile EDS, electron dose budget. Another instrument writes its own plan schema; only the envelope and trace are shared. |
| `action_space.schema.json` | the MCP server's `action_space()` tool (`asyncroscopy.mcp.action_space`) | What an endpoint lets a program control and observe: every tool with its canonical role, arguments (units, ranges, stage axes), risk and effect; the readable state; each job device's plan schema and example. See [docs/Operation/action_space.md](../../docs/Operation/action_space.md). |
| `trace_event.schema.json` | the MCP server, jobs, the approvals device | One line per instrument action, with who asked, whether a person approved it, and where the result went. This is the Instrument Trace that reward learning reads. |

## Tango naming

Tango command names are case-insensitive and every device already has a `Status` command, so on Tango devices the `status` verb is exposed as **`job_status`**. Its request and response are the `status` shapes above; the MCP tool is `<Class>_job_status`.

## Why a long job is not one command

A Tango command times out after seconds (3 s for a default client, 30 s from the MCP server), but a map or an indent grid takes minutes to hours. So a job is split: `reserve` checks a plan and returns an estimate, `start` returns at once with a job id, `job_status` reports progress, `abort` stops within one step and leaves the instrument safe (the Mapper blanks the beam), `results` returns the Tiled keys of what was produced, and `release` frees the instrument for the next caller. A job interrupted by a restart can be started again and skips finished steps.

## Implementing the envelope for a new instrument

Subclass `asyncroscopy.jobs.JobDevice` and implement `resolve_plan` and `run_step`, plus `on_stop` and `on_finish` to leave the instrument safe:

```python
class MyScan(JobDevice):
    JOB_KIND = "one line saying what the job does"
    PLAN_EXAMPLE = {...}                                  # a valid plan; agents see it

    def resolve_plan(self, plan: dict) -> dict:           # validate, expand into steps, estimate; raise ValueError to refuse
        ...
    def run_step(self, job: dict, index: int) -> dict:    # do one step; return what it produced
        ...
    def on_stop(self, job: dict, reason: str) -> None:    # abort or failure: make the instrument safe
        ...
```

[docs/Adding_New_Hardware/new_instrument.md](../../docs/Adding_New_Hardware/new_instrument.md) walks through a complete example.

The base class handles threading, state, abort, resume and tracing. Outside Python, implement the same JSON shapes; validate against the schema with any JSON Schema library.

## Limits and suspenders

Two checks run in the base class, after Bluesky's `check_limits` and suspenders:

- **Limits at reserve.** Implement `step_targets(job)` to say what each step will set: `[{"step": i, "device": address, "attribute": name, "value": number}, ...]`. `reserve` reads each attribute's Tango `min_value`/`max_value` (set them on your device's attributes) and refuses the plan with `plan_rejected` and a `violations` list (step, device, attribute, value, min, max) before anything moves. A target may carry its own `"min"`/`"max"` for limits the device reports some other way. A target whose limits cannot be read refuses the plan (`plan_rejected` with an `unchecked` list), because the plan cannot be checked; a device that sets its `require_limits` property to false reserves it anyway and lists those targets under `limits.unchecked` in the reply.
- **Suspenders between steps.** The `suspenders` device property is a JSON list of conditions on other devices' attributes:

  ```yaml
  mapper:
    properties:
      suspenders: '[{"attr": "lab/source/1/emission_current_ua", "min": 10, "resume_after_s": 30, "label": "emission current"}]'
  ```

  Each condition gives `min`, `max` or `equals`; an attribute that cannot be read counts as bad. Watch the source itself, not a value that also drops when the job blanks the beam between steps (the Mapper does). The digital twin has no such attribute yet. Before each step, a bad condition calls `on_suspend` (make the instrument safe there, e.g. blank the beam) and pauses the job in state `paused_suspended`, with the reason in `job_status`'s `suspended`. The job resumes by itself once every condition that tripped has been good for its `resume_after_s`. `abort` works while paused; `release` asks for an abort first. Conditions are checked between steps only, so keep steps short where a condition must stop work quickly.

## Versioning

Each schema carries `schema_version`. A change that breaks an existing caller raises the major version; an addition raises the minor version (envelope 1.1.0 added the `paused_suspended` state, `job_status`'s `suspended`, `reserve`'s `limits` and an error's `violations`). Replies carry the envelope's version. Copies of a schema in other repositories (the pyTEMlib fork keeps one for its jobs) must match byte for byte; `tests/test_contracts.py` checks this when both are installed.
