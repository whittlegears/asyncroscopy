# Contracts

Shared shapes that let different instruments, analysis servers and agents work together without importing each other's code.

| Schema | Used by | What it fixes |
| :-- | :-- | :-- |
| `envelope.schema.json` | `asyncroscopy.jobs.JobDevice` (e.g. `Mapper`), the pyTEMlib MCP server's jobs | The six verbs of a long-running job: `reserve`, `start`, `status`, `abort`, `results`, `release`. Each takes and returns one JSON object. |
| `map_plan.schema.json` | `Mapper.reserve` (STEM only) | What a STEM map run acquires: stage box, field of view, overlap, detectors (HAADF by default), per-tile EDS, electron dose budget. Another instrument writes its own plan schema; only the envelope and trace are shared. |
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

## Versioning

Each schema carries `schema_version`. A change that breaks an existing caller raises the major version. Copies of a schema in other repositories (the pyTEMlib fork keeps one for its jobs) must match byte for byte; `tests/test_contracts.py` checks this when both are installed.
