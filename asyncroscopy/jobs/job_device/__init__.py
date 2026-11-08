"""Tango base class for jobs that outlive a single command.

A Tango command times out after seconds; a map takes hours. ``JobDevice``
splits a job into the envelope verbs: ``reserve`` checks a plan and returns an
estimate, ``start`` launches a worker thread and returns at once, ``status``
reports progress, ``abort`` stops within one step, ``results`` returns what was
produced and ``release`` frees the device. State is saved after every step,
so a restarted device resumes a job and skips the steps already done.

A job is kept as three files in ``state_directory``: the plan (``<device>__<id>.plan.json``: the plan,
its resolution, estimate and limits), written once at reserve (again only by ``revise``, before the job
starts); a small progress record (``<device>__<id>.json``: state, done ranges, result keys, the hooks'
own progress) rewritten after every step; and the step results, appended one line per step
(``.steps.jsonl``). At Init only the jobs that are active or can be resumed are loaded; a finished one
(done or released) is opened from its files when it is asked for by id.

Subclasses implement ``resolve_plan`` and ``run_step`` (and optionally
``on_start``, ``on_stop``, ``on_finish``, ``on_suspend`` and ``step_targets``).
One reservation is active per device at a time: the device stands for one
instrument.

Two checks borrowed from Bluesky (its ``check_limits`` and suspenders), written
for Tango:

* ``reserve`` checks every value ``step_targets`` says a step will set against
  that attribute's Tango ``min_value``/``max_value`` (or limits the target
  carries), and refuses the plan with a list of violations before anything moves.
* The ``suspenders`` device property lists conditions on other devices'
  attributes, e.g. ``[{"attr": "sys/beam/1/current_pa", "min": 10,
  "resume_after_s": 30}]``. Between steps, a condition gone bad pauses the job
  (state ``paused_suspended``, with the reason in ``job_status``); it resumes by
  itself once every condition has been good for ``resume_after_s``. Abort works
  while paused.

A person supervising a job can ``pause`` it: after the current step the device
calls ``on_suspend`` (the instrument waits safely, e.g. beam blanked) and the job
waits in state ``paused`` holding the device. ``start`` on a paused job continues
it, through the same approval as any start; ``abort`` stops it. Before a reserved
plan starts, an operator can ``revise`` it: the plan is checked again as at
reserve and gets a new plan_id. ``get_jobs`` lists the jobs a supervisor can act on.

Dry run (decision D3 of docs/designs/concepts_brief.md): a plan with ``"dry_run": true``
is reserved, approved and started like any other, but inside its worker ``proxy()``
returns a ``RecordingProxy``: reads pass through, the device's ``safe_commands`` (making
the instrument safer, such as blanking the beam) really run, and every
other command or attribute write is recorded in the trace (``dry_run: true``), checked
by ``check_dry_call`` and not performed. ``dry_run`` is part of the plan, so it is fixed
at reserve, shown to the operator who approves the start, and kept on a resume. Only a device that
sets ``SUPPORTS_DRY_RUN`` takes one: its steps must cope with the placeholder a recorded command
returns. Any other device refuses ``dry_run`` at reserve, naming itself. The reserve request may carry
``risk_overrides`` (the MCP server passes its ``mcp.safety.risk``); a command they mark high is
recorded in a dry run even when its name says it only reads (an SPM ``get_image`` scans the tip).
"""

from __future__ import annotations

from .device import JobDevice
from .errors import JobError
from .progress import done_count, done_ranges_from, is_done, mark_done, read_last_steps, step_count
from .suspenders import parse_suspenders, suspender_problem
from .worker import lease_seconds

__all__ = [
    "JobDevice",
    "JobError",
    "done_count",
    "done_ranges_from",
    "is_done",
    "lease_seconds",
    "mark_done",
    "parse_suspenders",
    "read_last_steps",
    "step_count",
    "suspender_problem",
]
