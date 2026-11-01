"""Skills that run themselves: a skill's ``run`` list of tool calls, sent one after another in the background.

A small model drops a step of a long fixed procedure (qwen3:8b asked to find a particle and take a spectrum on
it left out one of about eight calls in most runs: the unblank, the image, or the spectrum itself). A skill whose
steps never change lists them in its front matter under ``run``; ``run_skill`` starts them as one background run
and answers at once, and ``wait_for_skill`` gives the outcome. Nothing new runs on the instrument: each step is a
tool the model could call itself, through the same endpoint, safety policy, approvals and trace.

    run:
      inputs: {fov_m: 3.0e-7}
      steps:
        - {tool: "*_set_fov", args: {fov: "{inputs.fov_m}"}}
        - {id: survey, tool: "*_acquire_scanned_image", args: {detector_list: [haadf]}}
        - {tool: "*_blank_beam", always: true}

``inputs`` are what the caller may set, with their defaults (null: the caller must give it). ``tool`` is a glob matched against the agent's
tools (each instrument's tools carry its own prefix). An argument that is exactly ``{path}`` becomes that value,
type and all: ``inputs.<name>``, or a step's ``id`` followed by keys and list indices into its reply
(``found.features.0.x_m``). A step that fails, is held for approval, or names a value an earlier reply does not
have, stops the run with why; the steps marked ``always`` run anyway (blanking the beam). A step's ``cut_at`` keeps
its reply's text only up to that text (a spectrum's peaks without its channel table, which crowded the peaks out of
a small model's view). ``answer`` tells the model where in a finished run's ``outputs`` its answer is. A finished
run's reply lists every call it made (``calls``), so traces, the claim check and benchmarks count them as the
agent's own.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import re
import uuid
from typing import Any

from langchain_core.tools import StructuredTool

from asyncroscopy.skills.registry import list_skills

RUN_TOOLS = ("run_skill", "wait_for_skill")
REFERENCE = re.compile(r"^\{([A-Za-z_][\w.]*)\}$")
CALL_RESULT_CHARS = 120  # each call's reply as listed in ``calls``: enough for a key or an error
OUTPUT_CHARS = 3000  # each named step's reply in ``outputs``: a spectrum's peak table fits
WAIT_S = 600.0  # longest one wait_for_skill call waits before saying the run is still going


class StepFailed(Exception):
    pass


def _reply(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _failed(text: str) -> bool:
    return text.startswith(("Error", "APPROVAL REQUIRED"))


def _lookup(path: str, inputs: dict, replies: dict) -> Any:
    head, *rest = path.split(".")
    if head == "inputs":
        value: Any = inputs
    elif head in replies:
        value = replies[head]
    else:
        raise StepFailed(f"{{{path}}} names no input or earlier step")
    for part in rest:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            shown = json.dumps(replies.get(head), default=str)[:400] if head != "inputs" else json.dumps(inputs)
            raise StepFailed(f"the {head} step's reply has no {path[len(head) + 1:]}: it was {shown}")
    return value


def fill(value: Any, inputs: dict, replies: dict) -> Any:
    """``value`` with every ``{path}`` string replaced by what it names."""
    if isinstance(value, str):
        match = REFERENCE.match(value)
        return _lookup(match.group(1), inputs, replies) if match else value
    if isinstance(value, list):
        return [fill(v, inputs, replies) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, inputs, replies) for k, v in value.items()}
    return value


def skill_run_calls(reply: Any) -> list[dict]:
    """The calls a skill run reports in its reply (``calls``), or none when the reply is not a skill run's."""
    value = _reply(reply) if isinstance(reply, str) else reply
    if not isinstance(value, dict) or "skill" not in value or not isinstance(value.get("calls"), list):
        return []
    return [c for c in value["calls"] if isinstance(c, dict) and "tool" in c]


class SkillRuns:
    """``run_skill`` and ``wait_for_skill`` over an agent's tools: one background run at a time."""

    def __init__(self, tools, skills=None):
        self.tools = {t.name: t for t in tools}
        found = skills if skills is not None else list_skills(set(self.tools))
        # Runnable only when every step's tool is offered: a run must not fail part way for a tool that is not there.
        self.skills = {s.name: s for s in found if s.run and s.available and all(self._offers(str(step["tool"])) for step in s.run["steps"])}
        self.runs: dict[str, dict] = {}

    def offered(self) -> bool:
        return bool(self.skills)

    def _offers(self, pattern: str) -> bool:
        return any(fnmatch.fnmatchcase(n, pattern) for n in self.tools)

    def _tool_for(self, pattern: str) -> str:
        matches = sorted(n for n in self.tools if fnmatch.fnmatchcase(n, pattern))
        if not matches:
            raise StepFailed(f"no tool matches {pattern}")
        return matches[0]

    def _inputs(self, skill, given: Any) -> dict:
        if isinstance(given, str):
            given = _reply(given) if given.strip() else {}
        given = dict(given or {})
        declared = dict(skill.run.get("inputs") or {})
        unknown = sorted(set(given) - set(declared))
        missing = sorted(k for k, v in declared.items() if v is None and given.get(k) is None)
        if unknown or missing:
            # The exact call to send: a small model passed a map and top_k to a skill that takes a position.
            example = json.dumps({"skill": skill.name, "inputs": {k: "<number>" for k in missing}} if missing else {"skill": skill.name})
            wrong = f"takes inputs {sorted(declared) or 'none'}, not {unknown}" if unknown else f"needs inputs {missing}"
            raise ValueError(f"{skill.name} {wrong}: send {example}")
        for name, default in declared.items():
            if (default is None or isinstance(default, (int, float))) and not isinstance(default, bool) and name in given:
                given[name] = float(given[name])
        return {**declared, **given}

    async def start(self, skill: str, inputs: Any = None) -> str:
        if skill not in self.skills:
            # A small model sent skills without a run list, and tools, to run_skill: say what the name is and what to do.
            if skill in self.tools:
                what = f"{skill} is a tool: call it directly"
            elif skill in {s.name for s in list_skills()}:
                what = f"{skill} is instructions to follow with its tools, step by step; it has no run list"
            else:
                what = f"there is no skill {skill!r}"
            return json.dumps({"error": f"{what}. run_skill runs only {sorted(self.skills)}"})
        going = next((r for r in self.runs.values() if not r["task"].done()), None)
        if going:
            return json.dumps({"error": f"skill run {going['run_id']} is still going: call wait_for_skill with {{\"run_id\": \"{going['run_id']}\"}}"})
        try:
            values = self._inputs(self.skills[skill], inputs)
            steps = [{**s, "tool": self._tool_for(str(s["tool"]))} for s in self.skills[skill].run["steps"]]
        except (ValueError, StepFailed) as exc:
            return json.dumps({"error": str(exc)})
        run_id = uuid.uuid4().hex[:8]
        run = {"run_id": run_id, "skill": skill, "inputs": values, "calls": [], "outputs": {}, "state": "running",
               "answer": self.skills[skill].run.get("answer")}
        run["task"] = asyncio.create_task(self._run(run, steps))
        self.runs[run_id] = run
        return json.dumps({"run_id": run_id, "skill": skill, "state": "running", "steps": len(steps),
                           "next": f"call wait_for_skill with {{\"run_id\": \"{run_id}\"}} for the outcome"})

    async def _call(self, run: dict, step: dict, replies: dict) -> None:
        args = fill(step.get("args") or {}, run["inputs"], replies)
        try:
            text = await self.tools[step["tool"]].ainvoke(args)
        except Exception as exc:  # a tool without handle_tool_error, or a dropped connection
            text = f"Error: {type(exc).__name__}: {exc}"
        text = text if isinstance(text, str) else json.dumps(text, default=str)
        failed = _failed(text)
        run["calls"].append({"tool": step["tool"], "args": args, "result": text[:CALL_RESULT_CHARS], "error": failed})
        if failed:
            raise StepFailed(f"{step['tool']} failed: {text[:600]}")
        if step.get("cut_at") and step["cut_at"] in text:
            text = text[: text.index(step["cut_at"])].rstrip()
        if step.get("id"):
            replies[step["id"]] = _reply(text)
            run["outputs"][step["id"]] = replies[step["id"]] if len(text) <= OUTPUT_CHARS else text[:OUTPUT_CHARS]

    async def _run(self, run: dict, steps: list[dict]) -> None:
        replies: dict = {}
        stopped = None
        for step in steps:
            if stopped and not step.get("always"):
                continue
            try:
                await self._call(run, step, replies)
            except StepFailed as exc:
                stopped = stopped or str(exc)
        run["state"] = "stopped" if stopped else "done"
        if stopped:
            run["stopped"] = stopped

    def _status(self, run: dict) -> dict:
        out = {k: run[k] for k in ("run_id", "skill", "state")}
        if "stopped" in run:
            out["stopped"] = run["stopped"]
        elif run["state"] == "done" and run["answer"]:
            out["next"] = f"{run['answer']} No more tool calls are needed."
        out.update({k: run[k] for k in ("outputs", "calls")})
        if run["state"] == "running":
            out["next"] = f"still going ({len(run['calls'])} calls so far): call wait_for_skill with {{\"run_id\": \"{run['run_id']}\"}} again"
        return out

    async def wait(self, run_id: str, wait_s: float = WAIT_S) -> str:
        run = self.runs.get(run_id)
        if run is None:
            return json.dumps({"error": f"no skill run {run_id!r}; runs: {sorted(self.runs)}"})
        try:
            await asyncio.wait_for(asyncio.shield(run["task"]), timeout=max(0.0, float(wait_s)))
        except TimeoutError:
            pass
        return json.dumps(self._status(run), default=str)

    def as_tools(self) -> list[StructuredTool]:
        names = ", ".join(sorted(self.skills))
        start = StructuredTool(
            name="run_skill",
            description=(f"Run a skill's fixed list of tool calls in the background, one after another ({names}). Only when a "
                         "skill for this task says to. Answers at once with a run_id; wait_for_skill gives the outcome."),
            args_schema={"type": "object", "properties": {
                "skill": {"type": "string", "description": "The skill's name."},
                "inputs": {"type": "object", "description": "The skill's inputs; leave out to use its defaults."}},
                "required": ["skill"]},
            coroutine=self.start,
        )
        wait = StructuredTool(
            name="wait_for_skill",
            description="Wait for a run_skill run to finish; its reply has every call made, each named step's reply (outputs), and why it stopped if it did.",
            args_schema={"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
            coroutine=self.wait,
        )
        return [start, wait]
