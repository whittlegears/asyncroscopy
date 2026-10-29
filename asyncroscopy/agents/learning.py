"""Skills learned from experience: drafted from runs that met their goal, used only once an operator promotes them.

After a goal run the checker says was met (from the servers' replies, not the
model's word), ``distill`` turns the calls that worked into a SKILL.md draft
in the memory folder (``<memory>/skills/<name>/SKILL.md``):

* the steps are the successful tool calls in order (repeated waits collapsed),
  with their arguments; data keys and job ids become placeholders;
* one short model call names it and writes its description and keywords
  (without a model, they come from the goal's words);
* ``tools:`` lists exactly the tools used, and the provenance (runs, dates,
  model, number of successes, the tool sequence) is kept in its front matter.

The same tool sequence succeeding again raises the success count of the
existing draft instead of making a new one. After a run that did not meet
its goal, ``lesson`` turns the unmet criteria and errors into one line for
LESSONS.md instead.

Safety: a learned skill starts as ``status: draft`` and is never put in a
prompt until promoted, by an operator (``python -m asyncroscopy.agents.skills
promote <name>``) or automatically after ``$ASYNCROSCOPY_AUTO_PROMOTE_AFTER``
successes (default 0: never). ``lint`` refuses one that is too long, tells
the agent to get around approvals or safety, names operator-only tools or
hard-codes data keys, or uses a tool the endpoint does not offer. A learned
skill only describes calls; every call still goes through the approval gate.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

from asyncroscopy.agents.goals import DATA_KEY, MAP_KEY, Call, Goal
from asyncroscopy.agents.memory import Memory, _slug
from asyncroscopy.skills.registry import content_words

MAX_LEARNED_CHARS = 4000
MAX_RUNS_KEPT = 20
HEX_ID = re.compile(r"\b[0-9a-f]{12}\b")
UNSAFE = [
    (re.compile(r"\b(?:bypass|skip|ignore|disable|avoid|circumvent|turn off|get around|work around)\b[^.\n]{0,40}\b(?:approvals?|safety|operators?|polic(?:y|ies)|limits?|gates?|interlocks?|dose budget)\b", re.I), "tells the agent to get around approvals or safety"),
    (re.compile(r"\bApprovals_(?:approve|deny|set_mode)\b|\bset_mode\b|\bexport_ground_truth\b", re.I), "names an operator-only tool"),
    (re.compile(r"\b(?:approve|grant)\b[^.\n]{0,20}\b(?:your own|itself|yourself)\b", re.I), "tells the agent to approve its own request"),
]
NAME_PROMPT = """Name a procedure for a lab's skill library. Reply with JSON only:
{"name": "<2-4 words, snake_case>", "description": "<one sentence: when to use it>", "keywords": ["<3-6 short phrases a request would contain>"]}"""


def auto_promote_after() -> int:
    return int(os.environ.get("ASYNCROSCOPY_AUTO_PROMOTE_AFTER", "0") or 0)


def tool_sequence(calls: list[Call]) -> list[str]:
    """The successful calls' tool names in order, with immediate repeats (polling, waiting) collapsed."""
    names: list[str] = []
    for call in calls:
        if not call.error and (not names or names[-1] != call.name):
            names.append(call.name)
    return names


def _general(value: Any) -> Any:
    """Arguments with run-specific values replaced: data keys, map keys and job ids become placeholders."""
    if isinstance(value, dict):
        return {k: _general(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_general(v) for v in value]
    if isinstance(value, str):
        if value.startswith("{"):
            try:
                return _general(json.loads(value))
            except ValueError:
                pass
        if DATA_KEY.fullmatch(value) or MAP_KEY.fullmatch(value):
            return "<key from an earlier step>"
        if HEX_ID.fullmatch(value):
            return "<id from an earlier step>"
    return value


def lint(text: str, tools: list[str], offered: set[str] | None = None) -> list[str]:
    """What is wrong with a learned skill's text (empty when it is acceptable)."""
    problems = []
    if len(text) > MAX_LEARNED_CHARS:
        problems.append(f"longer than {MAX_LEARNED_CHARS} characters")
    for pattern, why in UNSAFE:
        if pattern.search(text):
            problems.append(why)
    body = text.split("---", 2)[2] if text.startswith("---") else text
    if DATA_KEY.search(body) or MAP_KEY.search(body) or HEX_ID.search(body):
        problems.append("hard-codes a data key or job id")
    if offered is not None:
        extra = sorted(set(tools) - offered)
        if extra:
            problems.append(f"uses tools the endpoint does not offer this agent: {extra}")
    return problems


def _front(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    return yaml.safe_load(front) or {}, body


def _write(path: Path, meta: dict, body: str) -> str:
    text = "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n" + body.lstrip("\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def learned_skills(memory: Memory) -> dict[str, tuple[Path, dict]]:
    out = {}
    if memory.skills_dir.is_dir():
        for path in sorted(memory.skills_dir.glob("*/SKILL.md")):
            try:
                meta, _ = _front(path)
            except (ValueError, yaml.YAMLError):
                continue
            out[str(meta.get("name") or path.parent.name)] = (path, meta)
    return out


async def _naming(model, goal: Goal, sequence: list[str]) -> dict:
    fallback_words = [w for w in re.findall(r"[a-z]+", goal.statement.lower()) if w in content_words(goal.statement) or len(w) > 3][:4]
    fallback = {"name": "_".join(fallback_words) or "learned_procedure", "description": f"Learned procedure for requests like: {goal.statement[:120]}", "keywords": fallback_words[:5]}
    if model is None:
        return fallback
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        reply = await model.ainvoke([SystemMessage(NAME_PROMPT), HumanMessage(f"Request: {goal.statement}\nTool calls that worked, in order: {' -> '.join(sequence)}")])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        name = _slug(str(data.get("name") or ""), 40)
        keywords = [str(k)[:40] for k in (data.get("keywords") or []) if isinstance(k, str)][:6]
        description = " ".join(str(data.get("description") or "").split())[:200]
        if name and description:
            return {"name": name, "description": description, "keywords": keywords or fallback["keywords"]}
    except Exception:
        pass
    return fallback


async def distill(goal: Goal, calls: list[Call], memory: Memory, offered: set[str], model=None, model_name: str = "", run_id: str = "") -> tuple[Path | None, str]:
    """Record a successful run as a learned skill. Returns (path, what happened: created / merged / refused: why)."""
    sequence = tool_sequence(calls)
    if not sequence:
        return None, "refused: no successful tool calls to learn from"
    pattern = " > ".join(sequence)
    tools = sorted(set(sequence))
    today = time.strftime("%Y-%m-%d")
    for name, (path, meta) in learned_skills(memory).items():
        learned = meta.get("learned") or {}
        if learned.get("pattern") == pattern:
            learned["successes"] = int(learned.get("successes", 1)) + 1
            learned["runs"] = (list(learned.get("runs") or []) + [run_id])[-MAX_RUNS_KEPT:]
            learned["last"] = today
            meta["learned"] = learned
            if meta.get("status") == "draft" and auto_promote_after() and learned["successes"] >= auto_promote_after():
                _, body = _front(path)
                if not lint(body, tools, offered):
                    meta["status"] = "promoted"
            _write(path, meta, _front(path)[1])
            return path, f"merged into {name} ({learned['successes']} successes)"
    naming = await _naming(model, goal, sequence)
    name, existing = naming["name"], learned_skills(memory)
    base, n = name, 2
    while name in existing:
        name, n = f"{base}_{n}", n + 1
    steps = []
    for call in calls:
        if call.error or (steps and steps[-1][0] == call.name):
            continue
        steps.append((call.name, _general(call.args)))
    criteria = goal.describe() or "the request is done"
    body = "\n".join([
        f"# {naming['description']}",
        "",
        f"Learned from a run that met its goal: \"{goal.statement[:200]}\". The values are from that run: change them to fit the request.",
        "",
        "## Steps",
        "",
        *[f"{i + 1}. `{tool}` with `{json.dumps(args, default=str)[:300]}`" for i, (tool, args) in enumerate(steps)],
        "",
        "## Done when",
        "",
        f"- {criteria}",
        "",
        "If a call is held for approval, wait for the operator as usual; if one fails, read its message and fix the arguments.",
        "",
    ])
    meta = {
        "name": name,
        "description": naming["description"],
        "version": "0.1.0",
        "status": "draft",
        "keywords": naming["keywords"],
        "tools": tools,
        "learned": {"pattern": pattern, "successes": 1, "runs": [run_id], "first": today, "last": today, "model": model_name, "goal": goal.statement[:200]},
    }
    problems = lint(body + naming["description"] + " ".join(naming["keywords"]), tools, offered)
    if problems:
        return None, "refused: " + "; ".join(problems)
    path = memory.skills_dir / name / "SKILL.md"
    _write(path, meta, body)
    return path, f"created draft {name}"


def lesson(goal: Goal, verdict_results: list[dict], calls: list[Call]) -> str:
    """One line for LESSONS.md from a run that did not meet its goal: what was unmet and the errors seen."""
    unmet = [r["why"] for r in verdict_results if not r["met"]]
    errors = []
    for call in calls:
        if call.error:
            message = " ".join(call.result.split())[:120]
            if all(message != e.split(": ", 1)[-1] for e in errors):
                errors.append(f"{call.name}: {message}")
    text = f"\"{goal.statement[:100]}\" was not met: " + "; ".join(unmet[:3])
    if errors:
        text += ". Errors: " + " | ".join(errors[:3])
    return text


def promote(name: str, memory: Memory, offered: set[str] | None = None) -> str:
    """Make a learned draft usable by agents (after linting it again). Returns a message."""
    found = learned_skills(memory).get(name)
    if found is None:
        raise KeyError(f"no learned skill {name!r}; learned: {sorted(learned_skills(memory))}")
    path, meta = found
    _, body = _front(path)
    problems = lint(body + str(meta.get("description", "")), list(meta.get("tools") or []), offered)
    if problems:
        raise ValueError(f"{name} cannot be promoted: " + "; ".join(problems))
    meta["status"] = "promoted"
    _write(path, meta, body)
    return f"{name} promoted: agents now get it when a request matches"
