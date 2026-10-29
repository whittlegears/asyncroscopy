"""Keeping a small model's context and attention on the task: in-tool job waits, a context guard, a repeat guard.

A local model has a short context (qwen3:8b with ``num_ctx`` 8192 tokens). Live
runs showed what happens without care: a map's "done" replies and a dozen
waits filled the context, Ollama silently dropped the oldest messages (the
user's request among them), and the model repeated one call 36 times.

* ``JobWaitMiddleware``: when the model calls ``wait_for_job`` (or ``run_job``
  returns a job that is still running), the tool keeps waiting on the server in
  a loop until the job is done, failed or aborted, a cap (the estimate's
  duration x 1.5 + 120 s, else ``default_cap_s``, never past ``deadline``) or a
  stop; it reports progress through ``on_progress`` and returns ONE reply. A
  job of any length costs one tool round.
* ``fit_messages``: the messages a model call sees, within a character budget:
  the first and the latest request always, the latest tool reply in full (up to
  ``LATEST_REPLY_CHARS``), older replies as one-line digests that keep the keys
  they named, and the oldest finished tool rounds dropped whole if still over.
* ``repeats``: how many times the last tool round repeated the one before it
  exactly (same tool, arguments and reply). The skill middleware tells the
  model after one repeat and stops the run honestly after two.

Token counts are estimated as characters / ``CHARS_PER_TOKEN``.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

CHARS_PER_TOKEN = 3.5
OUTPUT_RESERVE_TOKENS = 1500  # left for the model's answer and tool call
LATEST_REPLY_CHARS = 6000
DIGEST_CHARS = 160
OLD_TEXT_CHARS = 600
ACTIVE = {"reserved", "queued", "running", "aborting", "paused_awaiting_approval", "paused_suspended"}
KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]*\.(?:h5|hdf5|hdf|nxs|tiff?)\b|\b(?=[A-Za-z0-9_\-.]*\d)[A-Za-z0-9][A-Za-z0-9_\-.]*_map\b")


def context_chars(num_ctx: int) -> int:
    """Characters a model call may use (system prompt, tool schemas and messages) for a context of ``num_ctx`` tokens."""
    return int(max(2048, num_ctx - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN)


def tools_chars(tools) -> int:
    """Characters the tool schemas take in the prompt."""
    total = 0
    for tool in tools or []:
        schema = getattr(tool, "args_schema", None)
        if schema is not None and not isinstance(schema, dict):
            try:
                schema = schema.model_json_schema()
            except Exception:
                schema = {}
        total += len(getattr(tool, "name", "")) + len(getattr(tool, "description", "") or "") + len(json.dumps(schema or {}, default=str))
    return total


def _text(message) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _size(message) -> int:
    calls = getattr(message, "tool_calls", None) or []
    return len(_text(message)) + sum(len(json.dumps(c.get("args", {}), default=str)) + len(c.get("name", "")) for c in calls) + 20


def _digest(message: ToolMessage) -> str:
    text = " ".join(_text(message).split())
    keys = sorted(set(KEY.findall(text)))[:5]
    head = text[:DIGEST_CHARS] + (" ..." if len(text) > DIGEST_CHARS else "")
    return f"[earlier reply, shortened: {head}]" + (f" keys: {', '.join(keys)}" if keys else "")


def fit_messages(messages: list, budget_chars: int) -> list:
    """The messages, shortened to fit ``budget_chars``; the first and latest requests always stay whole."""
    humans = [i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"]
    first, last_human = (humans[0], humans[-1]) if humans else (None, None)
    tool_indexes = [i for i, m in enumerate(messages) if getattr(m, "type", "") == "tool"]
    latest_tool = tool_indexes[-1] if tool_indexes else None
    last_ai = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "ai"), default=None)
    out = []
    for i, m in enumerate(messages):
        kind = getattr(m, "type", "")
        if kind == "tool":
            text = _text(m)
            if i == latest_tool:
                content = text if len(text) <= LATEST_REPLY_CHARS else text[:LATEST_REPLY_CHARS] + " ... (cut; ask again with less)"
            else:
                content = _digest(m)
            out.append(ToolMessage(content, tool_call_id=m.tool_call_id, name=getattr(m, "name", None), status=getattr(m, "status", "success")))
        elif kind == "ai" and i != last_ai and len(_text(m)) > OLD_TEXT_CHARS:
            out.append(AIMessage(_text(m)[:OLD_TEXT_CHARS] + " ...", tool_calls=list(getattr(m, "tool_calls", None) or [])))
        else:
            out.append(m)
    # Still over: drop the oldest finished tool rounds (an AI message with its replies), never a request.
    while sum(_size(m) for m in out) > budget_chars:
        drop = None
        for i, m in enumerate(out):
            if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None) and (first is None or i > first):
                ids = {c.get("id") for c in m.tool_calls}
                replies = [j for j in range(i + 1, len(out)) if getattr(out[j], "type", "") == "tool" and out[j].tool_call_id in ids]
                if replies and max(replies) < len(out) - 2:  # never the latest round
                    drop = [i, *replies]
                break
        if drop is None:
            break
        out = [m for j, m in enumerate(out) if j not in set(drop)]
    if first is not None and messages[first] not in out:  # never happens, but the request must survive
        out.insert(0, messages[first])
    if last_human is not None and messages[last_human] not in out:
        out.append(messages[last_human])
    return out


def tool_rounds(messages: list) -> list[tuple[str, str, str]]:
    """(tool, arguments, reply) of every tool call with its reply, in order."""
    replies = {getattr(m, "tool_call_id", None): _text(m) for m in messages if getattr(m, "type", "") == "tool"}
    rounds = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            if call.get("id") in replies:
                rounds.append((call.get("name", ""), json.dumps(call.get("args", {}), sort_keys=True, default=str), replies[call["id"]]))
    return rounds


# Waiting for an operator repeats the same "pending" reply by design; the approval's own expiry ends it.
NEVER_A_LOOP = ("approval_status",)


def _this_request(messages: list) -> list:
    """The messages since the user's latest message: earlier turns are other requests."""
    last = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=0)
    return messages[last:]


def repeats(messages: list) -> int:
    """How many times in a row the latest tool round repeated the one before (0: it did not)."""
    rounds = tool_rounds(_this_request(messages))
    if rounds and rounds[-1][0] in NEVER_A_LOOP:
        return 0
    count = 0
    for i in range(len(rounds) - 1, 0, -1):
        if rounds[i] != rounds[i - 1]:
            break
        count += 1
    return count


def cycles(messages: list) -> int:
    """How many earlier tool rounds, in a row or not, the latest one repeats exactly: a model cycling
    over the same few calls loops as surely as one repeating a single call. Going back to a place
    once is normal, so callers allow more of these than of repeats in a row."""
    rounds = tool_rounds(_this_request(messages))
    # An action without arguments (blank the beam, laser off) is repeated by normal protocols.
    if not rounds or rounds[-1][0] in NEVER_A_LOOP or rounds[-1][1] == "{}":
        return 0
    return rounds[:-1].count(rounds[-1])


def looping(messages: list) -> int:
    """0: no loop; 1: point the repeat out; 2: stop the run."""
    in_row, anywhere = repeats(messages), cycles(messages)
    return 2 if in_row >= 2 or anywhere >= 3 else 1 if in_row == 1 or anywhere == 2 else 0


# ---------------------------------------------------------------------- waiting inside one tool call
def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _progress(job_id: str, status: dict, waited: float) -> None:
    done = status.get("progress") or {}
    print(f"[wait] job {job_id} {status.get('state')} {done.get('done', '?')}/{done.get('total', '?')} after {waited:.0f} s", file=sys.stderr, flush=True)


class JobWaitMiddleware(AgentMiddleware):
    """Waits for a job inside one tool call: the model sees one reply when the job ends (see the module docstring)."""

    def __init__(self, tools, default_cap_s: float = 3600.0, deadline: float | None = None,
                 on_progress: Callable[[str, dict, float], None] | None = _progress, stop: Callable[[], bool] | None = None):
        super().__init__()
        self.wait_tool = next((t for t in tools if getattr(t, "name", "") == "wait_for_job"), None)
        self.default_cap_s = default_cap_s
        self.deadline = deadline  # time.monotonic() after which no wait goes on (a goal's budget)
        self.on_progress = on_progress
        self.stop = stop
        self.estimates: dict[str, float] = {}  # job id -> estimated duration (s), from run_job replies

    def _cap(self, job_id: str) -> float:
        estimate = self.estimates.get(job_id)
        cap = estimate * 1.5 + 120.0 if estimate else self.default_cap_s
        if self.deadline is not None:
            cap = min(cap, max(0.0, self.deadline - time.monotonic()))
        return cap

    def _follow(self, request, result) -> tuple[dict, str] | None:
        """(the reply's status, its job id) when this call left a job running that we should wait for."""
        name = request.tool_call.get("name")
        if self.wait_tool is None or name not in ("run_job", "wait_for_job") or not isinstance(result, ToolMessage):
            return None
        if getattr(result, "status", "success") == "error":
            return None
        status = _json(_text(result))
        if not isinstance(status, dict) or not isinstance(status.get("job_id"), str):
            return None
        estimate = (status.get("estimate") or {}).get("duration_s") if isinstance(status.get("estimate"), dict) else None
        if isinstance(estimate, (int, float)):
            self.estimates[status["job_id"]] = float(estimate)
        return (status, status["job_id"]) if status.get("state") in ACTIVE else None

    def _wait_request(self, request, job_id: str):
        call = {**request.tool_call, "name": "wait_for_job", "args": {"job_id": job_id}}
        return request.override(tool=self.wait_tool, tool_call=call)

    def _final(self, request, first: dict, status: dict, waited: float, checks: int, reason: str = "") -> ToolMessage:
        merged = {**first, **status, "waited_s": round(waited, 1), "checks": checks}
        if status.get("state") in ACTIVE:
            merged["note"] = reason or (f"still {status.get('state')} after waiting {waited:.0f} s; the job goes on. "
                                        "Tell the user it is running (with its job_id), or call wait_for_job again later.")
        else:
            merged.pop("note", None)
        return ToolMessage(json.dumps(merged, default=str), tool_call_id=request.tool_call.get("id"), name=request.tool_call.get("name"))

    def _stopped(self) -> bool:
        return bool(self.stop and self.stop())

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        followed = self._follow(request, result)
        if followed is None:
            return result
        first, job_id = followed
        status, began, checks = first, time.monotonic(), 1
        cap = self._cap(job_id)
        while status.get("state") in ACTIVE and time.monotonic() - began < cap and not self._stopped():
            reply = await handler(self._wait_request(request, job_id))
            checks += 1
            parsed = _json(_text(reply)) if isinstance(reply, ToolMessage) else None
            if not isinstance(parsed, dict) or getattr(reply, "status", "success") == "error":
                return self._final(request, first, status, time.monotonic() - began, checks,
                                   f"waiting failed: {_text(reply)[:300]}; the job may still be running: call wait_for_job again")
            status = parsed
            if self.on_progress:
                self.on_progress(job_id, status, time.monotonic() - began)
            if status.get("state") in ACTIVE and not status.get("note"):
                await asyncio.sleep(1.0)  # the server returned before its own wait ended: do not spin
        return self._final(request, first, status, time.monotonic() - began, checks)

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        followed = self._follow(request, result)
        if followed is None:
            return result
        first, job_id = followed
        status, began, checks = first, time.monotonic(), 1
        cap = self._cap(job_id)
        while status.get("state") in ACTIVE and time.monotonic() - began < cap and not self._stopped():
            reply = handler(self._wait_request(request, job_id))
            checks += 1
            parsed = _json(_text(reply)) if isinstance(reply, ToolMessage) else None
            if not isinstance(parsed, dict) or getattr(reply, "status", "success") == "error":
                return self._final(request, first, status, time.monotonic() - began, checks,
                                   f"waiting failed: {_text(reply)[:300]}; the job may still be running: call wait_for_job again")
            status = parsed
            if self.on_progress:
                self.on_progress(job_id, status, time.monotonic() - began)
            if status.get("state") in ACTIVE and not status.get("note"):
                time.sleep(1.0)
        return self._final(request, first, status, time.monotonic() - began, checks)


STOP_TEXT = ("I stopped: the same call ({name}) gave the same result three times, so trying it again "
             "would not change anything. Here is what I have so far: {reply}")
REPEAT_NOTE = ("\n\nYou already called {name} with these arguments and got this result. Do not call it again: "
               "answer the user from it, or do something different.")


def stop_message(messages: list) -> AIMessage:
    name, _args, reply = tool_rounds(messages)[-1]
    return AIMessage(STOP_TEXT.format(name=name, reply=" ".join(reply.split())[:400]))
