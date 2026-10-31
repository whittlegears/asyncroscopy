"""Claim check: a final answer may only claim what this run's tool calls did.

Benchmark runs with a small model (qwen3:8b) ended with answers such as "the
defocus was set back to 0 nm" when the last call set +20 nm, lists of image
keys no tool returned, and "you can view the spectrum" when the user asked what
the sample contains. On an instrument a claim with nothing behind it is a
safety problem, so before a final answer (no tool calls) reaches the user,
``ClaimCheckMiddleware`` checks it against the run's tool calls, without
another model call. Only claims that can be checked exactly are flagged:

==================== =================================================================
claim                unsupported when
==================== =================================================================
key                  a data key (``*.h5`` ...) or map key that no tool reply, request
                     or system prompt in this conversation contains
setting value        "<setting> set back to / reset to / is now <value>": the last value
                     set or read for it differs (or none was); "set <setting> to
                     <value>": no successful set call used that value. Lengths are
                     metres in tool calls; a unitless value is checked only when 0
action               "I took / acquired / captured ...", "moved to ...": no successful
                     call of that kind (an acquisition, a stage move, or a job) at all;
                     "a higher-magnification image", "zoomed in": no set_fov call succeeded
placement            "<key> was placed", "placed it on ...": place_image said "unplaced" for
                     that key (its latest call), or never said "placed" for any image
deferred read        the request asks what the data contain, the answer leaves viewing
                     them to the user, and no tool read any data (view_data, get_region)
==================== =================================================================

Negated and hedged sentences ("could not", "was not", "previously acquired")
are not claims. Facts come from ``goals.trace_from_messages`` and ``goals.Facts``,
the same reading of tool calls the goal checker uses.

On an unsupported claim the model gets one correction turn (a human message
named ``claim_check``) listing exactly what is unsupported; it may make the
missing call or correct the answer. If its next final answer still has
unsupported claims, a short note naming them is appended to that answer:
claims are never silently removed. ``ASYNCROSCOPY_CLAIM_CHECK=0`` turns the
check off.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from asyncroscopy.agents.goals import DATA_KEY, MAP_KEY, Facts, action_category, trace_from_messages
from asyncroscopy.agents.guards import CLAIM_CHECK, is_request

ENV = "ASYNCROSCOPY_CLAIM_CHECK"
READERS = ("view_data", "get_region")

NEGATION = re.compile(r"\b(?:not|never|no|unable|failed|cannot|couldn't|didn't|wasn't|weren't|haven't|hasn't|without|could|would|will|can|should|may|might|previously|earlier|already|existing|stored|old)\b|n't\b", re.I)
NUMBER = r"(?P<num>[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
UNIT = r"(?:\s*(?P<unit>nanomet(?:er|re)s?|micromet(?:er|re)s?|millimet(?:er|re)s?|picomet(?:er|re)s?|met(?:er|re)s?|nm|pm|µm|μm|um|mm|m)\b)?"
SCALE = {"nm": 1e-9, "pm": 1e-12, "µm": 1e-6, "μm": 1e-6, "um": 1e-6, "mm": 1e-3, "m": 1.0,
         "nanomet": 1e-9, "micromet": 1e-6, "millimet": 1e-3, "picomet": 1e-12, "met": 1.0}
FINAL_STATE = r"(?:(?:set|put|returned|reset|restored|brought|changed|moved)\s+back\s+to|(?:reset|restored|returned)\s+to|is\s+now(?:\s+at)?|now\s+at|left\s+at|ends?\s+at)"
SETTING_ALIASES = {"fov": ("fov", "field of view"), "defocus": ("defocus", "focus")}

ACQUIRED = re.compile(
    # Only the agent saying it did it: "the latest acquired image" describes data, it claims nothing.
    r"(?:\b(?:I|we)\s+(?:then\s+|also\s+|first\s+|have\s+|successfully\s+|just\s+)*|\b(?:and|then)\s+)(?:took|acquired|captured|recorded|collected)\s+(?:(?:an?|the|one|two|three|four|five|some|several|new|\d+)\s+)?(?:[\w-]+\s+){0,3}?"
    r"(?:images?|pictures?|micrographs?|spectr(?:um|a)|scans?|diffractograms?|data|frames?)\b"
    r"|\b(?:images?|pictures?|micrographs?|spectr(?:um|a)|scans?|diffractograms?)\s+(?:was|were|has\s+been|have\s+been)\s+(?:taken|acquired|captured|recorded|collected)\b"
    r"|\bI\s+(?:then\s+|also\s+|first\s+|have\s+|successfully\s+)*imaged\b",
    re.I,
)
MOVED = re.compile(
    r"\b(?:I|we)\s+(?:then\s+|also\s+|first\s+|have\s+|successfully\s+)*(?:moved|went|navigated|drove)\b"
    r"|\bstage\s+(?:was|has\s+been)\s+moved\b"
    r"|\b(?:moved|went|navigated)\s+(?:the\s+stage\s+)?(?:back\s+)?to\s+(?:the\s+)?(?:best|region|position|point|area|feature|particle|centre|center|origin|coordinates?)\b",
    re.I,
)
MAGNIFIED = re.compile(r"\b(?:higher|high|increased|greater|more)[- ]magnification\b|\bzoom(?:ed)?\s+in(?:to)?\b|\b(?:smaller|narrower)\s+field\s+of\s+view\b", re.I)
CONTENT_QUESTION = re.compile(
    r"\bwhat\b[^.?!]{0,40}\b(?:contains?|made\s+of|composed\s+of|composition|is\s+in\s+(?:it|the\s+(?:sample|spectrum|image|data)))\b"
    r"|\b(?:which|what)\s+elements?\b|\bcomposition\b|\belements?\s+(?:are|is)\s+(?:present|there)\b",
    re.I,
)
DEFERS = re.compile(
    r"\byou\s+(?:can|could|may|might)\s+(?:now\s+)?(?:view|inspect|analy[sz]e|look\s+at|open|examine|check)\b"
    r"|\bavailable\s+for\s+(?:further\s+)?(?:analysis|viewing|inspection|review)\b"
    r"|\bcan\s+be\s+(?:viewed|analy[sz]ed|inspected|examined)\b|\bfor\s+further\s+analysis\b",
    re.I,
)

PLACED = re.compile(r"\b(?:was|were|been|is|are|got|successfully|I|we)\s+(?:\w+\s+)?placed\b|\bplaced\s+(?:it|them|the\s+\w+(?:\s+\w+)?)\s+(?:on|in)\b|\bplacement\s+(?:succeeded|worked)\b", re.I)

# An answer that ends by announcing work instead of reporting it ("I'll now take an EDS spectrum on it.").
ANNOUNCES = re.compile(
    r"\b(?:I(?:'ll| will| am going to|'m going to)|let's|let\s+us|next,?\s+I(?:'ll| will))\s+(?:now\s+|then\s+|first\s+|go\s+ahead\s+and\s+)?"
    r"(?:take|acquire|record|collect|measure|find|locate|move|place|image|map|run|start|proceed|go|zoom|focus|scan|check|identify|select|prepare)\b",
    re.I,
)

CORRECTION = ("Check before you answer: your answer says things no tool call in this run did or returned:\n{items}\n"
              "Make the missing tool call now if the request needs it, or answer again without these claims. "
              "Say only what the tool replies show.")
NOTE = "\n\n[Claim check: these statements could not be verified from this run's tool calls:\n{items}]"


@dataclass
class Claim:
    kind: str  # key, setting, action, deferred_read
    text: str  # what the answer says
    why: str  # why no tool call supports it

    def line(self) -> str:
        return f'- "{self.text}": {self.why}'


def enabled_from_env() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def is_correction(message) -> bool:
    return getattr(message, "type", "") == "human" and not is_request(message)


def _text(message) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _request(messages) -> str:
    return next((_text(m) for m in reversed(messages) if is_request(m)), "")


def _sentences(answer: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+|;", answer) if s.strip()]


def _negated(sentence: str, match) -> bool:
    """A claim is hedged or negated when one of those words comes shortly before it in its sentence, or within it."""
    return bool(NEGATION.search(sentence[max(0, match.start() - 40):match.end()]))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            parsed = _json(value)
            return float(parsed) if isinstance(parsed, (int, float)) and not isinstance(parsed, bool) else None
    return None


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def _setting_of(tool: str, verb: str) -> str | None:
    match = re.match(rf"^[A-Z][A-Za-z0-9]*_{verb}_(\w+)$", tool)
    return match.group(1) if match and match.group(1) not in ("stage", "parameters") else None


class _Settings:
    """Values set and read for each setting in this run, from the calls in order."""

    def __init__(self, calls, tool_names):
        self.names = {n for t in tool_names for n in [_setting_of(t, "set")] if n} | {n for c in calls for n in [_setting_of(c.name, "set")] if n}
        self.last: dict[str, tuple[float, str]] = {}
        self.set: dict[str, list[float]] = {}
        for call in calls:
            if call.error:
                continue
            name = _setting_of(call.name, "set")
            if name:
                value = _number(call.args.get(name))
                numbers = [v for v in (_number(a) for a in call.args.values()) if v is not None]
                if value is None and len(numbers) == 1:
                    value = numbers[0]
                if value is not None:
                    self.set.setdefault(name, []).append(value)
                    self.last[name] = (value, call.name)
                continue
            name = _setting_of(call.name, "get")
            if name and _number(call.result) is not None:
                self.last[name] = (_number(call.result), call.name)
                continue
            for node in _walk(_json(call.result)):  # a reply that reports settings (get_parameters: defocus_m, fov_m)
                for key, value in node.items():
                    setting = key[:-2] if key.endswith("_m") else key
                    if setting in self.names and _number(value) is not None:
                        self.last[setting] = (_number(value), call.name)

    def phrases(self) -> list[tuple[str, str]]:
        """(setting, how an answer may name it), longest phrase first."""
        out = []
        for name in self.names:
            for phrase in {name.replace("_", " "), *SETTING_ALIASES.get(name, ())}:
                out.append((name, phrase))
        return sorted(out, key=lambda p: -len(p[1]))


def _scale(unit: str | None) -> float | None:
    if not unit:
        return None
    unit = unit.lower()
    for prefix, scale in SCALE.items():
        if unit == prefix or (len(prefix) > 2 and unit.startswith(prefix)):
            return scale
    return None


def _matches(stated: float, scale: float | None, actual: float) -> bool | None:
    """Whether a tool value (SI) agrees with a stated value; None when it cannot be checked exactly."""
    if scale is None:
        return abs(actual) <= 1e-9 if stated == 0 else None
    target = stated * scale
    return abs(actual - target) <= max(0.05 * abs(target), 0.5 * scale)


def _show(value: float, scale: float | None, unit: str | None) -> str:
    return f"{value / scale:g} {unit}" if scale else f"{value:g}"


def _setting_claims(answer: str, settings: _Settings) -> list[Claim]:
    claims: list[Claim] = []
    seen: set[tuple[str, int]] = set()
    for sentence in _sentences(answer):
        for name, phrase in settings.phrases():
            p = re.escape(phrase)
            patterns = (
                re.compile(rf"\b{p}\b[^.;\n]{{0,40}}?\b{FINAL_STATE}\s+{NUMBER}{UNIT}", re.I),
                re.compile(rf"\bset\s+(?:the\s+)?{p}\s+(?:back\s+)?(?:to|at)\s+{NUMBER}{UNIT}", re.I),
                re.compile(rf"\b{p}\s+(?:was|were|has\s+been)\s+set\s+(?:back\s+)?(?:to|at)\s+{NUMBER}{UNIT}", re.I),
            )
            for pattern in patterns:
                for m in pattern.finditer(sentence):
                    key = (sentence, m.start())
                    if key in seen or _negated(sentence, m):
                        continue
                    seen.add(key)
                    stated = float(m.group("num").replace("−", "-"))
                    unit = m.group("unit")
                    scale = _scale(unit)
                    said = m.group(0).strip()
                    if pattern is patterns[0] or re.search(r"\bback\b", said, re.I):
                        if name not in settings.last:
                            claims.append(Claim("setting", said, f"no tool call in this run set or read the {phrase}"))
                            continue
                        actual, source = settings.last[name]
                        if _matches(stated, scale, actual) is False:
                            claims.append(Claim("setting", said, f"the last {phrase} set or read in this run was {_show(actual, scale, unit)} ({source})"))
                    else:
                        values = settings.set.get(name, [])
                        if not values:
                            claims.append(Claim("setting", said, f"no successful set_{name} call in this run"))
                        elif all(_matches(stated, scale, v) is False for v in values):
                            shown = ", ".join(_show(v, scale, unit) for v in values)
                            claims.append(Claim("setting", said, f"the {phrase} values set in this run were {shown}"))
    return claims


def _action_claims(answer: str, calls) -> list[Claim]:
    ok = [c for c in calls if c.ok]
    jobs = any(action_category(c.name) == "job" for c in ok)
    acquired = jobs or any(action_category(c.name) == "acquire" for c in ok)
    moved = jobs or any(action_category(c.name) == "move" or c.name.endswith("_aim_image_shift") for c in ok)
    magnified = jobs or any(c.name.endswith("_set_fov") for c in ok)
    claims = []
    for sentence in _sentences(answer):
        for pattern, done, why in ((ACQUIRED, acquired, "no acquisition or job call succeeded in this run"),
                                   (MOVED, moved, "no stage move or job call succeeded in this run"),
                                   (MAGNIFIED, magnified, "no set_fov call succeeded in this run, so every image has the field of view it started with")):
            if done:
                continue
            m = pattern.search(sentence)
            if m and not _negated(sentence, m):
                claims.append(Claim("action", m.group(0).strip(), why))
    return claims


def _placements(calls) -> dict[str, str]:
    """The latest placement status for each image key: from place_image, or from a job that placed images (a Mapper
    zoom reports each tier's {"image", "placement": {"status"}})."""
    out: dict[str, str] = {}
    for call in calls:
        if call.error:
            continue
        reply = _json(call.result)
        if call.name.endswith("place_image"):
            if isinstance(reply, dict) and reply.get("status") in ("placed", "unplaced"):
                out[str(reply.get("child") or call.args.get("image"))] = reply["status"]
            continue
        for node in _walk(reply):
            placement = node.get("placement")
            if node.get("image") and isinstance(placement, dict) and placement.get("status") in ("placed", "unplaced"):
                out[str(node["image"])] = placement["status"]
    return out


def _placement_claims(answer: str, calls) -> list[Claim]:
    placements = _placements(calls)
    claims = []
    for sentence in _sentences(answer):
        m = PLACED.search(sentence)
        if not m or _negated(sentence, m):
            continue
        keys = [k for k in DATA_KEY.findall(sentence) if k in placements]
        refused = [k for k in keys if placements[k] == "unplaced"]
        if refused:
            claims.append(Claim("placement", sentence[:160], f"place_image said unplaced for {', '.join(refused)}; report its reason instead"))
        elif not keys and "placed" not in placements.values():
            claims.append(Claim("placement", m.group(0).strip(), "no place_image call in this run said placed"))
    return claims


def _unfinished_claims(answer: str) -> list[Claim]:
    """The answer's last sentences announce a next step the model never took (a small model ending its turn early)."""
    tail = _sentences(answer)[-2:]
    if any("?" in sentence for sentence in tail):
        return []  # a question to the user is a fair place to stop
    for sentence in tail:
        m = ANNOUNCES.search(sentence)
        if m:
            return [Claim("unfinished", sentence.strip()[:160], "the answer announces this step but ends without doing it; make the call now, or say why it cannot be done")]
    return []


def check_answer(messages: list, answer: str, tool_names=(), context: str = "") -> list[Claim]:
    """The claims in ``answer`` that the tool calls in ``messages`` do not support (see the module docstring).

    ``tool_names``: the tools the agent has (they name the settings it can set); ``context``: other text
    the model was given (its system prompt), whose keys it may name.
    """
    calls = trace_from_messages(messages)
    facts = Facts.from_calls(calls)
    claims: list[Claim] = []
    # Keys: any key a reply, a request or the prompt gave the model may be named (an error reply too).
    given = " ".join(_text(m) for m in messages if getattr(m, "type", "") in ("tool", "human")) + " " + context
    known = facts.seen_keys | set(DATA_KEY.findall(given)) | set(MAP_KEY.findall(given))
    named = list(dict.fromkeys(DATA_KEY.findall(answer) + MAP_KEY.findall(answer)))
    for key in named:
        if key not in known:
            claims.append(Claim("key", key, "no tool returned this key in this run"))
    claims += _setting_claims(answer, _Settings(calls, tool_names))
    claims += _action_claims(answer, calls)
    claims += _placement_claims(answer, calls)
    claims += _unfinished_claims(answer)
    if CONTENT_QUESTION.search(_request(messages)) and not any(c.ok and c.name in READERS for c in calls):
        m = DEFERS.search(answer)
        if m:
            claims.append(Claim("deferred_read", m.group(0).strip(),
                                "the request asks what the data contain, but no tool read them; read them (view_data) and say what they show"))
    return claims


def corrections_used(messages: list) -> int:
    """Correction turns since the user's latest request."""
    count = 0
    for m in reversed(messages):
        if is_correction(m):
            count += 1
        elif getattr(m, "type", "") == "human":
            break
    return count


def with_note(message: AIMessage, claims: list[Claim]) -> AIMessage:
    """The answer with a visible note naming the claims that could not be verified (nothing removed)."""
    text = _text(message) + NOTE.format(items="\n".join(c.line() for c in claims))
    return AIMessage(text, id=message.id, tool_calls=[], response_metadata=getattr(message, "response_metadata", {}) or {})


def _final(result: list) -> AIMessage | None:
    ai = next((m for m in reversed(result) if isinstance(m, AIMessage)), None)
    return ai if ai is not None and not ai.tool_calls else None


class ClaimCheckMiddleware(AgentMiddleware):
    """Checks a final answer's claims against the run's tool calls; one correction turn, then a visible note.

    Put it last in the middleware list so it wraps the model call itself: the correction is one more
    model call inside the same step, and the first answer, the correction and the new answer all go
    into the state (and so into traces).
    """

    def __init__(self, tool_names=(), enabled: bool | None = None, max_corrections: int = 1):
        super().__init__()
        self.tool_names = set(tool_names)
        self.enabled = enabled_from_env() if enabled is None else enabled
        self.max_corrections = max_corrections

    def _claims(self, request, history: list, answer: AIMessage) -> list[Claim]:
        system = getattr(request, "system_message", None)
        return check_answer(history, _text(answer), self.tool_names | {getattr(t, "name", "") for t in request.tools or []},
                            context=_text(system) if system is not None else "")

    def _plan(self, request, response):
        """(answer, claims, history) when the response is a final answer with unsupported claims, else None."""
        if not self.enabled:
            return None
        result = response.result if isinstance(response, ModelResponse) else [response]
        answer = _final(result)
        if answer is None:
            return None
        history = list(request.state.get("messages", [])) if getattr(request, "state", None) else list(request.messages)
        claims = self._claims(request, history, answer)
        return (answer, claims, history) if claims else None

    def _finish(self, request, history, answer, correction, retry):
        result = retry.result if isinstance(retry, ModelResponse) else [retry]
        second = _final(result)
        if second is not None:
            left = self._claims(request, [*history, answer, correction], second)
            if left:
                result = [with_note(second, left) if m is second else m for m in result]
        return ModelResponse(result=[answer, correction, *result], structured_response=getattr(retry, "structured_response", None))

    def _correct(self, request, planned):
        answer, claims, history = planned
        if corrections_used(history) >= self.max_corrections:
            return None, ModelResponse(result=[with_note(answer, claims)])
        correction = HumanMessage(CORRECTION.format(items="\n".join(c.line() for c in claims)), name=CLAIM_CHECK)
        return correction, request.override(messages=[*request.messages, answer, correction])

    def wrap_model_call(self, request, handler):
        response = handler(request)
        planned = self._plan(request, response)
        if planned is None:
            return response
        correction, next_request = self._correct(request, planned)
        if correction is None:
            return next_request
        return self._finish(request, planned[2], planned[0], correction, handler(next_request))

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        planned = self._plan(request, response)
        if planned is None:
            return response
        correction, next_request = self._correct(request, planned)
        if correction is None:
            return next_request
        return self._finish(request, planned[2], planned[0], correction, await handler(next_request))
